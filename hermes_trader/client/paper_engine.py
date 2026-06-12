"""Paper trading engine — simulated fills against LIVE market prices.

Activated by `"mode": "PAPER"` in .agent-config.json. The whole pipeline
(scan → TA → AI research → risk gates → DSL exits) runs exactly as in LIVE,
but every authenticated exchange call is intercepted:

- `place_hl_order`            → filled instantly against the live book touch
                                (best bid/ask via l2Book, fallback mid) plus
                                configurable slippage, minus taker fees
- `place_hl_trigger_order`    → stored as a virtual resting trigger, evaluated
                                against live mids on every account-state read
- `set_leverage` / cancels    → applied to the virtual book
- `fetch_account_state`       → returns the virtual portfolio in the exact
                                shape of HL's clearinghouseState aggregation

State is persisted atomically to .paper-state.json (HERMES_PAPER_STATE_FILE)
so the book survives daemon restarts, mirroring .dsl-state.json.

Config knobs (all in .agent-config.json):
- paper_starting_equity  (default 10_000 USD)
- paper_fee_bps          (default 4.5 — HL taker, charged per side)
- paper_slippage_bps     (default 2 — applied past the touch on fills,
                          and adversely on trigger fills)

Known approximations, by design:
- IOC orders always fill in full (no partial fills, no "could not
  immediately match"); paper can't reproduce book-depth exhaustion.
- Trigger orders fill AT the trigger price ± slippage; real markets gap.
- The pre-trade margin check uses realized cash, not marked equity.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOCK = threading.RLock()
_state: Optional[Dict[str, Any]] = None

_MAX_FILLS_KEPT = 300


def paper_mode_active() -> bool:
    """True when .agent-config.json says `"mode": "PAPER"` (case-insensitive)."""
    from hermes_trader.agents.config_store import read_agent_config
    return str(read_agent_config().get("mode", "OFF")).upper() == "PAPER"


def _cfg() -> Dict[str, Any]:
    from hermes_trader.agents.config_store import read_agent_config
    return read_agent_config()


def _state_path() -> str:
    return os.environ.get(
        "HERMES_PAPER_STATE_FILE", os.path.join(_ROOT, ".paper-state.json"))


def _fresh_state() -> Dict[str, Any]:
    start = float(_cfg().get("paper_starting_equity",
                             os.environ.get("HERMES_PAPER_EQUITY", 10_000)) or 10_000)
    return {
        "cash": start,
        "starting_equity": start,
        "positions": {},      # coin -> {szi, entry_px, leverage}
        "triggers": [],       # [{oid, coin, is_buy, trigger_px, kind, size}]
        "leverage": {},       # coin -> int (set_leverage results)
        "next_oid": 1,
        "realized_pnl": 0.0,
        "fees_paid": 0.0,
        "fills": [],
        "created_at": time.time(),
    }


def _load() -> Dict[str, Any]:
    global _state
    with _LOCK:
        if _state is not None:
            return _state
        try:
            with open(_state_path(), "r") as f:
                _state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _state = _fresh_state()
            logger.info(f"[paper] fresh book: ${_state['cash']:.2f} starting equity")
        return _state


def _save() -> None:
    with _LOCK:
        if _state is None:
            return
        tmp = _state_path() + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_state, f, indent=2)
        os.replace(tmp, _state_path())


def reset_book() -> Dict[str, Any]:
    """Wipe the paper book back to starting equity (operator/testing helper)."""
    global _state
    with _LOCK:
        _state = _fresh_state()
        _save()
        return dict(_state)


# ── Live market data (read-only, best-effort) ─────────────────────────────────

def _live_mid(coin: str, fallback: float = 0.0) -> float:
    """Current live mid for one coin; falls back to the caller's price."""
    from hermes_trader.client.hl_client import _http_post
    try:
        if ":" in coin:
            dex = coin.split(":", 1)[0]
            mids = _http_post("/info", {"type": "allMids", "dex": dex}) or {}
        else:
            mids = _http_post("/info", {"type": "allMids"}) or {}
        v = mids.get(coin)
        if v is not None:
            return float(v)
    except Exception as e:
        logger.warning(f"[paper] allMids failed for {coin}: {e}")
    return fallback


def _touch_price(coin: str, is_buy: bool, mid: float) -> float:
    """Best ask (buy) / best bid (sell) from a live l2Book; fallback mid."""
    from hermes_trader.client.hl_client import _http_post
    try:
        book = _http_post("/info", {"type": "l2Book", "coin": coin}) or {}
        levels = book.get("levels", [])
        bids, asks = levels[0], levels[1]
        if is_buy and asks:
            return float(asks[0]["px"])
        if not is_buy and bids:
            return float(bids[0]["px"])
    except Exception:
        pass  # thin/namespaced books flake; mid fallback is fine for paper
    return mid


def _fill_px(coin: str, is_buy: bool, mid: float) -> float:
    slip = float(_cfg().get("paper_slippage_bps", 2)) / 10_000.0
    px = _touch_price(coin, is_buy, mid)
    return px * (1 + slip) if is_buy else px * (1 - slip)


# ── Book mutations ─────────────────────────────────────────────────────────────

def _record_fill(st: Dict[str, Any], **fill: Any) -> None:
    fill["ts"] = time.time()
    st["fills"].append(fill)
    if len(st["fills"]) > _MAX_FILLS_KEPT:
        st["fills"] = st["fills"][-_MAX_FILLS_KEPT:]


def _apply_fill(st: Dict[str, Any], coin: str, delta: float, px: float,
                kind: str) -> float:
    """Apply a signed size delta at px to the book. Returns realized PnL.

    Handles netting: the portion opposing an existing position realizes PnL
    against its entry; any remainder opens (or extends) at the fill price
    with a size-weighted average entry.
    """
    fee_bps = float(_cfg().get("paper_fee_bps", 4.5)) / 10_000.0
    pos = st["positions"].get(coin)
    szi = float(pos["szi"]) if pos else 0.0
    entry = float(pos["entry_px"]) if pos else 0.0
    lev = int(pos["leverage"]) if pos else int(st["leverage"].get(coin, 5))

    realized = 0.0
    if szi != 0.0 and (szi > 0) != (delta > 0):
        closed = min(abs(szi), abs(delta))
        realized = (px - entry) * closed * (1 if szi > 0 else -1)
        st["cash"] += realized
        st["realized_pnl"] += realized

    new_szi = szi + delta
    if abs(new_szi) < 1e-12:
        st["positions"].pop(coin, None)
        st["triggers"] = [t for t in st["triggers"] if t["coin"] != coin]
    elif szi == 0.0 or (szi > 0) == (new_szi > 0) and abs(new_szi) > abs(szi):
        # opened or extended: size-weighted entry over the added portion
        added = abs(new_szi) - max(0.0, abs(szi))
        base = abs(szi) if (szi != 0 and (szi > 0) == (new_szi > 0)) else 0.0
        avg_entry = (entry * base + px * added) / (base + added) if base else px
        st["positions"][coin] = {"szi": new_szi, "entry_px": avg_entry, "leverage": lev}
    else:
        # reduced (same side, smaller) or flipped through zero
        flipped = (szi > 0) != (new_szi > 0)
        st["positions"][coin] = {
            "szi": new_szi,
            "entry_px": px if flipped else entry,
            "leverage": lev,
        }

    fee = abs(delta) * px * fee_bps
    st["cash"] -= fee
    st["fees_paid"] += fee
    _record_fill(st, coin=coin, side="buy" if delta > 0 else "sell",
                 px=px, sz=abs(delta), fee=fee, realized=realized, kind=kind)
    return realized


def _margin_used(st: Dict[str, Any]) -> float:
    return sum(abs(p["szi"]) * p["entry_px"] / max(1, int(p.get("leverage", 1)))
               for p in st["positions"].values())


# ── Exchange-API mirrors (called from exchange.py when paper mode is on) ──────

def place_order(is_buy: bool, size: float, mid_price: float, coin: str,
                reduce_only: bool = False) -> Dict[str, Any]:
    """Paper mirror of exchange.place_hl_order — instant full fill."""
    if size <= 0:
        return {"ok": False, "error": "invalid size"}
    mid = _live_mid(coin, fallback=mid_price)
    if mid <= 0:
        return {"ok": False, "error": f"invalid price for {coin}"}
    px = _fill_px(coin, is_buy, mid)

    with _LOCK:
        st = _load()
        pos = st["positions"].get(coin)
        szi = float(pos["szi"]) if pos else 0.0
        delta = size if is_buy else -size

        if reduce_only:
            if szi == 0.0 or (szi > 0) == (delta > 0):
                return {"ok": False,
                        "error": "reduce only order would increase position"}
            # HL fills only up to the live position size — clean flatten.
            delta = max(-abs(szi), min(abs(szi), delta))

        if not reduce_only:
            opening = abs(szi + delta) - abs(szi)
            if opening > 0:
                lev = max(1, int(st["leverage"].get(coin, 5)))
                projected = _margin_used(st) + (opening * px) / lev
                if projected > st["cash"]:
                    return {"ok": False,
                            "error": "Insufficient margin to place order (paper)"}

        _apply_fill(st, coin, delta, px, kind="ioc")
        oid = st["next_oid"]
        st["next_oid"] += 1
        _save()

    logger.info(f"[paper] FILL {coin} {'BUY' if is_buy else 'SELL'} "
                f"{abs(delta):.6f} @ {px:.6g} (reduce_only={reduce_only})")
    return {"ok": True, "order_id": str(oid), "avg_px": px,
            "total_sz": abs(delta), "paper": True}


def place_trigger_order(is_long_position: bool, size: float, trigger_px: float,
                        kind: str, coin: str) -> Dict[str, Any]:
    """Paper mirror of exchange.place_hl_trigger_order — virtual resting order."""
    if size <= 0 or trigger_px <= 0:
        return {"ok": False, "error": "invalid size/price"}
    with _LOCK:
        st = _load()
        oid = st["next_oid"]
        st["next_oid"] += 1
        st["triggers"].append({
            "oid": oid, "coin": coin, "is_buy": not is_long_position,
            "trigger_px": float(trigger_px), "kind": kind, "size": float(size),
        })
        _save()
    return {"ok": True, "order_id": str(oid), "paper": True}


def set_leverage(coin: str, leverage: int) -> Dict[str, Any]:
    with _LOCK:
        st = _load()
        st["leverage"][coin] = int(leverage)
        _save()
    return {"ok": True, "paper": True, "is_cross": True}


def cancel_order(oid: int) -> Dict[str, Any]:
    with _LOCK:
        st = _load()
        before = len(st["triggers"])
        st["triggers"] = [t for t in st["triggers"] if int(t["oid"]) != int(oid)]
        if len(st["triggers"]) != before:
            _save()
            return {"ok": True, "paper": True}
    return {"ok": False, "error": f"unknown paper order {oid}"}


def cancel_open_orders_for_coin(coin: str) -> int:
    with _LOCK:
        st = _load()
        before = len(st["triggers"])
        st["triggers"] = [t for t in st["triggers"] if t["coin"] != coin]
        n = before - len(st["triggers"])
        if n:
            _save()
            logger.info(f"[paper] cancelled {n} virtual trigger(s) for {coin}")
        return n


# ── Trigger evaluation + account state ─────────────────────────────────────────

def _check_triggers(st: Dict[str, Any], mids: Dict[str, float]) -> None:
    """Fire virtual SL/TP triggers crossed by the live mid (reduce-only)."""
    slip = float(_cfg().get("paper_slippage_bps", 2)) / 10_000.0
    fired: List[Dict[str, Any]] = []
    for t in list(st["triggers"]):
        pos = st["positions"].get(t["coin"])
        if not pos:
            st["triggers"].remove(t)
            continue
        mid = mids.get(t["coin"])
        if not mid or mid <= 0:
            continue
        long_pos = float(pos["szi"]) > 0
        trig = float(t["trigger_px"])
        if long_pos:
            hit = mid <= trig if t["kind"] == "sl" else mid >= trig
        else:
            hit = mid >= trig if t["kind"] == "sl" else mid <= trig
        if hit:
            fired.append(t)

    for t in fired:
        pos = st["positions"].get(t["coin"])
        if not pos:
            continue
        szi = float(pos["szi"])
        close_sz = min(float(t["size"]), abs(szi))
        delta = close_sz if t["is_buy"] else -close_sz
        # adverse slippage on the stop-out, none granted on the take-profit side
        px = float(t["trigger_px"])
        if t["kind"] == "sl":
            px = px * (1 + slip) if t["is_buy"] else px * (1 - slip)
        realized = _apply_fill(st, t["coin"], delta, px, kind=f"trigger_{t['kind']}")
        if t in st["triggers"]:
            st["triggers"].remove(t)
        logger.info(f"[paper] TRIGGER {t['kind'].upper()} fired {t['coin']} "
                    f"@ {px:.6g} (realized {realized:+.2f})")


def account_state(include_hip3: bool = False) -> Dict[str, Any]:
    """Paper mirror of hl_client.fetch_account_state — same shape, virtual book."""
    from hermes_trader.client.hl_client import fetch_all_mids
    with _LOCK:
        st = _load()
        need_hip3 = include_hip3 or any(":" in c for c in st["positions"])
    try:
        mids = {k: float(v) for k, v in fetch_all_mids(include_hip3=need_hip3).items()}
    except Exception as e:
        logger.warning(f"[paper] mids fetch failed, marking at entry: {e}")
        mids = {}

    with _LOCK:
        st = _load()
        _check_triggers(st, mids)
        _save()

        upnl = 0.0
        total_ntl = 0.0
        asset_positions = []
        for coin, p in st["positions"].items():
            szi = float(p["szi"])
            entry = float(p["entry_px"])
            mid = mids.get(coin, entry)
            pos_upnl = (mid - entry) * szi
            upnl += pos_upnl
            total_ntl += abs(szi) * mid
            asset_positions.append({
                "type": "oneWay",
                "position": {
                    "coin": coin,
                    "szi": f"{szi:.10g}",
                    "entryPx": f"{entry:.10g}",
                    "leverage": {"type": "cross", "value": int(p.get("leverage", 5))},
                    "unrealizedPnl": f"{pos_upnl:.6f}",
                    "positionValue": f"{abs(szi) * mid:.6f}",
                },
            })

        equity = st["cash"] + upnl
        available = max(0.0, equity - _margin_used(st))

        # Every dex reports "queried" so DSL reconciliation can both drop
        # closed paper positions and never falsely preserve stale trackers.
        queried: set = {""}
        if need_hip3:
            try:
                from hermes_trader.client.universe import list_hip3_dexes
                queried.update(list_hip3_dexes())
            except Exception:
                queried.update(c.split(":", 1)[0] for c in st["positions"] if ":" in c)

        return {
            "equity": equity,
            "available": available,
            "available_aggregated": available,
            "spot_usdc": 0.0,
            "total_usdc": equity,
            "total_ntl": total_ntl,
            "spot_balances": [],
            "asset_positions": asset_positions,
            "dex_equity": {d: equity for d in queried},
            "dex_available": {d: available for d in queried},
            "queried_dexes": queried,
            "paper": True,
        }
