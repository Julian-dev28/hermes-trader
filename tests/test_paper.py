"""Offline tests for the paper trading engine (mode: PAPER).

All market-data reads are monkeypatched — no network, no credentials.
"""
import json

import pytest

from hermes_trader.client import paper_engine


MIDS = {"BTC": 100_000.0, "ETH": 3_000.0, "xyz:NVDA": 180.0}


@pytest.fixture(autouse=True)
def _paper_sandbox(tmp_path, monkeypatch):
    """Isolated paper book per test: temp state file, PAPER config, fixed mids."""
    monkeypatch.setenv("HERMES_PAPER_STATE_FILE", str(tmp_path / "paper.json"))
    monkeypatch.setattr(paper_engine, "_state", None)
    monkeypatch.setattr(paper_engine, "_cfg", lambda: {
        "mode": "PAPER",
        "paper_starting_equity": 10_000,
        "paper_fee_bps": 4.5,
        "paper_slippage_bps": 0,  # deterministic fills at the mid for math tests
    })
    monkeypatch.setattr(paper_engine, "_live_mid",
                        lambda coin, fallback=0.0: MIDS.get(coin, fallback))
    monkeypatch.setattr(paper_engine, "_touch_price",
                        lambda coin, is_buy, mid: mid)
    # account_state lazy-imports fetch_all_mids from hl_client
    import hermes_trader.client.hl_client as hl
    monkeypatch.setattr(hl, "fetch_all_mids",
                        lambda include_hip3=False: {k: str(v) for k, v in MIDS.items()})
    yield
    paper_engine._state = None


def _fee(notional):
    return notional * 4.5 / 10_000.0


# ── fills & book math ───────────────────────────────────────────────────

def test_open_long_fills_and_charges_fee():
    res = paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    assert res["ok"] and res["paper"]
    assert res["avg_px"] == 100_000.0
    assert res["total_sz"] == 0.1
    st = paper_engine._load()
    assert st["positions"]["BTC"]["szi"] == pytest.approx(0.1)
    assert st["cash"] == pytest.approx(10_000 - _fee(10_000))


def test_account_state_matches_hl_shape_and_marks_to_mid():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    MIDS["BTC"] = 110_000.0
    try:
        state = paper_engine.account_state()
    finally:
        MIDS["BTC"] = 100_000.0
    pos = state["asset_positions"][0]["position"]
    assert pos["coin"] == "BTC"
    assert float(pos["szi"]) == pytest.approx(0.1)
    assert float(pos["entryPx"]) == pytest.approx(100_000.0)
    assert isinstance(pos["leverage"], dict) and "value" in pos["leverage"]
    # equity = cash + unrealized = (10k - fee) + 0.1 * 10k
    assert state["equity"] == pytest.approx(10_000 - _fee(10_000) + 1_000)
    assert "" in state["queried_dexes"]
    assert state["dex_equity"][""] == pytest.approx(state["equity"])


def test_close_realizes_pnl():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    MIDS["BTC"] = 105_000.0
    try:
        res = paper_engine.place_order(False, 0.1, 105_000.0, "BTC",
                                       reduce_only=True)
    finally:
        MIDS["BTC"] = 100_000.0
    assert res["ok"]
    st = paper_engine._load()
    assert "BTC" not in st["positions"]
    assert st["realized_pnl"] == pytest.approx(500.0)  # 0.1 × 5 000


def test_short_close_sign_is_correct():
    paper_engine.place_order(False, 1.0, 3_000.0, "ETH")  # short 1 ETH @ 3000
    MIDS["ETH"] = 2_700.0
    try:
        paper_engine.place_order(True, 1.0, 2_700.0, "ETH", reduce_only=True)
    finally:
        MIDS["ETH"] = 3_000.0
    st = paper_engine._load()
    assert st["realized_pnl"] == pytest.approx(300.0)  # short profits on the drop


def test_reduce_only_rejects_wrong_direction_and_clamps_size():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    same_dir = paper_engine.place_order(True, 0.1, 100_000.0, "BTC",
                                        reduce_only=True)
    assert not same_dir["ok"] and "increase" in same_dir["error"]
    oversized = paper_engine.place_order(False, 5.0, 100_000.0, "BTC",
                                         reduce_only=True)
    assert oversized["ok"]
    assert oversized["total_sz"] == pytest.approx(0.1)  # clamped, never flips
    assert "BTC" not in paper_engine._load()["positions"]


def test_margin_check_blocks_oversized_entry():
    paper_engine.set_leverage("BTC", 5)
    # 1 BTC @ 100k / 5x = 20k margin > 10k cash → reject
    res = paper_engine.place_order(True, 1.0, 100_000.0, "BTC")
    assert not res["ok"] and "margin" in res["error"].lower()
    assert paper_engine._load()["positions"] == {}


def test_extend_position_averages_entry():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    MIDS["BTC"] = 110_000.0
    try:
        paper_engine.place_order(True, 0.1, 110_000.0, "BTC")
    finally:
        MIDS["BTC"] = 100_000.0
    pos = paper_engine._load()["positions"]["BTC"]
    assert pos["szi"] == pytest.approx(0.2)
    assert pos["entry_px"] == pytest.approx(105_000.0)


# ── virtual triggers ────────────────────────────────────────────────────

def test_stop_loss_trigger_fires_on_long():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    paper_engine.place_trigger_order(True, 0.1, 97_500.0, "sl", "BTC")
    MIDS["BTC"] = 97_000.0  # below the stop
    try:
        state = paper_engine.account_state()
    finally:
        MIDS["BTC"] = 100_000.0
    assert state["asset_positions"] == []  # position closed by the stop
    st = paper_engine._load()
    assert st["triggers"] == []
    assert st["realized_pnl"] == pytest.approx(-250.0)  # 0.1 × −2 500
    assert any(f["kind"] == "trigger_sl" for f in st["fills"])


def test_take_profit_does_not_fire_early_and_fires_on_cross():
    paper_engine.place_order(False, 1.0, 3_000.0, "ETH")          # short
    paper_engine.place_trigger_order(False, 1.0, 2_850.0, "tp", "ETH")
    state = paper_engine.account_state()                          # mid 3 000
    assert len(state["asset_positions"]) == 1                     # not yet
    MIDS["ETH"] = 2_800.0
    try:
        state = paper_engine.account_state()
    finally:
        MIDS["ETH"] = 3_000.0
    assert state["asset_positions"] == []
    assert paper_engine._load()["realized_pnl"] == pytest.approx(150.0)


def test_cancel_open_orders_for_coin_drops_triggers():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    paper_engine.place_trigger_order(True, 0.1, 97_500.0, "sl", "BTC")
    paper_engine.place_trigger_order(True, 0.1, 104_000.0, "tp", "BTC")
    assert paper_engine.cancel_open_orders_for_coin("BTC") == 2
    assert paper_engine._load()["triggers"] == []


# ── persistence & reset ─────────────────────────────────────────────────

def test_state_survives_reload(tmp_path):
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    cash_before = paper_engine._load()["cash"]
    paper_engine._state = None  # simulate daemon restart
    st = paper_engine._load()
    assert st["positions"]["BTC"]["szi"] == pytest.approx(0.1)
    assert st["cash"] == pytest.approx(cash_before)


def test_reset_book_restores_starting_equity():
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    st = paper_engine.reset_book()
    assert st["positions"] == {} and st["cash"] == pytest.approx(10_000)


# ── integration: exchange-layer interception ────────────────────────────

def test_exchange_layer_routes_to_paper(monkeypatch):
    from hermes_trader.client import exchange
    monkeypatch.setattr(paper_engine, "paper_mode_active", lambda: True)
    res = exchange.place_hl_order(True, 0.05, 100_000.0, coin="BTC")
    assert res["ok"] and res.get("paper")
    assert exchange.set_leverage("BTC", 3)["paper"]
    trig = exchange.place_hl_trigger_order(True, 0.05, 95_000.0, "sl", coin="BTC")
    assert trig["ok"] and trig.get("paper")
    assert exchange.cancel_open_orders_for_coin("BTC") == 1


def test_fetch_account_state_routes_to_paper(monkeypatch):
    import hermes_trader.client.hl_client as hl
    monkeypatch.setattr(paper_engine, "paper_mode_active", lambda: True)
    paper_engine.place_order(True, 0.1, 100_000.0, "BTC")
    state = hl.fetch_account_state("whatever", include_hip3=True)
    assert state.get("paper") is True
    assert len(state["asset_positions"]) == 1
    assert hl.resolve_user_address() == "paper"
    assert hl.fetch_aggregate_contributions_since("paper", 1) == 0.0
