"""Regression tests for the partial-fill close fix and the DSL restart-clock fix.

- close_position_market must NOT deregister the DSL tracker when an IOC
  reduce-only close only partially fills (the residual would run stopless).
- rehydrate_from_exchange must stamp synthesized trackers with the position's
  REAL open time from fill history, not time.time() (which re-armed the
  hard-timeout clock on every restart).
"""
import time

import pytest

from hermes_trader.agents import dsl_exit, executor


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(dsl_exit, "DSL_STATE_FILE", str(tmp_path / "dsl.json"))
    dsl_exit._active_positions.clear()
    monkeypatch.setattr(executor.time, "sleep", lambda s: None)
    yield
    dsl_exit._active_positions.clear()


def _one_btc_long_state(szi=0.5, entry=100_000.0):
    return {
        "asset_positions": [{
            "position": {
                "coin": "BTC", "szi": str(szi), "entryPx": str(entry),
                "leverage": {"type": "cross", "value": 5},
                "positionValue": str(abs(szi) * entry),
            },
        }],
    }


def _wire_close(monkeypatch, order_results):
    """Stub everything close_position_market touches except the DSL registry."""
    calls = []

    def fake_order(is_buy, size, mid_price, coin="BTC", reduce_only=False):
        calls.append({"size": size, "reduce_only": reduce_only})
        return order_results[min(len(calls) - 1, len(order_results) - 1)]

    monkeypatch.setattr(executor, "resolve_user_address", lambda: "0xtest")
    monkeypatch.setattr(executor, "fetch_account_state",
                        lambda user, include_hip3=False: _one_btc_long_state())
    monkeypatch.setattr(executor, "get_hl_price", lambda coin="BTC": 100_000.0)
    monkeypatch.setattr(executor, "place_hl_order", fake_order)
    monkeypatch.setattr(executor, "cancel_open_orders_for_coin", lambda coin: 0)
    return calls


def test_partial_close_keeps_tracker_and_reports_failure(monkeypatch):
    dsl_exit.register_position("BTC", "long", 100_000.0)
    # Every attempt fills only 0.1 of the requested size → residual remains.
    calls = _wire_close(monkeypatch, [
        {"ok": True, "avg_px": 100_000.0, "total_sz": 0.1},
    ])
    res = executor.close_position_market("BTC")
    assert res["ok"] is False
    assert res["partial"] is True
    assert res["remaining_sz"] == pytest.approx(0.2)
    assert res["filled_sz"] == pytest.approx(0.3)
    assert len(calls) == 3  # initial + 2 residual retries
    # Each retry asks only for what's left, reduce-only.
    assert [round(c["size"], 6) for c in calls] == [0.5, 0.4, 0.3]
    assert all(c["reduce_only"] for c in calls)
    # THE fix: the tracker survives so the residual still has a stop/timeout.
    assert "BTC_long" in dsl_exit._active_positions


def test_partial_then_complete_close_deregisters(monkeypatch):
    dsl_exit.register_position("BTC", "long", 100_000.0)
    _wire_close(monkeypatch, [
        {"ok": True, "avg_px": 100_000.0, "total_sz": 0.3},  # partial
        {"ok": True, "avg_px": 99_900.0, "total_sz": 0.2},   # residual fills
    ])
    res = executor.close_position_market("BTC")
    assert res["ok"] is True and "partial" not in res
    # Weighted fill price across both fills: (100000×0.3 + 99900×0.2) / 0.5
    assert res["fill_px"] == pytest.approx(99_960.0)
    assert "BTC_long" not in dsl_exit._active_positions


def test_full_fill_close_unchanged(monkeypatch):
    dsl_exit.register_position("BTC", "long", 100_000.0)
    calls = _wire_close(monkeypatch, [
        {"ok": True, "avg_px": 100_050.0, "total_sz": 0.5},
    ])
    res = executor.close_position_market("BTC")
    assert res["ok"] is True and len(calls) == 1
    assert res["fill_px"] == pytest.approx(100_050.0)
    assert "BTC_long" not in dsl_exit._active_positions


def test_dust_residual_treated_as_flat(monkeypatch):
    dsl_exit.register_position("BTC", "long", 100_000.0)
    # Fills all but 1e-9 BTC (≈ $0.0001) — rounding dust, not a real residual.
    _wire_close(monkeypatch, [
        {"ok": True, "avg_px": 100_000.0, "total_sz": 0.5 - 1e-9},
    ])
    res = executor.close_position_market("BTC")
    assert res["ok"] is True
    assert "BTC_long" not in dsl_exit._active_positions


# ── restart clock ────────────────────────────────────────────────────────

def test_synthesized_tracker_uses_fill_history_open_time(monkeypatch):
    import hermes_trader.client.hl_client as hl
    opened_ms = (time.time() - 2 * 3600) * 1000  # opened 2h ago
    monkeypatch.setattr(hl, "resolve_user_address", lambda: "0xtest")
    monkeypatch.setattr(hl, "_http_post", lambda path, payload, timeout=5: [
        # newest first: a later pyramiding fill, then the open-from-flat
        {"coin": "BTC", "dir": "Open Long", "startPosition": "0.2",
         "time": opened_ms + 3_600_000},
        {"coin": "ETH", "dir": "Open Long", "startPosition": "0",
         "time": opened_ms + 1_000},
        {"coin": "BTC", "dir": "Open Long", "startPosition": "0",
         "time": opened_ms},
    ])
    dsl_exit.rehydrate_from_exchange(
        _one_btc_long_state()["asset_positions"], queried_dexes={""})
    t = dsl_exit._active_positions["BTC_long"]
    assert t.entry_time == pytest.approx(opened_ms / 1000.0)
    # The 2h-old position has only ~1h left on a 3h hard timeout — not 3h.
    age_min = (time.time() - t.entry_time) / 60
    assert 115 < age_min < 125


def test_open_time_falls_back_to_now_on_api_failure(monkeypatch):
    import hermes_trader.client.hl_client as hl
    monkeypatch.setattr(hl, "resolve_user_address", lambda: "0xtest")
    monkeypatch.setattr(hl, "_http_post",
                        lambda path, payload, timeout=5: None)
    before = time.time()
    dsl_exit.rehydrate_from_exchange(
        _one_btc_long_state()["asset_positions"], queried_dexes={""})
    t = dsl_exit._active_positions["BTC_long"]
    assert before <= t.entry_time <= time.time()
