"""Tests for the balanced value-layer optimisation (plan A/B/C).

These cover the *mechanics* of each knob (the net effect can only be measured on
the official platform across multiple runs — notes §2). Every change is
env-tunable and by construction never relaxes a compliance guard nor accepts an
unprofitable order (the net>0 filter is unchanged).

Run: ``python demo/tests/test_balanced_value_layer.py`` (no pytest dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

import agent.model_decision_service as mds  # noqa: E402
from agent.model_decision_service import (  # noqa: E402
    DriverRules,
    ModelDecisionService,
)

LAT, LNG = 22.92, 113.18


class _StubApi:
    def __init__(self, items=None, progress=0):
        self._items = items or []
        self._progress = progress

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._progress, "current_lat": LAT, "current_lng": LNG}

    def query_cargo(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        return {"items": self._items, "k": k.get("k", 60)}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {"choices": [{"message": {"content": "{}"}}]}


def _svc(items=None, progress=0):
    return ModelDecisionService(_StubApi(items, progress))


# ============================================================ A1: ranking score

def test_rank_score_neutral_without_chain_or_abs_boost() -> None:
    """With chain weight active but no destination liquidity, and abs-alpha 0,
    the score equals the plain overhead-amortised rate."""
    base = ModelDecisionService._amortized_rate(1000.0, 100)
    assert abs(ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0) - base) < 1e-9


def test_rank_score_chain_liquidity_boosts() -> None:
    """A liquid destination market boosts the ranking score (default weight)."""
    base = ModelDecisionService._amortized_rate(1000.0, 100)
    boosted = ModelDecisionService._candidate_rank_score(1000.0, 100, 100.0)
    assert boosted > base
    # bounded: ~ (1 + weight) at/above the reference market value
    assert boosted <= base * (1.0 + mds._CHAIN_VALUE_WEIGHT) + 1e-9


def test_rank_score_chain_can_flip_two_equal_net_orders() -> None:
    """Two equal net/duration orders: the one into the liquid market ranks
    higher, so chain value actually changes the choice."""
    dead = ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0)
    live = ModelDecisionService._candidate_rank_score(1000.0, 100, 90.0)
    assert live > dead


def test_rank_score_abs_net_alpha_emphasises_big() -> None:
    """When AGENT_ABS_NET_ALPHA>0 the absolute-net log term lifts a big haul's
    score relative to a small one beyond the overhead amortisation alone."""
    original = mds._ABS_NET_ALPHA
    mds._ABS_NET_ALPHA = 0.2
    try:
        big = ModelDecisionService._candidate_rank_score(8000.0, 600, 0.0)
        base_big = ModelDecisionService._amortized_rate(8000.0, 600)
        assert big > base_big
    finally:
        mds._ABS_NET_ALPHA = original


# ====================================================== A3: weak-local reposition

def _cargo(cid, *, price, cost_time, slat=LAT, slng=LNG):
    cargo = {
        "cargo_id": cid,
        "cargo_name": "普货",
        "start": {"lat": slat, "lng": slng, "city": "广州"},
        "end": {"lat": slat, "lng": slng, "city": "广州"},
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "load_time": None,
    }
    return {"cargo": cargo, "distance_km": 0.0}


def test_picked_order_value_recovers_net_per_h() -> None:
    svc = _svc()
    rules = DriverRules()
    item = _cargo("W", price=200.0, cost_time=120)  # zero-distance -> net 200, 100/h
    plan = {"_scan_items": (480, [item]), "monthly_deadhead_km": {}}
    order = {"action": "take_order", "params": {"cargo_id": "W"}}
    val = svc._picked_order_value(order, rules, plan, 480, LAT, LNG, 0, 1440)
    assert val is not None
    net, nph = val
    assert abs(net - 200.0) < 1.0 and abs(nph - 100.0) < 1.0, val


def test_anti_strand_min_net_gate_blocks_worse_target() -> None:
    """A3 gate: a relocation anchor that does not beat the local order's net is
    not taken (we never trade a sure order for a worse one)."""
    # remote rich cargo ~0.5deg east (~50km): reachable by reposition
    remote = _cargo("RICH", price=6000.0, cost_time=120, slat=LAT, slng=LNG + 0.5)
    svc = _svc([remote], progress=480)
    rules = DriverRules()
    plan = {
        "_scan_items": (480, [remote]), "strand_count": {}, "strand_repo": set(),
        "monthly_deadhead_km": {},
    }
    # min_net below the anchor's net -> diverts (returns a reposition)
    act_low = svc._anti_strand("D", rules, plan, 480, LAT, LNG, 0, 1440, min_net=100.0)
    assert act_low is not None and act_low["action"] == "reposition", act_low
    # min_net above any achievable net -> blocked
    plan2 = {
        "_scan_items": (480, [remote]), "strand_count": {}, "strand_repo": set(),
        "monthly_deadhead_km": {},
    }
    act_high = svc._anti_strand("D", rules, plan2, 480, LAT, LNG, 0, 1440, min_net=1_000_000.0)
    assert act_high is None, act_high


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
