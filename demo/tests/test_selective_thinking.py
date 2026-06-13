"""Tests for [D1] selective thinking + [A2] multi-day night-cross extra margin.

These cover the *mechanics* of the gating (the net effect can only be measured on
the official platform across multiple runs — notes §2). By construction selective
thinking never relaxes a compliance guard: it only chooses, per step, whether the
already-happening decision LLM call runs with ``enable_thinking`` — i.e. how much
of the (otherwise idle) reasoning budget to spend, bounded by a hard cumulative
per-driver wall-time cap so the finals 4h cap is never at risk.

Run: ``python demo/tests/test_selective_thinking.py`` (no pytest dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

import agent.model_decision_service as mds  # noqa: E402
from agent.model_decision_service import (  # noqa: E402
    DecisionHistory,
    DriverRules,
    ModelDecisionService,
)

LAT, LNG = 22.92, 113.18


class _StubApi:
    def __init__(self, items=None, progress=0):
        self._items = items or []
        self._progress = progress
        self.payloads: list[dict] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._progress, "current_lat": LAT, "current_lng": LNG}

    def query_cargo(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        return {"items": self._items, "k": k.get("k", 60)}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": "{}"}}]}


def _cargo(cid, *, price, cost_time, name="普货", slat=LAT, slng=LNG, elat=LAT, elng=LNG):
    cargo = {
        "cargo_id": cid,
        "cargo_name": name,
        "start": {"lat": slat, "lng": slng, "city": "广州"},
        "end": {"lat": elat, "lng": elng, "city": "广州"},
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "load_time": None,
    }
    return {"cargo": cargo, "distance_km": 0.0}


def _full_plan() -> dict:
    return {
        "rest_done": set(), "zeng_order_days": set(), "dated_single_done": set(),
        "dated_route_done": set(), "strand_repo": set(), "orders_today": {},
        "total_deadhead_km": 0.0, "monthly_deadhead_km": {}, "must_visit_days": {},
        "first_order_taken": set(), "home_done": set(), "monthly_longhual": {},
        "monthly_category_orders": {}, "failed_cargo_ids": set(),
        "failed_cargo_reasons": {}, "off_days": set(), "_consecutive_waits": 0,
        "strand_count": {},
    }


def _decide_once(svc, api, rules, *, now, day, tod):
    return svc._llm_decide_with_history(
        "D", api.get_driver_status("D"), rules, _full_plan(), DecisionHistory(),
        now, LAT, LNG, day, tod,
    )


def _last_thinking(api) -> bool:
    assert api.payloads, "LLM was not called"
    return bool(api.payloads[-1].get("enable_thinking", False))


# ===================================================== D1: high-stakes gating

def test_big_net_candidate_triggers_thinking() -> None:
    """A candidate whose net clears the high-stakes threshold makes the step
    spend the reasoning budget (enable_thinking + the 2000-token completion)."""
    big = _cargo("BIG", price=3000.0, cost_time=120)  # zero-dist net 3000 >= 1500
    api = _StubApi([big], progress=480)
    svc = ModelDecisionService(api)
    _decide_once(svc, api, DriverRules(), now=480, day=0, tod=480)
    assert _last_thinking(api) is True
    assert api.payloads[-1]["max_tokens"] == 2000
    # the thinking call was charged against the per-driver cumulative budget
    assert "D" in svc._thinking_spent


def test_low_stakes_step_stays_fast() -> None:
    """A small-net candidate with no category pressure does NOT spend thinking
    (stays in fast mode), conserving the budget for the picks that matter."""
    small = _cargo("S", price=400.0, cost_time=120)  # net 400 < 1500
    api = _StubApi([small], progress=480)
    svc = ModelDecisionService(api)
    _decide_once(svc, api, DriverRules(), now=480, day=0, tod=480)
    assert _last_thinking(api) is False
    assert api.payloads[-1]["max_tokens"] == 180
    assert "D" not in svc._thinking_spent  # nothing charged


def test_category_pressure_triggers_thinking_even_on_small_orders() -> None:
    """Unmet category quota is a high-stakes situation regardless of the table's
    net (the model should reason about reaching the quota)."""
    small = _cargo("S", price=400.0, cost_time=120)
    api = _StubApi([small], progress=480)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.monthly_category_targets = {0: {"水果": 12}}
    rules.rule_penalties["category_targets"] = 500.0
    _decide_once(svc, api, rules, now=480, day=0, tod=480)
    assert _last_thinking(api) is True


def test_cumulative_cap_disables_thinking() -> None:
    """Once the cumulative thinking budget is exhausted, even a big-net step
    stays fast — the hard guarantee that protects the finals 4h cap."""
    big = _cargo("BIG", price=3000.0, cost_time=120)
    api = _StubApi([big], progress=480)
    svc = ModelDecisionService(api)
    svc._thinking_spent["D"] = mds._THINKING_WALL_BUDGET_SECONDS  # fully spent
    _decide_once(svc, api, DriverRules(), now=480, day=0, tod=480)
    assert _last_thinking(api) is False


def test_pace_gate_throttles_when_ahead_of_budget() -> None:
    """Mid-season, spending faster than the pro-rated pace pauses thinking until
    the run falls back under the line (self-regulating, unlike legacy)."""
    big = _cargo("BIG", price=3000.0, cost_time=120)
    half = (mds.MONTH_DAYS * mds.DAY_MINUTES) // 2  # sim_frac == 0.5
    api = _StubApi([big], progress=half)
    svc = ModelDecisionService(api)
    # spent above budget*0.5 (over pace) but below the total budget (not capped)
    svc._thinking_spent["D"] = mds._THINKING_WALL_BUDGET_SECONDS * 0.5 + 100.0
    day = half // mds.DAY_MINUTES
    _decide_once(svc, api, DriverRules(), now=half, day=day, tod=half % mds.DAY_MINUTES)
    assert _last_thinking(api) is False


def test_selective_can_be_disabled_for_legacy_behaviour() -> None:
    """With selective off and no permanent disable yet, thinking runs as before
    (legacy guard); selective gating must not be the only path to thinking."""
    big = _cargo("BIG", price=3000.0, cost_time=120)
    api = _StubApi([big], progress=480)
    svc = ModelDecisionService(api)
    original = mds._THINKING_SELECTIVE
    mds._THINKING_SELECTIVE = False
    try:
        _decide_once(svc, api, DriverRules(), now=480, day=0, tod=480)
        assert _last_thinking(api) is True
    finally:
        mds._THINKING_SELECTIVE = original


def test_thinking_off_when_decision_thinking_disabled() -> None:
    big = _cargo("BIG", price=3000.0, cost_time=120)
    api = _StubApi([big], progress=480)
    svc = ModelDecisionService(api)
    original = mds._DECISION_THINKING
    mds._DECISION_THINKING = False
    try:
        _decide_once(svc, api, DriverRules(), now=480, day=0, tod=480)
        assert _last_thinking(api) is False
    finally:
        mds._DECISION_THINKING = original


# =============================================== A2: multi-day cross extra margin

NIGHT = (1260, 1800)  # 21:00 -> 06:00 next day


def _night_rules(pen=500.0):
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT)
    rules.rule_penalties["night_window"] = pen
    return rules


def _crossing_cargo(price, cost_time):
    cargo = {
        "cargo_id": "X", "cargo_name": "普货",
        "start": {"lat": LAT, "lng": LNG, "city": "广州"},
        "end": {"lat": LAT, "lng": LNG, "city": "广州"},
        "price": float(price), "cost_time_minutes": int(cost_time), "load_time": None,
    }
    return cargo, {"cargo": cargo, "distance_km": 0.0}


def test_two_day_crossing_counted() -> None:
    svc = ModelDecisionService(_StubApi())
    # depart 16:40 day0, run 2000 min -> crosses day0 AND day1 night windows
    assert svc._count_nodrive_crossings(_night_rules(), 1000, 3000) == 2


def test_extra_margin_default_is_noop_for_multiday() -> None:
    """Default extra margin (0) leaves the legacy flat-margin acceptance intact:
    a 2-day crossing with net just above the flat margin is still accepted."""
    svc = ModelDecisionService(_StubApi())
    rules = _night_rules(pen=500.0)
    # net = 1800 - 1000(2*pen) = 800; flat required = 1000*(1.5-1)=500 -> accept
    cargo, item = _crossing_cargo(price=1800.0, cost_time=2000)
    ev = svc._evaluate_cargo(cargo, item, rules, set(), 1000, 10 ** 7, LAT, LNG)
    assert ev is not None and abs(ev[0] - 800.0) < 1.0, ev


def test_extra_margin_rejects_thin_multiday_crossing() -> None:
    """With a positive extra-per-day margin the same thin 2-day crossing is
    rejected (each extra night demands more penalty-free net)."""
    svc = ModelDecisionService(_StubApi())
    rules = _night_rules(pen=500.0)
    original = mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY
    mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY = 1.0
    try:
        # required = 500 (flat) + 500*1.0*(2-1)=500 -> 1000; net 800 <= 1000 -> reject
        cargo, item = _crossing_cargo(price=1800.0, cost_time=2000)
        assert svc._evaluate_cargo(cargo, item, rules, set(), 1000, 10 ** 7, LAT, LNG) is None
        # a genuinely huge 2-day haul still clears the raised bar
        big_cargo, big_item = _crossing_cargo(price=4000.0, cost_time=2000)  # net 3000 > 1000
        assert svc._evaluate_cargo(big_cargo, big_item, rules, set(), 1000, 10 ** 7, LAT, LNG) is not None
    finally:
        mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY = original


def test_extra_margin_does_not_touch_single_day_crossing() -> None:
    """The extra-per-day term only applies to crossings > 1; a single-night
    crossing is unaffected even when the knob is on."""
    svc = ModelDecisionService(_StubApi())
    rules = _night_rules(pen=500.0)
    original = mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY
    mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY = 1.0
    try:
        # depart 16:40, run 400 min -> finish 23:20 same day: 1 crossing only
        cargo, item = _crossing_cargo(price=900.0, cost_time=400)  # net 400 > flat 250
        ev = svc._evaluate_cargo(cargo, item, rules, set(), 1000, 10 ** 7, LAT, LNG)
        assert ev is not None and abs(ev[0] - 400.0) < 1.0, ev
    finally:
        mds._NIGHT_CROSS_EXTRA_MARGIN_PER_DAY = original


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
