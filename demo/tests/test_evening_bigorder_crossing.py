"""Tests for the soft rest-window crossing in ``_evaluate_cargo``.

Business goal (from the repo owner): maximise *net* income, not minimise
penalty. The documented gross-income leak was capping the working day at the
no-drive (night-rest) window start and hard-rejecting every evening haul that
would finish inside it — even hauls whose net far exceeds the day's rest
penalty. The scorer charges that penalty *per day* (one violation per day
regardless of how far the haul runs in), so taking a single big evening order
that crosses the window is net-positive whenever ``net > penalty``.

This mirrors the existing soft long-haul cap. It is deliberately bounded and
fail-safe:

- only when the window has a KNOWN penalty (uncertain parse → hard reject, the
  guard that prevented the documented every-night −44万 blow-up);
- the haul must finish *within* the same overnight window (never run past its
  end into the next day's driving);
- the truck must depart *before* the window (so at most one crossing per day).

Run: ``python demo/tests/test_evening_bigorder_crossing.py`` (no pytest dep).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

import agent.model_decision_service as mds  # noqa: E402
from agent.model_decision_service import (  # noqa: E402
    COST_PER_KM,
    DriverRules,
    ModelDecisionService,
)

NIGHT = (1260, 1800)  # 21:00 -> next-day 06:00 (stored end > 1440 = overnight)
NIGHT_PENALTY = 2700.0
GZ_LAT, GZ_LNG = 22.92, 113.18  # driver/pickup spot (pickup_km ~ 0)


class _StubApi:
    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {"choices": [{"message": {"content": "{}"}}]}


def _svc() -> ModelDecisionService:
    return ModelDecisionService(_StubApi())


def _rules(*, night_penalty: float | None = NIGHT_PENALTY) -> DriverRules:
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT)
    if night_penalty is not None:
        rules.rule_penalties["night_window"] = night_penalty
    return rules


def _cargo(*, price: float, cost_time: int, haul_dst=(22.95, 113.22)):
    end_lat, end_lng = haul_dst
    cargo = {
        "cargo_id": "C1",
        "cargo_name": "普货",
        "start": {"lat": GZ_LAT, "lng": GZ_LNG, "city": "广州"},
        "end": {"lat": end_lat, "lng": end_lng, "city": "广州"},
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "load_time": None,
    }
    return cargo, {"cargo": cargo, "distance_km": 0.0}


def _eval(svc, rules, cargo, item, now, day_end):
    return svc._evaluate_cargo(cargo, item, rules, set(), now, day_end, GZ_LAT, GZ_LNG)


# ----------------------------------------------------------- window helper

def test_window_covering_detects_evening_crossing() -> None:
    svc = _svc()
    rules = _rules()
    # depart 18:00, finish 23:00 -> crosses the 21:00 window, departed before it
    assert svc._nodrive_window_covering(rules, 1080, 1380) == (1260, 1800)


def test_window_covering_rejects_departure_inside_window() -> None:
    svc = _svc()
    rules = _rules()
    # already 21:40 -> cannot *start* inside the window
    assert svc._nodrive_window_covering(rules, 1300, 1400) is None


def test_window_covering_none_when_no_overlap() -> None:
    svc = _svc()
    rules = _rules()
    # daytime haul 10:00 -> 12:00, no overlap
    assert svc._nodrive_window_covering(rules, 600, 720) is None


# ------------------------------------------------------ _evaluate_cargo

def test_big_evening_order_crosses_window_when_net_beats_penalty() -> None:
    """The core fix: a 5h evening haul finishing at 23:00 is accepted even
    though the (clamped) day_end is the 21:00 window start, because its net
    income beats the day's rest penalty."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=5000.0, cost_time=300)  # 18:00 -> 23:00
    # day_end clamped at the 21:00 window start (what the scheduler passes)
    ev = _eval(svc, rules, cargo, item, now=1080, day_end=1260)
    assert ev is not None, "big profitable evening order must be accepted"
    net, _req, occupied, _pk = ev
    # net is reported AFTER the rest penalty is subtracted
    assert net > 0
    assert net < 5000.0  # penalty + mileage were charged
    assert occupied == 300  # 18:00 -> 23:00


def test_small_evening_order_not_worth_penalty_is_rejected() -> None:
    """An evening crossing whose net does NOT beat the rest penalty is
    rejected — we don't blow the day's rest for a marginal order."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=2000.0, cost_time=300)  # net 2000-mileage-2700 < 0
    assert _eval(svc, rules, cargo, item, now=1080, day_end=1260) is None


def test_crossing_rejected_when_penalty_unknown_failsafe() -> None:
    """Fail-safe: if the window has no known penalty (uncertain parse), keep the
    hard reject — this is the guard that prevented the every-night blow-up."""
    svc = _svc()
    rules = _rules(night_penalty=None)
    cargo, item = _cargo(price=5000.0, cost_time=300)
    assert _eval(svc, rules, cargo, item, now=1080, day_end=1260) is None


def test_haul_running_past_window_end_is_rejected() -> None:
    """A haul that would run past the window end (06:00) into the next day's
    driving is rejected — only finishing *within* the window is allowed."""
    svc = _svc()
    rules = _rules()
    # depart 18:00, 13h haul -> finish 07:00 next day (1080 + 780 = 1860 > 1800)
    cargo, item = _cargo(price=20000.0, cost_time=780)
    assert _eval(svc, rules, cargo, item, now=1080, day_end=1260) is None


def test_cannot_start_order_inside_window() -> None:
    """Departing while already inside the rest window stays hard-rejected."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=5000.0, cost_time=120)  # start 21:40
    assert _eval(svc, rules, cargo, item, now=1300, day_end=1800) is None


def test_daytime_order_unaffected() -> None:
    """Non-crossing daytime orders behave exactly as before."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=1000.0, cost_time=120)  # 10:00 -> 12:00
    ev = _eval(svc, rules, cargo, item, now=600, day_end=1260)
    assert ev is not None
    net, _req, occupied, _pk = ev
    assert occupied == 120
    # no rest penalty charged for a daytime order
    assert net > 1000.0 - COST_PER_KM * 10 - 1.0


def test_crossing_net_equals_price_minus_mileage_minus_penalty() -> None:
    """The reported net is exactly price - mileage cost - rest penalty, so the
    selector ranks crossing orders on their true penalty-adjusted value."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=6000.0, cost_time=300, haul_dst=(GZ_LAT, GZ_LNG))
    ev = _eval(svc, rules, cargo, item, now=1080, day_end=1260)
    assert ev is not None
    net, _req, _occ, _pk = ev
    # zero-distance haul => mileage ~ 0 => net == price - penalty
    assert abs(net - (6000.0 - NIGHT_PENALTY)) < 1.0


# --------------------------------------------------- crossing safety margin

def _eval_with_margin(margin, svc, rules, cargo, item, now, day_end):
    """Evaluate ``cargo`` with ``_NIGHT_CROSS_MARGIN`` temporarily set."""
    original = mds._NIGHT_CROSS_MARGIN
    mds._NIGHT_CROSS_MARGIN = margin
    try:
        return _eval(svc, rules, cargo, item, now, day_end)
    finally:
        mds._NIGHT_CROSS_MARGIN = original


def test_default_margin_rejects_thin_crossing() -> None:
    """A crossing whose net only just beats the rest penalty (net=800 < the
    1350 = penalty*(1.5-1) margin) is dropped by the default 1.5 margin — these
    thin crossings drove the leaderboard penalty up without enough gross."""
    svc = _svc()
    rules = _rules()
    # zero-distance haul => net == price - penalty == 3500 - 2700 == 800
    cargo, item = _cargo(price=3500.0, cost_time=300, haul_dst=(GZ_LAT, GZ_LNG))
    assert _eval_with_margin(1.5, svc, rules, cargo, item, now=1080, day_end=1260) is None


def test_legacy_margin_accepts_thin_crossing() -> None:
    """With margin == 1.0 (legacy) the same thin-but-positive crossing is still
    accepted — proving the new gate is the only thing rejecting it."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=3500.0, cost_time=300, haul_dst=(GZ_LAT, GZ_LNG))
    ev = _eval_with_margin(1.0, svc, rules, cargo, item, now=1080, day_end=1260)
    assert ev is not None
    net, _req, _occ, _pk = ev
    assert abs(net - 800.0) < 1.0, net


def test_big_crossing_survives_default_margin() -> None:
    """A genuinely big evening haul (net well above the penalty) is still taken
    under the default margin — the feature's intent is preserved."""
    svc = _svc()
    rules = _rules()
    # net == 6000 - 2700 == 3300 > penalty*(1.5-1)=1350
    cargo, item = _cargo(price=6000.0, cost_time=300, haul_dst=(GZ_LAT, GZ_LNG))
    ev = _eval_with_margin(1.5, svc, rules, cargo, item, now=1080, day_end=1260)
    assert ev is not None
    net, _req, _occ, _pk = ev
    assert abs(net - 3300.0) < 1.0, net


def test_margin_does_not_affect_daytime_orders() -> None:
    """The margin only gates crossings: a non-crossing daytime order is
    unaffected regardless of the margin value."""
    svc = _svc()
    rules = _rules()
    cargo, item = _cargo(price=1000.0, cost_time=120, haul_dst=(GZ_LAT, GZ_LNG))
    ev = _eval_with_margin(3.0, svc, rules, cargo, item, now=600, day_end=1260)
    assert ev is not None  # daytime order never sees the crossing gate


# ----------------------------------------------- end-to-end scheduler path

class _SchedStub:
    """Stub returning a fixed cargo list + driver status for ``_schedule``."""

    def __init__(self, items, lat, lng, progress):
        self._items = items
        self._lat = lat
        self._lng = lng
        self._progress = progress

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {
            "simulation_progress_minutes": self._progress,
            "current_lat": self._lat,
            "current_lng": self._lng,
        }

    def query_cargo(self, driver_id, latitude, longitude, k):  # noqa: ANN001, ANN201
        return {"items": self._items, "k": k}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {"choices": [{"message": {"content": "{}"}}]}


def _plan() -> dict:
    return {
        "rest_done": set(), "zeng_order_days": set(), "dated_single_done": set(),
        "dated_route_done": set(), "strand_repo": set(), "orders_today": {},
        "total_deadhead_km": 0.0, "monthly_deadhead_km": {}, "must_visit_days": {},
        "first_order_taken": set(), "home_done": set(), "monthly_longhual": {},
        "monthly_category_orders": {}, "failed_cargo_ids": set(),
        "failed_cargo_reasons": {}, "off_days": set(), "_consecutive_waits": 0,
        "strand_count": {},
    }


def test_schedule_takes_big_evening_order_across_window() -> None:
    """Deterministic floor: a big evening order whose net beats the rest
    penalty is taken even though it finishes inside the night window."""
    cargo, item = _cargo(price=6000.0, cost_time=180)  # 20:00 -> 23:00
    svc = ModelDecisionService(_SchedStub([item], GZ_LAT, GZ_LNG, progress=1200))
    status = {"simulation_progress_minutes": 1200, "current_lat": GZ_LAT, "current_lng": GZ_LNG}
    action = svc._schedule("D001", status, _rules(), _plan(), 1200, GZ_LAT, GZ_LNG)
    assert action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "C1", action


def test_schedule_rests_inside_window_no_cascade() -> None:
    """Once inside the rest window the floor rests (does NOT keep taking orders),
    so a crossing is bounded to one penalised day."""
    cargo, item = _cargo(price=6000.0, cost_time=120)
    svc = ModelDecisionService(_SchedStub([item], GZ_LAT, GZ_LNG, progress=1380))  # 23:00
    status = {"simulation_progress_minutes": 1380, "current_lat": GZ_LAT, "current_lng": GZ_LNG}
    action = svc._schedule("D001", status, _rules(), _plan(), 1380, GZ_LAT, GZ_LNG)
    assert action["action"] == "wait", action


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
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
