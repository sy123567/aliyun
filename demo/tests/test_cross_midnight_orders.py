"""Tests for the cross-midnight order deadline and the stale-pick retry.

Old behavior being replaced:

- every order had to FINISH by today's midnight unless the driver was a
  flexible-rest driver (daily rest quota, no fixed window). A driver with no
  rest constraints at all was capped at midnight too — silently rejecting
  every profitable evening long-haul for no compliance reason;
- even for flexible-rest drivers the extension was dead code: _pick_order's
  final revalidation clamped the deadline back to midnight
  (min(day_end, plain _hard_order_deadline)), so every extended pick that the
  scoring pass accepted was rejected one line later;
- when the chosen order went stale during the scan (load window passed while
  the clock advanced), _pick_order returned None and the driver idled — now it
  drops the stale pick and re-scores the already-scanned items once.

Run: ``python demo/tests/test_cross_midnight_orders.py`` (no pytest dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DriverRules,
    ModelDecisionService,
)


class StubApi:
    """Minimal SimulationApiPort: fixed cargo list, scriptable clock."""

    def __init__(self, items=None, start_minutes=600, advance_per_status=0):
        self._items = list(items or [])
        self._now = start_minutes
        self._advance = advance_per_status
        self.status_calls = 0

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        self.status_calls += 1
        self._now += self._advance
        return {"simulation_progress_minutes": self._now}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": [dict(it) for it in self._items]}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        raise RuntimeError("model gateway not available in this test")


def _cargo(cargo_id, price, cost_time, load_time=None):
    c = {
        "cargo_id": cargo_id,
        "cargo_name": "普货",
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "start": {"lat": 23.05, "lng": 113.25, "city": "广州"},
        "end": {"lat": 23.5, "lng": 113.8, "city": "广州"},
    }
    if load_time is not None:
        c["load_time"] = load_time
    return {"distance_km": 7.6, "cargo": c}


def _plan():
    return {
        "orders_today": {},
        "monthly_longhual": {},
        "monthly_category_orders": {},
        "monthly_deadhead_km": {},
        "zeng_order_days": set(),
        "failed_cargo_ids": set(),
        "off_days": set(),
    }


def _status():
    return {"preferences": [], "current_lat": 23.0, "current_lng": 113.2}


def test_extended_end_unconstrained_reaches_next_midnight() -> None:
    """No rest rules at all → orders may finish any time before NEXT midnight."""
    svc = ModelDecisionService(StubApi())
    rules = DriverRules()
    end = svc._extended_order_end(rules, _plan(), now=1200, day=0)
    assert end == 2 * 1440, end


def test_extended_end_flexible_rest_leaves_room_for_rest_block() -> None:
    """Daily rest quota (no fixed window) → finish by next-day (1440 − block)."""
    svc = ModelDecisionService(StubApi())
    rules = DriverRules()
    rules.daily_rest_minutes = 300
    end = svc._extended_order_end(rules, _plan(), now=1200, day=0)
    assert end == 1440 + (1440 - 300), end


def test_extended_end_night_window_not_extended() -> None:
    """A driver with a 21:00→06:00 no-drive window gets NO extension and the
    deadline shrinks to window start − buffer."""
    svc = ModelDecisionService(StubApi())
    rules = DriverRules()
    rules.no_drive_windows = [(1260, 1800)]  # 21:00 → next 06:00
    end = svc._extended_order_end(rules, _plan(), now=1200, day=0)
    assert end == 1260 - 10, end


def test_extended_end_respects_home_rule_and_off_day() -> None:
    """home_by rules and non-ordinary next days must block the extension."""
    svc = ModelDecisionService(StubApi())
    rules = DriverRules()
    rules.home_by_minute = 23 * 60
    end = svc._extended_order_end(rules, _plan(), now=1200, day=0)
    assert end <= 1440, end

    rules2 = DriverRules()
    plan2 = _plan()
    plan2["off_days"] = {1}  # tomorrow is a full rest day
    end2 = svc._extended_order_end(rules2, plan2, now=1200, day=0)
    assert end2 == 1440, end2


def test_pick_order_takes_evening_long_haul_when_unconstrained() -> None:
    """20:00, 6h haul (finishes ~02:00) — an unconstrained driver must take it
    instead of idling the evening away."""
    api = StubApi(items=[_cargo("LH-1", price=3000, cost_time=360)], start_minutes=1200)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    plan = _plan()
    deadline = svc._extended_order_end(rules, plan, now=1200, day=0)
    action = svc._pick_order("DC01", _status(), rules, plan,
                             now=1200, lat=23.0, lng=113.2, day=0, day_end=deadline)
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "LH-1", action


def test_revalidation_keeps_cross_midnight_extension() -> None:
    """Regression: the final revalidation used to clamp the deadline back to
    midnight, rejecting every cross-midnight pick the scoring pass accepted."""
    api = StubApi(items=[_cargo("LH-2", price=3000, cost_time=360)], start_minutes=1200)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.daily_rest_minutes = 300  # flexible-rest driver
    plan = _plan()
    deadline = svc._extended_order_end(rules, plan, now=1200, day=0)
    assert deadline == 1440 + 1140, deadline
    action = svc._pick_order("DC02", _status(), rules, plan,
                             now=1200, lat=23.0, lng=113.2, day=0, day_end=deadline)
    assert action is not None and action["action"] == "take_order", action


def test_night_window_driver_still_rejects_cross_midnight() -> None:
    """The same evening long-haul stays rejected for a night-window driver."""
    api = StubApi(items=[_cargo("LH-3", price=3000, cost_time=360)], start_minutes=1200)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.no_drive_windows = [(1260, 1800)]  # 21:00 → next 06:00
    plan = _plan()
    deadline = svc._extended_order_end(rules, plan, now=1200, day=0)
    action = svc._pick_order("DC03", _status(), rules, plan,
                             now=1200, lat=23.0, lng=113.2, day=0, day_end=deadline)
    assert action is None, action


def test_stale_pick_retries_second_best() -> None:
    """The top pick's load window expires while the scan clock advances; the
    retry must fall back to the second-best order instead of idling."""
    # Clock: entry now=600; +10 per get_driver_status call. First scoring pass
    # runs at 610 (arrival 618 ≤ load_end 625 → A wins on price). Revalidation
    # runs at 620 (arrival 628 > 625 → stale). Retry re-scores at 620 → B; the
    # next revalidation (630) accepts B.
    cargo_a = _cargo("A", price=3000, cost_time=120,
                     load_time=["2026-03-01 08:00", "2026-03-01 10:25"])
    cargo_b = _cargo("B", price=2000, cost_time=120)
    api = StubApi(items=[cargo_a, cargo_b], start_minutes=600, advance_per_status=10)
    svc = ModelDecisionService(api)
    plan = _plan()
    action = svc._pick_order("DC04", _status(), DriverRules(), plan,
                             now=600, lat=23.0, lng=113.2, day=0, day_end=1440)
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "B", action


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
            print(f"FAIL {t.__name__}: unexpected {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
