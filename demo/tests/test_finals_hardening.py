"""Regression tests for the finals-hardening fixes (unknown驾驶员 robustness).

Covers the failure modes diagnosed from the four finals submissions (low gross
income + high preference penalties on the two unknown Guangdong / 江浙沪
drivers):

1. ``_pick_order`` referenced an undefined ``month_deadhead`` name — any driver
   with a monthly deadhead-cap preference raised ``NameError`` on every order
   pick and the orchestrator aborted that driver's ENTIRE month.
2. ``decide()`` had no top-level fail-safe: any unexpected exception killed the
   driver's whole simulation instead of degrading to a safe wait.
3. Orders were silently capped at midnight for drivers with no rest/night
   constraints (the past-midnight extension required ``daily_rest_minutes>0``),
   and the re-validation step clamped the deadline back to midnight anyway.
4. "每月至少N天…" quotas (full off-days) were spread over the whole 92-day
   season instead of being planted inside every calendar month.
5. A planner directive could *replace* (relax) the static night window on a
   weekday (e.g. the model believing Friday night is already the weekend).
6. Dialect phrasings (粤语/吴语) were not recognised by the offline night
   fail-safe keyword gates.

Run: ``python demo/tests/test_finals_hardening.py`` (no pytest dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DAY_MINUTES,
    MONTH_DAYS,
    DriverRules,
    ModelDecisionService,
)

NIGHT_STATIC = (1260, 1800)  # 21:00 -> next-day 06:00
WEEKEND_NIGHT = (1380, 1800)  # 23:00 -> next-day 06:00
# Simulation epoch 2026-03-01 is a SUNDAY: day 5 = Friday 2026-03-06,
# day 6 = Saturday 2026-03-07.
FRIDAY = 5
SATURDAY = 6


class _StubApi:
    def __init__(self, cargo_items=None, now=600):  # noqa: ANN001
        self._cargo_items = cargo_items or []
        self._now = now

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._now, "current_lat": 30.0, "current_lng": 120.0}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": self._cargo_items}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        raise RuntimeError("model unavailable in this test")


def _fresh_plan() -> dict:
    return {
        "monthly_category_orders": {},
        "monthly_longhual": {},
        "monthly_deadhead_km": {},
        "orders_today": {},
        "first_order_taken": set(),
        "zeng_order_days": set(),
        "failed_cargo_ids": set(),
        "off_days": set(),
        "home_done": set(),
    }


def _cargo_item(cargo_id="C1", price=1000.0, cost_time=120):  # noqa: ANN001
    """A profitable order ~48km from the driver at (30.0, 120.0)."""
    return {
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": "普货",
            "price": price,
            "cost_time_minutes": cost_time,
            "load_time": None,
            "start": {"lat": 30.0, "lng": 120.5, "city": "杭州"},
            "end": {"lat": 30.5, "lng": 120.5, "city": "湖州"},
        }
    }


# ---------------------------------------------------- 1. deadhead cap NameError

def test_pick_order_with_deadhead_cap_does_not_crash_and_enforces_cap() -> None:
    """With a monthly deadhead cap parsed, _pick_order must run (no NameError)
    and reject pickups that would bust the remaining budget."""
    svc = ModelDecisionService(_StubApi(cargo_items=[_cargo_item()]))
    rules = DriverRules()
    rules.monthly_deadhead_max_km = 100.0
    plan = _fresh_plan()
    plan["monthly_deadhead_km"] = {0: 95.0}  # 95 used; pickup ~48km busts it
    action = svc._pick_order("DX", {}, rules, plan, 600, 30.0, 120.0, 0, DAY_MINUTES)
    assert action is None, action


def test_pick_order_with_room_under_deadhead_cap_accepts() -> None:
    svc = ModelDecisionService(_StubApi(cargo_items=[_cargo_item()]))
    rules = DriverRules()
    rules.monthly_deadhead_max_km = 500.0
    plan = _fresh_plan()
    plan["monthly_deadhead_km"] = {0: 95.0}
    action = svc._pick_order("DX", {}, rules, plan, 600, 30.0, 120.0, 0, DAY_MINUTES)
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "C1"


# ----------------------------------------------------- 2. decide() fail-safe

def test_decide_degrades_to_wait_on_unexpected_exception() -> None:
    svc = ModelDecisionService(_StubApi())

    def _boom(driver_id):  # noqa: ANN001, ANN202
        raise RuntimeError("unexpected bug")

    svc._decide_impl = _boom  # type: ignore[assignment]
    action = svc.decide("DX")
    assert action == {"action": "wait", "params": {"duration_minutes": 60}}, action


# -------------------------------------------- 3. past-midnight order deadline

def test_unconstrained_driver_can_finish_orders_past_midnight() -> None:
    """A driver with NO rest/night constraints must not have orders silently
    capped at 24:00 (that idled every evening of the season)."""
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()  # no rest rules at all
    plan = {"off_days": set(), "home_done": set()}
    deadline = svc._order_finish_deadline(rules, plan, now=1200, day=0)
    assert deadline == 2 * DAY_MINUTES, deadline  # full headroom into next day


def test_flexible_rest_driver_extension_leaves_room_for_rest_block() -> None:
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()
    rules.daily_rest_minutes = 480  # "8h continuous anywhere"
    plan = {"off_days": set(), "home_done": set()}
    deadline = svc._order_finish_deadline(rules, plan, now=1200, day=0)
    assert deadline == DAY_MINUTES + (DAY_MINUTES - 480), deadline


def test_night_window_driver_keeps_conservative_deadline() -> None:
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    plan = {"off_days": set(), "home_done": set()}
    deadline = svc._order_finish_deadline(rules, plan, now=600, day=0)
    assert deadline <= NIGHT_STATIC[0], deadline  # stops before 21:00


def test_home_rule_driver_gets_no_midnight_extension() -> None:
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()
    rules.home_by_minute = 1380  # must be home by 23:00
    rules.home_lat, rules.home_lng = 30.0, 120.0
    plan = {"off_days": set(), "home_done": set()}
    deadline = svc._order_finish_deadline(rules, plan, now=600, day=0)
    assert deadline <= DAY_MINUTES, deadline


def test_pick_order_accepts_order_finishing_past_midnight() -> None:
    """End-to-end through _pick_order: the re-validation step used to clamp the
    deadline back to midnight and reject the already-chosen crossing order."""
    item = _cargo_item(cost_time=300)  # ~22:00 start + 5h haul -> finishes day 1
    svc = ModelDecisionService(_StubApi(cargo_items=[item], now=1320))
    rules = DriverRules()
    plan = _fresh_plan()
    deadline = svc._order_finish_deadline(rules, plan, now=1320, day=0)
    action = svc._pick_order("DX", {}, rules, plan, 1320, 30.0, 120.0, 0, deadline)
    assert action is not None and action["action"] == "take_order", action


# ------------------------------------------------- 4. per-month full off days

def test_off_days_are_planted_inside_every_calendar_month() -> None:
    rules = DriverRules()
    rules.off_days_min = 2
    off = ModelDecisionService._plan_off_days(rules)
    march = {d for d in off if 0 <= d < 31}
    april = {d for d in off if 31 <= d < 61}
    may = {d for d in off if 61 <= d < 92}
    assert len(march) >= 2, off
    assert len(april) >= 2, off
    assert len(may) >= 2, off
    assert all(0 <= d < MONTH_DAYS for d in off)


def test_off_days_avoid_reserved_event_days() -> None:
    rules = DriverRules()
    rules.off_days_min = 1
    rules.dated_single.append({"day": 15, "lat": 0.0, "lng": 0.0, "min_wait": 60, "before": 720})
    off = ModelDecisionService._plan_off_days(rules)
    assert 15 not in off, off


# --------------------------------------- 5. weekday replace must not relax

def test_directive_replace_rejected_on_weekday() -> None:
    """Friday night is NOT the weekend: a relaxing replacement on a weekday
    falls back to the additive union, keeping the static window enforced."""
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[FRIDAY] = {"windows": [WEEKEND_NIGHT], "replace": True, "notes": ""}
    eff = rules.no_drive_windows_for(FRIDAY)
    assert NIGHT_STATIC in eff, eff


def test_directive_replace_accepted_on_saturday() -> None:
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[SATURDAY] = {"windows": [WEEKEND_NIGHT], "replace": True, "notes": ""}
    assert rules.no_drive_windows_for(SATURDAY) == [WEEKEND_NIGHT]


# ------------------------------------------------ 6. dialect night fail-safe

def test_night_failsafe_handles_cantonese_phrasing() -> None:
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()
    svc._supplement_night_failsafe(["夜晚23点收工返屋企瞓觉，朝早6点先开工"], rules)
    assert (1380, 1800) in rules.no_drive_windows, rules.no_drive_windows
    assert rules.rest_window == (0, 360), rules.rest_window


def test_night_failsafe_handles_wu_phrasing() -> None:
    svc = ModelDecisionService(_StubApi())
    rules = DriverRules()
    svc._supplement_night_failsafe(["夜里向22点以后勿出车，困觉困到早上5点"], rules)
    assert (1320, 1740) in rules.no_drive_windows, rules.no_drive_windows


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
        except Exception as exc:  # noqa: BLE001 - surface crashes as failures
            failures += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
