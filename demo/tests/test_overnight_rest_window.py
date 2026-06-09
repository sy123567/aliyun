"""Regression tests for overnight rest-window enforcement.

Root cause they guard against: an overnight rest preference such as
"每夜 21:00 至次日 06:00 必须停车休息" used to be parsed into only the morning
half ``rest_window=(0, 360)`` (00:00-06:00). The evening half (21:00-24:00) was
silently dropped, so nothing stopped the driver from taking orders / repositioning
in the evening. Evening enforcement relied on the LLM *also* emitting a
``no_drive_windows`` entry AND that entry passing keyword grounding — neither is
guaranteed — so the night-rest rule was violated every single day (92/92).

The fix derives a ``no_drive_window`` directly from the overnight rest window so
the scheduler always blocks evening driving and waits through the whole window,
independent of LLM nondeterminism.

Run: ``python demo/tests/test_overnight_rest_window.py`` (no pytest dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DAY_MINUTES,
    DriverRules,
    ModelDecisionService,
)


class _StubApi:
    """Minimal SimulationApiPort stub; rule parsing never touches the model."""

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {}


def _svc() -> ModelDecisionService:
    return ModelDecisionService(_StubApi())


# Real (noisy) D001 night-rest preference text. Crucially it contains NONE of the
# no_drive action keywords on its own, so evening enforcement must NOT depend on
# keyword grounding of this text.
NIGHT_TEXT = "每夜21:00至日6:00，必停熄休，段间得单空驶但周两可晚个时休。"


def test_llm_overnight_rest_window_adds_no_drive_window() -> None:
    """LLM emits ONLY rest_window (21->6); evening must still be enforced."""
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"rest_window": {"start_hour": 21, "end_hour": 6}}, [NIGHT_TEXT])
    assert rules.rest_window == (0, 360), rules.rest_window
    # Whole overnight span (21:00 -> next-day 06:00) stored as end = em + 1 day.
    assert (1260, 360 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows
    assert rules.day_rest_block == 360, rules.day_rest_block


def test_evening_driving_is_blocked_after_fix() -> None:
    """At 21:00/21:40/23:00 the driver may not take orders or reposition, and a
    wait starting at 21:00 is extended to cover the full window (-> 06:00)."""
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"rest_window": {"start_hour": 21, "end_hour": 6}}, [NIGHT_TEXT])

    plan = {"home_done": set()}
    day = 0
    day_start = day * DAY_MINUTES
    for tod in (1260, 1300, 1380):  # 21:00, 21:40, 23:00
        now = day_start + tod
        hard_end = svc._hard_order_deadline(rules, plan, now, day)
        in_window = any(svc._tod_in_window(tod, ws, we) for ws, we in rules.no_drive_windows)
        assert in_window, f"tod={tod} should be inside the no-drive window"
        assert not (hard_end > now and not in_window), f"tod={tod} must not allow ordering"

    # A short wait beginning at 21:00 must be extended to the window end (06:00).
    extended = svc._extend_wait_for_no_drive(rules, day_start + 1260, 60)
    assert 1260 + extended == 1800, (extended, 1260 + extended)


def test_daytime_rest_window_is_not_converted() -> None:
    """A daytime window (e.g. 11:00-13:30) is a normal rest_window and must NOT
    spawn a no_drive_window (that is handled separately) nor inflate day_rest_block."""
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"rest_window": {"start_hour": 11, "end_hour": 13.5}}, ["午间11点到13点半休息"])
    assert rules.rest_window == (660, 810), rules.rest_window
    assert rules.no_drive_windows == [], rules.no_drive_windows


def test_duplicate_no_drive_window_is_not_added_twice() -> None:
    """If the LLM also emits the same overnight window via no_drive_windows, the
    derived one must not create a duplicate entry."""
    svc = _svc()
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 1800))  # pre-existing identical window
    svc._apply_rest_window(rules, 21 * 60, 6 * 60)
    assert rules.no_drive_windows.count((1260, 1800)) == 1, rules.no_drive_windows


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
