"""Tests for the daily compliance planner (per-day LLM directive layer).

Architecture under test: instead of relying solely on a one-shot NL→schema
parse (which silently fails on unseen finals drivers whose preferences are
phrased differently or are calendar-conditional), the agent asks the LLM once
per simulation day to concretize the *raw* preference texts into "today's"
constraints. Those windows are enforced deterministically through
``DriverRules.no_drive_windows_for``:

- additive by default (fail-safe: a missed daily window costs a recurring
  daily penalty, an extra one only costs a few idle hours);
- a directive may *replace* the static windows (e.g. weekend night rest that
  starts later) only when it acknowledges every static window;
- any planner failure leaves the static parse in charge.

Run: ``python demo/tests/test_daily_directive.py`` (no pytest dependency).
"""

from __future__ import annotations

import json
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

NIGHT_STATIC = (1260, 1800)  # 21:00 -> next-day 06:00
WEEKEND_NIGHT = (1380, 1800)  # 23:00 -> next-day 06:00


class _StubApi:
    """SimulationApiPort stub returning a canned model response.

    ``chat_calls`` counts only directive-planner calls (the replacement
    verification is a separate semantic yes/no call, counted in
    ``verify_calls``), so the per-day caching assertions keep their meaning.
    """

    def __init__(self, content: str | None = None, fail: bool = False,
                 verify_answer: bool = True):
        self.content = content
        self.fail = fail
        self.chat_calls = 0
        self.verify_calls = 0
        self.verify_answer = verify_answer

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        system = payload["messages"][0]["content"]
        if "语义判定助手" in system:
            self.verify_calls += 1
            return {"choices": [{"message": {"content": json.dumps({"answer": self.verify_answer})}}]}
        self.chat_calls += 1
        if self.fail:
            raise RuntimeError("model unavailable")
        return {"choices": [{"message": {"content": self.content}}]}


def _svc(api=None) -> ModelDecisionService:
    return ModelDecisionService(api or _StubApi())


# --------------------------------------------------------------- window merge

def test_no_directive_returns_static() -> None:
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    assert rules.no_drive_windows_for(3) == [NIGHT_STATIC]
    assert rules.has_no_drive(3) and rules.has_any_no_drive()


def test_union_when_not_replacing() -> None:
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[5] = {"windows": [(720, 840)], "replace": False, "notes": ""}
    eff = rules.no_drive_windows_for(5)
    assert NIGHT_STATIC in eff and (720, 840) in eff, eff
    # other days untouched
    assert rules.no_drive_windows_for(6) == [NIGHT_STATIC]


def test_replace_accepted_when_every_static_window_is_acknowledged() -> None:
    """Weekend relaxation: static 21:00-06:00, directive replaces it with
    23:00-06:00 (the two intersect, so the replacement is accepted)."""
    svc = _svc()
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[6] = {"windows": [WEEKEND_NIGHT], "replace": True, "notes": ""}
    eff = rules.no_drive_windows_for(6)
    assert eff == [WEEKEND_NIGHT], eff
    # 21:30 is now allowed, 23:30 still blocked
    assert not any(svc._tod_in_window(1290, ws, we) for ws, we in eff)
    assert any(svc._tod_in_window(1410, ws, we) for ws, we in eff)


def test_replace_rejected_when_dropping_a_static_window() -> None:
    """A directive that silently drops the standing night window must NOT be
    able to lift it: effective set falls back to the additive union."""
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    # 06:00-10:00 does not intersect 21:00->06:00 in time-of-day space
    rules.daily_directives[7] = {"windows": [(360, 600)], "replace": True, "notes": ""}
    eff = rules.no_drive_windows_for(7)
    assert NIGHT_STATIC in eff and (360, 600) in eff, eff


def test_replace_with_empty_windows_keeps_static() -> None:
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[8] = {"windows": [], "replace": True, "notes": "无约束"}
    assert rules.no_drive_windows_for(8) == [NIGHT_STATIC]


def test_replace_rejected_when_coverage_collapses() -> None:
    """A trivially small replacement window (intersecting but covering far
    less than the static constraint) must not lift the static window."""
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)  # 9h coverage
    rules.daily_directives[9] = {"windows": [(1260, 1290)], "replace": True, "notes": ""}
    eff = rules.no_drive_windows_for(9)
    assert NIGHT_STATIC in eff, eff


# ------------------------------------------------------------ window coercion

def test_coerce_directive_window() -> None:
    coerce = ModelDecisionService._coerce_directive_window
    assert coerce({"start_hour": 21, "end_hour": 6}) == (1260, 1800)
    assert coerce({"start_hour": 0, "end_hour": 6}) == (0, 360)
    assert coerce({"start_hour": 22.5, "end_hour": 24}) == (1350, 1440)
    assert coerce({"start_hour": 11, "end_hour": 13.5}) == (660, 810)
    assert coerce({"start_hour": 9, "end_hour": 9}) is None
    assert coerce({"start_hour": "x", "end_hour": 6}) is None
    assert coerce({"start_hour": 25, "end_hour": 6}) is None
    assert coerce("21-6") is None


# -------------------------------------------------------------------- planner

def test_planner_applies_directive_and_caches_per_day() -> None:
    api = _StubApi(content=json.dumps({
        "no_drive_today": [{"start_hour": 23, "end_hour": 6}],
        "replaces_default": True,
        "today_plan": "周六夜休可推迟到23点；本月水果还差3单",
        "category_focus": "水果",
    }, ensure_ascii=False))
    svc = _svc(api)
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    svc._pref_records["DX"] = [{"content": "夜里不出车，周末可以晚两小时", "penalty_amount": 2700}]
    plan: dict = {}

    svc._ensure_daily_directive("DX", rules, plan, 6)
    directive = rules.daily_directives.get(6)
    assert directive is not None
    assert directive["windows"] == [WEEKEND_NIGHT], directive
    assert directive["replace"] is True
    assert "水果" in directive["notes"]
    assert rules.no_drive_windows_for(6) == [WEEKEND_NIGHT]

    calls = api.chat_calls
    svc._ensure_daily_directive("DX", rules, plan, 6)  # same day -> cached
    assert api.chat_calls == calls, "directive must be computed once per day"
    svc._ensure_daily_directive("DX", rules, plan, 7)  # new day -> recomputed
    assert api.chat_calls == calls + 1


def test_planner_failure_falls_back_to_static_with_bounded_retries() -> None:
    api = _StubApi(fail=True)
    svc = _svc(api)
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    svc._pref_records["DX"] = [{"content": "晚九点后收车", "penalty_amount": None}]
    plan: dict = {}
    for _ in range(5):
        svc._ensure_daily_directive("DX", rules, plan, 0)
    assert rules.daily_directives == {}, "failed planner must not register a directive"
    assert rules.no_drive_windows_for(0) == [NIGHT_STATIC]
    # ≤2 attempts/day, each attempt makes ≤2 transport calls (format retry)
    assert api.chat_calls <= 4, api.chat_calls


def test_planner_skipped_without_preferences() -> None:
    api = _StubApi(content="{}")
    svc = _svc(api)
    rules = DriverRules()
    svc._ensure_daily_directive("DX", rules, {}, 0)
    assert api.chat_calls == 0


# ------------------------------------------------------- guard-level coverage

def test_directive_only_window_is_enforced_by_guards() -> None:
    """Even with NO statically parsed windows (schema parse missed the rule
    entirely), a directive window must block ordering and extend waits."""
    svc = _svc()
    rules = DriverRules()
    rules.daily_directives[0] = {"windows": [NIGHT_STATIC], "replace": False, "notes": ""}

    # wait at 21:00 must be extended through the window end (next 06:00)
    extended = svc._extend_wait_for_no_drive(rules, 1260, 30)
    assert 1260 + extended == 1800, extended

    # ordering deadline on day 0 must stop before 21:00
    hard_end = svc._hard_order_deadline(rules, {"home_done": set()}, 600, 0)
    assert hard_end <= 1260, hard_end

    # absolute-interval overlap honours the directive day anchoring
    assert svc._interval_overlaps_no_drive(rules, 1250, 1290)
    assert not svc._interval_overlaps_no_drive(rules, 600, 700)
    # day 1 has no directive -> 21:00 interval there is free
    assert not svc._interval_overlaps_no_drive(
        rules, DAY_MINUTES + 1850, DAY_MINUTES + 1900
    )


def test_weekend_replacement_relaxes_deadline() -> None:
    """With an accepted weekend replacement (23:00 start) the order deadline
    moves from 21:00 to 23:00 — recovering income lost to over-conservatism."""
    svc = _svc()
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    rules.daily_directives[6] = {"windows": [WEEKEND_NIGHT], "replace": True, "notes": ""}
    plan = {"home_done": set()}
    weekday_end = svc._hard_order_deadline(rules, plan, 5 * DAY_MINUTES + 600, 5)
    weekend_end = svc._hard_order_deadline(rules, plan, 6 * DAY_MINUTES + 600, 6)
    assert weekday_end <= 5 * DAY_MINUTES + 1260
    assert weekend_end > 6 * DAY_MINUTES + 1260, weekend_end
    assert weekend_end <= 6 * DAY_MINUTES + 1380


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
