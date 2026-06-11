"""Tests for the LLM-primary decision architecture.

What changed: previously the LLM was only advisory (invoked every 3rd step /
after a wait) and the deterministic scheduler made most decisions. Now the
LLM — grounded in the decision history — decides EVERY step where a real
choice exists:

- forced moves (off day / mandatory rest / already inside a no-drive window)
  are short-circuited WITHOUT an LLM call (compliance is mechanical there);
- an action that fails deterministic validation is fed back to the model once
  (with the rejection reason) so it can self-correct before the rule-engine
  fallback;
- the history layer adds per-day activity rollups so the model sees multi-day
  trends, and the daily planner distills a strategy from yesterday's rollup +
  realised per-city yield.

Run: ``python demo/tests/test_llm_primary_decisions.py`` (no pytest).
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
    DecisionHistory,
    DriverRules,
    ModelDecisionService,
)

_DECIDE_MARK = "智能货运调度决策AI"
_DIRECTIVE_MARK = "每日合规计划助手"

_CARGO = {
    "cargo_id": "c1", "cargo_name": "水果", "price": 5000.0,
    "cost_time_minutes": 180,
    "start": {"city": "广州", "lat": 23.0, "lng": 113.2},
    "end": {"city": "佛山", "lat": 23.0, "lng": 113.9},
}


class _StubApi:
    """Scripted gateway: returns queued decision responses; counts call kinds."""

    def __init__(self, decisions=None):  # noqa: ANN001
        self.now = 8 * 60
        self.decisions = list(decisions or [])
        self.calls: list[str] = []

    def get_driver_status(self, d):  # noqa: ANN001, ANN201
        return {"driver_id": d, "simulation_progress_minutes": self.now,
                "current_lat": 23.0, "current_lng": 113.2, "preferences": []}

    def query_cargo(self, driver_id, latitude, longitude, k):  # noqa: ANN001, ANN201
        return {"items": [{"cargo": dict(_CARGO)}]}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        sysmsg = payload["messages"][0]["content"]
        if _DECIDE_MARK in sysmsg:
            self.calls.append("decide")
            content = self.decisions.pop(0) if self.decisions else '{"action":"wait","params":{"duration_minutes":60}}'
        elif _DIRECTIVE_MARK in sysmsg:
            self.calls.append("directive")
            content = "{}"
        else:
            self.calls.append("other")
            content = "{}"
        return {"choices": [{"message": {"content": content}}]}


def _svc(api) -> ModelDecisionService:  # noqa: ANN001
    return ModelDecisionService(api)


# ------------------------------------------------------------ every-step LLM

def test_llm_decides_every_step() -> None:
    """3 consecutive non-wait steps -> 3 decision LLM calls (no step%3 gating)."""
    api = _StubApi(decisions=['{"action":"take_order","params":{"cargo_id":"c1"}}'] * 3)
    svc = _svc(api)
    for _ in range(3):
        a = svc.decide("DX")
        assert a["action"] == "take_order", a
        api.now += 240
    assert api.calls.count("decide") == 3, api.calls


def test_forced_window_skips_llm_call() -> None:
    """Inside a no-drive window the step is mechanical: wait it out, 0 tokens."""
    api = _StubApi()
    svc = _svc(api)
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 1800))  # 21:00 -> 06:00
    svc._rules["DX"] = rules
    api.now = 22 * 60  # 22:00, inside the window
    a = svc.decide("DX")
    assert a["action"] == "wait", a
    # wait must extend through the window end (06:00 next day)
    assert api.now % DAY_MINUTES + a["params"]["duration_minutes"] >= 1800
    assert api.calls.count("decide") == 0, api.calls


def test_wait_loop_hard_stop_skips_llm() -> None:
    """After 4 consecutive waits the rule engine takes over (no LLM call)."""
    api = _StubApi()
    svc = _svc(api)
    svc._plan["DX"] = {
        "rest_done": set(), "zeng_order_days": set(), "dated_single_done": set(),
        "dated_route_done": set(), "strand_repo": set(), "orders_today": {},
        "total_deadhead_km": 0.0, "monthly_deadhead_km": {}, "must_visit_days": {},
        "first_order_taken": set(), "home_done": set(), "monthly_longhual": {},
        "monthly_category_orders": {}, "failed_cargo_ids": set(),
        "failed_cargo_reasons": {}, "_consecutive_waits": 4,
    }
    svc.decide("DX")
    assert api.calls.count("decide") == 0, api.calls


# ------------------------------------------------------- retry with feedback

def test_invalid_pick_is_retried_with_feedback_then_corrected() -> None:
    """1st decision picks an unknown cargo -> rejection reason fed back ->
    2nd attempt picks the valid one and is executed."""
    api = _StubApi(decisions=[
        '{"action":"take_order","params":{"cargo_id":"nope"}}',
        '{"action":"take_order","params":{"cargo_id":"c1"}}',
    ])
    svc = _svc(api)
    a = svc.decide("DX")
    assert a["action"] == "take_order" and a["params"]["cargo_id"] == "c1", a
    assert api.calls.count("decide") == 2, api.calls


def test_invalid_twice_falls_back_to_rule_engine() -> None:
    """Two invalid picks -> deterministic fallback still returns the best order."""
    api = _StubApi(decisions=[
        '{"action":"take_order","params":{"cargo_id":"nope"}}',
        'not even json',
    ])
    svc = _svc(api)
    a = svc.decide("DX")
    assert a["action"] in ("take_order", "wait", "reposition"), a
    assert api.calls.count("decide") == 2, api.calls
    # rule engine picked the feasible cargo deterministically
    assert a["action"] == "take_order" and a["params"]["cargo_id"] == "c1", a


# --------------------------------------------------- history-grounded prompt

def test_day_rollups_in_history_summary() -> None:
    h = DecisionHistory()
    h.record(step=1, day=3, tod=600, action="wait", params={"duration_minutes": 120})
    h.record(step=2, day=3, tod=720, action="reposition", params={"latitude": 23.0, "longitude": 113.0})
    h.record(step=3, day=4, tod=600, action="take_order", params={"cargo_id": "x"})
    text = h.get_summary(5, {"orders_today": {3: 0, 4: 2}})
    assert "近几日概况" in text, text
    assert "D3:成交0单/等待2.0h/空驶1次" in text, text
    assert "D4:成交2单" in text, text


def test_directive_payload_includes_history_and_strategy_lands_in_notes() -> None:
    captured = {}

    class _Api(_StubApi):
        def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
            sysmsg = payload["messages"][0]["content"]
            if _DIRECTIVE_MARK in sysmsg:
                captured["user"] = payload["messages"][1]["content"]
                return {"choices": [{"message": {"content": json.dumps({
                    "no_drive_today": [], "replaces_default": False,
                    "today_plan": "正常运营", "category_focus": None,
                    "strategy_today": "优先去广州装货",
                }, ensure_ascii=False)}}]}
            return super().model_chat_completion(payload)

    api = _Api()
    svc = _svc(api)
    rules = DriverRules()
    svc._pref_records["DX"] = [{"content": "正常跑车", "penalty_amount": None}]
    history = DecisionHistory()
    history.record(step=1, day=4, tod=600, action="wait", params={"duration_minutes": 180})
    plan = {"orders_today": {4: 1}, "city_yield": {"广州": [3, 9000.0, 600]}}
    svc._ensure_daily_directive("DX", rules, plan, 5, history)
    assert "历史概况" in captured["user"], captured["user"]
    assert "昨日" in captured["user"] and "广州" in captured["user"], captured["user"]
    notes = rules.daily_directives[5]["notes"]
    assert "今日策略：优先去广州装货" in notes, notes


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
