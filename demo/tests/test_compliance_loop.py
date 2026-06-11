"""Regression tests for the closed-loop compliance architecture.

What they guard against: the old architecture was a ONE-SHOT pipeline —
parse natural-language preferences into a fixed schema, then hard-code the
execution. Anything the extractor missed (unknown phrasing, schema gaps) was
silently dropped and violated every single day for the whole season with no
detection mechanism, which is exactly what produced the six-figure penalty on
the unknown finals drivers (Guangdong / Yangtze-delta).

The new architecture closes the loop at three points, tested here:

1. Coverage verification: after extraction, an LLM reviewer compares the raw
   texts against a restatement of the structured rules; missing obligations get
   one focused repair pass, and whatever still cannot be structured is carried
   verbatim as a ``custom_directive`` (injected into decision prompts) instead
   of being dropped.
2. Daily compliance audit: once per simulated day, yesterday's actual action
   timeline is judged against the RAW preference text; reported violations may
   self-heal the rule set (add missed no-drive windows), so a systematic
   violation costs one day, not ninety-two.
3. Action shield: every outgoing action passes ``_guard_action``; driving
   actions inside a hard no-drive window are converted into compliant waits no
   matter which code path produced them.
4. Token budget governor: every model call is metered per driver; advisory
   calls throttle at the soft limit and stop at the hard limit, while
   compliance-critical calls keep running until the audit limit.

Run: ``python demo/tests/test_compliance_loop.py`` (no pytest dependency).
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
    TOKEN_AUDIT_LIMIT,
    TOKEN_HARD_LIMIT,
    TOKEN_SOFT_LIMIT,
    DriverRules,
    ModelDecisionService,
)


def _resp(payload: dict, total_tokens: int = 100) -> dict:
    return {
        "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
        "usage": {"total_tokens": total_tokens},
    }


class _ScriptedApi:
    """SimulationApiPort stub whose model returns scripted JSON payloads in order."""

    def __init__(self, model_payloads=None, records=None):  # noqa: ANN001
        self._payloads = list(model_payloads or [])
        self._records = records or []
        self.model_calls: list[dict] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": list(self._records)}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        self.model_calls.append(payload)
        if self._payloads:
            return _resp(self._payloads.pop(0))
        return {}


# --------------------------------------------------------------- action shield
def test_guard_blocks_take_order_inside_no_drive_window() -> None:
    svc = ModelDecisionService(_ScriptedApi())
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 360 + DAY_MINUTES))  # 21:00 -> next 06:00
    now = 5 * DAY_MINUTES + 1320  # day 5, 22:00
    out = svc._guard_action(rules, now, {"action": "take_order", "params": {"cargo_id": "X"}})
    assert out["action"] == "wait", out
    # the wait must cover the rest of the window: 22:00 -> 06:00 = 480 min
    assert out["params"]["duration_minutes"] >= 480, out


def test_guard_blocks_reposition_inside_window_and_passes_outside() -> None:
    svc = ModelDecisionService(_ScriptedApi())
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 360 + DAY_MINUTES))
    inside = svc._guard_action(
        rules, 1300, {"action": "reposition", "params": {"latitude": 23.0, "longitude": 113.0}}
    )
    assert inside["action"] == "wait", inside
    outside = svc._guard_action(
        rules, 600, {"action": "reposition", "params": {"latitude": 23.0, "longitude": 113.0}}
    )
    assert outside["action"] == "reposition", outside


def test_guard_extends_wait_that_ends_inside_window() -> None:
    svc = ModelDecisionService(_ScriptedApi())
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 360 + DAY_MINUTES))
    out = svc._guard_action(rules, 1200, {"action": "wait", "params": {"duration_minutes": 120}})
    # 20:00 + 120min would end 22:00 inside the window -> extended to 06:00
    assert out["params"]["duration_minutes"] == 1800 - 1200, out


# --------------------------------------------------------- daily compliance audit
def _night_violation_records() -> list[dict]:
    """Yesterday (day 0) the driver took an order at 23:00 — a night violation."""
    return [{
        "step_elapsed_minutes": 60,
        "query_scan_cost_minutes": 0,
        "action": {"action": "take_order", "params": {"cargo_id": "1"}},
        "result": {"accepted": True, "cargo_id": "1", "simulation_progress_minutes": 1440},
    }]


def test_audit_self_heals_missed_night_window() -> None:
    verdict = {
        "violations": ["昨日23:00在禁驶时段接单"],
        "add_no_drive_windows": [{"start_hour": 22, "end_hour": 6}],
        "notes": ["今晚22点前收车"],
    }
    api = _ScriptedApi(model_payloads=[verdict], records=_night_violation_records())
    svc = ModelDecisionService(api)
    svc._seen_prefs["D"] = {"入夜了就甭出车，天亮再说"}
    rules = DriverRules()
    plan: dict = {}
    svc._daily_compliance_audit("D", rules, plan, day=1)
    assert (1320, 360 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows
    assert plan.get("audit_notes") == ["今晚22点前收车"], plan.get("audit_notes")
    # idempotent: second call same day must not re-invoke the model
    calls_before = len(api.model_calls)
    svc._daily_compliance_audit("D", rules, plan, day=1)
    assert len(api.model_calls) == calls_before


def test_audit_without_violations_adds_no_windows() -> None:
    verdict = {"violations": [], "add_no_drive_windows": [{"start_hour": 8, "end_hour": 20}], "notes": []}
    api = _ScriptedApi(model_payloads=[verdict], records=_night_violation_records())
    svc = ModelDecisionService(api)
    svc._seen_prefs["D"] = {"x"}
    rules = DriverRules()
    svc._daily_compliance_audit("D", rules, {}, day=1)
    assert rules.no_drive_windows == [], rules.no_drive_windows


def test_audit_rejects_oversized_or_covered_windows() -> None:
    verdict = {
        "violations": ["v"],
        "add_no_drive_windows": [
            {"start_hour": 6, "end_hour": 23},   # 17h span: hallucination-sized
            {"start_hour": 22, "end_hour": 5},   # already covered by enforced window
        ],
        "notes": [],
    }
    api = _ScriptedApi(model_payloads=[verdict], records=_night_violation_records())
    svc = ModelDecisionService(api)
    svc._seen_prefs["D"] = {"x"}
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 360 + DAY_MINUTES))  # 21:00->06:00 enforced
    svc._daily_compliance_audit("D", rules, {}, day=1)
    assert rules.no_drive_windows == [(1260, 360 + DAY_MINUTES)], rules.no_drive_windows


def test_audit_failure_is_silent_noop() -> None:
    api = _ScriptedApi(model_payloads=None, records=_night_violation_records())  # model returns {}
    svc = ModelDecisionService(api)
    svc._seen_prefs["D"] = {"x"}
    rules = DriverRules()
    plan: dict = {}
    svc._daily_compliance_audit("D", rules, plan, day=1)
    assert rules.no_drive_windows == []
    assert "audit_notes" not in plan


# ------------------------------------------------------- coverage verification
def test_coverage_keeps_unstructured_obligation_as_custom_directive() -> None:
    # 1st reviewer call: one obligation missing; repair parse returns nothing
    # structurable; 2nd reviewer call: still missing -> custom directive.
    missing = {"missing": ["每逢初一十五要烧香，当天不接活"]}
    api = _ScriptedApi(model_payloads=[missing, {}, missing])
    svc = ModelDecisionService(api)
    rules = DriverRules()
    svc._verify_rule_coverage("D", ["每逢初一十五要烧香，当天不接活"], rules, None, None)
    assert rules.custom_directives == ["每逢初一十五要烧香，当天不接活"], rules.custom_directives
    # the directive must surface in the decision prompt rule block
    text = svc._format_rules_for_llm("D", rules, {"monthly_category_orders": {}, "monthly_longhual": {}}, day=0)
    assert "初一十五" in text, text


def test_coverage_repair_pass_can_structure_missing_rule() -> None:
    # reviewer finds a missing night window; the repair extraction emits it;
    # second review reports nothing missing -> no custom directive needed.
    payloads = [
        {"missing": ["每天23点后不接单不空驶"]},
        {"no_drive_windows": [{"start_hour": 23, "end_hour": 5}]},
        {"missing": []},
        # _confirm_rule_holds for the repaired window (semantic confirm)
    ]
    api = _ScriptedApi(model_payloads=payloads)
    svc = ModelDecisionService(api)
    # avoid an extra model call inside the merge's semantic confirm
    svc._confirm_rule_holds = lambda desc, all_text, default=True: True  # type: ignore[assignment]
    rules = DriverRules()
    svc._verify_rule_coverage("D", ["夜里23点往后就歇了"], rules, None, None)
    assert (1380, 300 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows
    assert rules.custom_directives == [], rules.custom_directives


def test_coverage_runs_once_per_preference_set() -> None:
    api = _ScriptedApi(model_payloads=[{"missing": []}])
    svc = ModelDecisionService(api)
    rules = DriverRules()
    svc._verify_rule_coverage("D", ["t1"], rules, None, None)
    calls = len(api.model_calls)
    svc._verify_rule_coverage("D", ["t1"], rules, None, None)  # same set: no new call
    assert len(api.model_calls) == calls
    svc._verify_rule_coverage("D", ["t1", "t2"], rules, None, None)  # new text: re-check
    assert len(api.model_calls) > calls


# --------------------------------------------------------- token budget governor
def test_chat_meters_tokens_per_driver() -> None:
    api = _ScriptedApi(model_payloads=[{"ok": 1}, {"ok": 1}])
    svc = ModelDecisionService(api)
    svc._current_driver = "D"
    svc._chat({"messages": []})
    svc._chat({"messages": []})
    assert svc._tokens_used("D") == 200, svc._tokens_used("D")


def test_budget_gates_throttle_and_stop_advisory_llm() -> None:
    svc = ModelDecisionService(_ScriptedApi())
    svc._token_usage["D"] = 0
    assert svc._llm_advice_allowed("D", step=3) is True
    svc._token_usage["D"] = TOKEN_SOFT_LIMIT
    assert svc._llm_advice_allowed("D", step=4) is True   # even steps only
    assert svc._llm_advice_allowed("D", step=5) is False
    svc._token_usage["D"] = TOKEN_HARD_LIMIT
    assert svc._llm_advice_allowed("D", step=4) is False
    assert svc._audit_allowed("D") is True               # compliance calls continue
    svc._token_usage["D"] = TOKEN_AUDIT_LIMIT
    assert svc._audit_allowed("D") is False


# ------------------------------------------------------------- decide() wiring
def test_decide_is_scheduler_first_and_always_guarded() -> None:
    """During a no-drive window decide() must emit a compliant wait and must not
    place any advisory model call."""
    api = _ScriptedApi()
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.no_drive_windows.append((1260, 360 + DAY_MINUTES))
    svc._rules["D"] = rules
    svc._seen_prefs["D"] = set()

    api.get_driver_status = lambda driver_id: {  # type: ignore[assignment]
        "simulation_progress_minutes": 1320,  # day 0, 22:00
        "current_lat": 23.0,
        "current_lng": 113.0,
        "preferences": [],
    }
    action = svc.decide("D")
    assert action["action"] == "wait", action
    # wait must cover up to 06:00 next day
    assert action["params"]["duration_minutes"] >= 1800 - 1320, action
    assert api.model_calls == [], "no advisory LLM call may happen inside a hard window"


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
