"""Tests for the penalty fail-safes added after the finals re-run.

Measured signal: every branch built on master (#73 daily planner + the
preference-driven long-haul cap) scored 64.5k-71.5k penalty, while the
#72-only branch scored 46k. The two mechanisms that can RAISE penalties
relative to #72 are hardened here:

1. directive replacement (relaxation) of static no-drive windows now needs a
   geometric subset check + an explicit default-REJECT semantic verification —
   a hallucinated "today is relaxed" can no longer lift a real window (it
   falls back to the additive union, which never causes penalties);
2. _supplement_critical_rules: an always-on deterministic backstop for the
   high-stakes scalar rules (monthly long-haul cap, daily continuous rest,
   monthly full off-days) so that an extraction+audit miss can no longer turn
   into season-long repeat penalties. It runs even when the LLM compile
   succeeded (unlike the offline-only _supplement_basic_rules).

Run: ``python demo/tests/test_penalty_failsafes.py`` (no pytest dependency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DriverRules,
    ModelDecisionService,
)

NIGHT_STATIC = (1260, 1800)  # 21:00 -> next-day 06:00


def _resp(obj) -> dict:
    return {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}


class ScriptedApi:
    """Stub scripting the LLM roles by system prompt."""

    def __init__(self, extract_responses=None, directive_response=None,
                 verify_answer=False):
        self.verify_questions: list[str] = []
        self._extract_responses = list(extract_responses or [])
        self._directive_response = directive_response
        self._verify_answer = verify_answer

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": 600}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "方言转写助手" in system:
            return _resp({"mandarin": user})  # identity → no augmentation
        if "偏好抽取器" in system:
            data = self._extract_responses.pop(0) if self._extract_responses else {}
            return _resp(data)
        if "覆盖审计器" in system:
            return _resp({"audits": []})
        if "每日合规计划助手" in system:
            return _resp(self._directive_response or {})
        if "是否确实包含某条约束" in system:
            return _resp({"holds": True})
        if "语义判定助手" in system:
            self.verify_questions.append(user)
            return _resp({"answer": self._verify_answer})
        return _resp({})


def _status(prefs):
    return {"preferences": prefs, "current_lat": 23.0, "current_lng": 113.2,
            "simulation_progress_minutes": 600}


# ------------------------------------------- directive replacement hardening

def _directive_setup(verify_answer: bool, windows=None):
    api = ScriptedApi(
        directive_response={
            "no_drive_today": windows or [{"start_hour": 23, "end_hour": 6}],
            "replaces_default": True,
            "today_plan": "周末晚两小时",
            "category_focus": None,
        },
        verify_answer=verify_answer,
    )
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT_STATIC)
    svc._pref_records["DH"] = [{"content": "夜里九点后收车，周末可以晚两小时", "penalty_amount": 2700}]
    return api, svc, rules


def test_replacement_rejected_without_semantic_confirmation() -> None:
    """Verifier says no (or is unavailable → default False): the static window
    must survive via the additive union."""
    api, svc, rules = _directive_setup(verify_answer=False)
    plan: dict = {}
    svc._ensure_daily_directive("DH", rules, plan, day=6)
    directive = rules.daily_directives[6]
    assert directive["replace"] is False, directive
    eff = rules.no_drive_windows_for(6)
    assert NIGHT_STATIC in eff, eff
    assert api.verify_questions, "verification must have been consulted"


def test_replacement_accepted_with_confirmation_and_subset() -> None:
    """Verifier confirms AND the relaxed window lies inside the static one →
    replacement takes effect (weekend +2h income window)."""
    _api, svc, rules = _directive_setup(verify_answer=True)
    plan: dict = {}
    svc._ensure_daily_directive("DH", rules, plan, day=6)
    directive = rules.daily_directives[6]
    assert directive["replace"] is True, directive
    assert rules.no_drive_windows_for(6) == [(1380, 1800)], rules.no_drive_windows_for(6)


def test_replacement_rejected_when_window_not_subset() -> None:
    """Even with a confirming verifier, a 'relaxation' that extends OUTSIDE the
    static window (20:00 start vs static 21:00) is no relaxation — geometric
    gate rejects, union keeps both."""
    api, svc, rules = _directive_setup(
        verify_answer=True, windows=[{"start_hour": 20, "end_hour": 6}],
    )
    plan: dict = {}
    svc._ensure_daily_directive("DH", rules, plan, day=6)
    directive = rules.daily_directives[6]
    assert directive["replace"] is False, directive
    eff = rules.no_drive_windows_for(6)
    assert NIGHT_STATIC in eff and (1200, 1800) in eff, eff
    assert api.verify_questions == [], "geometric gate must reject before any LLM call"


# ----------------------------------------------- always-on critical backstop

def test_longhaul_cap_backstop_when_llm_misses() -> None:
    """Extraction (succeeded but empty) + audit both miss the cap; the always-on
    backstop must still register it — without it the cap-removal change turns
    a missed parse into unlimited over-cap orders × per-order penalty."""
    api = ScriptedApi(extract_responses=[{}])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DH2", _status([{
        "content": "不爱接那种一跑就是大半天、人困马乏的远活，每个月超过八小时的长途只能接最多5单，多一单扣一次。",
        "penalty_amount": 1000,
    }]))
    assert rules.longhaul_cap_orders == 5, rules.longhaul_cap_orders
    assert rules.longhaul_min_minutes == 480, rules.longhaul_min_minutes


def test_daily_rest_and_off_days_backstop_when_llm_misses() -> None:
    api = ScriptedApi(extract_responses=[{}, {}])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DH3", _status([
        {"content": "每天要连续休息满8小时才扛得住。", "penalty_amount": 400},
        {"content": "每个月得有两天完全歇着，不出车不接活。", "penalty_amount": 3000},
    ]))
    assert rules.daily_rest_minutes == 480, rules.daily_rest_minutes
    assert rules.off_days_min == 2, rules.off_days_min


def test_backstop_does_not_invent_caps() -> None:
    """A long-haul-FOCUSED driver must not be throttled by the backstop."""
    api = ScriptedApi(extract_responses=[{}])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DH4", _status([{
        "content": "我就爱跑长途，每个月跑得越多越好，短途不爱接。",
        "penalty_amount": None,
    }]))
    assert rules.longhaul_cap_orders is None, rules.longhaul_cap_orders
    assert rules.off_days_min == 0 and rules.daily_rest_minutes == 0


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
