"""Tests for the closed-loop preference compilation architecture.

Architecture under test (generalisation to unknown finals drivers — 广东 / 江浙沪 —
whose natural-language preferences differ from the local D001 fixture):

1. per-preference compile: every preference text gets its own LLM extraction
   call (grounding checks run against the right text, no cross-pref noise);
2. audit & repair: a second LLM pass verifies that every obligation in each
   text is represented by the structured rules and patches misses — replacing
   the old D001-phrasing regex supplement layer;
3. no silent drop: anything still unrepresentable lands in
   ``rules.residual_constraints`` and is injected into the decision prompt;
4. semantic gates: scalar rules (daily_order_limit, haul_max_km, home_rule,
   allowed_regions, ...) are accepted/rejected by meaning, with the old keyword
   heuristics only as the offline default;
5. offline fallback: with the model unavailable the deterministic regex parser
   still produces rules.

Run: ``python demo/tests/test_preference_compile_loop.py`` (no pytest dependency).
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


def _resp(obj) -> dict:
    return {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}


class ScriptedApi:
    """SimulationApiPort stub that scripts the three LLM roles by system prompt."""

    def __init__(self, extract_responses=None, audit_response=None, confirm_holds=True,
                 fail_all=False):
        self.extract_payloads: list[dict] = []
        self.audit_payloads: list[dict] = []
        self.confirm_questions: list[str] = []
        self._extract_responses = list(extract_responses or [])
        self._audit_response = audit_response
        self._confirm_holds = confirm_holds
        self._fail_all = fail_all

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        if self._fail_all:
            raise RuntimeError("model gateway down")
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "偏好抽取器" in system:
            self.extract_payloads.append(json.loads(user))
            data = self._extract_responses.pop(0) if self._extract_responses else {}
            return _resp(data)
        if "覆盖审计器" in system:
            self.audit_payloads.append(json.loads(user))
            return _resp(self._audit_response or {"audits": []})
        if "是否确实包含某条约束" in system:
            self.confirm_questions.append(user)
            return _resp({"holds": self._confirm_holds})
        if "语义判定助手" in system:
            return _resp({"answer": True})
        return _resp({})


def _status(prefs):
    return {"preferences": prefs, "current_lat": 31.23, "current_lng": 121.47}


# Off-vocabulary (江浙沪/广东-flavoured) phrasings — deliberately NOT matching the
# old keyword whitelists nor the D001 fixture wording.
PREF_NIGHT = "天黑透了就收拾停当，廿二点之后到第二天早上六点，车子动也不动。"
PREF_OFFDAYS = "成个月里头起码两日完全唔开工，留返屋企陪屋企人。"
PREF_RAIN = "但凡落大雨嗰日，生鲜嘅货一律唔好同我派。"


def test_per_pref_compile_calls_llm_once_per_text() -> None:
    """Each preference text is compiled individually (not batched), with a
    3-sample ensemble per text whose results merge accretively."""
    api = ScriptedApi(
        extract_responses=[
            # pref 1: only one of the three samples catches the night window
            {"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]},
            {},
            {},
            # pref 2: samples agree on the off-days quota
            {"off_days_min": 2},
            {"off_days_min": 2},
            {},
        ],
        audit_response={"audits": [{"index": 0, "covered": True},
                                   {"index": 1, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX01", _status([
        {"content": PREF_NIGHT, "penalty_amount": 2700},
        {"content": PREF_OFFDAYS, "penalty_amount": 3000},
    ]))
    assert len(api.extract_payloads) == 6, api.extract_payloads
    for payload in api.extract_payloads:
        assert len(payload["preferences"]) == 1, payload
    assert (1320, 1800) in rules.no_drive_windows, rules.no_drive_windows
    assert rules.off_days_min == 2, rules.off_days_min
    assert rules.residual_constraints == [], rules.residual_constraints
    # idempotent: same prefs again -> no new extraction calls
    svc._ensure_rules("DX01", _status([
        {"content": PREF_NIGHT, "penalty_amount": 2700},
        {"content": PREF_OFFDAYS, "penalty_amount": 3000},
    ]))
    assert len(api.extract_payloads) == 6, "already-seen prefs must not be re-compiled"


def test_audit_repairs_missed_daily_window() -> None:
    """Extraction misses the night window entirely; the audit patch restores it."""
    api = ScriptedApi(
        extract_responses=[{}],  # extraction missed everything
        audit_response={"audits": [{
            "index": 0,
            "covered": False,
            "representable": True,
            "missing": "缺每日22:00-06:00禁驶窗",
            "patch": {"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]},
        }]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX02", _status([
        {"content": PREF_NIGHT, "penalty_amount": 2700},
    ]))
    assert (1320, 1800) in rules.no_drive_windows, rules.no_drive_windows
    # repaired -> not residual
    assert rules.residual_constraints == [], rules.residual_constraints
    # the audit saw the structured-rules dump and the penalty amount
    assert api.audit_payloads and "structured_rules" in api.audit_payloads[0]
    assert api.audit_payloads[0]["penalty_amounts"] == [2700]


def test_unrepresentable_pref_becomes_residual_and_reaches_prompt() -> None:
    """A constraint the schema cannot express is NEVER silently dropped: it must
    surface in the decision prompt and the daily self-audit block."""
    api = ScriptedApi(
        extract_responses=[{}],
        audit_response={"audits": [{
            "index": 0,
            "covered": False,
            "representable": False,
            "missing": "下雨天不接生鲜——天气条件无法结构化",
        }]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX03", _status([
        {"content": PREF_RAIN, "penalty_amount": 800},
    ]))
    assert len(rules.residual_constraints) == 1, rules.residual_constraints
    rc = rules.residual_constraints[0]
    assert rc["text"] == PREF_RAIN and rc["penalty"] == 800.0, rc

    prompt = svc._format_rules_for_llm("DX03", rules, {}, day=0)
    assert PREF_RAIN in prompt, prompt
    assert "800" in prompt and "未结构化偏好" in prompt, prompt

    from agent.model_decision_service import DecisionHistory
    audit_text = svc._compliance_self_audit(rules, DecisionHistory(), day=1)
    assert "未结构化偏好" in audit_text, audit_text


def test_semantic_confirm_accepts_offvocab_daily_limit() -> None:
    """daily_order_limit with phrasing outside the old keyword whitelist is kept
    when the semantic verifier confirms it."""
    api = ScriptedApi(confirm_holds=True)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"daily_order_limit": 3}, ["一日顶天接三票活，多咗唔制。"])
    assert rules.daily_order_limit == 3, rules.daily_order_limit


def test_semantic_confirm_rejects_hallucinated_daily_limit() -> None:
    """Even with whitelist keywords present, a clear verifier 'no' drops the rule."""
    api = ScriptedApi(confirm_holds=False)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"daily_order_limit": 3}, ["每天最多休息三次。"])
    assert rules.daily_order_limit is None, rules.daily_order_limit


def test_offline_fallback_uses_regex_parser() -> None:
    """Model unavailable -> the deterministic regex fallback still yields rules."""
    api = ScriptedApi(fail_all=True)
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX04", _status([
        {"content": "每天必须连续休息满8小时才扛得住。", "penalty_amount": 400},
    ]))
    assert rules.daily_rest_minutes == 480, rules.daily_rest_minutes


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
