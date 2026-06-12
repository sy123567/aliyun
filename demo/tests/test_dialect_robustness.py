"""Tests for finals-driver robustness: dialect preference handling + crash safety.

The finals drivers are unknown (one 广东, one 江浙沪) and their preference texts
may be written in 粤语 / 吴语 or heavy colloquial phrasing. This suite covers:

1. dialect normalization: each new preference text gets a standard-Mandarin
   paraphrase that is appended to the text fed to extraction / audit /
   grounding / the offline regex fallback, and stored on the pref record for
   the daily compliance planner;
2. fail-safety of the normalization layer: when the normalizer is unavailable
   or returns the identical text, the pipeline behaves exactly as before;
3. deterministic dialect category synonyms (粤语「生果」=「水果」…) that match
   without an LLM round-trip;
4. the "今天" calendar hint in the extraction payload ("本月/呢个月" targets
   resolve to the right month);
5. crash safety: _pick_order with a monthly deadhead cap must not raise
   (regression for a NameError that aborted the whole driver simulation), and
   decide() must degrade to a safe wait instead of propagating exceptions.

Run: ``python demo/tests/test_dialect_robustness.py`` (no pytest dependency).
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


PREF_CANTONESE = "晚黑十点之后唔好再接单走车，要返屋企熄火休息，挨到朝早六点先至好开工。"
MANDARIN_CANTONESE = "每天晚上22点之后到第二天早上6点不接单不空驶，必须回家熄火休息。"
PREF_WU = "夜里向十一点到第二天早浪向五点覅跑车，停勒屋里厢困觉。"


class DialectApi:
    """SimulationApiPort stub that scripts the LLM roles by system prompt."""

    def __init__(self, mandarin_map=None, extract_responses=None, audit_response=None,
                 normalize_fails=False):
        self.normalize_inputs: list[str] = []
        self.extract_payloads: list[dict] = []
        self.audit_payloads: list[dict] = []
        self.directive_payloads: list[dict] = []
        self.semantic_calls = 0
        self._mandarin_map = dict(mandarin_map or {})
        self._extract_responses = list(extract_responses or [])
        self._audit_response = audit_response
        self._normalize_fails = normalize_fails

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
            if self._normalize_fails:
                raise RuntimeError("normalizer down")
            self.normalize_inputs.append(user)
            return _resp({"mandarin": self._mandarin_map.get(user, user)})
        if "偏好抽取器" in system:
            self.extract_payloads.append(json.loads(user))
            data = self._extract_responses.pop(0) if self._extract_responses else {}
            return _resp(data)
        if "覆盖审计器" in system:
            self.audit_payloads.append(json.loads(user))
            return _resp(self._audit_response or {"audits": []})
        if "每日合规计划助手" in system:
            self.directive_payloads.append(json.loads(user))
            return _resp({"no_drive_today": [], "replaces_default": False,
                          "today_plan": "", "category_focus": None})
        if "是否确实包含某条约束" in system:
            return _resp({"holds": True})
        if "语义判定助手" in system:
            self.semantic_calls += 1
            return _resp({"answer": False})
        return _resp({})


def _status(prefs):
    return {"preferences": prefs, "current_lat": 23.0, "current_lng": 113.2,
            "simulation_progress_minutes": 600}


def test_dialect_pref_is_normalized_and_augmented() -> None:
    """A Cantonese preference is paraphrased; extraction sees 原文+释义 and 今天."""
    api = DialectApi(
        mandarin_map={PREF_CANTONESE: MANDARIN_CANTONESE},
        extract_responses=[{"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG01", _status([
        {"content": PREF_CANTONESE, "penalty_amount": 2700},
    ]))
    assert api.normalize_inputs == [PREF_CANTONESE], api.normalize_inputs
    assert len(api.extract_payloads) == 1, api.extract_payloads
    sent = api.extract_payloads[0]["preferences"][0]
    assert PREF_CANTONESE in sent and MANDARIN_CANTONESE in sent, sent
    assert "普通话释义" in sent, sent
    # the calendar hint lets "本月/呢个月" targets resolve to the right month
    assert "2026-03-01" in api.extract_payloads[0].get("今天", ""), api.extract_payloads[0]
    # the parsed cross-midnight window is enforced (22:00 → 次日06:00)
    assert (1320, 1800) in rules.no_drive_windows, rules.no_drive_windows
    # the audit also worked from the augmented text
    assert api.audit_payloads and MANDARIN_CANTONESE in api.audit_payloads[0]["preferences"][0]
    # the record carries the paraphrase for the daily planner
    rec = svc._pref_records["DG01"][0]
    assert rec.get("mandarin") == MANDARIN_CANTONESE, rec


def test_daily_directive_receives_mandarin_paraphrase() -> None:
    """The daily compliance planner payload carries 原文 + 普通话释义."""
    api = DialectApi(
        mandarin_map={PREF_CANTONESE: MANDARIN_CANTONESE},
        extract_responses=[{"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG02", _status([
        {"content": PREF_CANTONESE, "penalty_amount": 2700},
    ]))
    plan: dict = {}
    svc._ensure_daily_directive("DG02", rules, plan, day=0)
    assert api.directive_payloads, "directive call expected"
    pref_entry = api.directive_payloads[0]["偏好"][0]
    assert pref_entry["原文"] == PREF_CANTONESE, pref_entry
    assert pref_entry.get("普通话释义") == MANDARIN_CANTONESE, pref_entry


def test_normalizer_unavailable_pipeline_unchanged() -> None:
    """Normalizer down → extraction still runs with the raw text only."""
    api = DialectApi(
        normalize_fails=True,
        extract_responses=[{"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG03", _status([
        {"content": PREF_CANTONESE, "penalty_amount": 2700},
    ]))
    sent = api.extract_payloads[0]["preferences"][0]
    assert sent == PREF_CANTONESE, sent
    assert (1320, 1800) in rules.no_drive_windows, rules.no_drive_windows
    assert "mandarin" not in svc._pref_records["DG03"][0]


def test_identity_paraphrase_not_appended() -> None:
    """Already-standard text → no 释义 suffix, payload is the raw text."""
    plain = "每天22点到次日6点停车休息，不接单不空驶。"
    api = DialectApi(
        extract_responses=[{"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    svc._ensure_rules("DG04", _status([{"content": plain, "penalty_amount": 2700}]))
    sent = api.extract_payloads[0]["preferences"][0]
    assert sent == plain, sent


def test_dialect_category_synonyms_match_without_llm() -> None:
    """粤语「生果」↔「水果」 matches deterministically (no LLM round-trip)."""
    api = DialectApi()
    svc = ModelDecisionService(api)
    assert svc._category_matches_sem("生果", "水果")
    assert svc._category_matches_sem("水果", "生果")
    assert svc._category_matches_sem("菜蔬", "蔬菜")
    assert svc._category_matches_sem("建筑材料", "建材")
    assert api.semantic_calls == 0, "synonym map must short-circuit the LLM"


class DeadheadApi(DialectApi):
    """Stub that returns one profitable nearby cargo for _pick_order."""

    CARGO = {
        "cargo_id": "C-1",
        "cargo_name": "水果",
        "price": 2000.0,
        "cost_time_minutes": 120,
        "start": {"lat": 23.05, "lng": 113.25, "city": "广州"},
        "end": {"lat": 23.5, "lng": 113.8, "city": "广州"},
    }

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": [{"distance_km": 7.6, "cargo": dict(self.CARGO)}]}


def _fresh_plan() -> dict:
    return {
        "orders_today": {},
        "monthly_longhual": {},
        "monthly_category_orders": {},
        "monthly_deadhead_km": {},
        "zeng_order_days": set(),
        "failed_cargo_ids": set(),
        "off_days": set(),
    }


def test_pick_order_with_deadhead_cap_does_not_crash() -> None:
    """Regression: monthly_deadhead_max_km used to raise NameError inside
    _pick_order, which aborted the driver's whole month (decide() exceptions
    are fatal to the orchestrator loop)."""
    api = DeadheadApi()
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.monthly_deadhead_max_km = 1.0  # tiny cap → order must be rejected, not crash
    action = svc._pick_order("DG05", _status([]), rules, _fresh_plan(),
                             now=600, lat=23.0, lng=113.2, day=0, day_end=1440)
    assert action is None, action

    rules2 = DriverRules()
    rules2.monthly_deadhead_max_km = 1000.0  # generous cap → order goes through
    action2 = svc._pick_order("DG05", _status([]), rules2, _fresh_plan(),
                              now=600, lat=23.0, lng=113.2, day=0, day_end=1440)
    assert action2 is not None and action2["action"] == "take_order", action2
    assert action2["params"]["cargo_id"] == "C-1", action2


class ExplodingApi(DialectApi):
    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        raise RuntimeError("status endpoint down")


def test_decide_crash_falls_back_to_safe_wait() -> None:
    """decide() must never propagate: an exception ends the driver's whole
    simulation, so it degrades to a short safe wait instead."""
    svc = ModelDecisionService(ExplodingApi())
    action = svc.decide("DG06")
    assert action == {"action": "wait", "params": {"duration_minutes": 30}}, action


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
        except Exception as exc:  # noqa: BLE001 — regression tests assert "no crash"
            failures += 1
            print(f"FAIL {t.__name__}: unexpected {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
