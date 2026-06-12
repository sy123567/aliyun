"""Tests for dialect-robust preference compilation and gross-profit fixes.

Finals context: the two evaluation drivers are 广东 / 江浙沪 and their
natural-language preferences may be dialect (粤语 / 吴语), colloquial or
scrambled text. These tests cover:

1. Mandarin normalization pre-pass: every new preference is rewritten once and
   the rewrite is consumed *alongside* the raw text by extraction and the
   regex fail-safes (recall only, never replacing the raw text);
2. fail-safe: normalizer unavailable -> the pipeline runs raw-only, unchanged;
3. month-aware dated events: an explicit month anchors the event to the right
   calendar month, and a parsed day that already passed is re-anchored to the
   soonest month >= today (a visible preference cannot demand the past);
4. per-calendar-month off-day planning ("每月至少N天完全不出车" is judged per
   month, so the N days must land in every month);
5. deterministic-first decision flow: a feasible order is taken without any
   decision-LLM call; the LLM is consulted only as an idle rescue;
6. past-midnight finish allowance for drivers with no daily time constraints,
   applied consistently at scoring and at the pre-take revalidation.

Run: ``python demo/tests/test_dialect_and_profit.py`` (no pytest dependency).
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
    _MONTH_START_DAYS,
    _month_index_for_day,
)


def _resp(obj) -> dict:
    return {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}


# Cantonese night-rest phrasing: Chinese-numeral clock times, so the digit-based
# regex fail-safe can only fire through the Mandarin rewrite.
PREF_NIGHT_YUE = "夜晚黑十一点之后唔好同我派单，返屋企瞓觉，瞓到朝早六点先开工。"
NORM_NIGHT = "每天夜里23点之后不再接单，回家睡觉，睡到早上6点才出车。"


class ScriptedApi:
    """SimulationApiPort stub that scripts each LLM role by system prompt."""

    def __init__(self, extract_responses=None, audit_response=None,
                 normalize_response=None, normalize_fails=False):
        self.extract_payloads: list[dict] = []
        self.normalize_calls = 0
        self._extract_responses = list(extract_responses or [])
        self._audit_response = audit_response
        self._normalize_response = normalize_response
        self._normalize_fails = normalize_fails

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "普通话转写器" in system:
            self.normalize_calls += 1
            if self._normalize_fails:
                raise RuntimeError("normalizer down")
            return _resp(self._normalize_response or {})
        if "偏好抽取器" in system:
            self.extract_payloads.append(json.loads(user))
            data = self._extract_responses.pop(0) if self._extract_responses else {}
            return _resp(data)
        if "覆盖审计器" in system:
            return _resp(self._audit_response or {"audits": []})
        if "是否确实包含某条约束" in system:
            return _resp({"holds": True})
        if "语义判定助手" in system:
            return _resp({"answer": True})
        return _resp({})


def _status(prefs, now_minutes: int = 0):
    return {
        "preferences": prefs,
        "current_lat": 23.0,
        "current_lng": 113.2,
        "simulation_progress_minutes": now_minutes,
    }


def test_normalizer_feeds_extraction_and_failsafe() -> None:
    """The rewrite reaches the extraction payload AND the night fail-safe."""
    api = ScriptedApi(
        extract_responses=[{}],  # extraction missed the night window entirely
        audit_response={"audits": [{"index": 0, "covered": True}]},
        normalize_response={"normalized": NORM_NIGHT},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DY01", _status([
        {"content": PREF_NIGHT_YUE, "penalty_amount": 2700},
    ]))
    assert api.normalize_calls == 1, api.normalize_calls
    assert api.extract_payloads[0].get("normalized_preferences") == [NORM_NIGHT], \
        api.extract_payloads[0]
    # digit clock times only exist in the rewrite -> fail-safe must still fire
    assert (1380, 1800) in rules.no_drive_windows, rules.no_drive_windows
    # idempotent: same pref again -> no second normalization
    svc._ensure_rules("DY01", _status([{"content": PREF_NIGHT_YUE, "penalty_amount": 2700}]))
    assert api.normalize_calls == 1, "already-seen prefs must not be re-normalized"


def test_normalizer_failure_falls_back_to_raw() -> None:
    """Normalizer down -> extraction still runs on the raw text only."""
    api = ScriptedApi(
        extract_responses=[{"no_drive_windows": [{"start_hour": 23, "end_hour": 6}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
        normalize_fails=True,
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DY02", _status([
        {"content": PREF_NIGHT_YUE, "penalty_amount": 2700},
    ]))
    assert "normalized_preferences" not in api.extract_payloads[0], api.extract_payloads[0]
    assert (1380, 1800) in rules.no_drive_windows, rules.no_drive_windows


def test_dated_event_month_anchoring() -> None:
    """An explicit month anchors the event into the right calendar month."""
    svc = ModelDecisionService(ScriptedApi())
    rules = DriverRules()
    svc._merge_llm_rules(
        rules,
        {"dated_single": [{"date": 12, "month": 5, "lat": 23.15, "lng": 113.67,
                           "wait_minutes": 120}]},
        ["五月十二号去仓库（23.15，113.67）盘库两个钟头"],
    )
    assert rules.dated_single and rules.dated_single[0]["day"] == _MONTH_START_DAYS[2] + 11, \
        rules.dated_single


def test_past_dated_event_reanchored() -> None:
    """No explicit month + already-past day -> shifted to the soonest month."""
    api = ScriptedApi(
        extract_responses=[{"dated_single": [{"date": 3, "lat": 23.15, "lng": 113.67,
                                              "wait_minutes": 120}]}],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    # preference becomes visible on day 35 (April): "三号" cannot mean March 3rd
    rules = svc._ensure_rules("DY03", _status(
        [{"content": "三号要去仓库（23.15，113.67）盘点，约两个小时", "penalty_amount": 3000}],
        now_minutes=35 * DAY_MINUTES,
    ))
    assert rules.dated_single and rules.dated_single[0]["day"] == _MONTH_START_DAYS[2] + 2, \
        rules.dated_single


def test_existing_dated_event_not_reshifted() -> None:
    """A later compile pass must not re-anchor events from earlier passes."""
    api = ScriptedApi(
        extract_responses=[
            {"dated_single": [{"date": 3, "lat": 23.15, "lng": 113.67, "wait_minutes": 120}]},
            {},  # the later, unrelated preference
        ],
        audit_response={"audits": [{"index": 0, "covered": True}]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DY04", _status(
        [{"content": "三号去仓库（23.15，113.67）盘点两小时", "penalty_amount": 3000}],
        now_minutes=0,
    ))
    assert rules.dated_single[0]["day"] == 2, rules.dated_single
    # day 35: a new (date-windowed) preference appears; the old event already
    # completed on day 2 and must stay put.
    rules = svc._ensure_rules("DY04", _status(
        [
            {"content": "三号去仓库（23.15，113.67）盘点两小时", "penalty_amount": 3000},
            {"content": "四月里水果货至少接满十二单", "penalty_amount": 500},
        ],
        now_minutes=35 * DAY_MINUTES,
    ))
    assert rules.dated_single[0]["day"] == 2, rules.dated_single


def test_off_days_planned_per_calendar_month() -> None:
    """整休天数按自然月逐月铺点（每月N天，而不是整个赛季共N天）。"""
    rules = DriverRules()
    rules.off_days_min = 2
    off = ModelDecisionService._plan_off_days(rules)
    by_month = {0: 0, 1: 0, 2: 0}
    for d in off:
        by_month[_month_index_for_day(d)] += 1
    assert by_month == {0: 2, 1: 2, 2: 2}, (sorted(off), by_month)


# ---------------------------------------------------------------- decide() flow

CARGO_LATE = {
    "cargo": {
        "cargo_id": "C-LATE",
        "cargo_name": "建材",
        "price": 800.0,
        "cost_time_minutes": 600,  # 10h: started at 20:00 it finishes 06:00 next day
        "start": {"lat": 23.0, "lng": 113.2, "city": "佛山"},
        "end": {"lat": 23.4, "lng": 113.5, "city": "广州"},
        "load_time": None,
    },
    "distance_km": 0.0,
}


class SimStubApi:
    """Full decide() stub: fixed position/time, scripted cargo and LLM."""

    def __init__(self, items, now, decision_action=None, decision_fails=False,
                 preferences=None):
        self.items = items
        self.now = now
        self.decision_llm_calls = 0
        self.planner_calls = 0
        self._decision_action = decision_action
        self._decision_fails = decision_fails
        self._preferences = preferences or []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {
            "driver_id": driver_id,
            "simulation_progress_minutes": self.now,
            "current_lat": 23.0,
            "current_lng": 113.2,
            "preferences": self._preferences,
        }

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": self.items}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        system = payload["messages"][0]["content"]
        if "调度决策AI" in system:
            self.decision_llm_calls += 1
            if self._decision_fails:
                raise RuntimeError("decision llm down")
            return _resp(self._decision_action or {})
        if "每日合规计划助手" in system:
            self.planner_calls += 1
            return _resp({})
        if "是否确实包含某条约束" in system:
            return _resp({"holds": True})
        return _resp({})


def test_deterministic_take_order_skips_decision_llm() -> None:
    """A feasible order is taken directly — no decision-LLM call at all."""
    api = SimStubApi([CARGO_LATE], now=8 * 60)  # day 0, 08:00
    svc = ModelDecisionService(api)
    action = svc.decide("DZ01")
    assert action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "C-LATE", action
    assert api.decision_llm_calls == 0, api.decision_llm_calls


def test_prefless_driver_takes_past_midnight_order() -> None:
    """A driver with NO preferences at all may finish an order past midnight
    (nothing to violate), and the pre-take revalidation must agree with the
    scoring (regression: it used to recompute the bare deadline and reject
    every late pick)."""
    api = SimStubApi([CARGO_LATE], now=20 * 60)  # day 0, 20:00; finish 06:00 day 1
    svc = ModelDecisionService(api)
    action = svc.decide("DZ02")
    assert action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "C-LATE", action


def test_night_rest_driver_does_not_take_past_midnight_order() -> None:
    """A driver WITH a night window must still wrap up before the window."""
    api = SimStubApi([CARGO_LATE], now=20 * 60)
    svc = ModelDecisionService(api)
    rules = svc._rules.setdefault("DZ03", DriverRules())
    rules.no_drive_windows.append((21 * 60, 30 * 60))  # 21:00 -> 06:00
    action = svc.decide("DZ03")
    assert action["action"] == "wait", action


def test_driver_with_prefs_wraps_up_by_midnight() -> None:
    """Any driver WITH preferences wraps up by 24:00 even when no time rule was
    parsed — if the real rule is an unparsed/flattened night window, overnight
    hauls would violate it every single night (daily-compounding penalty)."""
    api = SimStubApi(
        [CARGO_LATE], now=20 * 60,
        preferences=[{"content": "成日要瞓够八个钟先顶得顺。", "penalty_amount": 400}],
    )
    svc = ModelDecisionService(api)
    action = svc.decide("DZ06")
    assert action["action"] != "take_order", action


def test_daily_planner_additive_only_in_decide() -> None:
    """The per-day planner runs (3 majority-vote samples) and can only ADD
    windows — replace semantics are permanently off after the round-2
    penalty regression."""
    api = SimStubApi(
        [CARGO_LATE], now=8 * 60,
        preferences=[{"content": "每天接单不超过3单。", "penalty_amount": 200}],
    )
    svc = ModelDecisionService(api)
    svc.decide("DZ07")
    assert api.planner_calls == 3, api.planner_calls
    directive = svc._rules["DZ07"].daily_directives.get(0)
    assert directive is not None and directive["replace"] is False, directive


def test_confirm_drop_requires_double_false() -> None:
    """A rule is only dropped when the verifier says 'false' twice; a single
    flaky 'false' (the old silent-delete path) keeps the rule."""

    class _ConfirmSeqApi:
        def __init__(self, verdicts):
            self.verdicts = list(verdicts)

        def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
            return {}

        def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            return {"items": []}

        def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
            return {"records": []}

        def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
            system = payload["messages"][0]["content"]
            if "是否确实包含某条约束" in system:
                return _resp({"holds": self.verdicts.pop(0)})
            return _resp({})

    # false then true -> kept (uncorroborated drop)
    svc = ModelDecisionService(_ConfirmSeqApi([False, True]))
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"forbidden_categories": ["生鲜"]}, ["生鲜嘅货唔好同我派"])
    assert "生鲜" in rules.forbidden_categories, rules.forbidden_categories

    # false twice -> dropped
    svc2 = ModelDecisionService(_ConfirmSeqApi([False, False]))
    rules2 = DriverRules()
    svc2._merge_llm_rules(rules2, {"forbidden_categories": ["生鲜"]}, ["每天最多接三单"])
    assert "生鲜" not in rules2.forbidden_categories, rules2.forbidden_categories


def test_residual_veto_downgrades_take_order() -> None:
    """With residual (unstructured) preferences, a clear LLM veto downgrades a
    deterministic take_order to a short wait — veto-only, so the LLM can
    refuse revenue but never cause a violation."""

    class _VetoApi(SimStubApi):
        def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
            system = payload["messages"][0]["content"]
            if "明显违反" in system:
                return _resp({"violates": True})
            return super().model_chat_completion(payload)

    api = _VetoApi([CARGO_LATE], now=8 * 60)
    svc = ModelDecisionService(api)
    rules = svc._rules.setdefault("DV01", DriverRules())
    rules.has_any_preference = True
    rules.residual_constraints.append(
        {"text": "落大雨嗰日生鲜嘅货唔好同我派", "penalty": 800, "note": ""}
    )
    action = svc.decide("DV01")
    assert action["action"] == "wait", action
    # without residuals the same setup takes the order (sanity check)
    api2 = _VetoApi([CARGO_LATE], now=8 * 60)
    svc2 = ModelDecisionService(api2)
    action2 = svc2.decide("DV02")
    assert action2["action"] == "take_order", action2


def test_idle_wait_consults_llm_for_rescue_reposition() -> None:
    """No compliant cargo anywhere -> the LLM idle rescue may reposition."""
    api = SimStubApi(
        [], now=8 * 60,
        decision_action={"action": "reposition",
                         "params": {"latitude": 23.5, "longitude": 113.5},
                         "reason": "去货源密集区"},
    )
    svc = ModelDecisionService(api)
    action = svc.decide("DZ04")
    assert api.decision_llm_calls == 1, api.decision_llm_calls
    assert action["action"] == "reposition", action


def test_idle_wait_survives_llm_outage() -> None:
    """LLM down during the idle rescue -> the deterministic 2h wait stands."""
    api = SimStubApi([], now=8 * 60, decision_fails=True)
    svc = ModelDecisionService(api)
    action = svc.decide("DZ05")
    assert action["action"] == "wait", action
    assert int(action["params"]["duration_minutes"]) == 120, action


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
