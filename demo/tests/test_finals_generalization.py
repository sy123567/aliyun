"""Tests for the finals-generalization hardening round (gross margin + penalty).

Covers the changes aimed at the two unknown finals drivers (one Guangdong, one
Yangtze-delta) whose preference texts may be phrased in dialect:

1. dialect-robust offline night fail-safe (Cantonese/Wu phrasings, Chinese-
   numeral clock times such as 晚上九点 / 廿三点);
2. penalty_cap plumbed through preference records, the parse/audit payloads
   and the decision prompt (capped vs uncapped economics);
3. semantic region membership fallback for region names outside the static
   table (e.g. 苏南) — both allowed_regions and forbidden_regions — with a
   per-step budget;
4. exact-name category quota progress counting when the parsed target exists
   verbatim in the dataset (the scorer credits quota orders by exact
   cargo_name), with semantic fallback for dialect/noisy targets (生果);
5. daily-directive hallucination gate: a directive window that does not
   reshape any static window must survive a semantic confirm before being
   enforced (fail-safe: kept on model failure);
6. past-midnight order finish deadline for drivers without any rest
   constraint, clamped to the simulation horizon;
7. wide-scan (k=600) throttling while parked near a recent fruitless wide
   scan during urgent category hunts.

Run: ``python demo/tests/test_finals_generalization.py`` (no pytest dependency).
"""

from __future__ import annotations

import json
import sys
import traceback
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


def _resp(obj) -> dict:
    return {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}


class ScriptedApi:
    """SimulationApiPort stub that scripts the LLM roles by system prompt."""

    def __init__(self, extract_responses=None, audit_response=None, confirm_holds=True,
                 fail_all=False, yes_no=None, directive_response=None, decision_response=None):
        self.extract_payloads: list[dict] = []
        self.audit_payloads: list[dict] = []
        self.directive_payloads: list[dict] = []
        self.confirm_payloads: list[dict] = []
        self.yes_no_questions: list[str] = []
        self.decision_prompts: list[str] = []
        self._extract_responses = list(extract_responses or [])
        self._audit_response = audit_response
        self._confirm_holds = confirm_holds
        self._fail_all = fail_all
        self._yes_no = yes_no  # callable(question) -> bool
        self._directive_response = directive_response
        self._decision_response = decision_response

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": 0}

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
        if "每日合规计划助手" in system:
            self.directive_payloads.append(json.loads(user))
            return _resp(self._directive_response or {})
        if "是否确实包含某条约束" in system:
            self.confirm_payloads.append(json.loads(user))
            return _resp({"holds": self._confirm_holds})
        if "语义判定助手" in system:
            self.yes_no_questions.append(user)
            ans = self._yes_no(user) if self._yes_no else True
            return _resp({"answer": ans})
        if "智能货运调度决策AI" in system:
            self.decision_prompts.append(user)
            return _resp(self._decision_response or {"action": "wait", "params": {"duration_minutes": 30}})
        return _resp({})


def _status(prefs):
    return {"preferences": prefs, "current_lat": 23.13, "current_lng": 113.26}


# ----------------------------------------------- 1. dialect night fail-safe

# Cantonese-flavoured night rest with Chinese-numeral clock times: no Arabic
# digits at all, none of the Mandarin keyword phrasings of the D001 fixture.
PREF_NIGHT_CANTONESE = "晚上九点之后就收车返屋企瞓觉，瞓到朝早六点先至开工。"
PREF_NIGHT_WU = "夜里向廿三点以后勿开车，困觉困到清晨五点。"


def test_dialect_night_failsafe_cantonese() -> None:
    """Offline (model down): the night fail-safe must still derive the
    overnight window from a Cantonese phrasing with CN-numeral clock times."""
    api = ScriptedApi(fail_all=True)
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG01", _status([
        {"content": PREF_NIGHT_CANTONESE, "penalty_amount": 2700, "penalty_cap": None},
    ]))
    assert (21 * 60, 30 * 60) in rules.no_drive_windows, rules.no_drive_windows
    assert rules.rest_window == (0, 6 * 60), rules.rest_window


def test_dialect_night_failsafe_wu_with_cn_numerals() -> None:
    api = ScriptedApi(fail_all=True)
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG02", _status([
        {"content": PREF_NIGHT_WU, "penalty_amount": 500, "penalty_cap": 5000},
    ]))
    assert (23 * 60, 29 * 60) in rules.no_drive_windows, rules.no_drive_windows


def test_extract_clock_times_pm_context() -> None:
    times = ModelDecisionService._extract_clock_times("晚上九点到朝早六点不动车")
    assert 21 * 60 in times and 6 * 60 in times, times
    times2 = ModelDecisionService._extract_clock_times("廿三点收车")
    assert 23 * 60 in times2, times2


# ----------------------------------------------- 2. penalty_cap plumbing

def test_penalty_cap_reaches_parse_audit_and_prompt() -> None:
    api = ScriptedApi(
        extract_responses=[{"no_drive_windows": [{"start_hour": 22, "end_hour": 6}]}],
        audit_response={"audits": [{
            "index": 0, "covered": False, "representable": False,
            "missing": "雨天生鲜不接无法结构化",
        }]},
    )
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DG03", _status([
        {"content": "落雨天生鲜唔好派俾我，夜晚十点后唔开车。",
         "penalty_amount": 800, "penalty_cap": 4000},
    ]))
    # parse + audit payloads carry the caps
    assert api.extract_payloads[0].get("penalty_caps") == [4000], api.extract_payloads[0]
    assert api.audit_payloads[0].get("penalty_caps") == [4000], api.audit_payloads[0]
    # residual constraint remembers the cap
    assert rules.residual_constraints and rules.residual_constraints[0]["cap"] == 4000.0
    # the decision prompt mentions the cap economics
    prompt = svc._format_rules_for_llm("DG03", rules, {}, day=0)
    assert "累计上限4000" in prompt, prompt


def test_uncapped_pref_is_flagged_in_prompt() -> None:
    api = ScriptedApi(extract_responses=[{}], audit_response={"audits": [{"index": 0, "covered": True}]})
    svc = ModelDecisionService(api)
    svc._ensure_rules("DG04", _status([
        {"content": "夜晚十点后唔开车。", "penalty_amount": 2700, "penalty_cap": None},
    ]))
    prompt = svc._format_rules_for_llm("DG04", svc._rules["DG04"], {}, day=0)
    assert "不封顶" in prompt, prompt


# ----------------------------------------------- 3. semantic region fallback

def test_allowed_region_semantic_membership() -> None:
    """苏南 is not in the static region table; without the semantic fallback
    every order would be rejected and the driver idles all month."""
    api = ScriptedApi(yes_no=lambda q: "『苏州』" in q)
    svc = ModelDecisionService(api)
    svc._sem_region_budget = 6
    rules = DriverRules()
    rules.allowed_regions = {"苏南"}
    assert svc._allowed_region_ok(rules, 31.30, 120.58, "苏州市") is True
    # cached: a second call must not consume budget nor re-ask
    n_questions = len(api.yes_no_questions)
    assert svc._allowed_region_ok(rules, 31.30, 120.58, "苏州市") is True
    assert len(api.yes_no_questions) == n_questions
    # a clearly-outside city is rejected
    assert svc._allowed_region_ok(rules, 23.13, 113.26, "广州市") is False


def test_forbidden_region_semantic_membership() -> None:
    api = ScriptedApi(yes_no=lambda q: "徐州" in q)
    svc = ModelDecisionService(api)
    svc._sem_region_budget = 6
    assert svc._forbidden_region_hit("苏北", "徐州市") is True
    assert svc._forbidden_region_hit("苏北", "苏州市") is False
    # plain substring still wins without any model call
    n = len(api.yes_no_questions)
    assert svc._forbidden_region_hit("惠州", "惠州市") is True
    assert len(api.yes_no_questions) == n


def test_region_budget_exhaustion_is_conservative_and_retryable() -> None:
    api = ScriptedApi(yes_no=lambda q: True)
    svc = ModelDecisionService(api)
    svc._sem_region_budget = 0
    rules = DriverRules()
    rules.allowed_regions = {"苏南"}
    # budget exhausted -> conservative False, NOT cached
    assert svc._allowed_region_ok(rules, 31.30, 120.58, "苏州市") is False
    assert not api.yes_no_questions
    # next step refills the budget -> the lookup happens and is cached
    svc._sem_region_budget = 6
    assert svc._allowed_region_ok(rules, 31.30, 120.58, "苏州市") is True


# ----------------------------------------------- 4. category quota counting

def test_category_progress_exact_when_target_verbatim_in_dataset() -> None:
    api = ScriptedApi(yes_no=lambda q: True)  # sem matcher would say yes to anything
    svc = ModelDecisionService(api)
    svc._cargo_names.update({"水果", "建材", "鲜果"})
    # target exists verbatim -> only exact names count (scorer semantics)
    assert svc._category_progress_match("水果", "水果") is True
    assert svc._category_progress_match("水果", "鲜果") is False
    assert svc._category_progress_match("水果", "水泥") is False


def test_category_progress_semantic_for_dialect_target() -> None:
    # 生果 (Cantonese for fruit) does not exist as a dataset cargo_name ->
    # semantic fallback decides
    api = ScriptedApi(yes_no=lambda q: "生果" in q and "水果" in q)
    svc = ModelDecisionService(api)
    svc._cargo_names.update({"水果", "建材"})
    assert svc._category_progress_match("生果", "水果") is True


# ----------------------------------------------- 5. directive hallucination gate

def _directive_svc(confirm_holds: bool, directive: dict) -> tuple[ModelDecisionService, DriverRules, dict]:
    api = ScriptedApi(confirm_holds=confirm_holds, directive_response=directive)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    svc._rules["DG10"] = rules
    svc._pref_records["DG10"] = [
        {"content": "夜晚十点后唔开车。", "penalty_amount": 2700, "penalty_cap": None}
    ]
    plan: dict = {}
    return svc, rules, plan


def test_directive_window_intersecting_static_kept_without_confirm() -> None:
    directive = {"no_drive_today": [{"start_hour": 23, "end_hour": 6}],
                 "replaces_default": False, "today_plan": "", "category_focus": None}
    svc, rules, plan = _directive_svc(confirm_holds=False, directive=directive)
    rules.no_drive_windows.append((22 * 60, 30 * 60))
    svc._ensure_daily_directive("DG10", rules, plan, day=3)
    assert (23 * 60, 30 * 60) in (rules.daily_directives.get(3) or {}).get("windows", []), rules.daily_directives
    # no confirm call was needed for an intersecting window
    assert not svc._api.confirm_payloads


def test_directive_hallucinated_window_dropped_when_confirm_rejects() -> None:
    directive = {"no_drive_today": [{"start_hour": 12, "end_hour": 14}],
                 "replaces_default": False, "today_plan": "", "category_focus": None}
    svc, rules, plan = _directive_svc(confirm_holds=False, directive=directive)
    rules.no_drive_windows.append((22 * 60, 30 * 60))
    svc._ensure_daily_directive("DG10", rules, plan, day=3)
    assert (rules.daily_directives.get(3) or {}).get("windows") == [], rules.daily_directives
    assert svc._api.confirm_payloads  # gate actually consulted the model


def test_directive_new_window_kept_when_confirm_accepts() -> None:
    directive = {"no_drive_today": [{"start_hour": 12, "end_hour": 14}],
                 "replaces_default": False, "today_plan": "", "category_focus": None}
    svc, rules, plan = _directive_svc(confirm_holds=True, directive=directive)
    svc._ensure_daily_directive("DG10", rules, plan, day=3)
    assert (12 * 60, 14 * 60) in (rules.daily_directives.get(3) or {}).get("windows", [])


# ----------------------------------------------- 6. past-midnight deadline

def test_rest_free_driver_can_finish_past_midnight() -> None:
    svc = ModelDecisionService(ScriptedApi())
    rules = DriverRules()  # no constraints at all
    plan = {"off_days": set()}
    day = 10
    now = day * DAY_MINUTES + 600
    deadline = svc._order_finish_deadline(rules, plan, now, day)
    assert deadline == (day + 2) * DAY_MINUTES, deadline


def test_flexible_rest_extension_preserved() -> None:
    svc = ModelDecisionService(ScriptedApi())
    rules = DriverRules()
    rules.daily_rest_minutes = 8 * 60
    plan = {"off_days": set()}
    day = 10
    now = day * DAY_MINUTES + 600
    deadline = svc._order_finish_deadline(rules, plan, now, day)
    assert deadline == (day + 1) * DAY_MINUTES + (DAY_MINUTES - 8 * 60), deadline


def test_night_window_driver_keeps_same_day_deadline() -> None:
    svc = ModelDecisionService(ScriptedApi())
    rules = DriverRules()
    rules.no_drive_windows.append((21 * 60, 30 * 60))
    plan = {"off_days": set()}
    day = 10
    now = day * DAY_MINUTES + 600
    deadline = svc._order_finish_deadline(rules, plan, now, day)
    # cut at window start minus buffer, never past midnight
    assert deadline <= day * DAY_MINUTES + 21 * 60, deadline


def test_deadline_clamped_to_simulation_horizon() -> None:
    svc = ModelDecisionService(ScriptedApi())
    rules = DriverRules()
    plan = {"off_days": set()}
    day = MONTH_DAYS - 1
    now = day * DAY_MINUTES + 600
    deadline = svc._order_finish_deadline(rules, plan, now, day)
    assert deadline <= MONTH_DAYS * DAY_MINUTES, deadline


# ----------------------------------------------- 7. wide-scan throttling

class ScanApi(ScriptedApi):
    def __init__(self, now: int):
        super().__init__()
        self._now = now
        self.scan_ks: list[int] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._now}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.scan_ks.append(int(kwargs.get("k", 0)))
        return {"items": []}


def _hunt_setup(now: int) -> tuple[ModelDecisionService, DriverRules, dict, ScanApi]:
    api = ScanApi(now)
    svc = ModelDecisionService(api)
    rules = DriverRules()
    rules.monthly_category_targets = {0: {"水果": 12}}
    plan = {
        "off_days": set(), "orders_today": {}, "first_order_taken": set(),
        "monthly_longhual": {}, "monthly_category_orders": {},
        "monthly_deadhead_km": {}, "zeng_order_days": set(),
        "failed_cargo_ids": set(), "failed_cargo_reasons": {},
        "total_deadhead_km": 0.0, "must_visit_days": {},
    }
    return svc, rules, plan, api


def test_urgent_hunt_uses_wide_scan_then_throttles() -> None:
    day = 25  # late in month 0 -> urgent (remaining days <= needed + 10)
    now = day * DAY_MINUTES + 600
    svc, rules, plan, api = _hunt_setup(now)
    svc._pick_order("DG20", {}, rules, plan, now, 23.1, 113.2, day, (day + 1) * DAY_MINUTES)
    assert api.scan_ks and api.scan_ks[0] == 600, api.scan_ks
    assert plan.get("_wide_scan") is not None
    # second step, same spot, minutes later: the wide scan is throttled
    api.scan_ks.clear()
    svc._pick_order("DG20", {}, rules, plan, now + 70, 23.1, 113.2, day, (day + 1) * DAY_MINUTES)
    assert api.scan_ks and max(api.scan_ks) <= 200, api.scan_ks
    # after relocating far away, the wide scan is allowed again
    api.scan_ks.clear()
    svc._pick_order("DG20", {}, rules, plan, now + 140, 26.0, 115.0, day, (day + 1) * DAY_MINUTES)
    assert 600 in api.scan_ks, api.scan_ks


class OneCargoApi(ScriptedApi):
    def __init__(self, now: int, cargo: dict):
        super().__init__()
        self._now = now
        self._cargo = cargo

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._now}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": [{"cargo": self._cargo, "distance_km": 1.0}]}


def test_monthly_deadhead_cap_driver_can_still_pick_orders() -> None:
    """Regression: _pick_order referenced month_deadhead that was only defined
    in _llm_decide_with_history — any driver with a monthly deadhead cap hit a
    NameError on every deterministic pick (losing the entire run)."""
    day = 5
    now = day * DAY_MINUTES + 480
    cargo = {
        "cargo_id": "C-1", "cargo_name": "建材", "price": 800.0,
        "cost_time_minutes": 120,
        "start": {"lat": 23.11, "lng": 113.21, "city": "广州市"},
        "end": {"lat": 23.40, "lng": 113.50, "city": "广州市"},
    }
    svc = ModelDecisionService(OneCargoApi(now, cargo))
    rules = DriverRules()
    rules.monthly_deadhead_max_km = 500.0
    plan = {
        "off_days": set(), "orders_today": {}, "first_order_taken": set(),
        "monthly_longhual": {}, "monthly_category_orders": {},
        "monthly_deadhead_km": {}, "zeng_order_days": set(),
        "failed_cargo_ids": set(), "failed_cargo_reasons": {},
        "total_deadhead_km": 0.0, "must_visit_days": {},
    }
    action = svc._pick_order("DG30", {}, rules, plan, now, 23.10, 113.20, day, (day + 1) * DAY_MINUTES)
    assert action is not None and action["action"] == "take_order", action


# ----------------------------------------------- 8. value layer (LLM-on-value)

def _fresh_plan() -> dict:
    return {
        "off_days": set(), "orders_today": {}, "first_order_taken": set(),
        "monthly_longhual": {}, "monthly_category_orders": {},
        "monthly_deadhead_km": {}, "zeng_order_days": set(),
        "failed_cargo_ids": set(), "failed_cargo_reasons": {},
        "total_deadhead_km": 0.0, "must_visit_days": {}, "rest_done": set(),
    }


def test_liquidity_stats_aggregation_and_recency() -> None:
    svc = ModelDecisionService(ScriptedApi())
    now = 10 * DAY_MINUTES
    svc._cargo_meta = {
        "a1": {"start_city": "苏州市", "net_per_h": 90.0, "seen_at": now - 100},
        "a2": {"start_city": "苏州市", "net_per_h": 110.0, "seen_at": now - 200},
        "a3": {"start_city": "苏州市", "net_per_h": 100.0, "seen_at": now - 300},
        "b1": {"start_city": "杭州市", "net_per_h": 200.0, "seen_at": now - 100},
        "c1": {"start_city": "南京市", "net_per_h": 500.0, "seen_at": now - 10 * DAY_MINUTES},  # stale
    }
    svc._city_centroid = {
        "苏州市": [31.3 * 3, 120.6 * 3, 3],
        "杭州市": [30.25, 120.16, 1],
        "南京市": [32.06, 118.80, 1],
    }
    rows = svc._cargo_liquidity_stats(now)
    cities = [r["city"] for r in rows]
    assert "南京市" not in cities, rows  # aged out
    # 苏州: 3 × avg100 = 300 beats 杭州: 1 × 200
    assert cities[0] == "苏州市", rows
    assert abs(rows[0]["net_per_h"] - 100.0) < 1e-6 and rows[0]["n"] == 3, rows
    assert abs(rows[0]["lat"] - 31.3) < 1e-6, rows


class ValueApi(ScriptedApi):
    """Decision-layer stub: real clock + one strong scannable cargo."""

    def __init__(self, now: int, items: list[dict], decision_response: dict):
        super().__init__(decision_response=decision_response)
        self._now = now
        self._items = items
        self.scan_ks: list[int] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._now,
                "current_lat": 23.10, "current_lng": 113.20, "preferences": []}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.scan_ks.append(int(kwargs.get("k", 0)))
        return {"items": self._items}


def _strong_item(cargo_id: str = "C-9") -> dict:
    return {"cargo": {
        "cargo_id": cargo_id, "cargo_name": "建材", "price": 900.0,
        "cost_time_minutes": 120,
        "start": {"lat": 23.11, "lng": 113.21, "city": "广州市"},
        "end": {"lat": 23.40, "lng": 113.50, "city": "佛山市"},
    }, "distance_km": 1.5}


def test_llm_long_wait_overridden_by_strong_candidate() -> None:
    """A >=60min strategic wait while a compliant high-net order is on the
    table gets converted into taking that order (documented failure mode of
    LLM-in-the-loop runs: idling away strong orders)."""
    day, tod = 5, 600
    now = day * DAY_MINUTES + tod
    api = ValueApi(now, [_strong_item()], {"action": "wait", "params": {"duration_minutes": 240}})
    svc = ModelDecisionService(api)
    svc._sim_now["DG40"] = now
    rules = DriverRules()
    from agent.model_decision_service import DecisionHistory
    action = svc._llm_decide_with_history(
        "DG40", {}, rules, _fresh_plan(), DecisionHistory(), now, 23.10, 113.20, day, tod
    )
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "C-9", action


def test_llm_short_wait_not_overridden() -> None:
    day, tod = 5, 600
    now = day * DAY_MINUTES + tod
    api = ValueApi(now, [_strong_item()], {"action": "wait", "params": {"duration_minutes": 30}})
    svc = ModelDecisionService(api)
    svc._sim_now["DG41"] = now
    rules = DriverRules()
    from agent.model_decision_service import DecisionHistory
    action = svc._llm_decide_with_history(
        "DG41", {}, rules, _fresh_plan(), DecisionHistory(), now, 23.10, 113.20, day, tod
    )
    assert action is not None and action["action"] == "wait", action


def test_llm_codecides_every_step() -> None:
    """The decision LLM is consulted on every step now (was every 3rd)."""
    day, tod = 5, 600
    now = day * DAY_MINUTES + tod
    api = ValueApi(now, [], {"action": "wait", "params": {"duration_minutes": 45}})
    svc = ModelDecisionService(api)
    a1 = svc.decide("DG42")  # step 1 — old gating (step % 3 == 0) would skip the LLM
    assert len(api.decision_prompts) == 1, api.decision_prompts
    assert a1["action"] == "wait", a1
    svc.decide("DG42")  # step 2 — still consulted
    assert len(api.decision_prompts) == 2


def test_decision_prompt_contains_market_table() -> None:
    day, tod = 5, 600
    now = day * DAY_MINUTES + tod
    api = ValueApi(now, [_strong_item()], {"action": "wait", "params": {"duration_minutes": 30}})
    svc = ModelDecisionService(api)
    svc._sim_now["DG43"] = now
    # pre-seed the scan cache so the liquidity table has content
    svc._cargo_meta = {"x": {"start_city": "佛山市", "net_per_h": 120.0, "seen_at": now - 50}}
    svc._city_centroid = {"佛山市": [23.02, 113.12, 1]}
    rules = DriverRules()
    from agent.model_decision_service import DecisionHistory
    svc._llm_decide_with_history(
        "DG43", {}, rules, _fresh_plan(), DecisionHistory(), now, 23.10, 113.20, day, tod
    )
    prompt = api.decision_prompts[-1]
    assert "活跃市场" in prompt and "佛山市" in prompt, prompt
    # the candidate going to 佛山市 is annotated with destination liquidity
    assert "to_liq" in prompt, prompt


def test_thinking_disabled_when_wall_budget_projected_exceeded() -> None:
    """Thinking mode must hand the driver back to fast mode when the run is
    projected past the per-driver wall budget (finals 4h cap is a hard kill)."""
    import agent.model_decision_service as mds
    import time as _time

    day, tod = 46, 600  # mid-season -> sim_frac ~0.5
    now = day * DAY_MINUTES + tod
    api = ValueApi(now, [], {"action": "wait", "params": {"duration_minutes": 30}})
    svc = ModelDecisionService(api)
    svc._sim_now["DG50"] = now
    # pretend the run started long ago: way over the pro-rated budget
    svc._wall_start["DG50"] = _time.time() - mds._THINKING_WALL_BUDGET_SECONDS
    old = mds._DECISION_THINKING
    mds._DECISION_THINKING = True
    try:
        from agent.model_decision_service import DecisionHistory
        svc._llm_decide_with_history(
            "DG50", {}, DriverRules(), _fresh_plan(), DecisionHistory(), now, 23.10, 113.20, day, tod
        )
        assert "DG50" in svc._thinking_off
        # within budget -> thinking stays on for another driver
        svc._sim_now["DG51"] = now
        svc._wall_start["DG51"] = _time.time() - 10
        svc._llm_decide_with_history(
            "DG51", {}, DriverRules(), _fresh_plan(), DecisionHistory(), now, 23.10, 113.20, day, tod
        )
        assert "DG51" not in svc._thinking_off
    finally:
        mds._DECISION_THINKING = old


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except Exception:
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
        else:
            print(f"PASS {t.__name__}")
            passed += 1
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
