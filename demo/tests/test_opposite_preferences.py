"""Tests for "opposite preference" generalization (finals drivers whose
preferences point the other way from D001's).

Root cause they guard against: D001's "monthly >8h long-haul ≤5 orders" rule
used to be GLOBAL CONSTANTS applied to every driver (decision prompt, LLM-pick
validator, deterministic scheduler). A finals driver with the opposite
preference (long-haul focused / long-haul quota) was actively throttled →
quota-shortfall penalties stack per missing order. Likewise the schema only
had a category *blacklist* and a haul *maximum*, so the mirrored preferences
("只拉X品类" whitelist, "低于X公里不接" minimum) were unrepresentable.

Now:
- the long-haul cap exists only when parsed from THIS driver's preferences
  (LLM field ``monthly_longhaul_cap`` or explicit legacy regex mode);
- ``allowed_categories`` whitelist and ``haul_min_km`` are first-class rules
  enforced by the deterministic evaluators.

Run: ``python demo/tests/test_opposite_preferences.py`` (no pytest dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DriverRules,
    ModelDecisionService,
)


class _StubApi:
    """Offline stub: model_chat_completion returns nothing parseable, so all
    semantic confirms fall back to their fail-safe defaults."""

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


def _validate_with_net(svc, rules, plan, *, cost_time, net):  # noqa: ANN001, ANN201
    items = [{"cargo": {"cargo_id": "99", "cost_time_minutes": cost_time, "cargo_name": "",
                        "start": {"city": ""}, "end": {"city": ""}}}]
    svc._evaluate_cargo = lambda *a, **k: (float(net), False, 600, 0.0)  # type: ignore[assignment]
    return svc._validate_llm_take_order(
        cargo_id="99", items=items, rules=rules, plan=plan,
        now=400, lat=0.0, lng=0.0, day=0, hard_end=10_000,
    )


# ------------------------------------------------------ long-haul cap removal

def test_no_cap_pref_means_unlimited_longhaul() -> None:
    """A driver WITHOUT a long-haul preference must not be throttled: the 6th
    low-net >8h order of the month is still accepted (old code rejected it)."""
    svc = _svc()
    rules = DriverRules()  # no longhaul preference parsed
    plan = {"monthly_longhual": {0: 5}}
    assert _validate_with_net(svc, rules, plan, cost_time=600, net=800) is True


def test_parsed_cap_still_enforced_softly() -> None:
    """A driver WITH a parsed cap keeps the old soft-cap economics."""
    svc = _svc()
    rules = DriverRules()
    rules.longhaul_cap_orders = 5
    plan = {"monthly_longhual": {0: 5}}
    assert _validate_with_net(svc, rules, plan, cost_time=600, net=800) is False
    assert _validate_with_net(svc, rules, plan, cost_time=600, net=5000) is True


def test_merge_llm_rules_sets_longhaul_cap() -> None:
    """The new parse field monthly_longhaul_cap lands on the rules (offline
    confirm default keeps it because the text mentions 长途)."""
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(
        rules,
        {"monthly_longhaul_cap": {"max_orders": 5, "min_hours": 8}},
        ["不爱接那种一跑就是大半天的远活，每个月超过八小时的长途只能接最多5单，多一单扣一次。"],
    )
    assert rules.longhaul_cap_orders == 5, rules.longhaul_cap_orders
    assert rules.longhaul_min_minutes == 480, rules.longhaul_min_minutes


def test_offline_regex_fallback_parses_d001_longhaul_text() -> None:
    """Offline fallback must still recover D001's cap from the raw text."""
    svc = _svc()
    rules = DriverRules()
    svc._supplement_basic_rules(
        "不爱接那种一跑就是大半天、人困马乏的远活，每个月超过八小时的长途只能接最多5单，多一单扣一次。",
        rules,
    )
    assert rules.longhaul_cap_orders == 5, rules.longhaul_cap_orders
    assert rules.longhaul_min_minutes == 480, rules.longhaul_min_minutes


def test_plain_text_does_not_invent_longhaul_cap() -> None:
    """Text without a long-haul order cap must not create one."""
    svc = _svc()
    rules = DriverRules()
    svc._supplement_basic_rules("每天连续休息满8小时，空驶超过五十公里别接。", rules)
    assert rules.longhaul_cap_orders is None, rules.longhaul_cap_orders


# ------------------------------------------------- category whitelist (只拉X)

def _cargo(name: str, *, haul_lat: float = 0.3) -> dict:
    return {
        "cargo_id": "c1", "cargo_name": name, "price": 9000.0,
        "cost_time_minutes": 200,
        "start": {"city": "A", "lat": 23.0, "lng": 113.0},
        "end": {"city": "B", "lat": 23.0 + haul_lat, "lng": 113.0},
    }


def test_allowed_categories_whitelist_rejects_other_cargo() -> None:
    svc = _svc()
    rules = DriverRules()
    rules.allowed_categories = {"水果"}
    ev_other = svc._evaluate_cargo(_cargo("钢材"), {}, rules, set(), 600, 100_000, 23.0, 113.0)
    ev_match = svc._evaluate_cargo(_cargo("水果"), {}, rules, set(), 600, 100_000, 23.0, 113.0)
    assert ev_other is None, ev_other
    assert ev_match is not None
    # known penalty → violation becomes a soft cost instead of a hard reject
    rules.rule_penalties["allowed_categories"] = 500.0
    ev_soft = svc._evaluate_cargo(_cargo("钢材"), {}, rules, set(), 600, 100_000, 23.0, 113.0)
    assert ev_soft is not None and ev_soft[0] < ev_match[0]


def test_merge_llm_rules_sets_allowed_categories() -> None:
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"allowed_categories": ["水果"]}, ["我只拉水果，别的货一概不接"])
    assert rules.allowed_categories == {"水果"}, rules.allowed_categories


def test_allowed_categories_not_invented_without_whitelist_marker() -> None:
    """Offline default must drop a hallucinated whitelist when the text has no
    only/whitelist wording (inventing one throttles everything)."""
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"allowed_categories": ["水果"]}, ["水果搬运要小心轻放"])
    assert rules.allowed_categories == set(), rules.allowed_categories


# ------------------------------------------------------- haul_min_km (反向上限)

def test_haul_min_km_rejects_short_hauls() -> None:
    svc = _svc()
    rules = DriverRules()
    rules.haul_min_km = 100.0
    short = svc._evaluate_cargo(_cargo("货", haul_lat=0.3), {}, rules, set(), 600, 100_000, 23.0, 113.0)
    long_ = svc._evaluate_cargo(_cargo("货", haul_lat=1.5), {}, rules, set(), 600, 100_000, 23.0, 113.0)
    assert short is None, short  # ~33km < 100km
    assert long_ is not None  # ~167km >= 100km


def test_merge_llm_rules_sets_haul_min_km() -> None:
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {"haul_min_km": 100}, ["运输距离低于一百公里的短活不接"])
    assert rules.haul_min_km == 100.0, rules.haul_min_km


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
