"""Regression tests for monthly long-haul cap enforcement.

Root cause they guard against: the agent tracked the monthly count of >8h
("long-haul") orders in ``plan["monthly_longhual"]``, and both the LLM-pick
validator and the deterministic scheduler reject further >8h orders once the
count reaches 5. BUT that counter was only ever incremented inside
``update_decision_result`` -- a hook the evaluation harness never calls (it only
calls ``decide()``). So the counter stayed 0 forever, the cap never fired, the
driver took unlimited long-haul orders and ate the monthly penalty every month.

The fix rebuilds the accepted-order accumulators inside ``decide()`` from the
authoritative ``query_decision_history`` records (cargo cost-time is recovered
from a metadata cache populated on every cargo scan). These tests verify the
reconstruction is correct and that the reconstructed count makes the cap fire.

Run: ``python demo/tests/test_longhaul_cap_history.py`` (no pytest dependency).
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
    def __init__(self, records=None, cargo_items=None):  # noqa: ANN001
        self._records = records or []
        self._cargo_items = cargo_items or []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": self._cargo_items}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": list(self._records)}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {}


def _accepted_longhaul_record(cargo_id: str, cost_minutes: int = 600) -> dict:
    """A take_order record accepted on day 0 (March) with the given haul time."""
    return {
        # action_start = end - step_elapsed + query_scan = 0 -> day 0 (March)
        "step_elapsed_minutes": 100,
        "query_scan_cost_minutes": 0,
        "action": {"action": "take_order", "params": {"cargo_id": cargo_id}},
        "result": {
            "accepted": True,
            "cargo_id": cargo_id,
            "simulation_progress_minutes": 100,
            "pickup_deadhead_km": 12.0,
        },
    }


def test_history_reconstructs_longhaul_count() -> None:
    """5 accepted >8h orders in March must rebuild monthly_longhual == {0: 5}."""
    records = [_accepted_longhaul_record(str(i)) for i in range(1, 6)]
    svc = ModelDecisionService(_StubApi(records=records))
    # Pretend these cargo were seen during scans (cost_time > 480 = long-haul).
    for i in range(1, 6):
        svc._cargo_meta[str(i)] = {"cost_time_minutes": 600, "cargo_name": "", "start_city": "", "end_city": ""}
    plan: dict = {}
    svc._sync_monthly_counts_from_history("D001", plan)
    assert plan["monthly_longhual"] == {0: 5}, plan["monthly_longhual"]
    assert plan["orders_today"].get(0) == 5, plan["orders_today"]
    assert plan["total_deadhead_km"] == 60.0, plan["total_deadhead_km"]


def test_short_haul_orders_are_not_counted_as_longhaul() -> None:
    """Orders with cost_time <= 480 must not inflate the long-haul count."""
    records = [_accepted_longhaul_record(str(i)) for i in range(1, 4)]
    svc = ModelDecisionService(_StubApi(records=records))
    for i in range(1, 4):
        svc._cargo_meta[str(i)] = {"cost_time_minutes": 400, "cargo_name": "", "start_city": "", "end_city": ""}
    plan: dict = {}
    svc._sync_monthly_counts_from_history("D001", plan)
    assert plan["monthly_longhual"] == {}, plan["monthly_longhual"]
    assert plan["orders_today"].get(0) == 3, plan["orders_today"]


def test_query_cargo_wrapper_caches_metadata() -> None:
    """_query_cargo must remember cost_time/name for every scanned cargo."""
    items = [{"cargo": {"cargo_id": "7", "cost_time_minutes": 540, "cargo_name": "水果",
                         "start": {"city": "A"}, "end": {"city": "B"}}}]
    svc = ModelDecisionService(_StubApi(cargo_items=items))
    svc._query_cargo(driver_id="D001", latitude=0.0, longitude=0.0, k=30)
    assert svc._cargo_meta["7"]["cost_time_minutes"] == 540, svc._cargo_meta
    assert svc._cargo_meta["7"]["cargo_name"] == "水果", svc._cargo_meta


def _validate_with_net(svc, plan, *, cost_time, net, cap=5):  # noqa: ANN001, ANN201
    """Drive _validate_llm_take_order with a controlled marginal net via a stub
    _evaluate_cargo, so we test only the soft long-haul gate (not geo feasibility).

    The long-haul cap is preference-driven now (no global hard-coded rule), so
    the test driver's parsed rules carry the cap explicitly (cap=None simulates
    a driver who never asked for one)."""
    rules = DriverRules()
    rules.longhaul_max_orders = cap
    items = [{"cargo": {"cargo_id": "99", "cost_time_minutes": cost_time, "cargo_name": "",
                        "start": {"city": ""}, "end": {"city": ""}}}]
    svc._evaluate_cargo = lambda *a, **k: (float(net), False, 600, 0.0)  # type: ignore[assignment]
    return svc._validate_llm_take_order(
        cargo_id="99", items=items, rules=rules, plan=plan,
        now=400, lat=0.0, lng=0.0, day=0, hard_end=10_000,
    )


def test_soft_cap_rejects_unprofitable_over_cap_longhaul() -> None:
    """Over the 5-order cap, a >8h order whose net <= penalty (1000) is rejected."""
    records = [_accepted_longhaul_record(str(i)) for i in range(1, 6)]
    svc = ModelDecisionService(_StubApi(records=records))
    for i in range(1, 6):
        svc._cargo_meta[str(i)] = {"cost_time_minutes": 600, "cargo_name": "", "start_city": "", "end_city": ""}
    plan: dict = {}
    svc._sync_monthly_counts_from_history("D001", plan)
    assert plan["monthly_longhual"].get(0) == 5
    assert _validate_with_net(svc, plan, cost_time=600, net=800) is False


def test_soft_cap_accepts_profitable_over_cap_longhaul() -> None:
    """Over the cap, a >8h order whose net > penalty is still accepted (net max)."""
    records = [_accepted_longhaul_record(str(i)) for i in range(1, 6)]
    svc = ModelDecisionService(_StubApi(records=records))
    for i in range(1, 6):
        svc._cargo_meta[str(i)] = {"cost_time_minutes": 600, "cargo_name": "", "start_city": "", "end_city": ""}
    plan: dict = {}
    svc._sync_monthly_counts_from_history("D001", plan)
    assert _validate_with_net(svc, plan, cost_time=600, net=5000) is True


def test_under_cap_longhaul_is_accepted_regardless_of_net() -> None:
    """Below the cap, a low-net >8h order is not penalised and is accepted."""
    svc = ModelDecisionService(_StubApi(records=[]))
    plan = {"monthly_longhual": {0: 4}}
    assert _validate_with_net(svc, plan, cost_time=600, net=800) is True


def test_driver_without_longhaul_preference_has_no_cap() -> None:
    """A driver whose preferences never mention a long-haul cap must not be
    capped by any hard-coded default (the rule is preference-driven)."""
    svc = ModelDecisionService(_StubApi(records=[]))
    plan = {"monthly_longhual": {0: 20}}
    assert _validate_with_net(svc, plan, cost_time=600, net=800, cap=None) is True


def test_parsed_longhaul_cap_is_merged_from_llm_output() -> None:
    """The compile step must turn monthly_longhaul_cap into DriverRules fields."""
    svc = ModelDecisionService(_StubApi(records=[]))
    rules = DriverRules()
    svc._merge_llm_rules(
        rules,
        {"monthly_longhaul_cap": {"max_orders": 5, "min_hours": 8}},
        ["不爱接那种一跑就是大半天的远活，每个月超过八小时的长途只能接最多5单，多一单扣一次。"],
    )
    assert rules.longhaul_max_orders == 5, rules.longhaul_max_orders
    assert rules.longhaul_threshold_minutes == 480, rules.longhaul_threshold_minutes


def test_empty_history_leaves_plan_untouched() -> None:
    """No records (e.g. session history not configured) must be a safe no-op."""
    svc = ModelDecisionService(_StubApi(records=[]))
    plan = {"monthly_longhual": {0: 3}}
    svc._sync_monthly_counts_from_history("D001", plan)
    assert plan["monthly_longhual"] == {0: 3}, plan["monthly_longhual"]


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
