"""Tests for the gross-income / value-side optimisations.

Business goal (leaderboard diagnosis): the submission had the LOWEST gross
income of the top teams despite the lowest penalty — it was too conservative
and left big orders on the table. These changes attack that *without* touching
the deterministic compliance guards:

1. ``_amortized_rate`` — rank candidates by net amortised over (occupied + a
   fixed per-order overhead), so a high-absolute-net long haul is no longer
   buried behind a marginally-faster small order (pure net-per-minute ranking
   would pick the small one).
2. ``_pick_order`` therefore prefers the bigger-value order when two are
   feasible.
3. The decision-LLM candidate rows expose absolute ``net`` (not just
   ``net_per_h``), are ordered big-value-first, and the system prompt tells the
   model to weigh absolute net + chain value — spending the (largely unused)
   token budget on better value choices.

These never accept an unprofitable order (the ``net>0`` feasibility filter is
unchanged) and never relax a compliance gate.

Run: ``python demo/tests/test_value_optimization.py`` (no pytest dep).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

import agent.model_decision_service as mds  # noqa: E402
from agent.model_decision_service import (  # noqa: E402
    DecisionHistory,
    DriverRules,
    ModelDecisionService,
    _ORDER_TIME_OVERHEAD_MIN,
)

LAT, LNG = 22.92, 113.18  # driver sits on the pickup so pickup_km ~ 0


def _cargo(cid: str, *, price: float, cost_time: int):
    """Zero-distance cargo (start == end == driver) so net ~= price."""
    cargo = {
        "cargo_id": cid,
        "cargo_name": "普货",
        "start": {"lat": LAT, "lng": LNG, "city": "广州"},
        "end": {"lat": LAT, "lng": LNG, "city": "广州"},
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "load_time": None,
    }
    return {"cargo": cargo, "distance_km": 0.0}


class _Stub:
    def __init__(self, items, progress, on_chat=None):
        self._items = items
        self._progress = progress
        self._on_chat = on_chat
        self.payloads: list[dict] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {
            "simulation_progress_minutes": self._progress,
            "current_lat": LAT,
            "current_lng": LNG,
        }

    def query_cargo(self, driver_id, latitude, longitude, k):  # noqa: ANN001, ANN201
        return {"items": self._items, "k": k}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        self.payloads.append(payload)
        if self._on_chat is not None:
            return self._on_chat(payload)
        return {"choices": [{"message": {"content": "{}"}}]}


def _plan() -> dict:
    return {
        "rest_done": set(), "zeng_order_days": set(), "dated_single_done": set(),
        "dated_route_done": set(), "strand_repo": set(), "orders_today": {},
        "total_deadhead_km": 0.0, "monthly_deadhead_km": {}, "must_visit_days": {},
        "first_order_taken": set(), "home_done": set(), "monthly_longhual": {},
        "monthly_category_orders": {}, "failed_cargo_ids": set(),
        "failed_cargo_reasons": {}, "off_days": set(), "_consecutive_waits": 0,
        "strand_count": {},
    }


# ------------------------------------------------------- _amortized_rate

def test_amortized_rate_flips_pure_per_minute_ranking() -> None:
    """A 10h/¥5000 haul has a LOWER raw net-per-minute than a 1h/¥600 order,
    yet a HIGHER overhead-amortised score — that flip is the whole point."""
    small_net, small_min = 600.0, 60
    big_net, big_min = 5000.0, 600
    # raw per-minute: small wins
    assert (small_net / small_min) > (big_net / big_min)
    # amortised: big wins
    assert ModelDecisionService._amortized_rate(big_net, big_min) > (
        ModelDecisionService._amortized_rate(small_net, small_min)
    )


def test_amortized_rate_uses_overhead_constant() -> None:
    net, occ = 1000.0, 100
    expected = net / (occ + _ORDER_TIME_OVERHEAD_MIN) * 60.0
    assert abs(ModelDecisionService._amortized_rate(net, occ) - expected) < 1e-6


# ------------------------------------------------------------ _pick_order

def test_pick_order_prefers_big_absolute_net_over_fast_small() -> None:
    small = _cargo("SMALL", price=600.0, cost_time=60)
    big = _cargo("BIG", price=5000.0, cost_time=600)
    svc = ModelDecisionService(_Stub([small, big], progress=480))  # 08:00, day 0
    status = {"simulation_progress_minutes": 480, "current_lat": LAT, "current_lng": LNG}
    action = svc._pick_order("D", status, DriverRules(), _plan(), 480, LAT, LNG, 0, 1440)
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "BIG", action


def test_pick_order_still_picks_higher_net_among_similar_sizes() -> None:
    """Sanity: among same-duration orders the higher-net one still wins."""
    a = _cargo("A", price=1000.0, cost_time=120)
    b = _cargo("B", price=1500.0, cost_time=120)
    svc = ModelDecisionService(_Stub([a, b], progress=480))
    status = {"simulation_progress_minutes": 480, "current_lat": LAT, "current_lng": LNG}
    action = svc._pick_order("D", status, DriverRules(), _plan(), 480, LAT, LNG, 0, 1440)
    assert action["params"]["cargo_id"] == "B", action


# ------------------------------------------------- decision-LLM candidates

def _capture_llm_payload() -> dict:
    small = _cargo("SMALL", price=600.0, cost_time=60)
    big = _cargo("BIG", price=5000.0, cost_time=600)
    stub = _Stub(
        [small, big],
        progress=480,
        on_chat=lambda p: {
            "choices": [{"message": {"content": json.dumps(
                {"action": "wait", "params": {"duration_minutes": 30}})}}]
        },
    )
    svc = ModelDecisionService(stub)
    status = {"simulation_progress_minutes": 480, "current_lat": LAT, "current_lng": LNG}
    svc._llm_decide_with_history(
        "D", status, DriverRules(), _plan(), DecisionHistory(),
        480, LAT, LNG, 0, 480,
    )
    assert stub.payloads, "LLM was not called"
    user_msg = stub.payloads[0]["messages"][1]["content"]
    sys_msg = stub.payloads[0]["messages"][0]["content"]
    return {"user": user_msg, "system": sys_msg}


def test_llm_candidates_expose_absolute_net() -> None:
    msgs = _capture_llm_payload()
    assert '"net":' in msgs["user"], "candidate rows must expose absolute net"


def test_llm_candidates_ranked_big_value_first() -> None:
    """The big-value order must appear before the small fast one in the list the
    model reads (overhead-amortised ordering)."""
    user = _capture_llm_payload()["user"]
    assert user.index("BIG") < user.index("SMALL"), user


def test_llm_system_prompt_weighs_absolute_net() -> None:
    sys_msg = _capture_llm_payload()["system"]
    assert "绝对净收益" in sys_msg


# ------------------------------------------- decision-LLM context width (§-17)
# Gross push v8: spend the spare token budget on a WIDER per-step context
# (more candidates + more observed-market rows) so the fast LLM can pick better
# without re-enabling thinking (which A/B-regressed: gross up, penalty doubled).

def _capture_payload(items, liq_rows=None) -> dict:
    stub = _Stub(
        items,
        progress=480,
        on_chat=lambda p: {
            "choices": [{"message": {"content": json.dumps(
                {"action": "wait", "params": {"duration_minutes": 30}})}}]
        },
    )
    svc = ModelDecisionService(stub)
    if liq_rows is not None:
        svc._cargo_liquidity_stats = lambda now, _r=liq_rows: _r  # type: ignore[assignment]
    status = {"simulation_progress_minutes": 480, "current_lat": LAT, "current_lng": LNG}
    svc._llm_decide_with_history(
        "D", status, DriverRules(), _plan(), DecisionHistory(),
        480, LAT, LNG, 0, 480,
    )
    assert stub.payloads, "LLM was not called"
    return {
        "user": stub.payloads[0]["messages"][1]["content"],
        "system": stub.payloads[0]["messages"][0]["content"],
    }


def test_context_width_defaults_widened() -> None:
    """v8 submission defaults: candidate list >= 40, market table >= 20."""
    assert mds._LLM_CARGO_SUMMARY_LIMIT >= 40, mds._LLM_CARGO_SUMMARY_LIMIT
    assert mds._LIQ_TOP_N >= 20, mds._LIQ_TOP_N


def test_llm_candidate_list_respects_widened_limit() -> None:
    """With more feasible candidates than the limit, the model sees exactly the
    widened number of rows (and strictly more than the old 24 default)."""
    items = [
        _cargo(f"C{i:02d}", price=600.0 + i * 10, cost_time=60)
        for i in range(mds._LLM_CARGO_SUMMARY_LIMIT + 5)
    ]
    user = _capture_payload(items)["user"]
    shown = user.count('"cargo_id":')
    assert shown == mds._LLM_CARGO_SUMMARY_LIMIT, shown
    assert shown > 24, f"widening must show more than the old 24, got {shown}"


def test_llm_market_table_respects_widened_limit() -> None:
    """With more observed markets than the limit, the table shows exactly the
    widened number of rows (and strictly more than the old 12 default)."""
    liq = [
        {"city": f"城{i:02d}", "n": 9, "net_per_h": 120.0, "lat": LAT + i * 0.1, "lng": LNG}
        for i in range(mds._LIQ_TOP_N + 5)
    ]
    user = _capture_payload([_cargo("X", price=600.0, cost_time=60)], liq_rows=liq)["user"]
    shown = user.count("近3天见")
    assert shown == mds._LIQ_TOP_N, shown
    assert shown > 12, f"widening must show more market rows than the old 12, got {shown}"


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
            print(f"ERROR {t.__name__}: {exc!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
