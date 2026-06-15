"""Tests for the balanced value-layer optimisation (plan A/B/C).

These cover the *mechanics* of each knob (the net effect can only be measured on
the official platform across multiple runs — notes §2). Every change is
env-tunable and by construction never relaxes a compliance guard nor accepts an
unprofitable order (the net>0 filter is unchanged).

Run: ``python demo/tests/test_balanced_value_layer.py`` (no pytest dep).
"""

from __future__ import annotations

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
)

LAT, LNG = 22.92, 113.18


class _StubApi:
    def __init__(self, items=None, progress=0):
        self._items = items or []
        self._progress = progress

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": self._progress, "current_lat": LAT, "current_lng": LNG}

    def query_cargo(self, *a, **k):  # noqa: ANN002, ANN003, ANN201
        return {"items": self._items, "k": k.get("k", 60)}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        return {"choices": [{"message": {"content": "{}"}}]}


def _svc(items=None, progress=0):
    return ModelDecisionService(_StubApi(items, progress))


# ============================================================ A1: ranking score

def test_rank_score_neutral_without_chain_or_abs_boost() -> None:
    """With chain weight active but no destination liquidity, and abs-alpha 0,
    the score equals the plain overhead-amortised rate. (abs-alpha now defaults
    to 0.2, so pin it to 0 here to verify the neutral baseline.)"""
    original = mds._ABS_NET_ALPHA
    mds._ABS_NET_ALPHA = 0.0
    try:
        base = ModelDecisionService._amortized_rate(1000.0, 100)
        assert abs(ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0) - base) < 1e-9
    finally:
        mds._ABS_NET_ALPHA = original


def test_rank_score_abs_net_alpha_on_by_default_lifts_big_haul() -> None:
    """Shipped default now emphasises big absolute net (2026-06-14 gross push):
    a big haul ranks strictly above its plain overhead-amortised rate out of the
    box, so big orders stop getting buried behind marginally-faster small ones."""
    assert mds._ABS_NET_ALPHA > 0.0, "default AGENT_ABS_NET_ALPHA must emphasise big net"
    big = ModelDecisionService._candidate_rank_score(8000.0, 600, 0.0)
    base_big = ModelDecisionService._amortized_rate(8000.0, 600)
    assert big > base_big


def test_rank_score_chain_liquidity_boosts() -> None:
    """A liquid destination market boosts the ranking score (default weight).
    Pin abs-alpha to 0 to isolate the chain term (abs-alpha now defaults > 0)."""
    original = mds._ABS_NET_ALPHA
    mds._ABS_NET_ALPHA = 0.0
    try:
        base = ModelDecisionService._amortized_rate(1000.0, 100)
        boosted = ModelDecisionService._candidate_rank_score(1000.0, 100, 100.0)
        assert boosted > base
        # bounded: ~ (1 + weight) at/above the reference market value
        assert boosted <= base * (1.0 + mds._CHAIN_VALUE_WEIGHT) + 1e-9
    finally:
        mds._ABS_NET_ALPHA = original


def test_rank_score_chain_can_flip_two_equal_net_orders() -> None:
    """Two equal net/duration orders: the one into the liquid market ranks
    higher, so chain value actually changes the choice."""
    dead = ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0)
    live = ModelDecisionService._candidate_rank_score(1000.0, 100, 90.0)
    assert live > dead


def test_rank_score_abs_net_alpha_emphasises_big() -> None:
    """When AGENT_ABS_NET_ALPHA>0 the absolute-net log term lifts a big haul's
    score relative to a small one beyond the overhead amortisation alone."""
    original = mds._ABS_NET_ALPHA
    mds._ABS_NET_ALPHA = 0.2
    try:
        big = ModelDecisionService._candidate_rank_score(8000.0, 600, 0.0)
        base_big = ModelDecisionService._amortized_rate(8000.0, 600)
        assert big > base_big
    finally:
        mds._ABS_NET_ALPHA = original


def test_rank_score_chain_depth_default_off_is_noop() -> None:
    """AGENT_CHAIN_DEPTH_WEIGHT defaults to 0 → supplying a destination depth
    count must not change the score (strict no-op until tuned on the platform)."""
    assert mds._CHAIN_DEPTH_WEIGHT == 0.0, "default AGENT_CHAIN_DEPTH_WEIGHT must be 0 (neutral)"
    without = ModelDecisionService._candidate_rank_score(1000.0, 100, 90.0)
    with_depth = ModelDecisionService._candidate_rank_score(1000.0, 100, 90.0, 50)
    assert abs(without - with_depth) < 1e-9


def test_rank_score_chain_depth_rewards_deeper_market() -> None:
    """With the depth term enabled, two orders with identical net AND identical
    mean destination liquidity rank by re-load depth: the deeper market (more
    recently-observed orders) wins, and the bonus is bounded by (1 + weight)."""
    original_alpha = mds._ABS_NET_ALPHA
    original_depth = mds._CHAIN_DEPTH_WEIGHT
    mds._ABS_NET_ALPHA = 0.0
    mds._CHAIN_DEPTH_WEIGHT = 0.3
    try:
        no_depth = ModelDecisionService._candidate_rank_score(1000.0, 100, 60.0, 0)
        shallow = ModelDecisionService._candidate_rank_score(1000.0, 100, 60.0, 1)
        deep = ModelDecisionService._candidate_rank_score(1000.0, 100, 60.0, 40)
        assert deep > shallow >= no_depth
        assert deep <= no_depth * (1.0 + mds._CHAIN_DEPTH_WEIGHT) + 1e-9
    finally:
        mds._ABS_NET_ALPHA = original_alpha
        mds._CHAIN_DEPTH_WEIGHT = original_depth


def test_rank_score_chain_depth_neutral_when_dest_illiquid() -> None:
    """Depth only applies to a liquidity-positive destination: a dead drop-off
    market (chain_liq<=0) gets no depth bonus even with many observed orders, so
    a high count never rescues a stranding destination."""
    original_depth = mds._CHAIN_DEPTH_WEIGHT
    mds._CHAIN_DEPTH_WEIGHT = 0.5
    try:
        dead_deep = ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0, 99)
        dead_none = ModelDecisionService._candidate_rank_score(1000.0, 100, 0.0, 0)
        assert abs(dead_deep - dead_none) < 1e-9
    finally:
        mds._CHAIN_DEPTH_WEIGHT = original_depth


# ===================================================== A1c: near-hub chain credit

_HUB = {"city": "广州", "n": 30, "net_per_h": 80.0, "lat": 23.13, "lng": 113.26}


def test_nearby_chain_liquidity_default_off_is_noop() -> None:
    """AGENT_CHAIN_NEAR_WEIGHT defaults to 0 → a drop-off right on top of a busy
    hub still earns no near-credit (strict no-op until tuned on the platform)."""
    assert mds._CHAIN_NEAR_WEIGHT == 0.0, "default AGENT_CHAIN_NEAR_WEIGHT must be 0 (neutral)"
    liq, n = ModelDecisionService._nearby_chain_liquidity([_HUB], _HUB["lat"], _HUB["lng"])
    assert liq == 0.0 and n == 0


def test_nearby_chain_liquidity_credits_close_hub_and_decays_with_distance() -> None:
    """With the lever on, a drop-off near a liquid hub is credited that hub's
    rate, decayed linearly to zero at the radius; full weight at zero distance."""
    original = mds._CHAIN_NEAR_WEIGHT
    mds._CHAIN_NEAR_WEIGHT = 1.0
    try:
        # on the hub: full rate (decay ~1)
        on_liq, on_n = ModelDecisionService._nearby_chain_liquidity(
            [_HUB], _HUB["lat"], _HUB["lng"]
        )
        assert on_n == _HUB["n"]
        assert abs(on_liq - _HUB["net_per_h"]) < 1.0, on_liq
        # ~half radius away → roughly half credit, still less than on-hub
        half_lat = _HUB["lat"] + (mds._CHAIN_NEAR_RADIUS_KM / 2.0) / 111.0
        half_liq, _ = ModelDecisionService._nearby_chain_liquidity([_HUB], half_lat, _HUB["lng"])
        assert 0.0 < half_liq < on_liq
        # weight scales the credit down linearly
        mds._CHAIN_NEAR_WEIGHT = 0.5
        scaled_liq, _ = ModelDecisionService._nearby_chain_liquidity(
            [_HUB], _HUB["lat"], _HUB["lng"]
        )
        assert abs(scaled_liq - on_liq * 0.5) < 1.0, (scaled_liq, on_liq)
    finally:
        mds._CHAIN_NEAR_WEIGHT = original


def test_nearby_chain_liquidity_zero_beyond_radius() -> None:
    """A hub farther than the radius (and an empty/illiquid table) credits
    nothing — the drop-off stays a dead city, so the lever never invents chains."""
    original = mds._CHAIN_NEAR_WEIGHT
    mds._CHAIN_NEAR_WEIGHT = 1.0
    try:
        far_lat = _HUB["lat"] + (mds._CHAIN_NEAR_RADIUS_KM * 2.0) / 111.0
        far_liq, far_n = ModelDecisionService._nearby_chain_liquidity([_HUB], far_lat, _HUB["lng"])
        assert far_liq == 0.0 and far_n == 0
        # empty table and unknown coords both credit nothing
        assert ModelDecisionService._nearby_chain_liquidity([], _HUB["lat"], _HUB["lng"]) == (0.0, 0)
        assert ModelDecisionService._nearby_chain_liquidity([_HUB], 0.0, 0.0) == (0.0, 0)
        # an illiquid (net_per_h<=0) hub is ignored even if it is right here
        dead_hub = {"city": "X", "n": 9, "net_per_h": 0.0, "lat": _HUB["lat"], "lng": _HUB["lng"]}
        assert ModelDecisionService._nearby_chain_liquidity([dead_hub], _HUB["lat"], _HUB["lng"]) == (0.0, 0)
    finally:
        mds._CHAIN_NEAR_WEIGHT = original


# ====================================================== A3: weak-local reposition

def _cargo(cid, *, price, cost_time, slat=LAT, slng=LNG):
    cargo = {
        "cargo_id": cid,
        "cargo_name": "普货",
        "start": {"lat": slat, "lng": slng, "city": "广州"},
        "end": {"lat": slat, "lng": slng, "city": "广州"},
        "price": float(price),
        "cost_time_minutes": int(cost_time),
        "load_time": None,
    }
    return {"cargo": cargo, "distance_km": 0.0}


def test_picked_order_value_recovers_net_per_h() -> None:
    svc = _svc()
    rules = DriverRules()
    item = _cargo("W", price=200.0, cost_time=120)  # zero-distance -> net 200, 100/h
    plan = {"_scan_items": (480, [item]), "monthly_deadhead_km": {}}
    order = {"action": "take_order", "params": {"cargo_id": "W"}}
    val = svc._picked_order_value(order, rules, plan, 480, LAT, LNG, 0, 1440)
    assert val is not None
    net, nph = val
    assert abs(net - 200.0) < 1.0 and abs(nph - 100.0) < 1.0, val


def test_anti_strand_min_net_gate_blocks_worse_target() -> None:
    """A3 gate: a relocation anchor that does not beat the local order's net is
    not taken (we never trade a sure order for a worse one)."""
    # remote rich cargo ~0.5deg east (~50km): reachable by reposition
    remote = _cargo("RICH", price=6000.0, cost_time=120, slat=LAT, slng=LNG + 0.5)
    svc = _svc([remote], progress=480)
    rules = DriverRules()
    plan = {
        "_scan_items": (480, [remote]), "strand_count": {}, "strand_repo": set(),
        "monthly_deadhead_km": {},
    }
    # min_net below the anchor's net -> diverts (returns a reposition)
    act_low = svc._anti_strand("D", rules, plan, 480, LAT, LNG, 0, 1440, min_net=100.0)
    assert act_low is not None and act_low["action"] == "reposition", act_low
    # min_net above any achievable net -> blocked
    plan2 = {
        "_scan_items": (480, [remote]), "strand_count": {}, "strand_repo": set(),
        "monthly_deadhead_km": {},
    }
    act_high = svc._anti_strand("D", rules, plan2, 480, LAT, LNG, 0, 1440, min_net=1_000_000.0)
    assert act_high is None, act_high


# ============================================================ B1: penalty cap

NIGHT = (1260, 1800)


def _night_rules(*, pen=500.0, cap=None):
    rules = DriverRules()
    rules.no_drive_windows.append(NIGHT)
    rules.rule_penalties["night_window"] = pen
    if cap is not None:
        rules.rule_caps["night_window"] = cap
    return rules


def test_night_window_capped_detects_exhausted_cap() -> None:
    svc = _svc()
    rules = _night_rules(pen=500.0, cap=1000.0)  # 2 crossings exhaust the cap
    plan = {"monthly_night_violations": {0: 2}}
    assert svc._night_window_capped(rules, plan, 0) is True
    plan_below = {"monthly_night_violations": {0: 1}}
    assert svc._night_window_capped(rules, plan_below, 0) is False
    # no cap parsed -> never credited
    assert svc._night_window_capped(_night_rules(cap=None), {"monthly_night_violations": {0: 9}}, 0) is False


def test_capped_night_crossing_is_free_in_evaluate() -> None:
    """Once the night cap is exhausted, a crossing order is priced with NO rest
    penalty (the scorer charges no more)."""
    svc = _svc()
    rules = _night_rules(pen=500.0, cap=1000.0)
    cargo = {
        "cargo_id": "C", "cargo_name": "普货",
        "start": {"lat": LAT, "lng": LNG, "city": "广州"},
        "end": {"lat": LAT, "lng": LNG, "city": "广州"},
        "price": 2000.0, "cost_time_minutes": 300, "load_time": None,
    }
    item = {"cargo": cargo, "distance_km": 0.0}
    # not capped: net = 2000 - 500 = 1500
    uncapped = svc._evaluate_cargo(cargo, item, rules, set(), 1080, 1260, LAT, LNG, night_capped=False)
    assert uncapped is not None and abs(uncapped[0] - 1500.0) < 1.0, uncapped
    # capped: penalty waived, net = 2000
    capped = svc._evaluate_cargo(cargo, item, rules, set(), 1080, 1260, LAT, LNG, night_capped=True)
    assert capped is not None and abs(capped[0] - 2000.0) < 1.0, capped


def test_longhaul_cap_credit_after_cap_exhausted() -> None:
    svc = _svc()
    rules = DriverRules()
    rules.longhaul_cap_orders = 5
    rules.rule_penalties["longhaul_cap"] = 1000.0
    rules.rule_caps["longhaul_cap"] = 2000.0  # 2 over-cap orders exhaust it
    # 7 long hauls -> 2 over cap -> 2*1000 == cap -> credited
    assert svc._longhaul_cap_credit(rules, {"monthly_longhual": {0: 7}}, 0) is True
    # 6 long hauls -> 1 over cap -> 1000 < 2000 -> not yet
    assert svc._longhaul_cap_credit(rules, {"monthly_longhual": {0: 6}}, 0) is False


def test_penalty_cap_credit_can_be_disabled() -> None:
    svc = _svc()
    rules = _night_rules(pen=500.0, cap=1000.0)
    original = mds._PENALTY_CAP_CREDIT
    mds._PENALTY_CAP_CREDIT = False
    try:
        assert svc._night_window_capped(rules, {"monthly_night_violations": {0: 9}}, 0) is False
    finally:
        mds._PENALTY_CAP_CREDIT = original


def test_rule_caps_parsed_from_llm_payload() -> None:
    svc = _svc()
    rules = DriverRules()
    svc._merge_llm_rules(rules, {
        "rule_penalties": {"night_window": 500},
        "rule_penalty_caps": {"night_window": 5000},
    })
    assert rules.rule_penalties.get("night_window") == 500
    assert rules.rule_caps.get("night_window") == 5000


# ===================================================== B2: category softening

def _full_plan() -> dict:
    return {
        "rest_done": set(), "zeng_order_days": set(), "dated_single_done": set(),
        "dated_route_done": set(), "strand_repo": set(), "orders_today": {},
        "total_deadhead_km": 0.0, "monthly_deadhead_km": {}, "must_visit_days": {},
        "first_order_taken": set(), "home_done": set(), "monthly_longhual": {},
        "monthly_category_orders": {}, "failed_cargo_ids": set(),
        "failed_cargo_reasons": {}, "off_days": set(), "_consecutive_waits": 0,
        "strand_count": {},
    }


def _urgent_category_rules():
    rules = DriverRules()
    rules.monthly_category_targets = {0: {"水果": 12}}  # March quota, none done
    rules.rule_penalties["category_targets"] = 500.0
    return rules


def test_category_soft_takes_big_noncat_order_when_net_beats_penalty() -> None:
    # day 20 of March: remaining 11 <= 12+10 -> urgent; only a big non-cat order
    big = _cargo("BIG", price=3000.0, cost_time=120)  # net ~3000 > 500
    svc = _svc([big], progress=20 * 1440)
    status = {"simulation_progress_minutes": 20 * 1440, "current_lat": LAT, "current_lng": LNG}
    action = svc._pick_order(
        "D", status, _urgent_category_rules(), _full_plan(), 20 * 1440, LAT, LNG, 20, 21 * 1440
    )
    assert action is not None and action["action"] == "take_order", action
    assert action["params"]["cargo_id"] == "BIG", action


def test_category_soft_off_keeps_urgent_skip() -> None:
    big = _cargo("BIG", price=3000.0, cost_time=120)
    svc = _svc([big], progress=20 * 1440)
    status = {"simulation_progress_minutes": 20 * 1440, "current_lat": LAT, "current_lng": LNG}
    original = mds._CATEGORY_SOFT
    mds._CATEGORY_SOFT = False
    try:
        action = svc._pick_order(
            "D", status, _urgent_category_rules(), _full_plan(), 20 * 1440, LAT, LNG, 20, 21 * 1440
        )
        assert action is None, action  # urgent skip preserved
    finally:
        mds._CATEGORY_SOFT = original


# ============================================================ C1: chain lookahead

def test_decision_prompt_has_chain_lookahead_guidance() -> None:
    """C1: the decision system prompt instructs the (thinking) model to do 2-3
    step chain lookahead using to_liq + the active-market table."""
    api = _StubApi(items=[], progress=480)
    captured: list[dict] = []
    api.model_chat_completion = lambda payload: (  # type: ignore[assignment]
        captured.append(payload) or {"choices": [{"message": {"content": "{}"}}]}
    )
    svc = ModelDecisionService(api)
    status = {"simulation_progress_minutes": 480, "current_lat": LAT, "current_lng": LNG}
    svc._llm_decide_with_history(
        "D", status, DriverRules(), _full_plan(), DecisionHistory(), 480, LAT, LNG, 0, 480
    )
    assert captured, "LLM was not called"
    sys_msg = captured[0]["messages"][0]["content"]
    assert "接力链路前瞻" in sys_msg, sys_msg


# ============================================ G1: selective decision-LLM gate

def _capturing_svc(items, progress=480):
    """Service whose model_chat_completion records every payload, so a test can
    assert whether the per-step decision LLM was actually consulted."""
    api = _StubApi(items=items, progress=progress)
    captured: list[dict] = []
    api.model_chat_completion = lambda payload: (  # type: ignore[assignment]
        captured.append(payload) or {"choices": [{"message": {"content": "{}"}}]}
    )
    return ModelDecisionService(api), captured


def _decide_status(progress=480):
    return {"simulation_progress_minutes": progress, "current_lat": LAT, "current_lng": LNG}


def test_decision_llm_gap_default_off_consults_llm() -> None:
    """AGENT_DECISION_LLM_GAP defaults to 0 → even a clearly dominant top
    candidate still consults the decision LLM (strict no-op until tuned)."""
    assert mds._DECISION_LLM_GAP == 0.0, "default AGENT_DECISION_LLM_GAP must be 0 (neutral)"
    items = [_cargo("BIG", price=8000.0, cost_time=120), _cargo("SMALL", price=200.0, cost_time=120)]
    svc, captured = _capturing_svc(items)
    svc._llm_decide_with_history(
        "D", _decide_status(), DriverRules(), _full_plan(), DecisionHistory(), 480, LAT, LNG, 0, 480
    )
    assert captured, "LLM must be consulted on every step when the gate is off"


def test_decision_llm_gap_skips_llm_on_clear_winner() -> None:
    """With the gate on, a step whose top candidate dominates the runner-up by
    more than the threshold skips the LLM round-trip entirely (returns None →
    the rule engine makes the deterministic pick)."""
    original = mds._DECISION_LLM_GAP
    mds._DECISION_LLM_GAP = 0.2
    try:
        items = [_cargo("BIG", price=8000.0, cost_time=120), _cargo("SMALL", price=200.0, cost_time=120)]
        svc, captured = _capturing_svc(items)
        out = svc._llm_decide_with_history(
            "D", _decide_status(), DriverRules(), _full_plan(), DecisionHistory(), 480, LAT, LNG, 0, 480
        )
        assert out is None, out
        assert not captured, "a clear-winner step must NOT consult the decision LLM"
    finally:
        mds._DECISION_LLM_GAP = original


def test_decision_llm_gap_consults_llm_on_close_candidates() -> None:
    """With the gate on, a step whose top two candidates are near-equal (rank gap
    below the threshold) still consults the LLM — that is exactly where the
    per-step model judgement is worth its tokens."""
    original = mds._DECISION_LLM_GAP
    mds._DECISION_LLM_GAP = 0.2
    try:
        items = [_cargo("A", price=1000.0, cost_time=120), _cargo("B", price=980.0, cost_time=120)]
        svc, captured = _capturing_svc(items)
        svc._llm_decide_with_history(
            "D", _decide_status(), DriverRules(), _full_plan(), DecisionHistory(), 480, LAT, LNG, 0, 480
        )
        assert captured, "an ambiguous (close top-2) step must consult the decision LLM"
    finally:
        mds._DECISION_LLM_GAP = original


def test_decision_llm_gap_consults_llm_when_single_candidate() -> None:
    """The gate only fires with >=2 feasible candidates; a thin market (0/1
    candidate), where repositioning judgement matters, still consults the LLM."""
    original = mds._DECISION_LLM_GAP
    mds._DECISION_LLM_GAP = 0.2
    try:
        svc, captured = _capturing_svc([_cargo("ONLY", price=5000.0, cost_time=120)])
        svc._llm_decide_with_history(
            "D", _decide_status(), DriverRules(), _full_plan(), DecisionHistory(), 480, LAT, LNG, 0, 480
        )
        assert captured, "a single-candidate step must still consult the decision LLM"
    finally:
        mds._DECISION_LLM_GAP = original


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
