"""Test that the LLM decision path ranks candidates by value, not distance.

query_cargo returns nearest-first; the old code truncated the feasible list at
8 in that order, so cheap nearby orders could crowd every high-net candidate
out of the LLM's view. Candidates must now be ranked (category-target first,
then net-per-hour) before truncation.

Run: ``python demo/tests/test_llm_candidate_ranking.py`` (no pytest dependency).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent.model_decision_service import (  # noqa: E402
    DecisionHistory,
    DriverRules,
    ModelDecisionService,
)


def _resp(obj) -> dict:
    return {"choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}]}


def _cargo(cargo_id, price, name="普货"):
    return {
        "distance_km": 7.6,
        "cargo": {
            "cargo_id": cargo_id,
            "cargo_name": name,
            "price": float(price),
            "cost_time_minutes": 120,
            "start": {"lat": 23.05, "lng": 113.25, "city": "广州"},
            "end": {"lat": 23.5, "lng": 113.8, "city": "广州"},
        },
    }


class CaptureApi:
    def __init__(self, items):
        self._items = items
        self.decision_user_prompts: list[str] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {"simulation_progress_minutes": 600}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": [dict(it) for it in self._items]}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "智能货运调度决策AI" in system:
            self.decision_user_prompts.append(user)
            return _resp({"action": "wait", "params": {"duration_minutes": 60}, "reason": "test"})
        if "语义判定助手" in system:
            return _resp({"answer": False})
        return _resp({})


def _candidates_from_prompt(prompt: str) -> list[dict]:
    m = re.search(r"候选货源\(\d+条\)\n(\[.*\])", prompt)
    assert m, prompt
    return json.loads(m.group(1))


def test_high_net_candidate_survives_truncation() -> None:
    # 9 cheap orders sit closer (listed first = nearest-first), 1 high-net
    # order comes last. The LLM must still see the high-net one, ranked first.
    items = [_cargo(f"C-LO{i}", price=800) for i in range(1, 10)]
    items.append(_cargo("C-HI", price=5000))
    api = CaptureApi(items)
    svc = ModelDecisionService(api)
    plan = {
        "orders_today": {},
        "monthly_longhual": {},
        "monthly_category_orders": {},
        "monthly_deadhead_km": {},
        "zeng_order_days": set(),
        "failed_cargo_ids": set(),
        "off_days": set(),
    }
    action = svc._llm_decide_with_history(
        "DR01", {}, DriverRules(), plan, DecisionHistory(),
        now=600, lat=23.0, lng=113.2, day=0, tod=600,
    )
    assert action is not None and action["action"] == "wait", action
    assert api.decision_user_prompts, "decision LLM was not called"
    cands = _candidates_from_prompt(api.decision_user_prompts[0])
    assert len(cands) == 8, [c["cargo_id"] for c in cands]
    assert cands[0]["cargo_id"] == "C-HI", [c["cargo_id"] for c in cands]
    assert cands[0]["net_per_h"] >= max(c["net_per_h"] for c in cands), cands


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
