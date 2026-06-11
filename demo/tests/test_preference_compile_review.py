"""Regression tests for the preference compile -> review -> merge pipeline.

Architecture they guard: natural-language preferences are compiled by the LLM
(stage-1 extraction + stage-2 audit/repair) into the generic constraint IR
(DriverRules). The merge layer performs ONLY deterministic structural
validation -- no phrasing-keyword whitelists -- so unknown finals drivers
(广东 / 江浙沪) whose wording differs from the local sample driver can never
lose a correctly-extracted rule to an off-vocabulary gate. The legacy regex
supplement layer must run ONLY in the offline fallback (model unreachable),
never on top of a successful LLM compile, so its sample-tuned patterns cannot
mis-fire on unknown drivers' text.

Run: ``python demo/tests/test_preference_compile_review.py`` (no pytest dependency).
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
)


class _SeqModelApi:
    """SimulationApiPort stub replaying a fixed sequence of model outputs."""

    def __init__(self, responses=None):  # noqa: ANN001
        self._responses = list(responses or [])
        self.calls: list[dict] = []

    def get_driver_status(self, driver_id):  # noqa: ANN001, ANN201
        return {}

    def query_cargo(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        return {"items": []}

    def query_decision_history(self, driver_id, step):  # noqa: ANN001, ANN201
        return {"records": []}

    def model_chat_completion(self, payload):  # noqa: ANN001, ANN201
        self.calls.append(payload)
        if not self._responses:
            raise RuntimeError("model unavailable")
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"choices": [{"message": {"content": json.dumps(r, ensure_ascii=False)}}]}


# Wordings deliberately OFF the old keyword whitelists (the finals scenario).
ODD_NIGHT_TEXT = "天一擦黑就得把车扔回库里，二十二点起到次晨五点人绝不沾方向盘。"
ODD_LIMIT_TEXT = "一天里头顶破天揽三票活，单程超两百千米的路咱不沾。"


def test_no_drive_window_survives_off_vocabulary_wording() -> None:
    """A correctly-extracted overnight window must be kept even when the text
    contains none of the legacy action keywords (the old gate dropped it)."""
    svc = ModelDecisionService(_SeqModelApi())
    rules = DriverRules()
    svc._merge_llm_rules(
        rules, {"no_drive_windows": [{"start_hour": 22, "end_hour": 5}]}, [ODD_NIGHT_TEXT]
    )
    assert (22 * 60, 5 * 60 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows


def test_scalar_limits_survive_off_vocabulary_wording() -> None:
    """daily_order_limit / haul_max_km must merge without keyword grounding."""
    svc = ModelDecisionService(_SeqModelApi())
    rules = DriverRules()
    svc._merge_llm_rules(
        rules, {"daily_order_limit": 3, "haul_max_km": 200}, [ODD_LIMIT_TEXT]
    )
    assert rules.daily_order_limit == 3, rules.daily_order_limit
    assert rules.haul_max_km == 200.0, rules.haul_max_km


def _pref_status(text: str) -> dict:
    return {
        "current_lat": 23.0,
        "current_lng": 113.2,
        "preferences": [{"content": text, "penalty_amount": 2700}],
    }


def test_review_pass_restores_missed_daily_window() -> None:
    """Stage-2 audit must repair a stage-1 miss of a daily no-drive window."""
    extract = {"no_drive_windows": [], "forbidden_categories": []}
    review = {"no_drive_windows": [{"start_hour": 21, "end_hour": 6}], "forbidden_categories": []}
    api = _SeqModelApi([extract, review])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX", _pref_status(ODD_NIGHT_TEXT))
    assert (21 * 60, 6 * 60 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows
    assert len(api.calls) == 2, len(api.calls)  # extract + one review call only


def test_review_pass_cannot_delete_stage1_windows() -> None:
    """Fail-safe: the audit may correct daily windows but never drop them all
    (a missed daily window compounds into the largest penalty class)."""
    extract = {"no_drive_windows": [{"start_hour": 22, "end_hour": 5}]}
    review = {"no_drive_windows": []}
    api = _SeqModelApi([extract, review])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX", _pref_status(ODD_NIGHT_TEXT))
    assert (22 * 60, 5 * 60 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows


def test_review_failure_keeps_stage1_extraction() -> None:
    """If the review call dies, the stage-1 extraction is used unchanged."""
    extract = {"no_drive_windows": [{"start_hour": 23, "end_hour": 4}]}
    api = _SeqModelApi([extract, RuntimeError("gateway down"), RuntimeError("gateway down")])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX", _pref_status(ODD_NIGHT_TEXT))
    assert (23 * 60, 4 * 60 + DAY_MINUTES) in rules.no_drive_windows, rules.no_drive_windows


def test_regex_supplements_do_not_run_on_successful_compile() -> None:
    """The sample-tuned regex layer must not add rules on top of a successful
    LLM compile (its patterns can mis-fire on unknown drivers' wording)."""
    # This text would trip the legacy blackout regex (不跑 + 号) and the daily
    # rest regex if supplements ran online.
    trap_text = "三号那天连轴转也无妨，不跑亏本买卖，连续歇够8小时再说。"
    extract = {"no_drive_windows": [], "blackout": [], "daily_rest_hours": None}
    review = dict(extract)
    api = _SeqModelApi([extract, review])
    svc = ModelDecisionService(api)
    rules = svc._ensure_rules("DX", _pref_status(trap_text))
    assert rules.blackout == [], rules.blackout
    assert rules.daily_rest_minutes == 0, rules.daily_rest_minutes


def test_relative_month_quota_resolves_to_current_month() -> None:
    """A quota phrased as 「本月…」(month missing/relative) must land in the
    month the preference became visible, not be silently dropped."""
    svc = ModelDecisionService(_SeqModelApi())
    rules = DriverRules()
    svc._merge_llm_rules(
        rules,
        {"monthly_category_targets": [{"month": None, "category": "聚酯切片", "min_orders": 8}]},
        ["本月聚酯切片这类化工货起码要拉满八车，缺一车罚一回钞票。"],
        default_month_idx=1,
    )
    assert rules.monthly_category_targets == {1: {"聚酯切片": 8}}, rules.monthly_category_targets


def test_offline_fallback_still_parses_longhaul_cap() -> None:
    """With the model unreachable, the regex fallback must still recover the
    sample driver's long-haul cap preference."""
    api = _SeqModelApi()  # every model call raises
    svc = ModelDecisionService(api)
    text = "不爱接那种一跑就是大半天的远活，每个月超过八小时的长途只能接最多5单，多一单扣一次。"
    rules = svc._ensure_rules("DX", _pref_status(text))
    assert rules.longhaul_max_orders == 5, rules.longhaul_max_orders
    assert rules.longhaul_threshold_minutes == 480, rules.longhaul_threshold_minutes


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
