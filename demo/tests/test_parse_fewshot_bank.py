"""Tests for the dynamic few-shot bank + retriever (parse_fewshot_bank.py).

The bank replaces the static inline few-shot that used to live in
``ModelDecisionService._PARSE_SYSTEM``. These tests validate it structurally
(the few-shot only affects the *model prompt*, so it has no behavioural effect
under the mocked-model unit tests — we therefore check the payloads directly):

- every example's ``out`` is valid JSON whose keys are a subset of the 27
  allowed schema fields;
- the bank is large (>= 400 examples) and contains zero-constraint negatives
  plus anchors;
- retrieval is relevant (a clear night-rest query surfaces a night-window
  example), bounded (<= k results), deterministic (same query -> same list),
  and always includes an anchor negative;
- the system prompt assembled by ``_fewshot_block`` still contains the literal
  "偏好抽取器" so the mocked extractor routing in the other test files keeps
  working, and respects the AGENT_FEWSHOT_TOPK=0 (disable) switch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from agent import parse_fewshot_bank as B  # noqa: E402
from agent.model_decision_service import ModelDecisionService  # noqa: E402


def test_bank_is_large() -> None:
    assert len(B.FEWSHOT_BANK) >= 400, len(B.FEWSHOT_BANK)


def test_every_out_is_valid_json_with_allowed_keys() -> None:
    for ex in B.FEWSHOT_BANK:
        # round-trips as JSON
        json.loads(json.dumps(ex["in"], ensure_ascii=False))
        out = json.loads(json.dumps(ex["out"], ensure_ascii=False))
        for key in out:
            assert key in B.SCHEMA_FIELDS, f"illegal schema field: {key}"


def test_in_has_preferences_list() -> None:
    for ex in B.FEWSHOT_BANK:
        prefs = ex["in"]["preferences"]
        assert isinstance(prefs, list) and prefs, ex["in"]
        assert all(isinstance(p, str) and p for p in prefs), ex["in"]


def test_bank_has_zero_constraint_negatives() -> None:
    """Over-extraction (hallucinated constraints) is the root cause of the
    lowest-gross standing, so the bank must carry many all-default negatives."""
    blank = B._blank()
    negatives = [ex for ex in B.FEWSHOT_BANK if ex["out"] == blank]
    assert len(negatives) >= 30, len(negatives)


def test_bank_has_anchors() -> None:
    anchors = [ex for ex in B.FEWSHOT_BANK if ex.get("anchor")]
    assert len(anchors) >= 3, len(anchors)


def test_penalty_examples_carry_penalty_inputs() -> None:
    """rule_penalties may only be emitted when the input carried penalty_amounts."""
    for ex in B.FEWSHOT_BANK:
        if ex["out"].get("rule_penalties"):
            assert "penalty_amounts" in ex["in"], ex["in"]
        if ex["out"].get("rule_penalty_caps"):
            assert "penalty_caps" in ex["in"], ex["in"]


def test_retrieval_is_bounded() -> None:
    res = B.select_examples("夜里十点到早上六点不开车睡觉", k=40)
    assert 0 < len(res) <= 40, len(res)
    res5 = B.select_examples("只跑长三角，江浙沪以外不接", k=5)
    assert len(res5) <= 5, len(res5)


def test_retrieval_is_deterministic() -> None:
    q = "每天23点前回到家两公里内，次日7点前不接单"
    a = B.select_examples(q, k=30)
    b = B.select_examples(q, k=30)
    assert [e["in"]["preferences"] for e in a] == [e["in"]["preferences"] for e in b]


def test_retrieval_is_relevant_for_night_query() -> None:
    """A clear night-rest query must surface at least one no_drive_windows example."""
    res = B.select_examples("晚上十点到早上六点不出车，停车睡觉休息", k=40)
    assert any(e["out"]["no_drive_windows"] for e in res), "no night-window example retrieved"


def test_retrieval_is_relevant_for_category_query() -> None:
    res = B.select_examples("生鲜货一律不接，碰都不碰", k=40)
    hit = any("生鲜" in e["out"].get("forbidden_categories", []) for e in res)
    assert hit, "no forbidden-category example retrieved for a clear category query"


def test_retrieval_always_includes_anchor() -> None:
    res = B.select_examples("单趟运距不超过六百公里", k=40, n_anchor=4)
    assert any(e.get("anchor") for e in res), "no anchor negative in retrieval result"


def test_topk_zero_returns_empty() -> None:
    assert B.select_examples("夜里十点不开车", k=0) == []


def test_fewshot_block_contains_extractor_marker() -> None:
    """The assembled extractor system prompt must keep '偏好抽取器' so the
    ScriptedApi routing in the other tests still recognises extract calls."""
    block = ModelDecisionService._fewshot_block(["夜里十点到早六点不出车"])
    system = ModelDecisionService._PARSE_SYSTEM + block
    assert "偏好抽取器" in system
    # the block itself should render examples in the 入/出 format
    assert "入:" in block and "出:" in block


def test_fewshot_block_renders_examples() -> None:
    block = ModelDecisionService._fewshot_block(["只拉冷链冷藏货"])
    # one "示例N：" header per retrieved example (plus the intro mentions 示例)
    assert block.count("示例") >= 10, block.count("示例")


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
