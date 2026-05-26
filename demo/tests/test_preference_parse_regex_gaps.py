"""正则兜底已知盲区（gap）跟踪。

这些变体表达**故意**会让纯正则路径漏解析；目的：
1. 量化换数据集时的风险——这些表达 LLM 必须兜住。
2. 接入真实 LLM 或微调后，应能让对应 case 转为通过。

测试用 ``expectedFailure`` 标记：
- 当前断言失败 = 预期（仍依赖 LLM）。
- 若未来变成 pass = 表明正则覆盖度提升或 LLM 已介入，应将 case 移到主 fixture。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent.preference_parser import parse_preferences  # noqa: E402
from tests.test_preference_parse_cases import _check_expectation  # noqa: E402

_GAP_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "preference_parse_regex_gaps.json")


def _load_gap_cases() -> list[dict]:
    with open(_GAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class RegexGapTrackingTest(unittest.TestCase):
    """跟踪正则在变体表达下的盲区；当前全部预期失败。"""


def _make_xfail(case: dict):
    @unittest.expectedFailure
    def test(self: RegexGapTrackingTest) -> None:
        pref = {
            "content": case["content"],
            "penalty_amount": case.get("penalty_amount"),
            "penalty_cap": case.get("penalty_cap"),
        }
        rules = parse_preferences([pref])  # 纯正则
        ok, reason = _check_expectation(case, rules)
        self.assertTrue(ok, f"[{case['id']}] {reason}")

    test.__name__ = f"test_regex_gap_{case['id']}"
    return test


for _case in _load_gap_cases():
    setattr(RegexGapTrackingTest, f"test_regex_gap_{_case['id']}", _make_xfail(_case))


if __name__ == "__main__":
    unittest.main()
