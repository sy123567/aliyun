"""表驱动评测：纯正则路径（无 LLM）能否覆盖常见偏好表达。

目的：换数据集时快速暴露 regex 兜底盲点，决定哪些 kind 必须由 LLM 兜住。
失败不一定是 bug——它告诉我们该规则在新表达下必须依赖 LLM 主解析。
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent.preference_parser import parse_preferences  # noqa: E402

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "preference_parse_cases.json")


def _load_cases() -> list[dict[str, Any]]:
    with open(_FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _check_expectation(case: dict[str, Any], rules) -> tuple[bool, str]:
    expect = case.get("expect", {})
    for key, want in expect.items():
        if key == "forbidden_category":
            if not all(name in rules.categories.forbidden for name in want):
                return False, f"forbidden_category missing {want}"
        elif key == "avoid_category":
            if not all(name in rules.categories.avoid for name in want):
                return False, f"avoid_category missing {want}"
        elif key == "no_drive_window":
            if not any(
                w.start_minute == want["start_minute"] and w.end_minute == want["end_minute"]
                for w in rules.no_drive_windows
            ):
                return False, f"no_drive_window {want} not found in {[(w.start_minute,w.end_minute) for w in rules.no_drive_windows]}"
        elif key == "no_drive_window_contains":
            if not any(
                w.start_minute == want["start_minute"] and w.end_minute == want["end_minute"]
                for w in rules.no_drive_windows
            ):
                return False, f"no_drive_window (sub) {want} not found"
        elif key == "home_rule":
            if rules.home_rule is None:
                return False, "home_rule missing"
            for k, v in want.items():
                if abs(getattr(rules.home_rule, k) - v) > 1e-6:
                    return False, f"home_rule.{k}={getattr(rules.home_rule, k)} != {v}"
        elif key == "monthly_day_off":
            if rules.monthly_day_off is None:
                return False, "monthly_day_off missing"
            if rules.monthly_day_off.required_days != want["required_days"]:
                return False, f"monthly_day_off required_days={rules.monthly_day_off.required_days}"
        elif key == "first_order_rule":
            if rules.first_order_rule is None:
                return False, "first_order_rule missing"
            if rules.first_order_rule.before_hour != want["before_hour"]:
                return False, f"first_order_rule.before_hour={rules.first_order_rule.before_hour}"
        elif key == "distance_limit":
            if not any(
                d.kind == want["kind"] and abs(d.max_km - want["max_km"]) < 1e-6
                for d in rules.distance_limits
            ):
                return False, f"distance_limit {want} not found"
        elif key == "daily_order_limit":
            if rules.daily_order_limit is None:
                return False, "daily_order_limit missing"
            if rules.daily_order_limit.max_orders != want["max_orders"]:
                return False, f"daily_order_limit.max_orders={rules.daily_order_limit.max_orders}"
        elif key == "preferred_cargo_id":
            if want not in rules.preferred_cargo_ids:
                return False, f"preferred_cargo_id {want} not in {rules.preferred_cargo_ids}"
        else:
            return False, f"unknown expectation key: {key}"
    return True, ""


class PreferenceParseFixtureTest(unittest.TestCase):
    """每个 fixture 用例独立断言，便于看到具体哪类规则正则没覆盖。"""

    maxDiff = None


def _make_test(case: dict[str, Any]):
    def test(self: PreferenceParseFixtureTest) -> None:
        pref = {
            "content": case["content"],
            "penalty_amount": case.get("penalty_amount"),
            "penalty_cap": case.get("penalty_cap"),
        }
        rules = parse_preferences([pref])  # 不传 llm_caller，纯正则
        ok, reason = _check_expectation(case, rules)
        self.assertTrue(ok, f"[{case['id']}] {reason}")

    test.__name__ = f"test_regex_{case['id']}"
    return test


for _case in _load_cases():
    setattr(PreferenceParseFixtureTest, f"test_regex_{_case['id']}", _make_test(_case))


if __name__ == "__main__":
    unittest.main()
