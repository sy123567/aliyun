"""Unit tests for first_order_rule deadline handling in scoring.

D004 regression: 首单晚于 12:00 的违规从 3 次上升到 7 次。
根因：
1. wait 跨越 first_order deadline 的惩罚仅为 penalty * 0.5，太弱。
2. 截止前 2h 内接单缺乏正向激励，agent 在有单时仍倾向 wait。

修复：
1. wait 跨越 deadline 的惩罚提升为全额 penalty。
2. 截止前 2h 内接单给予 urgency bonus（最高 penalty * 0.5）。
"""
from __future__ import annotations

import os
import sys
import unittest

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent import driver_memory, scoring  # noqa: E402
from agent.preference_parser import FirstOrderRule, ParsedRules  # noqa: E402


def _d004_rules() -> ParsedRules:
    """D004 基础规则：首单 12:00 前，penalty ¥200。"""
    rules = ParsedRules()
    rules.first_order_rule = FirstOrderRule(before_hour=12, penalty_amount=200.0)
    return rules


def _ctx_at(minutes: int) -> scoring.DecisionContext:
    return scoring.DecisionContext(
        driver_id="D004_FO",
        cost_per_km=1.5,
        truck_length="4.2m",
        current_lat=22.5,
        current_lng=114.0,
        current_minutes=minutes,
        horizon_minutes=31 * 1440,
    )


class WaitCrossesFirstOrderDeadlineTest(unittest.TestCase):
    """wait 跨越首单截止时间应施加全额罚分。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D004_FO")
        self.rules = _d004_rules()

    def test_wait_crossing_deadline_full_penalty(self) -> None:
        # 11:00 开始 wait 120min → 跨越 12:00
        scored = scoring.score_wait(120, self.rules, self.mem, _ctx_at(11 * 60), has_good_order=False)
        self.assertIn("first_order_deadline_wait_penalty", scored.breakdown)
        self.assertAlmostEqual(scored.breakdown["first_order_deadline_wait_penalty"], -200.0)

    def test_wait_not_crossing_deadline_no_penalty(self) -> None:
        # 10:00 开始 wait 60min → 11:00 结束，不跨越
        scored = scoring.score_wait(60, self.rules, self.mem, _ctx_at(10 * 60), has_good_order=False)
        self.assertNotIn("first_order_deadline_wait_penalty", scored.breakdown)

    def test_already_has_order_today_no_penalty(self) -> None:
        # 今天已有 1 单，即使跨越也不罚
        from agent import geo_utils
        today = geo_utils.date_str(11 * 60)
        self.mem.daily_orders[today] = 1
        scored = scoring.score_wait(120, self.rules, self.mem, _ctx_at(11 * 60), has_good_order=False)
        self.assertNotIn("first_order_deadline_wait_penalty", scored.breakdown)


class TakeOrderUrgencyBonusTest(unittest.TestCase):
    """截止前 2h 内接单应获得 urgency bonus。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D004_FO")
        self.rules = _d004_rules()

    def _make_item(self, distance_km: float = 10.0, price: float = 800.0) -> dict:
        return {
            "distance_km": distance_km,
            "cargo": {
                "cargo_id": "C_TEST",
                "start": {"lat": 22.6, "lng": 114.1},
                "end": {"lat": 23.0, "lng": 114.5},
                "price": price,
                "cost_time_minutes": 120,
                "cargo_name": "普通货物",
            },
        }

    def test_urgency_bonus_at_11_30(self) -> None:
        # 11:30 接单（距截止 30min），urgency = 1 - 30/120 = 0.75
        scored = scoring.score_take_order(self._make_item(), self.rules, self.mem, _ctx_at(11 * 60 + 30))
        self.assertIn("first_order_urgency_bonus", scored.breakdown)
        expected = 200.0 * 0.5 * 0.75
        self.assertAlmostEqual(scored.breakdown["first_order_urgency_bonus"], expected)

    def test_no_bonus_at_9_00(self) -> None:
        # 09:00 接单（距截止 3h > 2h），无 bonus
        scored = scoring.score_take_order(self._make_item(), self.rules, self.mem, _ctx_at(9 * 60))
        self.assertNotIn("first_order_urgency_bonus", scored.breakdown)

    def test_no_bonus_after_deadline(self) -> None:
        # 12:30 接单（已过截止），应有 late_penalty 而非 bonus
        scored = scoring.score_take_order(self._make_item(), self.rules, self.mem, _ctx_at(12 * 60 + 30))
        self.assertNotIn("first_order_urgency_bonus", scored.breakdown)
        self.assertIn("first_order_late_penalty", scored.breakdown)

    def test_no_bonus_if_already_has_order(self) -> None:
        from agent import geo_utils
        today = geo_utils.date_str(11 * 60 + 30)
        self.mem.daily_orders[today] = 1
        scored = scoring.score_take_order(self._make_item(), self.rules, self.mem, _ctx_at(11 * 60 + 30))
        self.assertNotIn("first_order_urgency_bonus", scored.breakdown)


if __name__ == "__main__":
    unittest.main()
