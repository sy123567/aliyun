"""Unit tests for cap-aware soft no_drive_window marginal charge.

根因（results_20260523_full_analysis.log）：
- D004「12-13 不接单不空驶」单日 ¥100、月度 cap ¥3,000，实际 28 次违规吃到 ¥2,800。
- 旧代码 `_is_preference_near_violation` 读 `memory.preference_penalty_accum`，但全代码库
  无任何写入点 → 永远 False → cap 永远不会被识别为饱和 → agent 月底仍按 ¥100/单
  全额扣分，过度回避能盈利的违规订单。
- 旧代码也未做「同日重复违规去重」：评测器同日多次违规只罚 1 次，agent 却累加。

修复内容：
1. ``DriverMemory._record_no_drive_window_violations`` 在吸收 take_order/reposition
   历史时按"评测器口径"做 per-day 去重 + cap 封顶累计。
2. ``scoring._soft_no_drive_marginal_charge`` 计算软窗动作的真实边际罚分：
   - cap 已饱和 95%+ → 0
   - 同日已记 1 次 → 0
   - 否则 = ``min(window_penalty × 新触发的天数, cap_remaining)``。
"""
from __future__ import annotations

import os
import sys
import unittest

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent import driver_memory, scoring  # noqa: E402
from agent.preference_parser import ParsedRules, TimeWindowRule  # noqa: E402


def _noon_window(penalty: float = 100.0, cap: float | None = 3000.0) -> TimeWindowRule:
    """复刻 D004「12-13 点不接单不空驶」软偏好窗。"""
    return TimeWindowRule(
        start_minute=12 * 60,
        end_minute=13 * 60,
        raw="12-13 noon soft window",
        penalty_amount=penalty,
        penalty_cap=cap,
    )


def _make_take_order_record(step: int, sim_end_minutes: int, exec_minutes: int) -> dict:
    return {
        "step": step,
        "action": {"action": "take_order", "params": {"cargo_id": f"C{step}"}},
        "simulation_end_time_minutes": sim_end_minutes,
        "step_elapsed_minutes": exec_minutes,
        "action_exec_cost_minutes": exec_minutes,
        "result": {
            "simulation_progress_minutes": sim_end_minutes,
            "accepted": True,
            "haul_distance_km": 0.0,
            "pickup_deadhead_km": 0.0,
        },
    }


def _make_reposition_record(step: int, sim_end_minutes: int, exec_minutes: int) -> dict:
    return {
        "step": step,
        "action": {
            "action": "reposition",
            "params": {"latitude": 22.5, "longitude": 114.0},
        },
        "simulation_end_time_minutes": sim_end_minutes,
        "step_elapsed_minutes": exec_minutes,
        "action_exec_cost_minutes": exec_minutes,
        "result": {
            "simulation_progress_minutes": sim_end_minutes,
            "distance_km": 0.0,
        },
    }


class NoDriveWindowAccumulationTest(unittest.TestCase):
    """``preference_penalty_accum`` 应在吸收历史时按日去重 + cap 封顶累计。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D_NDW")
        self.mem.rules = ParsedRules()
        self.mem.rules.no_drive_windows.append(_noon_window())

    def test_take_order_crossing_noon_records_one_violation(self) -> None:
        """穿越 12-13 的接单应记 1 次违规、累计 ¥100。"""
        # D=2026-03-01: 11:30 (690) → 12:30 (750)
        rec = _make_take_order_record(step=1, sim_end_minutes=750, exec_minutes=60)
        self.mem.absorb_history_records([rec])
        rule_id = "nodrive_720"
        self.assertIn("2026-03-01", self.mem.preference_violation_days[rule_id])
        self.assertAlmostEqual(self.mem.preference_penalty_accum[rule_id], 100.0)

    def test_same_day_repeat_does_not_double_count(self) -> None:
        """同一日两次穿越窗口应只累计一次（评测器同日只罚 1 次）。"""
        # 第一次：11:30→12:30
        self.mem.absorb_history_records(
            [_make_take_order_record(1, 750, 60)]
        )
        # 第二次同日：12:45→12:55
        self.mem.absorb_history_records(
            [_make_take_order_record(2, 775, 10)]
        )
        rule_id = "nodrive_720"
        self.assertEqual(len(self.mem.preference_violation_days[rule_id]), 1)
        self.assertAlmostEqual(self.mem.preference_penalty_accum[rule_id], 100.0)

    def test_distinct_days_accumulate_and_respect_cap(self) -> None:
        """30 个不同日的违规累计应封顶在 cap=3000。"""
        rule_id = "nodrive_720"
        # 模拟 35 个不同日，每日一次 11:30→12:30 接单；cap=3000 应止于 30 天的等量值。
        for i in range(35):
            sim_end = 750 + i * 1440  # 每日同时段
            self.mem.absorb_history_records([_make_take_order_record(i + 1, sim_end, 60)])
        # 35 个不同日均被记
        self.assertEqual(len(self.mem.preference_violation_days[rule_id]), 35)
        # 但累计在 cap 处饱和
        self.assertAlmostEqual(self.mem.preference_penalty_accum[rule_id], 3000.0)

    def test_reposition_crossing_window_also_accumulates(self) -> None:
        """reposition 穿越窗口同样应记入。"""
        # 11:50→12:10
        self.mem.absorb_history_records([_make_reposition_record(1, 730, 20)])
        rule_id = "nodrive_720"
        self.assertEqual(len(self.mem.preference_violation_days[rule_id]), 1)
        self.assertAlmostEqual(self.mem.preference_penalty_accum[rule_id], 100.0)

    def test_action_outside_window_does_not_accumulate(self) -> None:
        """完全不接触窗口的动作不应记违规。"""
        # 09:00→10:00（远离 12-13 窗口）
        self.mem.absorb_history_records([_make_take_order_record(1, 600, 60)])
        rule_id = "nodrive_720"
        self.assertNotIn(rule_id, self.mem.preference_penalty_accum)
        self.assertEqual(self.mem.preference_violation_days.get(rule_id, set()), set())

    def test_failed_take_order_does_not_accumulate(self) -> None:
        """take_order 未 accepted 不应记违规（仿真里只有成功才算违规活动）。"""
        rec = _make_take_order_record(1, 750, 60)
        rec["result"]["accepted"] = False
        self.mem.absorb_history_records([rec])
        rule_id = "nodrive_720"
        self.assertNotIn(rule_id, self.mem.preference_penalty_accum)


class SoftNoDriveMarginalChargeTest(unittest.TestCase):
    """``_soft_no_drive_marginal_charge`` 必须正确感知 cap 与同日去重。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D_CHARGE")
        self.window = _noon_window()

    def test_first_violation_charges_full_penalty(self) -> None:
        """无累计、未饱和、新一天 → 全额 ¥100。"""
        charge = scoring._soft_no_drive_marginal_charge(
            self.window, action_start_minutes=690, action_end_minutes=750, memory=self.mem
        )
        self.assertAlmostEqual(charge, 100.0)

    def test_same_day_repeat_charge_zero(self) -> None:
        """同日已被记一次 → 边际 0（评测器同日只罚 1 次）。"""
        rule_id = "nodrive_720"
        self.mem.preference_violation_days[rule_id].add("2026-03-01")
        self.mem.preference_penalty_accum[rule_id] = 100.0
        charge = scoring._soft_no_drive_marginal_charge(
            self.window, 690, 750, self.mem
        )
        self.assertAlmostEqual(charge, 0.0)

    def test_cap_saturated_charge_zero(self) -> None:
        """月度 cap ≥95% 已用 → 边际 0，避免月底过度回避盈利违规订单。"""
        rule_id = "nodrive_720"
        self.mem.preference_penalty_accum[rule_id] = 2900.0  # cap=3000, 96.7%
        charge = scoring._soft_no_drive_marginal_charge(
            self.window, 690, 750, self.mem
        )
        self.assertAlmostEqual(charge, 0.0)

    def test_cap_remaining_clamps_charge(self) -> None:
        """cap 剩余不足 ¥100 时按剩余空间收费。"""
        rule_id = "nodrive_720"
        self.mem.preference_penalty_accum[rule_id] = 2840.0  # 94.7%, 剩 ¥160
        # 10 天均触发，但 cap 限制
        # 用 10 个不同日：但函数只看 action 跨越的日。这里跨 1 天 → ¥100 vs cap 剩 ¥160 → ¥100
        charge = scoring._soft_no_drive_marginal_charge(
            self.window, 690, 750, self.mem
        )
        self.assertAlmostEqual(charge, 100.0)
        # 现在做一个跨 3 天的极端动作（理论），cap 剩 ¥160 → 限制为 ¥160
        action_start = 690  # D 11:30
        action_end = 690 + 2 * 1440 + 60  # 跨 3 天，每天都触碰中午窗口
        # 跨多天的动作罕见，但需断言 clamp 行为
        charge_multi = scoring._soft_no_drive_marginal_charge(
            self.window, action_start, action_end, self.mem
        )
        self.assertLessEqual(charge_multi, 160.0 + 1e-6)

    def test_no_cap_uses_full_penalty(self) -> None:
        """penalty_cap=None（少见但合法）时不做 cap 限制。"""
        window = _noon_window(cap=None)
        charge = scoring._soft_no_drive_marginal_charge(
            window, 690, 750, self.mem
        )
        self.assertAlmostEqual(charge, 100.0)

    def test_action_not_overlapping_window_charges_zero(self) -> None:
        """动作不与窗口重叠 → 边际 0。"""
        # 09:00 → 10:00
        charge = scoring._soft_no_drive_marginal_charge(
            self.window, 540, 600, self.mem
        )
        self.assertAlmostEqual(charge, 0.0)


class EndToEndSoftWindowTest(unittest.TestCase):
    """端到端：吸收 30 天违规历史 → cap 接近饱和 → 第 31 次评分边际 0。"""

    def test_thirty_days_saturate_then_marginal_zero(self) -> None:
        driver_memory.reset()
        mem = driver_memory.get_or_create("D004_SIM")
        mem.rules = ParsedRules()
        mem.rules.no_drive_windows.append(_noon_window())
        for i in range(30):
            sim_end = 750 + i * 1440
            mem.absorb_history_records([_make_take_order_record(i + 1, sim_end, 60)])
        # 30 天已 ¥3,000 = cap → 第 31 天评分应得边际 0
        rule_id = "nodrive_720"
        self.assertAlmostEqual(mem.preference_penalty_accum[rule_id], 3000.0)
        # 第 31 天 11:30→12:30 评分
        action_start = 690 + 30 * 1440
        action_end = action_start + 60
        charge = scoring._soft_no_drive_marginal_charge(
            mem.rules.no_drive_windows[0], action_start, action_end, mem
        )
        self.assertAlmostEqual(charge, 0.0)


class NearViolationWeightRegressionTest(unittest.TestCase):
    """D004 regression guard: soft no-drive cap must not globally triple preference risk."""

    def test_soft_no_drive_near_cap_does_not_trigger_global_preference_risk(self) -> None:
        driver_memory.reset()
        mem = driver_memory.get_or_create("D004_REGRESSION")
        rules = ParsedRules()
        rules.no_drive_windows.append(_noon_window())
        mem.preference_penalty_accum["nodrive_720"] = 2900.0
        self.assertFalse(scoring._is_preference_near_violation(rules, mem))

    def test_strict_no_drive_near_cap_still_triggers_global_preference_risk(self) -> None:
        driver_memory.reset()
        mem = driver_memory.get_or_create("D_STRICT")
        rules = ParsedRules()
        rules.no_drive_windows.append(
            TimeWindowRule(
                start_minute=23 * 60,
                end_minute=(24 + 4) * 60,
                raw="strict night window",
                penalty_amount=500.0,
                penalty_cap=15000.0,
            )
        )
        mem.preference_penalty_accum["nodrive_1380"] = 13000.0
        self.assertTrue(scoring._is_preference_near_violation(rules, mem))


class WaitSoftWindowRegressionTest(unittest.TestCase):
    def _ctx_at(self, minutes: int) -> scoring.DecisionContext:
        return scoring.DecisionContext(
            driver_id="D004_WAIT",
            cost_per_km=1.5,
            truck_length="4.2m",
            current_lat=22.5,
            current_lng=114.0,
            current_minutes=minutes,
            horizon_minutes=31 * 1440,
        )

    def test_wait_avoid_gain_is_zero_when_soft_window_same_day_already_violated(self) -> None:
        driver_memory.reset()
        mem = driver_memory.get_or_create("D004_WAIT")
        rules = ParsedRules()
        rules.no_drive_windows.append(_noon_window())
        mem.preference_violation_days["nodrive_720"].add("2026-03-01")
        mem.preference_penalty_accum["nodrive_720"] = 100.0

        scored = scoring.score_wait(60, rules, mem, self._ctx_at(12 * 60), has_good_order=False)

        self.assertNotIn("no_drive_window_avoid_gain", scored.breakdown)

    def test_wait_avoid_gain_uses_remaining_cap_for_soft_window(self) -> None:
        driver_memory.reset()
        mem = driver_memory.get_or_create("D004_WAIT")
        rules = ParsedRules()
        rules.no_drive_windows.append(_noon_window(cap=500.0))
        mem.preference_penalty_accum["nodrive_720"] = 450.0

        scored = scoring.score_wait(60, rules, mem, self._ctx_at(12 * 60), has_good_order=False)

        self.assertAlmostEqual(scored.breakdown["no_drive_window_avoid_gain"], 50.0)


if __name__ == "__main__":
    unittest.main()
