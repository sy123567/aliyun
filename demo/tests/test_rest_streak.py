"""Unit tests for the rest-streak day-segmentation fix.

根因：原 `_extend_rest_streak` 把整段 wait 计入结束日，跨午夜时让 agent
误以为 D+1 已经获得整段连续休息，导致 D002/D005/D006 等司机大量
"每日连续休息≥N 小时"违规（详见 monthly_income_202603.json）。

修复后：跨午夜 wait 必须分别按日计入"日内最长连续段"。
"""
from __future__ import annotations

import os
import sys
import unittest

# 让 ``from agent import ...`` 在不同 CWD 下都能解析
_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent import driver_memory  # noqa: E402
from agent import geo_utils  # noqa: E402


def _make_wait_record(step: int, sim_end_minutes: int, duration_minutes: int) -> dict:
    """构造一条 wait 历史记录，符合 ``_absorb_single_record`` 期望的字段。"""
    return {
        "step": step,
        "action": {"action": "wait", "params": {"duration_minutes": duration_minutes}},
        "simulation_end_time_minutes": sim_end_minutes,
        "step_elapsed_minutes": duration_minutes,
        "action_exec_cost_minutes": duration_minutes,
        "result": {"simulation_progress_minutes": sim_end_minutes},
    }


def _make_take_order_record(step: int, sim_end_minutes: int, exec_minutes: int) -> dict:
    return {
        "step": step,
        "action": {"action": "take_order", "params": {"cargo_id": "X"}},
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


class RestStreakSegmentationTest(unittest.TestCase):
    """覆盖跨午夜、同日续接、同日打断三大场景。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D_TEST")

    def test_wait_crossing_midnight_is_split_per_day(self) -> None:
        """22:00→04:00 的 6h wait：D 日记 120min，D+1 日记 240min。

        之前的 bug 会把全部 360min 都记入 D+1，让 agent 误判 D+1 已经
        休息够 5h。
        """
        # 仿真起点：2026-03-01 00:00
        # wait 从 D=2026-03-01 的 22:00 (1320 min) 开始，结束 D+1 04:00 (1680 min)
        record = _make_wait_record(step=1, sim_end_minutes=1680, duration_minutes=360)
        self.mem.absorb_history_records([record])

        # D 日 (2026-03-01) 应有 120 min（22:00-24:00）
        self.assertEqual(self.mem.longest_rest_today(1320), 120)
        # D+1 日 (2026-03-02) 应有 240 min（00:00-04:00），不是 360
        self.assertEqual(self.mem.longest_rest_today(1680), 240)

    def test_same_day_consecutive_waits_concatenate(self) -> None:
        """同日两段连续 wait 应合并为一个连续段（例：20:00→22:00 + 22:00→23:00 = 180min）。"""
        # 第一段：D=2026-03-01 20:00 (1200) → 22:00 (1320)，120 min
        self.mem.absorb_history_records([_make_wait_record(1, 1320, 120)])
        # 第二段：22:00 → 23:00 (1380)，60 min
        self.mem.absorb_history_records([_make_wait_record(2, 1380, 60)])
        # 应合并：单日最长连续段 = 180 min
        self.assertEqual(self.mem.longest_rest_today(1380), 180)

    def test_take_order_between_waits_breaks_streak(self) -> None:
        """同日 wait→take_order→wait 应该断开连续段，仅保留较长的那一段。"""
        # wait 60min 结束于 18:00 (1080)
        self.mem.absorb_history_records([_make_wait_record(1, 1080, 60)])
        # take_order 60min（打断 streak），结束于 19:00 (1140)
        self.mem.absorb_history_records([_make_take_order_record(2, 1140, 60)])
        # 第二段 wait 90min，结束于 20:30 (1230)
        self.mem.absorb_history_records([_make_wait_record(3, 1230, 90)])
        # 当日最长连续 = 90（不是 60+90）
        self.assertEqual(self.mem.longest_rest_today(1230), 90)

    def test_overnight_wait_then_morning_take_then_evening_wait(self) -> None:
        """端到端：跨夜 6h（D 120 / D+1 240）+ D+1 早上接单 + D+1 晚上 5h。

        D+1 最终应记最长 = 300min（晚上的连续段），而不是 240（早晨被打断的部分）
        也不是 540（错误的累加）。
        """
        # 跨夜 wait：D 22:00 → D+1 04:00，6h
        self.mem.absorb_history_records([_make_wait_record(1, 1680, 360)])
        # D+1 早 take_order：04:00 → 05:00，60 min（打断 streak）
        self.mem.absorb_history_records([_make_take_order_record(2, 1740, 60)])
        # D+1 晚 wait：17:00 → 22:00 (2460→2760)，300 min
        self.mem.absorb_history_records([_make_wait_record(3, 2760, 300)])
        # D+1 最长连续段 = 300（早上 240 被 take_order 切断，与晚上不连）
        self.assertEqual(self.mem.longest_rest_today(2760), 300)
        # D 仍应为 120
        self.assertEqual(self.mem.longest_rest_today(1320), 120)

    def test_zero_duration_wait_is_noop(self) -> None:
        """duration=0 不应改变任何状态。"""
        self.mem.absorb_history_records([_make_wait_record(1, 1000, 0)])
        self.assertEqual(self.mem.longest_rest_today(1000), 0)

    def test_three_day_spanning_wait(self) -> None:
        """极长 wait（48h+）跨多天：每日内段应单独计 1440min（理论极限）。

        模拟 D010 家事 stay 阶段：D 09:00 起 wait 60h（3600 min）→ D+2 21:00 结束。
        各日：D 段 = 1440-540 = 900 min（09:00-24:00）；D+1 = 1440；D+2 = 1260（00:00-21:00）。
        """
        # 起：D 09:00 (540)；终：540 + 3600 = 4140
        self.mem.absorb_history_records([_make_wait_record(1, 4140, 3600)])
        self.assertEqual(self.mem.longest_rest_today(540), 900)
        self.assertEqual(self.mem.longest_rest_today(540 + 1440), 1440)
        self.assertEqual(self.mem.longest_rest_today(4140), 1260)


class RestStreakLegacyApiTest(unittest.TestCase):
    """旧版 `_extend_rest_streak(date, duration)` 仍可被外部调用（兼容性保留）。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D_LEGACY")

    def test_legacy_extend_still_records(self) -> None:
        self.mem._extend_rest_streak("2026-03-05", 240)
        self.assertEqual(self.mem.daily_longest_rest_minutes.get("2026-03-05"), 240)


if __name__ == "__main__":
    unittest.main()
