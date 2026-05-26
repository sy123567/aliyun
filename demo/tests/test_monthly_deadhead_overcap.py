"""PR#29 单元测试：MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF 行为验证。

回归源：PR#27 把 OVERCAP_RESIDUAL_COEFF 设为 0.5，让 D003 在 cap 饱和后仍因高 pickup_dh 被强罚，
错过高价长单，毛收入 -¥10,727（净 -¥4,563）。PR#29 把系数降到 0.1，恢复经济最优决策。
"""
from __future__ import annotations

import os
import sys
import unittest

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent import config, driver_memory, scoring  # noqa: E402
from agent.preference_parser import DistanceLimitRule, ParsedRules  # noqa: E402


def _d003_rules(per_km: float = 10.0, cap: float = 2000.0, max_km: float = 100.0) -> ParsedRules:
    rules = ParsedRules()
    rules.distance_limits.append(
        DistanceLimitRule(kind="monthly_deadhead", max_km=max_km, penalty_amount=per_km, penalty_cap=cap)
    )
    return rules


def _ctx(minutes: int = 14400) -> scoring.DecisionContext:
    """ctx at 3/10 00:00, far from city."""
    return scoring.DecisionContext(
        driver_id="D003_OVERCAP",
        cost_per_km=1.5,
        truck_length="4.2m",
        current_lat=23.5,
        current_lng=113.5,
        current_minutes=minutes,
        horizon_minutes=31 * 1440,
    )


def _cargo_item(pickup_dh_km: float, price: float = 50000.0, haul_km: float = 150.0) -> dict:
    # 让 start 与 ctx pos (23.5, 113.5) 一致以保证 pickup_dh_km=0 时 haversine 也是 0。
    start_lat, start_lng = 23.5, 113.5
    if pickup_dh_km > 0:
        start_lat += pickup_dh_km / 111.0  # ~1 度 ≈ 111km
    return {
        "distance_km": pickup_dh_km,
        "cargo": {
            "cargo_id": "C_TEST",
            "start": {"lat": start_lat, "lng": start_lng},
            "end": {"lat": start_lat + haul_km / 111.0, "lng": start_lng},
            "price": price,
            "cost_time_minutes": 200,
            "cargo_name": "普通货物",
        },
    }


class MonthlyDeadheadOvercapTest(unittest.TestCase):
    """OVERCAP_RESIDUAL_COEFF=0.1 让 cap 饱和后接长 pickup_dh 高价单仍可行。"""

    def setUp(self) -> None:
        driver_memory.reset()
        self.mem = driver_memory.get_or_create("D003_OVERCAP")
        # 模拟 D003 已达 cap 饱和（total_deadhead_km 远超 max_km）
        self.mem.total_deadhead_km = 500.0  # max_km=100，已超 400km，cap 早已饱和
        self.rules = _d003_rules()

    def test_overcap_residual_is_small_after_pr29(self) -> None:
        """PR#29 验证：OVERCAP_RESIDUAL_COEFF 应被调小（≤0.2）。"""
        self.assertLessEqual(config.MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF, 0.2,
                             "PR#29: 系数过高会让 D003 cap 饱和后过度回避高 pickup_dh 的高价单")

    def test_high_price_long_pickup_dh_is_feasible(self) -> None:
        """高价单（¥50k）+ 长 pickup_dh（50km）+ cap 饱和：经过 OVERCAP 残余罚后仍应可行。"""
        item = _cargo_item(pickup_dh_km=50.0, price=50000.0)
        scored = scoring.score_take_order(item, self.rules, self.mem, _ctx())
        self.assertTrue(scored.feasible, f"高价长 dh 单应可行，note={scored.note}")
        # OVERCAP 残余罚分 = 10 × 50 × 0.1 = ¥50（PR#29 后），远小于 PR#27 的 ¥250
        residual_penalty = abs(scored.breakdown.get("distance_limit_penalty", 0.0))
        self.assertLessEqual(residual_penalty, 100.0,
                             f"PR#29 后 50km pickup_dh 的残余罚应 ≤¥100，实际 ¥{residual_penalty:.0f}")

    def test_zero_residual_when_no_new_overage(self) -> None:
        """pickup_dh=0（已在装货点）：不应增加任何 monthly_deadhead 罚分。"""
        item = _cargo_item(pickup_dh_km=0.0, price=50000.0)
        scored = scoring.score_take_order(item, self.rules, self.mem, _ctx())
        self.assertNotIn("distance_limit_penalty", scored.breakdown)

    def test_preempt_band_still_active(self) -> None:
        """预警区间 [50, 100] km 仍生效（PR#29 不变此行为）。"""
        self.mem.total_deadhead_km = 60.0  # 在预警区间
        # pickup_dh 10km → new_total=70，仍在预警区间
        item = _cargo_item(pickup_dh_km=10.0, price=50000.0)
        scored = scoring.score_take_order(item, self.rules, self.mem, _ctx())
        # 预警区间应有 some penalty（preempt logic）
        self.assertIn("distance_limit_penalty", scored.breakdown)
        self.assertTrue(scored.feasible, "预警区间应只是软罚，不阻拦")

    def test_residual_coeff_change_only_affects_overcap(self) -> None:
        """直接验证 OVERCAP 残余罚公式：penalty = per_km × over_km × COEFF。"""
        self.mem.total_deadhead_km = 500.0  # cap 已 100×10=1000 满，乘 0.95 阈值已过
        # pickup_dh 100km → over_km +100
        item = _cargo_item(pickup_dh_km=100.0, price=200000.0)
        scored = scoring.score_take_order(item, self.rules, self.mem, _ctx())
        residual = abs(scored.breakdown.get("distance_limit_penalty", 0.0))
        expected = 10.0 * 100.0 * config.MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF
        # PR#29: 0.1 系数 → ¥100；PR#27: 0.5 系数 → ¥500
        self.assertAlmostEqual(residual, expected, places=1,
                               msg=f"OVERCAP 残余 ¥{residual:.1f} 应 = ¥{expected:.1f}")


class PR27CapResidualRegressionTest(unittest.TestCase):
    """回归预防：避免未来把 COEFF 改回 ≥0.3（D003 会再次退化 -¥4,563）。"""

    def test_coeff_documented_lower_bound(self) -> None:
        """提醒后续维护：勿擅自调高 OVERCAP 系数，回归测评已证 0.5 损失 D003 -¥4,563。"""
        # 这是 sentinel 测试 - 若任何人改高了系数，CI 失败强制走 review
        self.assertLessEqual(config.MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF, 0.2)


if __name__ == "__main__":
    unittest.main()
