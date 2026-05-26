"""PR#28 评分架构改进的单元测试。

覆盖范围：
1. ``config.PENALTY_DEFAULTS`` 与 ``scoring._penalty_default`` 的一致性与边界。
2. ``scoring._finalize_score`` 出口校验钩子的三层校验逻辑。
3. 模块接口的可观测性（breakdown drift 标记 / NaN-Inf 拦截标记）。

这些测试与具体司机数据集解耦，验证数据集无关层的稳定行为。
"""
from __future__ import annotations

import math
import os
import sys
import unittest

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent import config, scoring  # noqa: E402


class PenaltyDefaultsConsistencyTest(unittest.TestCase):
    """PENALTY_DEFAULTS 内容与 _penalty_default 行为一致性测试。"""

    def test_all_known_kinds_resolve_to_positive_float(self) -> None:
        """所有已注册的 kind 应解析为正浮点数。"""
        self.assertGreater(len(config.PENALTY_DEFAULTS), 0, "PENALTY_DEFAULTS 不应为空")
        for kind, expected in config.PENALTY_DEFAULTS.items():
            actual = scoring._penalty_default(kind)
            self.assertIsInstance(actual, float, f"{kind} 应返回 float")
            self.assertGreater(actual, 0.0, f"{kind} 兜底值应为正数（防御性）")
            self.assertEqual(actual, float(expected), f"{kind} 解析值应与配置一致")

    def test_unknown_kind_returns_zero_in_lenient_mode(self) -> None:
        """未注册的 kind 在非严格模式下退化为 0.0（不抛异常）。"""
        # 默认 config.SCORING_STRICT_VALIDATION 不存在 → 走 lenient 分支
        self.assertEqual(scoring._penalty_default("__definitely_not_registered__"), 0.0)

    def test_required_kinds_are_registered(self) -> None:
        """覆盖 scoring.py 中所有用到的 kind 键，确保配置与代码同步。"""
        required = {
            "distance_limit_haul_pickup",
            "distance_limit_monthly_deadhead",
            "forbidden_zone",
            "no_drive_window_default",
            "daily_order_limit",
            "first_order_rule",
            "home_rule",
            "preferred_cargo",
            "timed_stay_event",
            "rest_rule",
            "monthly_day_off",
            "must_visit",
        }
        missing = required - set(config.PENALTY_DEFAULTS.keys())
        self.assertFalse(missing, f"PENALTY_DEFAULTS 缺少必需键: {missing}")


class FinalizeScoreValidationTest(unittest.TestCase):
    """_finalize_score 出口校验三层逻辑测试。"""

    def _make(self, score: float, breakdown: dict[str, float], feasible: bool = True) -> scoring.ScoredAction:
        return scoring.ScoredAction(
            action="take_order",
            params={"cargo_id": "C_TEST"},
            score=score,
            feasible=feasible,
            breakdown=dict(breakdown),
        )

    # ---------- (1) NaN/Inf 拦截 ----------

    def test_nan_score_is_blocked(self) -> None:
        scored = self._make(float("nan"), {"income": 100.0})
        result = scoring._finalize_score(scored)
        self.assertFalse(result.feasible)
        self.assertEqual(result.score, -float(scoring.HARD_CONSTRAINT_PENALTY))
        self.assertIn("_finalize_nan_inf", result.breakdown)
        self.assertIn("nan_inf_blocked", result.note)

    def test_inf_score_is_blocked(self) -> None:
        scored = self._make(float("inf"), {"income": 100.0})
        result = scoring._finalize_score(scored)
        self.assertFalse(result.feasible)
        self.assertEqual(result.score, -float(scoring.HARD_CONSTRAINT_PENALTY))

    def test_nan_in_breakdown_is_blocked(self) -> None:
        scored = self._make(50.0, {"income": 100.0, "weird": float("nan")})
        result = scoring._finalize_score(scored)
        self.assertFalse(result.feasible)
        self.assertNotIn("weird", result.breakdown, "NaN 项应被清理")
        self.assertIn("_finalize_nan_inf", result.breakdown)

    # ---------- (2) breakdown 一致性 ----------

    def test_consistent_breakdown_no_drift_marker(self) -> None:
        scored = self._make(150.0, {"income": 200.0, "cost": -50.0})
        result = scoring._finalize_score(scored)
        self.assertNotIn("_finalize_drift", result.breakdown)
        self.assertEqual(result.score, 150.0)

    def test_inconsistent_breakdown_drift_recorded(self) -> None:
        # breakdown 总和 = 150，但 score 显式设为 200，drift = 50
        scored = self._make(200.0, {"income": 200.0, "cost": -50.0})
        result = scoring._finalize_score(scored)
        self.assertIn("_finalize_drift", result.breakdown)
        self.assertAlmostEqual(result.breakdown["_finalize_drift"], 50.0)
        # score 本身不被修复（保持原值，不破坏已有调优）
        self.assertEqual(result.score, 200.0)

    def test_micro_drift_within_tolerance_ignored(self) -> None:
        scored = self._make(150.0 + 1e-9, {"income": 200.0, "cost": -50.0})
        result = scoring._finalize_score(scored)
        self.assertNotIn("_finalize_drift", result.breakdown)

    def test_empty_breakdown_is_ok(self) -> None:
        scored = self._make(-99999.0, {}, feasible=False)
        result = scoring._finalize_score(scored)
        # 空 breakdown 跳过一致性检查，infeasible 已为负 → 不修复
        self.assertEqual(result.score, -99999.0)
        self.assertFalse(result.feasible)

    # ---------- (3) infeasible 评分必须为负 ----------

    def test_infeasible_with_positive_score_is_fixed(self) -> None:
        scored = self._make(100.0, {"income": 100.0}, feasible=False)
        result = scoring._finalize_score(scored)
        self.assertLess(result.score, 0)
        self.assertEqual(result.score, -float(scoring.HARD_CONSTRAINT_PENALTY))
        self.assertIn("infeasible_score_fixed", result.note)

    def test_feasible_positive_score_unchanged(self) -> None:
        scored = self._make(100.0, {"income": 100.0}, feasible=True)
        result = scoring._finalize_score(scored)
        self.assertEqual(result.score, 100.0)
        self.assertTrue(result.feasible)
        self.assertNotIn("infeasible_score_fixed", result.note)

    # ---------- (4) 链式调用语义 ----------

    def test_returns_same_instance(self) -> None:
        scored = self._make(100.0, {"income": 100.0})
        result = scoring._finalize_score(scored)
        self.assertIs(result, scored, "_finalize_score 应就地修改并返回同一对象（链式语义）")


class FinalizeScoreStrictModeTest(unittest.TestCase):
    """严格模式下 _finalize_score 抛 AssertionError。"""

    def setUp(self) -> None:
        # 临时打开严格模式
        self._original = scoring._STRICT_VALIDATION
        scoring._STRICT_VALIDATION = True

    def tearDown(self) -> None:
        scoring._STRICT_VALIDATION = self._original

    def test_strict_raises_on_nan(self) -> None:
        scored = scoring.ScoredAction(
            action="wait", params={}, score=float("nan"), breakdown={"income": 100.0}
        )
        with self.assertRaises(AssertionError):
            scoring._finalize_score(scored)

    def test_strict_raises_on_drift(self) -> None:
        scored = scoring.ScoredAction(
            action="wait", params={}, score=200.0, breakdown={"income": 100.0}
        )
        with self.assertRaises(AssertionError):
            scoring._finalize_score(scored)

    def test_strict_raises_on_infeasible_positive(self) -> None:
        scored = scoring.ScoredAction(
            action="wait", params={}, score=100.0, feasible=False, breakdown={"income": 100.0}
        )
        with self.assertRaises(AssertionError):
            scoring._finalize_score(scored)


if __name__ == "__main__":
    unittest.main()
