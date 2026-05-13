"""DP-ORH-MS 全局参数中心。

所有评分权重、阈值、Token 预算与候选规模都在此集中，便于消融与调参。
其它模块通过 ``from . import config`` 引用，避免在算法代码中散落魔数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace


# ---------------- 仿真 horizon（agent 端假设） ----------------

def _resolve_agent_horizon_days() -> int:
    """读取智能体假设的仿真天数。

    平台正式评测固定为 31 天，``AGENT_HORIZON_DAYS`` 环境变量仅供本地缩短日数快速调试。
    """
    raw = os.environ.get("AGENT_HORIZON_DAYS", "").strip()
    if not raw:
        return 31
    try:
        days = int(raw)
        return max(1, days)
    except ValueError:
        return 31


AGENT_HORIZON_DAYS = _resolve_agent_horizon_days()
"""智能体内部假设的仿真总天数；默认 31 天。本地测试可通过环境变量 ``AGENT_HORIZON_DAYS=1`` 等覆盖。"""

AGENT_HORIZON_MINUTES = AGENT_HORIZON_DAYS * 24 * 60
"""智能体假设的仿真上界（分钟），用于 score_take_order 的 income_eligible 预判。"""


# ---------------- Token 预算（复赛资源约束） ----------------

PER_DRIVER_TOKEN_LIMIT = 5_000_000
"""每位司机 token 使用上限，复赛公告值。"""

TOKEN_DEGRADE_RATIO = 0.8
"""token 累计达到上限的该比例后，强制降级为纯规则模式。"""

TOKEN_DEGRADE_THRESHOLD = int(PER_DRIVER_TOKEN_LIMIT * TOKEN_DEGRADE_RATIO)


# ---------------- 候选规模 ----------------

TOP_ORDER_CANDIDATES = 100
"""每步进入评分的接单候选数量上限。"""

TOP_REPOSITION_TARGETS = 6
"""每步生成的空驶候选目标点数量上限。"""

HISTORY_LOOKBACK_STEPS = 64
"""每步向 ``query_decision_history`` 索取的近 N 步。"""

MIN_WAIT_FALLBACK_MINUTES = 30
"""一切候选都不可行时的安全休息时长。"""


# ---------------- 时空与成本基准 ----------------

DEFAULT_REPOSITION_SPEED_KMH = 60.0
"""与仿真器一致的空驶速度（仅用于本地估算，真正速度以接口为准）。"""

DEFAULT_OPPORTUNITY_COST_PER_MINUTE = 0.5
"""时间机会成本基准：元/分钟，约合 30 元/小时。"""

HORIZON_OVERFLOW_PENALTY = 5_000.0
"""候选完成时间超过仿真上界时的惩罚分。"""

HARD_CONSTRAINT_PENALTY = 1e9
"""硬约束违规分数：用于事实上的过滤。"""


# ---------------- 自适应权重触发阈值 ----------------

MONTH_END_REMAINING_DAYS = 3
"""剩余仿真天数 ≤ 该值时进入“月末模式”。"""

SCARCE_CARGO_THRESHOLD = 5
"""可见货源条数低于该值时进入“稀缺模式”。"""

NIGHT_HOUR_START = 22
NIGHT_HOUR_END = 6
"""夜间时段（含跨午夜），用于降低时间成本权重以鼓励夜间休息。"""

PREF_RISK_NEAR_VIOLATION_RATIO = 0.85
"""偏好风险即将累计到 penalty_cap 的该比例后视为“即将违规”。"""


# ---------------- 权重模型 ----------------


@dataclass(frozen=True)
class ScoringWeights:
    """各评分维度的乘数因子。

    所有打分项原始单位都按“元”估算，权重用于做相对放大/缩小。
    """

    income: float = 1.0
    distance_cost: float = 1.0
    time_cost: float = 1.0
    pickup_deadhead: float = 1.0
    waiting: float = 1.0
    preference_risk: float = 1.0
    horizon_risk: float = 1.0
    future_value: float = 1.0
    reposition_gain: float = 1.0

    def scaled(self, **kwargs: float) -> "ScoringWeights":
        """返回按字段乘以指定倍率后的新权重对象。"""
        update: dict[str, float] = {}
        for field_name, ratio in kwargs.items():
            current = getattr(self, field_name)
            update[field_name] = current * float(ratio)
        return replace(self, **update)


DEFAULT_WEIGHTS = ScoringWeights()


# ---------------- 偏好解析参数 ----------------

PREFERENCE_PARSE_RETRY_LIMIT = 1
"""LLM 偏好解析失败时的重试次数。"""

PREFERENCE_DEFAULT_AVOID_CATEGORIES: tuple[str, ...] = ()
"""LLM 完全不可用时的兜底品类避免清单，留空表示不主动规避。"""

PREFERENCE_LLM_MAX_INPUT_CHARS = 4000
"""单次 LLM 偏好解析输入字符上限，避免过长 token 占用。"""


# ---------------- 失败学习 / 反停滞调优（v2：低分根因修复） ----------------

CARGO_SUCCESS_RATE_MIN_ATTEMPTS = 4
"""至少观察 N 次 take_order 尝试后再用真实成功率折扣，避免初期偶然失败放大悲观。"""

CARGO_SUCCESS_RATE_FLOOR = 0.4
"""成功率折扣下限：即使长期失败也保留 40% 的预期收入，避免完全放弃接单。"""

CARGO_FAILURE_ATTEMPT_COST_YUAN = 80.0
"""单次 take_order 失败的固定隐性成本（10 分钟扫描 + 1 分钟尝试 + 重试机会损失）。"""

STAGNATION_WAIT_THRESHOLD = 2
"""连续 wait 次数 > 该阈值后开始增长惩罚，迫使智能体尝试 reposition。"""

STAGNATION_WAIT_PENALTY_PER_STEP = 120.0
"""每多一次连续 wait 增加的额外惩罚（元）；与休息收益对冲。"""

STAGNATION_FAIL_PENALTY_PER_STEP = 60.0
"""每次连续 take_order 失败给后续 take_order 候选增加的额外悲观折扣。"""

PICKUP_DEADHEAD_SOFT_THRESHOLD_KM = 20.0
"""接单空驶距离软惩罚起点；超过则按距离差线性加罚，覆盖 80% 司机偏好上限。"""

PICKUP_DEADHEAD_HARD_THRESHOLD_KM = 50.0
"""接单空驶距离硬惩罚起点（原值），与软罚共同形成两段式罚函数。"""

PICKUP_DEADHEAD_SOFT_COEFF = 0.5
"""软空驶罚相对于行驶成本的折算系数（小于 1 避免与已有 distance_cost 重复计）。"""

HORIZON_OVERFLOW_INCOME_VOIDED = True
"""完工时间超过仿真上界时，是否清零收入项（与 ``income_eligible=false`` 仿真行为一致）。"""

PREFERRED_CARGO_APPROACH_WINDOW_MINUTES = 8 * 60

PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES = 6 * 60
"""熟货上架时间过后 N 分钟内仍未接到，则视为放弃，停止远距离追逐避免里程浪费。"""

PREFERRED_CARGO_BONUS_MULTIPLIER = 1.2
PREFERRED_CARGO_PREPOSITION_WINDOW_MINUTES = 48 * 60
"""熟货提前预定位窗口：扩大到48小时确保高价值熟货有充分预定位时间。"""
PREFERRED_CARGO_ARRIVAL_BUFFER_MINUTES = 45
PREFERRED_CARGO_POSITION_GAIN_MULTIPLIER = 1.6
PREFERRED_CARGO_WAIT_GAIN_MULTIPLIER = 0.8
PREFERRED_CARGO_ACTIVE_BONUS_MULTIPLIER = 1.8
PREFERRED_CARGO_MAX_WAIT_MINUTES = 8 * 60

TIMED_EVENT_APPROACH_WINDOW_MINUTES = 2 * 60

TIMED_EVENT_PRE_LOCK_WINDOW_MINUTES = 24 * 60
"""事件开始前的提前对位窗口：在此窗内若接单后会被锁在远处，则视同违规并施加重罚。"""

TIMED_EVENT_PRE_LOCK_DISTANCE_KM = 80.0
"""提前对位窗内，订单完工点距离接人点超过此值即施加重罚。"""

TIMED_EVENT_FIXED_GAIN_MULTIPLIER = 1.2

TIMED_EVENT_STAY_CHUNK_MINUTES = 12 * 60
TIMED_EVENT_LONG_STAY_MAX_MINUTES = 4 * 24 * 60
TIMED_EVENT_APPROACH_GAIN_MULTIPLIER = 1.6
TIMED_EVENT_START_BUFFER_MINUTES = 30
TIMED_EVENT_PICKUP_OVERSTAY_MULTIPLIER = 2.0
HOME_RULE_PREP_WINDOW_MINUTES = 6 * 60
HOME_RULE_TARGET_GAIN_MULTIPLIER = 2.0
HOME_RULE_REACHABILITY_MULTIPLIER = 2.0
HOME_RULE_AWAY_WAIT_PENALTY_MULTIPLIER = 2.0
NO_DRIVE_ACTIVE_PENALTY_MULTIPLIER = 2.0
NO_DRIVE_SAFETY_BUFFER_MINUTES = 45
"""安全裕量：订单/空驶预计完成时间 + 该值若触碰禁行窗，则拒绝。防止仿真实际用时超估计。
从30提升至45分钟，覆盖更多估算误差。"""
