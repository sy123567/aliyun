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

EVALUATION_HORIZON_DAYS = min(AGENT_HORIZON_DAYS, 30)
"""收益评测对月度偏好只统计的天数窗口。"""

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

DEFAULT_OPPORTUNITY_COST_PER_MINUTE = 1.0
"""时间机会成本基准：元/分钟（参数搜索优化：0.5→1.0）。"""

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

CARGO_FAILURE_ATTEMPT_COST_YUAN = 40.0
"""单次 take_order 失败的固定隐性成本（参数搜索优化：80→40元）。"""

STAGNATION_WAIT_THRESHOLD = 6
"""连续 wait 次数 > 该阈值后开始增长惩罚（参数搜索优化：3→6）。"""

STAGNATION_WAIT_PENALTY_PER_STEP = 160.0
"""每多一次连续 wait 增加的额外惩罚（参数搜索优化：120→160元）。"""

STAGNATION_FAIL_PENALTY_PER_STEP = 60.0
"""每次连续 take_order 失败给后续 take_order 候选增加的额外悲观折扣。"""

PICKUP_DEADHEAD_SOFT_THRESHOLD_KM = 35.0
"""接单空驶距离软惩罚起点（参数搜索优化：20→35km）。"""

PICKUP_DEADHEAD_HARD_THRESHOLD_KM = 50.0
"""接单空驶距离硬惩罚起点（原值），与软罚共同形成两段式罚函数。"""

PICKUP_DEADHEAD_SOFT_COEFF = 1.0
"""软空驶罚相对于行驶成本的折算系数（参数搜索优化：0.5→1.0）。"""

HORIZON_OVERFLOW_INCOME_VOIDED = True
"""完工时间超过仿真上界时，是否清零收入项（与 ``income_eligible=false`` 仿真行为一致）。"""

PREFERRED_CARGO_APPROACH_WINDOW_MINUTES = 8 * 60

PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES = 6 * 60
"""熟货上架时间过后 N 分钟内仍未接到，则视为放弃，停止远距离追逐避免里程浪费。"""

PREFERRED_CARGO_BONUS_MULTIPLIER = 1.3
PREFERRED_CARGO_PREPOSITION_WINDOW_MINUTES = 48 * 60
"""熟货提前预定位窗口：扩大到48小时确保高价值熟货有充分预定位时间。"""
PREFERRED_CARGO_ARRIVAL_BUFFER_MINUTES = 45
PREFERRED_CARGO_POSITION_GAIN_MULTIPLIER = 1.6
PREFERRED_CARGO_WAIT_GAIN_MULTIPLIER = 0.8
PREFERRED_CARGO_ACTIVE_BONUS_MULTIPLIER = 1.8
PREFERRED_CARGO_MAX_WAIT_MINUTES = 8 * 60

TIMED_EVENT_APPROACH_WINDOW_MINUTES = 2 * 60

TIMED_EVENT_PRE_LOCK_WINDOW_MINUTES = 48 * 60
"""事件开始前的提前对位窗口：在此窗内若接单后会被锁在远处，则视同违规并施加重罚。
扩大到 48h 以确保 D010 等有多日家事约定的司机能提前对位。"""

TIMED_EVENT_EARLY_APPROACH_WINDOW_MINUTES = 72 * 60
"""事件开始前的早期趋近窗口：在此窗内给予软激励让司机向接人点靠拢。"""

TIMED_EVENT_PRE_LOCK_DISTANCE_KM = 80.0
"""提前对位窗内，订单完工点距离接人点超过此值即施加重罚。"""

TIMED_EVENT_FIXED_GAIN_MULTIPLIER = 1.2

TIMED_EVENT_STAY_CHUNK_MINUTES = 12 * 60
TIMED_EVENT_LONG_STAY_MAX_MINUTES = 4 * 24 * 60

TIMED_EVENT_HOME_MANDATORY_WINDOW_MINUTES = 24 * 60
"""事件开始前 N 分钟内，若司机不在家附近，硬阻拦所有 take_order，强制回家。"""

TIMED_EVENT_PRECOMPLETION_WINDOW_MINUTES = 48 * 60
"""事件开始前 N 分钟内，只允许接在事件开始前能完工的订单。"""

TIMED_EVENT_STAY_GAIN_MULTIPLIER = 5.0
"""stay 阶段 wait 增益倍数：确保司机留在家中不外出。（R5: 3.0→5.0）"""

TIMED_EVENT_SOFT_LIMIT_WINDOW_MINUTES = 48 * 60
"""事件开始前 N 分钟内，只允许接卸货点在家附近的订单（软限制窗口）。"""

TIMED_EVENT_SOFT_LIMIT_DISTANCE_KM = 200.0
"""软限制窗口内，订单卸货点距家超过此距离则施加重罚。"""

TIMED_EVENT_POST_STAY_BUFFER_MINUTES = 2 * 60
"""事件结束后 N 分钟内仍给予stay增益（避免事件刚结束就跑太远）。"""

TIMED_EVENT_APPROACH_GAIN_MULTIPLIER = 1.6
TIMED_EVENT_START_BUFFER_MINUTES = 30
TIMED_EVENT_PICKUP_OVERSTAY_MULTIPLIER = 2.0
HOME_RULE_PREP_WINDOW_MINUTES = 6 * 60
HOME_RULE_TARGET_GAIN_MULTIPLIER = 2.0
HOME_RULE_REACHABILITY_MULTIPLIER = 2.0
HOME_RULE_AWAY_WAIT_PENALTY_MULTIPLIER = 2.0

HOME_RULE_AFTERNOON_BLOCK_HOUR = 12
"""下午 N 点后且距家>200km 时开始阻拦新订单（参数搜索优化：14→12点）。"""

HOME_RULE_AFTERNOON_BLOCK_DISTANCE_KM = 200.0
"""下午阻拦距离阈值：超过此距离开始阻拦。"""
NO_DRIVE_ACTIVE_PENALTY_MULTIPLIER = 2.0
NO_DRIVE_SAFETY_BUFFER_MINUTES = 30
"""安全裕量：订单/空驶预计完成时间 + 该值若触碰禁行窗则拒绝（参数搜索优化：45→30min）。"""

HAUL_TIME_OVERESTIMATE_RATIO = 1.0
"""haversine 估算的 haul/pickup 时间比真实仿真低估约 10–15% 的事实虽然存在，
但实验表明：把该系数作用于 no_drive_window 检测时（即使 1.05×），会让 D003/D007 错失若干可安全完成的订单。
当前保持 1.0（不放大），仅靠 SAFETY_BUFFER + 软偏好阈值就足以让 D004 的中午软窗 + 整体罚分趋于最优。
留下常量是为后续可能场景（如长距离干线）按需再启用。"""

NO_DRIVE_SOFT_PENALTY_THRESHOLD = 120.0
"""禁行窗「软 / 严」划分阈值：单日罚金 ≤ 该值的 no_drive_window 走软偏好逻辑（仅接受软罚、不启用双重裕量、不硬拒）。
D004 「中午 12–13 不接」单日罚金 ¥100（上限 ¥3,000）是典型软偏好：硬拒该窗会让 agent 错失 30×单价¥1,000+ 的订单。
反之 D003 「凌晨 2–5 不接」¥200/天、D005/D007 夜禁均 ¥200+/天是严偏好，应硬拒。阈值 120 能区分 D004(100) 与 D003/D005(200)。"""


# ---------------- 月度休息日优化参数（第四/五/六轮） ----------------

MONTHLY_DAY_OFF_SPACING_COEFF = 0.8
"""月度休息日spacing惩罚系数（参数搜索优化：0.4→0.8）。"""

MONTHLY_DAY_OFF_MONTH_END_DAYS = 5
"""月末集中休息触发：剩余天数 ≤ 该值且deficit=1时，给予更高wait增益。"""

MONTHLY_DAY_OFF_URGENCY_THRESHOLD_EARLY = 0.08
"""R6-P1: 前半月urgency软罚分触发阈值（原0.1→0.08，更早开始抑制）。"""

MONTHLY_DAY_OFF_URGENCY_THRESHOLD_LATE = 0.06
"""R6-P1: 后半月urgency软罚分触发阈值（day≥20后更敏感）。"""

MONTHLY_DAY_OFF_LATE_MONTH_DAY = 20
"""R6-P1: 后半月起始日（0-indexed），此后使用更低的urgency阈值。"""

MONTHLY_DAY_OFF_FORCE_REST_DAY = 25
"""R6-P1: 强制休息检查日（0-indexed），day≥此值且deficit≥1时给极高wait增益。"""


# ---------------- PR#24 经验驱动优化参数 ----------------

INCOME_EFFICIENCY_MIN_ORDERS = 5
"""司机完成至少 N 单后才启用收入效率加成（避免样本不足导致偏差）。"""

INCOME_EFFICIENCY_BONUS_CAP = 80.0
"""收入效率加成上限（元），防止单因素过度影响评分。"""

HOTSPOT_DECAY_HALF_LIFE_MINUTES = 8 * 60
"""热点网格数据半衰期（分钟），越久远的观测权重越低。"""

REPOSITION_LONG_DISTANCE_THRESHOLD_KM = 150.0
"""非优先目标空驶超过此距离时施加额外时间成本惩罚。"""

REPOSITION_LONG_DISTANCE_PENALTY_MULTIPLIER = 0.5
"""远距离空驶额外时间成本乘数。"""

ANTI_STAGNATION_MILD_THRESHOLD = 3
"""轻度反停滞：连续 wait ≥ 此值且无明确收益时，对长 wait 施加温和惩罚。"""

ANTI_STAGNATION_MILD_PENALTY_PER_STEP = 30.0
"""轻度反停滞惩罚系数（远低于正式阈值 160，仅作为微调信号）。"""

NIGHT_WAIT_TO_DAWN_HOUR = 6
"""夜间等待候选的目标时刻（小时），用于覆盖无明确 no_drive_window 的司机。"""

MIN_NON_PRIORITY_REPOSITION_SLOTS = 2
"""空驶候选中至少保留 N 个非优先目标名额，确保市场探索多样性。"""


# ---------------- PR#27 经验驱动：失分精修参数 ----------------

MONTHLY_DEADHEAD_PREEMPT_RATIO = 0.5
"""月度空驶累计达到限额的该比例后开始施加渐增预警惩罚（PR#27: D003 提前预警起点）。"""

MONTHLY_DEADHEAD_PREEMPT_MAX_COEFF = 0.6
"""预警区间最高乘数：在 [PREEMPT_RATIO, 1.0] 区间内按线性比例从 0 → MAX_COEFF×pen_per_km 收取。"""

MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF = 0.5
"""月度空驶 cap 已耗尽后，仍对新增超额公里数施加 per_km × 该系数 的残余信号（PR#27: 防 D003 1700km 失控）。"""

SOFT_NODRIVE_DEFER_PROXIMITY_MINUTES = 90
"""当前时刻距离软禁行窗口起始 ≤ 此分钟数（且尚未进入窗口）时，软罚乘数随接近程度递增（PR#27: D004 中午延后判断）。"""

SOFT_NODRIVE_DEFER_INSIDE_MULTIPLIER = 2.5
"""若当前时刻已落入软窗内、且即将开始的动作可被推迟到窗口结束后，软罚乘数（PR#27）。"""

TIMED_EVENT_MULTIDAY_THRESHOLD_MINUTES = 24 * 60
"""事件 stay 时长 ≥ 此值视为多日 stay，触发更深的提前回家强制窗（PR#27: D010 家事约定）。"""

TIMED_EVENT_MULTIDAY_HOME_MANDATORY_WINDOW_MINUTES = 48 * 60
"""多日 stay 事件的提前强制回家窗口（PR#27: 由默认 24h 扩到 48h，确保 D010 提前两天回家）。"""

FIRST_ORDER_PROXIMITY_BOOST_WINDOW_MINUTES = 90
"""首单截止前 N 分钟内，若当天尚未接单则对 take_order 施加软增益（PR#27: D004 首单晚开工）。"""

FIRST_ORDER_PROXIMITY_BOOST_RATIO = 0.6
"""首单截止前软增益占规则单日罚金的比例上限（PR#27）。"""

FIRST_ORDER_WAIT_CROSS_MULTIPLIER = 1.0
"""wait 跨过 first_order 截止时的罚金系数（PR#27: 由 0.5 提高到 1.0，让 wait 与 take_order 边际匹配）。"""

DAILY_REST_RISK_EARLY_RATIO = 3.0
"""每日休息余量预警触发线：remaining_after < deficit × 该比例 时施加软罚（PR#27: 由 2.0→3.0 提前）。"""

DAILY_REST_RISK_TIGHT_RATIO = 1.5
"""每日休息余量紧迫触发线：remaining_after < deficit × 该比例 时施加强软罚（PR#27: 由 1.3→1.5）。"""
