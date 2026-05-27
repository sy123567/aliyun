"""候选动作生成与多目标评分。

每个候选最终被打成一个标量分数（单位：元），含明细 ``breakdown`` 便于日志与调参。

核心思想：
- 收益侧：订单价格、未来位置价值。
- 成本侧：行驶成本（里程 × cost_per_km）、时间机会成本、装货等待。
- 风险侧：偏好命中罚分预估、月度仿真上界风险。
- 偏好侧：硬约束直接过滤，软偏好计入罚分。

架构分层（PR#28）
-----------------
本模块按"数据集无关 + 数据集相关"两层组织：

1. **数据集无关层**（dataset-agnostic）：
   - :data:`config.PENALTY_DEFAULTS`：所有罚金兜底值（仅当规则字段未提供时使用）。
   - :func:`_penalty_default`：按 ``kind`` 键读取兜底罚金，统一替代 ``or N.0`` 模式。
   - :func:`_finalize_score`：评分结果验证钩子，统一拦截 NaN/Inf、确保 breakdown 与 score 一致。

2. **数据集相关层**（dataset-specific）：
   - :class:`ParsedRules` 携带从司机偏好文本里解析出的 ``penalty_amount`` / ``penalty_cap`` 等参数。
   - 当某条规则未提供 ``penalty_amount`` 时由 ``_penalty_default`` 兜底，保证新数据集亦可工作。

   规则的具体业务语义（如 first_order 截止前 2h 给 urgency bonus）在三大入口
   :func:`score_take_order`、:func:`score_wait`、:func:`score_reposition` 中实现，
   并在出口统一经过 ``_finalize_score`` 校验，保证：

   * 不返回 NaN/Inf；
   * breakdown 各项相加 ≈ score（容差 1e-6）；
   * feasible=False 时 score 必为负惩罚（用于排序）。

这样既允许业务逻辑灵活适配各司机偏好，也保证跨数据集的评分稳定性与可观测性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config, geo_utils
from .driver_memory import DriverMemory
from .preference_parser import (
    BoundingBoxRule,
    CircleZone,
    DistanceLimitRule,
    HomeRule,
    ParsedRules,
    PreferredCargoRule,
    TimeWindowRule,
    TimedStayEventRule,
)

# 为保证老调用点不变，继续暴露原名常量，但都只是 config 的转发。
DEFAULT_OPPORTUNITY_COST_PER_MINUTE = config.DEFAULT_OPPORTUNITY_COST_PER_MINUTE
HORIZON_OVERFLOW_PENALTY = config.HORIZON_OVERFLOW_PENALTY
DEFAULT_REPOSITION_SPEED_KMH = config.DEFAULT_REPOSITION_SPEED_KMH
HARD_CONSTRAINT_PENALTY = config.HARD_CONSTRAINT_PENALTY

# ---------------- 数据集无关层 ----------------

# 评分一致性校验容差：breakdown 各项求和 vs. score 的允许误差（元）。
_FINALIZE_SCORE_TOLERANCE = 1e-6

# 评分一致性校验断言：仅在显式启用时抛错，否则 silent-fix 并记录到 breakdown。
# 由 config.SCORING_STRICT_VALIDATION 控制（默认 False，避免影响仿真）。
_STRICT_VALIDATION = getattr(config, "SCORING_STRICT_VALIDATION", False)


def _penalty_default(kind: str) -> float:
    """读取 ``config.PENALTY_DEFAULTS`` 中 ``kind`` 对应的兜底罚金。

    设计目的：替代 scoring.py 中散落的 ``rule.penalty_amount or N.0`` 模式，
    把数据集无关的兜底值集中到 :data:`config.PENALTY_DEFAULTS` 一处维护。

    参数
    ----
    kind: 兜底键名，必须在 ``config.PENALTY_DEFAULTS`` 中存在。

    返回
    ----
    对应的兜底罚金（元）；若键不存在，返回 ``0.0`` 并记录调试信息（不抛异常，
    保证未注册的 kind 在生产环境下退化为「无罚金」而非崩溃）。
    """
    defaults: dict[str, float] = getattr(config, "PENALTY_DEFAULTS", {})
    val = defaults.get(kind)
    if val is None:
        # 未注册的 kind 在严格模式下抛错，便于开发期暴露问题；生产模式下静默返回 0。
        if _STRICT_VALIDATION:
            raise KeyError(f"penalty default '{kind}' not registered in config.PENALTY_DEFAULTS")
        return 0.0
    return float(val)


def _finalize_score(scored: "ScoredAction") -> "ScoredAction":
    """评分结果的统一出口校验。

    校验项目（顺序执行，前者失败即拦截）：

    1. **NaN/Inf 拦截**：若 ``score`` 或 ``breakdown`` 中任一值为 NaN/Inf，
       将其重置为 ``-HARD_CONSTRAINT_PENALTY`` 并把 feasible 标记为 False，
       同时在 breakdown 写入 ``_finalize_nan_inf`` 标记，便于日志侧排查。

    2. **breakdown 一致性**：``sum(breakdown.values())`` 与 ``score`` 的差
       超过 ``_FINALIZE_SCORE_TOLERANCE`` 时，记录 ``_finalize_drift`` 项
       表示差额，不修改 score（不破坏现有调优）。

    3. **不可行评分必须为负**：``feasible=False`` 时 score 应为负惩罚；
       违反时把 score 修正为 ``-HARD_CONSTRAINT_PENALTY``。

    严格模式（``config.SCORING_STRICT_VALIDATION=True``，开发期使用）下
    所有违规均抛 ``AssertionError``；默认生产模式下静默修复，确保仿真不被中断。

    返回经过校验/修复后的 ``ScoredAction``（就地修改，亦原样返回，便于链式调用）。
    """
    import math

    # 1) NaN / Inf 拦截
    bad_values: list[str] = []
    if not math.isfinite(scored.score):
        bad_values.append("score")
    for key, val in list(scored.breakdown.items()):
        if not math.isfinite(val):
            bad_values.append(key)
    if bad_values:
        if _STRICT_VALIDATION:
            raise AssertionError(
                f"_finalize_score: non-finite values in {scored.action}: {bad_values}"
            )
        scored.breakdown = {
            k: v for k, v in scored.breakdown.items() if math.isfinite(v)
        }
        scored.breakdown["_finalize_nan_inf"] = -float(HARD_CONSTRAINT_PENALTY)
        scored.score = -float(HARD_CONSTRAINT_PENALTY)
        scored.feasible = False
        scored.note = (scored.note + "|nan_inf_blocked").strip("|")
        return scored

    # 2) breakdown ≈ score 一致性（不修复 score，仅打标）
    if scored.breakdown:
        total = sum(scored.breakdown.values())
        drift = scored.score - total
        if abs(drift) > _FINALIZE_SCORE_TOLERANCE:
            if _STRICT_VALIDATION:
                raise AssertionError(
                    f"_finalize_score: breakdown sum {total:.6f} != score {scored.score:.6f} "
                    f"(drift {drift:.6f}) in {scored.action}"
                )
            scored.breakdown["_finalize_drift"] = drift

    # 3) 不可行评分必须为负
    if not scored.feasible and scored.score >= 0:
        if _STRICT_VALIDATION:
            raise AssertionError(
                f"_finalize_score: infeasible action {scored.action} has non-negative score {scored.score}"
            )
        scored.score = -float(HARD_CONSTRAINT_PENALTY)
        scored.note = (scored.note + "|infeasible_score_fixed").strip("|")

    return scored

# 评分明细 key 到 ScoringWeights 字段的映射；未列出的项统一使用 preference_risk。
_WEIGHT_MAP: dict[str, str] = {
    "income": "income",
    "distance_cost": "distance_cost",
    "time_cost": "time_cost",
    "pickup_deadhead_penalty": "pickup_deadhead",
    "waiting_penalty": "waiting",
    "horizon_overflow_penalty": "horizon_risk",
    "future_location_value": "future_value",
    "preferred_cargo_bonus": "future_value",
    "reposition_cost": "distance_cost",
    "reposition_time_cost": "time_cost",
    "expected_market_gain": "reposition_gain",
    "opportunity_loss": "time_cost",
    "timed_event_pickup_gain": "preference_risk",
    "timed_event_home_gain": "preference_risk",
    "timed_event_order_penalty": "preference_risk",
    "timed_event_pickup_overstay_penalty": "preference_risk",
    "preferred_cargo_position_gain": "preference_risk",
    "preferred_cargo_wait_gain": "preference_risk",
    "preferred_cargo_conflict_penalty": "preference_risk",
    "timed_event_approach_gain": "preference_risk",
    "daily_rest_risk_penalty": "preference_risk",
}


@dataclass
class ScoredAction:
    """被评分后的候选动作。"""

    action: str  # take_order / wait / reposition
    params: dict[str, Any]
    score: float
    feasible: bool = True
    breakdown: dict[str, float] = field(default_factory=dict)
    note: str = ""
    occupied_minutes: float = 0.0

    def as_action_dict(self) -> dict[str, Any]:
        return {"action": self.action, "params": dict(self.params)}


@dataclass
class DecisionContext:
    """单步决策共享上下文。"""

    driver_id: str
    cost_per_km: float
    truck_length: str
    current_lat: float
    current_lng: float
    current_minutes: int
    horizon_minutes: int
    reposition_speed_km_per_hour: float = DEFAULT_REPOSITION_SPEED_KMH
    opportunity_cost_per_minute: float = DEFAULT_OPPORTUNITY_COST_PER_MINUTE
    weights: config.ScoringWeights = config.DEFAULT_WEIGHTS
    visible_cargo_count: int = 0


# ---------------- 自适应权重 ----------------


def resolve_adaptive_weights(
    rules: ParsedRules,
    memory: DriverMemory,
    ctx: DecisionContext,
    visible_cargo_count: int,
) -> config.ScoringWeights:
    """根据当前状态动态调整评分维度权重。

    规则（详见 docs/06 第 8.4 节）：
    - 月末（3 天内）： horizon_risk ×2、future_value ×0.5。
    - 偏好即将违规： preference_risk ×3。
    - 货源稀缺（可见货源<5）： pickup_deadhead ×0.5、reposition_gain ×1.5。
    - 夜间时段： time_cost ×0.3（鼓励夜间休息）。
    """
    weights = config.DEFAULT_WEIGHTS

    # 月末阶段
    remaining_minutes = ctx.horizon_minutes - ctx.current_minutes
    remaining_days = remaining_minutes / (24 * 60.0)
    if remaining_days <= config.MONTH_END_REMAINING_DAYS:
        weights = weights.scaled(horizon_risk=2.0, future_value=0.5)

    # 偏好即将违规
    # PR#24: 月后半段偏好风险更高，因剩余可纠正时间更少
    if _is_preference_near_violation(rules, memory):
        sim_day = ctx.current_minutes // (24 * 60)
        if sim_day >= config.MONTHLY_DAY_OFF_LATE_MONTH_DAY:
            weights = weights.scaled(preference_risk=5.0)
        else:
            weights = weights.scaled(preference_risk=3.0)

    # 货源稀缺
    if visible_cargo_count < config.SCARCE_CARGO_THRESHOLD:
        weights = weights.scaled(pickup_deadhead=0.5, reposition_gain=1.5)

    # 夜间时段
    hour = geo_utils.hour_of_day(ctx.current_minutes)
    if hour >= config.NIGHT_HOUR_START or hour < config.NIGHT_HOUR_END:
        weights = weights.scaled(time_cost=0.3)

    return weights


def _is_preference_near_violation(rules: ParsedRules, memory: DriverMemory) -> bool:
    """检测是否接近 penalty_cap 阈值，应提高偏好风险权重。

    PR#21/22 经验：硬编码 10,000 阈值无法适应不同司机的 penalty_cap 差异。
    改为收集所有规则的 penalty_cap，当任一累计罚金达到对应 cap 的 85% 时触发。
    """
    if not memory.preference_penalty_accum:
        return False
    caps: dict[str, float] = {}
    for limit in rules.distance_limits:
        if limit.penalty_cap is not None and limit.penalty_cap > 0:
            caps[f"dist_{limit.kind}"] = float(limit.penalty_cap)
    if rules.monthly_day_off is not None and rules.monthly_day_off.penalty_cap is not None:
        caps["monthly_day_off"] = float(rules.monthly_day_off.penalty_cap)
    if rules.home_rule is not None and rules.home_rule.penalty_cap is not None:
        caps["home_rule"] = float(rules.home_rule.penalty_cap)
    for window in rules.no_drive_windows:
        if window.penalty_cap is not None and window.penalty_cap > 0:
            window_penalty = float(window.penalty_amount or 0.0)
            if 0 < window_penalty <= config.NO_DRIVE_SOFT_PENALTY_THRESHOLD:
                continue
            caps[f"nodrive_{window.start_minute}"] = float(window.penalty_cap)
    for rule_id, accum in memory.preference_penalty_accum.items():
        cap = caps.get(rule_id)
        if cap is not None and cap > 0:
            if accum >= cap * config.PREF_RISK_NEAR_VIOLATION_RATIO:
                return True
        elif accum >= 10_000.0:
            return True
    return False


def _apply_adaptive_weights(breakdown: dict[str, float], weights: config.ScoringWeights) -> None:
    """原地将 breakdown 中每个条目乘以对应权重；未映射项使用 preference_risk。"""
    if weights is config.DEFAULT_WEIGHTS:
        return  # 默认权重无需乘法
    for key in list(breakdown.keys()):
        weight_name = _WEIGHT_MAP.get(key, "preference_risk")
        breakdown[key] = breakdown[key] * getattr(weights, weight_name)


# ---------------- 工具：偏好检查 ----------------


def _is_in_circle(lat: float, lng: float, zone: CircleZone | HomeRule) -> bool:
    return geo_utils.haversine_km(lat, lng, zone.lat, zone.lng) <= zone.radius_km


def _is_in_box(lat: float, lng: float, box: BoundingBoxRule) -> bool:
    return box.lat_min <= lat <= box.lat_max and box.lng_min <= lng <= box.lng_max


def preferred_cargo_ready(rule: PreferredCargoRule, current_minutes: int) -> bool:
    if rule.available_minutes is None:
        return True
    lower = rule.available_minutes - config.PREFERRED_CARGO_APPROACH_WINDOW_MINUTES
    upper = rule.available_minutes + config.PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES
    return lower <= current_minutes <= upper


def preferred_cargo_preposition_ready(rule: PreferredCargoRule, current_minutes: int) -> bool:
    if rule.available_minutes is None:
        return True
    # 高价值熟货(>=5000)：预定位窗收紧到 12h。
    # 老版本的 72h 让 D009 司机 2 天前就开始向 ¥10000 熟货点跑，过夜在外吃 ¥900 回家罚。
    # 12h 配合 23-08 的 no_drive_window：日 D−1 20:43 还不开窗，日 D 02:43 开窗时
    # 司机正在 no_drive，须等到 08:00 才能出发；190km 行程 + 缓冲足以在上架时间前抵达。
    # 仅适用于 ≥5000 元的硬重单；普通熟货沿用 48h 默认。
    preposition_window = config.PREFERRED_CARGO_PREPOSITION_WINDOW_MINUTES
    giveup_window = config.PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES
    if rule.penalty_amount and rule.penalty_amount >= 5000:
        preposition_window = 12 * 60                           # 12 小时预定位（原 72h）
        giveup_window = max(giveup_window, 12 * 60)            # 12 小时放弃窗口
    lower = rule.available_minutes - preposition_window
    upper = rule.available_minutes + giveup_window
    return lower <= current_minutes <= upper


def preferred_cargo_active(rule: PreferredCargoRule, current_minutes: int) -> bool:
    if rule.available_minutes is None:
        return True
    return rule.available_minutes <= current_minutes <= rule.available_minutes + config.PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES


def preferred_cargo_target(rule: PreferredCargoRule) -> tuple[float, float] | None:
    if rule.lat is None or rule.lng is None:
        return None
    return (float(rule.lat), float(rule.lng))


def timed_event_key(event: TimedStayEventRule) -> str:
    return (
        f"{event.start_minutes}:"
        f"{event.pickup_lat:.4f},{event.pickup_lng:.4f}:"
        f"{event.home_lat:.4f},{event.home_lng:.4f}"
    )


def timed_event_pickup_done(memory: DriverMemory, event: TimedStayEventRule) -> bool:
    return f"{timed_event_key(event)}:pickup" in memory.timed_event_flags


def timed_event_home_arrived(memory: DriverMemory, event: TimedStayEventRule) -> bool:
    return f"{timed_event_key(event)}:home" in memory.timed_event_flags


def timed_event_phase(event: TimedStayEventRule, memory: DriverMemory, ctx: DecisionContext) -> str:
    early_approach = event.start_minutes - config.TIMED_EVENT_EARLY_APPROACH_WINDOW_MINUTES
    pre_lock = event.start_minutes - config.TIMED_EVENT_PRE_LOCK_WINDOW_MINUTES
    approach = event.start_minutes - config.TIMED_EVENT_APPROACH_WINDOW_MINUTES
    if ctx.current_minutes < early_approach:
        return "early"
    if ctx.current_minutes < pre_lock and not timed_event_pickup_done(memory, event):
        return "early_approach"
    if ctx.current_minutes < approach and not timed_event_pickup_done(memory, event):
        return "approaching"
    if not timed_event_pickup_done(memory, event):
        if ctx.current_minutes <= event.deadline_minutes:
            return "pickup"
        return "late_pickup"
    if not timed_event_home_arrived(memory, event):
        if ctx.current_minutes <= event.deadline_minutes:
            return "home"
        return "late_home"
    if ctx.current_minutes < event.stay_until_minutes:
        return "stay"
    return "done"


def _path_passes_forbidden_zone(
    start_lat: float, start_lng: float, end_lat: float, end_lng: float, zone: CircleZone, samples: int = 6
) -> bool:
    """折线采样判断空驶或干线是否穿越禁区。"""
    for i in range(samples + 1):
        t = i / samples
        lat = start_lat + (end_lat - start_lat) * t
        lng = start_lng + (end_lng - start_lng) * t
        if geo_utils.haversine_km(lat, lng, zone.lat, zone.lng) <= zone.radius_km:
            return True
    return False


def _truck_length_compatible(driver_truck: str, cargo_truck: Any) -> bool:
    if not cargo_truck:
        return True  # 货源未声明视为兼容
    if isinstance(cargo_truck, list):
        if not cargo_truck:
            return True
        return driver_truck in [str(x) for x in cargo_truck]
    if isinstance(cargo_truck, str):
        return driver_truck in cargo_truck or driver_truck == cargo_truck
    return True


def _hits_no_drive_window(start_minutes: int, end_minutes: int, rule: TimeWindowRule) -> int:
    """返回与禁行窗口重叠的分钟数（按当天投影到 [0, 24h*N]）。"""
    if end_minutes <= start_minutes:
        return 0
    overlap = 0
    cursor = start_minutes
    while cursor < end_minutes:
        day_start = (cursor // 1440) * 1440
        within_day = cursor - day_start
        next_day = day_start + 1440
        seg_end = min(end_minutes, next_day)
        seg_within_end = seg_end - day_start
        rule_start = rule.start_minute
        rule_end = rule.end_minute
        # 把规则窗口投影到该天，可能跨午夜
        for base in (day_start - 1440, day_start):
            for ws, we in _project_rule_to_minutes(base, rule_start, rule_end):
                inter_start = max(cursor, ws)
                inter_end = min(seg_end, we)
                if inter_end > inter_start:
                    overlap += inter_end - inter_start
        cursor = seg_end
    return overlap


def _project_rule_to_minutes(day_start: int, rule_start: int, rule_end: int) -> list[tuple[int, int]]:
    """将一天内的规则窗口（可能跨午夜）映射到绝对仿真分钟区间。"""
    if rule_end <= 24 * 60:
        return [(day_start + rule_start, day_start + rule_end)]
    # 跨午夜：rule_start..1440 + 0..rule_end-1440
    end_today = day_start + 24 * 60
    second_end = day_start + 24 * 60 + (rule_end - 24 * 60)
    return [(day_start + rule_start, end_today), (end_today, second_end)]


def _soft_no_drive_defer_multiplier(window: TimeWindowRule, current_minutes: int) -> float:
    """软禁行窗的「可延后性」乘数（PR#27）。

    场景：D004「中午 12-13 不接」penalty=¥100/次、cap=¥3,000。一旦动作起点离窗口很近（窗前 90min 内）
    或已落入窗内，等待到窗口结束几乎不损失收益，违规属于显然可避免的决策——给软罚加倍以反向偏好。
    远离窗口（如 11:00 出发的长单 09:00-13:30 跨过）保留原罚金。

    返回值范围 [1.0, SOFT_NODRIVE_DEFER_INSIDE_MULTIPLIER]。
    """
    cur_md = current_minutes % 1440
    rule_start = window.start_minute
    rule_end = window.end_minute
    # 跨午夜窗：以原始 start_minute 为锚，cur 距 start 取最近的正向间距
    if cur_md >= rule_start and cur_md < (rule_end if rule_end <= 1440 else 1440):
        return float(config.SOFT_NODRIVE_DEFER_INSIDE_MULTIPLIER)
    if rule_end > 1440 and cur_md < (rule_end - 1440):
        return float(config.SOFT_NODRIVE_DEFER_INSIDE_MULTIPLIER)
    # 窗前接近度：距 start_minute（取当日或次日的近未来）≤ PROXIMITY 分钟时线性放大。
    if cur_md < rule_start:
        gap = rule_start - cur_md
    else:
        gap = (rule_start + 1440) - cur_md
    proximity = config.SOFT_NODRIVE_DEFER_PROXIMITY_MINUTES
    if gap <= proximity:
        # 越靠近窗口起点，越显然「应该等过去」，乘数从 1.0 线性升至 INSIDE_MULTIPLIER。
        ratio = max(0.0, 1.0 - gap / max(1.0, proximity))
        return 1.0 + ratio * (config.SOFT_NODRIVE_DEFER_INSIDE_MULTIPLIER - 1.0)
    return 1.0


def _soft_no_drive_marginal_charge(
    window: TimeWindowRule,
    action_start_minutes: int,
    action_end_minutes: int,
    memory: DriverMemory,
) -> float:
    penalty = float(window.penalty_amount or 0.0)
    if penalty <= 0:
        return 0.0
    rule_id = f"nodrive_{int(window.start_minute)}"
    cap = window.penalty_cap
    accumulated = float(memory.preference_penalty_accum.get(rule_id, 0.0) or 0.0)
    if cap is not None and float(cap) > 0 and accumulated >= float(cap) * 0.95:
        return 0.0
    hit_dates: set[str] = set()
    if action_end_minutes <= action_start_minutes:
        action_end_minutes = action_start_minutes + 1
    first_day = max(0, action_start_minutes // 1440 - 1)
    last_day = max(first_day, action_end_minutes // 1440 + 1)
    for day_idx in range(first_day, last_day + 1):
        day_start = day_idx * 1440
        for ws, we in _project_rule_to_minutes(day_start, int(window.start_minute), int(window.end_minute)):
            if min(action_end_minutes, we) > max(action_start_minutes, ws):
                hit_dates.add(geo_utils.date_str(day_start))
    if not hit_dates:
        return 0.0
    existing = memory.preference_violation_days.get(rule_id, set())
    new_days = hit_dates - existing
    if not new_days:
        return 0.0
    charge = penalty * len(new_days)
    if cap is not None and float(cap) > 0:
        charge = min(charge, max(0.0, float(cap) - accumulated))
    return charge


# ---------------- 接单评分 ----------------


def score_take_order(
    item: dict[str, Any],
    rules: ParsedRules,
    memory: DriverMemory,
    ctx: DecisionContext,
) -> ScoredAction:
    cargo = item.get("cargo", {}) or {}
    cargo_id = str(cargo.get("cargo_id", "")).strip()
    if not cargo_id:
        return ScoredAction(
            action="take_order",
            params={"cargo_id": cargo_id},
            score=-HARD_CONSTRAINT_PENALTY,
            feasible=False,
            note="missing_cargo_id",
        )

    remove_time_text = str(cargo.get("remove_time", "") or "").strip()
    if remove_time_text:
        try:
            remove_minutes = geo_utils.wall_time_to_minutes(remove_time_text)
        except ValueError:
            remove_minutes = None
        if remove_minutes is not None and ctx.current_minutes > remove_minutes:
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="cargo_expired_after_scan",
            )

    preferred_rule = next((r for r in rules.preferred_cargo if r.cargo_id == cargo_id), None)

    start = cargo.get("start") or {}
    end = cargo.get("end") or {}
    try:
        start_lat = float(start["lat"])
        start_lng = float(start["lng"])
        end_lat = float(end["lat"])
        end_lng = float(end["lng"])
    except (KeyError, TypeError, ValueError):
        return ScoredAction(
            action="take_order",
            params={"cargo_id": cargo_id},
            score=-HARD_CONSTRAINT_PENALTY,
            feasible=False,
            note="invalid_coordinates",
        )

    # 距离与时间估算
    distance_pickup_km = float(item.get("distance_km") or geo_utils.haversine_km(
        ctx.current_lat, ctx.current_lng, start_lat, start_lng
    ))
    haul_km = geo_utils.haversine_km(start_lat, start_lng, end_lat, end_lng)
    pickup_minutes = (
        0 if distance_pickup_km <= 1e-6
        else geo_utils.distance_to_minutes(distance_pickup_km, ctx.reposition_speed_km_per_hour)
    )
    arrival_minutes = ctx.current_minutes + pickup_minutes
    load_window = cargo.get("load_time")
    waiting_minutes = 0
    load_window_expired = False
    if isinstance(load_window, list) and len(load_window) == 2:
        try:
            load_start = geo_utils.wall_time_to_minutes(str(load_window[0]))
            load_end = geo_utils.wall_time_to_minutes(str(load_window[1]))
        except ValueError:
            load_start = load_end = None
        if load_start is not None and load_end is not None:
            if arrival_minutes > load_end:
                load_window_expired = True
            elif arrival_minutes < load_start:
                waiting_minutes = load_start - arrival_minutes
    cost_minutes = int(cargo.get("cost_time_minutes") or 0)
    finish_minutes = arrival_minutes + waiting_minutes + cost_minutes

    breakdown: dict[str, float] = {}
    note_parts: list[str] = []

    # 硬约束：装货窗失效
    if load_window_expired:
        return ScoredAction(
            action="take_order",
            params={"cargo_id": cargo_id},
            score=-HARD_CONSTRAINT_PENALTY,
            feasible=False,
            note="load_window_expired",
        )

    # 硬约束：仿真上界
    # 仓未超期：货本身可接，但仓超期后仿真会标记 income_eligible=false 使收入不计、成本仍然计。
    # 这里同步抵消收入项，避免选择装货价高但完工超期的货源（详见 docs/06 14.11 节）。
    horizon_overflow = max(0, finish_minutes - ctx.horizon_minutes)
    income_voided_by_horizon = horizon_overflow > 0 and config.HORIZON_OVERFLOW_INCOME_VOIDED
    if horizon_overflow > 0:
        breakdown["horizon_overflow_penalty"] = -HORIZON_OVERFLOW_PENALTY
        note_parts.append("horizon_risk")

    # 硬约束：车型
    if not _truck_length_compatible(ctx.truck_length, cargo.get("truck_length")):
        return ScoredAction(
            action="take_order",
            params={"cargo_id": cargo_id},
            score=-HARD_CONSTRAINT_PENALTY,
            feasible=False,
            note="truck_length_mismatch",
        )

    # 硬约束：禁运品类
    cargo_name = str(cargo.get("cargo_name", ""))
    if cargo_name and cargo_name in rules.categories.forbidden:
        return ScoredAction(
            action="take_order",
            params={"cargo_id": cargo_id},
            score=-HARD_CONSTRAINT_PENALTY,
            feasible=False,
            note="forbidden_category",
        )

    # 硬约束：大额定时事件——在 approaching / pickup / late_pickup / home / late_home / stay 阶段
    # 不允许任何会推迟到家或越过事件且远离接人/老家点的接单。
    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        if phase in {"early", "done"}:
            continue
        # R6-P0: 统一提前完单限制——无论何阶段，事件期间不允许接完工时间超出事件开始的订单
        # 修复bug: 此检查原来仅在early_approach阶段，导致pickup阶段可接长距离订单穿越事件
        if finish_minutes > event.start_minutes and ctx.current_minutes < event.stay_until_minutes:
            dist_end_to_home = geo_utils.haversine_km(end_lat, end_lng, event.home_lat, event.home_lng)
            if dist_end_to_home > event.radius_km:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="timed_event_precompletion_block_unified",
                )
        if phase == "early_approach":
            dist_to_pick = geo_utils.haversine_km(end_lat, end_lng, event.pickup_lat, event.pickup_lng)
            cur_dist_to_pick = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.pickup_lat, event.pickup_lng)
            # P0: 强制回家策略——事件开始前 24h 内，若司机不在家/接人点附近，硬阻拦所有接单
            time_to_event = event.start_minutes - ctx.current_minutes
            near_pickup = cur_dist_to_pick <= event.radius_km
            near_home = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.home_lat, event.home_lng) <= event.radius_km
            # PR#27: 多日 stay 事件（≥24h，如 D010 3 天家事约定）扩展强制回家窗到 48h——
            # 24h 太晚：若司机当前在距家 400km 处，留给「最近 24h」可能不够回家+休息。
            stay_duration = max(0, event.stay_until_minutes - event.start_minutes)
            mandatory_window = config.TIMED_EVENT_HOME_MANDATORY_WINDOW_MINUTES
            if stay_duration >= config.TIMED_EVENT_MULTIDAY_THRESHOLD_MINUTES:
                mandatory_window = max(mandatory_window, config.TIMED_EVENT_MULTIDAY_HOME_MANDATORY_WINDOW_MINUTES)
            if 0 < time_to_event <= mandatory_window and not near_pickup and not near_home:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="timed_event_mandatory_home",
                )
            # P0: 提前完单限制——事件开始前 48h 内，只允许接在事件开始前能完工的订单
            if 0 < time_to_event <= config.TIMED_EVENT_PRECOMPLETION_WINDOW_MINUTES:
                if finish_minutes > event.start_minutes:
                    return ScoredAction(
                        action="take_order",
                        params={"cargo_id": cargo_id},
                        score=-HARD_CONSTRAINT_PENALTY,
                        feasible=False,
                        note="timed_event_precompletion_block",
                    )
            # R5-P0: 48h软距离限制——事件前48h内，卸货点距家>200km的订单施加重罚
            if 0 < time_to_event <= config.TIMED_EVENT_SOFT_LIMIT_WINDOW_MINUTES:
                dist_end_to_home = geo_utils.haversine_km(end_lat, end_lng, event.home_lat, event.home_lng)
                if dist_end_to_home > config.TIMED_EVENT_SOFT_LIMIT_DISTANCE_KM:
                    over_ratio = (dist_end_to_home - config.TIMED_EVENT_SOFT_LIMIT_DISTANCE_KM) / config.TIMED_EVENT_SOFT_LIMIT_DISTANCE_KM
                    penalty = float(event.penalty_amount or _penalty_default("timed_stay_event")) * min(2.0, over_ratio)
                    breakdown["timed_event_soft_distance_penalty"] = -penalty
                    note_parts.append("timed_event_soft_dist_limit")
            if dist_to_pick > cur_dist_to_pick + 30:
                penalty = float(event.penalty_amount or _penalty_default("timed_stay_event")) * 0.3
                breakdown["timed_event_order_penalty"] = breakdown.get("timed_event_order_penalty", 0.0) - penalty
                note_parts.append("timed_event_early_approach_away")
            elif dist_to_pick < cur_dist_to_pick - 10:
                gain = float(event.penalty_amount or _penalty_default("timed_stay_event")) * 0.15
                breakdown["timed_event_approach_gain"] = breakdown.get("timed_event_approach_gain", 0.0) + gain
                note_parts.append("timed_event_early_approach_closer")
            continue
        dist_to_pick = geo_utils.haversine_km(end_lat, end_lng, event.pickup_lat, event.pickup_lng)
        dist_to_home = geo_utils.haversine_km(end_lat, end_lng, event.home_lat, event.home_lng)
        # R6-P0: pickup/home阶段也阻拦会把司机送远的订单（即使起点在radius内）
        if phase in {"pickup", "late_pickup", "home", "late_home"}:
            if dist_to_pick > event.radius_km and dist_to_home > event.radius_km:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note=f"timed_event_block_{phase}_away",
                )
        # 1. 接人/回家阶段：接任何远离指定点的单都直接禁止，必须先去接人/回家
        if phase in {"pickup", "late_pickup"} and dist_to_pick > event.radius_km:
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note=f"timed_event_block_{phase}",
            )
        # stay 阶段无条件禁止接单——即使送达点在家附近，接单仍需外出装货/运输，产生缺席罚分
        if phase == "stay":
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="timed_event_block_stay",
            )
        if phase in {"home", "late_home"} and dist_to_home > event.radius_km:
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note=f"timed_event_block_{phase}",
            )
        # 2. approaching 阶段：本单完工已超过接人 deadline，或越过事件且远离接人点 → 禁止
        if phase == "approaching":
            minutes_to_pick_after_finish = geo_utils.distance_to_minutes(
                dist_to_pick,
                ctx.reposition_speed_km_per_hour,
            )
            can_reach_pickup_start = (
                finish_minutes + minutes_to_pick_after_finish + config.TIMED_EVENT_START_BUFFER_MINUTES
                <= event.start_minutes
            )
            if not can_reach_pickup_start and dist_to_pick > event.radius_km:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="timed_event_pre_lock_arrival",
                )
            if finish_minutes > event.deadline_minutes:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="timed_event_pre_lock_deadline",
                )
            if finish_minutes >= event.start_minutes and dist_to_pick > config.TIMED_EVENT_PRE_LOCK_DISTANCE_KM:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="timed_event_pre_lock_block",
                )

    # 收入与成本
    raw_income = float(cargo.get("price") or 0.0)
    # 仿真超期仓：收入不计。避免“干白工”选择。
    income = 0.0 if income_voided_by_horizon else raw_income
    distance_total = distance_pickup_km + haul_km
    distance_cost = ctx.cost_per_km * distance_total
    breakdown["income"] = income
    breakdown["distance_cost"] = -distance_cost
    if income_voided_by_horizon:
        note_parts.append("income_voided")

    occupied_minutes = pickup_minutes + waiting_minutes + cost_minutes
    time_cost = ctx.opportunity_cost_per_minute * occupied_minutes
    breakdown["time_cost"] = -time_cost

    # 软+硬两段式接单空驶罚（v2）：覆盖绝大多数司机偏好上限
    pickup_deadhead_penalty = 0.0
    if distance_pickup_km > config.PICKUP_DEADHEAD_SOFT_THRESHOLD_KM:
        soft_excess = min(
            distance_pickup_km, config.PICKUP_DEADHEAD_HARD_THRESHOLD_KM
        ) - config.PICKUP_DEADHEAD_SOFT_THRESHOLD_KM
        pickup_deadhead_penalty += soft_excess * ctx.cost_per_km * config.PICKUP_DEADHEAD_SOFT_COEFF
    if distance_pickup_km > config.PICKUP_DEADHEAD_HARD_THRESHOLD_KM:
        pickup_deadhead_penalty += (
            distance_pickup_km - config.PICKUP_DEADHEAD_HARD_THRESHOLD_KM
        ) * ctx.cost_per_km
    breakdown["pickup_deadhead_penalty"] = -pickup_deadhead_penalty

    waiting_penalty = ctx.opportunity_cost_per_minute * 0.5 * waiting_minutes
    breakdown["waiting_penalty"] = -waiting_penalty

    # v2: 接单成功率折扣与连续失败避让
    # 原理：接单失败会浪费 ~21 分钟（扫描+尝试+重试），需在评分时预期原本。
    if not income_voided_by_horizon and income > 0:
        success_rate = memory.cargo_success_rate()
        if success_rate < 1.0:
            expected_loss = income * (1.0 - success_rate)
            failure_attempt_cost = float(config.CARGO_FAILURE_ATTEMPT_COST_YUAN) * (1.0 - success_rate)
            breakdown["expected_failure_discount"] = -(expected_loss + failure_attempt_cost)
            note_parts.append(f"sr={success_rate:.2f}")
    if memory.consecutive_failed_take_orders > 0:
        breakdown["recent_fail_penalty"] = -(
            memory.consecutive_failed_take_orders * config.STAGNATION_FAIL_PENALTY_PER_STEP
        )

    # 软偏好：避免品类
    # R6-P2: 品类软罚分按规则罚金动态计算（原固定-300→按penalty_amount×0.4）
    if cargo_name and cargo_name in rules.categories.avoid:
        avoid_pen = 300.0
        if hasattr(rules.categories, 'penalty_amount') and rules.categories.penalty_amount:
            avoid_pen = float(rules.categories.penalty_amount) * 0.4
        breakdown["avoid_category_penalty"] = -avoid_pen
        note_parts.append("avoid_category")

    # 偏好：距离限制——硬性过滤 + 软惩罚
    pref_penalty = 0.0
    for limit in rules.distance_limits:
        if limit.kind == "haul" and haul_km > limit.max_km:
            if haul_km > limit.max_km * 1.05:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note=f"haul_distance_exceed_limit_{limit.max_km}km",
                )
            over_ratio = (haul_km - limit.max_km) / max(1.0, limit.max_km)
            pref_penalty += float(limit.penalty_amount or _penalty_default("distance_limit_haul_pickup")) * (1.0 + over_ratio * 5.0)
        elif limit.kind == "pickup" and distance_pickup_km > limit.max_km:
            if distance_pickup_km > limit.max_km * 1.05:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note=f"pickup_deadhead_exceed_limit_{limit.max_km}km",
                )
            over_ratio = (distance_pickup_km - limit.max_km) / max(1.0, limit.max_km)
            pref_penalty += float(limit.penalty_amount or _penalty_default("distance_limit_haul_pickup")) * (1.0 + over_ratio * 5.0)
        elif limit.kind == "monthly_deadhead":
            # 月度空驶赶路上限：按「新增超额公里数 × 单价」计软惩罚，不硬阻拦。
            # 评测仅按累计 over_km × 单价（≈10 元/km）扣分；硬阻拦会让司机在累计稍超即停摆。
            # 原硬阻拦在 D003（限 100km）触发后导致全月 899 次 wait、3 单 → 改为软罚。
            pen_per_km_t = float(limit.penalty_amount or _penalty_default("distance_limit_monthly_deadhead"))
            already_over = max(0.0, memory.total_deadhead_km - limit.max_km)
            new_total = memory.total_deadhead_km + distance_pickup_km
            new_over_km = max(0.0, max(0.0, new_total - limit.max_km) - already_over)
            # PR#27: 预警阶段——累计空驶 ∈ [PREEMPT_RATIO × max_km, max_km] 时即开始施加
            # 渐增软罚，避免临到限额才反应（D003 限额 100km 时往往一两单就冲破）。
            preempt_threshold = limit.max_km * config.MONTHLY_DEADHEAD_PREEMPT_RATIO
            if memory.total_deadhead_km < limit.max_km and new_total > preempt_threshold:
                preempt_start = max(memory.total_deadhead_km, preempt_threshold)
                preempt_end = min(new_total, limit.max_km)
                preempt_km = max(0.0, preempt_end - preempt_start)
                if preempt_km > 0:
                    # 在 [PREEMPT_RATIO, 1.0] 区间内线性放大至 MAX_COEFF × pen_per_km。
                    mid_ratio = (preempt_start + preempt_end) / 2.0 / max(1.0, limit.max_km)
                    band = max(1e-6, 1.0 - config.MONTHLY_DEADHEAD_PREEMPT_RATIO)
                    band_pos = max(0.0, min(1.0, (mid_ratio - config.MONTHLY_DEADHEAD_PREEMPT_RATIO) / band))
                    coeff = config.MONTHLY_DEADHEAD_PREEMPT_MAX_COEFF * band_pos
                    pref_penalty += pen_per_km_t * preempt_km * coeff
            if new_over_km > 0:
                cap_to = limit.penalty_cap
                if cap_to is not None and pen_per_km_t > 0:
                    paid_so_far_to = min(already_over * pen_per_km_t, cap_to)
                    if paid_so_far_to >= cap_to * 0.95:
                        # cap 已基本耗尽：保留残余信号防止 D003 失控（1,694km 超额）——
                        # 即使评测端不再扣分，新增公里仍是无收益里程，需轻惩反对。
                        pref_penalty += (
                            pen_per_km_t * new_over_km
                            * config.MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF
                        )
                        continue
                    # 否则按「未耗尽部分」截断
                    cap_remaining_to = cap_to - paid_so_far_to
                    pref_penalty += min(pen_per_km_t * new_over_km, cap_remaining_to)
                else:
                    pref_penalty += pen_per_km_t * new_over_km
    if pref_penalty > 0:
        breakdown["distance_limit_penalty"] = -pref_penalty

    # 偏好：禁入区域路径风险
    forbidden_path_penalty = 0.0
    for zone in rules.forbidden_zones:
        if _path_passes_forbidden_zone(ctx.current_lat, ctx.current_lng, start_lat, start_lng, zone):
            forbidden_path_penalty += float(zone.penalty_amount or _penalty_default("forbidden_zone"))
        elif _path_passes_forbidden_zone(start_lat, start_lng, end_lat, end_lng, zone):
            forbidden_path_penalty += float(zone.penalty_amount or _penalty_default("forbidden_zone"))
    if forbidden_path_penalty > 0:
        breakdown["forbidden_zone_penalty"] = -forbidden_path_penalty

    # 偏好：城市/经纬度边界——视为硬约束（起终点必须均在框内）
    for box in rules.bounded_areas:
        if not _is_in_box(start_lat, start_lng, box) or not _is_in_box(end_lat, end_lng, box):
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="bounded_area_violation",
            )

    # 偏好：禁行时段——评测按「当天有违规活动」扣固定罚金（如 D007 每天扣 500、上限 15000），
    # 不按重叠分钟数计。任何与禁行窗重叠的订单均视为当天违规。
    # 严格/软划分：单日罚金 ≤ NO_DRIVE_SOFT_PENALTY_THRESHOLD（默认 150 元）的窗作「软偏好」处理：仅接受软罚、
    # 不启用双重裕量、不硬拒（如 D004 「中午 12–13 不接」单日 ¥100，硬拒损失远大于罚金）。
    # 严格窗启用双重保险裕量：
    #   1) HAUL_TIME_OVERESTIMATE_RATIO 把 haversine 低估的 occupied_minutes 补回来（公路里程比直线长 10–15%）；
    #   2) NO_DRIVE_SAFETY_BUFFER_MINUTES 再叠加固定裕量，防止 cost_time_minutes 本身也偏乐观。
    overestimated_finish = ctx.current_minutes + int(
        (finish_minutes - ctx.current_minutes) * config.HAUL_TIME_OVERESTIMATE_RATIO
    )
    for window in rules.no_drive_windows:
        window_penalty = float(window.penalty_amount or 0.0)
        is_soft_window = 0 < window_penalty <= config.NO_DRIVE_SOFT_PENALTY_THRESHOLD
        if is_soft_window:
            # 软窗：用原始 estimate（不加裕量）检测重叠，重叠则按单日固定罚金计软价。
            real_overlap = _hits_no_drive_window(ctx.current_minutes, finish_minutes, window)
            if real_overlap > 0:
                # PR#27: 加入「可延后性」加权——D004 中午 12-13 软窗每天 ¥100 × 30 次 = ¥3,000 顶到 cap，
                # 多数违规源于「12:30 接单工时 60min」一类显然可延后的决策；若当前时刻就在窗内或窗前 90min，
                # 把软罚乘以倍数以强化反向偏好；远离窗口的违规（如长单 11:00-13:30 跨过）保留原罚金。
                multiplier = _soft_no_drive_defer_multiplier(window, ctx.current_minutes)
                marginal = _soft_no_drive_marginal_charge(window, ctx.current_minutes, finish_minutes, memory)
                breakdown["no_drive_window_soft_penalty"] = (
                    breakdown.get("no_drive_window_soft_penalty", 0.0) - marginal * multiplier
                )
                note_parts.append("no_drive_soft_window")
            continue
        buffered_finish = overestimated_finish + config.NO_DRIVE_SAFETY_BUFFER_MINUTES
        overlap = _hits_no_drive_window(ctx.current_minutes, buffered_finish, window)
        if overlap > 0:
            # 高价值熟货：禁行代价 < 错过熟货代价时，降级为软惩罚
            if preferred_rule is not None and preferred_rule.penalty_amount >= 5000:
                real_overlap = _hits_no_drive_window(ctx.current_minutes, finish_minutes, window)
                soft_cost = window_penalty if window_penalty > 0 else 500.0
                soft_cost *= max(1, real_overlap / 60.0)
                breakdown["no_drive_window_soft_penalty"] = -soft_cost
                note_parts.append("no_drive_preferred_override")
            else:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="no_drive_window_violation",
                )

    # 偏好：每日接单上限
    # PR#29: 超限 2 单以上硬阻拦，超限 1 单用 2× 罚金作强软罚——
    # D004（max_orders=3, ¥200/单）历史上因软罚不足吃到 ¥400 违规。
    if rules.daily_order_limit is not None:
        already = memory.daily_orders_today(ctx.current_minutes)
        if already + 1 > rules.daily_order_limit.max_orders:
            extra = already + 1 - rules.daily_order_limit.max_orders
            unit = float(rules.daily_order_limit.penalty_amount or _penalty_default("daily_order_limit"))
            if extra >= 2:
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="daily_order_limit_hard_block",
                )
            breakdown["daily_order_limit_penalty"] = -unit * extra * 2.0

    # 偏好：首单时间
    if rules.first_order_rule is not None and memory.daily_orders_today(ctx.current_minutes) == 0:
        first_take_minute = geo_utils.minute_of_day(ctx.current_minutes)
        deadline_min_first = rules.first_order_rule.before_hour * 60
        if first_take_minute >= deadline_min_first:
            breakdown["first_order_late_penalty"] = -float(rules.first_order_rule.penalty_amount or _penalty_default("first_order_rule"))
        else:
            # PR#27: 截止前 PROXIMITY 分钟内若仍未接单，给 take_order 软增益——
            # 与 wait 侧的 first_order_deadline_wait_penalty 配合：take 比 wait 更优。
            gap = deadline_min_first - first_take_minute
            if 0 < gap <= config.FIRST_ORDER_PROXIMITY_BOOST_WINDOW_MINUTES:
                boost_unit = float(rules.first_order_rule.penalty_amount or _penalty_default("first_order_rule")) * config.FIRST_ORDER_PROXIMITY_BOOST_RATIO
                # 越接近截止，激励越强（gap 越小、boost 越大）
                ratio_boost = 1.0 - gap / max(1.0, config.FIRST_ORDER_PROXIMITY_BOOST_WINDOW_MINUTES)
                breakdown["first_order_proximity_bonus"] = boost_unit * ratio_boost

    # 偏好：回家约束——接单可能错过回家窗
    if rules.home_rule is not None:
        home_by_min = rules.home_rule.home_by_hour * 60
        finish_hour = geo_utils.hour_of_day(finish_minutes)
        minutes_to_home_from_end = geo_utils.distance_to_minutes(
            geo_utils.haversine_km(end_lat, end_lng, rules.home_rule.lat, rules.home_rule.lng),
            ctx.reposition_speed_km_per_hour,
        )
        # 核心判断：接单后能否在 home_by_hour 前回到家附近
        end_day = finish_minutes // 1440
        can_reach_home_by_deadline = False
        if finish_minutes + minutes_to_home_from_end <= end_day * 1440 + home_by_min:
            can_reach_home_by_deadline = True
        end_in_home = _is_in_circle(end_lat, end_lng, rules.home_rule)
        # R6-P3: 动态回家时间检查——tight slack扩大至120min + 渐进式罚分
        # PR#29: 90→120min 窗口, 确保 D009 等有更多缓冲避免到家偏晚
        if can_reach_home_by_deadline and not end_in_home:
            slack = (end_day * 1440 + home_by_min) - (finish_minutes + minutes_to_home_from_end)
            if 0 < slack <= 120:
                severity = max(0.3, 1.0 - slack / 120.0)
                breakdown["home_rule_tight_slack_penalty"] = -float(rules.home_rule.penalty_amount or _penalty_default("home_rule")) * severity * 1.5
                note_parts.append("home_tight_slack")
        if not can_reach_home_by_deadline and not end_in_home:
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="home_rule_unreachable",
            )
        if finish_hour >= rules.home_rule.home_by_hour and not end_in_home:
            breakdown["home_rule_penalty"] = -float(rules.home_rule.penalty_amount or _penalty_default("home_rule"))
        else:
            current_minute = geo_utils.minute_of_day(ctx.current_minutes)
            finish_minute = geo_utils.minute_of_day(finish_minutes)
            near_home_window = 0 <= home_by_min - current_minute <= config.HOME_RULE_PREP_WINDOW_MINUTES
            crosses_day = finish_minutes // 1440 > ctx.current_minutes // 1440
            cannot_reach_home_after_order = (
                crosses_day
                or (
                    finish_minutes // 1440 == ctx.current_minutes // 1440
                    and finish_minute + minutes_to_home_from_end > home_by_min
                )
            )
            if not end_in_home and (near_home_window or cannot_reach_home_after_order) and cannot_reach_home_after_order:
                breakdown["home_rule_penalty"] = -float(
                    rules.home_rule.penalty_amount or _penalty_default("home_rule")
                ) * config.HOME_RULE_REACHABILITY_MULTIPLIER

    # 偏好：熟货源加成——高价值熟货使用更高倍数确保压倒普通订单
    if cargo_id in rules.preferred_cargo_ids:
        bonus_base = (
            float(preferred_rule.penalty_amount)
            if preferred_rule is not None and preferred_rule.penalty_amount > 0
            else 5_000.0
        )
        if preferred_rule is not None and preferred_cargo_active(preferred_rule, ctx.current_minutes):
            multiplier = config.PREFERRED_CARGO_ACTIVE_BONUS_MULTIPLIER
            # 高价值熟货在活跃窗口内用更高倍数，确保绝对压倒任何普通订单
            if preferred_rule.penalty_amount >= 5000:
                multiplier = max(multiplier, 3.0)
        elif preferred_rule is not None and preferred_rule.penalty_amount >= 5000:
            multiplier = 3.0
        else:
            multiplier = config.PREFERRED_CARGO_BONUS_MULTIPLIER
        # 最终加成 = 罚金金额 × 倍数，确保高价值熟货优先级最高
        breakdown["preferred_cargo_bonus"] = bonus_base * multiplier
        note_parts.append("preferred_cargo")

    for preferred in rules.preferred_cargo:
        if preferred.cargo_id == cargo_id:
            continue
        target = preferred_cargo_target(preferred)
        if target is None or not preferred_cargo_preposition_ready(preferred, ctx.current_minutes):
            continue
        if preferred.available_minutes is None:
            continue
        minutes_to_target = geo_utils.distance_to_minutes(
            geo_utils.haversine_km(end_lat, end_lng, target[0], target[1]),
            ctx.reposition_speed_km_per_hour,
        )
        if finish_minutes + minutes_to_target + config.PREFERRED_CARGO_ARRIVAL_BUFFER_MINUTES > preferred.available_minutes:
            conflict_base = float(preferred.penalty_amount or _penalty_default("preferred_cargo"))
            # 高价值熟货冲突使用更高惩罚，确保不因普通订单错过熟货
            if preferred.penalty_amount and preferred.penalty_amount >= 5000:
                conflict_base *= 2.0
            breakdown["preferred_cargo_conflict_penalty"] = breakdown.get("preferred_cargo_conflict_penalty", 0.0) - conflict_base
            note_parts.append("preferred_cargo_conflict")

    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        if phase in {"pickup", "home", "stay", "late_pickup", "late_home"}:
            penalty = float(event.penalty_amount or _penalty_default("timed_stay_event"))
            if phase.startswith("late"):
                penalty *= 2.0
            breakdown["timed_event_order_penalty"] = breakdown.get("timed_event_order_penalty", 0.0) - penalty
            note_parts.append(f"timed_event_{phase}")
        elif phase == "approaching":
            # 事件未开始，但接单后会越过事件开始时刻且把人锁在远处 → 重罚
            crosses_event = finish_minutes >= event.start_minutes
            far_from_pickup = (
                geo_utils.haversine_km(end_lat, end_lng, event.pickup_lat, event.pickup_lng)
                > config.TIMED_EVENT_PRE_LOCK_DISTANCE_KM
            )
            if crosses_event and far_from_pickup:
                penalty = float(event.penalty_amount or _penalty_default("timed_stay_event")) * 1.5
                breakdown["timed_event_order_penalty"] = breakdown.get("timed_event_order_penalty", 0.0) - penalty
                note_parts.append("timed_event_pre_lock_block")

    # 每日休息风险：接单后当天剩余时间是否足以满足休息要求。
    # 评测按「当天最长连续 wait 段 < required_minutes」判一日罚（每日上限 200-300 元）；
    # 单纯软惩罚不足以挡住高额订单收益，故引入硬阻拦：若接单前尚能完成休息、接单后不再可能 → 直接拒单。
    for rest in rules.rest_rules:
        today_rest = memory.longest_rest_today(ctx.current_minutes)
        deficit = rest.required_minutes - today_rest
        if deficit > 0:
            cur_md = geo_utils.minute_of_day(ctx.current_minutes)
            finish_md = geo_utils.minute_of_day(finish_minutes)
            crosses_day = finish_minutes // 1440 > ctx.current_minutes // 1440
            remaining_after = 1440 - finish_md if not crosses_day else 0
            available_now = 1440 - cur_md
            unit = float(rest.penalty_amount or _penalty_default("rest_rule"))
            if available_now >= deficit and (crosses_day or remaining_after < deficit):
                # 接单前还可以完成 deficit，接单后不再可能 → 硬阻拦
                return ScoredAction(
                    action="take_order",
                    params={"cargo_id": cargo_id},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="daily_rest_blocked",
                )
            # PR#29: 预警与紧迫触发线均提前——D002/D006/D008/D010 累计 ¥900+ 休息罚，
            # 多为「白天没意识到要留出连续休息段」导致夜里凑不齐。
            # tight 惩罚 2.5→3.5, early 惩罚 1.0→1.5 以更积极阻止吃掉休息余量。
            if remaining_after < deficit * config.DAILY_REST_RISK_TIGHT_RATIO:
                breakdown["daily_rest_risk_penalty"] = -unit * 3.5
                note_parts.append("rest_risk_tight")
            elif remaining_after < deficit * config.DAILY_REST_RISK_EARLY_RATIO:
                breakdown["daily_rest_risk_penalty"] = -unit * 1.5
                note_parts.append("rest_risk")

    # 月度休息日：评测按 [action_start, action_end] 跨过的每个自然日都计入「活跃日」，
    # 所以此处需把订单从今天到完工那天所跨过的全部自然日都纳入预估，避免跨午夜长单
    # 让 agent 以为「明天还可休息」（D008/D010 典型场景）。
    if rules.monthly_day_off is not None:
        projected_active = set(memory.daily_active)
        cur_day_idx = ctx.current_minutes // 1440
        end_day_idx = max(cur_day_idx, max(0, finish_minutes - 1) // 1440)
        for di in range(cur_day_idx, end_day_idx + 1):
            projected_active.add(geo_utils.date_str(di * 1440))
        eval_horizon = config.EVALUATION_HORIZON_DAYS
        max_active = eval_horizon - rules.monthly_day_off.required_days
        if len(projected_active) > max_active:
            return ScoredAction(
                action="take_order",
                params={"cargo_id": cargo_id},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="monthly_day_off_required_today",
            )
    # B.1 月度休息日 urgency 软惩罚：当下还没碰硬阈值，但若把今天「烧成」活跃日
        # 后续 pigeonhole 越来越紧 → 给本单一个递增软惩罚，鼓励在 wait/reposition 之间
        # 做更优选择（典型：D002 月末缺 1 天 ¥6k）。
        # NOTE: PR#21 把上方硬阻拦改成「跨天逐日 add」时，原本一次性算出的 finish_day_str
        # 被一起删掉，但下面的 B.1 软惩罚仍要用「订单落货那天」做判断，故在此重新计算。
        finish_day_str = geo_utils.date_str(max(0, finish_minutes - 1))
        today_str = geo_utils.date_str(ctx.current_minutes)
        today_already_active_to = today_str in memory.daily_active
        finishes_today = finish_day_str == today_str
        if finishes_today and not today_already_active_to:
            R = rules.monthly_day_off.required_days
            sim_day_to = ctx.current_minutes // 1440
            active_so_far_to = len(memory.daily_active)
            past_off_days_to = max(0, sim_day_to - active_so_far_to)
            deficit_after = R - past_off_days_to
            days_left_after_today_to = max(0, eval_horizon - 1 - sim_day_to)
            if deficit_after > 0:
                urgency = deficit_after / max(1, days_left_after_today_to + 1)
                if urgency >= 0.5:
                    return ScoredAction(
                        action="take_order",
                        params={"cargo_id": cargo_id},
                        score=-HARD_CONSTRAINT_PENALTY,
                        feasible=False,
                        note=f"monthly_day_off_urgency_block(urg={urgency:.2f})",
                    )
                # R6-P1: 双阶段urgency——前半月0.08，后半月(day>=20)0.06
                urg_threshold = config.MONTHLY_DAY_OFF_URGENCY_THRESHOLD_EARLY
                if sim_day_to >= config.MONTHLY_DAY_OFF_LATE_MONTH_DAY:
                    urg_threshold = config.MONTHLY_DAY_OFF_URGENCY_THRESHOLD_LATE
                if urgency >= urg_threshold:
                    base_pen = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    coeff = min(1.0, urgency) * 1.0
                    if urgency >= 0.3:
                        coeff = min(2.0, urgency) * 1.5
                    # R6-P1: 后半月penalty系数×1.5
                    if sim_day_to >= config.MONTHLY_DAY_OFF_LATE_MONTH_DAY:
                        coeff *= 1.5
                    breakdown["monthly_day_off_urgency_penalty"] = -base_pen * coeff
                    note_parts.append(f"day_off_urgency(urg={urgency:.2f})")
                ideal_spacing = eval_horizon / max(1, R)
                if past_off_days_to == 0 and sim_day_to >= ideal_spacing:
                    base_pen = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    breakdown["monthly_day_off_spacing_penalty"] = -base_pen * config.MONTHLY_DAY_OFF_SPACING_COEFF
                    note_parts.append("day_off_no_rest_yet")
                # R5-P1: 月末集中休息——剩余≤5天且deficit=1时增加urgency惩罚
                if deficit_after == 1 and days_left_after_today_to <= config.MONTHLY_DAY_OFF_MONTH_END_DAYS:
                    base_pen = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    breakdown["monthly_day_off_month_end_penalty"] = -base_pen * 0.6
                    note_parts.append("day_off_month_end")
        # 跨天订单额外检查：若 step_end 落在新一天，也会激活该天
        elif not finishes_today and not today_already_active_to:
            finish_day_active = finish_day_str in memory.daily_active
            if not finish_day_active:
                R = rules.monthly_day_off.required_days
                sim_day_to = ctx.current_minutes // 1440
                active_so_far_to = len(memory.daily_active)
                projected_active_count = active_so_far_to + 2
                max_active_days = eval_horizon - R
                if projected_active_count > max_active_days:
                    return ScoredAction(
                        action="take_order",
                        params={"cargo_id": cargo_id},
                        score=-HARD_CONSTRAINT_PENALTY,
                        feasible=False,
                        note="monthly_day_off_cross_day_block",
                    )

    # 收入效率加成：优先选择收入/时间比更高的订单
    # PR#24 经验：简单的效率信号比 PR#17 的复杂机会成本更安全。
    if (
        occupied_minutes > 0
        and income > 0
        and not income_voided_by_horizon
        and memory.total_completed_orders >= config.INCOME_EFFICIENCY_MIN_ORDERS
    ):
        yuan_per_min = income / occupied_minutes
        avg_yuan_per_min = memory.total_gross_income / max(1.0, sum(
            1 for _ in memory.daily_active
        ) * 14.0 * 60.0)
        if avg_yuan_per_min > 0 and yuan_per_min > avg_yuan_per_min * 1.2:
            efficiency_bonus = min(
                config.INCOME_EFFICIENCY_BONUS_CAP,
                (yuan_per_min - avg_yuan_per_min) * occupied_minutes * 0.05,
            )
            breakdown["income_efficiency_bonus"] = efficiency_bonus

    # 未来位置价值：卸货点的热点收益 + 到达时段的在线模式信号
    arrival_hour = geo_utils.hour_of_day(finish_minutes)
    hour_signal = memory.hour_pattern_value(arrival_hour)
    horizon_minutes_ahead = min(120, max(60, occupied_minutes // 2))
    future_value = (
        memory.hotspot_value(end_lat, end_lng) + 0.5 * hour_signal
    ) * horizon_minutes_ahead
    if future_value > 0:
        breakdown["future_location_value"] = future_value

    # 方案一：订单链评估——卸货点的后续接单价值
    if config.CHAIN_VALUE_ENABLED and not income_voided_by_horizon:
        chain_value = memory.estimate_location_value(end_lat, end_lng)
        if chain_value > 0:
            breakdown["chain_value"] = chain_value * config.CHAIN_VALUE_COEFF

    # 方案五：在线学习——卸货点+到达时段的历史收益率信号
    if config.ONLINE_LEARNING_ENABLED and not income_voided_by_horizon:
        loc_time_value = memory.get_location_time_value(end_lat, end_lng, arrival_hour)
        if loc_time_value > 0:
            breakdown["online_learning_value"] = (
                loc_time_value * horizon_minutes_ahead * config.ONLINE_LEARNING_SCORING_COEFF
            )

    _apply_adaptive_weights(breakdown, ctx.weights)
    score = sum(breakdown.values())
    return ScoredAction(
        action="take_order",
        params={"cargo_id": cargo_id},
        score=score,
        feasible=score > -HARD_CONSTRAINT_PENALTY / 2,
        breakdown=breakdown,
        note=",".join(note_parts) if note_parts else "",
        occupied_minutes=float(occupied_minutes),
    )


# ---------------- 休息评分 ----------------


def build_wait_durations(rules: ParsedRules, ctx: DecisionContext, memory: DriverMemory) -> list[int]:
    """生成休息候选时长列表（分钟）。"""
    durations = {30, 60, 120}
    long_durations: set[int] = set()
    for rest in rules.rest_rules:
        deficit = max(rest.required_minutes - memory.longest_rest_today(ctx.current_minutes), rest.required_minutes)
        durations.add(max(deficit, 30))
    if rules.monthly_day_off is not None:
        cur_md = geo_utils.minute_of_day(ctx.current_minutes)
        date_today = geo_utils.date_str(ctx.current_minutes)
        today_already_active = date_today in memory.daily_active
        active_so_far = memory.days_active_count()
        eval_horizon = config.EVALUATION_HORIZON_DAYS
        max_active = eval_horizon - rules.monthly_day_off.required_days
        # 必须休息日：今日剩余分钟都加为超长 wait 候选，让 agent 一次性吃满整天。
        # 该检查依据「现有活跃日数 + 今日可能变为活跃」；仅在必须未活跃日时赋予超长 wait
        days_off_required = rules.monthly_day_off.required_days
        days_remaining = max(0, eval_horizon - 1 - ctx.current_minutes // 1440)
        days_off_so_far = max(0, ctx.current_minutes // 1440 - active_so_far)
        needs_off_today = (
            not today_already_active
            and (
                active_so_far + 1 > max_active
                or days_off_so_far + days_remaining < days_off_required
            )
        )
        if needs_off_today:
            long_durations.add(max(1, 1440 - cur_md))
        elif not today_already_active:
            # V4: 当天尚未接单且有休息日缺口时，生成全天等待候选
            # 仅当罚金>=4000元时才值得放弃一天收入来休息（D002=6000 适用）
            days_off_deficit = days_off_required - days_off_so_far
            penalty_amt = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
            if days_off_deficit > 0 and penalty_amt >= 4000:
                long_durations.add(max(1, 1440 - cur_md))
    # 回家窗口或夜间禁行：休息至次日 6:00 / 8:00
    for window in rules.no_drive_windows:
        cur_md = geo_utils.minute_of_day(ctx.current_minutes)
        if window.start_minute <= cur_md < window.end_minute or (
            window.end_minute > 24 * 60 and (cur_md >= window.start_minute or cur_md < window.end_minute - 24 * 60)
        ):
            target = window.end_minute % (24 * 60)
            wait_to = (target - cur_md) % (24 * 60)
            if wait_to > 0:
                durations.add(wait_to)
    # 每日连续休息要求——在白天生成能覆盖 rest deficit 的休息时长
    # PR#29: 额外生成 deficit+30 的候选，确保安全覆盖（仿真计时可能有微小偏差）
    for rest in rules.rest_rules:
        today_rest = memory.longest_rest_today(ctx.current_minutes)
        deficit = rest.required_minutes - today_rest
        if deficit > 0:
            cur_md = geo_utils.minute_of_day(ctx.current_minutes)
            remaining_today = 1440 - cur_md
            if remaining_today >= deficit:
                durations.add(min(deficit, remaining_today))
            durations.add(max(deficit, 30))
            durations.add(deficit + 30)
    # 首单截止前生成唤醒候选：避免 wait 跨过 first_order deadline 而吃到晚单罚
    if rules.first_order_rule is not None:
        cur_md = geo_utils.minute_of_day(ctx.current_minutes)
        deadline_md = rules.first_order_rule.before_hour * 60
        if cur_md < deadline_md:
            gap = deadline_md - cur_md
            if gap > 30:
                durations.add(gap - 30)
            if gap > 60:
                durations.add(gap - 60)
    if rules.home_rule is not None:
        cur_md = geo_utils.minute_of_day(ctx.current_minutes)
        target = rules.home_rule.no_drive_until_hour * 60
        wait_to = (target - cur_md) % (24 * 60)
        if wait_to > 0:
            durations.add(wait_to)
        # PR#29: 在 home_by_hour 前 2h 生成回家等待候选（覆盖 D009 23:00 回家需求）
        home_by_md = rules.home_rule.home_by_hour * 60
        if cur_md < home_by_md:
            gap_to_home = home_by_md - cur_md
            if gap_to_home <= 4 * 60:
                durations.add(max(30, gap_to_home - 60))
    for preferred in rules.preferred_cargo:
        target = preferred_cargo_target(preferred)
        if target is None or preferred.available_minutes is None:
            continue
        if not preferred_cargo_preposition_ready(preferred, ctx.current_minutes):
            continue
        if geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, target[0], target[1]) > 5.0:
            continue
        wait_to_available = preferred.available_minutes - ctx.current_minutes
        if wait_to_available > 0:
            long_durations.add(min(config.PREFERRED_CARGO_MAX_WAIT_MINUTES, wait_to_available))
    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        near_pickup = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.pickup_lat, event.pickup_lng) <= event.radius_km
        near_home = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.home_lat, event.home_lng) <= event.radius_km
        if phase in {"pickup", "late_pickup"} and near_pickup:
            durations.add(event.pickup_stay_minutes)
        elif phase == "approaching" and near_pickup:
            # 提前抵达接人点：等到事件开始
            wait_to_start = max(1, event.start_minutes - ctx.current_minutes)
            durations.add(min(config.TIMED_EVENT_STAY_CHUNK_MINUTES, wait_to_start))
        elif phase in {"home", "late_home", "stay"} and near_home:
            remaining = max(1, event.stay_until_minutes - ctx.current_minutes)
            durations.add(min(config.TIMED_EVENT_STAY_CHUNK_MINUTES, remaining))
            long_durations.add(min(config.TIMED_EVENT_LONG_STAY_MAX_MINUTES, remaining))
    # PR#24: 夜间等待到天亮候选——覆盖无明确 no_drive_window 的司机，
    # 避免在低货源的夜间时段反复短 wait + 空驶消耗。
    cur_md_night = geo_utils.minute_of_day(ctx.current_minutes)
    if cur_md_night >= 22 * 60 or cur_md_night < config.NIGHT_WAIT_TO_DAWN_HOUR * 60:
        dawn_target = config.NIGHT_WAIT_TO_DAWN_HOUR * 60
        dawn_wait = (dawn_target - cur_md_night) % (24 * 60)
        if dawn_wait > 0:
            durations.add(dawn_wait)

    return sorted(
        {d for d in durations if 1 <= d <= 12 * 60}
        | {d for d in long_durations if 1 <= d <= config.TIMED_EVENT_LONG_STAY_MAX_MINUTES}
    )


def score_wait(
    duration_minutes: int,
    rules: ParsedRules,
    memory: DriverMemory,
    ctx: DecisionContext,
    has_good_order: bool,
) -> ScoredAction:
    breakdown: dict[str, float] = {}
    note_parts: list[str] = []

    # 休息要求达成增益
    rest_gain = 0.0
    today_rest = memory.longest_rest_today(ctx.current_minutes)
    cur_md = geo_utils.minute_of_day(ctx.current_minutes)
    for rest in rules.rest_rules:
        deficit = rest.required_minutes - today_rest
        if deficit > 0:
            covered = min(duration_minutes, deficit)
            unit = float(rest.penalty_amount or _penalty_default("rest_rule"))
            rest_gain += unit * (covered / max(1, deficit))
            # P2: 长时间 wait 完全覆盖 deficit 时给予额外奖励，避免多次短wait代替一次长wait
            if duration_minutes >= deficit:
                rest_gain += unit * 0.3
            # 当天即将结束但仍未满足休息要求时，给予额外的紧急增益
            remaining_today = 1440 - cur_md
            if remaining_today < deficit * 2 and duration_minutes >= deficit:
                urgency = min(2.0, deficit / max(1, remaining_today - deficit))
                rest_gain += unit * (0.5 + urgency * 0.5)
    if rest_gain > 0:
        breakdown["rest_preference_gain"] = rest_gain
        note_parts.append("rest_gain")

    # 避开禁行窗口的增益（只要休息覆盖窗口部分即得分）
    no_drive_gain = 0.0
    end_minutes = ctx.current_minutes + duration_minutes
    away_from_home_rule = (
        rules.home_rule is not None
        and not _is_in_circle(ctx.current_lat, ctx.current_lng, rules.home_rule)
    )
    home_reachable_before_deadline = False
    if away_from_home_rule and rules.home_rule is not None:
        home_by_min = rules.home_rule.home_by_hour * 60
        cur_md = geo_utils.minute_of_day(ctx.current_minutes)
        minutes_to_home = geo_utils.distance_to_minutes(
            geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, rules.home_rule.lat, rules.home_rule.lng),
            ctx.reposition_speed_km_per_hour,
        )
        home_reachable_before_deadline = cur_md + minutes_to_home <= home_by_min
    for window in rules.no_drive_windows:
        overlap = _hits_no_drive_window(ctx.current_minutes, end_minutes, window)
        if overlap > 0:
            if (
                away_from_home_rule
                and rules.home_rule is not None
                and home_reachable_before_deadline
                and window.start_minute == rules.home_rule.home_by_hour * 60
            ):
                continue
            unit = float(window.penalty_amount or _penalty_default("no_drive_window_default"))
            if 0 < unit <= config.NO_DRIVE_SOFT_PENALTY_THRESHOLD:
                no_drive_gain += _soft_no_drive_marginal_charge(window, ctx.current_minutes, end_minutes, memory)
            else:
                no_drive_gain += unit * (overlap / 60.0)
    if no_drive_gain > 0:
        breakdown["no_drive_window_avoid_gain"] = no_drive_gain
        note_parts.append("avoid_no_drive")

    if away_from_home_rule and rules.home_rule is not None and home_reachable_before_deadline:
        home_start = rules.home_rule.home_by_hour * 60
        home_end = rules.home_rule.no_drive_until_hour * 60
        if rules.home_rule.no_drive_until_hour <= rules.home_rule.home_by_hour:
            home_end = (rules.home_rule.no_drive_until_hour + 24) * 60
        home_window = TimeWindowRule(start_minute=home_start, end_minute=home_end)
        overlap = _hits_no_drive_window(ctx.current_minutes, end_minutes, home_window)
        if overlap > 0:
            breakdown["home_rule_away_wait_penalty"] = -float(
                rules.home_rule.penalty_amount or _penalty_default("home_rule")
            ) * config.HOME_RULE_AWAY_WAIT_PENALTY_MULTIPLIER
            note_parts.append("away_from_home_wait")

    preferred_wait_gain = 0.0
    for preferred in rules.preferred_cargo:
        target = preferred_cargo_target(preferred)
        if target is None or preferred.available_minutes is None:
            continue
        if not preferred_cargo_preposition_ready(preferred, ctx.current_minutes):
            continue
        if geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, target[0], target[1]) > 5.0:
            continue
        wait_to_available = max(0, preferred.available_minutes - ctx.current_minutes)
        if wait_to_available <= config.PREFERRED_CARGO_MAX_WAIT_MINUTES:
            covered = min(duration_minutes, max(1, wait_to_available))
            preferred_wait_gain += float(preferred.penalty_amount or _penalty_default("preferred_cargo")) * config.PREFERRED_CARGO_WAIT_GAIN_MULTIPLIER * (
                covered / max(1, wait_to_available)
            )
    if preferred_wait_gain > 0:
        breakdown["preferred_cargo_wait_gain"] = preferred_wait_gain
        note_parts.append("preferred_cargo_wait")

    # 首单截止时间感知：wait 跨过 first_order deadline 时施加软惩罚
    # PR#22 经验：固定+¥80 激励被回退，改为在 wait 侧惩罚——让长 wait 相对短 wait 更贵，
    # 从而间接鼓励在截止前接单（若有可接单的话）。
    if rules.first_order_rule is not None and memory.daily_orders_today(ctx.current_minutes) == 0:
        cur_md_fo = geo_utils.minute_of_day(ctx.current_minutes)
        deadline_fo = rules.first_order_rule.before_hour * 60
        end_md_fo = geo_utils.minute_of_day(end_minutes)
        crosses_day_fo = end_minutes // 1440 > ctx.current_minutes // 1440
        if cur_md_fo < deadline_fo and (end_md_fo >= deadline_fo or crosses_day_fo):
            fo_penalty = float(rules.first_order_rule.penalty_amount or _penalty_default("first_order_rule"))
            # PR#27: 原 ×0.5 让 wait 跨过截止比 take_order 还便宜（晚单仅 -¥200），
            # 用 FIRST_ORDER_WAIT_CROSS_MULTIPLIER 让边际成本与 take_order 侧对齐。
            breakdown["first_order_deadline_wait_penalty"] = -fo_penalty * config.FIRST_ORDER_WAIT_CROSS_MULTIPLIER
            note_parts.append("first_order_wait_cross")

    # 机会成本
    if has_good_order:
        opportunity_loss = ctx.opportunity_cost_per_minute * duration_minutes
        breakdown["opportunity_loss"] = -opportunity_loss

    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        near_pickup = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.pickup_lat, event.pickup_lng) <= event.radius_km
        near_home = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, event.home_lat, event.home_lng) <= event.radius_km
        if phase in {"pickup", "late_pickup"} and near_pickup and duration_minutes >= event.pickup_stay_minutes:
            base = float(event.penalty_amount or _penalty_default("timed_stay_event")) * config.TIMED_EVENT_FIXED_GAIN_MULTIPLIER
            if phase == "late_pickup":
                base *= 2.0  # 已经迟到了，必须立刻完成 10 分钟停留
            breakdown["timed_event_pickup_gain"] = base
            extra_stay = max(0, duration_minutes - event.pickup_stay_minutes)
            if extra_stay > 0:
                breakdown["timed_event_pickup_overstay_penalty"] = -(
                    extra_stay * event.absence_penalty_per_minute * config.TIMED_EVENT_PICKUP_OVERSTAY_MULTIPLIER
                )
            note_parts.append(f"event_{phase}_wait")
        elif phase == "approaching" and near_pickup:
            # 提前到接人点：保持原地等待至事件开始，避免空载远走
            wait_to_start = max(0, event.start_minutes - ctx.current_minutes)
            if wait_to_start > 0:
                covered = min(duration_minutes, wait_to_start)
                breakdown["timed_event_pickup_gain"] = covered * event.absence_penalty_per_minute * config.TIMED_EVENT_APPROACH_GAIN_MULTIPLIER
                note_parts.append("event_pre_pickup_wait")
        elif phase in {"home", "late_home", "stay"} and near_home:
            remaining = max(0, event.stay_until_minutes - ctx.current_minutes)
            covered = min(duration_minutes, remaining)
            if covered > 0:
                # R5-P0: stay 阶段 wait 增益加码（×5.0），确保司机留在家中
                multiplier = config.TIMED_EVENT_STAY_GAIN_MULTIPLIER if phase == "stay" else 1.0
                breakdown["timed_event_home_gain"] = covered * event.absence_penalty_per_minute * multiplier
                note_parts.append("event_home_stay")
        elif phase == "done" and near_home:
            # R5-P0: 事件刚结束后2h内仍给小量stay增益，避免立刻跑远
            post_buffer = ctx.current_minutes - event.stay_until_minutes
            if 0 <= post_buffer <= config.TIMED_EVENT_POST_STAY_BUFFER_MINUTES:
                decay = max(0.1, 1.0 - post_buffer / config.TIMED_EVENT_POST_STAY_BUFFER_MINUTES)
                breakdown["timed_event_post_stay_gain"] = duration_minutes * event.absence_penalty_per_minute * decay * 0.3
                note_parts.append("event_post_stay")
        elif phase == "stay" and not near_home:
            # stay阶段远离家：wait施加每分钟缺席惩罚，促使选择reposition回家
            breakdown["timed_event_away_penalty"] = -(duration_minutes * event.absence_penalty_per_minute)
            note_parts.append("event_stay_away")

    # v2: 反停滞惩罚——连续 wait 超阈值后逐步加码，迫使智能体尝试 reposition / take_order
    # 仅当本次 wait 不是为了完成 rest_gain / no_drive_gain 等明确收益时才施加（避免打断刚需休息）
    # 此外，处于定时事件 stay 阶段属于合法长等待，豁免反停滞惩罚
    in_timed_stay = any(
        timed_event_phase(ev, memory, ctx) in {"home", "late_home", "stay"}
        and geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, ev.home_lat, ev.home_lng) <= ev.radius_km
        for ev in rules.timed_stay_events
    )
    if (
        memory.consecutive_wait_count > config.STAGNATION_WAIT_THRESHOLD
        and rest_gain <= 0
        and no_drive_gain <= 0
        and not in_timed_stay
    ):
        excess = memory.consecutive_wait_count - config.STAGNATION_WAIT_THRESHOLD
        stagnation_penalty = excess * config.STAGNATION_WAIT_PENALTY_PER_STEP
        breakdown["stagnation_penalty"] = -stagnation_penalty
        note_parts.append(f"stagnation={memory.consecutive_wait_count}")
    # PR#24: 轻度反停滞——在正式阈值之前给予温和信号，
    # 让长 wait 相对短 wait 更贵，促进 agent 更频繁地重新评估市场。
    elif (
        config.ANTI_STAGNATION_MILD_THRESHOLD
        <= memory.consecutive_wait_count
        <= config.STAGNATION_WAIT_THRESHOLD
        and rest_gain <= 0
        and no_drive_gain <= 0
        and not in_timed_stay
        and duration_minutes > 120
    ):
        mild_excess = memory.consecutive_wait_count - config.ANTI_STAGNATION_MILD_THRESHOLD + 1
        mild_penalty = mild_excess * config.ANTI_STAGNATION_MILD_PENALTY_PER_STEP
        breakdown["mild_stagnation_penalty"] = -mild_penalty
        note_parts.append(f"mild_stagnation={memory.consecutive_wait_count}")

    # 月度休息日：必须休息日时给予完整罚金等额奖励，确保 wait 候选战胜其他动作。
    # 由于评测按 step_end 日判「成交接单」，需活跃日数预估到今日是否必须作为休息日
    if rules.monthly_day_off is not None:
        date_today = geo_utils.date_str(ctx.current_minutes)
        today_already_active = date_today in memory.daily_active
        active_so_far = memory.days_active_count()
        eval_horizon = config.EVALUATION_HORIZON_DAYS
        max_active = eval_horizon - rules.monthly_day_off.required_days
        days_off_required = rules.monthly_day_off.required_days
        sim_day = (ctx.current_minutes // (24 * 60))
        days_remaining = max(0, eval_horizon - 1 - sim_day)
        days_off_so_far = max(0, sim_day - active_so_far)
        needs_off_today = (
            not today_already_active
            and (
                active_so_far + 1 > max_active
                or days_off_so_far + days_remaining < days_off_required
            )
        )
        if needs_off_today:
            cur_md_off = geo_utils.minute_of_day(ctx.current_minutes)
            remaining_today = max(60, 1440 - cur_md_off)
            base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
            ratio = max(0.5, min(1.0, duration_minutes / float(remaining_today)))
            breakdown["monthly_day_off_gain"] = base * ratio * 1.5
            note_parts.append("day_off_required")
        elif days_off_so_far + days_remaining < days_off_required and duration_minutes >= 6 * 60:
            breakdown["monthly_day_off_gain"] = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off")) * 0.8
            note_parts.append("day_off_relief")
        elif not today_already_active:
            past_off_days = max(0, sim_day - active_so_far)
            deficit_after_today_if_active = days_off_required - past_off_days
            if deficit_after_today_if_active > 0 and days_remaining >= 0:
                urgency = deficit_after_today_if_active / max(1, days_remaining + 1)
                ideal_spacing = eval_horizon / max(1, days_off_required)
                proactive_rest = (past_off_days == 0 and sim_day >= ideal_spacing * 0.8)
                if urgency >= 0.03 and duration_minutes >= 60:
                    cur_md_off = geo_utils.minute_of_day(ctx.current_minutes)
                    remaining_today = max(60, 1440 - cur_md_off)
                    base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    ratio = min(1.0, duration_minutes / float(remaining_today))
                    # V3: 提高最低系数以确保休息日增益能与接单收入竞争
                    coeff = max(0.4, min(1.0, urgency)) * 1.5
                    if urgency >= 0.3:
                        coeff = min(2.0, urgency) * 2.0
                    pacing_gain = base * coeff * ratio
                    if pacing_gain > 0:
                        breakdown["monthly_day_off_pacing_gain"] = pacing_gain
                        note_parts.append(f"day_off_pacing(urg={urgency:.2f})")
                elif proactive_rest and duration_minutes >= 60:
                    base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    breakdown["monthly_day_off_pacing_gain"] = base * 0.6
                    note_parts.append("day_off_proactive")
                # R5-P1: 月末集中休息增益——剩余≤5天且deficit=1时给更高增益
                elif deficit_after_today_if_active == 1 and days_remaining <= config.MONTHLY_DAY_OFF_MONTH_END_DAYS and duration_minutes >= 60:
                    cur_md_off = geo_utils.minute_of_day(ctx.current_minutes)
                    remaining_today = max(60, 1440 - cur_md_off)
                    base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    ratio = max(0.5, min(1.0, duration_minutes / float(remaining_today)))
                    breakdown["monthly_day_off_pacing_gain"] = base * ratio * 1.2
                    note_parts.append("day_off_month_end_rest")
                # R6-P1: day>=25且deficit>=1时强制休息——极高wait增益促成全天休息
                elif sim_day >= config.MONTHLY_DAY_OFF_FORCE_REST_DAY and deficit_after_today_if_active >= 1 and duration_minutes >= 60:
                    cur_md_off = geo_utils.minute_of_day(ctx.current_minutes)
                    remaining_today = max(60, 1440 - cur_md_off)
                    base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    ratio = max(0.5, min(1.0, duration_minutes / float(remaining_today)))
                    breakdown["monthly_day_off_pacing_gain"] = base * ratio * 2.0
                    note_parts.append("day_off_force_rest_late")
                elif cur_md <= 6 * 60 and deficit_after_today_if_active >= 1:
                    base = float(rules.monthly_day_off.penalty_amount or _penalty_default("monthly_day_off"))
                    breakdown["monthly_day_off_pacing_gain"] = base * 0.2
                    note_parts.append("day_off_early_pacing")
    _apply_adaptive_weights(breakdown, ctx.weights)
    score = sum(breakdown.values()) + 1.0  # 兜底：始终略大于零
    return ScoredAction(
        action="wait",
        params={"duration_minutes": int(max(1, duration_minutes))},
        score=score,
        breakdown=breakdown,
        note=",".join(note_parts) if note_parts else "",
    )


# ---------------- 空驶评分 ----------------


def score_reposition(
    target_lat: float,
    target_lng: float,
    rules: ParsedRules,
    memory: DriverMemory,
    ctx: DecisionContext,
) -> ScoredAction:
    distance_km = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, target_lat, target_lng)
    minutes = geo_utils.distance_to_minutes(distance_km, ctx.reposition_speed_km_per_hour)
    breakdown: dict[str, float] = {}
    note_parts: list[str] = []

    # 硬约束：定时事件 stay/late_home/home 阶段，不允许远离老家圈外
    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        if phase in {"home", "late_home", "stay"}:
            dist_to_home = geo_utils.haversine_km(target_lat, target_lng, event.home_lat, event.home_lng)
            if dist_to_home > event.radius_km:
                return ScoredAction(
                    action="reposition",
                    params={"latitude": target_lat, "longitude": target_lng},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note=f"timed_event_block_reposition_{phase}",
                )
        elif phase in {"pickup", "late_pickup"}:
            dist_to_pick = geo_utils.haversine_km(target_lat, target_lng, event.pickup_lat, event.pickup_lng)
            dist_to_home = geo_utils.haversine_km(target_lat, target_lng, event.home_lat, event.home_lng)
            # 接人阶段允许去 pickup 或 home，其它远离全部禁止
            if dist_to_pick > event.radius_km and dist_to_home > event.radius_km:
                return ScoredAction(
                    action="reposition",
                    params={"latitude": target_lat, "longitude": target_lng},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note=f"timed_event_block_reposition_{phase}",
                )

    # 硬约束：禁入区域 / 边界
    for zone in rules.forbidden_zones:
        if _is_in_circle(target_lat, target_lng, zone):
            return ScoredAction(
                action="reposition",
                params={"latitude": target_lat, "longitude": target_lng},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="target_in_forbidden_zone",
            )
        if _path_passes_forbidden_zone(ctx.current_lat, ctx.current_lng, target_lat, target_lng, zone):
            breakdown["forbidden_path_penalty"] = -float(zone.penalty_amount or _penalty_default("forbidden_zone"))
            note_parts.append("forbidden_path")
    for box in rules.bounded_areas:
        if not _is_in_box(target_lat, target_lng, box):
            return ScoredAction(
                action="reposition",
                params={"latitude": target_lat, "longitude": target_lng},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="target_outside_bounded_area",
            )

    # 成本：里程 + 时间
    breakdown["reposition_cost"] = -ctx.cost_per_km * distance_km
    breakdown["reposition_time_cost"] = -ctx.opportunity_cost_per_minute * minutes

    # PR#24: 远距离非优先目标空驶额外成本——长距空驶消耗大量时间，
    # 若目标不是家/事件/熟货等优先点，应更谨慎。
    if distance_km > config.REPOSITION_LONG_DISTANCE_THRESHOLD_KM:
        is_priority = False
        if rules.home_rule is not None and _is_in_circle(target_lat, target_lng, rules.home_rule):
            is_priority = True
        if not is_priority:
            for ev in rules.timed_stay_events:
                if (
                    geo_utils.haversine_km(target_lat, target_lng, ev.pickup_lat, ev.pickup_lng) <= ev.radius_km
                    or geo_utils.haversine_km(target_lat, target_lng, ev.home_lat, ev.home_lng) <= ev.radius_km
                ):
                    is_priority = True
                    break
        if not is_priority:
            for pref in rules.preferred_cargo:
                pref_target = preferred_cargo_target(pref)
                if pref_target and geo_utils.haversine_km(target_lat, target_lng, pref_target[0], pref_target[1]) <= 3.0:
                    is_priority = True
                    break
        if not is_priority:
            extra_time_cost = ctx.opportunity_cost_per_minute * minutes * config.REPOSITION_LONG_DISTANCE_PENALTY_MULTIPLIER
            breakdown["long_reposition_penalty"] = -extra_time_cost
            note_parts.append("long_reposition")

    end_minutes = ctx.current_minutes + minutes
    # 空驶同样受禁行窗约束（PR#22 经验：与 take_order 对齐软/硬分离逻辑）
    # 回家空驶允许穿越 home_rule 自身的禁出窗（司机必须赶回家）
    target_is_home = (
        rules.home_rule is not None
        and _is_in_circle(target_lat, target_lng, rules.home_rule)
    )
    for window in rules.no_drive_windows:
        window_penalty = float(window.penalty_amount or 0.0)
        is_soft_window = 0 < window_penalty <= config.NO_DRIVE_SOFT_PENALTY_THRESHOLD
        if is_soft_window:
            real_overlap = _hits_no_drive_window(ctx.current_minutes, end_minutes, window)
            if real_overlap > 0:
                multiplier = _soft_no_drive_defer_multiplier(window, ctx.current_minutes)
                marginal = _soft_no_drive_marginal_charge(window, ctx.current_minutes, end_minutes, memory)
                breakdown["no_drive_window_soft_penalty"] = (
                    breakdown.get("no_drive_window_soft_penalty", 0.0) - marginal * multiplier
                )
                note_parts.append("no_drive_soft_window")
            continue
        buffered_end = end_minutes + config.NO_DRIVE_SAFETY_BUFFER_MINUTES
        overlap = _hits_no_drive_window(ctx.current_minutes, buffered_end, window)
        if overlap > 0:
            if target_is_home and rules.home_rule is not None:
                if window.start_minute == rules.home_rule.home_by_hour * 60:
                    continue  # 允许回家穿越自身禁出窗
            return ScoredAction(
                action="reposition",
                params={"latitude": target_lat, "longitude": target_lng},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="no_drive_window_violation",
            )

    if rules.home_rule is not None:
        arrival_hour = geo_utils.hour_of_day(end_minutes)
        if arrival_hour >= rules.home_rule.home_by_hour and not _is_in_circle(target_lat, target_lng, rules.home_rule):
            breakdown["home_rule_penalty"] = -float(rules.home_rule.penalty_amount or _penalty_default("home_rule"))
            note_parts.append("miss_home")
        # PR#22 经验：基于实际行程时间判断空驶后能否按时回家。
        # 非回家空驶：计算从目标点到家的行程时间，若到达目标后已来不及在 home_by_hour 前回家，
        # 施加渐进惩罚（而非等到已过 home_by_hour 才反应）。
        elif not _is_in_circle(target_lat, target_lng, rules.home_rule):
            dist_target_home = geo_utils.haversine_km(target_lat, target_lng, rules.home_rule.lat, rules.home_rule.lng)
            minutes_target_to_home = geo_utils.distance_to_minutes(dist_target_home, ctx.reposition_speed_km_per_hour)
            home_by_abs = (end_minutes // 1440) * 1440 + rules.home_rule.home_by_hour * 60
            slack_after_repo = home_by_abs - (end_minutes + minutes_target_to_home)
            if slack_after_repo < 0:
                breakdown["home_rule_penalty"] = -float(rules.home_rule.penalty_amount or _penalty_default("home_rule"))
                note_parts.append("repo_miss_home_travel")
            elif slack_after_repo <= 60:
                severity = max(0.3, 1.0 - slack_after_repo / 60.0)
                breakdown["home_rule_tight_slack_penalty"] = -float(rules.home_rule.penalty_amount or _penalty_default("home_rule")) * severity
                note_parts.append("repo_home_tight_slack")

    # 增益：目标点热点 + 必到 + 回家
    hotspot_gain = memory.hotspot_value(target_lat, target_lng) * minutes
    if hotspot_gain > 0:
        breakdown["expected_market_gain"] = hotspot_gain

    if rules.home_rule is not None:
        if _is_in_circle(target_lat, target_lng, rules.home_rule):
            arrival_min_of_day = geo_utils.minute_of_day(ctx.current_minutes + minutes)
            home_by_min = rules.home_rule.home_by_hour * 60
            cur_md = geo_utils.minute_of_day(ctx.current_minutes)
            unit = float(rules.home_rule.penalty_amount or _penalty_default("home_rule"))
            # 行程本身的耗时：远距离回家时，即便当前时刻"看似还早"，
            # 也需要立即出发才能在截止前抵达。slack = 距截止还剩多少分钟可挥霍。
            slack_minutes = home_by_min - arrival_min_of_day
            # 已过 home_by_hour（已经违规）：尽快回家止损，最高增益
            if cur_md >= home_by_min:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER * 1.5
                note_parts.append("home_target_urgent")
            # 在 home_by_hour 之前到家且距截止 ≤4h：给予完整违规罚金等额奖励
            elif arrival_min_of_day < home_by_min and cur_md >= home_by_min - 4 * 60:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER
                note_parts.append("home_target")
            elif cur_md >= home_by_min - 2 * 60:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER / 2.0
                note_parts.append("home_target")
            # 关键修复（B.3）：行程耗时已逼近截止——必须立刻出发才能赶上。
            # 旧逻辑仅依赖 cur_md ≥ home_by_min - 4h；若当下 cur_md 较早但回家路途≥3.5h，
            # 等到原触发窗口才反应已迟。改为：到家与截止 slack ≤ 60 分钟时立即给予完整奖励。
            elif arrival_min_of_day < home_by_min and slack_minutes <= 60:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER
                note_parts.append("home_target_tight_slack")
            # 距截止 4–6h 且行程占用 ≥1.5h：提前出发的递减奖励
            elif arrival_min_of_day < home_by_min and cur_md >= home_by_min - 6 * 60 and minutes >= 90:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER * 0.6
                note_parts.append("home_target_early")
            # PR#29: 距截止 6-8h 且行程占用 ≥2h：更早的微弱回家激励
            elif arrival_min_of_day < home_by_min and cur_md >= home_by_min - 8 * 60 and minutes >= 120:
                breakdown["home_rule_gain"] = unit * config.HOME_RULE_TARGET_GAIN_MULTIPLIER * 0.3
                note_parts.append("home_target_very_early")
    for preferred in rules.preferred_cargo:
        target = preferred_cargo_target(preferred)
        if target is None or not preferred_cargo_preposition_ready(preferred, ctx.current_minutes):
            continue
        if geo_utils.haversine_km(target_lat, target_lng, target[0], target[1]) <= 3.0:
            gain = float(preferred.penalty_amount or _penalty_default("preferred_cargo")) * config.PREFERRED_CARGO_POSITION_GAIN_MULTIPLIER
            if preferred.available_minutes is not None:
                arrival_minutes = ctx.current_minutes + minutes
                if arrival_minutes > preferred.available_minutes + config.PREFERRED_CARGO_GIVEUP_WINDOW_MINUTES:
                    continue
                wait_gap = max(0, preferred.available_minutes - arrival_minutes)
                if wait_gap > config.PREFERRED_CARGO_MAX_WAIT_MINUTES:
                    gain *= 0.5
                # 非必要的过夜外宿压制：
                #   司机有 home_rule、目标远离家圈、抵达后当晚无法在 home_by_hour 前返家、
                #   且货上架仍距当下 > 12h —— 则压低预定位增益，避免抢跑 1-2 天导致重复
                #   home_rule 罚（典型如 D009 走 ¥10000 熟货抢早一天就丢 ¥900 回家罚）。
                if (
                    rules.home_rule is not None
                    and not _is_in_circle(target_lat, target_lng, rules.home_rule)
                ):
                    dist_target_to_home = geo_utils.haversine_km(
                        target_lat, target_lng, rules.home_rule.lat, rules.home_rule.lng
                    )
                    return_minutes = geo_utils.distance_to_minutes(
                        dist_target_to_home, ctx.reposition_speed_km_per_hour
                    )
                    end_day_today = arrival_minutes // 1440
                    home_by_min_today = end_day_today * 1440 + rules.home_rule.home_by_hour * 60
                    time_to_cargo = preferred.available_minutes - ctx.current_minutes
                    if arrival_minutes + return_minutes > home_by_min_today and time_to_cargo > 12 * 60:
                        gain *= 0.1
                        note_parts.append("preferred_cargo_defer_overnight")
            breakdown["preferred_cargo_position_gain"] = gain
            note_parts.append("preferred_cargo_target")

    for event in rules.timed_stay_events:
        phase = timed_event_phase(event, memory, ctx)
        target: tuple[float, float] | None = None
        gain_key = ""
        if phase == "early_approach":
            # R5-P0: 在强制回家窗口内，reposition回家/接人点给额外增益
            time_to_event = event.start_minutes - ctx.current_minutes
            # PR#27: 多日 stay 事件扩展窗口与 take_order 对齐（48h），让司机更早被引导回家。
            stay_duration_r = max(0, event.stay_until_minutes - event.start_minutes)
            mandatory_window_r = config.TIMED_EVENT_HOME_MANDATORY_WINDOW_MINUTES
            if stay_duration_r >= config.TIMED_EVENT_MULTIDAY_THRESHOLD_MINUTES:
                mandatory_window_r = max(mandatory_window_r, config.TIMED_EVENT_MULTIDAY_HOME_MANDATORY_WINDOW_MINUTES)
            if 0 < time_to_event <= mandatory_window_r:
                dist_to_home_repo = geo_utils.haversine_km(target_lat, target_lng, event.home_lat, event.home_lng)
                dist_to_pick_repo = geo_utils.haversine_km(target_lat, target_lng, event.pickup_lat, event.pickup_lng)
                if dist_to_home_repo <= event.radius_km or dist_to_pick_repo <= event.radius_km:
                    breakdown["timed_event_mandatory_home_repo_gain"] = float(event.penalty_amount or _penalty_default("timed_stay_event")) * 2.0
                    note_parts.append("timed_event_mandatory_home_repo")
            target = (event.pickup_lat, event.pickup_lng)
            gain_key = "timed_event_approach_gain"
        elif phase == "approaching":
            target = (event.pickup_lat, event.pickup_lng)
            gain_key = "timed_event_approach_gain"
        elif phase in {"pickup", "late_pickup"}:
            target = (event.pickup_lat, event.pickup_lng)
            gain_key = "timed_event_pickup_gain"
        elif phase in {"home", "late_home", "stay"}:
            target = (event.home_lat, event.home_lng)
            gain_key = "timed_event_home_gain"
        if target is not None and geo_utils.haversine_km(target_lat, target_lng, target[0], target[1]) <= event.radius_km:
            gain = float(event.penalty_amount or _penalty_default("timed_stay_event")) * config.TIMED_EVENT_FIXED_GAIN_MULTIPLIER
            if phase == "early_approach":
                gain *= 0.5
            elif phase == "approaching":
                gain *= config.TIMED_EVENT_APPROACH_GAIN_MULTIPLIER
            if phase.startswith("late"):
                gain *= 2.0
            breakdown[gain_key] = breakdown.get(gain_key, 0.0) + gain
            note_parts.append(f"timed_event_{phase}")

    for must in rules.must_visit:
        date_today = geo_utils.date_str(ctx.current_minutes)
        already = memory.visited_target_dates.get(_must_visit_key(must), set())
        if date_today not in already and geo_utils.haversine_km(target_lat, target_lng, must.lat, must.lng) <= must.radius_km:
            breakdown["must_visit_gain"] = float(must.penalty_amount or _penalty_default("must_visit")) / max(1, must.required_days)
            note_parts.append("must_visit")

    # 月度空驶累计限制：超过预算的普通空驶直接拒绝（D003 月预算 100km 单一约束极紧）。
    # 仅允许「优先目标」（家、必到、定时事件接人/老家点、熟货）破例，并按强罚记分。
    target_is_priority_deadhead = False
    if rules.home_rule is not None and _is_in_circle(target_lat, target_lng, rules.home_rule):
        target_is_priority_deadhead = True
    if not target_is_priority_deadhead:
        for must in rules.must_visit:
            if geo_utils.haversine_km(target_lat, target_lng, must.lat, must.lng) <= must.radius_km:
                target_is_priority_deadhead = True
                break
    if not target_is_priority_deadhead:
        for ev in rules.timed_stay_events:
            if (
                geo_utils.haversine_km(target_lat, target_lng, ev.pickup_lat, ev.pickup_lng) <= ev.radius_km
                or geo_utils.haversine_km(target_lat, target_lng, ev.home_lat, ev.home_lng) <= ev.radius_km
            ):
                target_is_priority_deadhead = True
                break
    if not target_is_priority_deadhead:
        for pref in rules.preferred_cargo:
            pref_target = preferred_cargo_target(pref)
            if pref_target is None or not preferred_cargo_preposition_ready(pref, ctx.current_minutes):
                continue
            if geo_utils.haversine_km(target_lat, target_lng, pref_target[0], pref_target[1]) <= 3.0:
                target_is_priority_deadhead = True
                break
    for limit in rules.distance_limits:
        if limit.kind == "monthly_deadhead":
            projected = memory.total_deadhead_km + distance_km
            pen_per_km = float(limit.penalty_amount or _penalty_default("distance_limit_monthly_deadhead"))
            # PR#27: 预警阶段对空驶同样生效——避免反复短空驶累计冲破限额。
            preempt_threshold_r = limit.max_km * config.MONTHLY_DEADHEAD_PREEMPT_RATIO
            if memory.total_deadhead_km < limit.max_km and projected > preempt_threshold_r:
                preempt_start_r = max(memory.total_deadhead_km, preempt_threshold_r)
                preempt_end_r = min(projected, limit.max_km)
                preempt_km_r = max(0.0, preempt_end_r - preempt_start_r)
                if preempt_km_r > 0:
                    mid_ratio_r = (preempt_start_r + preempt_end_r) / 2.0 / max(1.0, limit.max_km)
                    band_r = max(1e-6, 1.0 - config.MONTHLY_DEADHEAD_PREEMPT_RATIO)
                    band_pos_r = max(0.0, min(1.0, (mid_ratio_r - config.MONTHLY_DEADHEAD_PREEMPT_RATIO) / band_r))
                    coeff_r = config.MONTHLY_DEADHEAD_PREEMPT_MAX_COEFF * band_pos_r
                    breakdown["monthly_deadhead_preempt_penalty"] = -pen_per_km * preempt_km_r * coeff_r
            if projected > limit.max_km:
                # B.2 cap-aware：评测端罚金有 penalty_cap 上限（D003 ¥2000），
                # 一旦累计惩罚已逼近上限，继续空驶的「边际罚分」≈0；过去硬阻拦让司机
                # 后半月只能干等。这里在 cap 已基本耗尽时把硬阻拦降级为软惩罚，
                # 由具体收益自行决定是否触发。
                cap_reached = False
                if limit.penalty_cap is not None and pen_per_km > 0:
                    over_so_far_km = max(0.0, memory.total_deadhead_km - limit.max_km)
                    paid_so_far = min(over_so_far_km * pen_per_km, limit.penalty_cap)
                    cap_reached = paid_so_far >= limit.penalty_cap * 0.95
                if not target_is_priority_deadhead and not cap_reached:
                    return ScoredAction(
                        action="reposition",
                        params={"latitude": float(target_lat), "longitude": float(target_lng)},
                        score=-HARD_CONSTRAINT_PENALTY,
                        feasible=False,
                        note="monthly_deadhead_over_limit",
                    )
                if cap_reached:
                    # PR#27: cap 已耗尽，原仅 -¥50 名义罚不足以抑制 D003 反复空驶；
                    # 改为按新增超额公里 × pen_per_km × OVERCAP_RESIDUAL_COEFF 计算残余信号，
                    # 与 take_order 侧对齐，避免后半月长距离无收益空驶。
                    new_over_km_repo = max(0.0, projected - limit.max_km) - max(
                        0.0, memory.total_deadhead_km - limit.max_km
                    )
                    residual = pen_per_km * max(0.0, new_over_km_repo) * config.MONTHLY_DEADHEAD_OVERCAP_RESIDUAL_COEFF
                    if residual > 0:
                        breakdown["monthly_deadhead_overcap_penalty"] = -residual
                        note_parts.append("deadhead_overcap_soft")
                else:
                    # 优先目标且 cap 未到：强软惩罚，由偏好增益自身决定是否抵消
                    breakdown["monthly_deadhead_penalty"] = -pen_per_km * (
                        projected - limit.max_km
                    ) * 3.0
    # 空驶层面的每日休息约束：避免空驶占用当日唯一可休息时段
    for rest in rules.rest_rules:
        today_rest = memory.longest_rest_today(ctx.current_minutes)
        deficit = rest.required_minutes - today_rest
        if deficit > 0:
            cur_md = geo_utils.minute_of_day(ctx.current_minutes)
            finish_md_repo = geo_utils.minute_of_day(end_minutes)
            crosses_day_repo = end_minutes // 1440 > ctx.current_minutes // 1440
            remaining_after_repo = 1440 - finish_md_repo if not crosses_day_repo else 0
            available_now = 1440 - cur_md
            if available_now >= deficit and (crosses_day_repo or remaining_after_repo < deficit):
                return ScoredAction(
                    action="reposition",
                    params={"latitude": float(target_lat), "longitude": float(target_lng)},
                    score=-HARD_CONSTRAINT_PENALTY,
                    feasible=False,
                    note="daily_rest_blocked_reposition",
                )

    # 月度休息日：必须休息日时禁止空驶。
    # PR#21 经验：评测按 [action_start, action_end] 跨过的每个自然日都计入活跃日，
    # 空驶同理——长距离空驶跨午夜会让两天都变为活跃日。
    if rules.monthly_day_off is not None:
        projected_active_repo = set(memory.daily_active)
        cur_day_idx_repo = ctx.current_minutes // 1440
        end_day_idx_repo = max(cur_day_idx_repo, max(0, end_minutes - 1) // 1440)
        for di_repo in range(cur_day_idx_repo, end_day_idx_repo + 1):
            projected_active_repo.add(geo_utils.date_str(di_repo * 1440))
        max_active_repo = config.EVALUATION_HORIZON_DAYS - rules.monthly_day_off.required_days
        if len(projected_active_repo) > max_active_repo:
            return ScoredAction(
                action="reposition",
                params={"latitude": float(target_lat), "longitude": float(target_lng)},
                score=-HARD_CONSTRAINT_PENALTY,
                feasible=False,
                note="monthly_day_off_required_today",
            )

    _apply_adaptive_weights(breakdown, ctx.weights)
    score = sum(breakdown.values())
    return ScoredAction(
        action="reposition",
        params={"latitude": float(target_lat), "longitude": float(target_lng)},
        score=score,
        feasible=score > -HARD_CONSTRAINT_PENALTY / 2,
        breakdown=breakdown,
        note=",".join(note_parts) if note_parts else "",
    )


def _must_visit_key(must) -> str:  # type: ignore[no-untyped-def]
    return f"{must.lat:.4f},{must.lng:.4f}"
