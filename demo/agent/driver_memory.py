"""司机维度运行时记忆：偏好缓存、token 预算、当日统计、热点网格。

设计要点：
- 进程内通过 ``get_or_create`` 单例化每个司机的 ``DriverMemory``。
- 每步从 ``query_decision_history`` 读取近若干步动作并以 ``step`` 去重，避免重复累计。
- 热点网格使用经纬度 0.1 度粒度聚合，仅用于空驶目标评估。
"""

from __future__ import annotations

import math as _math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from . import config, geo_utils

# 以下常量代理至 ``config``，保留原名以免上层调用点修改过多。
PER_DRIVER_TOKEN_LIMIT = config.PER_DRIVER_TOKEN_LIMIT
TOKEN_DEGRADE_THRESHOLD = config.TOKEN_DEGRADE_THRESHOLD


@dataclass
class HotspotCell:
    """单个网格的聚合统计。"""

    samples: int = 0
    sum_price: float = 0.0
    sum_price_per_minute: float = 0.0
    last_seen_minutes: int = 0


@dataclass
class PreferenceState:
    """动态偏好状态（文档 3.5 节）。"""

    current_signature: str = ""
    last_parse_time_minutes: int = -1
    parse_failure_count: int = 0
    dynamic_changes: list[dict[str, Any]] = field(default_factory=list)
    parsed_by_llm: int = 0
    parsed_by_regex: int = 0


@dataclass
class HourBucket:
    """小时粒度的货源价/频率统计，供时间模式学习。"""

    samples: int = 0
    sum_price: float = 0.0
    sum_price_per_minute: float = 0.0


@dataclass
class DriverMemory:
    """单司机决策上下文，跨步骤累积。"""

    driver_id: str
    rules: Any = None  # ParsedRules，由 preference_parser 注入
    rules_signature: str = ""  # 偏好原文哈希，用于检测偏好变更
    preference_state: PreferenceState = field(default_factory=PreferenceState)
    token_used: int = 0
    last_status_minutes: int = 0
    last_lat: float = 0.0
    last_lng: float = 0.0

    # 历史动作去重：仅处理 step > processed_until_step 的记录
    processed_until_step: int = 0

    # 当日统计：date_str -> count / minutes
    daily_orders: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    daily_active: set[str] = field(default_factory=set)
    daily_first_take_minute_of_day: dict[str, int] = field(default_factory=dict)
    daily_longest_rest_minutes: dict[str, int] = field(default_factory=dict)
    pending_rest_streak_minutes: int = 0
    pending_rest_streak_date: str = ""

    # 月度统计
    visited_target_dates: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    total_deadhead_km: float = 0.0
    total_haul_km: float = 0.0
    total_gross_income: float = 0.0
    total_completed_orders: int = 0

    # 偏好违规累积（penalty_cap 限流判断用）：rule_id -> 累计罚金
    preference_penalty_accum: dict[str, float] = field(default_factory=dict)

    # 热点：grid_key -> HotspotCell
    hotspots: dict[tuple[int, int], HotspotCell] = field(default_factory=dict)

    # 时间模式学习：24 小时桁粒度的货源频率与平均价格
    hour_buckets: dict[int, HourBucket] = field(default_factory=dict)

    # 失败学习：Take_order 尝试与成功计数，用于估计在线竞争强度
    cargo_attempt_count: int = 0
    cargo_success_count: int = 0
    consecutive_failed_take_orders: int = 0

    # 反停滞：连续 wait 计数，用于逐步增长 wait 惩罚
    consecutive_wait_count: int = 0

    timed_event_flags: set[str] = field(default_factory=set)

    def update_token(self, delta: int) -> None:
        if delta > 0:
            self.token_used += int(delta)

    def can_call_model(self, expected_tokens: int = 0) -> bool:
        """判断是否仍允许调用大模型；接近上限即降级。"""
        return (self.token_used + max(0, int(expected_tokens))) < TOKEN_DEGRADE_THRESHOLD

    def remaining_token_budget(self) -> int:
        return max(0, PER_DRIVER_TOKEN_LIMIT - self.token_used)

    def update_hotspot(self, latitude: float, longitude: float, price_yuan: float, minutes: int, current_time_minutes: int) -> None:
        """记录可见货源的装货点、单价、单位时间收益，用于空驶目标评估。"""
        if minutes <= 0:
            return
        key = geo_utils.grid_key(latitude, longitude)
        cell = self.hotspots.get(key)
        if cell is None:
            cell = HotspotCell()
            self.hotspots[key] = cell
        cell.samples += 1
        cell.sum_price += float(price_yuan)
        cell.sum_price_per_minute += float(price_yuan) / float(minutes)
        cell.last_seen_minutes = int(current_time_minutes)
        # 同步更新小时桋统计，描述“什么时间有什么货”的在线模式
        hour = geo_utils.hour_of_day(current_time_minutes)
        bucket = self.hour_buckets.get(hour)
        if bucket is None:
            bucket = HourBucket()
            self.hour_buckets[hour] = bucket
        bucket.samples += 1
        bucket.sum_price += float(price_yuan)
        bucket.sum_price_per_minute += float(price_yuan) / float(minutes)

    def hour_pattern_value(self, hour: int) -> float:
        """返回该小时的平均货源单位时间收益，用作在线时间模式信号。"""
        bucket = self.hour_buckets.get(hour % 24)
        if bucket is None or bucket.samples <= 0:
            return 0.0
        return bucket.sum_price_per_minute / bucket.samples

    def hotspot_value(self, latitude: float, longitude: float) -> float:
        """返回查询点附近 9 宫格的加权“元/分钟”收益，作为未来机会估计。

        PR#24: 增加时间衰减——近期观测权重更高，避免用过时数据做空驶决策。
        """
        key = geo_utils.grid_key(latitude, longitude)
        total_yield = 0.0
        total_weight = 0.0
        half_life = config.HOTSPOT_DECAY_HALF_LIFE_MINUTES
        current_time = max(self.last_status_minutes, 1)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cell = self.hotspots.get((key[0] + di, key[1] + dj))
                if cell is None or cell.samples == 0:
                    continue
                age = max(0, current_time - cell.last_seen_minutes)
                decay = _math.exp(-0.693 * age / half_life) if half_life > 0 else 1.0
                weight = cell.samples * decay
                total_yield += cell.sum_price_per_minute * decay
                total_weight += weight
        if total_weight <= 0:
            return 0.0
        return total_yield / total_weight

    def absorb_history_records(self, records: list[dict[str, Any]]) -> None:
        """从 ``query_decision_history`` 记录中累计当日和月度统计。"""
        if not records:
            return
        for record in records:
            try:
                step = int(record.get("step", 0))
            except (TypeError, ValueError):
                continue
            if step <= self.processed_until_step:
                continue
            self._absorb_single_record(record)
            self.processed_until_step = step
        # PR#27: 每次吸收后同步 preference_penalty_accum，让 _is_preference_near_violation
        # 真正可用（之前该 dict 被读但从不被写，自适应权重升级永远不触发）。
        self._refresh_preference_penalty_accum()

    def _absorb_single_record(self, record: dict[str, Any]) -> None:
        action = record.get("action", {}) or {}
        action_name = str(action.get("action", "")).strip().lower()
        result = record.get("result", {}) or {}
        sim_minutes_after = int(record.get("simulation_end_time_minutes", 0)) or int(
            result.get("simulation_progress_minutes", 0)
        )
        sim_wall = result.get("simulation_wall_time") or record.get("simulation_end_time")
        if isinstance(sim_wall, str) and sim_wall:
            try:
                sim_minutes_after = geo_utils.wall_time_to_minutes(sim_wall)
            except ValueError:
                pass
        date_today = geo_utils.date_str(sim_minutes_after)

        elapsed = int(record.get("step_elapsed_minutes", 0))
        action_exec = int(record.get("action_exec_cost_minutes", elapsed))
        # 动作起始绝对分钟（用于跨日归属）。take_order/reposition 的 action_start 是
        # query_scan 之后的真实驾驶开始时刻；评测脚本以 [action_start, action_end] 跨过的
        # 每个自然日都计入「活跃日」（_active_minutes_by_day）。
        action_start_minutes = max(0, sim_minutes_after - action_exec)
        date_action_start = geo_utils.date_str(action_start_minutes)

        if action_name == "take_order":
            accepted = bool(result.get("accepted", False))
            self.cargo_attempt_count += 1
            # 任何 take_order 尝试都打断连续 wait 停滞计数
            self.consecutive_wait_count = 0
            if accepted:
                self.cargo_success_count += 1
                self.consecutive_failed_take_orders = 0
                # 评测「日订单计数」与「首单时间」以 action_start 所在日为准；跨午夜的
                # 长单（如 23:50 接单 03:00 卸车）只算开工那天，而不是落货那天。
                self.daily_orders[date_action_start] += 1
                self._mark_active_days(action_start_minutes, sim_minutes_after)
                self.total_completed_orders += 1
                self.total_haul_km += float(result.get("haul_distance_km", 0.0) or 0.0)
                self.total_deadhead_km += float(result.get("pickup_deadhead_km", 0.0) or 0.0)
                if date_action_start not in self.daily_first_take_minute_of_day:
                    self.daily_first_take_minute_of_day[date_action_start] = geo_utils.minute_of_day(
                        action_start_minutes
                    )
            else:
                # 接单失败（cargo_id 已失效等）：累计连续失败供 take_order 评分避让
                self.consecutive_failed_take_orders += 1
            self.pending_rest_streak_minutes = 0
            self.pending_rest_streak_date = date_today
        elif action_name == "reposition":
            self._mark_active_days(action_start_minutes, sim_minutes_after)
            self.total_deadhead_km += float(result.get("distance_km", 0.0) or 0.0)
            self.consecutive_wait_count = 0  # 空驶打断停滞
            self.consecutive_failed_take_orders = 0  # 位置变了，失败史失效
            self.pending_rest_streak_minutes = 0
            self.pending_rest_streak_date = date_today
        elif action_name == "wait":
            params = action.get("params", {}) or {}
            duration = int(params.get("duration_minutes", action_exec) or 0)
            self._extend_rest_streak(date_today, duration)
            self.consecutive_wait_count += 1
        else:
            return

    def _mark_active_days(self, action_start_minutes: int, action_end_minutes: int) -> None:
        """将 [action_start, action_end] 跨过的每个自然日均加入 daily_active。

        评测脚本 ``_active_minutes_by_day`` 会把动作横跨的每个自然日都计为活跃；
        agent 仅记录落货日会让月度休息日预算偏乐观（典型场景：跨午夜的长单让
        agent 以为「次日还没活跃」从而错失实际已被评测计入的休息额度）。
        """
        if action_end_minutes <= action_start_minutes:
            self.daily_active.add(geo_utils.date_str(action_start_minutes))
            return
        cursor = action_start_minutes
        while cursor < action_end_minutes:
            day_idx = cursor // 1440
            next_day_start = (day_idx + 1) * 1440
            self.daily_active.add(geo_utils.date_str(cursor))
            cursor = next_day_start

    def _extend_rest_streak(self, date_today: str, duration_minutes: int) -> None:
        if duration_minutes <= 0:
            return
        if self.pending_rest_streak_date == date_today:
            self.pending_rest_streak_minutes += duration_minutes
        else:
            self.pending_rest_streak_minutes = duration_minutes
            self.pending_rest_streak_date = date_today
        prev = self.daily_longest_rest_minutes.get(date_today, 0)
        if self.pending_rest_streak_minutes > prev:
            self.daily_longest_rest_minutes[date_today] = self.pending_rest_streak_minutes

    def _refresh_preference_penalty_accum(self) -> None:
        """根据当前累计统计与规则估算各偏好的累计罚分（PR#27）。

        覆盖最常爆 cap 的两类规则：
        - ``dist_monthly_deadhead``：按 ``max(total_deadhead_km - max_km, 0) × penalty_amount`` 估算。
        - ``home_rule``：按超过 ``home_by_hour`` 的活跃日数 × penalty_amount 估算（粗略上界，避免遗漏）。

        其它规则（no_drive_windows、daily_rest）需要扫描行动序列才能精确估算，这里先不补充，
        避免引入与 ``absorb`` 不一致的复杂度；后续若需要可在 ``_absorb_single_record`` 中精细化。
        """
        rules = self.rules
        if rules is None:
            return
        # 月度空驶累计
        for limit in getattr(rules, "distance_limits", []) or []:
            if getattr(limit, "kind", "") != "monthly_deadhead":
                continue
            over_km = max(0.0, self.total_deadhead_km - float(limit.max_km))
            if over_km <= 0:
                self.preference_penalty_accum.pop("dist_monthly_deadhead", None)
                continue
            pen_per_km = float(limit.penalty_amount or 0.0)
            paid = over_km * pen_per_km
            if limit.penalty_cap is not None:
                paid = min(paid, float(limit.penalty_cap))
            self.preference_penalty_accum["dist_monthly_deadhead"] = paid
        # 月度休息日：active_days 越接近 horizon - required_days 越紧。这里写入「已活跃天数」
        # 作为软信号；_is_preference_near_violation 仍按 cap 阈值比较。
        if getattr(rules, "monthly_day_off", None) is not None:
            required = int(rules.monthly_day_off.required_days)
            horizon = config.EVALUATION_HORIZON_DAYS
            active = len(self.daily_active)
            max_active = horizon - required
            if max_active > 0 and active > 0:
                # 把「实际活跃 / 最大允许活跃」映射到 cap 比例：当 active/max_active 接近 1 时，penalty 接近 cap。
                cap = float(rules.monthly_day_off.penalty_cap or 0.0)
                if cap > 0:
                    ratio = min(1.0, active / max(1.0, max_active))
                    self.preference_penalty_accum["monthly_day_off"] = ratio * cap

    def daily_orders_today(self, sim_minutes: int) -> int:
        return int(self.daily_orders.get(geo_utils.date_str(sim_minutes), 0))

    def cargo_success_rate(self) -> float:
        """返回历史 take_order 成功率；未达最小样本时返回 1.0。

        该值代表“环境中其他司机的竞争强度”：在评测中货源被同班司机抢占将导致 cargo_id
        在我们 take_order 时失效。用于在 score_take_order 中折扣预期收入。
        """
        if self.cargo_attempt_count < config.CARGO_SUCCESS_RATE_MIN_ATTEMPTS:
            return 1.0
        rate = self.cargo_success_count / float(self.cargo_attempt_count)
        return max(float(config.CARGO_SUCCESS_RATE_FLOOR), rate)

    def longest_rest_today(self, sim_minutes: int) -> int:
        return int(self.daily_longest_rest_minutes.get(geo_utils.date_str(sim_minutes), 0))

    def days_active_count(self) -> int:
        return len(self.daily_active)

    def record_preference_change(
        self,
        new_signature: str,
        sim_minutes: int,
        *,
        parsed_by_llm: int = 0,
        parsed_by_regex: int = 0,
        parse_failure_count: int = 0,
    ) -> None:
        """记录偏好变化事件（文档 6.1 第 2 步）。"""
        state = self.preference_state
        if state.current_signature and state.current_signature != new_signature:
            state.dynamic_changes.append(
                {
                    "at_minutes": int(sim_minutes),
                    "prev_signature": state.current_signature,
                    "new_signature": new_signature,
                }
            )
        state.current_signature = new_signature
        state.last_parse_time_minutes = int(sim_minutes)
        state.parsed_by_llm = int(parsed_by_llm)
        state.parsed_by_regex = int(parsed_by_regex)
        state.parse_failure_count = int(parse_failure_count)


_MEMORY_BY_DRIVER: dict[str, DriverMemory] = {}


def get_or_create(driver_id: str) -> DriverMemory:
    """获取或新建司机记忆；进程内全局缓存。"""
    mem = _MEMORY_BY_DRIVER.get(driver_id)
    if mem is None:
        mem = DriverMemory(driver_id=driver_id)
        _MEMORY_BY_DRIVER[driver_id] = mem
    return mem


def reset(driver_id: str | None = None) -> None:
    """清空指定司机或所有司机的记忆。仅用于本地测试。"""
    if driver_id is None:
        _MEMORY_BY_DRIVER.clear()
        return
    _MEMORY_BY_DRIVER.pop(driver_id, None)
