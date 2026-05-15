"""司机维度运行时记忆：偏好缓存、token 预算、当日统计、热点网格。

设计要点：
- 进程内通过 ``get_or_create`` 单例化每个司机的 ``DriverMemory``。
- 每步从 ``query_decision_history`` 读取近若干步动作并以 ``step`` 去重，避免重复累计。
- 热点网格使用经纬度 0.1 度粒度聚合，仅用于空驶目标评估。
"""

from __future__ import annotations

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

    # R7: LLM辅助决策计数器
    llm_critical_count: int = 0

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
        """返回查询点附近 9 宫格的平均“元/分钟”收益，作为未来机会估计。"""
        key = geo_utils.grid_key(latitude, longitude)
        total_yield = 0.0
        total_samples = 0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                cell = self.hotspots.get((key[0] + di, key[1] + dj))
                if cell is None or cell.samples == 0:
                    continue
                total_yield += cell.sum_price_per_minute
                total_samples += cell.samples
        if total_samples <= 0:
            return 0.0
        return total_yield / total_samples

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

        if action_name == "take_order":
            accepted = bool(result.get("accepted", False))
            self.cargo_attempt_count += 1
            # 任何 take_order 尝试都打断连续 wait 停滞计数
            self.consecutive_wait_count = 0
            if accepted:
                self.cargo_success_count += 1
                self.consecutive_failed_take_orders = 0
                self.daily_orders[date_today] += 1
                self.daily_active.add(date_today)
                self.total_completed_orders += 1
                self.total_haul_km += float(result.get("haul_distance_km", 0.0) or 0.0)
                self.total_deadhead_km += float(result.get("pickup_deadhead_km", 0.0) or 0.0)
                if date_today not in self.daily_first_take_minute_of_day:
                    minute_at_start = max(0, sim_minutes_after - action_exec)
                    self.daily_first_take_minute_of_day[date_today] = geo_utils.minute_of_day(minute_at_start)
            else:
                # 接单失败（cargo_id 已失效等）：累计连续失败供 take_order 评分避让
                self.consecutive_failed_take_orders += 1
            self.pending_rest_streak_minutes = 0
            self.pending_rest_streak_date = date_today
        elif action_name == "reposition":
            self.daily_active.add(date_today)
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
