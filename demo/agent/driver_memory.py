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
class CompletedOrderRecord:
    """已完成订单的摘要记录，用于订单链评估（方案一）。"""

    end_lat: float
    end_lng: float
    income: float
    occupied_minutes: float
    completed_at_minutes: int


@dataclass
class LocationTimeStats:
    """某区域某时段的订单统计，用于在线学习（方案五）。"""

    grid_key: str
    hour_bucket: int
    total_income: float = 0.0
    total_orders: int = 0
    total_minutes: float = 0.0

    @property
    def avg_rate(self) -> float:
        """平均元/分钟。"""
        return self.total_income / max(1.0, self.total_minutes)


@dataclass
class HotspotCell:
    """单个网格的聚合统计。"""

    samples: int = 0
    sum_price: float = 0.0
    sum_price_per_minute: float = 0.0
    last_seen_minutes: int = 0
    sum_income: float = 0.0
    income_samples: int = 0


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


def _project_no_drive_window(day_start: int, start_minute: int, end_minute: int) -> list[tuple[int, int]]:
    if end_minute <= 24 * 60:
        return [(day_start + start_minute, day_start + end_minute)]
    return [
        (day_start + start_minute, day_start + 24 * 60),
        (day_start + 24 * 60, day_start + 24 * 60 + (end_minute - 24 * 60)),
    ]


def _no_drive_window_hit_dates(action_start_minutes: int, action_end_minutes: int, window: Any) -> set[str]:
    if action_end_minutes <= action_start_minutes:
        action_end_minutes = action_start_minutes + 1
    hit_dates: set[str] = set()
    first_day = max(0, action_start_minutes // 1440 - 1)
    last_day = max(first_day, action_end_minutes // 1440 + 1)
    for day_idx in range(first_day, last_day + 1):
        day_start = day_idx * 1440
        for ws, we in _project_no_drive_window(day_start, int(window.start_minute), int(window.end_minute)):
            if min(action_end_minutes, we) > max(action_start_minutes, ws):
                hit_dates.add(geo_utils.date_str(day_start))
    return hit_dates


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
    preference_violation_days: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

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

    # 方案一：已完成订单记录（订单链评估）
    completed_orders: list[CompletedOrderRecord] = field(default_factory=list)

    # PR#32 突破 #3：决策时即时累计（不依赖 result.income，规避仿真框架不回传 income 的限制）
    committed_income_total: float = 0.0
    committed_occupied_minutes_total: float = 0.0
    committed_take_count: int = 0
    committed_seen_cargo_ids: set[str] = field(default_factory=set)

    # 方案五：在线学习——区域×时段统计
    location_time_stats: dict[str, LocationTimeStats] = field(default_factory=dict)

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

    def record_completed_order(
        self,
        end_lat: float,
        end_lng: float,
        income: float,
        occupied_minutes: float,
        completed_at_minutes: int,
    ) -> None:
        """记录已完成订单摘要，供订单链评估使用（方案一）。"""
        self.completed_orders.append(
            CompletedOrderRecord(
                end_lat=end_lat,
                end_lng=end_lng,
                income=income,
                occupied_minutes=occupied_minutes,
                completed_at_minutes=completed_at_minutes,
            )
        )

    def estimate_location_value(self, lat: float, lng: float) -> float:
        """基于历史完单记录估算某位置的后续接单价值（方案一）。

        在 completed_orders 中查找 CHAIN_VALUE_RADIUS_KM 范围内的历史卸货点，
        以距离加权平均估算该位置的收益水平。
        """
        if len(self.completed_orders) < config.CHAIN_VALUE_MIN_RECORDS:
            return 0.0
        radius = config.CHAIN_VALUE_RADIUS_KM
        total_value = 0.0
        total_weight = 0.0
        for rec in self.completed_orders:
            dist = geo_utils.haversine_km(lat, lng, rec.end_lat, rec.end_lng)
            if dist < radius:
                w = max(0.0, 1.0 - dist / radius)
                total_value += rec.income * w
                total_weight += w
        if total_weight <= 0:
            return 0.0
        return total_value / total_weight

    def count_nearby_completed_orders(
        self, lat: float, lng: float, radius_km: float | None = None
    ) -> int:
        """统计指定位置 radius_km 内的历史完成订单数（PR#32 突破 #1：陌生区域判定）。"""
        radius = radius_km if radius_km is not None else config.CHAIN_VALUE_RADIUS_KM
        n = 0
        for rec in self.completed_orders:
            if geo_utils.haversine_km(lat, lng, rec.end_lat, rec.end_lng) < radius:
                n += 1
        return n

    def personal_rate_per_minute(self) -> float:
        """估算司机本月已实现的收益率（元/分钟）。

        优先使用 completed_orders（含 result.income）；当仿真框架不在 result 中回传 income
        时（实测确认），完成订单列表为空，则回退到决策时即时累计（committed_income_total）。
        """
        total_income = 0.0
        total_minutes = 0.0
        for rec in self.completed_orders:
            if rec.occupied_minutes <= 0 or rec.income <= 0:
                continue
            total_income += rec.income
            total_minutes += rec.occupied_minutes
        if total_minutes > 0:
            return total_income / total_minutes
        if (
            self.committed_occupied_minutes_total > 0
            and self.committed_income_total > 0
        ):
            return self.committed_income_total / self.committed_occupied_minutes_total
        return 0.0

    def record_committed_order(
        self,
        cargo_id: str,
        price: float,
        occupied_minutes: float,
        end_lat: float = 0.0,
        end_lng: float = 0.0,
        completed_at_minutes: int = 0,
    ) -> None:
        """PR#32 突破 #3：决策时即时累计 cargo.price 与 occupied_minutes。

        每个 cargo_id 只累计一次（防止 query_history 同步重放导致重复计数）。
        同时回填 completed_orders（PR#30 链评估依赖此列表）——规避仿真框架
        不在 result 中回传 income 导致 _absorb_single_record 跳过该记录。
        """
        if not cargo_id or price <= 0 or occupied_minutes <= 0:
            return
        if cargo_id in self.committed_seen_cargo_ids:
            return
        self.committed_seen_cargo_ids.add(cargo_id)
        self.committed_income_total += price
        self.committed_occupied_minutes_total += occupied_minutes
        self.committed_take_count += 1
        # 回填到 completed_orders 列表，让 chain_value / count_nearby 等
        # PR#30 链评估逻辑能基于"已承诺"的订单运作。
        if end_lat != 0.0 and end_lng != 0.0:
            self.completed_orders.append(
                CompletedOrderRecord(
                    end_lat=end_lat,
                    end_lng=end_lng,
                    income=price,
                    occupied_minutes=occupied_minutes,
                    completed_at_minutes=completed_at_minutes,
                )
            )

    def record_order_completion_stats(
        self,
        lat: float,
        lng: float,
        hour: int,
        income: float,
        minutes: float,
    ) -> None:
        """记录完成订单的区域-时段统计（方案五：在线学习）。"""
        grid_deg = config.ONLINE_LEARNING_GRID_DEG
        grid_key = f"{lat / grid_deg:.0f}_{lng / grid_deg:.0f}_{hour}"
        stats = self.location_time_stats.get(grid_key)
        if stats is None:
            stats = LocationTimeStats(grid_key=grid_key, hour_bucket=hour)
            self.location_time_stats[grid_key] = stats
        stats.total_income += income
        stats.total_orders += 1
        stats.total_minutes += max(1.0, minutes)

    def get_location_time_value(self, lat: float, lng: float, hour: int) -> float:
        """查询某位置某时段的历史收益率（方案五）。"""
        grid_deg = config.ONLINE_LEARNING_GRID_DEG
        grid_key = f"{lat / grid_deg:.0f}_{lng / grid_deg:.0f}_{hour}"
        stats = self.location_time_stats.get(grid_key)
        if stats is not None and stats.total_orders >= config.ONLINE_LEARNING_MIN_ORDERS:
            return stats.avg_rate
        return 0.0

    def get_high_value_reposition_targets(
        self,
        current_lat: float,
        current_lng: float,
        max_dist_km: float | None = None,
    ) -> list[tuple[float, float]]:
        """基于历史接单经验推荐高价值空驶目标（方案四）。

        ``max_dist_km`` 缺省使用 ``config.SMART_REPOSITION_MAX_DIST_KM``；
        PR#32 突破 #2 允许困境模式临时放宽到 ``STRANDED_REPOSITION_MAX_DIST_KM``。
        """
        cap = max_dist_km if max_dist_km is not None else config.SMART_REPOSITION_MAX_DIST_KM
        targets: list[tuple[float, float, float]] = []
        for key, cell in self.hotspots.items():
            if cell.samples < config.SMART_REPOSITION_MIN_SAMPLES:
                continue
            lat, lng = geo_utils.grid_center(key)
            dist = geo_utils.haversine_km(current_lat, current_lng, lat, lng)
            if dist < config.SMART_REPOSITION_MIN_DIST_KM:
                continue
            if dist > cap:
                continue
            avg_income = cell.sum_price / max(1, cell.samples)
            avg_yield = cell.sum_price_per_minute / max(1, cell.samples)
            score = avg_income * 0.3 + avg_yield * 100.0
            targets.append((lat, lng, score))
        targets.sort(key=lambda t: t[2], reverse=True)
        return [(t[0], t[1]) for t in targets[: config.SMART_REPOSITION_TOP_N]]

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
                # 方案一+五：记录完成订单用于链评估和在线学习
                order_income = float(result.get("income", 0.0) or 0.0)
                if order_income <= 0:
                    order_income = float(result.get("price", 0.0) or 0.0)
                self.total_gross_income += order_income
                pos_after = record.get("position_after", {}) or {}
                end_lat = float(pos_after.get("lat", 0.0) or 0.0)
                end_lng = float(pos_after.get("lng", 0.0) or 0.0)
                if end_lat != 0.0 and end_lng != 0.0 and order_income > 0:
                    self.record_completed_order(
                        end_lat=end_lat,
                        end_lng=end_lng,
                        income=order_income,
                        occupied_minutes=float(action_exec),
                        completed_at_minutes=sim_minutes_after,
                    )
                    if config.ONLINE_LEARNING_ENABLED:
                        self.record_order_completion_stats(
                            lat=end_lat,
                            lng=end_lng,
                            hour=geo_utils.hour_of_day(sim_minutes_after),
                            income=order_income,
                            minutes=float(action_exec),
                        )
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
        if self.rules is not None and action_name in {"take_order", "reposition"}:
            if action_name == "take_order" and not bool(result.get("accepted", False)):
                return
            self._record_no_drive_window_violations(action_start_minutes, sim_minutes_after)

    def _record_no_drive_window_violations(self, action_start_minutes: int, action_end_minutes: int) -> None:
        rules = self.rules
        if rules is None:
            return
        for window in getattr(rules, "no_drive_windows", []) or []:
            penalty = float(getattr(window, "penalty_amount", 0.0) or 0.0)
            if penalty <= 0:
                continue
            rule_id = f"nodrive_{int(window.start_minute)}"
            days = _no_drive_window_hit_dates(action_start_minutes, action_end_minutes, window)
            if not days:
                continue
            self.preference_violation_days[rule_id].update(days)
            raw = penalty * len(self.preference_violation_days[rule_id])
            cap = getattr(window, "penalty_cap", None)
            if cap is not None and float(cap) > 0:
                raw = min(raw, float(cap))
            self.preference_penalty_accum[rule_id] = raw

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
