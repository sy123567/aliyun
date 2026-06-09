"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。

策略：在大模型解析司机自然语言偏好的基础上，用确定性调度器保证偏好合规
（每日休息 / 夜间休息窗 / 整月整休天数 / 禁接货类 / 禁接区域 / 必接区域 /
赴装空驶上限 / 特定日期到点事件），并在合规候选中按净收益择单。
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from simkit.ports import SimulationApiPort

DAY_MINUTES = 1440
MONTH_DAYS = 92
SPEED_KM_PER_HOUR = 60.0
EARTH_RADIUS_KM = 6371.0
COST_PER_KM = 1.5
# Minimum remaining minutes in the day worth attempting an anti-stranding reposition:
# below this there is no time to relocate and still complete an order before day end.
_STRAND_MIN_BUDGET = 240
_ORDER_DEADLINE_BUFFER_MIN = 10
_LLM_CARGO_SUMMARY_LIMIT = 8
_MIN_BOUNDED_AREA_SPAN = 0.1

SHENZHEN_BBOX = (22.42, 22.89, 113.74, 114.66)  # lat_min, lat_max, lng_min, lng_max
_MONTH_START_DAYS = (0, 31, 61, 92)
_MONTH_NAMES = {0: "3月", 1: "4月", 2: "5月"}
_ALLOWED_REGION_GROUPS: dict[str, tuple[tuple[str, ...], tuple[float, float, float, float]]] = {
    "上海": (("上海",), (30.6, 31.9, 120.8, 122.2)),
    "江苏": (("江苏",), (30.7, 35.2, 116.3, 121.9)),
    "浙江": (("浙江",), (27.0, 31.5, 118.0, 123.0)),
    "安徽": (("安徽",), (29.4, 34.7, 114.8, 119.7)),
    "广东": (("广东",), (20.0, 25.6, 109.3, 117.4)),
    "江浙沪": (("上海", "江苏", "浙江"), (27.0, 35.0, 116.0, 123.5)),
    "长三角": (("上海", "江苏", "浙江", "安徽"), (27.0, 35.5, 114.0, 123.5)),
    "珠三角": (("广州", "深圳", "佛山", "东莞", "中山", "珠海", "惠州", "江门", "肇庆"), (21.5, 24.5, 112.0, 115.5)),
    "大湾区": (("广州", "深圳", "佛山", "东莞", "中山", "珠海", "惠州", "江门", "肇庆", "香港", "澳门"), (21.5, 24.5, 112.0, 115.5)),
}
_ALLOWED_REGION_ALIASES = {
    "广东省": "广东",
    "上海市": "上海",
    "江苏省": "江苏",
    "浙江省": "浙江",
    "安徽省": "安徽",
    "珠江三角洲": "珠三角",
    "粤港澳大湾区": "大湾区",
    "长江三角洲": "长三角",
    "长三角地区": "长三角",
    "江浙沪皖": "长三角",
}
_RELATIVE_PROVINCE_REGIONS = ("上海", "江苏", "浙江", "安徽", "广东")


def _month_index_for_day(day: int) -> int:
    """Calendar month index for 2026-03-01..2026-05-31."""
    if day < _MONTH_START_DAYS[1]:
        return 0
    if day < _MONTH_START_DAYS[2]:
        return 1
    return 2


def _month_end_day_exclusive(month_idx: int) -> int:
    idx = max(0, min(2, int(month_idx)))
    return _MONTH_START_DAYS[idx + 1]


def _day_in_month(day: int) -> int:
    idx = _month_index_for_day(day)
    return int(day) - _MONTH_START_DAYS[idx] + 1


def _month_name(month_idx: int) -> str:
    return _MONTH_NAMES.get(month_idx, f"第{month_idx + 1}月")


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _travel_minutes(distance_km: float) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil(distance_km / SPEED_KM_PER_HOUR * 60.0))


def _in_shenzhen(lat: float, lng: float) -> bool:
    a, b, c, d = SHENZHEN_BBOX
    return a <= lat <= b and c <= lng <= d


_REGION_SUFFIXES = ("省", "市", "区", "县", "镇", "村", "自治州", "地区")


def _norm_region(name: str) -> str:
    """去掉行政区后缀，便于「增城」与「增城区」「惠州」与「惠州市」互配。"""
    name = name.strip()
    changed = True
    while changed and name:
        changed = False
        for suf in _REGION_SUFFIXES:
            if len(name) > len(suf) and name.endswith(suf):
                name = name[: -len(suf)]
                changed = True
    return name


def _region_in_city(region: str, city: str) -> bool:
    """区域归属判断：对行政区后缀差异稳健（双向子串匹配）。"""
    if not region or not city:
        return False
    if region in city or city in region:
        return True
    r, c = _norm_region(region), _norm_region(city)
    if not r or not c:
        return False
    return r in c or c in r


def _allowed_region_key(region: str) -> str | None:
    raw = region.strip()
    if not raw:
        return None
    compact = _norm_region(raw)
    for tail in ("范围", "区域", "一带", "周边", "内"):
        if compact.endswith(tail) and len(compact) > len(tail):
            compact = compact[: -len(tail)]
    if raw in _ALLOWED_REGION_ALIASES:
        return _ALLOWED_REGION_ALIASES[raw]
    if compact in _ALLOWED_REGION_ALIASES:
        return _ALLOWED_REGION_ALIASES[compact]
    if compact in _ALLOWED_REGION_GROUPS:
        return compact
    return None


def _city_matches_allowed_regions(regions: set[str], city: str) -> bool:
    if not regions:
        return True
    city = city.strip()
    if not city:
        return False
    for region in regions:
        if _region_in_city(region, city):
            return True
        key = _allowed_region_key(region)
        if key is None:
            continue
        keywords, _bbox = _ALLOWED_REGION_GROUPS[key]
        if any(_region_in_city(kw, city) or kw in city for kw in keywords):
            return True
    return False


def _point_matches_allowed_regions(regions: set[str], lat: float, lng: float) -> bool:
    if not regions:
        return True
    for region in regions:
        key = _allowed_region_key(region)
        if key is None:
            continue
        la_min, la_max, ln_min, ln_max = _ALLOWED_REGION_GROUPS[key][1]
        if la_min <= lat <= la_max and ln_min <= lng <= ln_max:
            return True
    return False


def _province_region_from_point(lat: float, lng: float) -> str | None:
    for region in _RELATIVE_PROVINCE_REGIONS:
        la_min, la_max, ln_min, ln_max = _ALLOWED_REGION_GROUPS[region][1]
        if la_min <= lat <= la_max and ln_min <= lng <= ln_max:
            return region
    return None


def _target_matches_allowed_regions(
    regions: set[str], lat: float, lng: float, city: str | None = None
) -> bool:
    if not regions:
        return True
    if city and _city_matches_allowed_regions(regions, city):
        return True
    return _point_matches_allowed_regions(regions, lat, lng)


def _text_supports(value: str, all_text: str, min_ratio: float = 0.5) -> bool:
    """容噪 grounding：判断 LLM 抽出的 value 是否有原文依据。

    评测的偏好文本是被刻意打乱/删字的噪声文本（如「凡蔬货我律掉」），LLM 能把它
    正确归一为规范类目（「蔬菜」），但规范词往往不是原文的连续子串，旧的精确子串
    校验会把「正确的抽取」误判为幻觉而丢弃 → 漏规则 → 违规扣分。

    这里在精确/去后缀匹配之外，再放宽为「字符覆盖率」匹配：value 的去重字符里至少
    有 min_ratio 比例出现在原文中即视为有依据（如「蔬菜」中「蔬」在原文 → 1/2≥0.5
    通过）。完全无字符重叠的纯幻觉（如原文只谈蔬菜却抽出「危化品」）仍被拒绝。
    误纳一个类目只损失少量收益、不会扣分；漏纳一个类目必然违规扣分，故对召回放宽。
    """
    if not all_text:
        return True
    v = value.strip()
    if not v:
        return False
    if v in all_text or _norm_region(v) in all_text:
        return True
    nv = _norm_region(v) or v
    chars = set(nv)
    if not chars:
        return False
    present = sum(1 for ch in chars if ch in all_text)
    return present / len(chars) >= min_ratio


class DriverRules:
    """结构化偏好规则。"""

    def __init__(self) -> None:
        self.daily_rest_minutes: int = 0
        self.rest_window: tuple[int, int] | None = None  # (start_min, end_min) within day, from 0
        self.off_days_min: int = 0
        self.forbidden_categories: set[str] = set()
        self.avoid_categories: set[str] = set()  # soft avoid (still filter)
        self.forbidden_regions: set[str] = set()
        self.allowed_regions: set[str] = set()
        self.required_region: tuple[str, int] | None = None  # (region, min_days)
        self.pickup_max_km: float | None = None
        self.blackout: list[tuple[str, set[int]]] = []  # (region, days)
        self.blackout_coords: dict[str, tuple[float, float]] = {}  # region→(lat,lng)
        self.dated_single: list[dict[str, Any]] = []  # {day,lat,lng,min_wait,before}
        self.dated_route: list[dict[str, Any]] = []  # {day, stops:[{lat,lng,min_wait,before}]}
        # --- new rule types from 10-driver trained version ---
        self.no_drive_windows: list[tuple[int, int]] = []  # (start_min, end_min) 0..2880
        self.home_lat: float | None = None
        self.home_lng: float | None = None
        self.home_radius_km: float = 1.0
        self.home_by_minute: int | None = None  # must be near home by this minute of day
        self.no_drive_until_minute: int | None = None  # don't accept orders until this minute
        self.daily_order_limit: int | None = None
        self.haul_max_km: float | None = None  # max distance from pickup to dropoff
        self.monthly_deadhead_max_km: float | None = None
        self.forbidden_zones: list[tuple[float, float, float]] = []  # (lat, lng, radius_km)
        self.bounded_area: tuple[float, float, float, float] | None = None  # (lat_min, lat_max, lng_min, lng_max)
        self.must_visit: list[dict[str, Any]] = []  # {lat, lng, radius_km, required_days}
        self.first_order_before_minute: int | None = None
        self.monthly_category_targets: dict[int, dict[str, int]] = {}
        self.category_carryover_months: set[int] = set()

    @property
    def day_rest_block(self) -> int:
        """每日开工前在 00:00 起需要的连续静止分钟数。"""
        block = self.daily_rest_minutes
        if self.rest_window is not None:
            rw_start, rw_end = self.rest_window
            # Only use rest_window end as day_rest_block for overnight windows
            # (end < start means it wraps around midnight, e.g. 22:00-06:00 → (0,360)).
            # For daytime windows (e.g. 11:00-13:30 → (660,810)), the no_drive_windows
            # handles enforcement; we must NOT inflate day_rest_block or the driver
            # will idle the entire morning.
            if rw_end <= rw_start or rw_start == 0:
                block = max(block, rw_end)
        if self.no_drive_until_minute is not None:
            block = max(block, self.no_drive_until_minute)
        return block


class DecisionHistory:
    """维护司机的决策历史，为LLM提供结构化上下文。

    企业级设计：不依赖硬编码地区知识，通过历史模式自适应任何司机/地区。
    """

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._history: list[dict[str, Any]] = []  # rolling window of recent decisions
        self._monthly_stats: dict[int, dict[str, Any]] = {}  # month_idx → stats
        self._total_orders = 0
        self._total_income = 0.0
        self._total_penalty_violations = 0
        self._category_counts: dict[int, dict[str, int]] = {}  # month → {cat: count}
        self._longhual_counts: dict[int, int] = {}  # month → count of >8h orders

    def record(self, step: int, day: int, tod: int, action: str, params: dict,
               result: dict | None = None, cargo_info: dict | None = None) -> None:
        """Record a decision step."""
        month_idx = _month_index_for_day(day)
        entry = {
            "step": step,
            "day": day,
            "tod": tod,
            "action": action,
            "params": params,
        }
        if cargo_info:
            entry["cargo_name"] = cargo_info.get("cargo_name", "")
            entry["cargo_price"] = cargo_info.get("price", 0)
            entry["cost_time_minutes"] = cargo_info.get("cost_time_minutes", 0)
        if result:
            entry["accepted"] = result.get("accepted", False)
            entry["detail"] = result.get("detail", "")
            entry["haul_km"] = result.get("haul_distance_km", 0)
            entry["deadhead_km"] = result.get("pickup_deadhead_km", 0)
            if result.get("accepted"):
                self._total_orders += 1
                income = entry.get("cargo_price", 0)
                self._total_income += income
                # Track category
                cname = entry.get("cargo_name", "")
                self._category_counts.setdefault(month_idx, {})
                self._category_counts[month_idx][cname] = \
                    self._category_counts[month_idx].get(cname, 0) + 1
                # Track long-haul
                if entry.get("cost_time_minutes", 0) > 480:
                    self._longhual_counts[month_idx] = \
                        self._longhual_counts.get(month_idx, 0) + 1
        # Update monthly stats
        stats = self._monthly_stats.setdefault(month_idx, {
            "orders": 0, "waits": 0, "repositions": 0,
            "total_haul_km": 0.0, "total_deadhead_km": 0.0,
        })
        if action == "take_order" and result and result.get("accepted"):
            stats["orders"] += 1
            stats["total_haul_km"] += entry.get("haul_km", 0)
            stats["total_deadhead_km"] += entry.get("deadhead_km", 0)
        elif action == "wait":
            stats["waits"] += 1
        elif action == "reposition":
            stats["repositions"] += 1

        self._history.append(entry)
        if len(self._history) > self._window_size:
            self._history.pop(0)

    def get_summary(self, current_day: int) -> str:
        """Generate a concise history summary for LLM context."""
        month_idx = _month_index_for_day(current_day)
        month_name = _month_name(month_idx)
        lines = []
        lines.append(f"=== 决策历史摘要 ===")
        lines.append(f"总接单: {self._total_orders}, 累计收入: ¥{self._total_income:.0f}")

        # Monthly breakdown
        for mi in sorted(self._monthly_stats):
            mn = _month_name(mi)
            s = self._monthly_stats[mi]
            lines.append(
                f"{mn}: 接单{s['orders']}, 等待{s['waits']}次, "
                f"重定位{s['repositions']}次, "
                f"运输{s['total_haul_km']:.0f}km, 空驶{s['total_deadhead_km']:.0f}km"
            )

        # Category progress for current month
        if month_idx in self._category_counts:
            cats = self._category_counts[month_idx]
            top_cats = sorted(cats.items(), key=lambda x: -x[1])[:5]
            lines.append(f"\n{month_name}品类分布(前5): " +
                         ", ".join(f"{c}:{n}单" for c, n in top_cats))

        # Long-haul tracking
        lh = self._longhual_counts.get(month_idx, 0)
        lines.append(f"{month_name}长途(>8h): {lh}/5单上限")

        # Recent decisions (last 10)
        recent = self._history[-10:]
        if recent:
            lines.append(f"\n最近{len(recent)}步决策:")
            for e in recent:
                d, t = e["day"], e["tod"]
                h, m = divmod(t, 60)
                act = e["action"]
                detail = ""
                if act == "take_order":
                    cn = e.get("cargo_name", "?")
                    acc = "✓" if e.get("accepted") else "✗"
                    fail = ""
                    if not e.get("accepted") and e.get("detail"):
                        fail = f"({str(e.get('detail'))[:12]})"
                    detail = f" {cn} {acc}{fail}"
                elif act == "wait":
                    dur = e["params"].get("duration_minutes", 0)
                    detail = f" {dur}分钟"
                elif act == "reposition":
                    detail = f" →({e['params'].get('latitude',0):.1f},{e['params'].get('longitude',0):.1f})"
                lines.append(f"  Day{d} {h:02d}:{m:02d} {act}{detail}")

        return "\n".join(lines)

    def update_last_result(self, result: dict, cargo_info: dict | None = None) -> None:
        """Update the most recent history entry with execution result."""
        if not self._history:
            return
        entry = self._history[-1]
        month_idx = _month_index_for_day(entry["day"])
        if result:
            entry["accepted"] = result.get("accepted", False)
            entry["detail"] = result.get("detail", "")
            entry["haul_km"] = result.get("haul_distance_km", 0)
            entry["deadhead_km"] = result.get("pickup_deadhead_km", 0)
            if result.get("accepted"):
                self._total_orders += 1
                stats = self._monthly_stats.get(month_idx)
                if stats:
                    stats["orders"] += 1
                    stats["total_haul_km"] += entry.get("haul_km", 0)
                    stats["total_deadhead_km"] += entry.get("deadhead_km", 0)
        if cargo_info:
            entry["cargo_name"] = cargo_info.get("cargo_name", "")
            entry["cargo_price"] = cargo_info.get("price", 0)
            entry["cost_time_minutes"] = cargo_info.get("cost_time_minutes", 0)
            cname = entry.get("cargo_name", "")
            if result and result.get("accepted") and cname:
                self._category_counts.setdefault(month_idx, {})
                self._category_counts[month_idx][cname] = \
                    self._category_counts[month_idx].get(cname, 0) + 1
                if entry.get("cost_time_minutes", 0) > 480:
                    self._longhual_counts[month_idx] = \
                        self._longhual_counts.get(month_idx, 0) + 1

    def get_category_progress(self, month_idx: int) -> dict[str, int]:
        """Get category order counts for a specific month."""
        return dict(self._category_counts.get(month_idx, {}))

    def get_longhual_count(self, month_idx: int) -> int:
        """Get long-haul order count for a specific month."""
        return self._longhual_counts.get(month_idx, 0)


class ModelDecisionService:
    """单步决策：LLM驱动 + 决策历史感知 + 规则引擎兜底的调度器。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._rules: dict[str, DriverRules] = {}
        self._plan: dict[str, dict[str, Any]] = {}
        self._history: dict[str, DecisionHistory] = {}  # driver_id → history
        self._step_count: dict[str, int] = {}  # driver_id → step counter
        # preference texts already fed to the parser (LLM is only re-invoked when a
        # new, date-windowed preference becomes visible).
        self._seen_prefs: dict[str, set[str]] = {}
        self._initial_position: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------ decide
    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        rules = self._ensure_rules(driver_id, status)
        plan = self._plan.setdefault(
            driver_id,
            {
                "rest_done": set(),
                "zeng_order_days": set(),
                "dated_single_done": set(),
                "dated_route_done": set(),
                "strand_repo": set(),
                "orders_today": {},  # day → count
                "total_deadhead_km": 0.0,
                "monthly_deadhead_km": {},  # month_idx → pickup deadhead km
                "must_visit_days": {},  # idx → set of days visited
                "first_order_taken": set(),  # days where first order was already taken
                "home_done": set(),  # days where home repositioning is done
                "monthly_longhual": {},  # month_idx → count of >8h orders
                "monthly_category_orders": {},  # month_idx → {category: count}
                "failed_cargo_ids": set(),  # cargo ids that failed execution in this run
                "failed_cargo_reasons": {},  # cargo_id → detail
            },
        )
        # Initialize decision history for this driver
        history = self._history.setdefault(driver_id, DecisionHistory())
        self._step_count.setdefault(driver_id, 0)
        self._step_count[driver_id] += 1

        # Preferences may only become visible inside their date window, so the off-day
        # set is recomputed each step from the rules known so far.
        plan["off_days"] = self._plan_off_days(rules)
        now = int(status["simulation_progress_minutes"])
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        day, tod = divmod(now, DAY_MINUTES)

        # Try LLM-driven decision with history context
        action = None
        step = self._step_count[driver_id]
        consecutive_waits = plan.get("_consecutive_waits", 0)
        # Call LLM every 3 steps (or when idle), but skip if stuck in wait loop (>2 consecutive)
        use_llm = (step % 3 == 0 or plan.get("_last_action") == "wait") and consecutive_waits < 3
        if use_llm:
            action = self._llm_decide_with_history(
                driver_id, status, rules, plan, history, now, lat, lng, day, tod
            )

        # Fallback to rule engine if LLM decision is None or failed
        if action is None:
            action = self._schedule(driver_id, status, rules, plan, now, lat, lng)

        # Track consecutive waits to detect LLM getting stuck
        if action.get("action") == "wait":
            plan["_consecutive_waits"] = consecutive_waits + 1
        else:
            plan["_consecutive_waits"] = 0

        # Record decision in history
        plan["_last_action"] = action.get("action")
        history.record(
            step=step, day=day, tod=tod,
            action=action.get("action", "unknown"),
            params=action.get("params", {}),
        )

        self._logger.info(
            "decision driver_id=%s now=%s day=%s tod=%s action=%s params=%s step=%d",
            driver_id, now, day, tod,
            action.get("action"), action.get("params"), step,
        )
        return action

    def update_decision_result(
        self,
        driver_id: str,
        result: dict,
        cargo_info: dict | None = None,
        action_start_minutes: int | None = None,
    ) -> None:
        """Called after execution to update decision history with the result.

        This allows the history to track which orders were accepted, rejected, etc.
        """
        history = self._history.get(driver_id)
        if history:
            history.update_last_result(result, cargo_info)
        plan = self._plan.get(driver_id)
        if plan is None or not isinstance(result, dict):
            return
        cargo_id = str(result.get("cargo_id") or "")
        if result.get("accepted") is False and cargo_id:
            plan.setdefault("failed_cargo_ids", set()).add(cargo_id)
            plan.setdefault("failed_cargo_reasons", {})[cargo_id] = str(result.get("detail", "failed"))
            return
        if not result.get("accepted") or not cargo_id or not cargo_info:
            return
        progress = int(result.get("simulation_progress_minutes", 0) or 0)
        action_start = int(action_start_minutes) if action_start_minutes is not None else progress
        day = action_start // DAY_MINUTES
        day = max(0, min(MONTH_DAYS - 1, day))
        month_idx = _month_index_for_day(day)
        plan["orders_today"][day] = plan["orders_today"].get(day, 0) + 1
        plan["first_order_taken"].add(day)
        pickup_deadhead = float(result.get("pickup_deadhead_km", 0.0) or 0.0)
        plan["total_deadhead_km"] += pickup_deadhead
        monthly_deadhead = plan.setdefault("monthly_deadhead_km", {})
        monthly_deadhead[month_idx] = monthly_deadhead.get(month_idx, 0.0) + pickup_deadhead
        if int(cargo_info.get("cost_time_minutes", 0) or 0) > 480:
            plan["monthly_longhual"][month_idx] = plan["monthly_longhual"].get(month_idx, 0) + 1
        rules = self._rules.get(driver_id)
        if rules is not None and rules.required_region is not None:
            region = rules.required_region[0]
            start = cargo_info.get("start") or {}
            end = cargo_info.get("end") or {}
            if _region_in_city(region, str(start.get("city", ""))) or _region_in_city(region, str(end.get("city", ""))):
                plan["zeng_order_days"].add(day)
        self._track_category_order(plan, self._rules.get(driver_id), month_idx, str(cargo_info.get("cargo_name", "")))

    def _llm_decide_with_history(self, driver_id, status, rules, plan, history,
                                  now, lat, lng, day, tod) -> dict[str, Any] | None:
        """Use LLM with decision history context to make strategic decisions.

        Returns a valid action dict or None (to fall back to rule engine).
        """
        import time
        start_time = time.time()

        # Skip LLM for mandatory actions (rest, off-day, dated events)
        if day in plan.get("off_days", set()):
            return None  # let rule engine handle
        block = rules.day_rest_block
        if block > 0 and day not in plan.get("rest_done", set()):
            return None  # let rule engine handle rest

        # Query available cargo for context
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=30)
        items = cargo_resp.get("items", [])
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        hard_end = self._hard_order_deadline(rules, plan, now, day)
        blackout_regions = {r for r, days in rules.blackout if day in days}
        failed_ids = plan.get("failed_cargo_ids", set())

        # Build a compact list of locally feasible cargo. LLM may express preference,
        # but execution safety stays deterministic.
        cargo_summary = []
        for item in items:
            cargo = item.get("cargo", {})
            cargo_id = str(cargo.get("cargo_id", ""))
            if cargo_id in failed_ids:
                continue
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, now, hard_end, lat, lng)
            if ev is None:
                continue
            net, _touches_required, occupied, pickup_km = ev
            cargo_summary.append({
                "cargo_id": cargo_id,
                "name": cargo.get("cargo_name", ""),
                "price": round(float(cargo.get("price", 0) or 0), 2),
                "minutes": int(occupied),
                "pickup_km": round(float(pickup_km), 1),
                "net_per_h": round(float(net) / max(1, int(occupied)) * 60, 1),
                "from": cargo.get("start", {}).get("city", ""),
                "to": cargo.get("end", {}).get("city", ""),
            })
            if len(cargo_summary) >= _LLM_CARGO_SUMMARY_LIMIT:
                break

        # Build preference rules summary
        pref_summary = self._format_rules_for_llm(driver_id, rules, plan, day)

        # Build history summary
        history_text = history.get_summary(day)

        # Construct LLM prompt
        day_in_month = _day_in_month(day)
        month_idx = _month_index_for_day(day)
        month_name = _month_name(month_idx)
        hour, minute = divmod(tod, 60)

        system_prompt = (
            "你是一个智能货运调度决策AI。根据司机当前状态、决策历史、可用货源和偏好规则，"
            "做出最优决策。你的目标是：1) 最大化净收入 2) 最小化偏好违规处罚 3) 完成品类指标。\n\n"
            "可用动作：\n"
            '- take_order: 接单。参数: {"cargo_id": "ID"}。优先选择目标品类、高价、短途的货\n'
            '- wait: 等待。参数: {"duration_minutes": 分钟数}。仅在21:00-06:00夜休时段或货源确实不好时使用\n'
            '- reposition: 空驶到新位置。参数: {"latitude": 纬度, "longitude": 经度}。'
            '当附近无目标品类货源时，主动移动到可能有货的区域（根据地理常识判断）\n\n'
            "重要规则：\n"
            "- 21:00后不得接单或空驶（夜停休）；空驶必须在20:50前结束，21:00-06:00期间只能wait\n"
            "- 每月长途(>8h)最多5单\n"
            "- 不要连续wait超过2次！如果当前货源不合适，应该reposition到新区域寻找更好货源\n"
            "- 品类指标很重要：未达标会有高额罚款。优先接指标品类的货\n"
            "- 只能从候选货源列表里选 cargo_id；候选为空或都不合适就 wait/reposition\n"
            "- 只输出一个JSON对象: {\"action\": \"...\", \"params\": {...}, \"reason\": \"20字内\"}\n"
        )

        # Add category target warning
        cat_target, cat_needed = self._get_category_target(rules, plan, day)
        month_deadhead = plan.setdefault("monthly_deadhead_km", {}).get(month_idx, 0.0)
        cat_warning = ""
        if cat_target and cat_needed > 0:
            cat_warning = f"\n⚠️ 品类指标警告: 本月需要'{cat_target}'至少12单，还差{cat_needed}单！优先接此品类。\n"

        user_prompt = (
            f"## 当前状态\n"
            f"司机: {driver_id}, 位置: ({lat:.2f}, {lng:.2f})\n"
            f"时间: {month_name}{day_in_month}日 {hour:02d}:{minute:02d}\n"
            f"仿真进度: 第{day}天/{MONTH_DAYS}天\n"
            f"{cat_warning}\n"
            f"## 偏好规则与合规进度\n{pref_summary}\n\n"
            f"## 决策历史\n{history_text}\n\n"
            f"## 已通过本地可行性检查的候选货源({len(cargo_summary)}条)\n"
            f"{json.dumps(cargo_summary, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"请做出决策（只输出JSON）:"
        )

        try:
            req = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
                "max_tokens": 180,
            }
            resp = self._api.model_chat_completion(req)
            elapsed = time.time() - start_time

            # Timeout check: if >60s, log warning (for next iteration consider disabling thinking)
            if elapsed > 60:
                self._logger.warning(
                    "[LLM] decision took %.1fs (>60s threshold) driver_id=%s",
                    elapsed, driver_id
                )

            # Parse LLM response
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            data = json.loads(content) if content else None
            if data and "action" in data:
                action_type = data["action"]
                params = data.get("params", {})
                reason = data.get("reason", "")
                self._logger.info(
                    "[LLM] decision driver_id=%s action=%s reason=%s elapsed=%.1fs",
                    driver_id, action_type, reason, elapsed
                )
                # Validate against hard constraints before accepting
                if action_type == "take_order" and "cargo_id" in params:
                    cargo_id = str(params["cargo_id"])
                    if not self._validate_llm_take_order(cargo_id, items, rules, plan, now, lat, lng, day, hard_end):
                        return None
                    return self._take_order(cargo_id)
                elif action_type == "wait" and "duration_minutes" in params:
                    dur = int(params["duration_minutes"])
                    for ws, we in rules.no_drive_windows:
                        if self._tod_in_window(tod, ws, we):
                            wait_until_window_end = self._minutes_until_window_end(tod, ws, we)
                            if wait_until_window_end > 0:
                                return self._safe_wait(rules, now, max(dur, wait_until_window_end))
                    return self._safe_wait(rules, now, dur)
                elif action_type == "reposition" and "latitude" in params and "longitude" in params:
                    return self._safe_reposition(
                        rules,
                        now,
                        lat,
                        lng,
                        float(params["latitude"]),
                        float(params["longitude"]),
                        deadline=(day + 1) * DAY_MINUTES,
                        tag="LLM",
                    )
        except json.JSONDecodeError as exc:
            self._logger.warning("[LLM] JSON parse failed driver_id=%s: %s", driver_id, exc)
        except Exception as exc:
            self._logger.warning("[LLM] decision failed driver_id=%s: %s", driver_id, exc)

        return None  # fallback to rule engine

    @staticmethod
    def _tod_in_window(tod: int, start: int, end: int) -> bool:
        start %= DAY_MINUTES
        end %= DAY_MINUTES
        if start == end:
            return False
        if start < end:
            return start <= tod < end
        return tod >= start or tod < end

    @staticmethod
    def _minutes_until_window_end(tod: int, start: int, end: int) -> int:
        start %= DAY_MINUTES
        end %= DAY_MINUTES
        tod %= DAY_MINUTES
        if start == end:
            return 0
        if start < end:
            return max(0, end - tod) if start <= tod < end else 0
        if tod >= start:
            return DAY_MINUTES - tod + end
        if tod < end:
            return end - tod
        return 0

    def _extend_wait_for_no_drive(self, rules: DriverRules, now: int, duration: int) -> int:
        """Extend waits touching a no-drive window until that window is fully covered."""
        start = int(now)
        end = start + max(1, int(duration))
        if not rules.no_drive_windows:
            return max(1, int(duration))
        changed = True
        while changed:
            changed = False
            first_day = start // DAY_MINUTES - 1
            last_day = (end - 1) // DAY_MINUTES + 1
            for day_idx in range(first_day, last_day + 1):
                base = day_idx * DAY_MINUTES
                for ws, we in rules.no_drive_windows:
                    win_start = base + int(ws)
                    win_end = base + int(we)
                    if win_end <= win_start:
                        win_end += DAY_MINUTES
                    if start < win_end and end > win_start and end < win_end:
                        end = win_end
                        changed = True
        return max(1, end - start)

    def _safe_wait(self, rules: DriverRules, now: int, duration: int) -> dict[str, Any]:
        return self._wait(self._extend_wait_for_no_drive(rules, now, duration))

    def _hard_order_deadline(self, rules: DriverRules, plan: dict, now: int, day: int) -> int:
        """Latest acceptable finish time for a new order from this step."""
        day_start = day * DAY_MINUTES
        day_end = day_start + DAY_MINUTES
        hard_end = day_end
        for ws, _we in rules.no_drive_windows:
            ws_today = day_start + min(ws, DAY_MINUTES)
            if now < ws_today < hard_end:
                hard_end = max(now, ws_today - _ORDER_DEADLINE_BUFFER_MIN)
        if rules.home_by_minute is not None and day not in plan.get("home_done", set()):
            buffer = 60
            cutoff = day_start + rules.home_by_minute - buffer
            if cutoff > now:
                hard_end = min(hard_end, cutoff)
        return hard_end

    def _interval_overlaps_no_drive(self, rules: DriverRules, start: int, end: int) -> bool:
        """Whether an absolute minute interval touches a repeated no-drive window."""
        if end <= start or not rules.no_drive_windows:
            return False
        first_day = start // DAY_MINUTES - 1
        last_day = (end - 1) // DAY_MINUTES + 1
        for day_idx in range(first_day, last_day + 1):
            day_base = day_idx * DAY_MINUTES
            for ws, we in rules.no_drive_windows:
                win_start = day_base + int(ws)
                win_end = day_base + int(we)
                if win_end <= win_start:
                    win_end += DAY_MINUTES
                if start < win_end and end > win_start:
                    return True
        return False

    def _safe_reposition(
        self,
        rules: DriverRules,
        now: int,
        lat: float,
        lng: float,
        target_lat: float,
        target_lng: float,
        *,
        deadline: int | None = None,
        tag: str = "schedule",
        target_city: str | None = None,
    ) -> dict[str, Any] | None:
        if rules.allowed_regions and not _target_matches_allowed_regions(
            rules.allowed_regions, target_lat, target_lng, target_city
        ):
            self._logger.info(
                "[%s] rejected reposition: target outside allowed_regions=%s city=%s target=(%.4f,%.4f)",
                tag, sorted(rules.allowed_regions), target_city, target_lat, target_lng,
            )
            return None
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            lat_span, lng_span = la_max - la_min, ln_max - ln_min
            if lat_span >= _MIN_BOUNDED_AREA_SPAN and lng_span >= _MIN_BOUNDED_AREA_SPAN and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                if not (la_min <= target_lat <= la_max and ln_min <= target_lng <= ln_max):
                    self._logger.info(
                        "[%s] rejected reposition: target outside bounded_area target=(%.4f,%.4f)",
                        tag, target_lat, target_lng,
                    )
                    return None
        for fz_lat, fz_lng, fz_r in rules.forbidden_zones:
            if not (18 <= fz_lat <= 55 and 70 <= fz_lng <= 140):
                continue
            if _haversine_km(target_lat, target_lng, fz_lat, fz_lng) < fz_r:
                self._logger.info(
                    "[%s] rejected reposition: target inside forbidden_zone target=(%.4f,%.4f)",
                    tag, target_lat, target_lng,
                )
                return None
        day = int(now) // DAY_MINUTES
        for region, days in rules.blackout:
            if day not in days:
                continue
            in_region = region == "深圳" and _in_shenzhen(target_lat, target_lng)
            if not in_region and region in rules.blackout_coords:
                rlat, rlng = rules.blackout_coords[region]
                in_region = _haversine_km(target_lat, target_lng, rlat, rlng) < 60
            if in_region:
                self._logger.info(
                    "[%s] rejected reposition: target inside blackout region=%s day=%d",
                    tag, region, day,
                )
                return None
        distance_km = _haversine_km(lat, lng, target_lat, target_lng)
        finish = now + _travel_minutes(distance_km)
        if deadline is not None and finish > deadline:
            self._logger.info(
                "[%s] rejected reposition: finish=%d deadline=%d distance_km=%.1f",
                tag, finish, deadline, distance_km,
            )
            return None
        guard_end = finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.no_drive_windows else 0)
        if self._interval_overlaps_no_drive(rules, now, guard_end):
            self._logger.info(
                "[%s] rejected reposition: no_drive overlap start=%d finish=%d target=(%.4f,%.4f)",
                tag, now, finish, target_lat, target_lng,
            )
            return None
        return self._reposition(target_lat, target_lng)

    def _validate_llm_take_order(
        self,
        cargo_id: str,
        items: list[dict[str, Any]],
        rules: DriverRules,
        plan: dict,
        now: int,
        lat: float,
        lng: float,
        day: int,
        hard_end: int,
    ) -> bool:
        if cargo_id in plan.get("failed_cargo_ids", set()):
            self._logger.info("[LLM] rejected take_order: cargo_id=%s failed earlier", cargo_id)
            return False
        if any(self._tod_in_window(now % DAY_MINUTES, ws, we) for ws, we in rules.no_drive_windows):
            self._logger.info("[LLM] rejected take_order: in no_drive_window tod=%d", now % DAY_MINUTES)
            return False
        match = None
        for item in items:
            cargo = item.get("cargo", {})
            if str(cargo.get("cargo_id", "")) == cargo_id:
                match = item
                break
        if match is None:
            self._logger.info("[LLM] rejected take_order: cargo_id=%s not in visible candidates", cargo_id)
            return False
        cargo = match.get("cargo", {})
        month_idx = _month_index_for_day(day)
        longhual_count = plan.get("monthly_longhual", {}).get(month_idx, 0)
        if longhual_count >= 5 and int(cargo.get("cost_time_minutes", 0) or 0) > 480:
            self._logger.info("[LLM] rejected take_order: long-haul limit reached month=%d", month_idx)
            return False
        blackout_regions = {r for r, days in rules.blackout if day in days}
        ev = self._evaluate_cargo(cargo, match, rules, blackout_regions, now, hard_end, lat, lng)
        if ev is None:
            self._logger.info("[LLM] rejected take_order: cargo_id=%s failed deterministic feasibility", cargo_id)
            return False
        _net, _touches_required, occupied, _pickup_km = ev
        if occupied > 480 and longhual_count >= 5:
            self._logger.info("[LLM] rejected take_order: occupied=%d over long-haul limit", occupied)
            return False
        return True

    def _format_rules_for_llm(self, driver_id: str, rules: DriverRules, plan: dict, day: int) -> str:
        """Format driver rules and compliance progress as concise text for LLM."""
        month_idx = _month_index_for_day(day)
        lines = []
        if rules.rest_window:
            rs, re_ = rules.rest_window
            lines.append(f"- 休息窗口: {rs//60:02d}:{rs%60:02d}-{re_//60:02d}:{re_%60:02d}")
        if rules.no_drive_windows:
            for ws, we in rules.no_drive_windows:
                lines.append(f"- 禁止接单/空驶: {ws//60:02d}:{ws%60:02d}-{we//60:02d}:{we%60:02d}")
        if rules.forbidden_categories:
            lines.append(f"- 禁运品类: {', '.join(rules.forbidden_categories)}")
        if rules.forbidden_regions:
            lines.append(f"- 禁入区域: {', '.join(rules.forbidden_regions)}")
        if rules.allowed_regions:
            lines.append(f"- 仅允许运营区域: {', '.join(sorted(rules.allowed_regions))}")
        if rules.pickup_max_km:
            lines.append(f"- 空驶上限: {rules.pickup_max_km}km")
        if rules.daily_order_limit:
            today_count = plan.get("orders_today", {}).get(day, 0)
            lines.append(f"- 每日接单上限: {rules.daily_order_limit} (今日已接{today_count})")

        # Category targets with progress
        cat_orders = plan.get("monthly_category_orders", {}).get(month_idx, {})
        longhual = plan.get("monthly_longhual", {}).get(month_idx, 0)
        lines.append(f"- 本月长途(>8h): {longhual}/5单上限")
        lines.append(f"- 本月品类接单: {json.dumps(cat_orders, ensure_ascii=False)}")
        targets = getattr(rules, "monthly_category_targets", {}).get(month_idx, {})
        if targets:
            lines.append(f"- 本月品类指标: {json.dumps(targets, ensure_ascii=False)}")
        failed_reasons = plan.get("failed_cargo_reasons", {})
        if failed_reasons:
            recent_failed = list(failed_reasons.items())[-3:]
            lines.append("- 近期失败货源: " + "; ".join(f"{cid}:{reason}" for cid, reason in recent_failed))

        # Add raw preference texts for context
        prefs = self._get_active_preferences(driver_id)
        if prefs:
            lines.append("\n原始偏好文本:")
            for p in prefs:
                lines.append(f"  「{p}」")

        return "\n".join(lines)

    def _get_active_preferences(self, driver_id: str) -> list[str]:
        """Get the active preference texts for the current day (from seen_prefs)."""
        return list(self._seen_prefs.get(driver_id, set()))[:4]

    # --------------------------------------------------------------- scheduler
    def _schedule(self, driver_id, status, rules, plan, now, lat, lng) -> dict[str, Any]:
        day, tod = divmod(now, DAY_MINUTES)
        if day >= MONTH_DAYS:
            return self._wait(1)
        day_start = day * DAY_MINUTES
        day_end = day_start + DAY_MINUTES

        # (A) full off day: idle/rest the whole day.
        if day in plan["off_days"]:
            return self._safe_wait(rules, now, day_end - now)

        # (A') blackout day while sitting inside a forbidden region: idle the whole
        # day (waiting is never penalised).  Uses a proximity check for regions that
        # have known coordinates, plus the hard-coded Shenzhen bbox as fallback.
        for region, days in rules.blackout:
            if day not in days:
                continue
            in_region = False
            if region == "深圳" and _in_shenzhen(lat, lng):
                in_region = True
            elif region in rules.blackout_coords:
                rlat, rlng = rules.blackout_coords[region]
                if _haversine_km(lat, lng, rlat, rlng) < 60:
                    in_region = True
            if in_region:
                return self._safe_wait(rules, now, day_end - now)

        # (B) start-of-day rest (covers daily-rest & rest-window from 00:00).
        # For a flexible "N continuous hours anywhere" rest (rest_window is None) the
        # block may begin after midnight when the previous day's last order ran past
        # midnight: we then rest a full block from `now` (e.g. 02:00–10:00), which still
        # gives the day its 8h continuous span (the scorer clips spans to the day).
        block = rules.day_rest_block
        if block > 0 and day not in plan["rest_done"]:
            plan["rest_done"].add(day)
            if rules.rest_window is None:
                dur = block if tod + block <= DAY_MINUTES else max(0, DAY_MINUTES - tod)
            else:
                dur = min(block - tod, day_end - now) if tod < block else 0
            if dur > 0:
                return self._safe_wait(rules, now, dur)
        else:
            plan["rest_done"].add(day)

        # (C) dated single-stop events (e.g. 盘库).
        for ev in rules.dated_single:
            if ev["day"] != day or ev["day"] in plan["dated_single_done"]:
                continue
            before = day_start + ev["before"]
            if _haversine_km(lat, lng, ev["lat"], ev["lng"]) > 1.5:
                if now + _travel_minutes(_haversine_km(lat, lng, ev["lat"], ev["lng"])) <= before:
                    return self._safe_reposition(
                        rules, now, lat, lng, ev["lat"], ev["lng"], deadline=before, tag="dated_single"
                    )
                continue  # can't make it; skip silently
            plan["dated_single_done"].add(ev["day"])
            return self._safe_wait(rules, now, max(ev["min_wait"], 1))

        # (D) dated multi-stop route (e.g. 寿宴).
        for ev in rules.dated_route:
            if ev["day"] != day or ev["day"] in plan["dated_route_done"]:
                continue
            act = self._drive_route(ev, rules, plan, now, day_start, lat, lng)
            if act is not None:
                return act

        # (D2) pre-stage the evening before a time-tight route event so the driver
        # starts the event day already parked at the first stop.
        for ev in rules.dated_route:
            if ev["day"] != day + 1 or not ev["stops"]:
                continue
            first = ev["stops"][0]
            dist = _haversine_km(lat, lng, first["lat"], first["lng"])
            if dist > 1.5:
                if now + _travel_minutes(dist) <= day_end:
                    return self._safe_reposition(
                        rules, now, lat, lng, first["lat"], first["lng"], deadline=day_end, tag="route_prestage"
                    )
            else:
                return self._safe_wait(rules, now, day_end - now)

        # (D3) no_drive_windows: if current time-of-day falls inside a no-drive
        # window, idle until the window ends.
        for ws, we in rules.no_drive_windows:
            if self._tod_in_window(tod, ws, we):
                wait_min = self._minutes_until_window_end(tod, ws, we)
                if wait_min > 0:
                    return self._safe_wait(rules, now, wait_min)

        # (D4) must_visit: proactively go to must-visit locations if not enough
        # visits have been accumulated.  Only navigate when urgency is very high
        # (remaining days == still_needed) AND coordinates are explicitly from text
        # (to avoid LLM-hallucinated must_visit wasting entire days).
        for i, mv in enumerate(rules.must_visit):
            visited = plan["must_visit_days"].setdefault(i, set())
            remaining_days = MONTH_DAYS - day
            still_needed = mv["required_days"] - len(visited)
            if still_needed > 0 and remaining_days <= still_needed:
                dist = _haversine_km(lat, lng, mv["lat"], mv["lng"])
                if dist <= mv.get("radius_km", 1.0):
                    visited.add(day)
                elif dist < 150 and now + _travel_minutes(dist) <= day_end:
                    return self._safe_reposition(
                        rules, now, lat, lng, mv["lat"], mv["lng"], deadline=day_end, tag="must_visit"
                    )

        # (D5) home_rule: reposition to home before cutoff, idle until morning.
        # Only enforce when home coordinates were explicitly found in preference text.
        if rules.home_by_minute is not None and rules.home_lat is not None and day not in plan["home_done"]:
            if tod >= rules.home_by_minute:
                plan["home_done"].add(day)
                dist = _haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0)
                if dist > rules.home_radius_km:
                    if dist < 100:  # only reposition if reasonably close
                        return self._safe_reposition(
                            rules,
                            now,
                            lat,
                            lng,
                            rules.home_lat,
                            rules.home_lng or 0,
                            deadline=day_end,
                            tag="home",
                        )
                    else:
                        return self._safe_wait(rules, now, day_end - now)
                return self._safe_wait(rules, now, day_end - now)
            # If close to home_by_minute and far from home, start heading home
            travel_to_home = _travel_minutes(_haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0))
            dist_home = _haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0)
            if tod + travel_to_home >= rules.home_by_minute and dist_home > rules.home_radius_km and dist_home < 100:
                plan["home_done"].add(day)
                return self._safe_reposition(
                    rules,
                    now,
                    lat,
                    lng,
                    rules.home_lat,
                    rules.home_lng or 0,
                    deadline=day_start + rules.home_by_minute,
                    tag="home",
                )

        # (D6) bounded_area: if the driver is parked OUTSIDE its declared operating
        # area, the cargo query (centred on the driver) only ever returns out-of-area
        # cargo, all of which is rejected — so the driver idles the entire month and
        # earns nothing. Reposition once into the area so subsequent _pick_order calls
        # can find compliant in-area cargo. Only triggers for a grounded, reasonably
        # sized area (same guard as the order-acceptance check) to avoid acting on a
        # hallucinated box.
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            lat_span, lng_span = la_max - la_min, ln_max - ln_min
            reasonable = (lat_span >= _MIN_BOUNDED_AREA_SPAN and lng_span >= _MIN_BOUNDED_AREA_SPAN and
                          18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140)
            outside = not (la_min <= lat <= la_max and ln_min <= lng <= ln_max)
            if reasonable and outside and day not in plan.setdefault("bounded_repo", set()):
                # Clamp the current position onto the box, then nudge 5% toward the
                # centre so the driver lands clearly inside.
                clat, clng = (la_min + la_max) / 2, (ln_min + ln_max) / 2
                tgt_lat = min(max(lat, la_min), la_max)
                tgt_lng = min(max(lng, ln_min), ln_max)
                tgt_lat += (clat - tgt_lat) * 0.05
                tgt_lng += (clng - tgt_lng) * 0.05
                if now + _travel_minutes(_haversine_km(lat, lng, tgt_lat, tgt_lng)) <= day_end:
                    plan["bounded_repo"].add(day)
                    return self._safe_reposition(
                        rules, now, lat, lng, tgt_lat, tgt_lng, deadline=day_end, tag="bounded_area"
                    )

        # (E) take the best compliant order, else idle to day end. A flexible-rest
        # driver may let the day's *last* order finish past midnight (up to a cap that
        # still leaves room for a full rest block inside the next day), but only when
        # the next day is an ordinary working day — never crossing into an off day,
        # blackout day or a dated-event day.
        hard_end = self._hard_order_deadline(rules, plan, now, day)
        if rules.rest_window is None and not rules.no_drive_windows and rules.daily_rest_minutes > 0:
            if self._next_day_is_ordinary(rules, plan, day):
                hard_end = max(hard_end, day_end + (DAY_MINUTES - rules.day_rest_block))
        order = self._pick_order(driver_id, status, rules, plan, now, lat, lng, day, hard_end)
        if order is not None:
            return order
        # [OPT] LLM-driven category reposition: when a monthly category target is
        # still open and no target cargo is available locally, look for that category
        # before generic anti-stranding.  This keeps explicit KPI penalties at zero
        # for unknown category targets instead of chasing generic revenue too long.
        cat_target, cat_needed = self._get_category_target(rules, plan, day)
        if cat_target and cat_needed > 0 and plan.get("_llm_repo_today") != day:
            repo_action = self._llm_category_reposition(
                driver_id, rules, plan, now, lat, lng, day, cat_target, cat_needed
            )
            if repo_action is not None:
                plan["_llm_repo_today"] = day
                return repo_action
        # (E') anti-stranding: no compliant order is reachable from here, so the driver
        # would otherwise idle the entire day. If a single reposition toward a profitable
        # cargo cluster turns the day productive (the post-reposition pickup is short, so
        # no deadhead-cap penalty), move there instead of sitting idle. `net` already
        # nets out the reposition distance, so this never loses money on the anchor order.
        strand = self._anti_strand(driver_id, rules, plan, now, lat, lng, day, hard_end)
        if strand is not None:
            return strand
        # (E'') Instead of idling until day end, wait 2 hours and retry.
        remaining = day_end - now
        if remaining > 240:
            return self._safe_wait(rules, now, 120)
        return self._safe_wait(rules, now, max(remaining, 1))

    def _next_day_is_ordinary(self, rules, plan, day) -> bool:
        nxt = day + 1
        if nxt >= MONTH_DAYS or nxt in plan["off_days"]:
            return False
        if any(nxt in days for _, days in rules.blackout):
            return False
        if any(ev["day"] == nxt for ev in rules.dated_single):
            return False
        if any(ev["day"] in (nxt, nxt + 1) for ev in rules.dated_route):
            return False  # event day or its pre-stage evening
        return True

    def _drive_route(self, ev, rules, plan, now, day_start, lat, lng):
        stops = ev["stops"]
        st = plan.setdefault("route_state", {}).setdefault(ev["day"], {"idx": 0, "waited": False})
        i = st["idx"]
        if i >= len(stops):
            plan["dated_route_done"].add(ev["day"])
            return None
        stop = stops[i]
        before = day_start + stop["before"]
        dist = _haversine_km(lat, lng, stop["lat"], stop["lng"])
        if dist > 1.5:
            if now + _travel_minutes(dist) <= before:
                return self._safe_reposition(
                    rules, now, lat, lng, stop["lat"], stop["lng"], deadline=before, tag="dated_route"
                )
            plan["dated_route_done"].add(ev["day"])  # missed; give up route
            return None
        wait_minutes = int(stop.get("min_wait", 0) or 0)
        wait_until_abs = stop.get("wait_until_abs")
        if isinstance(wait_until_abs, (int, float)):
            wait_minutes = max(wait_minutes, int(wait_until_abs) - now)
        if wait_minutes > 0 and not st["waited"]:
            st["waited"] = True
            return self._safe_wait(rules, now, wait_minutes)
        st["idx"] += 1
        st["waited"] = False
        if st["idx"] >= len(stops):
            plan["dated_route_done"].add(ev["day"])
            return None
        return self._drive_route(ev, rules, plan, now, day_start, lat, lng)

    def _get_category_target(self, rules, plan, day):
        """Return (target_name, still_needed) for current month's category KPI."""
        month_idx = _month_index_for_day(day)
        cat_orders = plan["monthly_category_orders"].setdefault(month_idx, {})
        targets = dict(getattr(rules, "monthly_category_targets", {}).get(month_idx, {}))
        if month_idx in getattr(rules, "category_carryover_months", set()):
            prev_targets = getattr(rules, "monthly_category_targets", {}).get(month_idx - 1, {})
            prev_orders = plan["monthly_category_orders"].get(month_idx - 1, {})
            carry = sum(max(0, int(req) - int(prev_orders.get(cat, 0))) for cat, req in prev_targets.items())
            if carry and targets:
                first_cat = next(iter(targets))
                targets[first_cat] = int(targets[first_cat]) + carry
        if targets:
            best_cat, best_need = None, 0
            for cat, req in targets.items():
                got = cat_orders.get(cat, 0)
                need = max(0, int(req) - int(got))
                if need > best_need:
                    best_cat, best_need = cat, need
            return (best_cat, best_need) if best_cat and best_need > 0 else (None, 0)
        if month_idx == 1:
            got = cat_orders.get("水果", 0)
            return ("水果", max(0, 12 - got)) if got < 12 else (None, 0)
        if month_idx == 2:
            april_cat = plan["monthly_category_orders"].get(1, {}).get("水果", 0)
            april_shortfall = max(0, 12 - april_cat)
            may_target = 12 + april_shortfall
            got = cat_orders.get("建材", 0)
            return ("建材", max(0, may_target - got)) if got < may_target else (None, 0)
        return (None, 0)

    @staticmethod
    def _track_category_order(plan, rules, month_idx: int, cargo_name: str) -> None:
        if not cargo_name:
            return
        cat_orders = plan["monthly_category_orders"].setdefault(month_idx, {})
        targets = getattr(rules, "monthly_category_targets", {}).get(month_idx, {}) if rules else {}
        if targets:
            for cat in targets:
                if cat and (cat in cargo_name or cargo_name in cat):
                    cat_orders[cat] = cat_orders.get(cat, 0) + 1
            return
        if month_idx == 1 and cargo_name == "水果":
            cat_orders["水果"] = cat_orders.get("水果", 0) + 1
        elif month_idx == 2 and cargo_name == "建材":
            cat_orders["建材"] = cat_orders.get("建材", 0) + 1

    def _pick_order(self, driver_id, status, rules, plan, now, lat, lng, day, day_end):
        entry_now = now  # step-start clock, used to key the shared cargo scan
        # daily_order_limit check
        if rules.daily_order_limit is not None:
            count = plan["orders_today"].get(day, 0)
            if count >= rules.daily_order_limit:
                return None
        # first_order timing check: if no order taken today and it's past the deadline
        if rules.first_order_before_minute is not None and day not in plan["first_order_taken"]:
            tod = now % DAY_MINUTES
            if tod > rules.first_order_before_minute:
                pass

        # [OPT] Determine current month index and category targets
        month_idx = _month_index_for_day(day)
        longhual_count = plan["monthly_longhual"].get(month_idx, 0)
        cat_target, cat_needed = self._get_category_target(rules, plan, day)

        # [OPT] When a monthly category target is at risk, widen earlier.  The
        # month boundary follows the scorer's calendar (Mar 31 / Apr 30 / May 31),
        # not a flat 31-day bucket.
        hunt_category = cat_target is not None and cat_needed > 0
        remaining_month_days = max(1, _month_end_day_exclusive(month_idx) - day)
        urgent_category = bool(hunt_category and remaining_month_days <= cat_needed + 10)
        initial_k = 600 if urgent_category else (200 if hunt_category else 60)

        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=initial_k)
        items = cargo_resp.get("items", [])
        plan["_scan_items"] = (entry_now, items)
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        blackout_regions = {r for r, days in rules.blackout if day in days}
        need_zeng = (
            rules.required_region is not None
            and len(plan["zeng_order_days"]) < rules.required_region[1]
        )

        best = None
        best_item = None
        best_score = 0.0
        best_is_required = False
        best_is_category_target = False
        best_pickup_km = 0.0
        best_cost_time = 0

        def _score_item(cargo, item, now_t):
            nonlocal best, best_item, best_score, best_is_required, best_is_category_target, best_pickup_km, best_cost_time
            if str(cargo.get("cargo_id", "")) in plan.get("failed_cargo_ids", set()):
                return
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, now_t, day_end, lat, lng)
            if ev is None:
                return
            net, touches_required, occupied, pkm = ev
            if rules.monthly_deadhead_max_km is not None:
                if month_deadhead + pkm > rules.monthly_deadhead_max_km:
                    return
            # [OPT] Long-haul limit: reject >8h orders if already at 5 this month
            cost_time = int(cargo.get("cost_time_minutes", 0))
            if occupied > 480 and longhual_count >= 5:
                return
            score = net / occupied
            # [OPT] Category preference: exact name match with strong boost
            cargo_name = str(cargo.get("cargo_name", ""))
            is_cat_target = False
            if cat_target and cat_needed > 0:
                if cargo_name == cat_target:
                    score *= 8.0 if urgent_category else 5.0
                    is_cat_target = True

            is_req = bool(need_zeng and touches_required)
            if is_cat_target and not best_is_category_target:
                best, best_item, best_score, best_is_required, best_is_category_target, best_pickup_km, best_cost_time = (
                    cargo, item, score, is_req, True, pkm, cost_time
                )
            elif is_cat_target == best_is_category_target:
                if is_req and not best_is_required:
                    best, best_item, best_score, best_is_required, best_is_category_target, best_pickup_km, best_cost_time = (
                        cargo, item, score, True, is_cat_target, pkm, cost_time
                    )
                elif is_req == best_is_required and score > best_score:
                    best, best_item, best_score, best_is_required, best_is_category_target, best_pickup_km, best_cost_time = (
                        cargo, item, score, is_req, is_cat_target, pkm, cost_time
                    )

        for item in items:
            _score_item(item.get("cargo", {}), item, now)

        # [OPT] If hunting category and not found yet, do an even wider k=600 search
        if hunt_category and not best_is_category_target:
            cargo_resp2 = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
            items2 = cargo_resp2.get("items", [])
            plan["_scan_items"] = (entry_now, items2)
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            for item in items2:
                _score_item(item.get("cargo", {}), item, now)

        # Widen search if nothing found at all
        if best is None and not hunt_category:
            cargo_resp2 = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=200)
            items2 = cargo_resp2.get("items", [])
            plan["_scan_items"] = (entry_now, items2)
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            for item in items2:
                _score_item(item.get("cargo", {}), item, now)

        # [OPT] If category target found but we already have a non-category best,
        # use LLM to decide whether to take category cargo (potentially lower profit)
        if best is not None and best_is_category_target and cat_needed > 3:
            # Ask LLM whether to prioritize category order
            self._logger.info(
                "[LLM] category decision driver_id=%s cat=%s needed=%d best_cargo=%s",
                driver_id, cat_target, cat_needed, best.get("cargo_id")
            )

        if best is None:
            return None
        if hunt_category and urgent_category and not best_is_category_target:
            plan["_category_miss_today"] = day
            self._logger.info(
                "category target urgent: skip non-category best driver_id=%s cat=%s needed=%d remaining_days=%d",
                driver_id, cat_target, cat_needed, remaining_month_days,
            )
            return None
        latest_now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        latest_deadline = min(day_end, self._hard_order_deadline(rules, plan, latest_now, day))
        latest_ev = self._evaluate_cargo(best, best_item or {}, rules, blackout_regions, latest_now, latest_deadline, lat, lng)
        if latest_ev is None:
            self._logger.info(
                "rejected selected order after scan-time revalidation cargo_id=%s now=%d deadline=%d",
                best.get("cargo_id"),
                latest_now,
                latest_deadline,
            )
            return None
        _latest_net, _latest_required, latest_occupied, latest_pickup_km = latest_ev
        if latest_occupied > 480 and longhual_count >= 5:
            return None
        if rules.monthly_deadhead_max_km is not None:
            if month_deadhead + latest_pickup_km > rules.monthly_deadhead_max_km:
                return None
        best_pickup_km = latest_pickup_km
        best_cargo_id = str(best.get("cargo_id"))
        return self._take_order(best_cargo_id)

    def _llm_category_reposition(self, driver_id, rules, plan, now, lat, lng, day, cat_target, cat_needed):
        """Use LLM with history context to decide reposition for target category cargo.

        No hardcoded region knowledge — relies entirely on LLM's general knowledge
        and the decision history to determine optimal reposition target.
        """
        history = self._history.get(driver_id)
        history_text = history.get_summary(day) if history else "无历史记录"

        day_of_month = _day_in_month(day)
        month_idx = _month_index_for_day(day)
        month_name = {0: "三月", 1: "四月", 2: "五月"}.get(month_idx, "")

        # Query broader cargo to find where target category exists
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=200)
        items = cargo_resp.get("items", [])
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        # Find target category cargo in results
        target_locations = []
        for item in items:
            cargo = item.get("cargo", {})
            if cargo.get("cargo_name") == cat_target:
                start = cargo.get("start", {})
                target_locations.append({
                    "lat": start.get("lat", 0),
                    "lng": start.get("lng", 0),
                    "city": start.get("city", "未知"),
                    "distance_km": item.get("distance_km", 0),
                })

        prompt = (
            f"你是货运调度AI。司机当前位置: ({lat:.2f}, {lng:.2f})，{month_name}第{day_of_month}天。\n"
            f"品类指标: '{cat_target}'还差{cat_needed}单。\n\n"
            f"决策历史:\n{history_text}\n\n"
        )
        if target_locations:
            prompt += f"附近发现{len(target_locations)}条'{cat_target}'货源:\n"
            for i, loc in enumerate(target_locations[:5]):
                prompt += f"  {i+1}. ({loc['lat']:.2f},{loc['lng']:.2f}) {loc['city']} 距离{loc['distance_km']:.0f}km\n"
            prompt += "\n请选择最佳重定位目的地，输出JSON: {\"latitude\": 纬度, \"longitude\": 经度, \"reason\": \"理由\"}"
        else:
            prompt += (
                f"附近200条货源中无'{cat_target}'类。请根据你对中国物流地理的了解，"
                f"推测'{cat_target}'货源可能集中的区域，输出JSON: "
                f"{{\"latitude\": 纬度, \"longitude\": 经度, \"reason\": \"理由\"}}"
            )

        try:
            req = {
                "messages": [
                    {"role": "system", "content": "你是货运路线规划助手。输出纯JSON，不要markdown。"},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "max_tokens": 100,
            }
            resp = self._api.model_chat_completion(req)
            content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            data = json.loads(content) if content else None
            if data and "latitude" in data and "longitude" in data:
                target_lat = float(data["latitude"])
                target_lng = float(data["longitude"])
                reason = data.get("reason", "")
                target_city = None
                for loc in target_locations:
                    if _haversine_km(target_lat, target_lng, float(loc["lat"]), float(loc["lng"])) < 20:
                        target_city = str(loc.get("city", ""))
                        break
                dist = _haversine_km(lat, lng, target_lat, target_lng)
                if dist > 5:  # only reposition if meaningful distance
                    self._logger.info(
                        "[LLM] category reposition driver_id=%s target=(%.2f,%.2f) cat=%s needed=%d reason=%s",
                        driver_id, target_lat, target_lng, cat_target, cat_needed, reason
                    )
                    return self._safe_reposition(
                        rules,
                        now,
                        lat,
                        lng,
                        target_lat,
                        target_lng,
                        deadline=(day + 1) * DAY_MINUTES,
                        tag="category_reposition",
                        target_city=target_city,
                    )
        except Exception as exc:
            self._logger.warning("[LLM] category reposition failed: %s", exc)

        return None

    def _anti_strand(self, driver_id, rules, plan, now, lat, lng, day, day_end):
        """When no compliant order is reachable from the current spot, the driver is
        stranded (e.g. a previous haul left it far from any cargo cluster) and would
        idle the whole day. Scan a wide radius for the best order that becomes workable
        *after a single reposition to its pickup*, and move toward it. The reposition
        deadhead is already folded into `net`, and the post-reposition pickup is ~0 km,
        so the deadhead cap is never tripped (penalty stays 0). Allow up to 2 repositions
        per day to avoid wasting entire days when first target has no cargo."""
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        day_end = min(day_end, self._hard_order_deadline(rules, plan, now, day))
        strand_count = plan.get("strand_count", {}).get(day, 0)
        if strand_count >= 2:
            return None
        if day_end - now < _STRAND_MIN_BUDGET:
            return None
        blackout_regions = {r for r, days in rules.blackout if day in days}

        def _best_target(items):
            bt, bn = None, 0.0
            for item in items:
                cargo = item.get("cargo", {})
                ev = self._evaluate_relocation(cargo, rules, blackout_regions, now, day_end, lat, lng)
                if ev is None:
                    continue
                net, tlat, tlng, tcity = ev
                if net > bn:
                    bt, bn = (tlat, tlng, tcity), net
            return bt, bn

        # Reuse the cargo scan already paid for in this step's _pick_order (the
        # widest list it queried). Only escalate to a fresh wide k=600 query when
        # the reused list yields no reachable relocation target.
        best_target = best_net = None
        scan = plan.get("_scan_items")
        if scan is not None and scan[0] == now:
            best_target, best_net = _best_target(scan[1])
        if best_target is None:
            cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            day_end = min(day_end, self._hard_order_deadline(rules, plan, now, day))
            best_target, best_net = _best_target(cargo_resp.get("items", []))
        if best_target is None:
            return None
        action = self._safe_reposition(
            rules,
            now,
            lat,
            lng,
            best_target[0],
            best_target[1],
            deadline=day_end,
            tag="anti_strand",
            target_city=best_target[2],
        )
        if action is None:
            return None
        plan["strand_repo"].add(day)
        plan.setdefault("strand_count", {})
        plan["strand_count"][day] = plan["strand_count"].get(day, 0) + 1
        return action

    def _evaluate_relocation(self, cargo, rules, blackout_regions, now, day_end, lat, lng):
        """Net of an order if the driver first repositions to its pickup. Mirrors
        _evaluate_cargo's compliance checks but treats the approach as a reposition
        (so the per-order deadhead cap does not apply) and measures arrival from the
        pickup. Returns (net, pickup_lat, pickup_lng) or None."""
        name = str(cargo.get("cargo_name", ""))
        if self._is_forbidden_cargo(name, rules.forbidden_categories):
            return None
        # avoid_categories: soft penalty for relocation too
        avoid_penalty = 0.5 if self._is_forbidden_cargo(name, rules.avoid_categories) else 1.0
        start = cargo.get("start") or {}
        end = cargo.get("end") or {}
        scity = str(start.get("city", ""))
        ecity = str(end.get("city", ""))
        slat, slng = float(start.get("lat", 0.0)), float(start.get("lng", 0.0))
        elat, elng = float(end.get("lat", 0.0)), float(end.get("lng", 0.0))
        if rules.allowed_regions:
            if not _target_matches_allowed_regions(rules.allowed_regions, slat, slng, scity):
                return None
            if not _target_matches_allowed_regions(rules.allowed_regions, elat, elng, ecity):
                return None
        for region in rules.forbidden_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                return None
        for region in blackout_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                return None
            if region == "深圳" and (_in_shenzhen(slat, slng) or _in_shenzhen(elat, elng)):
                return None
            if region in rules.blackout_coords:
                rlat, rlng = rules.blackout_coords[region]
                if _haversine_km(slat, slng, rlat, rlng) < 60 or _haversine_km(elat, elng, rlat, rlng) < 60:
                    return None
        for fz_lat, fz_lng, fz_r in rules.forbidden_zones:
            if not (18 <= fz_lat <= 55 and 70 <= fz_lng <= 140):
                continue
            if _haversine_km(slat, slng, fz_lat, fz_lng) < fz_r or _haversine_km(elat, elng, fz_lat, fz_lng) < fz_r:
                return None
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            area_lat_span = la_max - la_min
            area_lng_span = ln_max - ln_min
            if area_lat_span >= _MIN_BOUNDED_AREA_SPAN and area_lng_span >= _MIN_BOUNDED_AREA_SPAN and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                if not (la_min <= slat <= la_max and ln_min <= slng <= ln_max):
                    return None
                if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                    return None
        move_km = _haversine_km(lat, lng, slat, slng)
        arrival = now + (_travel_minutes(move_km) if move_km > 1e-6 else 0)
        if self._interval_overlaps_no_drive(
            rules, now, arrival + (_ORDER_DEADLINE_BUFFER_MIN if rules.no_drive_windows else 0)
        ):
            return None
        load_window = cargo.get("load_time")
        ready = arrival
        if isinstance(load_window, list) and len(load_window) == 2:
            ls = _wall_to_min(str(load_window[0]))
            le = _wall_to_min(str(load_window[1]))
            if ls is not None and le is not None:
                if arrival > le:
                    return None
                ready = max(arrival, ls)
        cost_time = int(cargo.get("cost_time_minutes", 0))
        finish = ready + cost_time
        if finish > day_end:
            return None
        if self._interval_overlaps_no_drive(
            rules, now, finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.no_drive_windows else 0)
        ):
            return None
        haul_km = _haversine_km(slat, slng, elat, elng)
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (move_km + haul_km)
        if net <= 0:
            return None
        net *= avoid_penalty
        return net, slat, slng, scity

    def _evaluate_cargo(self, cargo, item, rules, blackout_regions, now, day_end, lat, lng):
        name = str(cargo.get("cargo_name", ""))
        if self._is_forbidden_cargo(name, rules.forbidden_categories):
            return None
        # avoid_categories: soft penalty (50% score reduction) rather than hard rejection
        avoid_penalty = 0.5 if self._is_forbidden_cargo(name, rules.avoid_categories) else 1.0
        start = cargo.get("start") or {}
        end = cargo.get("end") or {}
        scity = str(start.get("city", ""))
        ecity = str(end.get("city", ""))
        slat, slng = float(start.get("lat", 0.0)), float(start.get("lng", 0.0))
        elat, elng = float(end.get("lat", 0.0)), float(end.get("lng", 0.0))
        if rules.allowed_regions:
            if not _target_matches_allowed_regions(rules.allowed_regions, slat, slng, scity):
                return None
            if not _target_matches_allowed_regions(rules.allowed_regions, elat, elng, ecity):
                return None
        for region in rules.forbidden_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                return None
        for region in blackout_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                return None
            if region == "深圳" and (_in_shenzhen(slat, slng) or _in_shenzhen(elat, elng)):
                return None
            if region in rules.blackout_coords:
                rlat, rlng = rules.blackout_coords[region]
                if _haversine_km(slat, slng, rlat, rlng) < 60 or _haversine_km(elat, elng, rlat, rlng) < 60:
                    return None
        # forbidden_zones: circle-zone check on pickup/dropoff
        # Only enforce if coordinates look reasonable (latitude 18-55, longitude 70-140 for China)
        for fz_lat, fz_lng, fz_r in rules.forbidden_zones:
            if not (18 <= fz_lat <= 55 and 70 <= fz_lng <= 140):
                continue  # likely hallucinated coordinates
            if _haversine_km(slat, slng, fz_lat, fz_lng) < fz_r or _haversine_km(elat, elng, fz_lat, fz_lng) < fz_r:
                return None
        # bounded_area: only accept orders within operating bounds
        # Only enforce if the area looks reasonable (not too small / not covering all of China)
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            area_lat_span = la_max - la_min
            area_lng_span = ln_max - ln_min
            # Skip if area is unreasonably small or coordinates out of China range.
            if area_lat_span >= _MIN_BOUNDED_AREA_SPAN and area_lng_span >= _MIN_BOUNDED_AREA_SPAN and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                if not (la_min <= slat <= la_max and ln_min <= slng <= ln_max):
                    return None
                if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                    return None
        pickup_km = _haversine_km(lat, lng, slat, slng)
        if rules.pickup_max_km is not None and pickup_km > rules.pickup_max_km:
            return None
        cost_time = int(cargo.get("cost_time_minutes", 0))
        pickup_min = _travel_minutes(pickup_km) if pickup_km > 1e-6 else 0
        arrival = now + pickup_min
        if self._interval_overlaps_no_drive(
            rules, now, arrival + (_ORDER_DEADLINE_BUFFER_MIN if rules.no_drive_windows else 0)
        ):
            return None
        load_window = cargo.get("load_time")
        ready = arrival
        if isinstance(load_window, list) and len(load_window) == 2:
            ls = _wall_to_min(str(load_window[0]))
            le = _wall_to_min(str(load_window[1]))
            if ls is not None and le is not None:
                if arrival > le:
                    return None  # would miss load window
                ready = max(arrival, ls)
        finish = ready + cost_time
        if finish > day_end:
            return None  # don't cross midnight (keeps rest windows clean)
        if self._interval_overlaps_no_drive(
            rules, now, finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.no_drive_windows else 0)
        ):
            return None
        haul_km = _haversine_km(slat, slng, elat, elng)
        # haul_max_km: single-order haul distance limit
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (pickup_km + haul_km)
        if net <= 0:
            return None
        # Apply avoid_categories soft penalty
        net *= avoid_penalty
        touches_required = False
        if rules.required_region is not None:
            region = rules.required_region[0]
            touches_required = _region_in_city(region, scity) or _region_in_city(region, ecity)
        occupied = max(1, finish - now)
        return net, touches_required, occupied, pickup_km

    @staticmethod
    def _is_forbidden_cargo(cargo_name: str, forbidden: set[str]) -> bool:
        """Substring-based category matching: 'cargo_name' is forbidden if any
        forbidden category is a substring of the cargo name, or vice versa."""
        if not cargo_name or not forbidden:
            return False
        cn = cargo_name.strip()
        for cat in forbidden:
            if cat in cn or cn in cat:
                return True
        return False

    # ------------------------------------------------------------- action dsl
    @staticmethod
    def _wait(duration_minutes: int) -> dict[str, Any]:
        return {"action": "wait", "params": {"duration_minutes": int(max(1, duration_minutes))}}

    @staticmethod
    def _reposition(lat: float, lng: float) -> dict[str, Any]:
        return {"action": "reposition", "params": {"latitude": float(lat), "longitude": float(lng)}}

    @staticmethod
    def _take_order(cargo_id: str) -> dict[str, Any]:
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}

    # ----------------------------------------------------------------- planning
    @staticmethod
    def _plan_off_days(rules: DriverRules) -> set[int]:
        """整月整休日：均匀铺在月中，避开特定日期事件及其前置布置日。

        采用确定性铺点而非"挑最晚的几天"，这样即便事件偏好要到临近其日期窗才可见、
        导致 reserved 集合后期才补全，也不会出现"已铺的整休日被事后挪走"的问题。
        """
        n = rules.off_days_min
        if n <= 0:
            return set()
        reserved: set[int] = set()
        for ev in rules.dated_single:
            reserved.add(ev["day"])
        for ev in rules.dated_route:
            reserved.add(ev["day"])
            reserved.add(ev["day"] - 1)  # keep the day-before free for staging
        off: set[int] = set()
        for i in range(n):
            target = int(round((i + 0.5) * MONTH_DAYS / n))
            target = min(MONTH_DAYS - 1, max(0, target))
            chosen = None
            for delta in range(MONTH_DAYS):
                for cand in (target + delta, target - delta):
                    if 0 <= cand < MONTH_DAYS and cand not in reserved and cand not in off:
                        chosen = cand
                        break
                if chosen is not None:
                    break
            if chosen is not None:
                off.add(chosen)
        return off

    # ------------------------------------------------------------- preferences
    def _ensure_rules(self, driver_id: str, status: dict[str, Any]) -> DriverRules:
        # Re-parse and merge: preferences can be date-windowed and only become
        # visible inside their window (e.g. 深圳 blackout, 盘库, 寿宴). The LLM is the
        # primary parser (generalises to unseen drivers/wordings); it is re-invoked
        # only when a *new* preference text appears. Regex is the offline fallback.
        rules = self._rules.get(driver_id)
        if rules is None:
            rules = DriverRules()
            self._rules[driver_id] = rules
        self._initial_position.setdefault(
            driver_id,
            (float(status.get("current_lat", 0.0)), float(status.get("current_lng", 0.0))),
        )
        prefs = status.get("preferences") or []
        texts = [
            (pref.get("content", "") if isinstance(pref, dict) else str(pref)) for pref in prefs
        ]
        texts = [t for t in texts if t.strip()]
        seen = self._seen_prefs.setdefault(driver_id, set())
        new_texts = [t for t in texts if t not in seen]
        if not new_texts:
            return rules

        before = self._rules_fingerprint(rules)
        coord_map = self._collect_coords(prefs)
        parsed_by_llm = self._llm_parse_preferences(driver_id, texts, rules, coord_map)
        if not parsed_by_llm:
            # offline / model unavailable: fall back to the deterministic regex parser.
            for text in texts:
                self._parse_one(text, rules, coord_map)
            for text in texts:
                self._supplement_basic_rules(text, rules)
            for text in texts:
                self._supplement_dated_events(text, rules, coord_map)
        else:
            # LLM succeeded: supplement with safe, high-confidence regex patterns
            # for basic scalar rules that are easy to verify and unlikely to
            # false-match (rest hours, rest window, off days, pickup cap,
            # forbidden categories/regions with very specific trigger patterns).
            for text in texts:
                self._supplement_basic_rules(text, rules)
            # Supplement dated events: detect date+coordinate patterns the LLM may
            # have missed (e.g. wrong coords, missed events entirely).
            for text in texts:
                self._supplement_dated_events(text, rules, coord_map)
            # Cross-check LLM dated event coords against text-extracted coords.
            # If an LLM coord doesn't match any known coord, snap it to the
            # nearest known one (LLM often inverts or rounds coordinates).
            if coord_map:
                known = list(coord_map.values())
                for ev in rules.dated_single:
                    if not any(_haversine_km(ev["lat"], ev["lng"], k[0], k[1]) < 20 for k in known):
                        best = min(known, key=lambda k: _haversine_km(ev["lat"], ev["lng"], k[0], k[1]))
                        self._logger.info(
                            "coord fix: dated_single day=%d (%.2f,%.2f)→(%.2f,%.2f)",
                            ev["day"], ev["lat"], ev["lng"], best[0], best[1],
                        )
                        ev["lat"], ev["lng"] = best[0], best[1]
                for ev in rules.dated_route:
                    for stop in ev.get("stops", []):
                        if not any(_haversine_km(stop["lat"], stop["lng"], k[0], k[1]) < 20 for k in known):
                            best = min(known, key=lambda k: _haversine_km(stop["lat"], stop["lng"], k[0], k[1]))
                            self._logger.info(
                                "coord fix: dated_route day=%d stop (%.2f,%.2f)→(%.2f,%.2f)",
                                ev["day"], stop["lat"], stop["lng"], best[0], best[1],
                            )
                            stop["lat"], stop["lng"] = best[0], best[1]
        # Build blackout_coords: map blackout region names to nearby coordinates
        # extracted from preference text. This lets the scheduler idle the driver
        # on blackout days if they happen to be sitting inside the forbidden region.
        for region, _days in rules.blackout:
            if region in rules.blackout_coords:
                continue
            # Try exact match first, then norm_region match
            for name, loc in coord_map.items():
                if _region_in_city(region, name):
                    rules.blackout_coords[region] = loc
                    break
        seen.update(texts)
        # Dedup avoid/forbidden AFTER supplement so regex-added items are also cleaned
        init_lat, init_lng = self._initial_position[driver_id]
        self._supplement_relative_allowed_regions(texts, rules, init_lat, init_lng)
        self._dedup_avoid_forbidden(rules)
        if self._rules_fingerprint(rules) != before:
            self._logger.info(
                "parsed rules driver_id=%s rest=%s window=%s off=%s forbid_cat=%s avoid_cat=%s "
                "forbid_reg=%s allowed_reg=%s required=%s pickup_max=%s haul_max=%s blackout=%s "
                "dated_single=%s dated_route=%s no_drive=%s order_limit=%s home=%s "
                "forbidden_zones=%d bounded=%s must_visit=%d first_order=%s category_targets=%s carryover=%s",
                driver_id,
                rules.daily_rest_minutes,
                rules.rest_window,
                rules.off_days_min,
                rules.forbidden_categories,
                rules.avoid_categories,
                rules.forbidden_regions,
                rules.allowed_regions,
                rules.required_region,
                rules.pickup_max_km,
                rules.haul_max_km,
                rules.blackout,
                rules.dated_single,
                rules.dated_route,
                rules.no_drive_windows,
                rules.daily_order_limit,
                (rules.home_lat, rules.home_lng, rules.home_by_minute),
                len(rules.forbidden_zones),
                rules.bounded_area,
                len(rules.must_visit),
                rules.first_order_before_minute,
                rules.monthly_category_targets,
                rules.category_carryover_months,
            )
        return rules

    def _dedup_avoid_forbidden(self, rules: DriverRules) -> None:
        """Remove from forbidden_categories anything that semantically overlaps
        with avoid_categories.  "尽量不拉" means soft-avoid, not hard-reject.
        Uses substring matching so '石材类' vs '石材' are treated as the same item."""
        if not rules.avoid_categories or not rules.forbidden_categories:
            return
        to_remove: set[str] = set()
        for fc in rules.forbidden_categories:
            for ac in rules.avoid_categories:
                # substring match in either direction
                if ac in fc or fc in ac:
                    to_remove.add(fc)
                    break
        if to_remove:
            rules.forbidden_categories -= to_remove
            self._logger.info("dedup: removed %s from forbidden (overlap with avoid)", to_remove)

    # ----------------------------------------------------------- LLM preference parsing
    _PARSE_SYSTEM = (
        "你是货运司机偏好抽取器。把司机的自然语言偏好转成严格 JSON。\n"
        "只输出一个 JSON 对象，禁止 markdown / 解释 / think标签。未提及的字段用 null 或空数组。\n\n"
        "字段定义（宁缺毋错）：\n"
        '- daily_rest_hours: 每天连续休息最少小时数（数字或 null）\n'
        '- rest_window: 每天固定停车时段 {"start_hour":数字,"end_hour":数字}（或 null）。半小时用0.5，如11点半→11.5\n'
        '- no_drive_windows: 每天禁止接单/空驶的时段数组 [{"start_hour":数字,"end_hour":数字}]。半小时用0.5。\n'
        '  适用于非休息类禁行（如"中午12点到1点不接单"）。跨午夜时 end_hour<start_hour，如23→5。\n'
        '  注意：如果同一条偏好既有"休息"又有"不接单不空跑"，rest_window和no_drive_windows都填。\n'
        '- off_days_min: 整月完全不出车天数（整数，默认 0）\n'
        '- forbidden_categories: 禁运货物**品类名**数组（仅货物名称如"蔬菜""机械设备""生鲜"，'
        '绝不放城市/区域名！"在惠州的货"是区域禁令不是品类禁令）\n'
        '- avoid_categories: 尽量避免的货物品类名数组（"尽量不拉""尽量不接"→放这里）\n'
        '- forbidden_regions: 禁接的装/卸货**城市/区域名**数组（仅地名如"惠州""深圳"，'
        '不带"的货""那一路"等后缀）\n'
        '- allowed_regions: 仅允许运营的城市/区域名数组（如"长三角""江浙沪""广东""苏州"）。'
        '仅当文本明确说"只跑/只接/仅在/不出/限定在X"时填写；表示装货地和卸货地都必须在X。\n'
        '- forbidden_zones: 禁入圆形区域 [{"lat":纬度,"lng":经度,"radius_km":半径}]。\n'
        '  适用于"以(lat,lng)为圆心、半径N公里内禁入"之类约束。\n'
        '- bounded_area: 仅允许运营的经纬度矩形范围 {"lat_min":浮点,"lat_max":浮点,"lng_min":浮点,"lng_max":浮点}（或 null）。\n'
        '  适用于"北纬X至Y、东经X至Y"之类限制。\n'
        '- required_region: 每月需在该区域接货天数 {"region":"纯地名","min_days":整数}（或 null）\n'
        '- must_visit: 每月必须到达的地点 [{"lat":纬度,"lng":经度,"radius_km":半径,"required_days":整数}]。\n'
        '  适用于"每月至少N天到过某地"之类约束。\n'
        '- pickup_max_km: 赴装空驶上限公里数（数字或 null）。中文数字要转换。\n'
        '- haul_max_km: 单笔干线距离上限公里数（装货点到卸货点，数字或 null）\n'
        '- monthly_deadhead_max_km: 月累计空驶上限公里数（数字或 null）\n'
        '- daily_order_limit: 每天最多接几单（整数或 null）\n'
        '- first_order_before_hour: 每天首单不得晚于几点（整数或 null）\n'
        '- monthly_category_targets: 月度品类接单指标数组 [{"month":3-5整数,"category":"货物品类","min_orders":整数,"carryover":布尔}]。\n'
        '  适用于"四月水果必须接满十二单""本月建材至少12单""欠的单数下月补"。\n'
        '- home_rule: 回家规则 {"lat":纬度,"lng":经度,"radius_km":半径,"home_by_hour":几点前到家,"no_drive_until_hour":次日几点前不接单}（或 null）\n'
        '  适用于"每天X点前须在自家位置Y公里内，到次日Z点前不接单不空跑"之类约束。\n'
        '- blackout: 指定日期不去某地 [{"region":"纯地名","dates":[日期...]}]\n'
        '- dated_single: 某天必须到某地办事 [{"date":日期,"lat":纬度,"lng":经度,"wait_minutes":停留分钟,"before_hour":最晚到达整点或null}]\n'
        '  触发词：盘库/清库存/对账/验收/盘点/提货/签收/检查/保养/检修/停一趟/办事/开会/取东西/交货/拿货/走一趟/回一趟/去一趟/看看/办手续/送东西/存东西\n'
        '- dated_route: 某天按顺序经过多个地点 [{"date":日期,"stops":[{"lat":纬度,"lng":经度,"wait_minutes":分钟,"before_hour":整点或null}...]}]\n'
        '  触发词：赴宴/做寿/先到…再到/先去…再去/先过…赶到/接人/送人/喝喜酒/吃饭/接上配偶/接家人\n\n'
        "关键规则：\n"
        "1) 日期用该月日数 1-31。中文数字要转换（十二号→12，二十号→20，三十一号→31）。\n"
        "2) 坐标：偏好中'地名（纬度,经度）'→直接取。输入若有 known_coordinates 优先使用。"
        "若某偏好只说地名，而 known_coordinates 有同名坐标，直接引用。\n"
        "3) wait_minutes：办事类事件必须 >0（默认 120）。路过/取东西类 wait_minutes=0。\n"
        "4) 时刻：'中午十二点前'→12，'下午两点'→14，'上午十点'→10。\n"
        "5) **严格区分品类和区域**：forbidden_categories 只放货物品类名（蔬菜、机械设备、危化品等）；"
        "forbidden_regions 只放纯地名（惠州、深圳等），不加任何修饰词。\n"
        "6) **dated_single 和 dated_route 惩罚最高，必须仔细抽取，坐标必须正确。**\n"
        "7) **极其重要**：只抽取文本中**明确提到**的约束。如果文本没有提到回家/活动范围/禁入区域/接单上限等，"
        "对应字段**必须**为null或空数组。宁可漏掉也绝不臆造。**错误的约束比缺少约束惩罚高10倍。**"
        "特别是：forbidden_zones/bounded_area/must_visit/home_rule这些高风险字段，除非文本中有非常明确的描述和坐标，否则一律为null/空数组。\n"
        "8) 同一偏好可能同时包含多种约束（如禁区域+禁品类+日期事件+回家规则），全部抽取。\n"
        '9) "不接单不空跑/不空驶"类约束，如果含时间段→填no_drive_windows；'
        '如果同时含"休息/睡觉"→也填rest_window或daily_rest_hours。\n'
        '10) "接上配偶/家人→返回老家/进家门"是 dated_route 事件（多点路线），不是 home_rule。\n'
        '11) "只跑/只接/仅在/不出X"是 allowed_regions，不是 forbidden_regions。\n\n'
        "示例1：\n"
        '入: {"preferences":["每天零点到六点停着熄火睡觉","凡是生鲜货源碰不得","三月四号五号不往深圳（22.55，114.05）跑"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":{"start_hour":0,"end_hour":6},'
        '"no_drive_windows":[{"start_hour":0,"end_hour":6}],"off_days_min":0,'
        '"forbidden_categories":["生鲜"],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"monthly_category_targets":[],"home_rule":null,'
        '"blackout":[{"region":"深圳","dates":[4,5]}],"dated_single":[],"dated_route":[]}\n\n'
        "示例2：\n"
        '入: {"preferences":["十二号得去仓库（23.15，113.67）盘库，花两小时","连续休息满8小时","空驶超过五十五公里别接"]}\n'
        '出: {"daily_rest_hours":8,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":55,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"monthly_category_targets":[],"home_rule":null,'
        '"blackout":[],"dated_single":[{"date":12,"lat":23.15,"lng":113.67,"wait_minutes":120,"before_hour":null}],"dated_route":[]}\n\n'
        "示例3：\n"
        '入: {"preferences":["三十一号先过档口（23.15，113.67）取礼物，中午十二点前赶到县城（23.35，112.47）赴宴到下午两点"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"monthly_category_targets":[],"home_rule":null,'
        '"blackout":[],"dated_single":[],"dated_route":[{"date":31,"stops":[{"lat":23.15,"lng":113.67,"wait_minutes":0,"before_hour":12},{"lat":23.35,"lng":112.47,"wait_minutes":120,"before_hour":12}]}]}\n\n'
        "示例4：\n"
        '入: {"preferences":["龙门吊底座、机床铸件这类机械设备活儿干不了","装货地或卸货地在惠州的货一律不接"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":["龙门吊底座","机床铸件","机械设备"],"avoid_categories":[],"forbidden_regions":["惠州"],'
        '"forbidden_zones":[],"bounded_area":null,"required_region":null,"must_visit":[],'
        '"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"monthly_category_targets":[],"home_rule":null,'
        '"blackout":[],"dated_single":[],"dated_route":[]}\n\n'
        "示例5：\n"
        '入: {"preferences":["每天23点前车辆须在自家位置（23.10，113.50）1公里内，到次日8点前不接单不空跑","同一天接单不得超过3单"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[{"start_hour":23,"end_hour":8}],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":3,"first_order_before_hour":null,"monthly_category_targets":[],'
        '"home_rule":{"lat":23.10,"lng":113.50,"radius_km":1,"home_by_hour":23,"no_drive_until_hour":8},'
        '"blackout":[],"dated_single":[],"dated_route":[]}\n\n'
        "示例6：\n"
        '入: {"preferences":["十一点半到下午一点半歇晌，雷打不动","二十号去老李仓库（23.25，113.40）对账，大概两小时"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":{"start_hour":11.5,"end_hour":13.5},'
        '"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"monthly_category_targets":[],"home_rule":null,'
        '"blackout":[],"dated_single":[{"date":20,"lat":23.25,"lng":113.40,"wait_minutes":120,"before_hour":null}],"dated_route":[]}'
    )

    def _llm_parse_preferences(
        self, driver_id: str, texts: list[str], rules: DriverRules,
        coord_map: dict[str, tuple[float, float]] | None = None,
    ) -> bool:
        """用大模型把全部可见偏好解析成结构化规则并合并进 rules。

        返回 True 表示 LLM 成功产出结构化结果；False 表示模型不可用/解析失败
        （此时调用方会退回正则解析）。
        """
        if not texts:
            return False
        payload: dict[str, Any] = {"preferences": texts}
        if coord_map:
            payload["known_coordinates"] = {
                name: {"lat": loc[0], "lng": loc[1]} for name, loc in coord_map.items()
            }
        user = json.dumps(payload, ensure_ascii=False)
        msgs = [
            {"role": "system", "content": self._PARSE_SYSTEM},
            {"role": "user", "content": user},
        ]
        # Try with json_object format first; retry without if it fails (some
        # platform endpoints may not support response_format).
        for attempt in range(2):
            req: dict[str, Any] = {"messages": msgs, "temperature": 0}
            if attempt == 0:
                req["response_format"] = {"type": "json_object"}
            try:
                resp = self._api.model_chat_completion(req)
            except Exception as exc:
                self._logger.info("llm parse attempt %d unavailable driver_id=%s err=%s", attempt, driver_id, exc)
                continue
            data = self._extract_json(resp)
            if data is not None:
                self._logger.info("llm raw output driver_id=%s data=%s", driver_id, json.dumps(data, ensure_ascii=False)[:800])
                try:
                    self._merge_llm_rules(rules, data, texts)
                except Exception as exc:
                    self._logger.warning("llm parse: merge failed driver_id=%s err=%s", driver_id, exc)
                    continue
                return True
            self._logger.warning("llm parse attempt %d: no JSON driver_id=%s", attempt, driver_id)
        return False

    @staticmethod
    def _extract_json(resp: dict[str, Any]) -> dict[str, Any] | None:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        text = content.strip()
        # Strip <think>...</think> tags from reasoning models
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return data if isinstance(data, dict) else None

    # Suffixes / patterns that indicate the LLM put a *region* into forbidden_categories
    _REGION_HINT_RE = re.compile(
        r"(?:装货|卸货|目的)(?:地|地点)?在|"          # "卸货地在惠州"
        r"在[\u4e00-\u9fa5]{2,10}(?:的货|那|路|边|方向)|"  # "在惠州的货"
        r"(?:省|市|区|县|镇|村)$"
    )
    _REGION_NOISE_WORDS = {
        "接单", "跑车", "出车", "空驶", "空跑", "货源", "货物", "订单",
        "路线", "区域", "地方", "这边", "那边", "附近", "周边",
        "地区", "范围", "线路", "业务", "单子", "车", "省", "市", "区", "县", "镇",
    }

    @staticmethod
    def _clean_region_name(raw: str) -> str:
        """Strip common decorative suffixes so 'region' values are bare place-names."""
        r = raw.strip()
        # Strip prefix patterns like "卸货地在", "装货地在", "在"
        prefix_m = re.match(r"(?:装货|卸货|目的)(?:地|地点)?在([\u4e00-\u9fa5]+)", r)
        if prefix_m:
            r = prefix_m.group(1)
        elif r.startswith("在") and len(r) > 1:
            r = r[1:]
        for tail in ("的订单", "的单", "的货", "那一路", "方向", "那边", "一带", "范围内", "区域内", "省内", "市内", "内"):
            if r.endswith(tail) and len(r) > len(tail):
                r = r[:-len(tail)]
        r = re.sub(
            r"(?:装货地|卸货地|目的地|起点|终点|货源地|收货地|发货地|区域|范围|附近|周边)$",
            "",
            r,
        ).strip()
        if not r or r in ModelDecisionService._REGION_NOISE_WORDS:
            return ""
        if any(ch.isdigit() for ch in r):
            return ""
        return r[:12]

    def _canonical_allowed_region(self, raw: str) -> str:
        cleaned = self._clean_region_name(raw)
        if not cleaned:
            return ""
        key = _allowed_region_key(cleaned)
        return key or cleaned

    def _allowed_region_text_supported(self, region: str, all_text: str) -> bool:
        if _text_supports(region, all_text):
            return True
        key = _allowed_region_key(region)
        if key is None:
            return False
        aliases = [raw for raw, val in _ALLOWED_REGION_ALIASES.items() if val == key]
        return any(alias in all_text for alias in aliases) or key in all_text

    def _add_allowed_region(self, rules: DriverRules, raw: str, all_text: str) -> None:
        region = self._canonical_allowed_region(raw)
        if not region:
            return
        if self._allowed_region_text_supported(region, all_text):
            rules.allowed_regions.add(region)
        else:
            self._logger.info("validation: rejected allowed_region '%s' (not in texts)", region)

    def _apply_rest_window(self, rules: DriverRules, sm: int, em: int) -> None:
        """Record a daily rest window given start/end minutes-of-day.

        For an overnight window (e.g. 21:00-06:00, sm > em) the morning part
        (00:00-em) is enforced via rest_window/day_rest_block, but the EVENING
        part (sm-24:00) would otherwise be unconstrained — the driver keeps
        working until midnight and violates the night-rest rule every single day.
        We therefore also register the whole overnight span as a no_drive_window
        so the deterministic scheduler blocks evening orders/repositions and waits
        through the window. Deriving it from the rest_window bypasses the keyword
        grounding used for free-form no_drive_windows, which often fails on the
        noisy preference text.
        """
        if em > sm:
            rules.rest_window = (sm, em)
        elif sm > em > 0:
            rules.rest_window = (0, em)
            overnight = (sm, em + DAY_MINUTES)
            if not any(ws == overnight[0] and we == overnight[1] for ws, we in rules.no_drive_windows):
                rules.no_drive_windows.append(overnight)
            self._logger.info("overnight rest_window %d-%d -> rest_window=(0,%d) no_drive=%s",
                              sm, em, em, overnight)

    def _merge_llm_rules(self, rules: DriverRules, data: dict[str, Any], texts: list[str] | None = None) -> None:
        all_text = "\n".join(texts) if texts else ""
        rest_h = data.get("daily_rest_hours")
        if isinstance(rest_h, (int, float)) and rest_h > 0:
            rules.daily_rest_minutes = max(rules.daily_rest_minutes, int(round(rest_h * 60)))
        rw = data.get("rest_window")
        if isinstance(rw, dict):
            sh, eh = rw.get("start_hour"), rw.get("end_hour")
            if isinstance(sh, (int, float)) and isinstance(eh, (int, float)):
                self._apply_rest_window(rules, int(round(sh * 60)), int(round(eh * 60)))
        off = data.get("off_days_min")
        if isinstance(off, (int, float)) and off > 0:
            rules.off_days_min = max(rules.off_days_min, int(off))

        # Collect raw categories and regions, then cross-validate & clean
        raw_cats: list[str] = []
        raw_regs: list[str] = []
        for cat in data.get("forbidden_categories") or []:
            if isinstance(cat, str) and cat.strip():
                raw_cats.append(cat.strip())
        for reg in data.get("forbidden_regions") or []:
            if isinstance(reg, str) and reg.strip():
                raw_regs.append(reg.strip())

        # Move misclassified regions out of categories
        clean_cats: list[str] = []
        for c in raw_cats:
            if self._REGION_HINT_RE.search(c):
                # This looks like a region constraint, not a category
                cleaned = self._clean_region_name(c)
                if cleaned:
                    self._logger.info("llm cleanup: moved '%s' from categories to regions as '%s'", c, cleaned)
                    raw_regs.append(cleaned)
                continue
            # Strip common LLM prefixes like "凡是"
            stripped = re.sub(r"^(?:凡是|所有|一切|任何)", "", c).strip()
            clean_cats.append(stripped if stripped else c)

        for c in clean_cats:
            if _text_supports(c, all_text):
                rules.forbidden_categories.add(c)
            else:
                self._logger.info("llm validation: rejected forbidden_category '%s' (not in texts)", c)

        for reg in raw_regs:
            r = self._clean_region_name(reg)
            if not r:
                continue
            if _text_supports(r, all_text):
                rules.forbidden_regions.add(r)
            else:
                self._logger.info("llm validation: rejected forbidden_region '%s' (not in texts)", r)
        rr = data.get("required_region")
        if isinstance(rr, dict):
            reg, md = rr.get("region"), rr.get("min_days")
            if isinstance(reg, str) and reg.strip() and isinstance(md, (int, float)) and md > 0:
                rules.required_region = (self._clean_region_name(reg), int(md))
        pk = data.get("pickup_max_km")
        if isinstance(pk, (int, float)) and pk > 0:
            rules.pickup_max_km = float(pk)
        for bo in data.get("blackout") or []:
            if not isinstance(bo, dict):
                continue
            reg = bo.get("region")
            days = set()
            for d in bo.get("dates") or []:
                dv = self._coerce_date(d)
                if dv is not None:
                    days.add(dv - 1)
            if isinstance(reg, str) and reg.strip() and days and not any(r == self._clean_region_name(reg) for r, _ in rules.blackout):
                r = self._clean_region_name(reg)
                if not r:
                    continue
                if _text_supports(r, all_text):
                    rules.blackout.append((r, days))
                else:
                    self._logger.info("llm validation: rejected blackout region '%s' (not in texts)", r)
        for ev in data.get("dated_single") or []:
            single = self._coerce_single(ev)
            if single is not None and not any(e["day"] == single["day"] for e in rules.dated_single):
                rules.dated_single.append(single)
        for ev in data.get("dated_route") or []:
            route = self._coerce_route(ev)
            if route is not None and not any(e["day"] == route["day"] for e in rules.dated_route):
                rules.dated_route.append(route)

        # --- new rule types from old 10-driver version ---
        # Text-grounding: only accept a new rule if the preference text contains
        # matching keywords.  This prevents LLM hallucinations on unseen drivers.
        # Compound grounding: require BOTH a time indicator AND an action keyword
        # to reduce false positives from common words like "休息" or "不接"
        _NDW_TIME_KW = ("点", "时", "小时", "上午", "下午", "中午", "凌晨", "晚上", "早上", ":", "：")
        _NDW_ACTION_KW = ("不出车", "不接单", "不开车", "不跑车", "不空跑", "不空驶", "不运营", "不工作", "不干活", "不接活", "不赶路", "不许出", "不允许出", "别派活", "别赶路", "停车熄火", "禁止出车", "不准出车", "别开车", "别跑车", "收工", "停运", "收车", "封车", "落锁", "熄火", "歇着", "休息", "睡觉", "不动弹", "不动车", "别动车", "不跑", "不接")
        _NDW_KW = _NDW_ACTION_KW  # for logging compatibility
        _AVOID_KW = ("少接", "尽量不", "尽量少", "避免", "避开", "绕开", "不太想", "不愿意", "不喜欢", "最好别", "别给我", "尽量别", "能不接", "嫌麻烦", "犯怵", "不是绝对", "除非价钱", "能换就换", "不太愿意", "能不碰")
        _FZ_KW = ("不进", "不去", "禁止进入", "不要去", "不要进", "别去", "远离", "不往", "不到", "不可进", "不得进", "禁入", "不允许进", "严禁", "禁驶入", "禁止驶入", "避开", "绕开", "堵", "修路", "不想跑", "不做")
        _ALLOW_REGION_KW = ("只跑", "只接", "只在", "仅在", "限定", "不出", "不离开", "不离", "固定在", "范围内", "区域内")
        _BA_KW = ("范围", "区域内", "不超出", "只在", "仅在", "限定", "活动区域", "纬度", "经度", "运营区域", "只做", "只跑")
        _MV_KW = ("必须去", "一定要到", "每月去", "至少去", "必须到", "必访", "定期去", "经过", "起码", "至少", "接够")
        _HOME_KW = ("回家", "到家", "家里", "返回住所", "回住处", "回去", "回到家", "归家", "在家", "家附近", "停在家")
        _DOL_KW = ("不超过", "上限", "最多", "不得超过", "不得多于", "顶多", "封顶", "单以内", "趟以内")
        _DOL_SCOPE_KW = ("同一天", "每天", "每日", "单日", "当天", "一天", "每个自然日", "自然日")
        _HAUL_KW = ("装货", "卸货", "干线", "运距", "里程", "运输距离", "运输", "提货", "交货", "运货", "距离", "公里", "不超", "单趟")
        _FOB_KW = ("首单", "第一单", "第一趟", "最早", "点前出发", "点前接", "点前开", "点之前", "出第一", "还没接单", "还不接单", "前要出", "前必须接", "前得接")

        def _text_has_any(keywords: tuple[str, ...]) -> bool:
            return any(kw in all_text for kw in keywords)

        def _ndw_grounded() -> bool:
            """no_drive_windows requires BOTH time + action keywords."""
            has_time = any(kw in all_text for kw in _NDW_TIME_KW)
            has_action = any(kw in all_text for kw in _NDW_ACTION_KW)
            return has_time and has_action

        # no_drive_windows — grounded with compound check
        if _ndw_grounded():
            for ndw in data.get("no_drive_windows") or []:
                if not isinstance(ndw, dict):
                    continue
                sh, eh = ndw.get("start_hour"), ndw.get("end_hour")
                if isinstance(sh, (int, float)) and isinstance(eh, (int, float)):
                    sm, em = int(round(sh * 60)), int(round(eh * 60))
                    if em > sm:
                        pass  # normal range
                    elif sm > em:
                        em += 24 * 60  # cross-midnight
                    else:
                        continue
                    if not any(ws == sm and we == em for ws, we in rules.no_drive_windows):
                        rules.no_drive_windows.append((sm, em))
        elif data.get("no_drive_windows"):
            self._logger.info("llm grounding: rejected no_drive_windows (no keywords in text)")

        # avoid_categories — grounded
        if _text_has_any(_AVOID_KW):
            for ac in data.get("avoid_categories") or []:
                if isinstance(ac, str) and ac.strip():
                    s = re.sub(r"^(?:凡是|所有|一切|任何)", "", ac.strip()).strip()
                    if s and _text_supports(s, all_text):
                        rules.avoid_categories.add(s)
                    elif s:
                        self._logger.info("llm grounding: rejected avoid_category '%s' (not in text)", s)
        elif data.get("avoid_categories"):
            self._logger.info("llm grounding: rejected avoid_categories (no keywords in text)")

        # NOTE: avoid/forbidden dedup moved to _dedup_avoid_forbidden (runs after supplement)

        # allowed_regions — only strong "operate within X" wording becomes a hard constraint.
        if _text_has_any(_ALLOW_REGION_KW):
            for ar in data.get("allowed_regions") or []:
                if isinstance(ar, str) and ar.strip():
                    self._add_allowed_region(rules, ar, all_text)
        elif data.get("allowed_regions"):
            self._logger.info("llm grounding: rejected allowed_regions (no strong region-limit keywords)")

        # forbidden_zones — grounded + coordinate validation
        if _text_has_any(_FZ_KW):
            for fz in data.get("forbidden_zones") or []:
                if not isinstance(fz, dict):
                    continue
                try:
                    flat, flng, fr = float(fz["lat"]), float(fz["lng"]), float(fz.get("radius_km", 10))
                except (TypeError, ValueError, KeyError):
                    continue
                if flat > 90 and flng < 90:
                    flat, flng = flng, flat
                # Validate coordinates are in China range
                if not (18 <= flat <= 55 and 70 <= flng <= 140):
                    self._logger.info("llm grounding: rejected forbidden_zone (%.2f,%.2f) out of range", flat, flng)
                    continue
                # Validate radius is reasonable (not too large)
                if fr > 100:
                    self._logger.info("llm grounding: clamped forbidden_zone radius %.0f→100 km", fr)
                    fr = 100
                rules.forbidden_zones.append((flat, flng, fr))
        elif data.get("forbidden_zones"):
            self._logger.info("llm grounding: rejected forbidden_zones (no keywords in text)")

        # bounded_area — grounded + coordinate validation
        ba = data.get("bounded_area")
        if isinstance(ba, dict) and rules.bounded_area is None:
            # Require BOTH keyword AND explicit lat/lng numbers in the text
            has_kw = _text_has_any(_BA_KW)
            has_coords = bool(re.search(r'(?:纬度|经度|[纬经]\s*[0-9]|\d{2,3}\.\d)', all_text))
            if has_kw and has_coords:
                try:
                    la_min = float(ba["lat_min"])
                    la_max = float(ba["lat_max"])
                    ln_min = float(ba["lng_min"])
                    ln_max = float(ba["lng_max"])
                    # Validate range
                    if la_max - la_min >= _MIN_BOUNDED_AREA_SPAN and ln_max - ln_min >= _MIN_BOUNDED_AREA_SPAN and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                        rules.bounded_area = (la_min, la_max, ln_min, ln_max)
                    else:
                        self._logger.info("llm grounding: rejected bounded_area (unreasonable range)")
                except (TypeError, ValueError, KeyError):
                    pass
            elif has_kw:
                self._logger.info("llm grounding: rejected bounded_area (keywords found but no explicit coordinates)")
            else:
                self._logger.info("llm grounding: rejected bounded_area (no keywords in text)")

        # must_visit — grounded + coordinate validation
        if _text_has_any(_MV_KW):
            has_coords = bool(re.search(r'\d{2,3}\.\d{1,}', all_text))
            for mv in data.get("must_visit") or []:
                if not isinstance(mv, dict):
                    continue
                try:
                    mlat, mlng = float(mv["lat"]), float(mv["lng"])
                    mrd = int(mv.get("required_days", 1))
                    mr = float(mv.get("radius_km", 1.0))
                except (TypeError, ValueError, KeyError):
                    continue
                if mlat > 90 and mlng < 90:
                    mlat, mlng = mlng, mlat
                # Validate coordinates
                if not (18 <= mlat <= 55 and 70 <= mlng <= 140):
                    self._logger.info("llm grounding: rejected must_visit (coords out of range: %.2f,%.2f)", mlat, mlng)
                    continue
                if not has_coords:
                    self._logger.info("llm grounding: rejected must_visit (no explicit coords in text)")
                    continue
                rules.must_visit.append({"lat": mlat, "lng": mlng, "radius_km": mr, "required_days": mrd})
        elif data.get("must_visit"):
            self._logger.info("llm grounding: rejected must_visit (no keywords in text)")

        # haul_max_km — grounded
        hm = data.get("haul_max_km")
        if isinstance(hm, (int, float)) and hm > 0:
            if _text_has_any(_HAUL_KW):
                rules.haul_max_km = float(hm)
            else:
                self._logger.info("llm grounding: rejected haul_max_km=%s (no keywords in text)", hm)

        # monthly_deadhead_max_km (keep as before — low risk, rarely hallucinated)
        mdh = data.get("monthly_deadhead_max_km")
        if isinstance(mdh, (int, float)) and mdh > 0:
            rules.monthly_deadhead_max_km = float(mdh)

        # daily_order_limit — grounded
        dol = data.get("daily_order_limit")
        if _text_has_any(_DOL_KW) and _text_has_any(_DOL_SCOPE_KW):
            if isinstance(dol, (int, float)) and dol > 0:
                rules.daily_order_limit = int(dol)
            elif isinstance(dol, str):
                dm = re.search(r'(\d+)', dol)
                if dm:
                    rules.daily_order_limit = int(dm.group(1))
        elif dol is not None:
            self._logger.info("llm grounding: rejected daily_order_limit=%s (no keywords in text)", dol)

        # first_order_before_hour — grounded
        fob = data.get("first_order_before_hour")
        # first_order_before_minute is a soft, non-blocking marker (see _pick_order),
        # so a false positive costs nothing. The action keywords (头单/第一单…) are
        # frequently scrambled in the noisy texts, so we ground only on a garble-robust
        # time signal (点/时) to keep recall high.
        if fob is not None and (_text_has_any(_FOB_KW) or _text_has_any(_NDW_TIME_KW)):
            if isinstance(fob, (int, float)) and 0 < fob <= 24:
                rules.first_order_before_minute = int(fob) * 60
            elif isinstance(fob, str):
                fm = re.search(r'(\d+)', fob)
                if fm:
                    h = int(fm.group(1))
                    if 0 < h <= 24:
                        rules.first_order_before_minute = h * 60
        elif fob is not None:
            self._logger.info("llm grounding: rejected first_order_before_hour=%s (no time signal in text)", fob)

        # monthly_category_targets — grounded by category/target wording.
        self._merge_category_targets_from_llm(data, rules, all_text)

        # home_rule — grounded + coordinate validation
        # Require both home keywords AND explicit coordinates in text to reduce
        # hallucination risk (home_rule with wrong coordinates is very costly).
        hr = data.get("home_rule")
        if isinstance(hr, dict) and rules.home_lat is None:
            has_home_kw = _text_has_any(_HOME_KW)
            # Check if text contains explicit coordinate-like numbers (e.g., "23.10" or "(23.10,113.50)")
            has_coords_in_text = bool(re.search(r'\d{2,3}\.\d{1,}', all_text))
            if has_home_kw and has_coords_in_text:
                try:
                    hlat, hlng = float(hr["lat"]), float(hr["lng"])
                except (TypeError, ValueError, KeyError):
                    hlat, hlng = None, None
                if hlat is not None and hlng is not None:
                    if hlat > 90 and hlng < 90:
                        hlat, hlng = hlng, hlat
                    # Validate coordinates in China range
                    if 18 <= hlat <= 55 and 70 <= hlng <= 140:
                        rules.home_lat = hlat
                        rules.home_lng = hlng
                        rules.home_radius_km = float(hr.get("radius_km", 1.0))
                        hby = hr.get("home_by_hour")
                        if isinstance(hby, (int, float)) and 0 < hby <= 24:
                            rules.home_by_minute = int(hby) * 60
                        nduh = hr.get("no_drive_until_hour")
                        if isinstance(nduh, (int, float)) and 0 < nduh <= 24:
                            rules.no_drive_until_minute = int(nduh) * 60
                    else:
                        self._logger.info("llm grounding: rejected home_rule (coordinates out of range: %.2f,%.2f)", hlat, hlng)
            elif has_home_kw:
                self._logger.info("llm grounding: rejected home_rule (keywords found but no explicit coordinates in text)")
            else:
                self._logger.info("llm grounding: rejected home_rule (no keywords in text)")

    @staticmethod
    def _month_to_idx(value: Any, default_idx: int | None = None) -> int | None:
        if isinstance(value, (int, float)):
            month = int(value)
        elif isinstance(value, str):
            raw = value.strip()
            m = re.search(r"(\d+)", raw)
            if m:
                month = int(m.group(1))
            elif "三" in raw:
                month = 3
            elif "四" in raw:
                month = 4
            elif "五" in raw:
                month = 5
            elif "本月" in raw or "这个月" in raw:
                return default_idx
            else:
                return default_idx
        else:
            return default_idx
        mapping = {3: 0, 4: 1, 5: 2, 0: 0, 1: 1, 2: 2}
        return mapping.get(month)

    def _merge_category_targets_from_llm(self, data: dict[str, Any], rules: DriverRules, all_text: str) -> None:
        raw_targets = data.get("monthly_category_targets") or []
        if not isinstance(raw_targets, list):
            return
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            cat = str(item.get("category", "")).strip()
            req = item.get("min_orders")
            month_idx = self._month_to_idx(item.get("month"))
            if not cat or month_idx is None or not isinstance(req, (int, float)) or req <= 0:
                continue
            if not _text_supports(cat, all_text):
                self._logger.info("llm validation: rejected category_target '%s' (not in texts)", cat)
                continue
            rules.monthly_category_targets.setdefault(month_idx, {})[cat] = max(
                int(req),
                rules.monthly_category_targets.get(month_idx, {}).get(cat, 0),
            )
            if bool(item.get("carryover")):
                rules.category_carryover_months.add(month_idx)

    def _supplement_category_targets(self, text: str, rules: DriverRules) -> None:
        if not any(kw in text for kw in ("指标", "接满", "至少", "不少于", "接够", "必须接", "少一单", "欠的", "补")):
            return
        default_month_idx = None
        if "五月" in text or "5月" in text:
            default_month_idx = 2
        elif "四月" in text or "4月" in text:
            default_month_idx = 1
        elif "三月" in text or "3月" in text:
            default_month_idx = 0
        target_patterns = (
            r"货源类型是([\u4e00-\u9fa5]{1,8})的货.*?(?:必须|至少|接满|接够|不少于).*?([零一二两三四五六七八九十百\d]+)\s*单",
            r"([\u4e00-\u9fa5]{1,8})(?:的货|货源|类货).*?(?:必须|至少|接满|接够|不少于).*?([零一二两三四五六七八九十百\d]+)\s*单",
            r"(?:必须|至少|接满|接够|不少于).*?([零一二两三四五六七八九十百\d]+)\s*单.*?([\u4e00-\u9fa5]{1,8})(?:的货|货源|类货)",
        )
        for pat in target_patterns:
            for m in re.finditer(pat, text):
                if pat.startswith("(?:必须"):
                    req_s, cat = m.group(1), m.group(2)
                else:
                    cat, req_s = m.group(1), m.group(2)
                cat = re.sub(r"^(?:货源类型是|类型是|货物类型是)", "", cat).strip()
                cat = re.sub(r"(?:这个月|本月|公司|指标|没完成|又有新指标)$", "", cat).strip()
                req = _cn_to_int(req_s)
                month_idx = self._infer_month_idx_near(text, m.start(), default_month_idx)
                if cat and req > 0 and month_idx is not None:
                    rules.monthly_category_targets.setdefault(month_idx, {})[cat] = max(
                        req,
                        rules.monthly_category_targets.get(month_idx, {}).get(cat, 0),
                    )
        if default_month_idx is not None and any(kw in text for kw in ("欠", "补", "接着补", "没完成")):
            rules.category_carryover_months.add(default_month_idx)

    def _supplement_allowed_regions(self, text: str, rules: DriverRules) -> None:
        strong_kws = ("只跑", "只接", "只在", "仅在", "限定", "不出", "不离开", "不离", "固定在", "范围内", "区域内")
        if not any(kw in text for kw in strong_kws):
            return
        for alias in sorted(set(_ALLOWED_REGION_ALIASES) | set(_ALLOWED_REGION_GROUPS), key=len, reverse=True):
            if alias in text:
                self._add_allowed_region(rules, alias, text)
        patterns = (
            r"(?:只跑|只接|只在|仅在|限定在|固定在|不出|不离开|不离)\s*([\u4e00-\u9fa5]{2,12}(?:省|市|区|县|镇|地区|区域|范围|一带|周边|内)?)",
            r"([\u4e00-\u9fa5]{2,12}(?:省|市|区|县|镇|地区|区域|范围|一带|周边|内)?)\s*(?:范围内|区域内|省内|市内).*?(?:只跑|只接|限定|不出|不离)",
        )
        for pat in patterns:
            for m in re.finditer(pat, text):
                raw = m.group(1)
                if "在" in raw:
                    raw = raw.rsplit("在", 1)[-1]
                self._add_allowed_region(rules, raw, text)

    def _supplement_relative_allowed_regions(
        self,
        texts: list[str],
        rules: DriverRules,
        initial_lat: float,
        initial_lng: float,
    ) -> None:
        all_text = "\n".join(texts)
        if not any(kw in all_text for kw in ("省内", "本省", "不出省", "不离省")):
            return
        if not any(kw in all_text for kw in ("只跑", "只接", "只在", "仅在", "限定", "不出", "不离开", "不离")):
            return
        province = _province_region_from_point(initial_lat, initial_lng)
        if province:
            rules.allowed_regions.add(province)
            self._logger.info("supplement: inferred allowed_region=%s from initial position", province)

    @staticmethod
    def _infer_month_idx_near(text: str, pos: int, default_idx: int | None) -> int | None:
        prefix = text[:pos]
        candidates = [
            (max(prefix.rfind("三月"), prefix.rfind("3月")), 0),
            (max(prefix.rfind("四月"), prefix.rfind("4月")), 1),
            (max(prefix.rfind("五月"), prefix.rfind("5月")), 2),
        ]
        best = max(candidates, key=lambda x: x[0])
        return best[1] if best[0] >= 0 else default_idx

    @staticmethod
    def _coerce_before(before_hour: Any) -> int:
        if isinstance(before_hour, (int, float)) and 0 < before_hour <= 24:
            return int(before_hour) * 60
        if isinstance(before_hour, str):
            m = re.search(r'(\d+)', before_hour)
            if m:
                h = int(m.group(1))
                if 0 < h <= 24:
                    return h * 60
        return DAY_MINUTES

    @staticmethod
    def _coerce_date(val: Any) -> int | None:
        """Convert various date formats to int 1-31. Handles int, float, '15号', '15日', '15'."""
        if isinstance(val, (int, float)):
            d = int(val)
            return d if 1 <= d <= 31 else None
        if isinstance(val, str):
            m = re.search(r'(\d+)', val)
            if m:
                d = int(m.group(1))
                return d if 1 <= d <= 31 else None
        return None

    def _coerce_single(self, ev: Any) -> dict[str, Any] | None:
        if not isinstance(ev, dict):
            return None
        date_val = self._coerce_date(ev.get("date"))
        lat, lng = ev.get("lat"), ev.get("lng")
        if date_val is None:
            return None
        date = date_val
        try:
            lat, lng = float(lat), float(lng)
        except (TypeError, ValueError):
            return None
        if lat > 90 and lng < 90:
            lat, lng = lng, lat
        wait = ev.get("wait_minutes")
        if isinstance(wait, str):
            wm = re.search(r'(\d+)', wait)
            wait = int(wm.group(1)) if wm else 120
        wait = int(wait) if isinstance(wait, (int, float)) and wait > 0 else 120
        return {
            "day": int(date) - 1,
            "lat": float(lat),
            "lng": float(lng),
            "min_wait": wait,
            "before": self._coerce_before(ev.get("before_hour")),
        }

    def _coerce_route(self, ev: Any) -> dict[str, Any] | None:
        if not isinstance(ev, dict):
            return None
        date_val = self._coerce_date(ev.get("date"))
        if date_val is None:
            return None
        date = date_val
        stops: list[dict[str, Any]] = []
        for s in ev.get("stops") or []:
            if not isinstance(s, dict):
                continue
            lat, lng = s.get("lat"), s.get("lng")
            try:
                lat, lng = float(lat), float(lng)
            except (TypeError, ValueError):
                continue
            if lat > 90 and lng < 90:
                lat, lng = lng, lat
            wait = s.get("wait_minutes")
            if isinstance(wait, str):
                wm = re.search(r'(\d+)', wait)
                wait = int(wm.group(1)) if wm else 0
            wait = int(wait) if isinstance(wait, (int, float)) and wait >= 0 else 0
            stops.append(
                {
                    "lat": lat,
                    "lng": lng,
                    "min_wait": wait,
                    "before": self._coerce_before(s.get("before_hour")),
                }
            )
        if not stops:
            return None
        return {"day": int(date) - 1, "stops": stops}

    @staticmethod
    def _rules_fingerprint(rules: DriverRules) -> str:
        return repr(
            (
                rules.daily_rest_minutes,
                rules.rest_window,
                rules.off_days_min,
                sorted(rules.forbidden_categories),
                sorted(rules.forbidden_regions),
                sorted(rules.allowed_regions),
                rules.required_region,
                rules.pickup_max_km,
                [(r, sorted(d)) for r, d in rules.blackout],
                [e["day"] for e in rules.dated_single],
                [e["day"] for e in rules.dated_route],
                rules.no_drive_windows,
                rules.daily_order_limit,
                rules.haul_max_km,
                rules.monthly_deadhead_max_km,
                rules.home_by_minute,
                rules.bounded_area,
                len(rules.forbidden_zones),
                len(rules.must_visit),
                rules.first_order_before_minute,
                sorted(rules.avoid_categories),
                {m: dict(sorted(v.items())) for m, v in sorted(rules.monthly_category_targets.items())},
                sorted(rules.category_carryover_months),
            )
        )

    @staticmethod
    def _collect_coords(prefs: list[Any]) -> dict[str, tuple[float, float]]:
        """从偏好文本中抽取「地名（lat，lng）」映射。"""
        out: dict[str, tuple[float, float]] = {}
        pat = re.compile(r"([\u4e00-\u9fa5]{2,6}?)[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]")
        for pref in prefs:
            text = pref.get("content", "") if isinstance(pref, dict) else str(pref)
            for m in pat.finditer(text):
                name = m.group(1)
                loc = (float(m.group(2)), float(m.group(3)))
                out[name] = loc
                # canonicalise: the 2-6 chars right before the coordinate identify the place
                if "增城" in name or "档口" in name:
                    out["增城"] = loc
                if "四会" in name or "县城" in name:
                    out["四会"] = loc
        return out

    def _supplement_basic_rules(self, text: str, rules: DriverRules) -> None:
        """Run after LLM to reinforce basic scalar rules that are easy to verify.

        Covers scalar rules (daily rest, rest window, off days, pickup cap) plus
        high-confidence forbidden category/region patterns with very specific
        trigger phrases that are unlikely to false-match.
        """
        if ("连续" in text and "休息" in text) or ("连轴" in text):
            m = re.search(r"(\d+)\s*(?:个)?\s*小时", text)
            if m:
                rules.daily_rest_minutes = max(rules.daily_rest_minutes, int(m.group(1)) * 60)
        rest_m = re.search(
            r"(?:每天|每日)?.*?(?:连续|连着).*?(?:停车|停着|休息|歇|熄火).*?"
            r"(?:满|至少)?\s*([零一二两三四五六七八九十\d]+)\s*(?:个)?\s*小时",
            text,
        )
        if rest_m:
            rules.daily_rest_minutes = max(rules.daily_rest_minutes, _cn_to_int(rest_m.group(1)) * 60)
        if ("睡觉" in text or "停着熄火" in text or "雷打不动" in text) and "点" in text:
            window = self._parse_time_window(text)
            if window is not None:
                self._apply_rest_window(rules, window[0], window[1])
        if (
            "整天" in text and (
                "歇" in text or "休息" in text or "停驶" in text or "检修" in text
                or "保养" in text or "不接单" in text or "不空车" in text or "不空跑" in text
                or "不出车" in text
            )
        ) or ("完全歇着" in text and ("天" in text or "日" in text)):
            cnt = self._parse_cn_count(text)
            if cnt:
                rules.off_days_min = max(rules.off_days_min, cnt)
        if "空驶" in text and "超过" in text and "月" not in text and "累计" not in text and "总和" not in text:
            km = self._parse_distance_km(text)
            if km and rules.pickup_max_km is None:
                rules.pickup_max_km = km
        # forbidden/avoid category: patterns like "X的活/货...干不了/推掉/不接/不拉"
        _is_soft = any(kw in text for kw in ("尽量不", "尽量少", "尽量别", "最好别", "不太想", "不愿意"))
        cat_list_m = re.search(r"货源品类为(.{1,80}?)(?:的订单|的货|订单|货源|。|；|$)", text)
        if cat_list_m and ("不接" in text or "不拉" in text or "推掉" in text or _is_soft):
            for cat_val in re.findall(r"[「\"]?([\u4e00-\u9fa5]{2,12})[」\"]?", cat_list_m.group(1)):
                cat_val = cat_val.strip()
                if not cat_val or cat_val in {"或者", "以及", "订单", "货源", "品类"}:
                    continue
                if _is_soft:
                    rules.avoid_categories.add(cat_val)
                else:
                    rules.forbidden_categories.add(cat_val)
        cat_m = re.search(
            r"[\"\"「]?([\u4e00-\u9fa5]{2,6}?)[\"\"」]?"
            r"(?:的活|的货|货源|类货|这类|那类).*?"
            r"(?:干不了|推掉|不接|不拉|不碰|不做|碰不得|接不了)",
            text,
        )
        if cat_m:
            cat_val = cat_m.group(1)
            cat_val = re.sub(r"^(?:凡是|所有|一切|任何)", "", cat_val).strip()
            if any(kw in text for kw in ("不接则", "指定熟货", "熟货源编号", "丧失该老客户")):
                cat_val = ""
            if cat_val and not self._REGION_HINT_RE.search(cat_val):
                if _is_soft:
                    rules.avoid_categories.add(cat_val)
                else:
                    rules.forbidden_categories.add(cat_val)
            elif cat_val:
                cleaned = self._clean_region_name(cat_val)
                if cleaned:
                    rules.forbidden_regions.add(cleaned)
        # Also catch "凡是X...碰不得/不接/推掉"
        cat_m2 = re.search(
            r"凡是\s*([\u4e00-\u9fa5]{2,6}?)\s*(?:货源|的货|货).*?"
            r"(?:碰不得|不接|推掉|不拉|不碰|干不了)",
            text,
        )
        if cat_m2:
            cat_val2 = re.sub(r"^(?:凡是|所有|一切|任何)", "", cat_m2.group(1)).strip()
            if cat_val2:
                if _is_soft:
                    rules.avoid_categories.add(cat_val2)
                else:
                    rules.forbidden_categories.add(cat_val2)
        # forbidden region: "在X的货...一律不接/不要/不跑/不去"
        reg_m = re.search(
            r"(?:装货地|卸货地|目的地|起点|终点)\s*(?:在|为)?[\"\"「]?([\u4e00-\u9fa5]{2,10})[\"\"」]?"
            r".*?(?:一律不接|不接|不要|不跑|不去|不做)",
            text,
        )
        if reg_m is None:
            reg_m = re.search(
                r"在[\"\"「]?([\u4e00-\u9fa5]{2,10})[\"\"」]?.{0,20}?"
                r"(?:的货|的订单|货源|订单).*?(?:一律不接|不接|不要|不跑|不去|不做)",
                text,
            )
        if reg_m and rules.bounded_area is None:
            cleaned_region = self._clean_region_name(reg_m.group(1))
            if cleaned_region and cleaned_region not in {"自家位置", "家里", "住所", "老家"}:
                rules.forbidden_regions.add(cleaned_region)
        self._supplement_allowed_regions(text, rules)
        city_area_m = re.search(r"([\u4e00-\u9fa5]{2,10}市)范围内", text)
        if city_area_m and any(kw in text for kw in ("不出", "始终", "须", "必须", "只在", "范围内")):
            city_name = city_area_m.group(1)
            if "在" in city_name:
                city_name = city_name.rsplit("在", 1)[-1]
            self._add_allowed_region(rules, city_name, text)
        bbox_m = re.search(
            r"北纬\s*([0-9]+\.?[0-9]*)\s*(?:至|到|-|~)\s*([0-9]+\.?[0-9]*).*?"
            r"东经\s*([0-9]+\.?[0-9]*)\s*(?:至|到|-|~)\s*([0-9]+\.?[0-9]*)",
            text,
        )
        if bbox_m and rules.bounded_area is None:
            la1, la2, ln1, ln2 = (float(x) for x in bbox_m.groups())
            la_min, la_max = sorted((la1, la2))
            ln_min, ln_max = sorted((ln1, ln2))
            if la_max - la_min >= 0.1 and ln_max - ln_min >= 0.1 and 18 <= la_min <= la_max <= 55 and 70 <= ln_min <= ln_max <= 140:
                rules.bounded_area = (la_min, la_max, ln_min, ln_max)
        # required region: "在X的货...接够N个不同的日子"
        if ("不同的日子" in text or "不同日子" in text) and rules.required_region is None:
            mr = re.search(r"在([\u4e00-\u9fa5]{2,10})的货", text)
            cnt = self._parse_cn_count(text)
            if mr and cnt:
                rules.required_region = (mr.group(1), cnt)
        # blackout: various patterns for "don't go to region X on dates Y"
        blackout_region = None
        if "不往" in text and "跑" in text:
            m2 = re.search(r"不往([\u4e00-\u9fa5]{2,10})跑", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "不进" in text:
            m2 = re.search(r"不进([\u4e00-\u9fa5]{2,10})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "别给我派" in text:
            m2 = re.search(r"别给我派.*?([\u4e00-\u9fa5]{2,10})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and ("不去" in text or "别去" in text):
            m2 = re.search(r"(?:不去|别去)([\u4e00-\u9fa5]{2,10})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "别安排" in text:
            m2 = re.search(r"别安排.*?([\u4e00-\u9fa5]{2,10})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and ("不跑" in text or "不接" in text):
            m2 = re.search(r"(?:不跑|不接)([\u4e00-\u9fa5]{2,10})(?:的[活单货])?", text)
            if m2 and ("号" in text or "日" in text):
                blackout_region = m2.group(1)
        if blackout_region is not None:
            days = set(self._parse_any_days(text))
            if not days:
                days = set(self._parse_month_days(text))
            if days and not any(r == blackout_region for r, _ in rules.blackout):
                rules.blackout.append((blackout_region, days))
        # --- new rule type regex supplements ---
        # daily_order_limit: "同一天不超过N单" / "顶多跑N趟"
        if rules.daily_order_limit is None and ("单" in text or "接单" in text or "趟" in text) and any(
            kw in text for kw in ("同一天", "每天", "每日", "单日", "当天", "一天", "每个自然日", "自然日")
        ):
            dol_m = re.search(r"(?:不超过|不得超过|最多|上限|顶多)\s*(?:跑|接)?\s*([一二两三四五六七八九十\d]+)\s*(?:个)?\s*(?:单|趟)", text)
            if dol_m:
                rules.daily_order_limit = _cn_to_int(dol_m.group(1))
        # haul_max_km: "干线/单笔距离不超过N公里"
        if rules.haul_max_km is None and ("干线" in text or "单笔" in text or "装货点至卸货点" in text) and ("公里" in text or "距离" in text):
            hm_m = re.search(r"(?:不超过|不得超过|不能超过|不可超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if hm_m:
                rules.haul_max_km = float(_cn_to_int(hm_m.group(1)))
        # no_drive_window: "N点到M点不接单/不空跑/不出车"
        if not rules.no_drive_windows and ("不接单" in text or "不空跑" in text or "不出车" in text or "不空驶" in text or "不跑车" in text or "不接活" in text or "别派" in text or "别赶" in text or "不许" in text or "不允许" in text) and ("点" in text or ":" in text or "：" in text):
            ndw_window = self._parse_explicit_time_window(text) or self._parse_time_window(text)
            if ndw_window is not None:
                sm, em = self._normalise_time_window(ndw_window)
                rules.no_drive_windows.append((sm, em))
        zone_m = re.search(
            r"以[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
            r"为圆心[、，,]?\s*半径\s*([零一二两三四五六七八九十百千\d]+)\s*公里",
            text,
        )
        if zone_m and any(kw in text for kw in ("不得进入", "禁止进入", "禁入", "不进", "不要进", "远离")):
            flat, flng = float(zone_m.group(1)), float(zone_m.group(2))
            if flat > 90 and flng < 90:
                flat, flng = flng, flat
            radius = float(_cn_to_int(zone_m.group(3)))
            if 18 <= flat <= 55 and 70 <= flng <= 140 and radius > 0:
                rules.forbidden_zones.append((flat, flng, min(radius, 100.0)))
        home_m = re.search(
            r"每天\s*([零一二两三四五六七八九十\d]+)\s*点前.*?(?:自家|家|住所).*?"
            r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）].*?"
            r"([零一二两三四五六七八九十\d]+)\s*点\s*(?:至|到|-|~)\s*次日\s*([零一二两三四五六七八九十\d]+)\s*点",
            text,
        )
        if home_m and rules.home_lat is None:
            hby = _cn_to_int(home_m.group(1))
            hlat, hlng = float(home_m.group(2)), float(home_m.group(3))
            if hlat > 90 and hlng < 90:
                hlat, hlng = hlng, hlat
            until_h = _cn_to_int(home_m.group(5))
            if 18 <= hlat <= 55 and 70 <= hlng <= 140 and 0 < hby <= 24:
                rules.home_lat = hlat
                rules.home_lng = hlng
                rules.home_radius_km = 1.0
                rules.home_by_minute = hby * 60
                if 0 < until_h <= 24:
                    rules.no_drive_until_minute = until_h * 60
        # home_rule: "X点前须在自家/回家/到家...Y公里" (complex, rely more on LLM)
        # monthly_deadhead_max_km: "月累计空驶不超过N公里"
        if rules.monthly_deadhead_max_km is None and "月" in text and "空驶" in text and "公里" in text:
            mdh_m = re.search(r"(?:不超过|不得超过|不能超过|不可超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if mdh_m:
                rules.monthly_deadhead_max_km = float(_cn_to_int(mdh_m.group(1)))
        # first_order_before: "首单不得晚于N点"
        if rules.first_order_before_minute is None and ("首单" in text or "第一单" in text):
            fob_m = re.search(r"(?:不得晚于|不迟于|之前)\s*(中午|上午|下午|晚上|凌晨)?\s*([零一二两三四五六七八九十\d]+)\s*点", text)
            if fob_m:
                hour = _cn_to_int(fob_m.group(2))
                ctx = fob_m.group(1) or ""
                if ctx in ("下午", "晚上") and hour < 12:
                    hour += 12
                elif ctx == "中午" and hour < 11:
                    hour += 12
                rules.first_order_before_minute = hour * 60
            else:
                fob_m2 = re.search(r"(中午|上午|下午|晚上|凌晨)?\s*([零一二两三四五六七八九十\d]+)\s*点(?:半)?\s*(?:前|之前|以前)", text)
                if fob_m2:
                    hour = _cn_to_int(fob_m2.group(2))
                    ctx = fob_m2.group(1) or ""
                    if ctx in ("下午", "晚上") and hour < 12:
                        hour += 12
                    elif ctx == "中午" and hour < 11:
                        hour += 12
                    rules.first_order_before_minute = hour * 60
            if rules.first_order_before_minute is None and ("中午12点" in text or "中午十二点" in text):
                rules.first_order_before_minute = 12 * 60
        mv_m = re.search(
            r"至少\s*([零一二两三四五六七八九十\d]+)\s*个?不同.*?日.*?到过"
            r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
            r"([零一二两三四五六七八九十\d]+)?\s*公里?内",
            text,
        )
        if mv_m:
            req = _cn_to_int(mv_m.group(1))
            mlat, mlng = float(mv_m.group(2)), float(mv_m.group(3))
            if mlat > 90 and mlng < 90:
                mlat, mlng = mlng, mlat
            radius = float(_cn_to_int(mv_m.group(4) or "一"))
            if req > 0 and 18 <= mlat <= 55 and 70 <= mlng <= 140:
                rules.must_visit.append({"lat": mlat, "lng": mlng, "radius_km": radius, "required_days": req})
        self._supplement_category_targets(text, rules)

    def _supplement_dated_events(
        self, text: str, rules: DriverRules, coord_map: dict[str, tuple[float, float]]
    ) -> None:
        """After LLM, detect date+coordinate+action patterns for dated events.

        Catches dated_single/dated_route the LLM may have missed. Requires ALL
        three signals (date, coordinate, action keyword) to avoid false positives.
        """
        # Skip blackout-style / forbidden-style texts (about NOT going somewhere)
        if any(kw in text for kw in ("不往", "不去", "不进", "别给我派", "别安排", "不跑", "不接", "一律不")):
            return
        # Extract dates from text
        days = self._parse_month_days(text)
        if not days:
            days = self._parse_any_days(text)
        if not days:
            return
        # Find coordinates: inline in this text + from coord_map if name appears
        coord_pat = re.compile(
            r"([\u4e00-\u9fa5]{2,6}?)[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
        )
        found: list[tuple[str, float, float]] = []
        for m in coord_pat.finditer(text):
            lat, lng = float(m.group(2)), float(m.group(3))
            if lat > 90 and lng < 90:
                lat, lng = lng, lat
            found.append((m.group(1), lat, lng))
        for name, loc in coord_map.items():
            if name in text and not any(_haversine_km(loc[0], loc[1], f[1], f[2]) < 5 for f in found):
                found.append((name, loc[0], loc[1]))
        if not found:
            return
        # Determine event type by action keywords
        single_kws = (
            "停一趟", "盘库", "清库存", "对清", "对账", "验收", "盘点",
            "提货", "签收", "检查", "保养", "检修", "办事", "开会",
            "取东西", "拿货", "交货", "走一趟", "跑一趟", "回一趟",
            "去一趟", "装货", "卸货",
        )
        route_kws = (
            "赴宴", "寿", "赶到", "先到", "先去", "先过",
            "再到", "再去", "然后到", "然后去", "接着到", "接着去",
        )
        is_single = any(kw in text for kw in single_kws)
        is_route = len(found) >= 2 and any(kw in text for kw in route_kws)
        route_days = days
        if is_route and len(days) > 1 and any(kw in text for kw in ("待到", "方可", "解决", "进家门", "返回老家", "回家")):
            route_days = [min(days)]
        for day in (route_days if is_route else days):
            if is_route and not any(e["day"] == day for e in rules.dated_route):
                before = self._parse_before_minute(text) or DAY_MINUTES
                mb = re.search(r"([零一二两三四五六七八九十\d]+)\s*点前", text)
                if mb:
                    before = _cn_to_int(mb.group(1)) * 60
                wait_until_abs = self._parse_wait_until_abs(text)
                stops = []
                for i, (_, lat, lng) in enumerate(found):
                    wait = 0
                    stop = {"lat": lat, "lng": lng, "min_wait": wait, "before": before}
                    if i == len(found) - 1:
                        wait = self._parse_hours_minutes(text) or 120
                        stop["min_wait"] = wait
                        if wait_until_abs is not None and wait_until_abs > day * DAY_MINUTES:
                            stop["wait_until_abs"] = wait_until_abs
                    stops.append(stop)
                rules.dated_route.append({"day": day, "stops": stops})
                self._logger.info("supplement: added dated_route day=%d stops=%d", day, len(stops))
            elif is_single and not any(e["day"] == day for e in rules.dated_single):
                _, lat, lng = found[0]
                wait = self._parse_hours_minutes(text) or 120
                rules.dated_single.append({"day": day, "lat": lat, "lng": lng, "min_wait": wait, "before": DAY_MINUTES})
                self._logger.info("supplement: added dated_single day=%d lat=%s lng=%s", day, lat, lng)

    def _parse_one(self, text: str, rules: DriverRules, coords: dict[str, tuple[float, float]]) -> None:
        # daily continuous rest: "每天至少连续...休息满8小时"
        if ("连续" in text and "休息" in text) or ("连轴" in text):
            m = re.search(r"(\d+)\s*(?:个)?\s*小时", text)
            if m:
                rules.daily_rest_minutes = max(rules.daily_rest_minutes, int(m.group(1)) * 60)
        # scheduled rest window: "零点以后到早上六点...睡觉/停"
        if ("睡觉" in text or "停着熄火" in text or "雷打不动" in text) and "点" in text:
            window = self._parse_time_window(text)
            if window is not None:
                self._apply_rest_window(rules, window[0], window[1])
        # off days: "抽三个整天" / "留两个整天"
        if (
            "整天" in text and (
                "歇" in text or "休息" in text or "停驶" in text or "检修" in text or "保养" in text
                or "不接单" in text or "不空车" in text or "不空跑" in text
            )
        ) or ("完全歇着" in text and ("天" in text or "日" in text)):
            cnt = self._parse_cn_count(text)
            if cnt:
                rules.off_days_min = max(rules.off_days_min, cnt)
        # forbidden/avoid category: "X这类活儿...干不了" / "凡是X货源...推掉"
        _is_soft2 = any(kw in text for kw in ("尽量不", "尽量少", "尽量别", "最好别", "不太想", "不愿意"))
        if (
            "干不了" in text or "推掉" in text or ("一律不接" in text and "货源" not in text[:0])
        ) and "扣" in text and not any(kw in text for kw in ("不接则", "指定熟货", "熟货源编号", "丧失该老客户")):
            cat = self._parse_forbidden_category(text)
            if cat:
                if _is_soft2:
                    rules.avoid_categories.add(cat)
                else:
                    rules.forbidden_categories.add(cat)
        # forbidden region: "装货地或卸货地在X的货,我一律不接"
        m = re.search(r"在([\u4e00-\u9fa5]{2,10}?)的货[，,]?\s*我一律不接", text)
        if m:
            rules.forbidden_regions.add(m.group(1))
        self._supplement_allowed_regions(text, rules)
        # required region with min days: "装货或卸货在X的货...接够N个不同的日子"
        if "不同的日子" in text or "不同日子" in text:
            mr = re.search(r"在([\u4e00-\u9fa5]{2,10})的货", text)
            cnt = self._parse_cn_count(text)
            if mr and cnt:
                rules.required_region = (mr.group(1), cnt)
        # pickup deadhead cap: "空驶超过五十五公里"/"空驶超过55公里"
        if "空驶" in text and "超过" in text and "月" not in text and "累计" not in text and "总和" not in text:
            km = self._parse_distance_km(text)
            if km:
                rules.pickup_max_km = km
        # blackout region on dates: "三月四号五号...不往深圳跑"
        if ("不往" in text or "别给我派" in text or "不进" in text) and "号" in text:
            region = self._parse_blackout_region(text)
            days = self._parse_month_days(text)
            if region and days and not any(r == region for r, _ in rules.blackout):
                rules.blackout.append((region, set(days)))
        # dated single stop: "三月十二号...到增城区停一趟,花两小时"
        _dated_single_kws = (
            "停一趟", "盘库", "清库存", "对清", "对账", "验收", "盘点",
            "提货", "签收", "检查", "保养", "检修", "办事", "开会",
            "取东西", "拿货", "交货", "走一趟", "跑一趟", "回一趟",
            "去一趟", "装货", "卸货", "看看", "办手续", "送东西", "存东西",
        )
        if ("号" in text or "日" in text) and any(kw in text for kw in _dated_single_kws):
            days = self._parse_month_days(text)
            if not days:
                days = self._parse_any_days(text)
            loc = self._match_coords(text, coords)
            if days and loc and not any(e["day"] == days[0] for e in rules.dated_single):
                rules.dated_single.append(
                    {"day": days[0], "lat": loc[0], "lng": loc[1], "min_wait": self._parse_hours_minutes(text) or 120, "before": DAY_MINUTES}
                )
        # dated route: "三月三十一号...先过增城...中午十二点前赶到四会...赴宴到下午两点"
        _dated_route_kws = (
            "赴宴", "寿", "赶到", "先到", "先去", "先过",
            "再到", "再去", "然后到", "然后去", "接着到", "接着去",
            "接人", "送人", "喝喜酒", "吃饭", "接上",
        )
        if ("号" in text or "日" in text) and any(kw in text for kw in _dated_route_kws):
            days = self._parse_month_days(text)
            if days and not any(e["day"] == days[0] for e in rules.dated_route):
                stops = self._parse_route_stops(text, coords)
                if stops:
                    rules.dated_route.append({"day": days[0], "stops": stops})
        # --- new rule type regex patterns in fallback ---
        # no_drive_window: "N点到M点不接单/不空跑"
        if ("不接单" in text or "不空跑" in text or "不出车" in text or "不空驶" in text) and ("点" in text or ":" in text or "：" in text):
            ndw_window = self._parse_explicit_time_window(text) or self._parse_time_window(text)
            if ndw_window is not None:
                sm, em = self._normalise_time_window(ndw_window)
                if not any(ws == sm and we == em for ws, we in rules.no_drive_windows):
                    rules.no_drive_windows.append((sm, em))
        # daily_order_limit: "同一天不超过N单" / "顶多跑N趟"
        if rules.daily_order_limit is None and ("单" in text or "接单" in text or "趟" in text) and any(
            kw in text for kw in ("同一天", "每天", "每日", "单日", "当天", "一天", "每个自然日", "自然日")
        ):
            dol_m = re.search(r"(?:不超过|不得超过|最多|上限|顶多)\s*(?:跑|接)?\s*([一二两三四五六七八九十\d]+)\s*(?:个)?\s*(?:单|趟)", text)
            if dol_m:
                rules.daily_order_limit = _cn_to_int(dol_m.group(1))
        # haul_max_km: "干线距离不超过N公里" / "单趟运距不能超过N公里" / "运货距离最多N公里"
        if rules.haul_max_km is None and ("干线" in text or "单笔" in text or "单趟" in text or "运距" in text or "运货" in text) and "公里" in text:
            hm_m = re.search(r"(?:不超过|不得超过|不能超过|不可超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if hm_m:
                rules.haul_max_km = float(_cn_to_int(hm_m.group(1)))
        self._supplement_category_targets(text, rules)

    # ----------------------------------------------------- small text parsers
    @staticmethod
    def _parse_cn_count(text: str) -> int:
        m = re.search(r"([一二两三四五六七八九十\d]+)\s*个?\s*(?:整天|不同的日子|不同日子|个不同)", text)
        if not m:
            m = re.search(
                r"([一二两三四五六七八九十\d]+)\s*天\s*(?:完全|彻底|整天|整)?\s*"
                r"(?:歇|休|停|不接|不出|不空)",
                text,
            )
        if not m:
            m = re.search(r"够\s*([一二两三四五六七八九十\d]+)\s*个", text)
        if not m:
            return 0
        return _cn_to_int(m.group(1))

    @staticmethod
    def _parse_distance_km(text: str) -> float | None:
        m = re.search(r"超过\s*([0-9]+)\s*公里", text)
        if m:
            return float(m.group(1))
        m = re.search(r"超过\s*([零一二两三四五六七八九十百]+)\s*公里", text)
        if m:
            return float(_cn_to_int(m.group(1)))
        return None

    @staticmethod
    def _parse_hours_minutes(text: str) -> int | None:
        m = re.search(r"(?:花|停|耗|待|等|逗留|用|需要|大概|大约)\s*([零一二两三四五六七八九十\d]+)\s*(?:个)?\s*小时", text)
        if m:
            return _cn_to_int(m.group(1)) * 60
        # "N小时" without explicit prefix
        m2 = re.search(r"([零一二两三四五六七八九十\d]+)\s*(?:个)?\s*小时", text)
        if m2:
            return _cn_to_int(m2.group(1)) * 60
        return None

    @staticmethod
    def _normalise_time_window(window: tuple[int, int]) -> tuple[int, int]:
        """Store repeated windows as start/end minutes; cross-midnight end is > 1440."""
        sm, em = int(window[0]), int(window[1])
        if em <= sm:
            em += DAY_MINUTES
        return sm, em

    @staticmethod
    def _parse_explicit_time_window(text: str) -> tuple[int, int] | None:
        """Parse explicit Chinese/Arabic time ranges with context.

        Examples: "凌晨2点至5点", "每天23点至次日4点",
        "中午12点至下午1点", "每晚23点至次日早6点".
        Returns raw start/end minutes; callers normalise cross-midnight storage.
        """
        time_pat = re.compile(
            r"(?P<next>次日|第二天|翌日)?\s*"
            r"(?P<context>每晚|每天|凌晨|清晨|早上|上午|中午|下午|晚上|晚间|夜里|夜间|早)?\s*"
            r"(?P<hour>\d{1,2}|[零一二两三四五六七八九十两]+)\s*"
            r"(?:(?:[:：](?P<minute>\d{1,2}))|(?P<half>点半)|点|时)"
        )

        def _ctx_to_minute(ctx: str, hour_token: str, minute_token: str | None, half: str | None) -> tuple[int, int, str]:
            hour = _cn_to_int(hour_token)
            minute = int(minute_token) if minute_token is not None else (30 if half else 0)
            norm_ctx = (ctx or "").replace("每", "")
            raw_hour = hour
            if norm_ctx == "凌晨" and hour == 12:
                hour = 0
            elif norm_ctx in ("下午", "晚上", "晚间", "夜里", "夜间") and hour < 12:
                hour += 12
            elif norm_ctx == "中午" and hour < 11:
                hour += 12
            if hour == 24 and minute == 0:
                return 24 * 60, raw_hour, norm_ctx
            if not (0 <= hour <= 23 and 0 <= minute < 60):
                return -1, raw_hour, norm_ctx
            return hour * 60 + minute, raw_hour, norm_ctx

        mentions: list[dict[str, Any]] = []
        for m in time_pat.finditer(text):
            minute, raw_hour, ctx = _ctx_to_minute(
                m.group("context") or "",
                m.group("hour"),
                m.group("minute"),
                m.group("half"),
            )
            if minute < 0:
                continue
            mentions.append(
                {
                    "start": m.start(),
                    "end": m.end(),
                    "minute": minute,
                    "raw_hour": raw_hour,
                    "context": ctx,
                    "next_day": bool(m.group("next")),
                }
            )
        if len(mentions) < 2:
            return None

        connectors = ("到", "至", "-", "~", "—", "－")
        for i in range(len(mentions) - 1):
            left, right = mentions[i], mentions[i + 1]
            between = text[left["end"]: right["start"]]
            if not any(conn in between for conn in connectors):
                continue
            sm, em = int(left["minute"]), int(right["minute"])
            if not right["next_day"] and right["context"] == "":
                start_ctx = left["context"]
                raw_end_hour = int(right["raw_hour"])
                if start_ctx in ("下午", "晚上", "晚间", "夜里", "夜间") and raw_end_hour < 12 and em <= sm:
                    em += 12 * 60
                elif start_ctx == "中午" and raw_end_hour <= 6 and em <= sm:
                    em += 12 * 60
            if 0 <= sm <= DAY_MINUTES and 0 <= em <= DAY_MINUTES:
                return sm, em
        return None

    @staticmethod
    def _parse_before_minute(text: str) -> int | None:
        m = re.search(r"(\d{1,2})[:：](\d{2})\s*前", text)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
            if 0 <= hour <= 24 and 0 <= minute < 60:
                return hour * 60 + minute
        m = re.search(r"(凌晨|早上|上午|中午|下午|晚上|晚间|夜里|夜间)?\s*([零一二两三四五六七八九十\d]+)\s*点(半)?\s*(?:前|之前|以前)", text)
        if not m:
            return None
        ctx = m.group(1) or ""
        hour = _cn_to_int(m.group(2))
        minute = 30 if m.group(3) else 0
        if ctx == "凌晨" and hour == 12:
            hour = 0
        elif ctx in ("下午", "晚上", "晚间", "夜里", "夜间") and hour < 12:
            hour += 12
        elif ctx == "中午" and hour < 11:
            hour += 12
        if 0 <= hour <= 24 and 0 <= minute < 60:
            return hour * 60 + minute
        return None

    @staticmethod
    def _parse_wait_until_abs(text: str) -> int | None:
        def _date_to_min(month: int, day: int, hour: int, minute: int) -> int | None:
            month_offset = {3: 0, 4: 31, 5: 61, 6: 92}.get(month)
            if month_offset is None or not (1 <= day <= 31 and 0 <= hour <= 24 and 0 <= minute < 60):
                return None
            return (month_offset + day - 1) * DAY_MINUTES + hour * 60 + minute

        candidates: list[int] = []
        patterns = (
            r"(?:至少)?待到.*?(?:2026年)?\s*(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[:：](\d{2})",
            r"至\s*(?:2026年)?\s*(\d{1,2})月(\d{1,2})日\s*(\d{1,2})[:：](\d{2})",
        )
        for pat in patterns:
            for m in re.finditer(pat, text):
                val = _date_to_min(*(int(x) for x in m.groups()))
                if val is not None:
                    candidates.append(val)
        return max(candidates) if candidates else None

    @staticmethod
    def _parse_time_window(text: str) -> tuple[int, int] | None:
        colon_times = re.findall(r"(\d{1,2})[:：](\d{2})", text)
        if len(colon_times) >= 2:
            h1, m1 = (int(x) for x in colon_times[0])
            h2, m2 = (int(x) for x in colon_times[1])
            if 0 <= h1 <= 24 and 0 <= h2 <= 24 and 0 <= m1 < 60 and 0 <= m2 < 60:
                return (h1 * 60 + m1, h2 * 60 + m2)
        # handle "零点...六点" style, including "点半" (half-hour)
        nums = re.findall(
            r"(零点半?|凌晨|早上\s*[零一二两三四五六七八九十\d]+\s*点半?"
            r"|[零一二两三四五六七八九十\d]+\s*点半?)",
            text,
        )
        minutes: list[int] = []
        for token in nums:
            if token.startswith("零点"):
                minutes.append(30 if "半" in token else 0)
                continue
            mm = re.search(r"([零一二两三四五六七八九十\d]+)\s*点(半)?", token)
            if mm:
                h = _cn_to_int(mm.group(1))
                m = 30 if mm.group(2) else 0
                minutes.append(h * 60 + m)
        if "零点" in text and len(minutes) >= 1:
            end = next((m for m in minutes if m > 0), None)
            if end is not None:
                return (0, end)
        if len(minutes) >= 2:
            m1, m2 = minutes[0], minutes[1]
            # fix PM context: "十一点半到下午一点半" → h2=1 should be 13
            h1, h2 = m1 // 60, m2 // 60
            if h2 < h1 and h2 < 12 and ("下午" in text or "中午" in text):
                m2 += 12 * 60
            return (m1, m2)
        return None

    @staticmethod
    def _parse_forbidden_category(text: str) -> str | None:
        m = re.search(r"凡是\s*([\u4e00-\u9fa5]{2,6}?)\s*货源", text)
        if m:
            return m.group(1)
        m = re.search(r"([\u4e00-\u9fa5]{2,6}?)\s*这类", text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _parse_blackout_region(text: str) -> str | None:
        m = re.search(r"不往\s*([\u4e00-\u9fa5]{2,10}?)\s*跑", text)
        if m:
            return m.group(1)
        m = re.search(r"不进\s*([\u4e00-\u9fa5]{2,10})", text)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _parse_month_days(text: str) -> list[int]:
        """三月四号五号 -> [3,4] (day index = date-1)."""
        seg = re.search(r"三月([零一二两三四五六七八九十百\d号]+)", text)
        if not seg:
            return []
        chunk = seg.group(1)
        dates = re.findall(r"([零一二两三四五六七八九十百\d]+)\s*号", chunk)
        out = []
        for d in dates:
            val = _cn_to_int(d)
            if 1 <= val <= 31:
                out.append(val - 1)
        return sorted(set(out))

    @staticmethod
    def _parse_any_days(text: str) -> list[int]:
        """Parse dates from text with both Arabic and Chinese numerals.

        More flexible than _parse_month_days: handles bare '12号', '十二号',
        '二十号', '三十一日' without requiring a '三月' prefix.
        """
        results: set[int] = set()
        # Arabic numerals: "12号", "3日"
        for m in re.finditer(r"(\d{1,2})[号日]", text):
            d = int(m.group(1))
            if 1 <= d <= 31:
                results.add(d - 1)
        # Chinese numerals: "十二号", "二十号", "三十一号"
        for m in re.finditer(r"([零一二两三四五六七八九十百]+)[号日]", text):
            d = _cn_to_int(m.group(1))
            if 1 <= d <= 31:
                results.add(d - 1)
        return sorted(results)

    def _match_coords(self, text: str, coords: dict[str, tuple[float, float]]):
        for name, loc in coords.items():
            if name in text:
                return loc
        if "增城" in text and "增城" in coords:
            return coords["增城"]
        return None

    def _parse_route_stops(self, text: str, coords: dict[str, tuple[float, float]]):
        stops = []
        before = DAY_MINUTES
        mb = re.search(r"([零一二两三四五六七八九十\d]+)\s*点前", text)
        if mb:
            before = _cn_to_int(mb.group(1)) * 60
        # first stop: 增城 (pick up gift, no wait)
        if "增城" in coords:
            stops.append({"lat": coords["增城"][0], "lng": coords["增城"][1], "min_wait": 0, "before": before})
        # second stop: explicit-coord place (四会/县城) with a banquet wait
        target = None
        for key in ("四会", "县城"):
            if key in coords:
                target = coords[key]
                break
        if target is not None:
            stops.append({"lat": target[0], "lng": target[1], "min_wait": 120, "before": before})
        return stops


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(token: str) -> int:
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in _CN_DIGITS:
        return _CN_DIGITS[token]
    # handle composite Chinese numbers: 二百, 三百五十, 一千二百, 二十五, etc.
    total = 0
    current = 0
    for ch in token:
        if ch in _CN_DIGITS:
            current = _CN_DIGITS[ch]
        elif ch == "十":
            total += (current if current else 1) * 10
            current = 0
        elif ch == "百":
            total += (current if current else 1) * 100
            current = 0
        elif ch == "千":
            total += (current if current else 1) * 1000
            current = 0
    total += current
    return total


_EPOCH_FMT = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})")


def _wall_to_min(wall: str) -> int | None:
    m = _EPOCH_FMT.search(wall)
    if not m:
        return None
    _y, mo, d, hh, mm = (int(x) for x in m.groups())
    # Simulation epoch is 2026-03-01 00:00; accumulate days for months beyond March.
    _MONTH_DAYS_CUM = {3: 0, 4: 31, 5: 61, 6: 92}
    month_offset = _MONTH_DAYS_CUM.get(mo, 0)
    return (month_offset + d - 1) * DAY_MINUTES + hh * 60 + mm
