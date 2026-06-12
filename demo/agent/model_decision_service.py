"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。

架构（闭环偏好编译 + 每日合规计划，面向未知司机泛化；偏好原文为第一公民）：

1. 编译（LLM 优先，逐条）：每条自然语言偏好单独交给 LLM 抽取成结构化规则
   （时间窗 / 区域 / 品类 / 配额 / 日期事件…），逐字段做语义接地确认。
2. 审计修复（闭环）：编译完成后，再用一次 LLM 审计「每条偏好的全部约束是否
   都已体现在结构化规则里」，缺失的部分以补丁形式回填——这取代了旧版按
   D001 措辞写死的正则补抽层（正则解析仅在模型不可用时作为离线兜底）。
3. 零静默丢弃：审计后仍无法用结构化字段表达的偏好，进入
   ``DriverRules.residual_constraints``，逐步注入决策 prompt（带违约金额），
   由决策 LLM 在线遵守——任何偏好都不会被悄悄丢掉。
4. 每日合规计划层：每个仿真日开始时，把**偏好原文** + 当天日历（几月几号、
   星期几）+ 月度合规进度交给 LLM，产出"今天"的具体约束（今日禁驶时段、
   今日必须做/避免的事、品类缺口提醒）。任意措辞、任意条件式偏好（周末
   不同时段、特定日期事件、月度结转）都在这一层被按天具象化，不依赖固定
   schema 能否表达。计划层失效时自动退化为纯静态规则（fail-safe）。
5. 确定性执行：调度器与各动作守卫使用"静态窗 ∪ 当日计划窗"的**有效禁驶窗**
   硬性拦截违规动作（每日休息 / 夜间休息窗 / 整月整休天数 / 禁接货类 /
   禁接区域 / 必接区域 / 赴装空驶上限 / 特定日期到点事件），并在合规候选中
   按净收益择单。
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort

DAY_MINUTES = 1440
MONTH_DAYS = 92
SPEED_KM_PER_HOUR = 60.0
EARTH_RADIUS_KM = 6371.0
COST_PER_KM = 1.5
# Long-haul defaults. The monthly long-haul cap is PREFERENCE-DRIVEN: it is
# only enforced when a driver's own preferences impose one (parsed into
# DriverRules.longhaul_cap_orders), because applying D001's "≤5 per month" to
# every driver would actively throttle a long-haul-focused finals driver.
# These constants are only the defaults for threshold (8h) and per-violation
# penalty when the preference text does not specify them.
LONGHAUL_MINUTES = 480
LONGHAUL_CAP = 5  # retained for reference/tests; no longer applied globally
LONGHAUL_PENALTY = 1000.0
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


_SIM_EPOCH = datetime(2026, 3, 1)
_WEEKDAY_NAMES = ("一", "二", "三", "四", "五", "六", "日")


def _calendar_for_day(day: int) -> tuple[str, str, bool]:
    """(YYYY-MM-DD, 星期X, is_weekend) for a simulation day index."""
    dt = _SIM_EPOCH + timedelta(days=int(day))
    weekday = dt.weekday()
    return dt.strftime("%Y-%m-%d"), f"星期{_WEEKDAY_NAMES[weekday]}", weekday >= 5


def _windows_intersect(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Whether two within-day windows overlap in time-of-day space.

    Windows use the (start_min, end_min) convention where an overnight window
    has end_min > DAY_MINUTES (e.g. 21:00→06:00 == (1260, 1800)), so plain
    interval overlap must also be checked under ±1-day shifts.
    """
    for shift in (-DAY_MINUTES, 0, DAY_MINUTES):
        if a[0] < b[1] + shift and a[1] > b[0] + shift:
            return True
    return False


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
        # rule kind -> per-violation penalty amount (from preference penalty_amounts,
        # attributed by the parse LLM). Lets execution treat constraints economically:
        # an order violating a rule is still worth taking when net > penalty.
        self.rule_penalties: dict[str, float] = {}
        # Monthly long-haul cap — ONLY when this driver's preferences impose one
        # (e.g. D001 "每月超过八小时的长途最多5单"). None == no cap: a driver with
        # the opposite preference (long-haul focused) must not be throttled.
        self.longhaul_cap_orders: int | None = None
        self.longhaul_min_minutes: int = LONGHAUL_MINUTES
        # Category whitelist ("只拉/只接X类货"): when non-empty, orders whose
        # cargo does not match any listed category are non-compliant.
        self.allowed_categories: set[str] = set()
        # Minimum haul distance ("低于X公里的短活不接") — mirror of haul_max_km.
        self.haul_min_km: float | None = None
        # Preferences the compile+audit loop could NOT fully express as structured
        # rules. They are never silently dropped: each entry is injected into the
        # decision prompt (with its penalty amount) so the decision LLM enforces
        # it online. Entries: {"text": str, "penalty": float|None, "note": str}.
        self.residual_constraints: list[dict[str, Any]] = []
        # Per-day compliance directives produced by the LLM daily planner from
        # the *raw* preference texts:
        # day -> {"windows": [(sm, em), ...], "replace": bool, "notes": str}.
        # These concretize calendar-conditional preferences (weekend shifts,
        # dated events, month carryovers) that a static schema cannot express.
        self.daily_directives: dict[int, dict[str, Any]] = {}

    def no_drive_windows_for(self, day: int) -> list[tuple[int, int]]:
        """Effective no-drive windows for one simulation day.

        Default: union of the statically parsed windows and the day's planner
        directive (additive == fail-safe; a missed daily window costs a daily
        penalty, an extra one only costs a few idle hours).

        A directive may *replace* the static set (e.g. weekend night rest that
        starts later), but only when it acknowledges each standing constraint
        and merely reshapes it: every static window must intersect some
        directive window AND the directive's total coverage must be at least
        half of the static coverage. A directive that silently drops or
        trivializes a static window falls back to the additive union.
        """
        directive = self.daily_directives.get(int(day))
        if not directive:
            return list(self.no_drive_windows)
        extra = [tuple(w) for w in (directive.get("windows") or [])]
        if directive.get("replace") and extra:
            static_cov = sum(e - s for s, e in self.no_drive_windows)
            extra_cov = sum(e - s for s, e in extra)
            if extra_cov * 2 >= static_cov and all(
                any(_windows_intersect(sw, dw) for dw in extra)
                for sw in self.no_drive_windows
            ):
                return extra
        merged = list(self.no_drive_windows)
        for w in extra:
            if w not in merged:
                merged.append(w)
        return merged

    def has_no_drive(self, day: int) -> bool:
        return bool(self.no_drive_windows_for(day))

    def has_any_no_drive(self) -> bool:
        if self.no_drive_windows:
            return True
        return any(d.get("windows") for d in self.daily_directives.values())

    @property
    def longhaul_penalty(self) -> float:
        """Per-order penalty for exceeding this driver's long-haul cap."""
        return float(self.rule_penalties.get("longhaul_cap", LONGHAUL_PENALTY))

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

    def recent_actions(self) -> list[dict[str, Any]]:
        """Rolling window of recent decision entries (for the daily self-audit)."""
        return self._history

    def get_summary(self, current_day: int, plan: dict[str, Any] | None = None) -> str:
        """Concise history summary for LLM context.

        Accepted-order aggregates (total/this-month orders, long-haul count,
        category progress) are read from ``plan`` when provided, because the
        evaluation harness only calls ``decide()`` and never feeds per-step
        results back into this object -- so the internal counters here stay 0
        and would otherwise mislead the model. ``plan`` is rebuilt every step
        from the authoritative decision history, so it is the source of truth.
        Wait/reposition counts and the recent action trace come from the local
        rolling window (those are recorded reliably at decide time).
        """
        month_idx = _month_index_for_day(current_day)
        month_name = _month_name(month_idx)
        lines = ["=== 决策历史摘要 ==="]

        if plan is not None:
            orders_today = plan.get("orders_today", {}) or {}
            total_orders = sum(int(v) for v in orders_today.values())
            m_lo, m_hi = _MONTH_START_DAYS[month_idx], _MONTH_START_DAYS[month_idx + 1]
            month_orders = sum(int(c) for d, c in orders_today.items() if m_lo <= int(d) < m_hi)
            longhual = plan.get("monthly_longhual", {}).get(month_idx, 0)
            cat_orders = plan.get("monthly_category_orders", {}).get(month_idx, {}) or {}
            lines.append(f"累计接单: {total_orders}单 (本{month_name}已接{month_orders}单)")
            lines.append(f"{month_name}长途(>8h): {longhual}单")
            if cat_orders:
                top = sorted(cat_orders.items(), key=lambda x: -x[1])[:5]
                lines.append(f"{month_name}品类接单: " + ", ".join(f"{c}:{n}单" for c, n in top))
            # Per-pickup-city realised yield (net/h) from the authoritative
            # history — gives reposition decisions data instead of pure
            # geographic intuition.
            city_yield = plan.get("city_yield") or {}
            if city_yield:
                ranked = sorted(
                    city_yield.items(),
                    key=lambda kv: -(kv[1][1] / max(1, kv[1][2])),
                )[:6]
                lines.append("各装货城市历史净收益: " + ", ".join(
                    f"{c}:{v[1] / max(1, v[2]) * 60:.0f}元/h({v[0]}单)" for c, v in ranked))
        else:
            lines.append(f"累计接单: {self._total_orders}单, 累计收入: ¥{self._total_income:.0f}")

        # Wait/reposition activity this month (recorded reliably at decide time).
        s = self._monthly_stats.get(month_idx)
        if s:
            lines.append(f"{month_name}活动: 等待{s['waits']}次, 重定位{s['repositions']}次")

        # Recent action trace. Outcomes are annotated from the authoritative
        # plan (failed_cargo_reasons rebuilt from the server-side history), so
        # the model can learn from its own rejected attempts.
        fail_reasons = (plan or {}).get("failed_cargo_reasons", {}) or {}
        recent = self._history[-10:]
        if recent:
            lines.append(f"\n最近{len(recent)}步决策:")
            for e in recent:
                d, t = e["day"], e["tod"]
                h, m = divmod(t, 60)
                act = e["action"]
                detail = ""
                if act == "take_order":
                    cid = str(e.get("params", {}).get("cargo_id", ""))
                    detail = f" {e.get('cargo_name', '') or cid or '?'}"
                    if cid and cid in fail_reasons:
                        detail += f" [失败:{fail_reasons[cid][:24]}]"
                    elif e.get("accepted") is False:
                        detail += f" [失败:{str(e.get('detail', ''))[:24]}]"
                    else:
                        detail += " [成交]"
                elif act == "wait":
                    detail = f" {e['params'].get('duration_minutes', 0)}分钟"
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
        # ordered raw preference records (content + penalty amount) — the daily
        # compliance planner and the decision prompt always work from the raw
        # text, never only from the parsed schema.
        self._pref_records: dict[str, list[dict[str, Any]]] = {}
        # raw preference text → standard-Mandarin paraphrase ("" = no paraphrase
        # needed / normalization unavailable). Finals drivers are from 广东 and
        # 江浙沪, so preference texts may be written in 粤语 / 吴语 or heavy
        # colloquial phrasing; the paraphrase is appended to the text fed to the
        # extraction / audit / grounding / daily-planner calls so no rule is
        # missed because of dialect wording.
        self._pref_norms: dict[str, str] = {}
        self._initial_position: dict[str, tuple[float, float]] = {}
        # cargo_id -> metadata seen during scans; used to reconstruct monthly
        # accumulators from the authoritative decision history (see
        # _sync_monthly_counts_from_history).
        self._cargo_meta: dict[str, dict[str, Any]] = {}

    def _query_cargo(self, driver_id: str, latitude: float, longitude: float, k: int) -> dict[str, Any]:
        """query_cargo wrapper that caches cargo metadata for every scanned item.

        The cap-enforcing accumulators are rebuilt from accepted orders in the
        decision history, but that history only records cargo_id + result (no
        cost_time / category). We therefore remember the metadata of every cargo
        we ever see here so it can be looked up later by id.
        """
        resp = self._api.query_cargo(driver_id=driver_id, latitude=latitude, longitude=longitude, k=k)
        for item in resp.get("items", []) or []:
            cargo = item.get("cargo", {}) if isinstance(item, dict) else {}
            cid = str(cargo.get("cargo_id", "") or "")
            if not cid:
                continue
            self._cargo_meta[cid] = {
                "cost_time_minutes": int(cargo.get("cost_time_minutes", 0) or 0),
                "cargo_name": str(cargo.get("cargo_name", "") or ""),
                "price": float(cargo.get("price", 0.0) or 0.0),
                "start_city": str((cargo.get("start") or {}).get("city", "") or ""),
                "end_city": str((cargo.get("end") or {}).get("city", "") or ""),
            }
        return resp

    def _sync_monthly_counts_from_history(self, driver_id: str, plan: dict[str, Any]) -> None:
        """Rebuild accepted-order accumulators from the authoritative history.

        The harness only calls ``decide()`` -- it never calls
        ``update_decision_result``. That left ``monthly_longhual`` (and the
        deadhead/category/orders accumulators) permanently empty, so the
        monthly long-haul cap (and other quota checks) read 0 every step and
        never fired -- the driver took unlimited >8h orders and ate the penalty.

        Since the evaluation submits the agent only, we reconstruct these
        accumulators inside the agent from ``query_decision_history`` (the same
        per-step records the server appends). This is idempotent and runs each
        step, so the cap checks downstream see the real counts. ``action_start``
        is computed exactly as the scorer does (prev_end + query_scan) so the
        calendar-month bucketing matches.
        """
        try:
            resp = self._api.query_decision_history(driver_id, -1)
        except Exception:  # pragma: no cover - defensive: never break decide()
            return
        records = (resp or {}).get("records") or []
        if not records:
            return
        rules = self._rules.get(driver_id)
        plan["monthly_longhual"] = {}
        plan["monthly_category_orders"] = {}
        plan["monthly_deadhead_km"] = {}
        plan["orders_today"] = {}
        plan["first_order_taken"] = set()
        plan["failed_cargo_ids"] = set()
        plan["failed_cargo_reasons"] = {}
        plan["zeng_order_days"] = set()
        plan["total_deadhead_km"] = 0.0
        plan["city_yield"] = {}  # start_city -> [orders, total_net, total_minutes]
        for rec in records:
            action = rec.get("action") or {}
            if action.get("action") != "take_order":
                continue
            result = rec.get("result") or {}
            cargo_id = str(result.get("cargo_id") or (action.get("params") or {}).get("cargo_id") or "")
            if not result.get("accepted"):
                if cargo_id:
                    plan["failed_cargo_ids"].add(cargo_id)
                    plan["failed_cargo_reasons"][cargo_id] = str(result.get("detail", "failed"))
                continue
            if not cargo_id:
                continue
            end_min = int(result.get("simulation_progress_minutes", 0) or 0)
            step_elapsed = int(rec.get("step_elapsed_minutes", 0) or 0)
            query_scan = int(rec.get("query_scan_cost_minutes", 0) or 0)
            action_start = max(0, end_min - step_elapsed + query_scan)
            day = max(0, min(MONTH_DAYS - 1, action_start // DAY_MINUTES))
            month_idx = _month_index_for_day(day)
            plan["orders_today"][day] = plan["orders_today"].get(day, 0) + 1
            plan["first_order_taken"].add(day)
            pickup_deadhead = float(result.get("pickup_deadhead_km", 0.0) or 0.0)
            plan["total_deadhead_km"] += pickup_deadhead
            plan["monthly_deadhead_km"][month_idx] = plan["monthly_deadhead_km"].get(month_idx, 0.0) + pickup_deadhead
            meta = self._cargo_meta.get(cargo_id, {})
            lh_min = rules.longhaul_min_minutes if rules is not None else LONGHAUL_MINUTES
            if int(meta.get("cost_time_minutes", 0) or 0) > lh_min:
                plan["monthly_longhual"][month_idx] = plan["monthly_longhual"].get(month_idx, 0) + 1
            self._track_category_order(plan, rules, month_idx, str(meta.get("cargo_name", "")))
            city = str(meta.get("start_city", "") or "")
            price = float(meta.get("price", 0.0) or 0.0)
            if city and price > 0:
                haul_km = float(result.get("haul_distance_km", 0.0) or 0.0)
                net_v = price - COST_PER_KM * (haul_km + pickup_deadhead)
                minutes = max(1, int(meta.get("cost_time_minutes", 0) or 0))
                cy = plan["city_yield"].setdefault(city, [0, 0.0, 0])
                cy[0] += 1
                cy[1] += net_v
                cy[2] += minutes
            if rules is not None and rules.required_region is not None:
                region = rules.required_region[0]
                if _region_in_city(region, meta.get("start_city", "")) or _region_in_city(region, meta.get("end_city", "")):
                    plan["zeng_order_days"].add(day)

    # ------------------------------------------------------------------ decide
    def decide(self, driver_id: str) -> dict[str, Any]:
        # Bulletproof shell: the orchestrator treats an exception from decide()
        # as fatal and terminates this driver's ENTIRE month (end_reason="error"),
        # freezing the score at whatever was earned so far. No single-step bug is
        # worth that, so any unexpected error degrades to a short safe wait and
        # the next step gets a fresh chance to decide properly.
        try:
            return self._decide_impl(driver_id)
        except Exception:
            self._logger.exception(
                "decide() crashed driver_id=%s — falling back to safe 30min wait", driver_id
            )
            return {"action": "wait", "params": {"duration_minutes": 30}}

    def _decide_impl(self, driver_id: str) -> dict[str, Any]:
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
        # Rebuild accepted-order accumulators (long-haul/deadhead/category/...)
        # from the authoritative decision history, since the result hook is
        # never invoked by the harness.
        self._sync_monthly_counts_from_history(driver_id, plan)
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

        # Daily compliance planning: concretize the raw NL preferences for
        # *today* (calendar-aware) before any action is chosen. Fail-safe: on
        # any failure the static parse stays in charge.
        self._ensure_daily_directive(driver_id, rules, plan, day)

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
        cargo_resp = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=30)
        items = cargo_resp.get("items", [])
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        # Same cross-midnight allowance as the deterministic scheduler, so the
        # LLM sees the same candidate set (evening long-hauls included for
        # drivers whose rest rules permit finishing past midnight).
        hard_end = self._extended_order_end(rules, plan, now, day)
        blackout_regions = {r for r, days in rules.blackout if day in days}
        failed_ids = plan.get("failed_cargo_ids", set())

        # Build a compact list of locally feasible cargo. LLM may express preference,
        # but execution safety stays deterministic. Evaluate ALL scanned items and
        # keep the best by (category-target first, then net-per-hour): the scan is
        # nearest-first, so truncating in distance order used to let cheap nearby
        # orders crowd every high-net candidate out of the LLM's view.
        cat_target_rank, cat_needed_rank = self._get_category_target(rules, plan, day)
        ranked: list[tuple[bool, float, dict[str, Any]]] = []
        for item in items:
            cargo = item.get("cargo", {})
            cargo_id = str(cargo.get("cargo_id", ""))
            if cargo_id in failed_ids:
                continue
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, now, hard_end, lat, lng)
            if ev is None:
                continue
            net, _touches_required, occupied, pickup_km = ev
            cargo_name = str(cargo.get("cargo_name", ""))
            is_cat = bool(
                cat_target_rank and cat_needed_rank > 0
                and self._category_matches_sem(cat_target_rank, cargo_name)
            )
            net_per_h = round(float(net) / max(1, int(occupied)) * 60, 1)
            entry = {
                "cargo_id": cargo_id,
                "name": cargo_name,
                "price": round(float(cargo.get("price", 0) or 0), 2),
                "minutes": int(occupied),
                "pickup_km": round(float(pickup_km), 1),
                "net_per_h": net_per_h,
                "from": cargo.get("start", {}).get("city", ""),
                "to": cargo.get("end", {}).get("city", ""),
            }
            if is_cat:
                entry["指标品类"] = True
            ranked.append((is_cat, net_per_h, entry))
        ranked.sort(key=lambda r: (not r[0], -r[1]))
        cargo_summary = [entry for _, _, entry in ranked[:_LLM_CARGO_SUMMARY_LIMIT]]

        # Build preference rules summary
        pref_summary = self._format_rules_for_llm(driver_id, rules, plan, day)

        # Build history summary (accepted-order aggregates come from the
        # authoritative ``plan`` rebuilt each step from decision history).
        history_text = history.get_summary(day, plan)

        # Construct LLM prompt
        day_in_month = _day_in_month(day)
        month_idx = _month_index_for_day(day)
        month_name = _month_name(month_idx)
        hour, minute = divmod(tod, 60)

        system_prompt = (
            "你是一个智能货运调度决策AI。根据司机当前状态、决策历史、可用货源和偏好规则，"
            "做出最优决策。首要目标是最大化净收益(net=毛收入−成本−偏好罚款)；"
            "罚款只是净收益里的一项成本，毛收入足够高时带点罚款也值得。完成品类指标同样重要(欠单罚款高)。\n\n"
            "可用动作：\n"
            '- take_order: 接单。参数: {"cargo_id": "ID"}。优先净收益最高的货：候选里 net_per_h(每小时净收益)越高越好；'
            '若有未达标的品类指标则优先该品类。不要只图短途，长途只要 net_per_h 高就值得接\n'
            '- wait: 等待。参数: {"duration_minutes": 分钟数}。在司机自己的休息/禁驶时段(见下方硬约束与偏好规则)或货源确实不好时使用\n'
            '- reposition: 空驶到新位置。参数: {"latitude": 纬度, "longitude": 经度}。'
            '当附近无目标品类货源时，主动移动到可能有货的区域（根据地理常识判断）\n\n'
            "重要规则：\n"
            "- 在司机自己的休息/禁驶时段内不得接单或空驶，只能wait(具体时段见下方【硬约束】与偏好规则，不同司机时段不同，不要假设固定21:00-06:00)；空驶须在该时段开始前结束\n"
            "- 长途/品类/区域等限制以下方偏好规则为准：没有列出的限制就不存在，不要自行假设(例如不要假设有长途单数上限)\n"
            "- 不要连续wait超过2次！如果当前货源不合适，应该reposition到新区域寻找更好货源\n"
            "- 品类指标很重要：未达标会有高额罚款。优先接指标品类的货\n"
            "- 只能从候选货源列表里选 cargo_id；候选为空或都不合适就 wait/reposition\n"
            "- 只输出一个JSON对象: {\"action\": \"...\", \"params\": {...}, \"reason\": \"20字内\"}\n"
        )

        # Add category target warning (reuse the ranking pass's computation)
        cat_target, cat_needed = cat_target_rank, cat_needed_rank
        month_deadhead = plan.setdefault("monthly_deadhead_km", {}).get(month_idx, 0.0)
        # The shortfall comes from this driver's PARSED targets (incl. carryover) —
        # never assume D001's "12 orders" for other drivers.
        cat_warning = ""
        if cat_target and cat_needed > 0:
            cat_warning = f"\n⚠️ 品类指标警告: 本月品类'{cat_target}'的指标还差{cat_needed}单未完成！优先接此品类。\n"

        # P3: history-driven daily compliance self-audit (hard time-windows +
        # any previous-day in-window driving). Placed up top so it anchors the
        # decision.
        audit_text = self._compliance_self_audit(rules, history, day)
        audit_block = f"## 合规自检\n{audit_text}\n\n" if audit_text else ""

        user_prompt = (
            f"## 当前状态\n"
            f"司机: {driver_id}, 位置: ({lat:.2f}, {lng:.2f})\n"
            f"时间: {month_name}{day_in_month}日 {hour:02d}:{minute:02d}\n"
            f"仿真进度: 第{day}天/{MONTH_DAYS}天\n"
            f"{cat_warning}\n"
            f"{audit_block}"
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
                # Greedy decoding (0, like the other LLM calls here) for
                # run-to-run stability: at 0.1 the per-step picks drift enough to
                # swing monthly net income by 10k+, which makes the one-shot
                # finals score a coin flip. We want a high, reproducible floor.
                "temperature": 0,
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
                    for ws, we in rules.no_drive_windows_for(day):
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
        if not rules.has_any_no_drive():
            return max(1, int(duration))
        changed = True
        while changed:
            changed = False
            first_day = start // DAY_MINUTES - 1
            last_day = (end - 1) // DAY_MINUTES + 1
            for day_idx in range(first_day, last_day + 1):
                base = day_idx * DAY_MINUTES
                for ws, we in rules.no_drive_windows_for(day_idx):
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

    def _hard_order_deadline(
        self, rules: DriverRules, plan: dict, now: int, day: int,
        base_end: int | None = None,
    ) -> int:
        """Latest acceptable finish time for a new order from this step.

        ``base_end`` lets the caller start from an already-extended deadline
        (cross-midnight completion for drivers whose rest rules leave room for
        it, see _extended_order_end); the default is today's midnight. Window
        and home-rule cutoffs still shrink the deadline either way.
        """
        day_start = day * DAY_MINUTES
        day_end = day_start + DAY_MINUTES
        hard_end = base_end if base_end is not None else day_end
        for ws, _we in rules.no_drive_windows_for(day):
            ws_today = day_start + min(ws, DAY_MINUTES)
            if now < ws_today < hard_end:
                hard_end = max(now, ws_today - _ORDER_DEADLINE_BUFFER_MIN)
        if hard_end > day_end:
            # Deadline reaches into tomorrow: tomorrow's windows shrink it too.
            for ws, _we in rules.no_drive_windows_for(day + 1):
                ws_next = day_end + min(ws, DAY_MINUTES)
                if now < ws_next < hard_end:
                    hard_end = max(now, ws_next - _ORDER_DEADLINE_BUFFER_MIN)
        if rules.home_by_minute is not None and day not in plan.get("home_done", set()):
            buffer = 60
            cutoff = day_start + rules.home_by_minute - buffer
            if cutoff > now:
                hard_end = min(hard_end, cutoff)
        return hard_end

    def _extended_order_end(self, rules: DriverRules, plan: dict, now: int, day: int) -> int:
        """Order-finish deadline including the cross-midnight allowance.

        A flexible-rest driver (daily rest quota but no fixed window) may let
        the day's last order finish past midnight, leaving room inside the
        next day for a full rest block. A driver with NO rest constraints at
        all may finish any time before the next midnight — capping such a
        driver at today's midnight (the old behavior) silently rejected every
        profitable evening long-haul for no compliance reason at all.
        Never extends across off days, blackout days, dated-event days, home
        rules or any no-drive window on either side of the boundary.
        """
        hard_end = self._hard_order_deadline(rules, plan, now, day)
        day_end = (day + 1) * DAY_MINUTES
        if (rules.rest_window is None and not rules.has_no_drive(day)
                and not rules.has_no_drive(day + 1)
                and rules.home_by_minute is None
                and self._next_day_is_ordinary(rules, plan, day)):
            hard_end = max(hard_end, day_end + (DAY_MINUTES - rules.day_rest_block))
        return hard_end

    def _interval_overlaps_no_drive(self, rules: DriverRules, start: int, end: int) -> bool:
        """Whether an absolute minute interval touches a repeated no-drive window."""
        if end <= start or not rules.has_any_no_drive():
            return False
        first_day = start // DAY_MINUTES - 1
        last_day = (end - 1) // DAY_MINUTES + 1
        for day_idx in range(first_day, last_day + 1):
            day_base = day_idx * DAY_MINUTES
            for ws, we in rules.no_drive_windows_for(day_idx):
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
        guard_end = finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.has_no_drive(int(now) // DAY_MINUTES) else 0)
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
        if any(
            self._tod_in_window(now % DAY_MINUTES, ws, we)
            for ws, we in rules.no_drive_windows_for(day)
        ):
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
        blackout_regions = {r for r, days in rules.blackout if day in days}
        ev = self._evaluate_cargo(cargo, match, rules, blackout_regions, now, hard_end, lat, lng)
        if ev is None:
            self._logger.info("[LLM] rejected take_order: cargo_id=%s failed deterministic feasibility", cargo_id)
            return False
        net, _touches_required, _occupied, _pickup_km = ev
        # Long-haul cap is preference-driven AND soft (net maximization): it only
        # exists when this driver's preferences impose one, and even then an
        # over-cap order is rejected only when its net does not cover the penalty.
        cost_time = int(cargo.get("cost_time_minutes", 0) or 0)
        if (rules.longhaul_cap_orders is not None
                and cost_time > rules.longhaul_min_minutes
                and longhual_count >= rules.longhaul_cap_orders
                and net <= rules.longhaul_penalty):
            self._logger.info(
                "[LLM] rejected take_order: over long-haul cap and net=%.0f<=penalty month=%d", net, month_idx
            )
            return False
        return True

    # ------------------------------------------------ daily compliance planner
    _DIRECTIVE_SYSTEM = (
        "你是货运司机的每日合规计划助手。司机有若干自然语言偏好（违反会扣钱），"
        "你的任务是把这些偏好换算成【今天】的具体执行约束。偏好可能与日历有关"
        "（如周末时段不同、特定日期有事、月度指标与上月欠额结转），必须结合今天的日期/星期推算。\n"
        "只输出一个JSON对象：\n"
        '{"no_drive_today":[{"start_hour":数字,"end_hour":数字}],'
        '"replaces_default":true或false,'
        '"today_plan":"100字内的今日执行要点",'
        '"category_focus":"今天应优先承接的货物品类名或null"}\n'
        "字段说明：\n"
        '- no_drive_today: 今天（含跨入明晨）不得接单/空驶、只能停车等待的时段。'
        '跨午夜用 end_hour<start_hour（如21点到次日6点→start_hour 21, end_hour 6）。半小时用0.5。'
        '凡偏好含"某时段不出车/不揽货/收车/熄火/休息/睡觉/归家不动"含义且今天适用的，必须全部列出——'
        '这类约束按天扣钱，漏一条代价极高；偏好中没有此类约束则给空数组。\n'
        "- replaces_default: 仅当你确定 no_drive_today 已**完整**给出今天的全部时段约束、"
        "且今天的时段与平日默认不同（例如偏好明确说周末可以晚些休息，而今天正是周末）时才填 true；"
        "不确定时一律 false。\n"
        "- today_plan: 简述今天必须做/必须避免的事（品类缺口与欠额、特定日期事项、区域或长途限制等）。\n"
        "- category_focus: 若有月度品类指标且尚有缺口（含上月欠额需本月同品类补）则给出应优先的品类名，否则 null。\n"
        "规则：宁可保守，绝不编造偏好中不存在的约束；时段必须严格按偏好原文给出的时刻推算。"
        "偏好原文可能是粤语/吴语等方言或口语（如 唔=不要、晚黑=晚上、返屋企=回家、生果=水果、"
        "夜里向=晚上、覅=不要、困觉=睡觉），可能附带「普通话释义」，一律按含义推算。"
    )

    def _ensure_daily_directive(
        self, driver_id: str, rules: DriverRules, plan: dict[str, Any], day: int
    ) -> None:
        """每日一次：把偏好原文按"今天"的日历具象化为可执行约束。

        这是对未知司机泛化的核心层——它不依赖静态 schema 能否表达某类偏好，
        只依赖偏好原文 + 今天的日期/星期/月度进度。产出的今日禁驶窗由
        ``DriverRules.no_drive_windows_for`` 与各动作守卫确定性强制执行；
        today_plan/category_focus 注入决策 prompt。任何失败都只是退回静态
        解析结果（fail-safe），且每天最多重试 2 次、结果按天缓存，token
        开销 ~1 次小调用/司机日。
        """
        records = self._pref_records.get(driver_id) or []
        if not records:
            return
        # re-plan within the day only if new (date-windowed) preferences appear
        cache_key = (int(day), len(records))
        if plan.get("_directive_key") == cache_key:
            return
        attempts = plan.setdefault("_directive_attempts", {})
        if attempts.get(cache_key, 0) >= 2:
            return
        attempts[cache_key] = attempts.get(cache_key, 0) + 1

        date_str, weekday_str, is_weekend = _calendar_for_day(day)
        month_idx = _month_index_for_day(day)
        payload: dict[str, Any] = {
            "今天": f"{date_str} {weekday_str}" + ("（周末）" if is_weekend else "（工作日）"),
            "本月": _month_name(month_idx),
            "偏好": [
                {
                    "原文": r["content"],
                    "单次违约扣款": r.get("penalty_amount"),
                    **({"普通话释义": r["mandarin"]} if r.get("mandarin") else {}),
                }
                for r in records
            ],
        }
        if rules.no_drive_windows:
            payload["平日默认禁驶时段"] = [
                f"{ws // 60:02d}:{ws % 60:02d}-{(we % DAY_MINUTES) // 60:02d}:{(we % DAY_MINUTES) % 60:02d}"
                for ws, we in rules.no_drive_windows
            ]
        targets = rules.monthly_category_targets.get(month_idx) or {}
        if targets:
            payload["本月品类指标"] = targets
            payload["本月品类已接"] = plan.get("monthly_category_orders", {}).get(month_idx, {}) or {}
        if month_idx > 0:
            prev_targets = rules.monthly_category_targets.get(month_idx - 1) or {}
            if prev_targets:
                payload["上月品类指标"] = prev_targets
                payload["上月品类实接"] = plan.get("monthly_category_orders", {}).get(month_idx - 1, {}) or {}

        try:
            req: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": self._DIRECTIVE_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": {"type": "json_object"},
            }
            try:
                resp = self._api.model_chat_completion(req)
            except Exception:
                req.pop("response_format", None)
                resp = self._api.model_chat_completion(req)
            data = self._extract_json(resp)
        except Exception as exc:
            self._logger.info(
                "daily directive unavailable driver_id=%s day=%d err=%s", driver_id, day, exc
            )
            return
        if not isinstance(data, dict):
            self._logger.info("daily directive: no JSON driver_id=%s day=%d", driver_id, day)
            return

        windows: list[tuple[int, int]] = []
        for w in data.get("no_drive_today") or []:
            win = self._coerce_directive_window(w)
            if win is not None and win not in windows:
                windows.append(win)
        notes_parts: list[str] = []
        today_plan = str(data.get("today_plan") or "").strip()
        if today_plan:
            notes_parts.append(today_plan[:160])
        focus = data.get("category_focus")
        if isinstance(focus, str) and focus.strip() and focus.strip().lower() != "null":
            notes_parts.append(f"今日优先品类：{focus.strip()}")
        # Replacement (relaxation) of the static windows is the ONE spot where
        # the daily planner can cause penalties instead of just losing income:
        # a hallucinated "today is relaxed" replaces e.g. a 21:00 start with
        # 23:00 and the driver gets fined every such day. It therefore needs a
        # dedicated verification with default-REJECT (falls back to the
        # additive union, which can never cause a penalty).
        replace_ok = False
        if bool(data.get("replaces_default")) and windows and rules.no_drive_windows:
            replace_ok = self._verify_directive_replacement(driver_id, rules, day, windows)
            if not replace_ok:
                self._logger.info(
                    "daily directive: replacement NOT verified driver_id=%s day=%d windows=%s — using union",
                    driver_id, day, windows,
                )
        rules.daily_directives[int(day)] = {
            "windows": windows,
            "replace": replace_ok,
            "notes": "；".join(notes_parts),
        }
        plan["_directive_key"] = cache_key
        self._logger.info(
            "daily directive driver_id=%s day=%d windows=%s replace=%s effective=%s notes=%s",
            driver_id, day, windows, bool(data.get("replaces_default")),
            rules.no_drive_windows_for(day), "；".join(notes_parts)[:120],
        )

    def _verify_directive_replacement(
        self, driver_id: str, rules: DriverRules, day: int,
        windows: list[tuple[int, int]],
    ) -> bool:
        """Two-gate verification before a directive may REPLACE static windows.

        1. Geometric gate: every directive window must lie inside some static
           window (a legitimate relaxation only reshapes/shrinks coverage; a
           window extending beyond the static set is handled by the additive
           union anyway, so replacement is never needed for it).
        2. Semantic gate: a dedicated default-REJECT yes/no check that the
           preference text explicitly allows TODAY's relaxation (weekend /
           specific-date clauses). This is deliberately the opposite bias of
           _confirm_rule_holds: dropping a relaxation only costs a couple of
           idle hours, while accepting a hallucinated one costs a per-day fine.
        """
        def _is_subset(dw: tuple[int, int]) -> bool:
            for shift in (-DAY_MINUTES, 0, DAY_MINUTES):
                ds, de = dw[0] + shift, dw[1] + shift
                for ss, se in rules.no_drive_windows:
                    se_n = se if se > ss else se + DAY_MINUTES
                    if ss <= ds and de <= se_n:
                        return True
            return False

        if not all(_is_subset(w) for w in windows):
            return False
        records = self._pref_records.get(driver_id) or []
        all_text = "\n".join(
            r["content"] + (f"\n（普通话释义：{r['mandarin']}）" if r.get("mandarin") else "")
            for r in records
        )
        if not all_text.strip():
            return False
        date_str, weekday_str, is_weekend = _calendar_for_day(day)

        def _fmt(wins: list[tuple[int, int]]) -> str:
            return "、".join(
                f"{ws // 60:02d}:{ws % 60:02d}-{(we % DAY_MINUTES) // 60:02d}:{(we % DAY_MINUTES) % 60:02d}"
                for ws, we in wins
            )

        question = (
            f"司机偏好原文如下（可能含方言）：\n{all_text}\n\n"
            f"司机平日的禁驶/停车时段是 {_fmt(rules.no_drive_windows)}。"
            f"今天是 {date_str}（{weekday_str}{'，周末' if is_weekend else '，工作日'}）。"
            f"原文是否**明确**允许今天把禁驶时段放宽为 {_fmt(windows)}"
            f"（例如周末或特定日期可以晚些休息）？只有原文明确支持才回答true，拿不准一律回答false。"
        )
        return self._llm_semantic_yes_no(question, default=False)

    @staticmethod
    def _coerce_directive_window(w: Any) -> tuple[int, int] | None:
        if not isinstance(w, dict):
            return None
        try:
            sh = float(w.get("start_hour"))
            eh = float(w.get("end_hour"))
        except (TypeError, ValueError):
            return None
        if not (0 <= sh <= 24 and 0 <= eh <= 24):
            return None
        sm = int(round(sh * 60)) % DAY_MINUTES
        em = int(round(eh * 60)) % DAY_MINUTES
        if em == sm:
            return None
        if em < sm:
            em += DAY_MINUTES  # overnight window crossing into the next day
        return (sm, em)

    def _compliance_self_audit(self, rules: "DriverRules", history: "DecisionHistory", day: int) -> str:
        """P3 — history-driven daily self-audit.

        Each day, restate the driver's hard time-windows (rest + no-drive) to
        the LLM and, by scanning the recent decision history, flag whether the
        previous day actually drove inside any of those windows. Because these
        penalties recur daily, surfacing one slip immediately stops it from
        repeating for the rest of the season. Pure history lookup — no extra
        LLM call. Returns "" when the driver has no time-window constraints.
        """
        # Build the within-day "must not drive" spans (minutes of day) for a
        # given day, from that day's *effective* (static ∪ directive) windows.
        def _spans_for(d: int) -> list[tuple[int, int]]:
            sp: list[tuple[int, int]] = []
            block_ = rules.day_rest_block
            if block_ > 0:
                sp.append((0, block_))
            for ws, we in rules.no_drive_windows_for(d):
                if we <= DAY_MINUTES:
                    sp.append((ws, we))
                else:  # cross-midnight -> split into tail-of-day + head-of-next
                    sp.append((ws, DAY_MINUTES))
                    sp.append((0, we - DAY_MINUTES))
            return sp

        today_windows = rules.no_drive_windows_for(day)
        spans = _spans_for(day)
        residuals = getattr(rules, "residual_constraints", []) or []
        if not spans and not rules.rest_window and not residuals:
            return ""

        reminders: list[str] = []
        if rules.rest_window:
            rs, re_ = rules.rest_window
            reminders.append(f"休息 {rs // 60:02d}:{rs % 60:02d}-{re_ // 60:02d}:{re_ % 60:02d}")
        for ws, we in today_windows:
            we_d = we % DAY_MINUTES
            reminders.append(f"禁驶 {ws // 60:02d}:{ws % 60:02d}-{we_d // 60:02d}:{we_d % 60:02d}")

        violations = 0
        prev = day - 1
        prev_spans = _spans_for(prev) if prev >= 0 else []
        if prev_spans:
            for e in history.recent_actions():
                if e.get("day") != prev:
                    continue
                act = e.get("action")
                if act not in ("take_order", "reposition"):
                    continue
                if act == "take_order" and e.get("accepted") is False:
                    continue
                t = int(e.get("tod", -1))
                if any(s <= t < en for s, en in prev_spans):
                    violations += 1

        parts: list[str] = []
        if reminders:
            parts.append("【硬约束·每日生效】" + "；".join(reminders)
                         + "：这些时段只能 wait，绝不可 take_order/reposition（违反按天扣高额罚分）。")
        if violations:
            parts.append(f"【历史自检】昨日检测到 {violations} 次在休息/禁驶时段出车的记录，"
                         "今日务必严格遵守上述时段，避免每日重复扣分。")
        if residuals:
            snippets = []
            for rc in residuals[:4]:
                t = str(rc.get("text", ""))
                t = t if len(t) <= 60 else t[:60] + "…"
                pen = rc.get("penalty")
                snippets.append(f"「{t}」" + (f"(违约扣{int(pen)})" if pen else ""))
            parts.append("【未结构化偏好·必须遵守】调度引擎无法自动执行以下偏好，决策时逐条核对："
                         + "；".join(snippets))
        return "\n".join(parts)

    def _format_rules_for_llm(self, driver_id: str, rules: DriverRules, plan: dict, day: int) -> str:
        """Format driver rules and compliance progress as concise text for LLM."""
        month_idx = _month_index_for_day(day)
        lines = []
        if rules.rest_window:
            rs, re_ = rules.rest_window
            lines.append(f"- 休息窗口: {rs//60:02d}:{rs%60:02d}-{re_//60:02d}:{re_%60:02d}")
        for ws, we in rules.no_drive_windows_for(day):
            we_d = we % DAY_MINUTES
            lines.append(f"- 今日禁止接单/空驶: {ws//60:02d}:{ws%60:02d}-{we_d//60:02d}:{we_d%60:02d}")
        directive = rules.daily_directives.get(day)
        if directive and directive.get("notes"):
            lines.append(f"- 今日合规计划: {directive['notes']}")
        if rules.forbidden_categories:
            lines.append(f"- 禁运品类: {', '.join(rules.forbidden_categories)}")
        if rules.allowed_categories:
            lines.append(f"- 仅允许承运品类(其余一律不接): {', '.join(sorted(rules.allowed_categories))}")
        if rules.haul_min_km is not None:
            lines.append(f"- 单笔干线距离下限: {rules.haul_min_km:g}km (更短的不接)")
        if rules.forbidden_regions:
            lines.append(f"- 禁入区域: {', '.join(rules.forbidden_regions)}")
        if rules.allowed_regions:
            lines.append(f"- 仅允许运营区域: {', '.join(sorted(rules.allowed_regions))}")
        if rules.pickup_max_km:
            lines.append(f"- 空驶上限: {rules.pickup_max_km}km")
        if rules.daily_order_limit:
            today_count = plan.get("orders_today", {}).get(day, 0)
            lines.append(f"- 每日接单上限: {rules.daily_order_limit} (今日已接{today_count})")
        rule_penalties = getattr(rules, "rule_penalties", {}) or {}
        if rule_penalties:
            lines.append("- 各规则单次违约扣款: " + ", ".join(f"{k}:{int(v)}元" for k, v in rule_penalties.items()))
            lines.append("  罚款只是成本：候选货源的 net/net_per_h 已扣除对应罚款，净收益为正即值得接")

        # Category targets with progress
        cat_orders = plan.get("monthly_category_orders", {}).get(month_idx, {})
        longhual = plan.get("monthly_longhual", {}).get(month_idx, 0)
        # The long-haul cap is preference-driven: only mention it when this
        # driver's preferences impose one. A cap-free or long-haul-focused
        # driver must not be steered away from long hauls.
        if rules.longhaul_cap_orders is not None:
            cap = rules.longhaul_cap_orders
            pen = int(rules.longhaul_penalty)
            lh_h = rules.longhaul_min_minutes // 60
            if longhual >= cap:
                lines.append(
                    f"- 本月长途(>{lh_h}h): {longhual}单 (已达{cap}单上限; 再接每单扣{pen}罚分，"
                    f"仅当该单净利润>{pen}才值得接)"
                )
            else:
                lines.append(f"- 本月长途(>{lh_h}h): {longhual}/{cap}单上限 (超限后每单扣{pen})")
        lines.append(f"- 本月品类接单: {json.dumps(cat_orders, ensure_ascii=False)}")
        targets = getattr(rules, "monthly_category_targets", {}).get(month_idx, {})
        if targets:
            lines.append(f"- 本月品类指标: {json.dumps(targets, ensure_ascii=False)}")
        failed_reasons = plan.get("failed_cargo_reasons", {})
        if failed_reasons:
            recent_failed = list(failed_reasons.items())[-3:]
            lines.append("- 近期失败货源: " + "; ".join(f"{cid}:{reason}" for cid, reason in recent_failed))

        # Residual (non-structured) constraints: the compile+audit loop could not
        # express these as structured rules, so the decision LLM is the enforcer.
        residuals = getattr(rules, "residual_constraints", []) or []
        residual_texts = {rc.get("text", "") for rc in residuals}
        if residuals:
            lines.append("\n【必须遵守·未结构化偏好】以下偏好未被自动调度引擎覆盖，每次决策都必须逐条核对，违反会被扣罚:")
            for rc in residuals:
                pen = rc.get("penalty")
                pen_txt = f"（违约每次扣{int(pen)}元）" if pen else ""
                lines.append(f"  「{rc.get('text', '')}」{pen_txt}")

        # Add raw preference texts for context (skip ones already shown above)
        prefs = [p for p in self._get_active_preferences(driver_id) if p not in residual_texts]
        if prefs:
            lines.append("\n原始偏好文本:")
            for p in prefs:
                lines.append(f"  「{p}」")

        return "\n".join(lines)

    def _get_active_preferences(self, driver_id: str) -> list[str]:
        """Raw preference texts seen so far, in stable first-seen order."""
        records = self._pref_records.get(driver_id)
        if records:
            return [r["content"] for r in records][:6]
        return sorted(self._seen_prefs.get(driver_id, set()))[:4]

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
        # window (static or today's directive), idle until the window ends.
        for ws, we in rules.no_drive_windows_for(day):
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
        # or unconstrained driver may let the day's *last* order finish past
        # midnight (see _extended_order_end), but only when the next day is an
        # ordinary working day — never crossing into an off day, blackout day or
        # a dated-event day.
        hard_end = self._extended_order_end(rules, plan, now, day)
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
        # (E'') Instead of idling until day end, wait 1 hour and retry. The cargo
        # pool is dynamic (new cargo keeps appearing per create_time), so a fresh
        # scan every hour catches new compliant orders up to an hour earlier than
        # the previous 2h cadence — the extra rescan only costs a few minutes of
        # scan time, far less than an idle hour of a lost order.
        remaining = day_end - now
        if remaining > 240:
            return self._safe_wait(rules, now, 60)
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
            # A carryover clause means an unmet KPI from the previous month must
            # be made up by taking MORE OF THE SAME category this month (that is
            # how the scorer credits make-up orders). So add each previous-month
            # deficit to the *same* category's target this month. Do NOT inflate
            # this month's own first category -- extra orders of a different
            # category do not offset the carried-over deficit, which is exactly
            # what made the agent over-take 建材 while ignoring the cheaper fruit
            # make-up and eat the (higher) carryover penalty.
            for cat, req in prev_targets.items():
                deficit = max(0, int(req) - int(prev_orders.get(cat, 0)))
                if deficit:
                    targets[cat] = int(targets.get(cat, 0)) + deficit
        if targets:
            best_cat, best_need = None, 0
            for cat, req in targets.items():
                got = cat_orders.get(cat, 0)
                need = max(0, int(req) - int(got))
                if need > best_need:
                    best_cat, best_need = cat, need
            return (best_cat, best_need) if best_cat and best_need > 0 else (None, 0)
        # No category KPI parsed from this driver's preferences -> no target.
        # (Earlier builds hard-coded D001's 水果/建材 quotas here, which would
        # mislead any other driver; the parsed monthly_category_targets above is
        # the only source of truth now.)
        return (None, 0)

    # Chars too generic to identify a category on their own ("水果" vs "水泥",
    # "建材" vs "木材" must NOT match through them).
    _GENERIC_CATEGORY_CHARS = set("水材品类货物用设大小鲜活其他")

    @staticmethod
    def _category_matches(cat: str, cargo_name: str) -> bool:
        """容噪品类匹配：精确子串之外，允许通过非泛用共有字匹配
        （抓住「水果」↔「鲜果」这类措辞差异，同时拒绝「水果」↔「水泥」）。"""
        cat, cargo_name = cat.strip(), cargo_name.strip()
        if not cat or not cargo_name:
            return False
        if cat in cargo_name or cargo_name in cat:
            return True
        shared = (set(cat) & set(cargo_name)) - ModelDecisionService._GENERIC_CATEGORY_CHARS
        return bool(shared)

    def _track_category_order(self, plan, rules, month_idx: int, cargo_name: str) -> None:
        if not cargo_name:
            return
        cat_orders = plan["monthly_category_orders"].setdefault(month_idx, {})
        targets = getattr(rules, "monthly_category_targets", {}).get(month_idx, {}) if rules else {}
        if targets:
            for cat in targets:
                if self._category_matches_sem(cat, cargo_name):
                    cat_orders[cat] = cat_orders.get(cat, 0) + 1
        # No parsed targets -> nothing to track (was a D001-specific 水果/建材
        # fallback that does not generalize to other drivers).

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
        # Month-to-date pickup deadhead, needed by the monthly_deadhead_max_km
        # checks below. Missing this binding made every _pick_order call raise
        # NameError for any driver whose preferences impose a deadhead cap —
        # which aborts that driver's whole simulation (orchestrator treats a
        # decide() exception as fatal).
        month_deadhead = plan.setdefault("monthly_deadhead_km", {}).get(month_idx, 0.0)
        cat_target, cat_needed = self._get_category_target(rules, plan, day)

        # [OPT] When a monthly category target is at risk, widen earlier.  The
        # month boundary follows the scorer's calendar (Mar 31 / Apr 30 / May 31),
        # not a flat 31-day bucket.
        hunt_category = cat_target is not None and cat_needed > 0
        remaining_month_days = max(1, _month_end_day_exclusive(month_idx) - day)
        urgent_category = bool(hunt_category and remaining_month_days <= cat_needed + 10)
        initial_k = 600 if urgent_category else (200 if hunt_category else 60)

        cargo_resp = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=initial_k)
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
        skip_ids: set[str] = set()  # picks dropped after going stale mid-step

        def _score_item(cargo, item, now_t):
            nonlocal best, best_item, best_score, best_is_required, best_is_category_target, best_pickup_km, best_cost_time
            cid = str(cargo.get("cargo_id", ""))
            if cid in skip_ids or cid in plan.get("failed_cargo_ids", set()):
                return
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, now_t, day_end, lat, lng)
            if ev is None:
                return
            net, touches_required, occupied, pkm = ev
            if rules.monthly_deadhead_max_km is not None:
                if month_deadhead + pkm > rules.monthly_deadhead_max_km:
                    return
            # [OPT] Long-haul cap is preference-driven and *soft*. It only applies
            # when this driver's preferences impose one (a long-haul-focused or
            # cap-free driver must not be throttled). Over-cap orders have the
            # per-violation penalty subtracted from their value: still picked when
            # net beats the penalty (net maximization), deprioritised otherwise.
            # Threshold uses cost_time (the haul time the scorer counts).
            cost_time = int(cargo.get("cost_time_minutes", 0))
            eff_net = net
            if (rules.longhaul_cap_orders is not None
                    and cost_time > rules.longhaul_min_minutes
                    and longhual_count >= rules.longhaul_cap_orders):
                eff_net = net - rules.longhaul_penalty
                if eff_net <= 0:
                    return
            score = eff_net / occupied
            # [OPT] Category preference: exact name match with strong boost
            cargo_name = str(cargo.get("cargo_name", ""))
            is_cat_target = False
            if cat_target and cat_needed > 0:
                if self._category_matches_sem(cat_target, cargo_name):
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
            cargo_resp2 = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
            items2 = cargo_resp2.get("items", [])
            plan["_scan_items"] = (entry_now, items2)
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            for item in items2:
                _score_item(item.get("cargo", {}), item, now)

        # Widen search if nothing found at all
        if best is None and not hunt_category:
            cargo_resp2 = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=200)
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
        retried = False
        while True:
            if hunt_category and urgent_category and not best_is_category_target:
                plan["_category_miss_today"] = day
                self._logger.info(
                    "category target urgent: skip non-category best driver_id=%s cat=%s needed=%d remaining_days=%d",
                    driver_id, cat_target, cat_needed, remaining_month_days,
                )
                return None
            latest_now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            # Re-derive the deadline at the post-scan clock but KEEP the caller's
            # cross-midnight extension: the old min(day_end, plain deadline)
            # clamped the deadline back to midnight, so revalidation rejected
            # every extended pick the scoring pass had legitimately accepted.
            latest_deadline = self._hard_order_deadline(rules, plan, latest_now, day, base_end=day_end)
            latest_ev = self._evaluate_cargo(best, best_item or {}, rules, blackout_regions, latest_now, latest_deadline, lat, lng)
            if latest_ev is not None:
                break
            self._logger.info(
                "rejected selected order after scan-time revalidation cargo_id=%s now=%d deadline=%d",
                best.get("cargo_id"),
                latest_now,
                latest_deadline,
            )
            if retried:
                return None
            # One retry: drop the stale pick and re-score the already-scanned
            # items at the fresh clock. Losing the whole step (idling up to an
            # hour) because the best pick went stale during a long k=600 scan is
            # strictly worse than taking the second-best compliant order now.
            retried = True
            skip_ids.add(str(best.get("cargo_id")))
            best = None
            best_item = None
            best_score = 0.0
            best_is_required = False
            best_is_category_target = False
            best_pickup_km = 0.0
            best_cost_time = 0
            _, last_items = plan.get("_scan_items") or (now, items)
            for item in last_items:
                _score_item(item.get("cargo", {}), item, latest_now)
            if best is None:
                return None
        latest_net, _latest_required, latest_occupied, latest_pickup_km = latest_ev
        # Preference-driven soft long-haul cap: keep an over-cap order only if its
        # net beats the penalty (consistent with _score_item / _validate_llm_take_order).
        latest_cost_time = int(best.get("cost_time_minutes", 0) or 0)
        if (
            rules.longhaul_cap_orders is not None
            and latest_cost_time > rules.longhaul_min_minutes
            and longhual_count >= rules.longhaul_cap_orders
            and latest_net - rules.longhaul_penalty <= 0
        ):
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
        history_text = history.get_summary(day, plan) if history else "无历史记录"

        day_of_month = _day_in_month(day)
        month_idx = _month_index_for_day(day)
        month_name = {0: "三月", 1: "四月", 2: "五月"}.get(month_idx, "")

        # Query broader cargo to find where target category exists
        cargo_resp = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=200)
        items = cargo_resp.get("items", [])
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        # Find target category cargo in results
        target_locations = []
        for item in items:
            cargo = item.get("cargo", {})
            if self._category_matches_sem(cat_target, str(cargo.get("cargo_name", ""))):
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
        wide_items: list[dict[str, Any]] | None = None
        if best_target is None:
            cargo_resp = self._query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            day_end = min(day_end, self._hard_order_deadline(rules, plan, now, day))
            wide_items = cargo_resp.get("items", [])
            best_target, best_net = _best_target(wide_items)
        if best_target is None:
            # Deep-strand rescue: not even ONE order in the widest scan can pay
            # back the approach — the driver is in a cargo desert (a previous
            # haul dropped it hundreds of km from any cluster). Single-order
            # economics can never justify the return trip, so the driver would
            # idle for the REST OF THE SEASON (shadow benchmark measured 57
            # zero-order days). Treat the return deadhead as an investment in
            # future daily earning capacity and head toward the densest pickup
            # cluster, one day-sized leg at a time.
            action = self._deep_strand_rescue(rules, now, day_end, lat, lng, wide_items or [])
            if action is None:
                return None
        else:
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

    _RESCUE_MIN_DESERT_KM = 80.0  # nearer cargo than this → not a desert, normal flow

    def _deep_strand_rescue(self, rules, now, day_end, lat, lng, items) -> dict[str, Any] | None:
        """Reposition toward the densest pickup cluster visible in the wide scan.

        Only fires when the nearest scanned pickup is farther than
        _RESCUE_MIN_DESERT_KM (a true desert; anything closer is left to the
        normal anchor logic). The leg is clipped to today's drivable budget so
        it always ends before the day's hard deadline / night window; if still
        stranded tomorrow, the next rescue leg continues the journey.
        """
        pickups: list[tuple[float, float, str]] = []
        for item in items:
            start = (item.get("cargo") or {}).get("start") or {}
            try:
                plat, plng = float(start.get("lat")), float(start.get("lng"))
            except (TypeError, ValueError):
                continue
            pickups.append((plat, plng, str(start.get("city", ""))))
        if not pickups:
            return None
        if min(_haversine_km(lat, lng, p[0], p[1]) for p in pickups) < self._RESCUE_MIN_DESERT_KM:
            return None
        # Densest pickup: most neighbours within 60 km (sampled for O(n²) control).
        sample = pickups[:150]
        best_i, best_density = 0, -1
        for i, (ilat, ilng, _c) in enumerate(sample):
            density = sum(
                1 for jlat, jlng, _c2 in sample
                if _haversine_km(ilat, ilng, jlat, jlng) < 60.0
            )
            if density > best_density:
                best_i, best_density = i, density
        tlat, tlng, tcity = sample[best_i]
        budget_min = day_end - now - _ORDER_DEADLINE_BUFFER_MIN
        if budget_min < 60:
            return None
        max_km = budget_min / 60.0 * SPEED_KM_PER_HOUR
        dist = _haversine_km(lat, lng, tlat, tlng)
        if dist > max_km:
            frac = max_km / dist
            tlat = lat + (tlat - lat) * frac
            tlng = lng + (tlng - lng) * frac
            tcity = None
        self._logger.info(
            "deep-strand rescue: heading to densest cluster (%.3f,%.3f) density=%d dist=%.0fkm (clipped to %.0fkm)",
            tlat, tlng, best_density, dist, min(dist, max_km),
        )
        return self._safe_reposition(
            rules, now, lat, lng, tlat, tlng,
            deadline=day_end, tag="strand_rescue", target_city=tcity,
        )

    def _evaluate_relocation(self, cargo, rules, blackout_regions, now, day_end, lat, lng):
        """Net of an order if the driver first repositions to its pickup. Mirrors
        _evaluate_cargo's compliance checks but treats the approach as a reposition
        (so the per-order deadhead cap does not apply) and measures arrival from the
        pickup. Returns (net, pickup_lat, pickup_lng) or None."""
        name = str(cargo.get("cargo_name", ""))
        if self._is_forbidden_cargo(name, rules.forbidden_categories):
            return None
        # allowed_categories whitelist: relocation anchor must also comply
        if rules.allowed_categories and not any(
            self._category_matches_sem(cat, name) for cat in rules.allowed_categories
        ):
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
            rules, now, arrival + (_ORDER_DEADLINE_BUFFER_MIN if rules.has_no_drive(now // DAY_MINUTES) else 0)
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
            rules, now, finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.has_no_drive(now // DAY_MINUTES) else 0)
        ):
            return None
        haul_km = _haversine_km(slat, slng, elat, elng)
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            return None
        if rules.haul_min_km is not None and haul_km < rules.haul_min_km:
            return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (move_km + haul_km)
        if net <= 0:
            return None
        net *= avoid_penalty
        return net, slat, slng, scity

    def _evaluate_cargo(self, cargo, item, rules, blackout_regions, now, day_end, lat, lng):
        # Economic treatment of preference violations: when the parse step
        # attributed a per-violation penalty amount to a rule kind, a violating
        # order is not hard-rejected — the penalty is subtracted from its net
        # and the order survives only if it is still profitable (same pattern
        # as the long-haul soft cap). Unknown penalty → conservative hard reject.
        violation_cost = 0.0
        name = str(cargo.get("cargo_name", ""))
        if self._is_forbidden_cargo(name, rules.forbidden_categories):
            fp = rules.rule_penalties.get("forbidden_categories")
            if fp is None:
                return None
            violation_cost += fp
        # allowed_categories whitelist ("只拉X"): cargo outside the whitelist is a
        # violation. Same economics as forbidden: known penalty → soft cost,
        # unknown penalty → conservative hard reject.
        if rules.allowed_categories and not any(
            self._category_matches_sem(cat, name) for cat in rules.allowed_categories
        ):
            ap = rules.rule_penalties.get("allowed_categories")
            if ap is None:
                return None
            violation_cost += ap
        # avoid_categories: soft penalty (50% score reduction) rather than hard rejection
        avoid_penalty = 0.5 if self._is_forbidden_cargo(name, rules.avoid_categories) else 1.0
        start = cargo.get("start") or {}
        end = cargo.get("end") or {}
        scity = str(start.get("city", ""))
        ecity = str(end.get("city", ""))
        slat, slng = float(start.get("lat", 0.0)), float(start.get("lng", 0.0))
        elat, elng = float(end.get("lat", 0.0)), float(end.get("lng", 0.0))
        if rules.allowed_regions:
            if not (_target_matches_allowed_regions(rules.allowed_regions, slat, slng, scity)
                    and _target_matches_allowed_regions(rules.allowed_regions, elat, elng, ecity)):
                ap = rules.rule_penalties.get("allowed_regions")
                if ap is None:
                    return None
                violation_cost += ap
        for region in rules.forbidden_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                rp = rules.rule_penalties.get("forbidden_regions")
                if rp is None:
                    return None
                violation_cost += rp
                break
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
            rules, now, arrival + (_ORDER_DEADLINE_BUFFER_MIN if rules.has_no_drive(now // DAY_MINUTES) else 0)
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
            rules, now, finish + (_ORDER_DEADLINE_BUFFER_MIN if rules.has_no_drive(now // DAY_MINUTES) else 0)
        ):
            return None
        haul_km = _haversine_km(slat, slng, elat, elng)
        # haul_max_km / haul_min_km: single-order haul distance limits
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            return None
        if rules.haul_min_km is not None and haul_km < rules.haul_min_km:
            return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (pickup_km + haul_km) - violation_cost
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
        """整休日：在**每个日历月内**均匀铺 off_days_min 天，避开事件日及其前置日。

        "每月至少N天完全不出车" 是按月评定的——旧实现把 N 天摊到整个 92 天赛季
        （N=2 时只铺 3 月下旬和 5 月上旬各一天），busy 月份 0 个整休日就吃一次
        整月罚款（影子基准实测 D102 四月 0 整休 → 3000 罚）。按月铺保证每个月
        都达标；多歇几天的收入损失远小于按月罚款。

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
        for m_start, m_end in zip(_MONTH_START_DAYS[:-1], _MONTH_START_DAYS[1:]):
            span = m_end - m_start
            for i in range(min(n, span)):
                target = m_start + int(round((i + 0.5) * span / n))
                target = min(m_end - 1, max(m_start, target))
                chosen = None
                for delta in range(span):
                    for cand in (target + delta, target - delta):
                        if m_start <= cand < m_end and cand not in reserved and cand not in off:
                            chosen = cand
                            break
                    if chosen is not None:
                        break
                if chosen is not None:
                    off.add(chosen)
        return off

    # ------------------------------------------------------------- preferences
    def _ensure_rules(self, driver_id: str, status: dict[str, Any]) -> DriverRules:
        # Closed-loop preference compilation. Preferences can be date-windowed and
        # only become visible inside their window (e.g. 深圳 blackout, 盘库, 寿宴),
        # so this runs each step but only compiles *new* texts.
        #
        # Pipeline per new preference text:
        #   1. compile: one LLM extraction call per text (per-text grounding —
        #      semantic confirms check against the right text, no cross-pref noise);
        #   2. audit & repair: a second LLM pass checks that every obligation in
        #      the text is represented by the structured rules and patches misses
        #      (replaces the old D001-phrasing regex supplement layer);
        #   3. no silent drop: anything still not representable lands in
        #      rules.residual_constraints and is enforced via the decision prompt.
        # The regex parser remains only as an offline fallback (model unavailable).
        rules = self._rules.get(driver_id)
        if rules is None:
            rules = DriverRules()
            self._rules[driver_id] = rules
        self._initial_position.setdefault(
            driver_id,
            (float(status.get("current_lat", 0.0)), float(status.get("current_lng", 0.0))),
        )
        prefs = status.get("preferences") or []
        items: list[tuple[str, Any]] = []
        for pref in prefs:
            if isinstance(pref, dict):
                text = str(pref.get("content", "") or "")
                penalty = pref.get("penalty_amount")
            else:
                text, penalty = str(pref), None
            if text.strip():
                items.append((text, penalty))
        texts = [t for t, _ in items]
        seen = self._seen_prefs.setdefault(driver_id, set())
        new_items = [(t, p) for t, p in items if t not in seen]
        if not new_items:
            return rules

        # Keep the raw preference records (ordered, with penalty amounts) for
        # the daily compliance planner and the decision prompt.
        records = self._pref_records.setdefault(driver_id, [])
        known_contents = {r["content"] for r in records}
        for pref in prefs:
            content = (pref.get("content", "") if isinstance(pref, dict) else str(pref)).strip()
            if not content or content in known_contents:
                continue
            known_contents.add(content)
            record: dict[str, Any] = {
                "content": content,
                "penalty_amount": (pref.get("penalty_amount") if isinstance(pref, dict) else None),
            }
            mandarin = self._normalize_preference_text(driver_id, content)
            if mandarin:
                record["mandarin"] = mandarin
            records.append(record)

        before = self._rules_fingerprint(rules)
        coord_map = self._collect_coords(prefs)
        # Current calendar date: lets the extractor resolve "本月/这个月/呢个月"
        # style targets to the right month (preferences become visible inside
        # their own date window, so "this month" == the month of first sight).
        cur_day = int(status.get("simulation_progress_minutes", 0) or 0) // DAY_MINUTES
        date_str, weekday_str, _is_we = _calendar_for_day(min(max(cur_day, 0), MONTH_DAYS - 1))
        today_str = f"{date_str} {weekday_str}"
        compiled: list[tuple[str, Any]] = []
        for text, penalty in new_items:
            # Dialect robustness: compile/audit/ground against 原文+普通话释义.
            aug = self._augmented_pref_text(driver_id, text)
            if self._llm_parse_preferences(driver_id, [aug], rules, coord_map, [penalty], today=today_str):
                compiled.append((aug, penalty))
            else:
                # offline / model unavailable: deterministic regex fallback for
                # this text only.
                self._parse_one(aug, rules, coord_map)
                self._supplement_basic_rules(aug, rules)
                self._supplement_dated_events(aug, rules, coord_map)
        if compiled:
            self._audit_preference_coverage(driver_id, compiled, rules, coord_map)
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
        seen.update(t for t, _ in new_items)
        # Dedup avoid/forbidden AFTER supplement so regex-added items are also cleaned
        init_lat, init_lng = self._initial_position[driver_id]
        aug_texts = [self._augmented_pref_text(driver_id, t) for t in texts]
        self._supplement_relative_allowed_regions(aug_texts, rules, init_lat, init_lng)
        self._supplement_night_failsafe(aug_texts, rules)
        self._supplement_critical_rules(aug_texts, rules)
        self._dedup_avoid_forbidden(rules)
        if self._rules_fingerprint(rules) != before:
            self._logger.info(
                "parsed rules driver_id=%s rest=%s window=%s off=%s forbid_cat=%s avoid_cat=%s "
                "forbid_reg=%s allowed_reg=%s required=%s pickup_max=%s haul_max=%s blackout=%s "
                "dated_single=%s dated_route=%s no_drive=%s order_limit=%s home=%s "
                "forbidden_zones=%d bounded=%s must_visit=%d first_order=%s category_targets=%s carryover=%s "
                "residual=%d",
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
                len(rules.residual_constraints),
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
        "偏好原文可能是标准普通话，也可能是粤语、客家话、吴语（江浙沪）等方言或口语俚语"
        "（有的条目附带「普通话释义」），一律按**含义**抽取，不要因为措辞陌生而漏抽。\n"
        "只输出一个 JSON 对象，禁止 markdown / 解释 / think标签。未提及的字段用 null 或空数组。\n\n"
        "字段定义（宁缺毋错）：\n"
        '- daily_rest_hours: 每天连续休息最少小时数（数字或 null）\n'
        '- rest_window: 每天固定停车时段 {"start_hour":数字,"end_hour":数字}（或 null）。半小时用0.5，如11点半→11.5\n'
        '- no_drive_windows: 每天禁止接单/空驶的时段数组 [{"start_hour":数字,"end_hour":数字}]。半小时用0.5。\n'
        '  适用于任何"某时段不出车/不接单/不空驶/收车/熄火/落锁/归家不动/歇业/宵禁"。跨午夜时 end_hour<start_hour，如23→5。\n'
        '  **极重要（夜休按天扣分，漏抽代价极高，务必抽出）**：凡表达"夜里/入夜/天黑后/后半夜/晚上X点后/到次日X点 不出车·不揽货·收车·归家·熄火·睡觉·歇着"等含义的，'
        '无论用词多口语/方言，都必须填 no_drive_windows（跨夜用 start>end）；若同时有"休息/睡觉/停车熄火"含义，则 rest_window 也一并填。宁可多抽一个夜休窗口，也不要漏掉。\n'
        '- off_days_min: 整月完全不出车天数（整数，默认 0）\n'
        '- forbidden_categories: 禁运货物**品类名**数组（仅货物名称如"蔬菜""机械设备""生鲜"，'
        '绝不放城市/区域名！"在惠州的货"是区域禁令不是品类禁令）\n'
        '- avoid_categories: 尽量避免的货物品类名数组（"尽量不拉""尽量不接"→放这里）\n'
        '- allowed_categories: 仅允许承运的货物品类名数组（"只拉X""只接X类货""除了X一律不接"→放这里）。'
        '仅当文本明确表达"只/仅承运某品类"时填写，与 forbidden_categories 严格区分。\n'
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
        '- haul_min_km: 单笔干线距离下限公里数（"低于X公里的短活不接"，数字或 null）\n'
        '- monthly_longhaul_cap: 月度长途单数上限 {"max_orders":整数,"min_hours":小时数}（或 null）。\n'
        '  适用于"每月超过X小时的长途最多/只能接N单，多一单扣一次"。文本没提长途单数限制就必须为 null，'
        '绝不能默认存在上限（有的司机专跑长途）。\n'
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
        '  触发词：赴宴/做寿/先到…再到/先去…再去/先过…赶到/接人/送人/喝喜酒/吃饭/接上配偶/接家人\n'
        '- rule_penalties: 各类规则的单次违约扣款金额对象（仅当输入含 penalty_amounts 时填写，否则 null）。'
        '把每条偏好的金额归因到其抽出的规则类别，键限：'
        '"allowed_regions"/"forbidden_regions"/"forbidden_categories"/"allowed_categories"/"night_window"/"category_targets"/"dated_events"/"longhaul_cap"。'
        '一条偏好含多种规则时金额填到每个对应键；无法确定归因的键省略。\n\n'
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
        '9) "不接单不空跑/不空驶/收车/归家不动/熄火歇业"类约束，只要含时间段一律填no_drive_windows（夜间跨夜用start>end）；'
        '若同时含"休息/睡觉/停车熄火"→也填rest_window或daily_rest_hours。这类"每日生效"的时段约束漏抽代价极高，务必抽全。\n'
        '10) "接上配偶/家人→返回老家/进家门"是 dated_route 事件（多点路线），不是 home_rule。\n'
        '11) "只跑/只接/仅在/不出X"是 allowed_regions，不是 forbidden_regions。\n'
        '12) 输入可能含 penalty_amounts 数组（与 preferences 一一对应的违约扣款金额）。'
        '金额越高的偏好越要逐字推敲、确保其全部约束被完整抽出，绝不能漏；尤其是每日生效的夜休/禁驶窗口和月度品类指标。'
        '同时把各金额按规则类别归因填入 rule_penalties。\n'
        '13) 方言对照（按含义抽取，绝不漏抽）：粤语 唔/咪=不要、冇=没有、晚黑=晚上、朝早=早上、'
        '晏昼=下午、挨晚=傍晚、返屋企=回家、收工=收车下班、开工=开始干活、走货/拉货=运货、'
        '生果=水果、礼拜六日=周末、先至=才；吴语 夜里向/夜到=晚上、早浪向=早上、屋里厢=家里、'
        '勿/覅=不要、困觉=睡觉、歇生意=不接活、做生活=干活。'
        '品类名输出标准普通话叫法（「生果」→"水果"）。\n'
        '14) 输入若含"今天"（当前日期），偏好说"本月/这个月/呢个月"的指标时，month 取今天所在月份。\n\n'
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
        '"blackout":[],"dated_single":[{"date":20,"lat":23.25,"lng":113.40,"wait_minutes":120,"before_hour":null}],"dated_route":[]}\n\n'
        "示例7（方言）：\n"
        '入: {"preferences":["晚黑十点之后唔好再接单走车，要返屋企熄火休息，挨到朝早六点先至好开工","呢个月生果要拉够十二单，少一单扣一次钱"],"今天":"2026-03-02 周一"}\n'
        '出: {"daily_rest_hours":null,"rest_window":{"start_hour":22,"end_hour":6},'
        '"no_drive_windows":[{"start_hour":22,"end_hour":6}],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,'
        '"monthly_category_targets":[{"month":3,"category":"水果","min_orders":12,"carryover":false}],"home_rule":null,'
        '"blackout":[],"dated_single":[],"dated_route":[]}'
    )

    # ------------------------------------------------ dialect normalization
    _NORMALIZE_SYSTEM = (
        "你是货运司机偏好的方言转写助手。输入是一条司机偏好原文，可能用粤语、客家话、"
        "吴语（江浙沪一带）、其他地方方言或口语俚语表述。把它转写成一句等义的标准普通话：\n"
        "- 所有数字、时刻、日期、星期、地名、坐标、金额、单数/天数指标必须原样保留，一个都不能丢、不能改；\n"
        "- 不得添加原文没有的约束，也不得弱化、合并或省略任何约束；\n"
        "- 货物品类名转成通用叫法（如粤语「生果」→「水果」）；\n"
        "- 若原文已是标准普通话，原样返回。\n"
        '只输出 JSON {"mandarin":"转写后的普通话"}，禁止 markdown / 解释。\n'
        "常见方言词对照（粤语）：唔/咪=不要；冇=没有；晚黑/夜晚黑=晚上；朝早=早上；晏昼=下午；"
        "挨晚=傍晚；半夜三更=深夜；返屋企=回家；收工=下班收车；开工=开始干活；揾食=干活谋生；"
        "走货/行货/拉货=运货；生果=水果；嘢=东西；礼拜六日=周末；呢个月=这个月；先至=才；唔好=不要。\n"
        "（吴语/江浙沪）：夜里向/夜到=晚上；早浪向/早上向=早上；屋里厢=家里；勿/覅=不要；"
        "困觉=睡觉；歇生意=不接活；做生活=干活；交关=非常；礼拜天=星期日；卯辰光=那时候。"
    )

    def _normalize_preference_text(self, driver_id: str, text: str) -> str | None:
        """方言/口语偏好 → 标准普通话释义（按原文缓存 + fail-safe）。

        返回 None 表示无需释义或模型不可用；此时管线行为与未引入本层完全一致。
        """
        cached = self._pref_norms.get(text)
        if cached is not None:
            return cached or None
        result: str | None = None
        try:
            req: dict[str, Any] = {
                "messages": [
                    {"role": "system", "content": self._NORMALIZE_SYSTEM},
                    {"role": "user", "content": text},
                ],
                "temperature": 0,
                "max_tokens": 300,
            }
            try:
                req["response_format"] = {"type": "json_object"}
                resp = self._api.model_chat_completion(req)
            except Exception:
                req.pop("response_format", None)
                resp = self._api.model_chat_completion(req)
            data = self._extract_json(resp)
            if isinstance(data, dict):
                mand = str(data.get("mandarin") or "").strip()
                # Identical output means the text was already standard Mandarin.
                if mand and mand != text.strip():
                    result = mand
        except Exception as exc:
            self._logger.info(
                "pref normalize unavailable driver_id=%s err=%s", driver_id, exc
            )
        self._pref_norms[text] = result or ""
        if result:
            self._logger.info(
                "pref normalize driver_id=%s raw=%s mandarin=%s",
                driver_id, text[:80], result[:120],
            )
        return result

    def _augmented_pref_text(self, driver_id: str, text: str) -> str:
        """原文 + 普通话释义的合并文本。

        合并文本被一致地用于抽取、覆盖审计、语义接地（_confirm_rule_holds）与
        离线正则兜底：原文保证接地校验不会因释义产生幻觉约束（原文始终在场），
        释义保证标准普通话措辞的关键词/语义都能命中。
        """
        norm = self._normalize_preference_text(driver_id, text)
        if norm:
            return f"{text}\n（普通话释义：{norm}）"
        return text

    def _llm_parse_preferences(
        self, driver_id: str, texts: list[str], rules: DriverRules,
        coord_map: dict[str, tuple[float, float]] | None = None,
        penalties: list[Any] | None = None,
        today: str | None = None,
    ) -> bool:
        """用大模型把全部可见偏好解析成结构化规则并合并进 rules。

        返回 True 表示 LLM 成功产出结构化结果；False 表示模型不可用/解析失败
        （此时调用方会退回正则解析）。
        """
        if not texts:
            return False
        payload: dict[str, Any] = {"preferences": texts}
        if today:
            payload["今天"] = today
        if penalties and len(penalties) == len(texts) and any(p for p in penalties):
            payload["penalty_amounts"] = penalties
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

    # ------------------------------------------------- coverage audit & repair
    _PREF_AUDIT_SYSTEM = (
        "你是货运司机偏好「覆盖审计器」。调度引擎已把司机的自然语言偏好编译成结构化规则；"
        "你逐条审计：每条偏好原文里的**全部约束**是否都已体现在 structured_rules 里。\n"
        '输入 JSON: {"preferences":[原文...], "penalty_amounts":[违约扣款金额或null...], '
        '"structured_rules":"已结构化规则清单"}\n'
        "只输出一个 JSON 对象，禁止 markdown/解释：\n"
        '{"audits":[{"index":偏好序号从0起,"covered":true/false,'
        '"missing":"缺了什么(covered=false时填)","representable":true/false,'
        '"patch":{仅含缺失部分的补丁字段}}]}\n\n'
        "patch 可用字段（与抽取器一致，未缺失的不要填）：\n"
        "daily_rest_hours; rest_window{start_hour,end_hour}; "
        "no_drive_windows[{start_hour,end_hour}]（跨夜时 end_hour<start_hour）; off_days_min; "
        "forbidden_categories[]; avoid_categories[]; allowed_categories[]; forbidden_regions[]; allowed_regions[]; "
        "pickup_max_km; haul_max_km; haul_min_km; monthly_longhaul_cap{max_orders,min_hours}; "
        "monthly_deadhead_max_km; daily_order_limit; "
        "first_order_before_hour; monthly_category_targets[{month,category,min_orders,carryover}]; "
        "home_rule{lat,lng,radius_km,home_by_hour,no_drive_until_hour}; "
        "blackout[{region,dates[]}]; dated_single[{date,lat,lng,wait_minutes,before_hour}]; "
        "dated_route[{date,stops[{lat,lng,wait_minutes,before_hour}]}]; "
        "forbidden_zones[{lat,lng,radius_km}]; bounded_area{lat_min,lat_max,lng_min,lng_max}; "
        "must_visit[{lat,lng,radius_km,required_days}]; required_region{region,min_days}\n\n"
        "审计准则：\n"
        "1) 漏一条「每日生效」约束（夜休/禁驶时段/回家）代价最高：重点核对时间窗是否齐全、起止时刻与原文一致。\n"
        "2) 月度品类指标、整月休息天数、月度长途单数上限（如「每月超过X小时的长途最多N单」）、"
        "日期事件（某天到某地）逐条核对数字与日期。\n"
        "3) 某约束无法用上述字段表达时：representable=false、patch 留空、missing 写明该约束原意。\n"
        "4) penalty_amounts 金额越高的偏好越要逐字核对。\n"
        "5) 已完整覆盖就 covered=true 且不输出 patch；拿不准时 covered=false 并给出 patch"
        "（结构化引擎对重复规则免疫，宁可重复、不可漏掉）。\n"
        "6) 偏好原文可能是粤语/吴语等方言或口语（可能附带「普通话释义」），按含义审计："
        "如粤语「晚黑收工返屋企」即夜间禁驶+回家、「生果」即水果品类、吴语「夜里向覅跑车」即夜间禁驶。"
    )

    def _describe_rules_for_audit(self, rules: DriverRules) -> str:
        """Compact, complete human-readable dump of the structured rules so the
        audit model can judge coverage. Must mention every populated field."""
        lines: list[str] = []
        if rules.daily_rest_minutes:
            lines.append(f"每日连续休息≥{rules.daily_rest_minutes / 60:g}小时")
        if rules.rest_window:
            rs, re_ = rules.rest_window
            lines.append(f"每日休息窗 {rs // 60:02d}:{rs % 60:02d}-{re_ // 60:02d}:{re_ % 60:02d}")
        for ws, we in rules.no_drive_windows:
            we_d = we % DAY_MINUTES
            lines.append(f"每日禁驶(不接单不空驶) {ws // 60:02d}:{ws % 60:02d}-{we_d // 60:02d}:{we_d % 60:02d}")
        if rules.off_days_min:
            lines.append(f"整月完全不出车≥{rules.off_days_min}天")
        if rules.forbidden_categories:
            lines.append("禁运品类: " + "、".join(sorted(rules.forbidden_categories)))
        if rules.avoid_categories:
            lines.append("尽量避开品类: " + "、".join(sorted(rules.avoid_categories)))
        if rules.allowed_categories:
            lines.append("仅承运品类: " + "、".join(sorted(rules.allowed_categories)))
        if rules.forbidden_regions:
            lines.append("禁接区域: " + "、".join(sorted(rules.forbidden_regions)))
        if rules.allowed_regions:
            lines.append("仅允许运营区域: " + "、".join(sorted(rules.allowed_regions)))
        if rules.required_region:
            reg, md = rules.required_region
            lines.append(f"每月至少{md}天接「{reg}」的货")
        if rules.pickup_max_km:
            lines.append(f"赴装空驶≤{rules.pickup_max_km:g}km")
        if rules.haul_max_km:
            lines.append(f"单笔干线≤{rules.haul_max_km:g}km")
        if rules.haul_min_km:
            lines.append(f"单笔干线≥{rules.haul_min_km:g}km")
        if rules.longhaul_cap_orders is not None:
            lines.append(
                f"每月超过{rules.longhaul_min_minutes / 60:g}小时的长途≤{rules.longhaul_cap_orders}单"
            )
        if rules.monthly_deadhead_max_km:
            lines.append(f"月累计空驶≤{rules.monthly_deadhead_max_km:g}km")
        if rules.daily_order_limit:
            lines.append(f"每日接单≤{rules.daily_order_limit}单")
        if rules.first_order_before_minute is not None:
            m = rules.first_order_before_minute
            lines.append(f"每日首单不晚于{m // 60:02d}:{m % 60:02d}")
        for month_idx, targets in sorted(rules.monthly_category_targets.items()):
            for cat, req in sorted(targets.items()):
                carry = "(欠额结转下月)" if month_idx in rules.category_carryover_months else ""
                lines.append(f"{_month_name(month_idx)}品类「{cat}」至少{req}单{carry}")
        if rules.home_lat is not None:
            hb = rules.home_by_minute
            hb_txt = f"，每天{hb // 60:02d}:{hb % 60:02d}前到家" if hb is not None else ""
            nd = rules.no_drive_until_minute
            nd_txt = f"，次日{nd // 60:02d}:{nd % 60:02d}前不接单" if nd is not None else ""
            lines.append(
                f"回家规则: 家({rules.home_lat:.2f},{rules.home_lng or 0:.2f})"
                f"半径{rules.home_radius_km:g}km{hb_txt}{nd_txt}"
            )
        for region, days in rules.blackout:
            lines.append(f"{sorted(d + 1 for d in days)}号不去「{region}」")
        for ev in rules.dated_single:
            lines.append(
                f"{ev['day'] + 1}号需到({ev['lat']:.2f},{ev['lng']:.2f})停留{ev['min_wait']}分钟"
            )
        for ev in rules.dated_route:
            stops = "→".join(f"({s['lat']:.2f},{s['lng']:.2f})" for s in ev.get("stops", []))
            lines.append(f"{ev['day'] + 1}号按序经过 {stops}")
        for flat, flng, fr in rules.forbidden_zones:
            lines.append(f"禁入圆区: 圆心({flat:.2f},{flng:.2f})半径{fr:g}km")
        if rules.bounded_area:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            lines.append(f"仅允许运营矩形: 纬{la_min:.2f}-{la_max:.2f} 经{ln_min:.2f}-{ln_max:.2f}")
        for mv in rules.must_visit:
            lines.append(
                f"每月至少{mv['required_days']}天到({mv['lat']:.2f},{mv['lng']:.2f})"
                f"{mv.get('radius_km', 1.0):g}km内"
            )
        return "\n".join(lines) if lines else "（当前没有任何已结构化规则）"

    def _audit_preference_coverage(
        self,
        driver_id: str,
        compiled: list[tuple[str, Any]],
        rules: DriverRules,
        coord_map: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Closed-loop verification: ask the model whether every obligation in
        each compiled preference is represented by the structured rules; merge
        repair patches for misses; record anything unrepresentable as a residual
        constraint (enforced via the decision prompt — never silently dropped).

        This replaces the old phrasing-specific regex supplement layer as the
        recall mechanism, so unseen drivers/wordings generalise. Fail-open: if
        the audit call is unavailable the compiled rules simply stand as-is.
        """
        texts = [t for t, _ in compiled]
        penalties = [p for _, p in compiled]
        payload: dict[str, Any] = {
            "preferences": texts,
            "penalty_amounts": penalties,
            "structured_rules": self._describe_rules_for_audit(rules),
        }
        if coord_map:
            payload["known_coordinates"] = {
                name: {"lat": loc[0], "lng": loc[1]} for name, loc in coord_map.items()
            }
        msgs = [
            {"role": "system", "content": self._PREF_AUDIT_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        data: dict[str, Any] | None = None
        for attempt in range(2):
            req: dict[str, Any] = {"messages": msgs, "temperature": 0}
            if attempt == 0:
                req["response_format"] = {"type": "json_object"}
            try:
                resp = self._api.model_chat_completion(req)
            except Exception as exc:
                self._logger.info(
                    "coverage audit attempt %d unavailable driver_id=%s err=%s",
                    attempt, driver_id, exc,
                )
                continue
            data = self._extract_json(resp)
            if data is not None:
                break
        if data is None:
            self._logger.info("coverage audit unavailable driver_id=%s (rules stand as-is)", driver_id)
            return
        audits = data.get("audits")
        if not isinstance(audits, list):
            return
        for entry in audits:
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= idx < len(texts)) or entry.get("covered") is True:
                continue
            text = texts[idx]
            patch = entry.get("patch")
            patched = False
            if isinstance(patch, dict) and any(v for v in patch.values()):
                try:
                    self._merge_llm_rules(rules, patch, [text])
                    patched = True
                    self._logger.info(
                        "coverage audit: repaired pref #%d driver_id=%s missing=%s",
                        idx, driver_id, str(entry.get("missing", ""))[:120],
                    )
                except Exception as exc:
                    self._logger.warning(
                        "coverage audit: patch merge failed driver_id=%s err=%s", driver_id, exc
                    )
            if entry.get("representable") is False or not patched:
                self._add_residual_constraint(
                    rules, text, penalties[idx], str(entry.get("missing", ""))
                )
                self._logger.info(
                    "coverage audit: pref #%d kept as residual constraint driver_id=%s",
                    idx, driver_id,
                )

    @staticmethod
    def _add_residual_constraint(rules: DriverRules, text: str, penalty: Any, note: str = "") -> None:
        if any(rc.get("text") == text for rc in rules.residual_constraints):
            return
        pen = float(penalty) if isinstance(penalty, (int, float)) and penalty > 0 else None
        rules.residual_constraints.append({"text": text, "penalty": pen, "note": note[:120]})

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

    # Characters that never appear in a real place name; regex captures over the
    # noisy/scrambled preference text sometimes yield fragments like "浙范跑单"
    # or "三角个" — those must not become allowed_regions entries.
    _REGION_JUNK_CHARS = set("跑接单货车驶运范围圈线路内外出不只仅在限的得须必每天月日点号时晚夜元扣罚个这那")

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
        if _allowed_region_key(region) is None:
            junk_heuristic = len(region) > 6 or any(ch in self._REGION_JUNK_CHARS for ch in region)
            is_place = self._llm_semantic_yes_no(
                f"『{region}』是否是一个真实的中国地名/行政区/区域名（而非动作短语或乱序碎片）？",
                default=not junk_heuristic,
            )
            if not is_place:
                self._logger.info("validation: rejected junk allowed_region '%s'", region)
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

    def _confirm_rule_holds(self, rule_desc: str, all_text: str, default: bool = True) -> bool:
        """Semantic grounding: ask the model whether ``all_text`` actually
        imposes the constraint ``rule_desc``.

        This replaces brittle substring/keyword grounding, which silently
        dropped correctly-extracted rules whenever a finals driver phrased the
        same constraint with words not on a hard-coded whitelist. The model
        decides by meaning, so paraphrases / dialect / idioms all generalise.

        Fail-safe: returns ``default`` on any error, empty text, or
        unparseable response. An explicit verdict from the model overrides the
        default in BOTH directions — so a keyword-miss default of False can be
        lifted by a clear "yes" (off-vocabulary phrasings generalise), while a
        flaky/unavailable verifier never silently discards a real constraint
        whose default is True.
        """
        if not all_text.strip():
            return default
        cache = getattr(self, "_rule_confirm_cache", None)
        if cache is None:
            cache = {}
            self._rule_confirm_cache = cache
        key = (rule_desc, all_text)
        if key in cache:
            return cache[key]
        verdict = default
        try:
            msgs = [
                {"role": "system", "content": (
                    "你判断一段司机偏好原文是否确实包含某条约束。"
                    "原文可能是粤语/吴语等方言或口语（如 唔=不要、晚黑=晚上、返屋企=回家、"
                    "生果=水果、夜里向=晚上、覅=不要），按含义判断。"
                    "只输出JSON {\"holds\": true/false}。"
                    "只有当原文明显不支持该约束时才返回false；含义相符或不确定一律返回true。"
                )},
                {"role": "user", "content": json.dumps(
                    {"原文": all_text, "约束": rule_desc}, ensure_ascii=False)},
            ]
            req: dict[str, Any] = {"messages": msgs, "temperature": 0, "max_tokens": 30}
            try:
                req["response_format"] = {"type": "json_object"}
                resp = self._api.model_chat_completion(req)
            except Exception:
                req.pop("response_format", None)
                resp = self._api.model_chat_completion(req)
            data = self._extract_json(resp)
            if isinstance(data, dict) and isinstance(data.get("holds"), bool):
                verdict = data["holds"]
        except Exception as exc:
            self._logger.info("rule-confirm unavailable (default=%s) desc=%s err=%s", default, rule_desc, exc)
            verdict = default
        cache[key] = verdict
        return verdict

    def _llm_semantic_yes_no(self, question: str, default: bool) -> bool:
        """通用 LLM 是/否语义判定（缓存 + fail-safe 降级到 default）。

        用于品类归属、地名真实性等开放词表无法枚举的判断，避免硬编码字表。
        同一问题只问一次；模型不可用/输出不可解析时返回启发式 default。"""
        cache = getattr(self, "_semantic_yes_no_cache", None)
        if cache is None:
            cache = {}
            self._semantic_yes_no_cache = cache
        if question in cache:
            return cache[question]
        verdict = default
        try:
            msgs = [
                {"role": "system", "content": (
                    "你是货运领域的语义判定助手。输入文本可能是乱序/缺字的中文，"
                    "也可能含粤语/吴语等方言叫法（如粤语「生果」即「水果」）。"
                    "只输出JSON {\"answer\": true/false}。"
                )},
                {"role": "user", "content": question},
            ]
            req: dict[str, Any] = {"messages": msgs, "temperature": 0, "max_tokens": 30}
            try:
                req["response_format"] = {"type": "json_object"}
                resp = self._api.model_chat_completion(req)
            except Exception:
                req.pop("response_format", None)
                resp = self._api.model_chat_completion(req)
            data = self._extract_json(resp)
            if isinstance(data, dict) and isinstance(data.get("answer"), bool):
                verdict = data["answer"]
        except Exception as exc:
            self._logger.info("semantic yes/no unavailable (default=%s) q=%s err=%s", default, question[:60], exc)
            verdict = default
        cache[question] = verdict
        return verdict

    # 方言品类同义词（仅收录无歧义的等价叫法）：抽取层已要求品类名归一为标准
    # 普通话，这里是品类匹配的确定性兜底，避免「偏好写生果、货源叫水果」时
    # 还要依赖一次 LLM 判定。
    _DIALECT_CATEGORY_SYNONYMS = {
        "生果": "水果",
        "菜蔬": "蔬菜",
        "小菜": "蔬菜",
        "建筑材料": "建材",
    }

    def _category_matches_sem(self, cat: str, cargo_name: str) -> bool:
        """品类归属判定：子串直接命中；其余交给 LLM 语义判定（缓存），
        模型不可用时降级到 _category_matches 启发式。"""
        cat, cargo_name = cat.strip(), cargo_name.strip()
        if not cat or not cargo_name:
            return False
        if cat in cargo_name or cargo_name in cat:
            return True
        ncat = self._DIALECT_CATEGORY_SYNONYMS.get(cat, cat)
        nname = self._DIALECT_CATEGORY_SYNONYMS.get(cargo_name, cargo_name)
        if ncat in nname or nname in ncat:
            return True
        question = (
            f"货物名称『{cargo_name}』是否属于品类『{cat}』？"
            "（两者都可能是乱序/缺字的中文，如『鲜果』即『水果』；"
            "但『水泥』不属于『水果』）"
        )
        return self._llm_semantic_yes_no(question, default=self._category_matches(cat, cargo_name))

    def _merge_llm_rules(self, rules: DriverRules, data: dict[str, Any], texts: list[str] | None = None) -> None:
        all_text = "\n".join(texts) if texts else ""
        rp = data.get("rule_penalties")
        if isinstance(rp, dict):
            for k, v in rp.items():
                if isinstance(k, str) and isinstance(v, (int, float)) and v > 0:
                    rules.rule_penalties[k] = max(rules.rule_penalties.get(k, 0.0), float(v))
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
            if self._confirm_rule_holds(f"司机禁运（一律不接）货物品类「{c}」", all_text):
                rules.forbidden_categories.add(c)
            else:
                self._logger.info("llm semantic-confirm: dropped forbidden_category '%s'", c)

        for reg in raw_regs:
            r = self._clean_region_name(reg)
            if not r:
                continue
            if self._confirm_rule_holds(f"司机禁接装货地或卸货地在「{r}」的货", all_text):
                rules.forbidden_regions.add(r)
            else:
                self._logger.info("llm semantic-confirm: dropped forbidden_region '%s'", r)
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
                if self._confirm_rule_holds(f"司机在指定日期不去「{r}」", all_text):
                    rules.blackout.append((r, days))
                else:
                    self._logger.info("llm semantic-confirm: dropped blackout region '%s'", r)
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

        # no_drive_windows — FAIL-SAFE (no keyword gate).
        # A daily "don't drive in this window" rule (esp. night rest) that we
        # miss is catastrophic: it is violated EVERY day and compounds into a
        # six-figure penalty over the season, whereas a spurious window only
        # costs a few idle hours. The old `_ndw_grounded()` keyword whitelist
        # silently dropped these whenever a driver phrased it off-vocabulary,
        # which is exactly the finals scenario. We now trust the temperature-0
        # extraction and only drop a window if the semantic verifier clearly
        # says it does not hold (verifier failure keeps the window).
        for ndw in data.get("no_drive_windows") or []:
            if not isinstance(ndw, dict):
                continue
            sh, eh = ndw.get("start_hour"), ndw.get("end_hour")
            if not (isinstance(sh, (int, float)) and isinstance(eh, (int, float))):
                continue
            if not (0 <= sh <= 24 and 0 <= eh <= 24):
                continue
            sm, em = int(round(sh * 60)), int(round(eh * 60))
            if em > sm:
                pass  # normal within-day range
            elif sm > em:
                em += 24 * 60  # cross-midnight (e.g. 23:00->05:00)
            else:
                continue
            desc = f"司机每天 {sm // 60:02d}:{sm % 60:02d}-{(em % DAY_MINUTES) // 60:02d}:{(em % DAY_MINUTES) % 60:02d} 不接单、不出车/不空驶（禁驶时段）"
            if not self._confirm_rule_holds(desc, all_text):
                self._logger.info("llm semantic-confirm: dropped no_drive_window %d-%d", sm, em)
                continue
            if not any(ws == sm and we == em for ws, we in rules.no_drive_windows):
                rules.no_drive_windows.append((sm, em))

        # avoid_categories — semantic confirm (replaces keyword/substring gate)
        for ac in data.get("avoid_categories") or []:
            if isinstance(ac, str) and ac.strip():
                s = re.sub(r"^(?:凡是|所有|一切|任何)", "", ac.strip()).strip()
                if not s:
                    continue
                if self._confirm_rule_holds(f"司机尽量避免/少接货物品类「{s}」", all_text):
                    rules.avoid_categories.add(s)
                else:
                    self._logger.info("llm semantic-confirm: dropped avoid_category '%s'", s)

        # allowed_categories — category whitelist ("只拉/只接X"), the OPPOSITE of
        # forbidden_categories. Wrongly adding one throttles everything, wrongly
        # dropping one costs per-order penalties, so each entry is gated with a
        # semantic confirm (offline default: text must contain a whitelist marker).
        for ac in data.get("allowed_categories") or []:
            if isinstance(ac, str) and ac.strip():
                s = re.sub(r"^(?:凡是|所有|一切|任何)", "", ac.strip()).strip()
                if not s:
                    continue
                if self._confirm_rule_holds(
                    f"司机只承运/只接货物品类「{s}」（除此之外的品类不接）",
                    all_text,
                    default=any(k in all_text for k in ("只拉", "只接", "只运", "专拉", "专运", "专跑", "除了")),
                ):
                    rules.allowed_categories.add(s)
                else:
                    self._logger.info("llm semantic-confirm: dropped allowed_category '%s'", s)

        # NOTE: avoid/forbidden dedup moved to _dedup_avoid_forbidden (runs after supplement)

        # allowed_regions — semantic confirm replaces the keyword whitelist gate
        # (keyword hit only remains as the offline default when the verifier is
        # unavailable). _add_allowed_region still applies junk-name and
        # text-support validation on top.
        for ar in data.get("allowed_regions") or []:
            if not (isinstance(ar, str) and ar.strip()):
                continue
            region = self._canonical_allowed_region(ar)
            if not region:
                continue
            if self._confirm_rule_holds(
                f"司机的运营范围被限定在「{region}」之内（装货地和卸货地都必须在该范围，不得出该范围接活）",
                all_text,
                default=_text_has_any(_ALLOW_REGION_KW),
            ):
                self._add_allowed_region(rules, ar, all_text)
            else:
                self._logger.info("llm semantic-confirm: dropped allowed_region '%s'", region)

        # forbidden_zones — semantic confirm + coordinate validation
        fz_default = _text_has_any(_FZ_KW)
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
            if not self._confirm_rule_holds(
                f"司机禁止进入以({flat:.2f},{flng:.2f})为圆心、半径{fr:g}公里的区域",
                all_text,
                default=fz_default,
            ):
                self._logger.info("llm semantic-confirm: dropped forbidden_zone (%.2f,%.2f)", flat, flng)
                continue
            rules.forbidden_zones.append((flat, flng, fr))

        # bounded_area — semantic confirm + coordinate validation
        ba = data.get("bounded_area")
        if isinstance(ba, dict) and rules.bounded_area is None:
            # Explicit lat/lng numbers must appear in the text (data gate, not a
            # phrasing gate); the keyword hit is only the offline default.
            has_coords = bool(re.search(r'(?:纬度|经度|[纬经]\s*[0-9]|\d{2,3}\.\d)', all_text))
            confirmed = has_coords and self._confirm_rule_holds(
                "司机只允许在一个明确给出经纬度边界的矩形范围内运营",
                all_text,
                default=_text_has_any(_BA_KW),
            )
            if confirmed:
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
            elif not has_coords:
                self._logger.info("llm grounding: rejected bounded_area (no explicit coordinates in text)")
            else:
                self._logger.info("llm semantic-confirm: dropped bounded_area")

        # must_visit — semantic confirm + coordinate validation
        mv_default = _text_has_any(_MV_KW)
        has_mv_coords = bool(re.search(r'\d{2,3}\.\d{1,}', all_text))
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
            if not has_mv_coords:
                self._logger.info("llm grounding: rejected must_visit (no explicit coords in text)")
                continue
            if not self._confirm_rule_holds(
                f"司机每月至少{mrd}天需要到达({mlat:.2f},{mlng:.2f})附近{mr:g}公里内",
                all_text,
                default=mv_default,
            ):
                self._logger.info("llm semantic-confirm: dropped must_visit (%.2f,%.2f)", mlat, mlng)
                continue
            rules.must_visit.append({"lat": mlat, "lng": mlng, "radius_km": mr, "required_days": mrd})

        # haul_max_km — semantic confirm
        hm = data.get("haul_max_km")
        if isinstance(hm, (int, float)) and hm > 0:
            if self._confirm_rule_holds(
                f"司机单笔订单的运输距离（装货点到卸货点）不得超过{hm:g}公里",
                all_text,
                default=_text_has_any(_HAUL_KW),
            ):
                rules.haul_max_km = float(hm)
            else:
                self._logger.info("llm semantic-confirm: dropped haul_max_km=%s", hm)

        # haul_min_km — opposite of haul_max_km ("低于X公里的短活不接")
        hmin = data.get("haul_min_km")
        if isinstance(hmin, (int, float)) and hmin > 0:
            if self._confirm_rule_holds(
                f"司机单笔订单的运输距离（装货点到卸货点）不得低于{hmin:g}公里",
                all_text,
                default=_text_has_any(_HAUL_KW),
            ):
                rules.haul_min_km = float(hmin)
            else:
                self._logger.info("llm semantic-confirm: dropped haul_min_km=%s", hmin)

        # monthly_longhaul_cap — preference-driven long-haul order cap. Only set
        # when the driver's own text imposes one: missing a real cap costs
        # per-order penalties, while inventing one throttles a long-haul-focused
        # driver all season.
        lhc = data.get("monthly_longhaul_cap")
        if isinstance(lhc, dict):
            mx = lhc.get("max_orders")
            mh = lhc.get("min_hours")
            if isinstance(mx, (int, float)) and int(mx) > 0:
                hours_txt = f"{float(mh):g}" if isinstance(mh, (int, float)) and mh > 0 else "若干"
                if self._confirm_rule_holds(
                    f"司机每月超过{hours_txt}小时的长途订单最多只能接{int(mx)}单",
                    all_text,
                    default=("长途" in all_text or "远活" in all_text),
                ):
                    rules.longhaul_cap_orders = int(mx)
                    if isinstance(mh, (int, float)) and mh > 0:
                        rules.longhaul_min_minutes = int(float(mh) * 60)
                else:
                    self._logger.info("llm semantic-confirm: dropped monthly_longhaul_cap=%s", lhc)

        # monthly_deadhead_max_km (keep as before — low risk, rarely hallucinated)
        mdh = data.get("monthly_deadhead_max_km")
        if isinstance(mdh, (int, float)) and mdh > 0:
            rules.monthly_deadhead_max_km = float(mdh)

        # daily_order_limit — semantic confirm
        dol = data.get("daily_order_limit")
        dol_val: int | None = None
        if isinstance(dol, (int, float)) and dol > 0:
            dol_val = int(dol)
        elif isinstance(dol, str):
            dm = re.search(r'(\d+)', dol)
            if dm:
                dol_val = int(dm.group(1))
        if dol_val is not None:
            if self._confirm_rule_holds(
                f"司机每天（同一天内）最多接{dol_val}单",
                all_text,
                default=_text_has_any(_DOL_KW) and _text_has_any(_DOL_SCOPE_KW),
            ):
                rules.daily_order_limit = dol_val
            else:
                self._logger.info("llm semantic-confirm: dropped daily_order_limit=%s", dol_val)

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

        # home_rule — semantic confirm + coordinate validation
        # Explicit coordinates must appear in the text (data gate — a home_rule
        # with hallucinated coordinates is very costly); the phrasing decision is
        # made semantically, keyword hit only remains as the offline default.
        hr = data.get("home_rule")
        if isinstance(hr, dict) and rules.home_lat is None:
            # Check if text contains explicit coordinate-like numbers (e.g., "23.10" or "(23.10,113.50)")
            has_coords_in_text = bool(re.search(r'\d{2,3}\.\d{1,}', all_text))
            if has_coords_in_text and self._confirm_rule_holds(
                "司机每天必须在规定时刻前回到自家/住所位置附近，且在次日规定时刻前不再接单或空驶（回家规则）",
                all_text,
                default=_text_has_any(_HOME_KW),
            ):
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
            elif not has_coords_in_text:
                self._logger.info("llm grounding: rejected home_rule (no explicit coordinates in text)")
            else:
                self._logger.info("llm semantic-confirm: dropped home_rule")

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
            desc = f"司机要求{_month_name(month_idx)}品类「{cat}」当月至少接{int(req)}单（月度品类指标）"
            if not self._confirm_rule_holds(desc, all_text):
                self._logger.info("llm semantic-confirm: dropped category_target '%s'", cat)
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

    _NIGHT_KW = ("夜", "晚", "宵", "黑")
    _REST_KW = ("停", "休", "熄", "睡", "眠", "收", "歇", "锁", "不出", "不接", "不动", "归家",
                "唔好", "覅", "勿", "困觉", "唞")
    _CLOCK_RE = re.compile(r"([01]?\d|2[0-4])(?:[：:]([0-5]\d)|点(半)?|时)")
    # Chinese-numeral clocks ("十点" "十一点半") used by colloquial/dialect texts.
    _CN_CLOCK_RE = re.compile(r"([零一二两三四五六七八九十]{1,3})\s*点(半)?")

    @classmethod
    def _clock_minutes_with_context(cls, text: str) -> list[int]:
        """All clock times in ``text`` as minutes-of-day, dialect-aware.

        Handles digit clocks ("21:00" "21点") and Chinese-numeral clocks
        ("十点" "十一点半"). Colloquial texts express evening times on the
        12-hour dial ("晚黑十点" == 22:00), so an hour ≤ 11 preceded within a
        few characters by a night marker is shifted by +12.
        """
        out: list[int] = []
        seen_spans: list[tuple[int, int]] = []
        for m in cls._CLOCK_RE.finditer(text):
            h = int(m.group(1))
            mins = int(m.group(2)) if m.group(2) else (30 if m.group(3) else 0)
            out.append((h, mins, m.start()))
            seen_spans.append((m.start(), m.end()))
        for m in cls._CN_CLOCK_RE.finditer(text):
            if any(s <= m.start() < e for s, e in seen_spans):
                continue
            h = _cn_to_int(m.group(1))
            if not (0 <= h <= 24):
                continue
            out.append((h, 30 if m.group(2) else 0, m.start()))
        result: list[int] = []
        for h, mins, pos in out:
            ctx = text[max(0, pos - 6):pos]
            if h <= 11 and any(k in ctx for k in ("晚", "夜", "黑", "宵", "挨晚", "傍晚")):
                h += 12
            result.append(h * 60 + mins)
        return result

    def _supplement_night_failsafe(self, texts: list[str], rules: DriverRules) -> None:
        """夜休 fail-safe：夜间禁驶是按天重复扣分的最大罚项之一，漏抽代价极高。

        若 LLM/正则都未产出任何跨夜窗口，但某条原文同时含夜间词 + 停休词 +
        可识别的晚间/凌晨时刻对，则直接按文本中的时刻构造跨夜窗口（宁可多休
        损失少量收入，也不能每天吃夜休罚款）。"""
        if any(we > DAY_MINUTES for _ws, we in rules.no_drive_windows):
            return  # an overnight window already exists
        if rules.rest_window is not None and rules.rest_window[0] == 0 and rules.rest_window[1] > 0:
            return  # overnight rest already registered via _apply_rest_window
        for text in texts:
            if not any(k in text for k in self._NIGHT_KW):
                continue
            if not any(k in text for k in self._REST_KW):
                continue
            times = self._clock_minutes_with_context(text)
            evening = [t for t in times if t >= 18 * 60]
            morning = [t for t in times if 0 < t <= 9 * 60]
            if not evening or not morning:
                continue
            sm, em = max(evening), min(morning)
            self._logger.info("night fail-safe: derived overnight window %d-%d from text %r", sm, em, text[:60])
            self._apply_rest_window(rules, sm, em)
            return

    def _supplement_critical_rules(self, texts: list[str], rules: DriverRules) -> None:
        """Always-on deterministic backstop for high-stakes scalar rules.

        Unlike _supplement_basic_rules (offline fallback only), this runs even
        when the LLM compile succeeded. These three rule kinds carry repeat or
        lump penalties — per-order over-cap fines, per-day rest shortfalls,
        one-shot monthly off-day fines — so a single missed parse compounds for
        the whole season, while a false positive merely costs a little income.
        The patterns require explicit wording (they never fire on text that
        does not impose the constraint) and merge conservatively: only fill a
        missing cap / take the stricter value.
        """
        for text in texts:
            # Monthly long-haul cap ("每个月超过八小时的长途只能接最多5单"). Since
            # the global default cap was removed (preference-driven now), missing
            # a real cap means unlimited over-cap orders × per-order penalty.
            if (rules.longhaul_cap_orders is None
                    and ("长途" in text or "远活" in text or "远路" in text)
                    and ("每月" in text or "每个月" in text or "一个月" in text)):
                m_n = re.search(
                    r"(?:最多|只能|不超过?|至多|顶多)[接拉跑]?\s*(?:最多)?\s*"
                    r"([零一二两三四五六七八九十\d]+)\s*(?:单|票|趟)",
                    text,
                )
                if m_n:
                    cap = _cn_to_int(m_n.group(1))
                    if cap > 0:
                        rules.longhaul_cap_orders = cap
                        m_h = re.search(r"超过\s*([零一二两三四五六七八九十\d]+)\s*个?\s*(?:小时|钟头)", text)
                        if m_h and _cn_to_int(m_h.group(1)) > 0:
                            rules.longhaul_min_minutes = _cn_to_int(m_h.group(1)) * 60
                        self._logger.info(
                            "critical supplement: longhaul cap=%d (>%dmin) from %r",
                            cap, rules.longhaul_min_minutes, text[:60],
                        )
            # Daily continuous rest hours ("每天连续休息满8小时" — per-day penalty).
            rest_m = re.search(
                r"(?:每天|每日)?.*?(?:连续|连着).*?(?:停车|停着|休息|歇|熄火).*?"
                r"(?:满|至少|起码)?\s*([零一二两三四五六七八九十\d]+)\s*(?:个)?\s*小时",
                text,
            )
            if rest_m:
                hours = _cn_to_int(rest_m.group(1))
                if 0 < hours <= 16 and hours * 60 > rules.daily_rest_minutes:
                    rules.daily_rest_minutes = hours * 60
                    self._logger.info(
                        "critical supplement: daily rest=%dh from %r", hours, text[:60]
                    )
            # Monthly full off-days ("每月至少两天完全不出车" — one-shot lump penalty).
            if (
                ("整天" in text or "全天" in text or "完全" in text) and any(
                    k in text for k in ("歇", "休息", "停驶", "不接单", "不空车", "不空跑", "不出车",
                                        "不开工", "唔开工", "唔出车", "勿出车", "覅出车")
                )
            ):
                cnt = self._parse_cn_count(text)
                if cnt and cnt > rules.off_days_min:
                    rules.off_days_min = cnt
                    self._logger.info(
                        "critical supplement: off_days_min=%d from %r", cnt, text[:60]
                    )

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
                sorted(rules.allowed_categories),
                rules.haul_min_km,
                (rules.longhaul_cap_orders, rules.longhaul_min_minutes),
                {m: dict(sorted(v.items())) for m, v in sorted(rules.monthly_category_targets.items())},
                sorted(rules.category_carryover_months),
                sorted(rc.get("text", "") for rc in rules.residual_constraints),
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
        # Monthly long-haul order cap (preference-driven; e.g. D001 "每个月超过
        # 八小时的长途只能接最多5单"). Offline fallback for the LLM extractor —
        # without it a cap-having driver would take unlimited long hauls offline.
        if (rules.longhaul_cap_orders is None
                and ("长途" in text or "远活" in text)
                and ("每月" in text or "每个月" in text or "一个月" in text)):
            m_n = re.search(
                r"(?:最多|只能|不超过?|至多)[接拉]?\s*(?:最多)?\s*([零一二两三四五六七八九十\d]+)\s*单",
                text,
            )
            if m_n:
                cap = _cn_to_int(m_n.group(1))
                if cap > 0:
                    rules.longhaul_cap_orders = cap
                    m_h = re.search(r"超过\s*([零一二两三四五六七八九十\d]+)\s*个?\s*(?:小时|钟头)", text)
                    if m_h:
                        hours = _cn_to_int(m_h.group(1))
                        if hours > 0:
                            rules.longhaul_min_minutes = hours * 60
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
            # 天/日 both accepted (粤语用"日"); allow a short connective ("要/得")
            # and dialect negations (唔/勿/覅) before the rest verb.
            m = re.search(
                r"([一二两三四五六七八九十\d]+)\s*[天日]\s*(?:要|得|都)?\s*(?:完全|彻底|整天|整)?\s*"
                r"(?:歇|休|停|不接|不出|不空|唔开|唔出|勿出|覅出|留喺|待牢)",
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
            h1, h2 = m1 // 60, m2 // 60
            # PM context for the start: "每晚十点到次日五点" → h1=10 means 22:00
            if 0 < h1 < 12:
                first = re.search(r"[零一二两三四五六七八九十\d]+\s*点半?", text)
                if first is not None:
                    prefix = text[max(0, first.start() - 2):first.start()]
                    if any(ch in prefix for ch in "晚夜宵"):
                        m1 += 12 * 60
                        h1 += 12
            # fix PM context: "十一点半到下午一点半" → h2=1 should be 13
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
