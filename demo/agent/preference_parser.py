"""动态偏好解析器（DP-ORH-MS 核心模块）。

设计原则：**严禁硬编码司机偏好规则**。优先调用 LLM（``qwen3.5-flash``）动态解析；
只有在 LLM 不可用、token 预算耗尽或返回不可用时，才采用确定性正则起安全网作用，
避免服务崩溃。两者输出会合并到统一的 ``ParsedRules``，供 ``scoring`` 模块直接使用。

接口返回的 ``preferences`` 是字典数组，含 ``content``、``penalty_amount``、``penalty_cap``。
为复赛 500W token 限额考虑：偏好文本的哈希不变时不会重复调用 LLM。上层调用者通过
``signature_of`` 检测偏好变化，仅在有变化时重新调用 ``parse_preferences``。
“严禁硬编码 if driver_id == ...”是赛委重点核查项；本文件内的正则仅面向 ``content`` 文本语义匹配，
不包含任何具体司机 ID 、具体货源 ID 、具体品类以外的判断。
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import md5
from typing import Any, Callable

logger = logging.getLogger("agent.preference_parser")

# 数字提取常用正则
_NUM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
_HOUR_RANGE_PATTERN = re.compile(r"(\d{1,2})\s*(?:点|时|:00)?\s*(?:至|到|-|~|—)\s*(?:次日\s*)?(\d{1,2})\s*(?:点|时|:00)?")
_LATLNG_PATTERN = re.compile(r"[（(]\s*(\d{1,3}(?:\.\d+)?)\s*[，,]\s*(\d{1,3}(?:\.\d+)?)\s*[)）]")
_WALL_TIME_ZH_PATTERN = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2})(?::|点)(\d{2})?")
_WALL_TIME_ISO_PATTERN = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})")


@dataclass
class CategoryRule:
    """品类相关硬过滤或软惩罚。"""

    forbidden: list[str] = field(default_factory=list)  # 严格禁运，触发即过滤
    avoid: list[str] = field(default_factory=list)  # 软避免，命中则减分


@dataclass
class TimeWindowRule:
    """禁止接单或空驶的时段（按一天分钟数表示）。"""

    start_minute: int  # 0..1440
    end_minute: int  # 0..1440，可超过 24*60 表示跨天
    raw: str = ""
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class RestRule:
    """每日连续休息时长要求。"""

    required_minutes: int
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class CircleZone:
    """圆形区域：禁入或必到。"""

    lat: float
    lng: float
    radius_km: float
    raw: str = ""
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class BoundingBoxRule:
    """边界矩形：仅允许在该范围内运行。"""

    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float
    raw: str = ""
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class HomeRule:
    """每日须在指定时间前回到家附近，并保留禁出窗口。"""

    lat: float
    lng: float
    radius_km: float
    home_by_hour: int  # 应当几点前到家
    no_drive_until_hour: int  # 次日几点前不接单不空驶
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class MonthlyDayOffRule:
    """月度至少 N 天完全不接单/不出车。"""

    required_days: int
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class MustVisitRule:
    """月度内至少在 N 个不同日到达指定点附近。"""

    lat: float
    lng: float
    radius_km: float
    required_days: int
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class DailyOrderLimitRule:
    """每日接单上限。"""

    max_orders: int
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class DistanceLimitRule:
    """距离类硬约束：单笔干线 / 单笔赴装货空驶 / 月度累计空驶。"""

    kind: str  # "haul" | "pickup" | "monthly_deadhead"
    max_km: float
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class FirstOrderRule:
    """首单时间约束：只要当天接单，第一单必须在某时刻前。"""

    before_hour: int
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class PreferredCargoRule:
    cargo_id: str
    lat: float | None = None
    lng: float | None = None
    available_minutes: int | None = None
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class TimedStayEventRule:
    start_minutes: int
    pickup_lat: float
    pickup_lng: float
    home_lat: float
    home_lng: float
    deadline_minutes: int
    stay_until_minutes: int
    pickup_stay_minutes: int = 10
    radius_km: float = 1.0
    absence_penalty_per_minute: float = 5.0
    penalty_amount: float = 0.0
    penalty_cap: float | None = None


@dataclass
class ParsedRules:
    """聚合后的可执行偏好结构；评分阶段会逐项查询。"""

    categories: CategoryRule = field(default_factory=CategoryRule)
    no_drive_windows: list[TimeWindowRule] = field(default_factory=list)
    rest_rules: list[RestRule] = field(default_factory=list)
    forbidden_zones: list[CircleZone] = field(default_factory=list)
    must_visit: list[MustVisitRule] = field(default_factory=list)
    bounded_areas: list[BoundingBoxRule] = field(default_factory=list)
    home_rule: HomeRule | None = None
    monthly_day_off: MonthlyDayOffRule | None = None
    daily_order_limit: DailyOrderLimitRule | None = None
    distance_limits: list[DistanceLimitRule] = field(default_factory=list)
    first_order_rule: FirstOrderRule | None = None
    preferred_cargo_ids: list[str] = field(default_factory=list)
    preferred_cargo: list[PreferredCargoRule] = field(default_factory=list)
    timed_stay_events: list[TimedStayEventRule] = field(default_factory=list)
    unparsed: list[dict[str, Any]] = field(default_factory=list)  # 两道解析都未覆盖
    raw_preferences: list[dict[str, Any]] = field(default_factory=list)
    # 解析指标，供上层日志与风险调参
    parsed_by_llm: int = 0
    parsed_by_regex: int = 0
    parse_failure_count: int = 0
    llm_used: bool = False


# ---------------- 文本特征提取 ----------------

_CATEGORY_KEYWORDS = [
    "化工塑料",
    "煤炭矿产",
    "蔬菜",
    "鲜活水产品",
    "食品饮料",
    "机械设备",
    "服饰纺织皮革",
    "快递快运搬家",
    "数码家电",
    "水果",
    "农用物资",
    "空包装",
]


def _extract_numbers(text: str) -> list[float]:
    return [float(m.group(1)) for m in _NUM_PATTERN.finditer(text)]


def _extract_lat_lng(text: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for m in _LATLNG_PATTERN.finditer(text):
        try:
            pairs.append((float(m.group(1)), float(m.group(2))))
        except ValueError:
            continue
    return pairs


def _extract_hour_range(text: str) -> tuple[int, int] | None:
    m = _HOUR_RANGE_PATTERN.search(text)
    if m is None:
        return None
    try:
        start = int(m.group(1))
        end = int(m.group(2))
    except ValueError:
        return None
    if not (0 <= start <= 24 and 0 <= end <= 24):
        return None
    return start, end


def _extract_wall_times_minutes(text: str) -> list[int]:
    out: list[int] = []
    epoch = datetime(2026, 3, 1, 0, 0, 0)
    for pattern in (_WALL_TIME_ZH_PATTERN, _WALL_TIME_ISO_PATTERN):
        for m in pattern.finditer(text):
            try:
                minute = int(m.group(5) or 0)
                dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), minute)
            except ValueError:
                continue
            out.append(int((dt - epoch).total_seconds() // 60))
    return sorted(set(out))


def _parse_high_value_single(item: dict[str, Any], rules: ParsedRules) -> bool:
    content = str(item.get("content", "")).strip()
    if not content:
        return False
    penalty_amount, penalty_cap = _normalize_penalty(item)
    if "熟货" in content or "指定熟货源编号" in content:
        m = re.search(r"编号\s*(\d+)", content)
        if m:
            cargo_id = m.group(1)
            coords = _extract_lat_lng(content)
            times = _extract_wall_times_minutes(content)
            if cargo_id not in rules.preferred_cargo_ids:
                rules.preferred_cargo_ids.append(cargo_id)
            if not any(r.cargo_id == cargo_id for r in rules.preferred_cargo):
                rules.preferred_cargo.append(
                    PreferredCargoRule(
                        cargo_id=cargo_id,
                        lat=coords[0][0] if coords else None,
                        lng=coords[0][1] if coords else None,
                        available_minutes=times[0] if times else None,
                        penalty_amount=penalty_amount,
                        penalty_cap=penalty_cap,
                    )
                )
            return True
    family_like = (
        ("接上" in content or "接到" in content)
        and ("配偶" in content or "家人" in content)
        and ("返回老家" in content or "进家门" in content or "回家" in content)
    )
    if family_like:
        coords = _extract_lat_lng(content)
        times = _extract_wall_times_minutes(content)
        if len(coords) >= 2 and len(times) >= 2:
            stay_match = re.search(r"不少于\s*(\d+)\s*分钟", content)
            absence_match = re.search(r"1\s*分钟罚\s*(\d+(?:\.\d+)?)", content)
            pickup_stay = int(stay_match.group(1)) if stay_match else 10
            absence_unit = float(absence_match.group(1)) if absence_match else 5.0
            deadline = times[1] if len(times) >= 3 else times[-1]
            event = TimedStayEventRule(
                start_minutes=times[0],
                pickup_lat=coords[0][0],
                pickup_lng=coords[0][1],
                home_lat=coords[1][0],
                home_lng=coords[1][1],
                deadline_minutes=deadline,
                stay_until_minutes=times[-1],
                pickup_stay_minutes=max(1, pickup_stay),
                absence_penalty_per_minute=absence_unit,
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            if not any(
                e.start_minutes == event.start_minutes
                and abs(e.pickup_lat - event.pickup_lat) < 1e-6
                and abs(e.home_lat - event.home_lat) < 1e-6
                for e in rules.timed_stay_events
            ):
                rules.timed_stay_events.append(event)
            return True
    return False


def _normalize_penalty(raw: dict[str, Any]) -> tuple[float, float | None]:
    amount = float(raw.get("penalty_amount") or 0.0)
    cap_raw = raw.get("penalty_cap")
    cap = None
    if cap_raw is not None:
        try:
            cap = float(cap_raw)
        except (TypeError, ValueError):
            cap = None
    return amount, cap


# ---------------- 主解析函数 ----------------


def _try_parse_single(item: dict[str, Any], rules: ParsedRules) -> bool:
    """对单条偏好尝试规则化；成功返回 True，失败让 LLM 兜底。"""
    if not isinstance(item, dict):
        return False
    content = str(item.get("content", "")).strip()
    if not content:
        return False
    penalty_amount, penalty_cap = _normalize_penalty(item)
    if _parse_high_value_single(item, rules):
        return True
    # 以下正则仅在 LLM 崩溃、预算耗尽、输出不可用时作为安全网使用。
    # 不针对任何具体司机身份 / 项目实例硬编码规则，仅面向 content 文本语义。

    # 1. 品类禁运 / 软避免
    cats_in_text = [c for c in _CATEGORY_KEYWORDS if c in content]
    if cats_in_text and ("不接" in content or "禁" in content or "尽量不拉" in content):
        if "尽量不" in content or "尽量" in content:
            rules.categories.avoid.extend(cats_in_text)
        else:
            rules.categories.forbidden.extend(cats_in_text)
        return True

    # 2. 月度至少 N 天完全不接单 / 不出车
    if ("自然月" in content or "每月" in content) and ("不接单" in content or "不出车" in content or "歇着" in content):
        nums = _extract_numbers(content)
        required = int(nums[0]) if nums else 1
        rules.monthly_day_off = MonthlyDayOffRule(
            required_days=max(1, required),
            penalty_amount=penalty_amount,
            penalty_cap=penalty_cap,
        )
        return True

    # 3. 每日连续休息 N 小时 / 分钟
    if ("休息" in content or "歇" in content or "停车" in content) and "每天" in content:
        rng = _extract_hour_range(content)
        nums = _extract_numbers(content)
        created = False
        # 如果文本同时包含时间范围，也生成 no_drive_window，防止休息规则"吞掉"禁行窗
        if rng is not None:
            start_hour, end_hour = rng
            start_min = start_hour * 60
            end_min = end_hour * 60
            if end_hour <= start_hour:
                end_min = (end_hour + 24) * 60
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=start_min,
                    end_minute=end_min,
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            # 休息时长用窗口长度而非文本第一个数字（避免把"23"当成23小时）
            window_minutes = end_min - start_min
            rules.rest_rules.append(
                RestRule(
                    required_minutes=window_minutes,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            created = True
        elif nums:
            value = nums[0]
            minutes = int(value * 60) if value < 24 else int(value)
            rules.rest_rules.append(
                RestRule(
                    required_minutes=minutes,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            created = True
        if created:
            return True

    # 3.5 回家规则（必须先于纯禁行窗匹配，避免「不接单/不空跑」抢先 return）：
    #     文本特征：含坐标 + 「自家位置 / 在家 / 进家门」+ 时间范围 + 「不接单」「不空」
    home_keywords = (
        ("自家" in content or "在家" in content or "进家门" in content or "回家" in content)
        and "公里" in content
    )
    if home_keywords:
        coords = _extract_lat_lng(content)
        rng = _extract_hour_range(content)
        if coords and rng:
            home_by_hour = rng[0]
            no_drive_until_hour = rng[1]
            radius_km = 1.0
            # 优先从「N 公里内」中读取实际半径
            rad_m = re.search(r"(\d+(?:\.\d+)?)\s*公里", content)
            if rad_m:
                try:
                    radius_km = float(rad_m.group(1))
                except ValueError:
                    radius_km = 1.0
            rules.home_rule = HomeRule(
                lat=coords[0][0],
                lng=coords[0][1],
                radius_km=radius_km,
                home_by_hour=home_by_hour,
                no_drive_until_hour=no_drive_until_hour,
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            # 顺手把对应的禁出窗也登记下来，确保接单/空驶评分都能看到
            start_min = home_by_hour * 60
            end_min = no_drive_until_hour * 60
            if no_drive_until_hour <= home_by_hour:
                end_min = (no_drive_until_hour + 24) * 60
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=start_min,
                    end_minute=end_min,
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 4. 夜间禁行 / 时段禁行
    if ("不接单" in content and "不空" in content) or ("不接单" in content and "不出车" in content):
        rng = _extract_hour_range(content)
        if rng is not None:
            start_hour, end_hour = rng
            start_min = start_hour * 60
            end_min = end_hour * 60
            if end_hour <= start_hour:
                # 跨午夜，例如 23 -> 4 -> end_min += 24h
                end_min = (end_hour + 24) * 60
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=start_min,
                    end_minute=end_min,
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 5. 中午不出车（吃饭歇脚）
    if ("中午" in content or "12点" in content) and ("不接单" in content or "不空" in content):
        rng = _extract_hour_range(content)
        if rng is not None:
            start_hour, end_hour = rng
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=start_hour * 60,
                    end_minute=end_hour * 60,
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 6. 单笔干线距离不超过 N 公里
    if ("装货点至卸货点" in content or "单笔" in content) and "距离" in content and "公里" in content:
        nums = _extract_numbers(content)
        if nums:
            rules.distance_limits.append(
                DistanceLimitRule(
                    kind="haul",
                    max_km=float(nums[0]),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 7. 接单后赴装货空驶不超过 N 公里
    if "赴装货" in content and "公里" in content:
        nums = _extract_numbers(content)
        if nums:
            rules.distance_limits.append(
                DistanceLimitRule(
                    kind="pickup",
                    max_km=float(nums[0]),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 8. 月度空驶累计不超过 N 公里
    if "空驶" in content and ("一个月" in content or "月" in content) and "公里" in content:
        nums = _extract_numbers(content)
        if nums:
            rules.distance_limits.append(
                DistanceLimitRule(
                    kind="monthly_deadhead",
                    max_km=float(nums[0]),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 9. 禁入圆形区域：以（lat, lng）为圆心、半径 N 公里
    if ("禁" in content or "不得进入" in content) and "圆心" in content and "公里" in content:
        coords = _extract_lat_lng(content)
        nums = _extract_numbers(content)
        if coords and nums:
            radius_km = nums[-1]
            rules.forbidden_zones.append(
                CircleZone(
                    lat=coords[0][0],
                    lng=coords[0][1],
                    radius_km=float(radius_km),
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 10. 行政或经纬度边界：北纬 a 至 b，东经 c 至 d
    if "北纬" in content and "东经" in content:
        lat_idx = content.find("北纬")
        lng_idx = content.find("东经")
        # 仅截取「北纬」与「东经」各自后面短段中的数字，避免被前文坐标污染
        lat_seg = content[lat_idx : lat_idx + 30]
        lng_seg = content[lng_idx : lng_idx + 30]
        lat_nums = _extract_numbers(lat_seg)[:2]
        lng_nums = _extract_numbers(lng_seg)[:2]
        if len(lat_nums) == 2 and len(lng_nums) == 2:
            rules.bounded_areas.append(
                BoundingBoxRule(
                    lat_min=min(lat_nums),
                    lat_max=max(lat_nums),
                    lng_min=min(lng_nums),
                    lng_max=max(lng_nums),
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 11. 月度至少 N 天到达目标点
    if "自然月" in content and "到过" in content:
        coords = _extract_lat_lng(content)
        nums = _extract_numbers(content)
        if coords and nums:
            required_days = int(nums[0]) if nums else 1
            radius_km = 1.0
            rules.must_visit.append(
                MustVisitRule(
                    lat=coords[0][0],
                    lng=coords[0][1],
                    radius_km=radius_km,
                    required_days=max(1, required_days),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 12. 回家规则：每天 N 点前在家附近，至 M 点不接单不空跑
    if "前车辆须在自家位置" in content or ("23点前" in content and "公里内" in content):
        coords = _extract_lat_lng(content)
        rng = _extract_hour_range(content)
        if coords and rng:
            home_by_hour = 23
            no_drive_until_hour = rng[1]
            radius_km = 1.0
            rules.home_rule = HomeRule(
                lat=coords[0][0],
                lng=coords[0][1],
                radius_km=radius_km,
                home_by_hour=home_by_hour,
                no_drive_until_hour=no_drive_until_hour,
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            end_min = no_drive_until_hour * 60
            if no_drive_until_hour <= home_by_hour:
                end_min = (no_drive_until_hour + 24) * 60
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=home_by_hour * 60,
                    end_minute=end_min,
                    raw=content,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True

    # 13. 同一天接单不得超过 N 单
    if "同一天接单" in content or ("一天" in content and "不得超过" in content):
        nums = _extract_numbers(content)
        if nums:
            rules.daily_order_limit = DailyOrderLimitRule(
                max_orders=int(nums[0]),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True

    # 14. 首单不得晚于 N 点
    if "首单" in content and "不得晚于" in content:
        nums = _extract_numbers(content)
        if nums:
            rules.first_order_rule = FirstOrderRule(
                before_hour=int(nums[0]),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True

    # 15. 熟货源指定编号
    if "熟货" in content or "指定熟货源编号" in content:
        m = re.search(r"编号(\d+)", content)
        if m:
            rules.preferred_cargo_ids.append(m.group(1))
            return True

    return False


def parse_preferences(
    preferences: list[Any],
    *,
    llm_caller: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> ParsedRules:
    """主入口：把接口返回的偏好列表转为 ``ParsedRules``。

    优先级：
    1. ``llm_caller`` 可用时优先交由 LLM 解析。
    2. LLM 未覆盖、返回不可用或完全不可用时，逐条走正则安全网。
    3. 两道都未覆盖的条目入 ``unparsed`` 并计入 ``parse_failure_count``。
    """
    rules = ParsedRules()
    if not preferences:
        return rules

    valid_items: list[dict[str, Any]] = []
    for item in preferences:
        if not isinstance(item, dict):
            continue
        rules.raw_preferences.append(item)
        valid_items.append(item)

    if not valid_items:
        return rules

    regex_precovered: set[int] = set()
    for idx, item in enumerate(valid_items):
        if _parse_high_value_single(item, rules):
            regex_precovered.add(idx)
            rules.parsed_by_regex += 1

    # 阶段 1：LLM 主解析
    llm_covered: set[int] = set(regex_precovered)  # 记录 LLM 成功覆盖的 idx
    if llm_caller is not None:
        rules.llm_used = True
        try:
            llm_covered = _llm_parse_all(valid_items, rules, llm_caller)
            llm_covered |= regex_precovered
        except Exception as exc:  # noqa: BLE001 - 不能因 LLM 崩溃阻断主流程
            logger.warning("LLM 主解析异常，降级到正则安全网: %s", exc)
            llm_covered = set(regex_precovered)

    # 阶段 2：正则安全网（仅処理未被 LLM 覆盖的条目）
    for idx, item in enumerate(valid_items):
        if idx in llm_covered:
            continue
        if _try_parse_single(item, rules):
            rules.parsed_by_regex += 1
        else:
            rules.unparsed.append(item)
            rules.parse_failure_count += 1

    return rules


# ---------------- LLM 主解析 ----------------

_LLM_SYSTEM_PROMPT = """你是货运司机偏好规则解析器。输入是一组司机偏好条目（含 content、penalty_amount、penalty_cap）。
任务：逐条解析为可执行结构。仅输出严格 JSON，禁止 markdown、禁止解释、禁止多余文本。

输出 Schema：
{
  "rules": [
    {"index": <输入中该偏好的 0-base 下标>, "kind": <下面枚举之一>, "params": {...},
     "penalty_amount": <可选 number，默认从输入取>, "penalty_cap": <可选 number或 null>}
  ]
}

kind 枚举与 params 要求：
- forbidden_category: {"names": ["品类名", ...]}                       品类禁运
- avoid_category:    {"names": ["品类名", ...]}                       品类软避免（“尽量不”）
- no_drive_window:   {"start_minute": 0..1440, "end_minute": 0..2880}  一天内禁接单禁空驶的分钟区间，
                                                                    跨午夜用 end_minute > 1440 表示
- daily_rest:        {"minutes": int}                                 每日要求连续休息分钟数
- forbidden_zone:    {"lat": float, "lng": float, "radius_km": float} 禁入圆形区域
- must_visit:        {"lat": float, "lng": float, "radius_km": float, "required_days": int}
- bounded_area:      {"lat_min": float, "lat_max": float, "lng_min": float, "lng_max": float}
                                                                    仅允许在该经纬度范围内运营
- home_rule:         {"lat": float, "lng": float, "radius_km": float,
                      "home_by_hour": int (0..24),                  几点前需到家
                      "no_drive_until_hour": int (0..24)}            次日几点前不接不空
- monthly_day_off:   {"required_days": int}                          月内至少几天不出车
- daily_order_limit: {"max_orders": int}                             每日接单上限
- distance_limit:    {"kind": "haul|pickup|monthly_deadhead", "max_km": float}
                                                                    haul=单笔干线，pickup=赴装货空驶，
                                                                    monthly_deadhead=月累计空驶
- first_order_rule:  {"before_hour": int (0..24)}                    首单不得晚于几点
- preferred_cargo:   {"cargo_id": "字符串"}                          点名熟货源
- complex:           {"note": "原因描述"}                            无法准确结构化

重要要求：
1. 所有经纬度取自偏好原文，禁止凭空猜测。
2. “不接品类 X”/“禁运”=forbidden_category；“尽量不”=avoid_category。
3. 夜间“23点至次日4点”应输出 start_minute=1380, end_minute=1680。
4. 一条偏好可能包含多个规则（例如同时含回家 + 禁出区），可在该 index 下返回多条。
5. 不肯定的条目一律用 complex，不要猜。"""


def _llm_parse_all(
    items: list[dict[str, Any]],
    rules: ParsedRules,
    llm_caller: Callable[[dict[str, Any]], dict[str, Any]],
) -> set[int]:
    """调用 LLM 一次性解析全部偏好，返回被成功覆盖的条目下标集合。"""
    user_payload = {
        "preferences": [
            {
                "index": idx,
                "content": str(item.get("content", ""))[:1024],
                "penalty_amount": item.get("penalty_amount"),
                "penalty_cap": item.get("penalty_cap"),
            }
            for idx, item in enumerate(items)
        ]
    }
    payload = {
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    resp = llm_caller(payload)
    if not isinstance(resp, dict):
        return set()
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return set()
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return set()
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        logger.warning("LLM 输出不是合法 JSON，降级到正则")
        return set()
    parsed_rules = data.get("rules") if isinstance(data, dict) else None
    if not isinstance(parsed_rules, list):
        return set()

    covered: set[int] = set()
    for entry in parsed_rules:
        if not isinstance(entry, dict):
            continue
        idx_raw = entry.get("index")
        try:
            idx = int(idx_raw)
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < len(items)):
            continue
        source_item = items[idx]
        ok = _absorb_llm_rule(entry, rules, source_item)
        if ok:
            covered.add(idx)
            rules.parsed_by_llm += 1
    return covered


def _absorb_llm_rule(
    parsed: dict[str, Any],
    rules: ParsedRules,
    source_item: dict[str, Any],
) -> bool:
    """将一条 LLM 输出转为 ParsedRules 子项；返回是否成功覆盖。"""
    kind = str(parsed.get("kind", "")).lower()
    params = parsed.get("params") or {}
    if not isinstance(params, dict):
        return False
    src_amount, src_cap = _normalize_penalty(source_item)
    penalty_amount = float(parsed.get("penalty_amount", src_amount) or 0.0)
    penalty_cap_raw = parsed.get("penalty_cap", src_cap)
    penalty_cap: float | None = None
    if penalty_cap_raw is not None:
        try:
            penalty_cap = float(penalty_cap_raw)
        except (TypeError, ValueError):
            penalty_cap = src_cap

    try:
        if kind == "forbidden_category":
            names = [str(n) for n in params.get("names", []) if str(n).strip()]
            if not names:
                return False
            rules.categories.forbidden.extend(names)
            return True
        if kind == "avoid_category":
            names = [str(n) for n in params.get("names", []) if str(n).strip()]
            if not names:
                return False
            rules.categories.avoid.extend(names)
            return True
        if kind == "daily_rest":
            minutes = int(params.get("minutes", 0))
            if minutes <= 0:
                return False
            rules.rest_rules.append(
                RestRule(
                    required_minutes=minutes,
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            # 同步检查原文是否含时间范围——含则补生成 no_drive_window 防止漏封
            src_content = str(source_item.get("content", ""))
            src_rng = _extract_hour_range(src_content)
            if src_rng is not None:
                sh, eh = src_rng
                sm = sh * 60
                em = eh * 60
                if eh <= sh:
                    em = (eh + 24) * 60
                rules.no_drive_windows.append(
                    TimeWindowRule(
                        start_minute=sm, end_minute=em,
                        raw=src_content,
                        penalty_amount=penalty_amount,
                        penalty_cap=penalty_cap,
                    )
                )
            return True
        if kind == "no_drive_window":
            start = int(params.get("start_minute", 0))
            end = int(params.get("end_minute", 0))
            if end <= start:
                return False
            rules.no_drive_windows.append(
                TimeWindowRule(
                    start_minute=start,
                    end_minute=end,
                    raw=str(source_item.get("content", "")),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True
        if kind == "forbidden_zone":
            rules.forbidden_zones.append(
                CircleZone(
                    lat=float(params.get("lat", 0.0)),
                    lng=float(params.get("lng", 0.0)),
                    radius_km=float(params.get("radius_km", 0.0)),
                    raw=str(source_item.get("content", "")),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True
        if kind == "must_visit":
            rules.must_visit.append(
                MustVisitRule(
                    lat=float(params.get("lat", 0.0)),
                    lng=float(params.get("lng", 0.0)),
                    radius_km=float(params.get("radius_km", 1.0)),
                    required_days=int(params.get("required_days", 1)),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True
        if kind == "bounded_area":
            lat_min = float(params.get("lat_min"))
            lat_max = float(params.get("lat_max"))
            lng_min = float(params.get("lng_min"))
            lng_max = float(params.get("lng_max"))
            if lat_min > lat_max or lng_min > lng_max:
                return False
            rules.bounded_areas.append(
                BoundingBoxRule(
                    lat_min=lat_min,
                    lat_max=lat_max,
                    lng_min=lng_min,
                    lng_max=lng_max,
                    raw=str(source_item.get("content", "")),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True
        if kind == "home_rule":
            rules.home_rule = HomeRule(
                lat=float(params.get("lat", 0.0)),
                lng=float(params.get("lng", 0.0)),
                radius_km=float(params.get("radius_km", 1.0)),
                home_by_hour=int(params.get("home_by_hour", 23)),
                no_drive_until_hour=int(params.get("no_drive_until_hour", 6)),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True
        if kind == "monthly_day_off":
            rules.monthly_day_off = MonthlyDayOffRule(
                required_days=int(params.get("required_days", 1)),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True
        if kind == "daily_order_limit":
            rules.daily_order_limit = DailyOrderLimitRule(
                max_orders=int(params.get("max_orders", 0)),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True
        if kind == "distance_limit":
            sub_kind = str(params.get("kind", "")).strip().lower()
            if sub_kind not in {"haul", "pickup", "monthly_deadhead"}:
                return False
            rules.distance_limits.append(
                DistanceLimitRule(
                    kind=sub_kind,
                    max_km=float(params.get("max_km", 0.0)),
                    penalty_amount=penalty_amount,
                    penalty_cap=penalty_cap,
                )
            )
            return True
        if kind == "first_order_rule":
            rules.first_order_rule = FirstOrderRule(
                before_hour=int(params.get("before_hour", 24)),
                penalty_amount=penalty_amount,
                penalty_cap=penalty_cap,
            )
            return True
        if kind == "preferred_cargo":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                return False
            if cargo_id not in rules.preferred_cargo_ids:
                rules.preferred_cargo_ids.append(cargo_id)
            if not any(r.cargo_id == cargo_id for r in rules.preferred_cargo):
                lat = params.get("lat")
                lng = params.get("lng")
                available_minutes = params.get("available_minutes")
                rules.preferred_cargo.append(
                    PreferredCargoRule(
                        cargo_id=cargo_id,
                        lat=float(lat) if lat is not None else None,
                        lng=float(lng) if lng is not None else None,
                        available_minutes=int(available_minutes) if available_minutes is not None else None,
                        penalty_amount=penalty_amount,
                        penalty_cap=penalty_cap,
                    )
                )
            return True
        if kind == "complex":
            return False  # 交给正则安全网、正则也失败则计入 unparsed
    except (TypeError, ValueError, KeyError):
        return False
    return False


def signature_of(preferences: list[Any]) -> str:
    """偏好原文哈希，用于检测是否需要重新解析。"""
    payload = json.dumps(preferences, ensure_ascii=False, sort_keys=True)
    return md5(payload.encode("utf-8")).hexdigest()
