"""地理与时间换算工具：与 simkit 中的 Haversine + ceil 取整规则保持一致。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

EARTH_RADIUS_KM = 6371.0
SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
MONTH_HORIZON_MINUTES = 31 * 24 * 60
WALL_TIME_FMT = "%Y-%m-%d %H:%M:%S"


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """与 simkit.simulation_actions.haversine_km 等价的纯 Python 实现。"""
    p1 = math.radians(lat1)
    l1 = math.radians(lng1)
    p2 = math.radians(lat2)
    l2 = math.radians(lng2)
    dp = p2 - p1
    dl = l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * (math.sin(dl * 0.5) ** 2)
    h = min(1.0, max(0.0, h))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def distance_to_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    """与 simkit 一致的 ceil 取整规则；零距离返回 1 分钟，与 reposition 口径相同。"""
    if distance_km <= 0:
        return 1
    if speed_km_per_hour <= 0:
        return 1
    return max(1, math.ceil((distance_km / speed_km_per_hour) * 60))


def minutes_to_wall_time(simulation_progress_minutes: int) -> datetime:
    """仿真分钟偏移 → 墙钟 datetime。"""
    return SIMULATION_EPOCH + timedelta(minutes=int(simulation_progress_minutes))


def wall_time_to_minutes(wall_time_str: str) -> int:
    """%Y-%m-%d %H:%M:%S 格式墙钟 → 仿真分钟偏移。"""
    dt = datetime.strptime(wall_time_str.strip(), WALL_TIME_FMT)
    delta = dt - SIMULATION_EPOCH
    return int(delta.total_seconds() // 60)


def minute_of_day(simulation_progress_minutes: int) -> int:
    """当天中第几分钟，范围 [0, 1440)。"""
    return int(simulation_progress_minutes) % (24 * 60)


def hour_of_day(simulation_progress_minutes: int) -> int:
    """当天第几小时，范围 [0, 24)。"""
    return minute_of_day(simulation_progress_minutes) // 60


def date_str(simulation_progress_minutes: int) -> str:
    """当天日期字符串 YYYY-MM-DD，便于按日聚合。"""
    return minutes_to_wall_time(simulation_progress_minutes).strftime("%Y-%m-%d")


def is_in_window(minute_of_day_value: int, start_minute: int, end_minute: int) -> bool:
    """判断当天分钟值是否落在 [start, end] 区间内（支持跨午夜）。"""
    if start_minute <= end_minute:
        return start_minute <= minute_of_day_value < end_minute
    return minute_of_day_value >= start_minute or minute_of_day_value < end_minute


def grid_key(latitude: float, longitude: float, cell_size_deg: float = 0.1) -> tuple[int, int]:
    """经纬度网格化：默认 0.1 度 ≈ 10 公里粒度，用于热点统计。"""
    return (
        int(math.floor(latitude / cell_size_deg)),
        int(math.floor(longitude / cell_size_deg)),
    )


def grid_center(key: tuple[int, int], cell_size_deg: float = 0.1) -> tuple[float, float]:
    """网格 key 还原为单元中心点经纬度。"""
    lat = (key[0] + 0.5) * cell_size_deg
    lng = (key[1] + 0.5) * cell_size_deg
    return (lat, lng)
