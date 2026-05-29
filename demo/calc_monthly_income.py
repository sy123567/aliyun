"""计算 2026 年 3 月每个司机累计收益（与仿真结果 JSONL 对齐）。

偏好罚分与 config/drivers.json 中每条 preference 的 penalty 及本脚本内嵌评测口径一致。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT / "demo") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "demo"))

from simkit.simulation_actions import haversine_km

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
REPOSITION_SPEED_KM_PER_HOUR = 60.0

SHENZHEN_LAT_MIN, SHENZHEN_LAT_MAX = 22.42, 22.89
SHENZHEN_LNG_MIN, SHENZHEN_LNG_MAX = 113.74, 114.66

D001_FORBIDDEN_REGION = "惠州"
D001_SHENZHEN_DAYS = [3, 4]

D002_REQUIRED_REGION = "增城"
D002_REQUIRED_MIN_DAYS = 4
D002_ZENGCHENG_LAT, D002_ZENGCHENG_LNG = 23.15, 113.67
D002_SIHUI_LAT, D002_SIHUI_LNG = 23.32, 112.83
D002_STOCKTAKE_DAY = 11
D002_BANQUET_DAY = 30


def _parse_epoch_minutes(ts: str) -> int:
    return int((_SIMULATION_EPOCH.fromisoformat(ts.strip().replace(" ", "T")) - _SIMULATION_EPOCH).total_seconds() // 60)


@dataclass(frozen=True)
class PreferenceRuleSpec:
    content: str
    start_minutes: int
    end_minutes: int
    penalty_amount: float
    penalty_cap: float | None


@dataclass(frozen=True)
class RouteStop:
    day: int
    lat: float
    lng: float
    min_wait_minutes: int
    arrive_before_minute: int | None = None


def _resolve_config_json(server_config_dir: Path) -> Path:
    primary = server_config_dir / "config.json"
    if primary.is_file():
        return primary
    fallback = server_config_dir / "config.example.json"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"缺少 server 配置: {primary} 或 {fallback}")


def load_reposition_speed_km_per_hour(config_path: Path) -> float:
    value = json.loads(config_path.read_text(encoding="utf-8")).get("reposition_speed_km_per_hour")
    if not isinstance(value, (int, float)) or float(value) <= 0:
        raise ValueError(f"{config_path.name} 缺少有效 reposition_speed_km_per_hour")
    return float(value)


def load_drivers_path(config_path: Path, server_root: Path) -> Path:
    rel = json.loads(config_path.read_text(encoding="utf-8")).get("drivers_path")
    if not rel or not isinstance(rel, str):
        raise ValueError(f"{config_path.name} 缺少有效 drivers_path")
    path = Path(rel)
    return (path if path.is_absolute() else server_root / path).resolve()


def load_cargo_map(path: Path) -> dict[str, dict[str, Any]]:
    cargo_map: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cargo_id = str(item.get("cargo_id", "")).strip()
            if not cargo_id:
                continue
            start, end = item.get("start", {}), item.get("end", {})
            distance_km = haversine_km(float(start["lat"]), float(start["lng"]), float(end["lat"]), float(end["lng"]))
            load_start_minutes: int | None = None
            load_end_minutes: int | None = None
            load_window = item.get("load_time")
            if isinstance(load_window, list) and len(load_window) == 2:
                load_start_minutes = _parse_epoch_minutes(str(load_window[0]))
                load_end_minutes = _parse_epoch_minutes(str(load_window[1]))
                if load_end_minutes < load_start_minutes:
                    load_start_minutes = load_end_minutes = None
            cargo_map[cargo_id] = {
                "price": float(item.get("price", 0.0)) / 100.0,
                "distance_km": distance_km,
                "create_minutes": _parse_epoch_minutes(str(item["create_time"])),
                "remove_minutes": _parse_epoch_minutes(str(item["remove_time"])),
                "start_lat": float(start["lat"]),
                "start_lng": float(start["lng"]),
                "end_lat": float(end["lat"]),
                "end_lng": float(end["lng"]),
                "cost_time_minutes": int(item.get("cost_time_minutes", 0) or 0),
                "load_start_minutes": load_start_minutes,
                "load_end_minutes": load_end_minutes,
                "cargo_name": str(item.get("cargo_name", "") or "").strip(),
                "start_city": str(start.get("city", "") or "").strip(),
                "end_city": str(end.get("city", "") or "").strip(),
            }
    return cargo_map


def load_driver_cost_map(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["driver_id"]).strip(): float(item.get("cost_per_km", 0.0))
        for item in raw
        if str(item.get("driver_id", "")).strip()
    }


def load_driver_preference_rules(path: Path) -> dict[str, list[PreferenceRuleSpec]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[PreferenceRuleSpec]] = {}
    for item in raw:
        driver_id = str(item.get("driver_id", "")).strip()
        if not driver_id:
            continue
        rules: list[PreferenceRuleSpec] = []
        for entry in item.get("preferences") or []:
            if not isinstance(entry, dict):
                continue
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            cap_raw = entry.get("penalty_cap")
            rules.append(
                PreferenceRuleSpec(
                    content=content,
                    start_minutes=_parse_epoch_minutes(str(entry.get("start_time", "2026-03-01 00:00:00"))),
                    end_minutes=_parse_epoch_minutes(str(entry.get("end_time", "2026-03-31 23:59:59"))),
                    penalty_amount=float(entry.get("penalty_amount", 0.0) or 0.0),
                    penalty_cap=None if cap_raw is None else float(cap_raw),
                )
            )
        out[driver_id] = rules
    return out


def iter_result_files(results_dir: Path) -> list[Path]:
    latest_by_driver: dict[str, Path] = {}
    for path in sorted(results_dir.glob("actions_202603_*.jsonl")):
        parts = path.name.split("_")
        if len(parts) < 4:
            continue
        driver_id = parts[2]
        prev = latest_by_driver.get(driver_id)
        if prev is None or path.name > prev.name:
            latest_by_driver[driver_id] = path
    return sorted(latest_by_driver.values(), key=lambda p: p.name)


def load_simulation_duration_days(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8")).get("simulation_duration_days")
    if not isinstance(value, int) or value <= 0:
        raise ValueError("run_summary_202603.json 缺少有效 simulation_duration_days")
    return min(value, 31)


def load_simulate_time_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("simulate_time_seconds")
    return round(float(value), 2) if isinstance(value, (int, float)) else None


def _nearly_equal(a: float, b: float, eps: float = 1e-4) -> bool:
    return abs(float(a) - float(b)) <= eps


def _distance_minutes(distance_km: float, speed_km_per_hour: float) -> int:
    if distance_km <= 0:
        return 1
    return max(1, int(math.ceil((distance_km / speed_km_per_hour) * 60.0)))


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged: list[tuple[int, int]] = []
    for s, e in intervals:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def _longest_merged_span_minutes(intervals: list[tuple[int, int]]) -> int:
    return max((e - s for s, e in _merge_intervals(intervals)), default=0)


def _in_shenzhen(lat: float, lng: float) -> bool:
    return SHENZHEN_LAT_MIN <= lat <= SHENZHEN_LAT_MAX and SHENZHEN_LNG_MIN <= lng <= SHENZHEN_LNG_MAX


def _near(lat: float, lng: float, target_lat: float, target_lng: float, radius_km: float) -> bool:
    return haversine_km(lat, lng, target_lat, target_lng) <= radius_km


def _ctx_overlaps_rule(ctx: dict[str, Any], rule: PreferenceRuleSpec) -> bool:
    return _interval_overlap(ctx["action_start"], ctx["action_end"], rule.start_minutes, rule.end_minutes + 1)


def _penalty_count(count: float, rule: PreferenceRuleSpec) -> float:
    amount = float(count) * rule.penalty_amount
    return amount if rule.penalty_cap is None else min(amount, rule.penalty_cap)


def _penalty_failed(failed: bool, rule: PreferenceRuleSpec) -> float:
    if not failed:
        return 0.0
    cap = rule.penalty_cap if rule.penalty_cap is not None else rule.penalty_amount
    return min(rule.penalty_amount, cap)


def _append_rule(detail_rules: list[dict[str, Any]], rule_label: str, penalty: float, rule: PreferenceRuleSpec, **extra: Any) -> None:
    detail_rules.append({"rule": rule_label, "penalty": round(penalty, 2), "preference_text": rule.content, **extra})


def _cargo_touches_region(cargo: dict[str, Any], region: str) -> bool:
    return region in str(cargo.get("start_city", "")) or region in str(cargo.get("end_city", ""))


def _wait_intervals_for_day(ctxs: list[dict[str, Any]], day: int) -> list[tuple[int, int]]:
    d0, d1 = day * 1440, (day + 1) * 1440
    intervals: list[tuple[int, int]] = []
    for c in ctxs:
        if c["action_name"] != "wait" or c["action_exec_cost"] <= 0:
            continue
        s, e = max(c["step_start"], d0), min(c["step_end"], d1)
        if e > s:
            intervals.append((s, e))
    return intervals


def _active_minutes_by_day(ctxs: list[dict[str, Any]], days: list[int]) -> dict[int, int]:
    active = {d: 0 for d in days}
    for c in ctxs:
        if c["action_name"] not in {"take_order", "reposition"}:
            continue
        cur = c["action_start"]
        while cur < c["action_end"]:
            day_idx = cur // 1440
            day_end = (day_idx + 1) * 1440
            if day_idx in active:
                active[day_idx] += min(day_end, c["action_end"]) - cur
            cur = day_end
    return active


def _build_step_contexts(file_path: Path) -> list[dict[str, Any]]:
    ctxs: list[dict[str, Any]] = []
    prev_end = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record = json.loads(row)
            step_elapsed = int(record["step_elapsed_minutes"])
            query_scan = int(record["query_scan_cost_minutes"])
            action_exec = int(record["action_exec_cost_minutes"])
            result = record.get("result", {})
            end_minutes = int(result["simulation_progress_minutes"])
            action_obj = record.get("action") or {}
            pos_before = record.get("position_before") or {}
            pos_after = record.get("position_after") or {}
            ctxs.append(
                {
                    "line_no": line_no,
                    "action_name": str(action_obj.get("action", "")).strip().lower(),
                    "params": action_obj.get("params") or {},
                    "result": result if isinstance(result, dict) else {},
                    "step_start": prev_end,
                    "action_start": prev_end + query_scan,
                    "action_end": prev_end + query_scan + action_exec,
                    "step_end": end_minutes,
                    "action_exec_cost": action_exec,
                    "before_lat": float(pos_before.get("lat", 0.0)),
                    "before_lng": float(pos_before.get("lng", 0.0)),
                    "after_lat": float(pos_after.get("lat", 0.0)),
                    "after_lng": float(pos_after.get("lng", 0.0)),
                }
            )
            prev_end = end_minutes
    return ctxs


def _first_arrival_minute(
    ctxs: list[dict[str, Any]],
    lat: float,
    lng: float,
    radius_km: float,
    window_start: int,
    window_end: int,
    after_min: int = 0,
) -> int | None:
    for ctx in ctxs:
        if ctx["step_end"] <= after_min or ctx["step_end"] < window_start or ctx["step_end"] > window_end:
            continue
        if _near(ctx["after_lat"], ctx["after_lng"], lat, lng, radius_km):
            return ctx["step_end"]
    return None


def _wait_minutes_near(
    ctxs: list[dict[str, Any]],
    lat: float,
    lng: float,
    radius_km: float,
    window_start: int,
    window_end: int,
    after_min: int = 0,
) -> int:
    run = 0
    best = 0
    for ctx in ctxs:
        if ctx["step_end"] <= after_min or ctx["step_start"] >= window_end:
            continue
        if ctx["action_name"] == "wait" and _near(ctx["after_lat"], ctx["after_lng"], lat, lng, radius_km):
            run += ctx["action_exec_cost"]
            best = max(best, run)
        elif not (
            ctx["action_name"] == "wait"
            and _near(ctx["before_lat"], ctx["before_lng"], lat, lng, radius_km)
            and _near(ctx["after_lat"], ctx["after_lng"], lat, lng, radius_km)
        ):
            run = 0
    return best


def _eval_daily_rest(ctxs, days, min_hours, rule, label, detail_rules):
    violations = sum(1 for day in days if _longest_merged_span_minutes(_wait_intervals_for_day(ctxs, day)) < min_hours * 60)
    penalty = _penalty_count(violations, rule)
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_scheduled_rest_window(ctxs, days, start_hour, start_minute, end_hour, end_minute, rule, label, detail_rules):
    window_minutes = (end_hour * 60 + end_minute) - (start_hour * 60 + start_minute)
    if window_minutes <= 0:
        window_minutes += 24 * 60
    violations = 0
    for day in days:
        window_start = day * 1440 + start_hour * 60 + start_minute
        window_end = day * 1440 + end_hour * 60 + end_minute
        if window_end <= window_start:
            window_end += 1440
        intervals: list[tuple[int, int]] = []
        for ctx in ctxs:
            if ctx["action_name"] != "wait" or ctx["action_exec_cost"] <= 0:
                continue
            s, e = max(ctx["step_start"], window_start), min(ctx["step_end"], window_end)
            if e > s:
                intervals.append((s, e))
        if sum(e - s for s, e in _merge_intervals(intervals)) < window_minutes:
            violations += 1
    penalty = _penalty_count(violations, rule)
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_forbidden_categories(ctxs, cargo_map, categories, rule, label, detail_rules):
    penalty = violations = 0
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        cargo_id = str((ctx["params"] or {}).get("cargo_id", "")).strip()
        if str(cargo_map.get(cargo_id, {}).get("cargo_name", "")) in categories:
            violations += 1
            penalty += rule.penalty_amount
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_forbidden_region_cargo(ctxs, cargo_map, region, rule, label, detail_rules):
    violations = 0
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
        if cargo is not None and _cargo_touches_region(cargo, region):
            violations += 1
    penalty = _penalty_count(violations, rule)
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_pickup_deadhead(ctxs, cargo_map, max_km, rule, label, detail_rules):
    violations = 0
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
        if cargo is None:
            continue
        pickup_km = haversine_km(ctx["before_lat"], ctx["before_lng"], float(cargo["start_lat"]), float(cargo["start_lng"]))
        if pickup_km > max_km:
            violations += 1
    penalty = _penalty_count(violations, rule)
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_off_days(ctxs, days, min_off_days, rule, label, detail_rules):
    active = _active_minutes_by_day(ctxs, days)
    off_days = sum(1 for day in days if active.get(day, 0) == 0)
    penalty = _penalty_failed(off_days < min_off_days, rule)
    _append_rule(detail_rules, label, penalty, rule, off_days=off_days)
    return penalty


def _eval_required_region_cargo_days(ctxs, cargo_map, region, min_days, rule, label, detail_rules):
    order_days: set[int] = set()
    for ctx in ctxs:
        if ctx["action_name"] != "take_order" or not bool(ctx["result"].get("accepted", False)):
            continue
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
        if cargo is not None and _cargo_touches_region(cargo, region):
            order_days.add(ctx["action_start"] // 1440)
    penalty = _penalty_failed(len(order_days) < min_days, rule)
    _append_rule(detail_rules, label, penalty, rule, order_days=len(order_days))
    return penalty


def _eval_d001_shenzhen_march(
    ctxs: list[dict[str, Any]],
    cargo_map: dict[str, dict[str, Any]],
    days: list[int],
    rule: PreferenceRuleSpec,
    label: str,
    detail_rules: list[dict[str, Any]],
) -> float:
    violations = 0
    day_set = set(days)
    for ctx in ctxs:
        if not _ctx_overlaps_rule(ctx, rule):
            continue
        on_days = any(_interval_overlap(ctx["action_start"], ctx["action_end"], d * 1440, (d + 1) * 1440) for d in day_set)
        if not on_days:
            continue
        if ctx["action_name"] == "take_order" and bool(ctx["result"].get("accepted", False)):
            cargo = cargo_map.get(str((ctx["params"] or {}).get("cargo_id", "")).strip())
            if cargo is not None and _cargo_touches_region(cargo, "深圳"):
                violations += 1
        elif ctx["action_name"] in {"take_order", "reposition"}:
            if _in_shenzhen(ctx["before_lat"], ctx["before_lng"]) or _in_shenzhen(ctx["after_lat"], ctx["after_lng"]):
                violations += 1
    penalty = _penalty_count(violations, rule)
    _append_rule(detail_rules, label, penalty, rule, violations=violations)
    return penalty


def _eval_wait_at_location_on_day(ctxs, day, lat, lng, radius_km, min_wait_minutes, rule, label, detail_rules, arrive_before_minute=None):
    day_start = day * 1440
    day_end = day_start + 1440
    deadline = day_start + (arrive_before_minute if arrive_before_minute is not None else 1440)
    arrival = _first_arrival_minute(ctxs, lat, lng, radius_km, day_start, deadline)
    if arrival is None:
        waited = 0
        failed = True
    else:
        waited = _wait_minutes_near(ctxs, lat, lng, radius_km, arrival, day_end, after_min=arrival)
        failed = waited < min_wait_minutes
    penalty = _penalty_failed(failed, rule)
    _append_rule(detail_rules, label, penalty, rule, satisfied=not failed, waited_minutes=waited)
    return penalty


def _eval_route_stops(ctxs, stops, rule, label, detail_rules, radius_km=2.0):
    failed = False
    after_min = 0
    for stop in stops:
        day_start = stop.day * 1440
        deadline = day_start + (stop.arrive_before_minute if stop.arrive_before_minute is not None else 1440)
        arrival = _first_arrival_minute(ctxs, stop.lat, stop.lng, radius_km, day_start, deadline, after_min)
        if arrival is None:
            failed = True
            break
        if stop.min_wait_minutes > 0:
            waited = _wait_minutes_near(
                ctxs, stop.lat, stop.lng, radius_km, arrival, day_start + 1440, after_min=arrival
            )
            if waited < stop.min_wait_minutes:
                failed = True
                break
            after_min = max(after_min, arrival + stop.min_wait_minutes)
        else:
            after_min = max(after_min, arrival)
    penalty = _penalty_failed(failed, rule)
    _append_rule(detail_rules, label, penalty, rule, satisfied=not failed)
    return penalty


class DriverPreferenceCalculatorBase(ABC):
    driver_id: str
    expected_rules: int

    @abstractmethod
    def compute(self, ctxs, cargo_map, rules, simulation_duration_days):
        raise NotImplementedError


class DriverD001PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D001"
    expected_rules = 5

    def compute(self, ctxs, cargo_map, rules, simulation_duration_days):
        if len(rules) != self.expected_rules:
            raise ValueError(f"D001 需要 {self.expected_rules} 条偏好，当前 {len(rules)}")
        detail: list[dict[str, Any]] = []
        total = 0.0
        all_days = list(range(simulation_duration_days))
        r0, r1, r2, r3, r4 = rules
        total += _eval_daily_rest(ctxs, all_days, 8, r0, "每日连续休息≥8小时", detail)
        total += _eval_forbidden_categories(ctxs, cargo_map, {"机械设备"}, r1, "禁接机械设备", detail)
        total += _eval_forbidden_region_cargo(ctxs, cargo_map, D001_FORBIDDEN_REGION, r2, "不接惠州货源", detail)
        total += _eval_off_days(ctxs, all_days, 3, r3, "每月至少3整天休息", detail)
        total += _eval_d001_shenzhen_march(ctxs, cargo_map, D001_SHENZHEN_DAYS, r4, "三四号不进深圳", detail)
        return round(total, 2), {"rules": detail}


class DriverD002PreferenceCalculator(DriverPreferenceCalculatorBase):
    driver_id = "D002"
    expected_rules = 7

    def compute(self, ctxs, cargo_map, rules, simulation_duration_days):
        if len(rules) != self.expected_rules:
            raise ValueError(f"D002 需要 {self.expected_rules} 条偏好，当前 {len(rules)}")
        detail: list[dict[str, Any]] = []
        total = 0.0
        all_days = list(range(simulation_duration_days))
        r0, r1, r2, r3, r4, r5, r6 = rules
        total += _eval_scheduled_rest_window(ctxs, all_days, 0, 0, 6, 0, r0, "每日0–6点休息", detail)
        total += _eval_forbidden_categories(ctxs, cargo_map, {"蔬菜"}, r1, "禁接蔬菜", detail)
        total += _eval_required_region_cargo_days(
            ctxs, cargo_map, D002_REQUIRED_REGION, D002_REQUIRED_MIN_DAYS, r2, "月度增城≥4日", detail
        )
        total += _eval_pickup_deadhead(ctxs, cargo_map, 55.0, r3, "赴装货空驶≤55km", detail)
        total += _eval_off_days(ctxs, all_days, 2, r4, "每月至少2整天休息", detail)
        total += _eval_wait_at_location_on_day(
            ctxs, D002_STOCKTAKE_DAY, D002_ZENGCHENG_LAT, D002_ZENGCHENG_LNG, 2.0, 120, r5, "三月十二号增城盘库", detail
        )
        total += _eval_route_stops(
            ctxs,
            [
                RouteStop(D002_BANQUET_DAY, D002_ZENGCHENG_LAT, D002_ZENGCHENG_LNG, 0, 12 * 60),
                RouteStop(D002_BANQUET_DAY, D002_SIHUI_LAT, D002_SIHUI_LNG, 120, 12 * 60),
            ],
            r6,
            "三月三十一号舅公寿宴",
            detail,
        )
        return round(total, 2), {"rules": detail}


_PREFERENCE_CALCULATORS = {
    "D001": DriverD001PreferenceCalculator(),
    "D002": DriverD002PreferenceCalculator(),
}


def _evaluate_preferences(driver_id, file_path, rules, cargo_map, simulation_duration_days):
    calc = _PREFERENCE_CALCULATORS.get(driver_id)
    if calc is None:
        return 0.0, {"rules": []}
    ctxs = _build_step_contexts(file_path)
    if not ctxs:
        return 0.0, {"rules": []}
    return calc.compute(ctxs, cargo_map, rules, simulation_duration_days)


def _validate_and_compute_income_by_driver(
    file_path: Path,
    cargo_map: dict[str, dict[str, Any]],
    cost_per_km: float,
    reposition_speed_km_per_hour: float,
    simulation_horizon_minutes: int | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    prev_end_minutes = 0
    with file_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = line.strip()
            if not row:
                continue
            record = json.loads(row)
            action_name = str(record.get("action", {}).get("action", "")).strip().lower()
            if action_name not in {"wait", "reposition", "take_order"}:
                raise ValueError(f"{file_path.name} 第 {line_no} 行 action 非法: {action_name}")
            result = record.get("result", {})
            params = record.get("action", {}).get("params", {})
            raw_usage = record.get("token_usage", {})
            if isinstance(raw_usage, dict):
                for key in token_usage:
                    token_usage[key] += int(raw_usage.get(key, 0))
            end_minutes = int(result["simulation_progress_minutes"])
            step_elapsed = int(record["step_elapsed_minutes"])
            query_scan = int(record["query_scan_cost_minutes"])
            action_exec = int(record["action_exec_cost_minutes"])
            if end_minutes - prev_end_minutes != step_elapsed:
                raise ValueError(f"{file_path.name} 第 {line_no} 行时间推进不一致")
            action_start = prev_end_minutes + query_scan
            before = record.get("position_before", {})
            after = record.get("position_after", {})
            before_lat, before_lng = float(before["lat"]), float(before["lng"])
            after_lat, after_lng = float(after["lat"]), float(after["lng"])
            if action_name == "wait":
                if action_exec != int((params or {}).get("duration_minutes", 1)):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 时间不一致")
                if not _nearly_equal(before_lat, after_lat) or not _nearly_equal(before_lng, after_lng):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 wait 不应改变位置")
            elif action_name == "reposition":
                target_lat, target_lng = float(params["latitude"]), float(params["longitude"])
                expected_km = haversine_km(before_lat, before_lng, target_lat, target_lng)
                if action_exec != _distance_minutes(expected_km, reposition_speed_km_per_hour):
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 reposition 时间不一致")
                income["distance_km"] += float(result.get("distance_km", 0.0))
            elif action_name == "take_order":
                cargo_id = str((params or {}).get("cargo_id", "")).strip()
                cargo = cargo_map.get(cargo_id)
                if cargo is None:
                    raise ValueError(f"{file_path.name} 第 {line_no} 行 cargo_id 不存在")
                if bool(result.get("accepted", False)):
                    if not (int(cargo["create_minutes"]) <= action_start <= int(cargo["remove_minutes"])):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单时点不在货源有效期")
                    pickup_km = haversine_km(before_lat, before_lng, float(cargo["start_lat"]), float(cargo["start_lng"]))
                    pickup_minutes = _distance_minutes(pickup_km, reposition_speed_km_per_hour) if pickup_km > 1e-6 else 0
                    arrival = action_start + pickup_minutes
                    wait_minutes = 0
                    if isinstance(cargo.get("load_start_minutes"), int) and isinstance(cargo.get("load_end_minutes"), int):
                        if arrival > int(cargo["load_end_minutes"]):
                            raise ValueError(f"{file_path.name} 第 {line_no} 行成功接单但已超装货时间窗")
                        wait_minutes = max(0, int(cargo["load_start_minutes"]) - arrival)
                    if action_exec != pickup_minutes + wait_minutes + int(cargo["cost_time_minutes"]):
                        raise ValueError(f"{file_path.name} 第 {line_no} 行接单耗时不一致")
                    if simulation_horizon_minutes is None or end_minutes <= simulation_horizon_minutes:
                        income["gross_income"] += float(cargo["price"])
                    income["distance_km"] += float(result.get("pickup_deadhead_km", 0.0) or 0.0) + float(
                        result.get("haul_distance_km", 0.0) or cargo["distance_km"]
                    )
            prev_end_minutes = end_minutes
    income["cost"] = income["distance_km"] * cost_per_km
    income["net_income"] = income["gross_income"] - income["cost"]
    for key in ("gross_income", "distance_km", "cost", "net_income"):
        income[key] = round(float(income[key]), 2)
    return income, token_usage


def compute_income(files, cargo_map, driver_cost_map, driver_preference_rules, reposition_speed_km_per_hour, simulation_duration_days):
    stats, token_stats, validation_errors, preference_details = {}, {}, {}, {}
    zero_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "net_income": 0.0}
    zero_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    horizon = simulation_duration_days * 1440
    for file_path in files:
        driver_id = file_path.name.split("_")[2]
        try:
            income_item, token_item = _validate_and_compute_income_by_driver(
                file_path, cargo_map, float(driver_cost_map.get(driver_id, 0.0)), reposition_speed_km_per_hour, horizon
            )
            penalty, pref_detail = _evaluate_preferences(
                driver_id, file_path, driver_preference_rules.get(driver_id, []), cargo_map, simulation_duration_days
            )
            income_item["preference_penalty"] = round(penalty, 2)
            income_item["net_income"] = round(income_item["net_income"] - penalty, 2)
            preference_details[driver_id] = pref_detail
            stats[driver_id] = income_item
            token_stats[driver_id] = token_item
        except Exception as exc:
            stats[driver_id] = dict(zero_income)
            token_stats[driver_id] = dict(zero_tokens)
            validation_errors[driver_id] = f"{type(exc).__name__}: {exc}"
            preference_details[driver_id] = {"rules": []}
    total_token_usage = {k: sum(int(t[k]) for t in token_stats.values()) for k in zero_tokens}
    return stats, token_stats, total_token_usage, validation_errors, preference_details


def build_drivers_payload(income, token_by_driver, validation_errors, preference_details):
    default_income = {"gross_income": 0.0, "distance_km": 0.0, "cost": 0.0, "preference_penalty": 0.0, "net_income": 0.0}
    default_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
    rows = []
    for driver_id in sorted(set(income) | set(token_by_driver)):
        rows.append(
            {
                "driver_id": driver_id,
                "income": {**default_income, **income.get(driver_id, {})},
                "token_usage": {**default_tokens, **token_by_driver.get(driver_id, {})},
                "calculation_aborted": driver_id in validation_errors,
                "validation_error": validation_errors.get(driver_id),
                "preference_check": preference_details.get(driver_id, {"rules": []}),
            }
        )
    return rows


def main(
    *,
    results_dir: Path,
    data_dir: Path | None = None,
    project_root: Path | None = None,
    reposition_speed_km_per_hour: float | None = None,
) -> None:
    layout_root = (project_root or _SCRIPT_DIR).resolve()
    results_dir = results_dir.resolve()
    server_root = layout_root / "server"
    if data_dir is not None:
        cargo_dataset = data_dir / "cargo_dataset.jsonl"
        drivers_dataset = data_dir / "drivers.json"
        speed = float(reposition_speed_km_per_hour or REPOSITION_SPEED_KM_PER_HOUR)
    else:
        config_path = _resolve_config_json(server_root / "config")
        cargo_dataset = server_root / "data" / "cargo_dataset.jsonl"
        drivers_dataset = load_drivers_path(config_path, server_root)
        speed = float(reposition_speed_km_per_hour or load_reposition_speed_km_per_hour(config_path))
    output_file = results_dir / "monthly_income_202603.json"
    run_summary_file = results_dir / "run_summary_202603.json"
    if not cargo_dataset.is_file():
        raise FileNotFoundError(f"缺少货源数据: {cargo_dataset}")
    if not drivers_dataset.is_file():
        raise FileNotFoundError(f"缺少司机数据: {drivers_dataset}")

    cargo_map = load_cargo_map(cargo_dataset)
    driver_cost_map = load_driver_cost_map(drivers_dataset)
    driver_preference_rules = load_driver_preference_rules(drivers_dataset)
    simulation_duration_days = load_simulation_duration_days(run_summary_file)
    result_files = iter_result_files(results_dir)
    income, token_by_driver, total_token_usage, validation_errors, preference_details = compute_income(
        result_files,
        cargo_map,
        driver_cost_map,
        driver_preference_rules,
        reposition_speed_km_per_hour=speed,
        simulation_duration_days=simulation_duration_days,
    )
    drivers = build_drivers_payload(income, token_by_driver, validation_errors, preference_details)
    payload = {
        "month": "2026-03",
        "simulate_time_seconds": load_simulate_time_seconds(run_summary_file),
        "result_files_count": len(result_files),
        "drivers": drivers,
        "summary": {
            "total_net_income_all_drivers": round(sum(float(d["income"]["net_income"]) for d in drivers), 2),
            "total_preference_penalty": round(sum(float(d["income"].get("preference_penalty", 0.0)) for d in drivers), 2),
            "total_token_usage": total_token_usage,
            "failed_driver_count": len(validation_errors),
            "failed_drivers": validation_errors,
        },
        "cost_meaning": "cost = distance_km * cost_per_km",
        "cost_metric": "net_income = gross_income - cost - preference_penalty",
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 2026 年 3 月司机累计收益")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--reposition-speed", type=float, default=None)
    args = parser.parse_args()
    layout_root = (args.project_root or _SCRIPT_DIR).resolve()
    results_dir = (args.results_dir or layout_root / "results").resolve()
    main(
        results_dir=results_dir,
        data_dir=args.data_dir,
        project_root=layout_root,
        reposition_speed_km_per_hour=args.reposition_speed,
    )
