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
MONTH_DAYS = 31
SPEED_KM_PER_HOUR = 60.0
EARTH_RADIUS_KM = 6371.0
COST_PER_KM = 1.5
QUERY_SCAN_BATCH_SIZE = 10
# Minimum remaining minutes in the day worth attempting an anti-stranding reposition:
# below this there is no time to relocate and still complete an order before day end.
_STRAND_MIN_BUDGET = 240

SHENZHEN_BBOX = (22.42, 22.89, 113.74, 114.66)  # lat_min, lat_max, lng_min, lng_max


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


def _query_scan_minutes(k: int) -> int:
    return math.ceil(max(1, k) / QUERY_SCAN_BATCH_SIZE)


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


class DriverRules:
    """结构化偏好规则。"""

    def __init__(self) -> None:
        self.daily_rest_minutes: int = 0
        self.rest_window: tuple[int, int] | None = None  # (start_min, end_min) within day, from 0
        self.off_days_min: int = 0
        self.forbidden_categories: set[str] = set()
        self.forbidden_category_penalty: dict[str, float] = {}
        self.avoid_categories: set[str] = set()  # soft avoid (still filter)
        self.forbidden_regions: set[str] = set()
        self.forbidden_region_penalty: dict[str, float] = {}
        self.required_region: tuple[str, int] | None = None  # (region, min_days)
        self.pickup_max_km: float | None = None
        self.pickup_max_penalty: float = 180.0
        self.blackout: list[tuple[str, set[int]]] = []  # (region, days)
        self.blackout_penalty: dict[str, float] = {}
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
        self.haul_max_penalty: float = 1200.0
        self.monthly_deadhead_max_km: float | None = None
        self.forbidden_zones: list[tuple[float, float, float]] = []  # (lat, lng, radius_km)
        self.forbidden_zone_penalty: float = 2500.0
        self.bounded_area: tuple[float, float, float, float] | None = None  # (lat_min, lat_max, lng_min, lng_max)
        self.bounded_area_penalty: float = 2500.0
        self.must_visit: list[dict[str, Any]] = []  # {lat, lng, radius_km, required_days}
        self.first_order_before_minute: int | None = None

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


class ModelDecisionService:
    """单步决策：合规优先 + 净收益择单的确定性调度器。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")
        self._rules: dict[str, DriverRules] = {}
        self._plan: dict[str, dict[str, Any]] = {}
        # preference texts already fed to the parser (LLM is only re-invoked when a
        # new, date-windowed preference becomes visible).
        self._seen_prefs: dict[str, set[str]] = {}

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
                "must_visit_days": {},  # idx → set of days visited
                "first_order_taken": set(),  # days where first order was already taken
                "home_done": set(),  # days where home repositioning is done
            },
        )
        # Preferences may only become visible inside their date window, so the off-day
        # set is recomputed each step from the rules known so far.
        plan["off_days"] = self._plan_off_days(rules)
        now = int(status["simulation_progress_minutes"])
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        action = self._schedule(driver_id, status, rules, plan, now, lat, lng)
        self._logger.info(
            "decision driver_id=%s now=%s day=%s tod=%s action=%s params=%s",
            driver_id,
            now,
            now // DAY_MINUTES,
            now % DAY_MINUTES,
            action.get("action"),
            action.get("params"),
        )
        return action

    # --------------------------------------------------------------- scheduler
    def _schedule(self, driver_id, status, rules, plan, now, lat, lng) -> dict[str, Any]:
        day, tod = divmod(now, DAY_MINUTES)
        if day >= MONTH_DAYS:
            return self._wait(1)
        day_start = day * DAY_MINUTES
        day_end = day_start + DAY_MINUTES

        # (A) full off day: idle/rest the whole day.
        if day in plan["off_days"]:
            return self._wait(day_end - now)

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
                return self._wait(day_end - now)

        # (B) start-of-day rest (covers daily-rest & rest-window from 00:00).
        # For a flexible "N continuous hours anywhere" rest (rest_window is None) the
        # block may begin after midnight when the previous day's last order ran past
        # midnight: we then rest a full block from `now` (e.g. 02:00–10:00), which still
        # gives the day its 8h continuous span (the scorer clips spans to the day).
        block = rules.day_rest_block
        if block > 0 and day not in plan["rest_done"]:
            plan["rest_done"].add(day)
            if rules.rest_window is None:
                # Flexible rest: rest from midnight (low-cargo period, 0-6am typical)
                # but cap to block duration
                dur = block if tod + block <= DAY_MINUTES else max(0, DAY_MINUTES - tod)
            else:
                dur = min(block - tod, day_end - now) if tod < block else 0
            if dur > 0:
                return self._wait(dur)
        else:
            plan["rest_done"].add(day)

        # (B2) daytime rest_window enforcement: if the driver has a daytime
        # rest window (e.g. 12:00-14:00) and the current time-of-day falls
        # inside it, idle until the window ends.
        if rules.rest_window is not None:
            rw_s, rw_e = rules.rest_window
            if rw_e > rw_s and rw_s > 0:
                if rw_s <= tod < rw_e:
                    return self._wait(rw_e - tod)

        # (C) dated single-stop events (e.g. 盘库).
        for ev in rules.dated_single:
            if ev["day"] != day or ev["day"] in plan["dated_single_done"]:
                continue
            before = day_start + ev["before"]
            if _haversine_km(lat, lng, ev["lat"], ev["lng"]) > 1.5:
                dist = _haversine_km(lat, lng, ev["lat"], ev["lng"])
                if now + _travel_minutes(dist) <= before:
                    repo = self._planned_reposition(rules, plan, lat, lng, ev["lat"], ev["lng"])
                    if repo is not None:
                        return repo
                continue  # can't make it; skip silently
            plan["dated_single_done"].add(ev["day"])
            return self._wait(max(ev["min_wait"], 1))

        # (D) dated multi-stop route (e.g. 寿宴).
        for ev in rules.dated_route:
            if ev["day"] != day or ev["day"] in plan["dated_route_done"]:
                continue
            act = self._drive_route(ev, plan, now, day_start, lat, lng)
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
                    repo = self._planned_reposition(rules, plan, lat, lng, first["lat"], first["lng"])
                    if repo is not None:
                        return repo
            else:
                return self._wait(day_end - now)

        # (D3) no_drive_windows: if current time-of-day falls inside a no-drive
        # window, idle until the window ends.  Skip windows that are fully covered
        # by rest_window to avoid double-idle.
        for ws, we in rules.no_drive_windows:
            if rules.rest_window is not None:
                rws, rwe = rules.rest_window
                if rws <= ws and rwe >= min(we, DAY_MINUTES):
                    continue  # already covered by rest_window
            we_today = we if we <= DAY_MINUTES else DAY_MINUTES
            if ws <= tod < we_today:
                return self._wait(we_today - tod)

        # (D4) must_visit: proactively go to must-visit locations if not enough
        # visits have been accumulated.  Only navigate when urgency is very high
        # (remaining days == still_needed) AND coordinates are explicitly from text
        # (to avoid LLM-hallucinated must_visit wasting entire days).
        for i, mv in enumerate(rules.must_visit):
            visited = plan["must_visit_days"].setdefault(i, set())
            remaining_days = MONTH_DAYS - day
            still_needed = mv["required_days"] - len(visited)
            if still_needed > 0 and remaining_days <= still_needed + 2:
                dist = _haversine_km(lat, lng, mv["lat"], mv["lng"])
                if dist <= mv.get("radius_km", 1.0):
                    visited.add(day)
                elif dist < 250 and now + _travel_minutes(dist) <= day_end:
                    repo = self._planned_reposition(rules, plan, lat, lng, mv["lat"], mv["lng"])
                    if repo is not None:
                        return repo

        # (D5) home_rule: reposition to home before cutoff, idle until morning.
        # Only enforce when home coordinates were explicitly found in preference text.
        if rules.home_by_minute is not None and rules.home_lat is not None and day not in plan["home_done"]:
            if tod >= rules.home_by_minute:
                plan["home_done"].add(day)
                dist = _haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0)
                if dist > rules.home_radius_km:
                    if now + _travel_minutes(dist) <= day_end:
                        return self._reposition(rules.home_lat, rules.home_lng or 0)
                    return self._wait(day_end - now)
                return self._wait(day_end - now)
            # If close to home_by_minute and far from home, start heading home
            travel_to_home = _travel_minutes(_haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0))
            dist_home = _haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0)
            if tod + travel_to_home + 180 >= rules.home_by_minute and dist_home > rules.home_radius_km:
                plan["home_done"].add(day)
                return self._reposition(rules.home_lat, rules.home_lng or 0)

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
            reasonable = (lat_span >= 0.5 and lng_span >= 0.5 and
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
                    repo = self._planned_reposition(rules, plan, lat, lng, tgt_lat, tgt_lng)
                    if repo is not None:
                        plan["bounded_repo"].add(day)
                        return repo

        if rules.daily_order_limit is not None:
            if plan["orders_today"].get(day, 0) >= rules.daily_order_limit:
                return self._wait(day_end - now)

        if self._first_order_deadline_unreachable(rules, plan, day, tod, 60):
            return self._wait(day_end - now)

        # (E) take the best compliant order, else idle to day end. A flexible-rest
        # driver may let the day's *last* order finish past midnight (up to a cap that
        # still leaves room for a full rest block inside the next day), but only when
        # the next day is an ordinary working day — never crossing into an off day,
        # blackout day or a dated-event day.
        hard_end = day_end
        # For home_rule, give a buffer before home_by_minute instead of hard-cutting.
        # Allow orders that finish up to 60 min before home_by_minute (leaves time to travel home).
        if rules.home_by_minute is not None and day not in plan.get("home_done", set()):
            buffer = max(60, _travel_minutes(rules.home_radius_km * 5) if rules.home_lat else 60)
            cutoff = day_start + rules.home_by_minute - buffer
            if cutoff > now:
                hard_end = min(hard_end, cutoff)
        if rules.rest_window is None and rules.daily_rest_minutes > 0:
            if self._next_day_is_ordinary(rules, plan, day):
                hard_end = max(hard_end, day_end + (DAY_MINUTES - rules.day_rest_block))
        order = self._pick_order(driver_id, status, rules, plan, now, lat, lng, day, hard_end)
        if order is not None:
            return order
        # (E') anti-stranding: no compliant order is reachable from here, so the driver
        # would otherwise idle the entire day. If a single reposition toward a profitable
        # cargo cluster turns the day productive (the post-reposition pickup is short, so
        # no deadhead-cap penalty), move there instead of sitting idle. `net` already
        # nets out the reposition distance, so this never loses money on the anchor order.
        strand = self._anti_strand(driver_id, rules, plan, now, lat, lng, day, hard_end)
        if strand is not None:
            return strand
        # (E'') Adaptive retry: more frequent during high-cargo hours (8-18),
        # less frequent during off-hours to avoid wasting API calls.
        remaining = day_end - now
        if remaining > 240:
            hour_of_day = (now % DAY_MINUTES) // 60
            if 8 <= hour_of_day < 18:
                retry = 90   # peak hours: retry every 1.5h
            else:
                retry = 150  # off-peak: retry every 2.5h
            if rules.home_by_minute is not None and rules.home_lat is not None and day not in plan.get("home_done", set()):
                dist_home = _haversine_km(lat, lng, rules.home_lat, rules.home_lng or 0)
                if dist_home > rules.home_radius_km:
                    latest_depart = day_start + rules.home_by_minute - _travel_minutes(dist_home) - 180
                    if latest_depart <= now:
                        plan["home_done"].add(day)
                        return self._reposition(rules.home_lat, rules.home_lng or 0)
                    retry = min(retry, max(1, latest_depart - now))
            return self._wait(retry)
        return self._wait(max(remaining, 1))

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

    def _first_order_deadline_unreachable(self, rules, plan, day: int, tod: int, query_k: int) -> bool:
        deadline = rules.first_order_before_minute
        if deadline is None or day in plan["first_order_taken"]:
            return False
        return tod + _query_scan_minutes(query_k) > deadline

    def _drive_route(self, ev, plan, now, day_start, lat, lng):
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
                return self._reposition(stop["lat"], stop["lng"])
            plan["dated_route_done"].add(ev["day"])  # missed; give up route
            return None
        if stop["min_wait"] > 0 and not st["waited"]:
            st["waited"] = True
            return self._wait(stop["min_wait"])
        st["idx"] += 1
        st["waited"] = False
        if st["idx"] >= len(stops):
            plan["dated_route_done"].add(ev["day"])
            return None
        return self._drive_route(ev, plan, now, day_start, lat, lng)

    def _pick_order(self, driver_id, status, rules, plan, now, lat, lng, day, day_end):
        # daily_order_limit check
        if rules.daily_order_limit is not None:
            count = plan["orders_today"].get(day, 0)
            if count >= rules.daily_order_limit:
                return None
        # first_order timing check: if no order taken today and it's past the deadline
        if self._first_order_deadline_unreachable(rules, plan, day, now % DAY_MINUTES, 60):
            return None
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=180)
        items = cargo_resp.get("items", [])
        now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
        blackout_regions = {r for r, days in rules.blackout if day in days}
        need_zeng = (
            rules.required_region is not None
            and len(plan["zeng_order_days"]) < rules.required_region[1]
        )
        best = None
        best_score = 0.0
        best_is_required = False
        best_pickup_km = 0.0
        for item in items:
            cargo = item.get("cargo", {})
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, plan, now, day_end, lat, lng)
            if ev is None:
                continue
            net, touches_required, occupied, pkm, hkm = ev
            # monthly deadhead cap: skip if this order would exceed limit
            if rules.monthly_deadhead_max_km is not None:
                if plan["total_deadhead_km"] + pkm > rules.monthly_deadhead_max_km:
                    continue
            score = self._enhanced_score(
                net, occupied, hkm, touches_required, need_zeng,
                cargo, rules, plan, day)
            is_req = bool(need_zeng and touches_required)
            if is_req and not best_is_required:
                best, best_score, best_is_required, best_pickup_km = (cargo, score, True, pkm)
            elif is_req == best_is_required and score > best_score:
                best, best_score, best_is_required, best_pickup_km = (cargo, score, is_req, pkm)
        # Widen search if the first pass found no positive risk-adjusted order.
        if best is None:
            if self._first_order_deadline_unreachable(rules, plan, day, now % DAY_MINUTES, 350):
                return None
            cargo_resp2 = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=350)
            items2 = cargo_resp2.get("items", [])
            now = int(self._api.get_driver_status(driver_id)["simulation_progress_minutes"])
            for item in items2:
                cargo = item.get("cargo", {})
                ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, plan, now, day_end, lat, lng)
                if ev is None:
                    continue
                net, touches_required, occupied, pkm, hkm = ev
                if rules.monthly_deadhead_max_km is not None:
                    if plan["total_deadhead_km"] + pkm > rules.monthly_deadhead_max_km:
                        continue
                score = self._enhanced_score(
                    net, occupied, hkm, touches_required, need_zeng,
                    cargo, rules, plan, day)
                is_req = bool(need_zeng and touches_required)
                if is_req and not best_is_required:
                    best, best_score, best_is_required, best_pickup_km = (cargo, score, True, pkm)
                elif is_req == best_is_required and score > best_score:
                    best, best_score, best_is_required, best_pickup_km = (cargo, score, is_req, pkm)
        if best is None:
            return None
        if best_is_required:
            plan["zeng_order_days"].add(day)
        # Track order count, first-order flag, and deadhead
        plan["orders_today"][day] = plan["orders_today"].get(day, 0) + 1
        plan["first_order_taken"].add(day)
        plan["total_deadhead_km"] += best_pickup_km
        return self._take_order(str(best.get("cargo_id")))


    def _enhanced_score(self, net, occupied, haul_km, touches_required, need_zeng,
                        cargo, rules, plan, day):
        """General-purpose scoring that works across unknown driver/cargo distributions."""
        score = net / occupied

        # (1) Dynamic haul-distance penalty: long hauls strand drivers.
        # Threshold adapts to bounded_area size; defaults to 200km without one.
        haul_thresh = 200.0
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            diag = _haversine_km(la_min, ln_min, la_max, ln_max)
            haul_thresh = max(100.0, diag * 0.8)
        if haul_km > haul_thresh:
            score *= max(0.4, 1.0 - (haul_km - haul_thresh) / (haul_thresh * 2))

        # (2) Dropoff stays in bounded_area: prefer orders that don't push driver out.
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            end = cargo.get("end") or {}
            elat = float(end.get("lat", 0.0))
            elng = float(end.get("lng", 0.0))
            if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                score *= 0.6

        # (3) Required-region boost when quota not met.
        if need_zeng and touches_required:
            score *= 1.5

        # (4) First-order-of-day boost: avoid wasting morning on marginal comparisons.
        if plan["orders_today"].get(day, 0) == 0:
            score *= 1.15

        return score

    def _anti_strand(self, driver_id, rules, plan, now, lat, lng, day, day_end):
        """When no compliant order is reachable from the current spot, the driver is
        stranded (e.g. a previous haul left it far from any cargo cluster) and would
        idle the whole day. Scan a wide radius for the best order that becomes workable
        *after a single reposition to its pickup*, and move toward it. The reposition
        deadhead is already folded into `net`, and the post-reposition pickup is ~0 km,
        so the deadhead cap is never tripped (penalty stays 0). Allow up to 2 repositions
        per day to avoid wasting entire days when first target has no cargo."""
        strand_count = plan.get("strand_count", {}).get(day, 0)
        if strand_count >= 3:
            return None
        if day_end - now < _STRAND_MIN_BUDGET:
            return None
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
        items = cargo_resp.get("items", [])
        blackout_regions = {r for r, days in rules.blackout if day in days}
        need_zeng = (
            rules.required_region is not None
            and len(plan.get("zeng_order_days", set())) < rules.required_region[1]
        )
        best_target = None
        best_net = 0.0
        best_is_req = False
        for item in items:
            cargo = item.get("cargo", {})
            ev = self._evaluate_relocation(cargo, rules, blackout_regions, now, day_end, lat, lng)
            if ev is None:
                continue
            net, tlat, tlng = ev
            # Boost score for required-region cargo during anti-strand
            is_req = False
            if need_zeng:
                start = cargo.get("start") or {}
                end = cargo.get("end") or {}
                region = rules.required_region[0]
                is_req = (_region_in_city(region, str(start.get("city", "")))
                          or _region_in_city(region, str(end.get("city", ""))))
                if is_req:
                    net *= 1.5
            # Penalize dropoff outside bounded_area
            if rules.bounded_area is not None:
                la_min, la_max, ln_min, ln_max = rules.bounded_area
                end = cargo.get("end") or {}
                elat = float(end.get("lat", 0.0))
                elng = float(end.get("lng", 0.0))
                if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                    net *= 0.5
            if is_req and not best_is_req:
                best_target, best_net, best_is_req = (tlat, tlng), net, True
            elif is_req == best_is_req and net > best_net:
                best_target, best_net, best_is_req = (tlat, tlng), net, is_req
        if best_target is None:
            return None
        repo = self._planned_reposition(rules, plan, lat, lng, best_target[0], best_target[1])
        if repo is None:
            return None
        plan["strand_repo"].add(day)
        plan.setdefault("strand_count", {})
        plan["strand_count"][day] = plan["strand_count"].get(day, 0) + 1
        return repo

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
            if area_lat_span >= 0.5 and area_lng_span >= 0.5 and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                if not (la_min <= slat <= la_max and ln_min <= slng <= ln_max):
                    return None
                if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                    return None
        move_km = _haversine_km(lat, lng, slat, slng)
        arrival = now + (_travel_minutes(move_km) if move_km > 1e-6 else 0)
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
        haul_km = _haversine_km(slat, slng, elat, elng)
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (move_km + haul_km)
        if net <= 0:
            return None
        net *= avoid_penalty
        return net, slat, slng

    def _evaluate_cargo(self, cargo, item, rules, blackout_regions, plan, now, day_end, lat, lng):
        name = str(cargo.get("cargo_name", ""))
        preference_penalty = 0.0
        if self._is_forbidden_cargo(name, rules.forbidden_categories):
            preference_penalty += self._matched_category_penalty(name, rules.forbidden_category_penalty, 2200.0)
        # avoid_categories: soft penalty (50% score reduction) rather than hard rejection
        avoid_penalty = 0.5 if self._is_forbidden_cargo(name, rules.avoid_categories) else 1.0
        start = cargo.get("start") or {}
        end = cargo.get("end") or {}
        scity = str(start.get("city", ""))
        ecity = str(end.get("city", ""))
        slat, slng = float(start.get("lat", 0.0)), float(start.get("lng", 0.0))
        elat, elng = float(end.get("lat", 0.0)), float(end.get("lng", 0.0))
        for region in rules.forbidden_regions:
            if _region_in_city(region, scity) or _region_in_city(region, ecity):
                preference_penalty += rules.forbidden_region_penalty.get(region, 1800.0)
        for region in blackout_regions:
            blackout_hit = _region_in_city(region, scity) or _region_in_city(region, ecity)
            if region == "深圳" and (_in_shenzhen(slat, slng) or _in_shenzhen(elat, elng)):
                blackout_hit = True
            if region in rules.blackout_coords:
                rlat, rlng = rules.blackout_coords[region]
                if _haversine_km(slat, slng, rlat, rlng) < 60 or _haversine_km(elat, elng, rlat, rlng) < 60:
                    blackout_hit = True
            if blackout_hit:
                preference_penalty += rules.blackout_penalty.get(region, 3000.0)
        # forbidden_zones: circle-zone check on pickup/dropoff
        # Only enforce if coordinates look reasonable (latitude 18-55, longitude 70-140 for China)
        for fz_lat, fz_lng, fz_r in rules.forbidden_zones:
            if not (18 <= fz_lat <= 55 and 70 <= fz_lng <= 140):
                continue  # likely hallucinated coordinates
            if _haversine_km(slat, slng, fz_lat, fz_lng) < fz_r or _haversine_km(elat, elng, fz_lat, fz_lng) < fz_r:
                preference_penalty += rules.forbidden_zone_penalty
        # bounded_area: only accept orders within operating bounds
        # Only enforce if the area looks reasonable (not too small / not covering all of China)
        if rules.bounded_area is not None:
            la_min, la_max, ln_min, ln_max = rules.bounded_area
            area_lat_span = la_max - la_min
            area_lng_span = ln_max - ln_min
            # Skip if area is unreasonably small (<0.5 degree) or coordinates out of China range
            if area_lat_span >= 0.5 and area_lng_span >= 0.5 and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
                if not (la_min <= slat <= la_max and ln_min <= slng <= ln_max):
                    preference_penalty += rules.bounded_area_penalty
                if not (la_min <= elat <= la_max and ln_min <= elng <= ln_max):
                    preference_penalty += rules.bounded_area_penalty
        pickup_km = _haversine_km(lat, lng, slat, slng)
        if rules.pickup_max_km is not None and pickup_km > rules.pickup_max_km:
            preference_penalty += rules.pickup_max_penalty
        cost_time = int(cargo.get("cost_time_minutes", 0))
        pickup_min = _travel_minutes(pickup_km) if pickup_km > 1e-6 else 0
        arrival = now + pickup_min
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
        if rules.rest_window is not None:
            rs, re = rules.rest_window
            if re <= rs:
                re += DAY_MINUTES
            for base in (now // DAY_MINUTES - 1, now // DAY_MINUTES, now // DAY_MINUTES + 1):
                ws, we = base * DAY_MINUTES + rs, base * DAY_MINUTES + re
                if now < we and finish > ws:
                    return None
        if rules.home_by_minute is not None and rules.home_lat is not None:
            day = now // DAY_MINUTES
            deadline = day * DAY_MINUTES + rules.home_by_minute
            if day not in plan.get("home_done", set()) and now < deadline:
                travel_home = _travel_minutes(_haversine_km(elat, elng, rules.home_lat, rules.home_lng or 0))
                if finish + travel_home + 180 > deadline:
                    return None
        haul_km = _haversine_km(slat, slng, elat, elng)
        # haul_max_km: single-order haul distance limit
        if rules.haul_max_km is not None and haul_km > rules.haul_max_km:
            preference_penalty += rules.haul_max_penalty
        # no_drive_windows: check if order execution overlaps a no-drive window
        # Skip windows fully covered by rest_window (already handled in schedule)
        _, start_tod = divmod(now, DAY_MINUTES)
        _, finish_tod = divmod(finish, DAY_MINUTES)
        for ws, we in rules.no_drive_windows:
            if rules.rest_window is not None:
                rws, rwe = rules.rest_window
                if rws <= ws and rwe >= min(we, DAY_MINUTES):
                    continue  # already covered by rest_window
            we_clamp = min(we, DAY_MINUTES)
            if start_tod < we_clamp and finish_tod > ws:
                return None
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (pickup_km + haul_km) - preference_penalty
        if net <= 0:
            return None
        if preference_penalty > 0 and net < max(300.0, preference_penalty * 0.25):
            return None
        # Apply avoid_categories soft penalty
        net *= avoid_penalty
        touches_required = False
        if rules.required_region is not None:
            region = rules.required_region[0]
            touches_required = _region_in_city(region, scity) or _region_in_city(region, ecity)
        occupied = max(1, finish - now)
        return net, touches_required, occupied, pickup_km, haul_km

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

    @staticmethod
    def _matched_category_penalty(cargo_name: str, penalties: dict[str, float], default: float) -> float:
        if not cargo_name:
            return default
        for cat, penalty in penalties.items():
            if cat in cargo_name or cargo_name in cat:
                return float(penalty)
        return default

    # ------------------------------------------------------------- action dsl
    @staticmethod
    def _wait(duration_minutes: int) -> dict[str, Any]:
        return {"action": "wait", "params": {"duration_minutes": int(max(1, duration_minutes))}}

    def _planned_reposition(self, rules, plan, lat: float, lng: float, target_lat: float, target_lng: float) -> dict[str, Any] | None:
        dist = _haversine_km(lat, lng, target_lat, target_lng)
        if rules.monthly_deadhead_max_km is not None:
            if plan["total_deadhead_km"] + dist > rules.monthly_deadhead_max_km:
                return None
        plan["total_deadhead_km"] += dist
        return self._reposition(target_lat, target_lng)

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
        self._apply_penalty_metadata(prefs, rules)
        seen.update(texts)
        # Dedup avoid/forbidden AFTER supplement so regex-added items are also cleaned
        self._dedup_avoid_forbidden(rules)
        if self._rules_fingerprint(rules) != before:
            self._logger.info(
                "parsed rules driver_id=%s rest=%s window=%s off=%s forbid_cat=%s avoid_cat=%s "
                "forbid_reg=%s required=%s pickup_max=%s haul_max=%s blackout=%s "
                "dated_single=%s dated_route=%s no_drive=%s order_limit=%s home=%s "
                "forbidden_zones=%d bounded=%s must_visit=%d first_order=%s",
                driver_id,
                rules.daily_rest_minutes,
                rules.rest_window,
                rules.off_days_min,
                rules.forbidden_categories,
                rules.avoid_categories,
                rules.forbidden_regions,
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

    def _apply_penalty_metadata(self, prefs: list[Any], rules: DriverRules) -> None:
        for pref in prefs:
            if not isinstance(pref, dict):
                continue
            text = str(pref.get("content", ""))
            try:
                penalty = float(pref.get("penalty_amount", 0) or 0)
            except (TypeError, ValueError):
                penalty = 0.0
            if penalty <= 0:
                continue
            cat = self._parse_forbidden_category(text)
            if cat:
                for known in rules.forbidden_categories:
                    if cat in known or known in cat:
                        rules.forbidden_category_penalty[known] = penalty
            reg_m = re.search(r"在([\u4e00-\u9fa5]{2,4})的货", text)
            if reg_m:
                reg = self._clean_region_name(reg_m.group(1))
                for known in rules.forbidden_regions:
                    if _region_in_city(reg, known) or _region_in_city(known, reg):
                        rules.forbidden_region_penalty[known] = penalty
            if "空驶" in text and "超过" in text and "月" not in text:
                rules.pickup_max_penalty = penalty
            if ("干线" in text or "单笔" in text or "单趟" in text or "运距" in text or "运货" in text) and "公里" in text:
                rules.haul_max_penalty = penalty
            if ("北纬" in text or "纬" in text) and ("东经" in text or "经" in text):
                rules.bounded_area_penalty = penalty
            if "半径" in text and "公里" in text and ("不进" in text or "不去" in text or "禁" in text):
                rules.forbidden_zone_penalty = penalty
            region = self._parse_blackout_region(text)
            if region:
                rules.blackout_penalty[region] = penalty

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
        '10) "接上配偶/家人→返回老家/进家门"是 dated_route 事件（多点路线），不是 home_rule。\n\n'
        "示例1：\n"
        '入: {"preferences":["每天零点到六点停着熄火睡觉","凡是生鲜货源碰不得","三月四号五号不往深圳（22.55，114.05）跑"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":{"start_hour":0,"end_hour":6},'
        '"no_drive_windows":[{"start_hour":0,"end_hour":6}],"off_days_min":0,'
        '"forbidden_categories":["生鲜"],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"home_rule":null,'
        '"blackout":[{"region":"深圳","dates":[4,5]}],"dated_single":[],"dated_route":[]}\n\n'
        "示例2：\n"
        '入: {"preferences":["十二号得去仓库（23.15，113.67）盘库，花两小时","连续休息满8小时","空驶超过五十五公里别接"]}\n'
        '出: {"daily_rest_hours":8,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":55,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"home_rule":null,'
        '"blackout":[],"dated_single":[{"date":12,"lat":23.15,"lng":113.67,"wait_minutes":120,"before_hour":null}],"dated_route":[]}\n\n'
        "示例3：\n"
        '入: {"preferences":["三十一号先过档口（23.15，113.67）取礼物，中午十二点前赶到县城（23.35，112.47）赴宴到下午两点"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"home_rule":null,'
        '"blackout":[],"dated_single":[],"dated_route":[{"date":31,"stops":[{"lat":23.15,"lng":113.67,"wait_minutes":0,"before_hour":12},{"lat":23.35,"lng":112.47,"wait_minutes":120,"before_hour":12}]}]}\n\n'
        "示例4：\n"
        '入: {"preferences":["龙门吊底座、机床铸件这类机械设备活儿干不了","装货地或卸货地在惠州的货一律不接"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":["龙门吊底座","机床铸件","机械设备"],"avoid_categories":[],"forbidden_regions":["惠州"],'
        '"forbidden_zones":[],"bounded_area":null,"required_region":null,"must_visit":[],'
        '"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"home_rule":null,'
        '"blackout":[],"dated_single":[],"dated_route":[]}\n\n'
        "示例5：\n"
        '入: {"preferences":["每天23点前车辆须在自家位置（23.10，113.50）1公里内，到次日8点前不接单不空跑","同一天接单不得超过3单"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":null,"no_drive_windows":[{"start_hour":23,"end_hour":8}],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":3,"first_order_before_hour":null,'
        '"home_rule":{"lat":23.10,"lng":113.50,"radius_km":1,"home_by_hour":23,"no_drive_until_hour":8},'
        '"blackout":[],"dated_single":[],"dated_route":[]}\n\n'
        "示例6：\n"
        '入: {"preferences":["十一点半到下午一点半歇晌，雷打不动","二十号去老李仓库（23.25，113.40）对账，大概两小时"]}\n'
        '出: {"daily_rest_hours":null,"rest_window":{"start_hour":11.5,"end_hour":13.5},'
        '"no_drive_windows":[],"off_days_min":0,'
        '"forbidden_categories":[],"avoid_categories":[],"forbidden_regions":[],"forbidden_zones":[],"bounded_area":null,'
        '"required_region":null,"must_visit":[],"pickup_max_km":null,"haul_max_km":null,"monthly_deadhead_max_km":null,'
        '"daily_order_limit":null,"first_order_before_hour":null,"home_rule":null,'
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
        r"在[\u4e00-\u9fa5]{2,4}(?:的货|那|路|边|方向)|"  # "在惠州的货"
        r"(?:省|市|区|县|镇|村)$"
    )

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
        for tail in ("的货", "那一路", "方向", "那边", "一带"):
            if r.endswith(tail) and len(r) > len(tail):
                r = r[:-len(tail)]
        return r.strip()

    def _merge_llm_rules(self, rules: DriverRules, data: dict[str, Any], texts: list[str] | None = None) -> None:
        all_text = "\n".join(texts) if texts else ""
        rest_h = data.get("daily_rest_hours")
        if isinstance(rest_h, (int, float)) and rest_h > 0:
            rules.daily_rest_minutes = max(rules.daily_rest_minutes, int(round(rest_h * 60)))
        rw = data.get("rest_window")
        if isinstance(rw, dict):
            sh, eh = rw.get("start_hour"), rw.get("end_hour")
            if isinstance(sh, (int, float)) and isinstance(eh, (int, float)):
                sm, em = int(round(sh * 60)), int(round(eh * 60))
                if em > sm:
                    rules.rest_window = (sm, em)
                elif sm > em > 0:
                    # Overnight window (e.g. 22:00-05:00): morning part as rest_window
                    rules.rest_window = (0, em)
                    overnight_min = 24 * 60 - sm + em
                    rules.daily_rest_minutes = max(rules.daily_rest_minutes, overnight_min)
                    self._logger.info("llm: overnight rest_window %d-%d -> rest_window=(0,%d) rest_min=%d",
                                      sm, em, em, overnight_min)
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
            if not all_text or c in all_text or _norm_region(c) in all_text:
                rules.forbidden_categories.add(c)
            else:
                self._logger.info("llm validation: rejected forbidden_category '%s' (not in texts)", c)

        for reg in raw_regs:
            r = self._clean_region_name(reg)
            if not r:
                continue
            if not all_text or r in all_text or _norm_region(r) in all_text:
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
                if not all_text or r in all_text or _norm_region(r) in all_text:
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
        _NDW_TIME_KW = ("点", "时", "小时", "上午", "下午", "中午", "凌晨", "晚上", "早上")
        _NDW_ACTION_KW = ("不出车", "不接单", "不开车", "不跑车", "不空跑", "不空驶", "不运营", "不工作", "不干活", "不接活", "不赶路", "不许出", "不允许出", "别派活", "别赶路", "停车熄火", "禁止出车", "不准出车", "别开车", "别跑车", "收工", "歇着", "休息", "睡觉", "不动弹", "不跑", "不接")
        _NDW_KW = _NDW_ACTION_KW  # for logging compatibility
        _AVOID_KW = ("少接", "尽量不", "尽量少", "避免", "不太想", "不愿意", "不喜欢", "最好别", "别给我", "尽量别", "能不接", "嫌麻烦", "不是绝对", "除非价钱", "能换就换", "不太愿意", "能不碰")
        _FZ_KW = ("不进", "不去", "禁止进入", "不要去", "不要进", "别去", "远离", "不往", "不到", "不可进", "不得进", "禁入", "不允许进", "严禁", "禁驶入", "禁止驶入", "堵", "修路", "不想跑", "不做")
        _BA_KW = ("范围", "区域内", "不超出", "只在", "仅在", "限定", "活动区域", "纬度", "经度", "运营区域", "只做", "只跑")
        _MV_KW = ("必须去", "一定要到", "每月去", "至少去", "必须到", "必访", "定期去", "经过", "起码", "至少", "接够")
        _HOME_KW = ("回家", "到家", "家里", "返回住所", "回住处", "回去", "回到家", "归家", "在家", "家附近", "停在家")
        _DOL_KW = ("不超过", "上限", "最多", "不得超过", "不得多于", "顶多", "封顶", "单以内", "趟以内")
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
                    if s and (s in all_text or _norm_region(s) in all_text):
                        rules.avoid_categories.add(s)
                    elif s:
                        self._logger.info("llm grounding: rejected avoid_category '%s' (not in text)", s)
        elif data.get("avoid_categories"):
            self._logger.info("llm grounding: rejected avoid_categories (no keywords in text)")

        # NOTE: avoid/forbidden dedup moved to _dedup_avoid_forbidden (runs after supplement)

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
                    if la_max - la_min >= 0.5 and ln_max - ln_min >= 0.5 and 18 <= la_min and la_max <= 55 and 70 <= ln_min and ln_max <= 140:
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
        if _text_has_any(_DOL_KW):
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
        if fob is not None and _text_has_any(_FOB_KW):
            if isinstance(fob, (int, float)) and 0 < fob <= 24:
                rules.first_order_before_minute = int(fob) * 60
            elif isinstance(fob, str):
                fm = re.search(r'(\d+)', fob)
                if fm:
                    h = int(fm.group(1))
                    if 0 < h <= 24:
                        rules.first_order_before_minute = h * 60
        elif fob is not None:
            self._logger.info("llm grounding: rejected first_order_before_hour=%s (no keywords in text)", fob)

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
        if ("睡觉" in text or "停着熄火" in text or "雷打不动" in text) and "点" in text:
            window = self._parse_time_window(text)
            if window is not None:
                rules.rest_window = window
        if "整天" in text and ("歇" in text or "休息" in text or "停驶" in text or "检修" in text or "保养" in text):
            cnt = self._parse_cn_count(text)
            if cnt:
                rules.off_days_min = max(rules.off_days_min, cnt)
        if "空驶" in text and "超过" in text and "月" not in text:
            km = self._parse_distance_km(text)
            if km and rules.pickup_max_km is None:
                rules.pickup_max_km = km
        # forbidden/avoid category: patterns like "X的活/货...干不了/推掉/不接/不拉"
        _is_soft = any(kw in text for kw in ("尽量不", "尽量少", "尽量别", "最好别", "不太想", "不愿意"))
        cat_m = re.search(
            r"[\"\"「]?([\u4e00-\u9fa5]{2,6}?)[\"\"」]?"
            r"(?:的活|的货|货源|类货|这类|那类).*?"
            r"(?:干不了|推掉|不接|不拉|不碰|不做|碰不得|接不了)",
            text,
        )
        if cat_m:
            cat_val = cat_m.group(1)
            cat_val = re.sub(r"^(?:凡是|所有|一切|任何)", "", cat_val).strip()
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
            r"(?:装货地|卸货地|目的地)?在[\"\"「]?([\u4e00-\u9fa5]{2,4})[\"\"」]?"
            r".*?(?:一律不接|不接|不要|不跑|不去|不做)",
            text,
        )
        if reg_m and rules.bounded_area is None:
            rules.forbidden_regions.add(self._clean_region_name(reg_m.group(1)))
        # required region: "在X的货...接够N个不同的日子"
        if ("不同的日子" in text or "不同日子" in text) and rules.required_region is None:
            mr = re.search(r"在([\u4e00-\u9fa5]{2,4})的货", text)
            cnt = self._parse_cn_count(text)
            if mr and cnt:
                rules.required_region = (mr.group(1), cnt)
        # blackout: various patterns for "don't go to region X on dates Y"
        blackout_region = None
        if "不往" in text and "跑" in text:
            m2 = re.search(r"不往([\u4e00-\u9fa5]{2,4})跑", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "不进" in text:
            m2 = re.search(r"不进([\u4e00-\u9fa5]{2,4})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "别给我派" in text:
            m2 = re.search(r"别给我派.*?([\u4e00-\u9fa5]{2,4})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and ("不去" in text or "别去" in text):
            m2 = re.search(r"(?:不去|别去)([\u4e00-\u9fa5]{2,4})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and "别安排" in text:
            m2 = re.search(r"别安排.*?([\u4e00-\u9fa5]{2,4})", text)
            if m2:
                blackout_region = m2.group(1)
        if blackout_region is None and ("不跑" in text or "不接" in text):
            m2 = re.search(r"(?:不跑|不接)([\u4e00-\u9fa5]{2,4})(?:的[活单货])?", text)
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
        if rules.daily_order_limit is None and ("单" in text or "接单" in text or "趟" in text):
            dol_m = re.search(r"(?:不超过|不得超过|最多|上限|顶多)\s*(?:跑|接)?\s*([一二两三四五六七八九十\d]+)\s*(?:个)?\s*(?:单|趟)", text)
            if dol_m:
                rules.daily_order_limit = _cn_to_int(dol_m.group(1))
        # haul_max_km: "干线/单笔距离不超过N公里"
        if rules.haul_max_km is None and ("干线" in text or "单笔" in text) and ("公里" in text or "距离" in text):
            hm_m = re.search(r"(?:不超过|不得超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if hm_m:
                rules.haul_max_km = float(_cn_to_int(hm_m.group(1)))
        # no_drive_window: "N点到M点不接单/不空跑/不出车"
        if not rules.no_drive_windows and ("不接单" in text or "不空跑" in text or "不出车" in text or "不空驶" in text or "不跑车" in text or "不接活" in text or "别派" in text or "别赶" in text or "不许" in text or "不允许" in text) and "点" in text:
            ndw_window = self._parse_time_window(text)
            if ndw_window is not None:
                sm, em = ndw_window
                # handle cross-midnight: end < start → wrap to next day
                if em <= sm:
                    em += DAY_MINUTES
                rules.no_drive_windows.append((sm, em))
        # home_rule: "X点前须在自家/回家/到家...Y公里" (complex, rely more on LLM)
        # monthly_deadhead_max_km: "月累计空驶不超过N公里"
        if rules.monthly_deadhead_max_km is None and "月" in text and "空驶" in text and "公里" in text:
            mdh_m = re.search(r"(?:不超过|不得超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if mdh_m:
                rules.monthly_deadhead_max_km = float(_cn_to_int(mdh_m.group(1)))
        # first_order_before: "首单不得晚于N点"
        if rules.first_order_before_minute is None and ("首单" in text or "第一单" in text):
            fob_m = re.search(r"(?:不得晚于|不迟于|之前)\s*([零一二两三四五六七八九十\d]+)\s*点", text)
            if fob_m:
                rules.first_order_before_minute = _cn_to_int(fob_m.group(1)) * 60

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
        for day in days:
            if is_route and not any(e["day"] == day for e in rules.dated_route):
                before = DAY_MINUTES
                mb = re.search(r"([零一二两三四五六七八九十\d]+)\s*点前", text)
                if mb:
                    before = _cn_to_int(mb.group(1)) * 60
                stops = []
                for i, (_, lat, lng) in enumerate(found):
                    wait = 0
                    if i == len(found) - 1:
                        wait = self._parse_hours_minutes(text) or 120
                    stops.append({"lat": lat, "lng": lng, "min_wait": wait, "before": before})
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
                rules.rest_window = window
        # off days: "抽三个整天" / "留两个整天"
        if "整天" in text and ("歇" in text or "休息" in text or "停驶" in text or "检修" in text or "保养" in text):
            cnt = self._parse_cn_count(text)
            if cnt:
                rules.off_days_min = max(rules.off_days_min, cnt)
        # forbidden/avoid category: "X这类活儿...干不了" / "凡是X货源...推掉"
        _is_soft2 = any(kw in text for kw in ("尽量不", "尽量少", "尽量别", "最好别", "不太想", "不愿意"))
        if ("干不了" in text or "推掉" in text or ("一律不接" in text and "货源" not in text[:0])) and "扣" in text:
            cat = self._parse_forbidden_category(text)
            if cat:
                if _is_soft2:
                    rules.avoid_categories.add(cat)
                else:
                    rules.forbidden_categories.add(cat)
        # forbidden region: "装货地或卸货地在X的货,我一律不接"
        m = re.search(r"在([\u4e00-\u9fa5]{2,4}?)的货[，,]?\s*我一律不接", text)
        if m:
            rules.forbidden_regions.add(m.group(1))
        # required region with min days: "装货或卸货在X的货...接够N个不同的日子"
        if "不同的日子" in text or "不同日子" in text:
            mr = re.search(r"在([\u4e00-\u9fa5]{2,4})的货", text)
            cnt = self._parse_cn_count(text)
            if mr and cnt:
                rules.required_region = (mr.group(1), cnt)
        # pickup deadhead cap: "空驶超过五十五公里"/"空驶超过55公里"
        if "空驶" in text and "超过" in text:
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
        if ("不接单" in text or "不空跑" in text or "不出车" in text or "不空驶" in text) and "点" in text:
            ndw_window = self._parse_time_window(text)
            if ndw_window is not None:
                sm, em = ndw_window
                if em <= sm:
                    em += DAY_MINUTES
                if not any(ws == sm and we == em for ws, we in rules.no_drive_windows):
                    rules.no_drive_windows.append((sm, em))
        # daily_order_limit: "同一天不超过N单" / "顶多跑N趟"
        if rules.daily_order_limit is None and ("单" in text or "接单" in text or "趟" in text):
            dol_m = re.search(r"(?:不超过|不得超过|最多|上限|顶多)\s*(?:跑|接)?\s*([一二两三四五六七八九十\d]+)\s*(?:个)?\s*(?:单|趟)", text)
            if dol_m:
                rules.daily_order_limit = _cn_to_int(dol_m.group(1))
        # haul_max_km: "干线距离不超过N公里" / "单趟运距不能超过N公里" / "运货距离最多N公里"
        if rules.haul_max_km is None and ("干线" in text or "单笔" in text or "单趟" in text or "运距" in text or "运货" in text) and "公里" in text:
            hm_m = re.search(r"(?:不超过|不得超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if hm_m:
                rules.haul_max_km = float(_cn_to_int(hm_m.group(1)))
        # monthly_deadhead_max_km: "月累计空驶不超过N公里"
        if rules.monthly_deadhead_max_km is None and "月" in text and "空驶" in text and "公里" in text:
            mdh_m = re.search(r"(?:不超过|不得超过|上限)\s*([零一二两三四五六七八九十百千\d]+)\s*公里", text)
            if mdh_m:
                rules.monthly_deadhead_max_km = float(_cn_to_int(mdh_m.group(1)))
        # first_order_before: "首单不得晚于N点"
        if rules.first_order_before_minute is None and ("首单" in text or "第一单" in text):
            fob_m = re.search(r"(?:不得晚于|不迟于|之前)\s*([零一二两三四五六七八九十\d]+)\s*点", text)
            if fob_m:
                rules.first_order_before_minute = _cn_to_int(fob_m.group(1)) * 60
        # bounded_area: "北纬X到Y、东经A到B"
        if rules.bounded_area is None and ("北纬" in text or "纬" in text) and ("东经" in text or "经" in text):
            ba_m = re.search(
                r"(?:北纬|纬)\s*([0-9]+\.?[0-9]*)\s*到\s*([0-9]+\.?[0-9]*)[、，, ]+"
                r"(?:东经|经)\s*([0-9]+\.?[0-9]*)\s*到\s*([0-9]+\.?[0-9]*)",
                text,
            )
            if ba_m:
                la1, la2, ln1, ln2 = map(float, ba_m.groups())
                la_min, la_max = sorted((la1, la2))
                ln_min, ln_max = sorted((ln1, ln2))
                if la_max - la_min >= 0.3 and ln_max - ln_min >= 0.3:
                    rules.bounded_area = (la_min, la_max, ln_min, ln_max)
        # forbidden_zones: "坐标...半径N公里内不进"
        if ("半径" in text and "公里" in text and ("不进" in text or "不去" in text or "禁" in text)):
            z_m = re.search(
                r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）].*?"
                r"半径\s*([零一二两三四五六七八九十百\d]+)\s*公里",
                text,
            )
            if z_m:
                flat, flng = float(z_m.group(1)), float(z_m.group(2))
                fr = float(_cn_to_int(z_m.group(3)))
                if flat > 90 and flng < 90:
                    flat, flng = flng, flat
                if 18 <= flat <= 55 and 70 <= flng <= 140:
                    rules.forbidden_zones.append((flat, flng, fr))
        # must_visit: "至少N天到过X(lat,lng)R公里内"
        if "至少" in text and "天" in text and ("到过" in text or "去过" in text):
            mv_m = re.search(
                r"至少\s*([一二两三四五六七八九十\d]+)\s*天.*?"
                r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
                r".*?([零一二两三四五六七八九十百\d]+)\s*公里内",
                text,
            )
            if mv_m:
                days = _cn_to_int(mv_m.group(1))
                mlat, mlng = float(mv_m.group(2)), float(mv_m.group(3))
                mr = float(_cn_to_int(mv_m.group(4)))
                if mlat > 90 and mlng < 90:
                    mlat, mlng = mlng, mlat
                if 18 <= mlat <= 55 and 70 <= mlng <= 140:
                    rules.must_visit.append({"lat": mlat, "lng": mlng, "radius_km": mr, "required_days": days})
        # home_rule: "每天X点前/必须回...自家位置(lat,lng)R公里内，到次日Y点前不接单不空跑"
        if rules.home_by_minute is None and any(kw in text for kw in ("自家", "回家", "到家", "家里", "回到", "进家")):
            hr_m = re.search(
                r"每天\s*([零一二两三四五六七八九十\d]+)\s*点(?:前|.*?(?:必须|须)).*?"
                r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
                r"\s*([零一二两三四五六七八九十百\d]+)\s*公里内.*?"
                r"次日\s*([零一二两三四五六七八九十\d]+)\s*点前",
                text,
            )
            if not hr_m:
                hr_m = re.search(
                    r"每[天日]\s*([零一二两三四五六七八九十\d]+)\s*点.*?"
                    r"(?:回到?|须在|必须).*?"
                    r"[（(]\s*([0-9]+\.?[0-9]*)\s*[，,]\s*([0-9]+\.?[0-9]*)\s*[)）]"
                    r"\s*([零一二两三四五六七八九十百\d]+)\s*公里内.*?"
                    r"(?:次日|到.*?日)\s*([零一二两三四五六七八九十\d]+)\s*点",
                    text,
                )
            if hr_m:
                rules.home_by_minute = _cn_to_int(hr_m.group(1)) * 60
                hlat, hlng = float(hr_m.group(2)), float(hr_m.group(3))
                if hlat > 90 and hlng < 90:
                    hlat, hlng = hlng, hlat
                rules.home_lat = hlat
                rules.home_lng = hlng
                rules.home_radius_km = float(_cn_to_int(hr_m.group(4)))
                rules.no_drive_until_minute = _cn_to_int(hr_m.group(5)) * 60

    # ----------------------------------------------------- small text parsers
    @staticmethod
    def _parse_cn_count(text: str) -> int:
        m = re.search(r"([一二两三四五六七八九十\d]+)\s*个?\s*(?:整天|不同的日子|不同日子|个不同)", text)
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
    def _parse_time_window(text: str) -> tuple[int, int] | None:
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
        m = re.search(r"不往\s*([\u4e00-\u9fa5]{2,4}?)\s*跑", text)
        if m:
            return m.group(1)
        m = re.search(r"不进\s*([\u4e00-\u9fa5]{2,4})", text)
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
    _y, _mo, d, hh, mm = (int(x) for x in m.groups())
    return (d - 1) * DAY_MINUTES + hh * 60 + mm
