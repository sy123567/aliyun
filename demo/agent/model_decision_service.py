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
        self.forbidden_regions: set[str] = set()
        self.required_region: tuple[str, int] | None = None  # (region, min_days)
        self.pickup_max_km: float | None = None
        self.blackout: list[tuple[str, set[int]]] = []  # (region, days)
        self.dated_single: list[dict[str, Any]] = []  # {day,lat,lng,min_wait,before}
        self.dated_route: list[dict[str, Any]] = []  # {day, stops:[{lat,lng,min_wait,before}]}

    @property
    def day_rest_block(self) -> int:
        """每日开工前在 00:00 起需要的连续静止分钟数。"""
        block = self.daily_rest_minutes
        if self.rest_window is not None:
            block = max(block, self.rest_window[1])
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

        # (A') blackout day while sitting inside a forbidden bbox: any move would be
        # penalised, so idle the whole day (waiting is never penalised).
        for region, days in rules.blackout:
            if day in days and region == "深圳" and _in_shenzhen(lat, lng):
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
                dur = block if tod + block <= DAY_MINUTES else max(0, DAY_MINUTES - tod)
            else:
                dur = min(block - tod, day_end - now) if tod < block else 0
            if dur > 0:
                return self._wait(dur)
        else:
            plan["rest_done"].add(day)

        # (C) dated single-stop events (e.g. 盘库).
        for ev in rules.dated_single:
            if ev["day"] != day or ev["day"] in plan["dated_single_done"]:
                continue
            before = day_start + ev["before"]
            if _haversine_km(lat, lng, ev["lat"], ev["lng"]) > 1.5:
                if now + _travel_minutes(_haversine_km(lat, lng, ev["lat"], ev["lng"])) <= before:
                    return self._reposition(ev["lat"], ev["lng"])
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
                    return self._reposition(first["lat"], first["lng"])
            else:
                return self._wait(day_end - now)

        # (E) take the best compliant order, else idle to day end. A flexible-rest
        # driver may let the day's *last* order finish past midnight (up to a cap that
        # still leaves room for a full rest block inside the next day), but only when
        # the next day is an ordinary working day — never crossing into an off day,
        # blackout day or a dated-event day.
        hard_end = day_end
        if rules.rest_window is None and rules.daily_rest_minutes > 0:
            if self._next_day_is_ordinary(rules, plan, day):
                hard_end = day_end + (DAY_MINUTES - rules.day_rest_block)
        order = self._pick_order(driver_id, status, rules, plan, now, lat, lng, day, hard_end)
        if order is not None:
            return order
        return self._wait(max(day_end - now, 1))

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
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=60)
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
        for item in items:
            cargo = item.get("cargo", {})
            ev = self._evaluate_cargo(cargo, item, rules, blackout_regions, now, day_end, lat, lng)
            if ev is None:
                continue
            net, touches_required, occupied = ev
            # Value-density selection: net income per occupied minute. Picking the
            # single highest-net order is myopic — one long haul eats the whole day and
            # blocks a second order. Maximising net/minute packs more profitable work
            # into the fixed daily working window (rest windows stay clean), which both
            # lifts gross and trims deadhead cost vs. plain max-net.
            score = net / occupied
            is_req = bool(need_zeng and touches_required)
            # Prefer a required-region order (worth the monthly penalty) over plain net.
            if is_req and not best_is_required:
                best, best_score, best_is_required = (cargo, score, True)
            elif is_req == best_is_required and score > best_score:
                best, best_score, best_is_required = (cargo, score, is_req)
        if best is None:
            return None
        if best_is_required:
            plan["zeng_order_days"].add(day)
        return self._take_order(str(best.get("cargo_id")))

    def _evaluate_cargo(self, cargo, item, rules, blackout_regions, now, day_end, lat, lng):
        name = str(cargo.get("cargo_name", ""))
        if name in rules.forbidden_categories:
            return None
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
        pickup_km = _haversine_km(lat, lng, slat, slng)
        if rules.pickup_max_km is not None and pickup_km > rules.pickup_max_km:
            return None
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
        haul_km = _haversine_km(slat, slng, elat, elng)
        price = float(cargo.get("price", 0.0))
        net = price - COST_PER_KM * (pickup_km + haul_km)
        if net <= 0:
            return None
        touches_required = False
        if rules.required_region is not None:
            region = rules.required_region[0]
            touches_required = _region_in_city(region, scity) or _region_in_city(region, ecity)
        occupied = max(1, finish - now)
        return net, touches_required, occupied

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
        parsed_by_llm = self._llm_parse_preferences(driver_id, texts, rules)
        if not parsed_by_llm:
            # offline / model unavailable: fall back to the deterministic regex parser.
            coord_map = self._collect_coords(prefs)
            for text in texts:
                self._parse_one(text, rules, coord_map)
        seen.update(texts)
        if self._rules_fingerprint(rules) != before:
            self._logger.info(
                "parsed rules driver_id=%s rest=%s window=%s off=%s forbid_cat=%s forbid_reg=%s "
                "required=%s pickup_max=%s blackout=%s dated_single=%s dated_route=%s",
                driver_id,
                rules.daily_rest_minutes,
                rules.rest_window,
                rules.off_days_min,
                rules.forbidden_categories,
                rules.forbidden_regions,
                rules.required_region,
                rules.pickup_max_km,
                rules.blackout,
                rules.dated_single,
                rules.dated_route,
            )
        return rules

    # ----------------------------------------------------------- LLM preference parsing
    _PARSE_SYSTEM = (
        "你是货运司机偏好抽取器。把司机的自然语言偏好转成严格 JSON，供调度器执行。"
        "只输出一个 JSON 对象，禁止 markdown / 解释 / 多余文本。未提及的字段用 null 或空数组。"
        "字段定义（含义务必准确，宁缺毋错）：\n"
        '- daily_rest_hours: 数字或 null。每天必须「连续静止休息」的最少小时数（如“每天连续休息满8小时”→8）。\n'
        '- rest_window: {"start_hour":整数,"end_hour":整数} 或 null。每天固定必须停车静止的时段（如“每天0点到6点必须停车熄火”→{"start_hour":0,"end_hour":6}）。\n'
        '- off_days_min: 整数。整月需要的「一整天完全不出车」的天数（如“每月至少留3个整天休息”→3）；无则 0。\n'
        '- forbidden_categories: 字符串数组。任何时候都禁止承运的货物名称（用货物本身的名字，如“机械设备”“蔬菜”）。\n'
        '- forbidden_regions: 字符串数组。任何时候装货地或卸货地落在该城市/地区就不接（如“惠州”）。\n'
        '- required_region: {"region":字符串,"min_days":整数} 或 null。每月在该地区接货需达到的不同天数。\n'
        '- pickup_max_km: 数字或 null。赴装货的空驶里程上限（公里）。\n'
        '- blackout: 数组，元素 {"region":字符串,"dates":[该月日期数字...]}。指定日期内不去某地区。\n'
        '- dated_single: 数组，元素 {"date":日期数字,"lat":数字,"lng":数字,"wait_minutes":整数,"before_hour":整数或null}。'
        "某天必须亲自到某地点停留办事（盘库 / 清库存 / 对账 / 验收 等），即使不接单也要去；"
        "wait_minutes 取需停留时长，before_hour 是当天必须到达的最晚整点，无明确时限则 null。\n"
        '- dated_route: 数组，元素 {"date":日期数字,"stops":[{"lat":数字,"lng":数字,"wait_minutes":整数,"before_hour":整数或null}...]}。'
        "某天按顺序经过多个地点的赴约/办事路线（如先取物再赴宴），stops 按先后顺序排列。\n"
        "通用规则：\n"
        "1) 日期一律用该月的日数（1-31）。\n"
        "2) 坐标直接取经纬度数字；若某条偏好只说了地点名（如某档口/某仓库/某地区）却没给经纬度，"
        "而另一条偏好给出了同一地点的经纬度，则跨偏好引用那个经纬度填进来。\n"
        "3) 停留时长：‘停一趟，花两小时’→wait_minutes=120；"
        "‘赴宴到下午两点’且需中午前赶到→在该点停留到结束，wait_minutes 约等于(结束-到达)，按 120 估；"
        "凡是‘到点办事/赴宴/盘点’类事件，wait_minutes 必须为正（>0），不能填 0。\n"
        "4) 时刻换算：‘中午十二点前’→before_hour=12；‘下午两点’→14；‘早上六点’→6。\n"
        "只抽取明确写出的约束，不要臆造未提及的规则。"
    )

    def _llm_parse_preferences(
        self, driver_id: str, texts: list[str], rules: DriverRules
    ) -> bool:
        """用大模型把全部可见偏好解析成结构化规则并合并进 rules。

        返回 True 表示 LLM 成功产出结构化结果；False 表示模型不可用/解析失败
        （此时调用方会退回正则解析）。
        """
        if not texts:
            return False
        user = json.dumps({"preferences": texts}, ensure_ascii=False)
        try:
            resp = self._api.model_chat_completion(
                {
                    "messages": [
                        {"role": "system", "content": self._PARSE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    "response_format": {"type": "json_object"},
                }
            )
        except Exception as exc:  # model endpoint unavailable (e.g. offline eval)
            self._logger.info("llm parse unavailable driver_id=%s err=%s", driver_id, exc)
            return False
        data = self._extract_json(resp)
        if data is None:
            self._logger.warning("llm parse: could not extract JSON driver_id=%s", driver_id)
            return False
        try:
            self._merge_llm_rules(rules, data)
        except Exception as exc:
            self._logger.warning("llm parse: merge failed driver_id=%s err=%s", driver_id, exc)
            return False
        return True

    @staticmethod
    def _extract_json(resp: dict[str, Any]) -> dict[str, Any] | None:
        try:
            content = resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None
        if not isinstance(content, str) or not content.strip():
            return None
        text = content.strip()
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

    def _merge_llm_rules(self, rules: DriverRules, data: dict[str, Any]) -> None:
        rest_h = data.get("daily_rest_hours")
        if isinstance(rest_h, (int, float)) and rest_h > 0:
            rules.daily_rest_minutes = max(rules.daily_rest_minutes, int(round(rest_h * 60)))
        rw = data.get("rest_window")
        if isinstance(rw, dict):
            sh, eh = rw.get("start_hour"), rw.get("end_hour")
            if isinstance(sh, (int, float)) and isinstance(eh, (int, float)) and eh > sh:
                rules.rest_window = (int(sh) * 60, int(eh) * 60)
        off = data.get("off_days_min")
        if isinstance(off, (int, float)) and off > 0:
            rules.off_days_min = max(rules.off_days_min, int(off))
        for cat in data.get("forbidden_categories") or []:
            if isinstance(cat, str) and cat.strip():
                rules.forbidden_categories.add(cat.strip())
        for reg in data.get("forbidden_regions") or []:
            if isinstance(reg, str) and reg.strip():
                rules.forbidden_regions.add(reg.strip())
        rr = data.get("required_region")
        if isinstance(rr, dict):
            reg, md = rr.get("region"), rr.get("min_days")
            if isinstance(reg, str) and reg.strip() and isinstance(md, (int, float)) and md > 0:
                rules.required_region = (reg.strip(), int(md))
        pk = data.get("pickup_max_km")
        if isinstance(pk, (int, float)) and pk > 0:
            rules.pickup_max_km = float(pk)
        for bo in data.get("blackout") or []:
            if not isinstance(bo, dict):
                continue
            reg = bo.get("region")
            days = {int(d) - 1 for d in (bo.get("dates") or []) if isinstance(d, (int, float)) and 1 <= d <= 31}
            if isinstance(reg, str) and reg.strip() and days and not any(r == reg.strip() for r, _ in rules.blackout):
                rules.blackout.append((reg.strip(), days))
        for ev in data.get("dated_single") or []:
            single = self._coerce_single(ev)
            if single is not None and not any(e["day"] == single["day"] for e in rules.dated_single):
                rules.dated_single.append(single)
        for ev in data.get("dated_route") or []:
            route = self._coerce_route(ev)
            if route is not None and not any(e["day"] == route["day"] for e in rules.dated_route):
                rules.dated_route.append(route)

    @staticmethod
    def _coerce_before(before_hour: Any) -> int:
        if isinstance(before_hour, (int, float)) and 0 < before_hour <= 24:
            return int(before_hour) * 60
        return DAY_MINUTES

    def _coerce_single(self, ev: Any) -> dict[str, Any] | None:
        if not isinstance(ev, dict):
            return None
        date, lat, lng = ev.get("date"), ev.get("lat"), ev.get("lng")
        if not (isinstance(date, (int, float)) and 1 <= date <= 31):
            return None
        if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
            return None
        wait = ev.get("wait_minutes")
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
        date = ev.get("date")
        if not (isinstance(date, (int, float)) and 1 <= date <= 31):
            return None
        stops: list[dict[str, Any]] = []
        for s in ev.get("stops") or []:
            if not isinstance(s, dict):
                continue
            lat, lng = s.get("lat"), s.get("lng")
            if not (isinstance(lat, (int, float)) and isinstance(lng, (int, float))):
                continue
            wait = s.get("wait_minutes")
            wait = int(wait) if isinstance(wait, (int, float)) and wait >= 0 else 0
            stops.append(
                {
                    "lat": float(lat),
                    "lng": float(lng),
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
        # forbidden category: "X这类活儿...干不了" / "凡是X货源...推掉"
        if ("干不了" in text or "推掉" in text or ("一律不接" in text and "货源" not in text[:0])) and "扣" in text:
            cat = self._parse_forbidden_category(text)
            if cat:
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
        if "号" in text and ("停一趟" in text or "盘库" in text or "清库存" in text or "对清" in text):
            days = self._parse_month_days(text)
            loc = self._match_coords(text, coords)
            if days and loc and not any(e["day"] == days[0] for e in rules.dated_single):
                rules.dated_single.append(
                    {"day": days[0], "lat": loc[0], "lng": loc[1], "min_wait": self._parse_hours_minutes(text) or 120, "before": DAY_MINUTES}
                )
        # dated route: "三月三十一号...先过增城...中午十二点前赶到四会...赴宴到下午两点"
        if "号" in text and ("赴宴" in text or "寿" in text or "赶到" in text) and "增城" in text:
            days = self._parse_month_days(text)
            if days and not any(e["day"] == days[0] for e in rules.dated_route):
                stops = self._parse_route_stops(text, coords)
                if stops:
                    rules.dated_route.append({"day": days[0], "stops": stops})

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
        m = re.search(r"(?:花|停)\s*([零一二两三四五六七八九十\d]+)\s*(?:个)?\s*小时", text)
        if m:
            return _cn_to_int(m.group(1)) * 60
        return None

    @staticmethod
    def _parse_time_window(text: str) -> tuple[int, int] | None:
        # handle "零点...六点" style
        nums = re.findall(r"(零点|凌晨|早上\s*[零一二两三四五六七八九十\d]+\s*点|[零一二两三四五六七八九十\d]+\s*点)", text)
        hours: list[int] = []
        for token in nums:
            if "零点" in token:
                hours.append(0)
                continue
            mm = re.search(r"([零一二两三四五六七八九十\d]+)\s*点", token)
            if mm:
                hours.append(_cn_to_int(mm.group(1)))
        if "零点" in text and len(hours) >= 1:
            end = next((h for h in hours if h > 0), None)
            if end is not None:
                return (0, end * 60)
        if len(hours) >= 2:
            return (hours[0] * 60, hours[1] * 60)
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
    # handle 十/二十/十二 etc.
    if "十" in token:
        parts = token.split("十")
        tens = _CN_DIGITS.get(parts[0], 1) if parts[0] else 1
        ones = _CN_DIGITS.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    total = 0
    for ch in token:
        if ch in _CN_DIGITS:
            total = total * 10 + _CN_DIGITS[ch]
    return total


_EPOCH_FMT = re.compile(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2})")


def _wall_to_min(wall: str) -> int | None:
    m = _EPOCH_FMT.search(wall)
    if not m:
        return None
    _y, _mo, d, hh, mm = (int(x) for x in m.groups())
    return (d - 1) * DAY_MINUTES + hh * 60 + mm
