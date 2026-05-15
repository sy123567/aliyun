"""DP-ORH-MS 决策服务：动态偏好感知的在线滚动时域多目标评分。

依赖 ``simkit.ports.SimulationApiPort``：评测进程会注入具体的环境实现。

主流程（详见 ``docs/06-设计过程思路总文档.md`` 6.1 节）：
1. 读取司机状态（``get_driver_status``）。
2. 检测偏好变化，仅在变化时重新调用 LLM 解析。
3. 同步历史动作到司机记忆（``query_decision_history``）。
4. 查询候选货源并更新热点 + 小时桋。
5. 生成接单/休息/空驶候选 + 自适应权重评分。
6. 选择最高分动作，过 ``action_validator`` 后输出。
7. 任何阶段异常回退到安全休息，不会抛出未捕获异常。
"""

from __future__ import annotations

import logging
import math
from typing import Any

from simkit.ports import SimulationApiPort

from . import action_validator, config, driver_memory, geo_utils, preference_parser, scoring
from .scoring import DecisionContext, ScoredAction

_LOGGER = logging.getLogger("agent.decision_service")

_HISTORY_LOOKBACK_STEPS = config.HISTORY_LOOKBACK_STEPS
_TOP_ORDER_CANDIDATES = config.TOP_ORDER_CANDIDATES
_TOP_REPOSITION_TARGETS = config.TOP_REPOSITION_TARGETS
_MIN_WAIT_FALLBACK_MINUTES = config.MIN_WAIT_FALLBACK_MINUTES
_TOP_LOG_CANDIDATES = 5


class ModelDecisionService:
    """参赛智能体单步决策入口。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = _LOGGER

    # ---------------- 主入口 ----------------

    def decide(self, driver_id: str) -> dict[str, Any]:
        """主决策入口；全过程异常回退到安全休息。"""
        try:
            return self._decide_inner(driver_id)
        except Exception as exc:  # noqa: BLE001 - 最外层兜底，避免评测进程被决策损坏
            self._logger.exception("decide 未捕获异常 driver_id=%s err=%s", driver_id, exc)
            return action_validator.safe_wait(_MIN_WAIT_FALLBACK_MINUTES, note="unhandled_exception")

    def _decide_inner(self, driver_id: str) -> dict[str, Any]:
        try:
            status = self._api.get_driver_status(driver_id)
        except Exception as exc:  # noqa: BLE001 - 状态接口异常时退化为短休息
            self._logger.warning("get_driver_status 失败 driver_id=%s err=%s", driver_id, exc)
            return action_validator.safe_wait(_MIN_WAIT_FALLBACK_MINUTES, note="status_unavailable")

        memory = driver_memory.get_or_create(driver_id)
        self._sync_history(driver_id, memory)

        sim_minutes = int(status.get("simulation_progress_minutes") or 0)
        rules = self._ensure_rules_parsed(driver_id, status, memory, sim_minutes)

        current_lat = float(status.get("current_lat") or 0.0)
        current_lng = float(status.get("current_lng") or 0.0)

        cargo_items = self._safe_query_cargo(driver_id, current_lat, current_lng)
        try:
            status = self._api.get_driver_status(driver_id)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("query_cargo 后刷新 get_driver_status 失败 driver_id=%s err=%s", driver_id, exc)

        sim_minutes = int(status.get("simulation_progress_minutes") or sim_minutes)
        rules = self._ensure_rules_parsed(driver_id, status, memory, sim_minutes)
        current_lat = float(status.get("current_lat") or current_lat)
        current_lng = float(status.get("current_lng") or current_lng)
        truck_length = str(status.get("truck_length") or "")
        memory.last_status_minutes = sim_minutes
        memory.last_lat = current_lat
        memory.last_lng = current_lng

        cost_per_km = float(status.get("cost_per_km") or 1.5)
        ctx = DecisionContext(
            driver_id=driver_id,
            cost_per_km=cost_per_km,
            truck_length=truck_length,
            current_lat=current_lat,
            current_lng=current_lng,
            current_minutes=sim_minutes,
            horizon_minutes=config.AGENT_HORIZON_MINUTES,
        )

        self._update_hotspots(memory, cargo_items, sim_minutes)

        # 自适应权重（文档 8.4 节）：根据月末/夜间/稀缺/违规预警调位
        ctx.weights = scoring.resolve_adaptive_weights(
            rules=rules,
            memory=memory,
            ctx=ctx,
            visible_cargo_count=len(cargo_items),
        )
        ctx.visible_cargo_count = len(cargo_items)

        order_candidates = self._build_order_candidates(cargo_items, rules, memory, ctx)
        has_good_order = any(c.feasible and c.score > 0 for c in order_candidates)

        wait_candidates = self._build_wait_candidates(rules, memory, ctx, has_good_order)
        reposition_candidates = self._build_reposition_candidates(
            cargo_items, rules, memory, ctx, has_good_order
        )

        all_candidates: list[ScoredAction] = []
        all_candidates.extend(order_candidates)
        all_candidates.extend(wait_candidates)
        all_candidates.extend(reposition_candidates)
        feasible = [c for c in all_candidates if c.feasible]

        self._log_top_candidates(driver_id, sim_minutes, len(cargo_items), all_candidates)

        if not feasible:
            filtered_notes = sorted({c.note for c in all_candidates if c.note})
            self._logger.warning(
                "无可行候选 driver_id=%s sim_min=%s items=%s filtered=%s -> safe_wait",
                driver_id,
                sim_minutes,
                len(cargo_items),
                filtered_notes,
            )
            return action_validator.safe_wait(_MIN_WAIT_FALLBACK_MINUTES, note="no_feasible_candidate")

        best = max(feasible, key=lambda c: c.score)

        # R7: LLM辅助关键决策
        if config.LLM_CRITICAL_DECISION_ENABLED and memory.can_call_model(expected_tokens=2000):
            llm_override = self._llm_critical_decision(
                best, feasible, rules, memory, ctx, cargo_items
            )
            if llm_override is not None:
                best = llm_override

        allowed_cargo_ids: set[str] | None = None
        if best.action == "take_order":
            allowed_cargo_ids = {
                str((item.get("cargo") or {}).get("cargo_id", "")).strip()
                for item in cargo_items
            }
            allowed_cargo_ids.discard("")

        try:
            validated = action_validator.validate_action(
                best.as_action_dict(),
                allowed_cargo_ids=allowed_cargo_ids,
            )
        except action_validator.ActionInvalid as exc:
            self._logger.warning(
                "action_validator 拒绝 driver_id=%s action=%s err=%s -> safe_wait",
                driver_id,
                best.as_action_dict(),
                exc,
            )
            return action_validator.safe_wait(_MIN_WAIT_FALLBACK_MINUTES, note=f"invalid_action:{exc}")

        self._logger.info(
            "decision driver_id=%s sim_min=%s items=%s action=%s score=%.2f note=%s token_used=%s",
            driver_id,
            sim_minutes,
            len(cargo_items),
            validated.get("action"),
            best.score,
            best.note,
            memory.token_used,
        )
        return validated

    # ---------------- 历史记忆同步 ----------------

    def _sync_history(self, driver_id: str, memory: driver_memory.DriverMemory) -> None:
        try:
            history = self._api.query_decision_history(driver_id, _HISTORY_LOOKBACK_STEPS)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("query_decision_history 失败 driver_id=%s err=%s", driver_id, exc)
            return
        records = history.get("records") if isinstance(history, dict) else None
        if isinstance(records, list):
            memory.absorb_history_records(records)
            if memory.rules is not None:
                self._update_timed_event_flags(memory, memory.rules, records)

    def _update_timed_event_flags(
        self,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        records: list[dict[str, Any]],
    ) -> None:
        for event in rules.timed_stay_events:
            key = scoring.timed_event_key(event)
            pickup_run = 0
            for record in records:
                action = record.get("action", {}) or {}
                action_name = str(action.get("action", "")).strip().lower()
                result = record.get("result", {}) or {}
                pos_before = record.get("position_before", {}) or {}
                pos_after = record.get("position_after", {}) or {}
                try:
                    before_lat = float(pos_before.get("lat"))
                    before_lng = float(pos_before.get("lng"))
                    after_lat = float(pos_after.get("lat"))
                    after_lng = float(pos_after.get("lng"))
                    step_end = int(result.get("simulation_progress_minutes", 0) or 0)
                    action_exec = int(record.get("action_exec_cost_minutes", 0) or 0)
                except (TypeError, ValueError):
                    continue
                near_pick_after = geo_utils.haversine_km(after_lat, after_lng, event.pickup_lat, event.pickup_lng) <= event.radius_km
                near_pick_before = geo_utils.haversine_km(before_lat, before_lng, event.pickup_lat, event.pickup_lng) <= event.radius_km
                near_home_after = geo_utils.haversine_km(after_lat, after_lng, event.home_lat, event.home_lng) <= event.radius_km
                if step_end >= event.start_minutes and action_name == "wait" and near_pick_after:
                    pickup_run += action_exec
                    if pickup_run >= event.pickup_stay_minutes:
                        memory.timed_event_flags.add(f"{key}:pickup")
                elif not (near_pick_before and near_pick_after):
                    pickup_run = 0
                if step_end >= event.start_minutes and near_home_after:
                    memory.timed_event_flags.add(f"{key}:home")

    # ---------------- 偏好解析（带缓存） ----------------

    def _ensure_rules_parsed(
        self,
        driver_id: str,
        status: dict[str, Any],
        memory: driver_memory.DriverMemory,
        sim_minutes: int,
    ) -> preference_parser.ParsedRules:
        preferences = status.get("preferences") or []
        signature = preference_parser.signature_of(preferences)

        # R7-T5: 仅当偏好签名变化（仿真器暴露新偏好）时才重新解析
        force_reparse = False
        if config.LLM_DAILY_REPARSE_ENABLED and memory.rules is not None:
            current_day = sim_minutes // 1440
            last_parse_day = memory.preference_state.last_parse_time_minutes // 1440
            if current_day > last_parse_day and signature != memory.rules_signature and memory.can_call_model(expected_tokens=3000):
                force_reparse = True
                self._logger.info(
                    "R7-T5 签名变化重解析偏好 driver_id=%s day=%s->%s sig=%s->%s",
                    driver_id, last_parse_day, current_day,
                    memory.rules_signature[:16], signature[:16],
                )

        if memory.rules is not None and memory.rules_signature == signature and not force_reparse:
            return memory.rules  # type: ignore[return-value]

        # 仅在 token 预算允许时交由 LLM 主解析；否则走正则安全网
        llm_caller = None
        if memory.can_call_model(expected_tokens=2000):
            llm_caller = self._make_llm_caller(driver_id, memory)
        else:
            self._logger.warning(
                "token 预算接近上限，偏好解析降级为纯正则 driver_id=%s token_used=%s",
                driver_id,
                memory.token_used,
            )

        rules = preference_parser.parse_preferences(preferences, llm_caller=llm_caller)
        # 持久化保留：把先前已解析的高额规则（家事 / 熟货）合并到本次结果。
        # 仿真器按墙钟隐藏偏好（如家事仅 3/10–3/13 可见），但 agent 需要提前对位。
        prior_rules = memory.rules
        if prior_rules is not None:
            seen_keys = {(e.start_minutes, round(e.pickup_lat, 4), round(e.home_lat, 4)) for e in rules.timed_stay_events}
            for ev in prior_rules.timed_stay_events:
                key = (ev.start_minutes, round(ev.pickup_lat, 4), round(ev.home_lat, 4))
                if key in seen_keys:
                    continue
                if ev.stay_until_minutes <= sim_minutes:
                    continue  # 已结束的事件不再保留
                rules.timed_stay_events.append(ev)
            seen_cargo = {r.cargo_id for r in rules.preferred_cargo}
            for pc in prior_rules.preferred_cargo:
                if pc.cargo_id in seen_cargo:
                    continue
                rules.preferred_cargo.append(pc)
                if pc.cargo_id not in rules.preferred_cargo_ids:
                    rules.preferred_cargo_ids.append(pc.cargo_id)
        memory.rules = rules
        memory.rules_signature = signature
        memory.record_preference_change(
            new_signature=signature,
            sim_minutes=sim_minutes,
            parsed_by_llm=rules.parsed_by_llm,
            parsed_by_regex=rules.parsed_by_regex,
            parse_failure_count=rules.parse_failure_count,
        )
        self._logger.info(
            "偏好解析完成 driver_id=%s total=%s llm=%s regex=%s failed=%s changes=%s",
            driver_id,
            len(rules.raw_preferences),
            rules.parsed_by_llm,
            rules.parsed_by_regex,
            rules.parse_failure_count,
            len(memory.preference_state.dynamic_changes),
        )
        return rules

    def _make_llm_caller(self, driver_id: str, memory: driver_memory.DriverMemory):
        def _caller(payload: dict[str, Any]) -> dict[str, Any]:
            try:
                resp = self._api.model_chat_completion(payload)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("model_chat_completion 失败 driver_id=%s err=%s", driver_id, exc)
                return {}
            if not isinstance(resp, dict):
                return {}
            usage = resp.get("usage")
            if isinstance(usage, dict):
                memory.update_token(int(usage.get("total_tokens", 0)))
            return resp

        return _caller

    def _log_top_candidates(
        self,
        driver_id: str,
        sim_minutes: int,
        items_count: int,
        candidates: list[ScoredAction],
    ) -> None:
        """依文档 12.3 节要求记录 Top 候选评分明细供人工复核。"""
        if not candidates:
            return
        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:_TOP_LOG_CANDIDATES]
        for rank, cand in enumerate(ranked, start=1):
            top_breakdown = sorted(
                cand.breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True
            )[:5]
            self._logger.debug(
                "top%s driver=%s sim_min=%s items=%s action=%s feasible=%s score=%.2f params=%s breakdown=%s note=%s",
                rank,
                driver_id,
                sim_minutes,
                items_count,
                cand.action,
                cand.feasible,
                cand.score,
                cand.params,
                top_breakdown,
                cand.note,
            )

    # ---------------- 货源查询 ----------------

    def _safe_query_cargo(self, driver_id: str, lat: float, lng: float) -> list[dict[str, Any]]:
        try:
            resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("query_cargo 失败 driver_id=%s err=%s", driver_id, exc)
            return []
        items = resp.get("items") if isinstance(resp, dict) else None
        return list(items) if isinstance(items, list) else []

    def _update_hotspots(
        self,
        memory: driver_memory.DriverMemory,
        items: list[dict[str, Any]],
        sim_minutes: int,
    ) -> None:
        for item in items:
            cargo = item.get("cargo") or {}
            start = cargo.get("start") or {}
            try:
                lat = float(start["lat"])
                lng = float(start["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            price = float(cargo.get("price") or 0.0)
            minutes = int(cargo.get("cost_time_minutes") or 0)
            memory.update_hotspot(lat, lng, price, max(1, minutes), sim_minutes)

    # ---------------- 候选生成 ----------------

    def _build_order_candidates(
        self,
        items: list[dict[str, Any]],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
    ) -> list[ScoredAction]:
        candidates: list[ScoredAction] = []
        preferred_ids = set(rules.preferred_cargo_ids)
        selected = list(items[:_TOP_ORDER_CANDIDATES])
        if preferred_ids:
            selected_ids = {
                str(((item.get("cargo") or {}).get("cargo_id", ""))).strip()
                for item in selected
            }
            for item in items[_TOP_ORDER_CANDIDATES:]:
                cargo_id = str(((item.get("cargo") or {}).get("cargo_id", ""))).strip()
                if cargo_id in preferred_ids and cargo_id not in selected_ids:
                    selected.append(item)
                    selected_ids.add(cargo_id)
        for item in selected:
            scored = scoring.score_take_order(item, rules, memory, ctx)
            candidates.append(scored)
        return candidates

    def _build_wait_candidates(
        self,
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        has_good_order: bool,
    ) -> list[ScoredAction]:
        durations = scoring.build_wait_durations(rules, ctx, memory)
        candidates = [
            scoring.score_wait(d, rules, memory, ctx, has_good_order=has_good_order) for d in durations
        ]
        return candidates

    def _build_reposition_candidates(
        self,
        items: list[dict[str, Any]],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        has_good_order: bool,
    ) -> list[ScoredAction]:
        # 高优先级偏好目标点：事件接人点 / 老家 / 熟货 / 必到 / 回家
        priority_targets: list[tuple[float, float]] = []
        for event in rules.timed_stay_events:
            phase = scoring.timed_event_phase(event, memory, ctx)
            if phase in {"approaching", "pickup", "late_pickup"}:
                priority_targets.append((event.pickup_lat, event.pickup_lng))
            elif phase in {"home", "late_home"}:
                priority_targets.append((event.home_lat, event.home_lng))
        for preferred in rules.preferred_cargo:
            target = scoring.preferred_cargo_target(preferred)
            if target is not None and scoring.preferred_cargo_preposition_ready(preferred, ctx.current_minutes):
                priority_targets.append(target)
        for must in rules.must_visit:
            priority_targets.append((must.lat, must.lng))
        if rules.home_rule is not None:
            priority_targets.append((rules.home_rule.lat, rules.home_rule.lng))

        # 当前位置已有可接好单时，避免无意义远距离空驶。
        targets: list[tuple[float, float]] = list(priority_targets)
        if not has_good_order:
            targets.extend(self._reposition_targets_from_cargo(items, ctx))
            targets.extend(self._reposition_targets_from_hotspots(memory, ctx))
        # 去重
        seen: set[tuple[float, float]] = set()
        deduped: list[tuple[float, float]] = []
        for t in targets:
            key = (round(t[0], 3), round(t[1], 3))
            if key in seen:
                continue
            if math.hypot(t[0] - ctx.current_lat, t[1] - ctx.current_lng) < 0.01:
                continue
            seen.add(key)
            deduped.append(t)
        return [
            scoring.score_reposition(t[0], t[1], rules, memory, ctx)
            for t in deduped[:_TOP_REPOSITION_TARGETS]
        ]

    def _reposition_targets_from_cargo(
        self,
        items: list[dict[str, Any]],
        ctx: DecisionContext,
    ) -> list[tuple[float, float]]:
        # 取价格-时间比 Top 货源的装货点作为空驶候选；过滤过近的点
        scored: list[tuple[float, float, float]] = []
        for item in items[:_TOP_ORDER_CANDIDATES]:
            cargo = item.get("cargo") or {}
            start = cargo.get("start") or {}
            try:
                lat = float(start["lat"])
                lng = float(start["lng"])
            except (KeyError, TypeError, ValueError):
                continue
            price = float(cargo.get("price") or 0.0)
            minutes = max(1, int(cargo.get("cost_time_minutes") or 60))
            ratio = price / minutes
            distance_km = float(item.get("distance_km") or 0.0)
            if distance_km < 5:
                continue
            scored.append((ratio, lat, lng))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(lat, lng) for _, lat, lng in scored[:_TOP_REPOSITION_TARGETS]]

    def _reposition_targets_from_hotspots(
        self,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
    ) -> list[tuple[float, float]]:
        scored: list[tuple[float, tuple[float, float]]] = []
        for key, cell in memory.hotspots.items():
            if cell.samples < 2:
                continue
            lat, lng = geo_utils.grid_center(key)
            if math.hypot(lat - ctx.current_lat, lng - ctx.current_lng) < 0.05:
                continue
            avg_yield = cell.sum_price_per_minute / max(1, cell.samples)
            scored.append((avg_yield, (lat, lng)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [coord for _, coord in scored[:_TOP_REPOSITION_TARGETS]]


    # ---------------- R7: LLM 辅助关键决策 ----------------

    def _llm_critical_decision(
        self,
        best: ScoredAction,
        feasible: list[ScoredAction],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        cargo_items: list[dict[str, Any]],
    ) -> ScoredAction | None:
        """在高风险决策点调用LLM辅助决策，返回覆盖选项或None保持原选择。"""
        if not memory.can_call_model(expected_tokens=2000):
            return None
        llm_call_count = getattr(memory, "llm_critical_count", 0)
        if llm_call_count >= config.LLM_CRITICAL_DECISION_MAX_PER_DRIVER:
            return None

        # T1: 事件临近审核
        override = self._t1_event_proximity_check(best, feasible, rules, memory, ctx, cargo_items)
        if override is not None:
            return override

        # T2: 月末休息deficit审核
        override = self._t2_rest_deficit_check(best, feasible, rules, memory, ctx)
        if override is not None:
            return override

        # T3: 回家时间紧迫审核
        override = self._t3_home_urgency_check(best, feasible, rules, memory, ctx, cargo_items)
        if override is not None:
            return override

        # T4: 评分接近仲裁
        override = self._t4_score_arbitration(best, feasible, rules, memory, ctx, cargo_items)
        if override is not None:
            return override

        return None

    def _call_llm_for_decision(
        self,
        driver_id: str,
        memory: driver_memory.DriverMemory,
        prompt: str,
    ) -> str:
        """调用LLM获取决策建议，返回响应文本。"""
        llm_caller = self._make_llm_caller(driver_id, memory)
        payload = {
            "messages": [
                {"role": "system", "content": "你是货运司机决策助手。请直接给出决策结论，格式为ACCEPT或REJECT或REST或WORK，然后用一句话说明理由。不要使用思考标签。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        try:
            resp = llm_caller(payload)
        except Exception:  # noqa: BLE001
            return ""
        if not isinstance(resp, dict):
            return ""
        choices = resp.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        memory.llm_critical_count = getattr(memory, "llm_critical_count", 0) + 1
        return str(content).strip()

    def _t1_event_proximity_check(
        self,
        best: ScoredAction,
        feasible: list[ScoredAction],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        cargo_items: list[dict[str, Any]],
    ) -> ScoredAction | None:
        """T1: 距事件≤48h且best是take_order时，让LLM审核是否安全。"""
        if best.action != "take_order":
            return None
        if not rules.timed_stay_events:
            return None

        for event in rules.timed_stay_events:
            time_to_event = event.start_minutes - ctx.current_minutes
            if not (0 < time_to_event <= config.LLM_EVENT_HORIZON_MINUTES):
                continue

            cargo_id = best.params.get("cargo_id", "")
            cargo_info = self._find_cargo_info(cargo_items, cargo_id)
            if not cargo_info:
                continue

            finish_min = int(cargo_info.get("cost_time_minutes", 0) or 0) + ctx.current_minutes
            end_lat = float(cargo_info.get("end_lat", 0))
            end_lng = float(cargo_info.get("end_lng", 0))
            dist_to_home = geo_utils.haversine_km(end_lat, end_lng, event.home_lat, event.home_lng)
            hours_to_event = time_to_event / 60.0

            prompt = (
                f"当前时间：仿真第{ctx.current_minutes // 1440 + 1}天 {geo_utils.hour_of_day(ctx.current_minutes)}:{ctx.current_minutes % 60:02d}\n"
                f"司机位置：({ctx.current_lat:.2f}, {ctx.current_lng:.2f})\n"
                f"即将到来的家事/约定事件：{hours_to_event:.1f}小时后开始，需在({event.home_lat:.2f}, {event.home_lng:.2f})附近\n"
                f"推荐接单：cargo_id={cargo_id}，预计用时{cargo_info.get('cost_time_minutes', 0)}分钟\n"
                f"完单后距家事地点：{dist_to_home:.0f}km\n"
                f"事件缺席罚金：每分钟{event.absence_penalty_per_minute}元\n\n"
                f"问题：接这个单是否安全？会不会影响按时参加事件？\n"
                f"请回答ACCEPT（接单）或REJECT（拒绝改为等待/回家）"
            )

            response = self._call_llm_for_decision(ctx.driver_id, memory, prompt)
            self._logger.info(
                "R7-T1 LLM事件审核 driver=%s event_in=%.1fh cargo=%s dist_home=%.0fkm response=%s",
                ctx.driver_id, hours_to_event, cargo_id, dist_to_home, response[:80],
            )

            if "REJECT" in response.upper():
                best_wait = max(
                    (c for c in feasible if c.action == "wait" and c.feasible),
                    key=lambda c: c.score,
                    default=None,
                )
                if best_wait:
                    best_wait.note = f"llm_event_reject({best_wait.note})"
                    return best_wait

        return None

    def _t2_rest_deficit_check(
        self,
        best: ScoredAction,
        feasible: list[ScoredAction],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
    ) -> ScoredAction | None:
        """T2: 月度休息日deficit≥1且day≥18时，让LLM决定接单还是休息。"""
        if best.action != "take_order":
            return None
        if rules.monthly_day_off is None:
            return None

        sim_day = ctx.current_minutes // 1440
        if sim_day < config.LLM_REST_DEFICIT_DAY:
            return None

        required = rules.monthly_day_off.required_days
        active_days_so_far = len(memory.daily_active)
        remaining_days = 31 - sim_day
        rest_days = sim_day - active_days_so_far  # approximate rest days so far
        current_day_date = geo_utils.date_str(ctx.current_minutes)
        current_day_active = current_day_date in memory.daily_active
        deficit = required - rest_days

        if deficit < 1 or current_day_active:
            return None

        penalty_amount = rules.monthly_day_off.penalty_amount or 3000.0
        prompt = (
            f"今天是仿真第{sim_day + 1}天（还剩{remaining_days}天）\n"
            f"月度休息日要求：{required}天，已休{rest_days}天，还差{deficit}天\n"
            f"今天尚未接单（休息中）\n"
            f"每缺少1天休息日罚金：¥{penalty_amount}\n"
            f"推荐接单评分：{best.score:.0f}分\n\n"
            f"问题：今天应该继续休息还是接单工作？考虑到剩余{remaining_days}天内还需要休息{deficit}天。\n"
            f"请回答WORK（接单）或REST（今天休息）"
        )

        response = self._call_llm_for_decision(ctx.driver_id, memory, prompt)
        self._logger.info(
            "R7-T2 LLM休息决策 driver=%s day=%s deficit=%s response=%s",
            ctx.driver_id, sim_day, deficit, response[:80],
        )

        if "REST" in response.upper():
            best_wait = max(
                (c for c in feasible if c.action == "wait" and c.feasible),
                key=lambda c: c.score,
                default=None,
            )
            if best_wait:
                best_wait.note = f"llm_rest_decision({best_wait.note})"
                return best_wait

        return None

    def _t3_home_urgency_check(
        self,
        best: ScoredAction,
        feasible: list[ScoredAction],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        cargo_items: list[dict[str, Any]],
    ) -> ScoredAction | None:
        """T3: 距回家截止≤4h且best是take_order时审核。"""
        if best.action != "take_order":
            return None
        if rules.home_rule is None:
            return None

        hr = rules.home_rule
        home_by_min = hr.home_by_hour * 60
        current_mod = ctx.current_minutes % 1440
        time_until_home = home_by_min - current_mod
        if time_until_home < 0:
            time_until_home += 1440

        if time_until_home > config.LLM_HOME_CRITICAL_HOURS * 60:
            return None

        dist_to_home = geo_utils.haversine_km(ctx.current_lat, ctx.current_lng, hr.lat, hr.lng)
        if dist_to_home <= hr.radius_km:
            return None

        cargo_id = best.params.get("cargo_id", "")
        cargo_info = self._find_cargo_info(cargo_items, cargo_id)
        if not cargo_info:
            return None

        end_lat = float(cargo_info.get("end_lat", 0))
        end_lng = float(cargo_info.get("end_lng", 0))
        dist_end_to_home = geo_utils.haversine_km(end_lat, end_lng, hr.lat, hr.lng)
        cost_time = int(cargo_info.get("cost_time_minutes", 0) or 0)
        travel_home_min = dist_end_to_home / (config.DEFAULT_REPOSITION_SPEED_KMH / 60.0)

        prompt = (
            f"当前时间：{geo_utils.hour_of_day(ctx.current_minutes)}:{ctx.current_minutes % 60:02d}\n"
            f"必须{hr.home_by_hour}:00前到家\n"
            f"距到家截止还有{time_until_home}分钟\n"
            f"当前距家{dist_to_home:.0f}km\n"
            f"推荐接单用时{cost_time}分钟，完单后距家{dist_end_to_home:.0f}km（约{travel_home_min:.0f}分钟车程）\n"
            f"接单后总用时预估：{cost_time + travel_home_min:.0f}分钟\n"
            f"回家违规罚金：¥{hr.penalty_amount}\n\n"
            f"问题：接单后还能按时到家吗？\n"
            f"请回答ACCEPT或REJECT"
        )

        response = self._call_llm_for_decision(ctx.driver_id, memory, prompt)
        self._logger.info(
            "R7-T3 LLM回家审核 driver=%s time_left=%dmin dist=%.0fkm response=%s",
            ctx.driver_id, time_until_home, dist_to_home, response[:80],
        )

        if "REJECT" in response.upper():
            # 优先选reposition回家，其次wait
            best_repo = max(
                (c for c in feasible if c.action == "reposition" and c.feasible),
                key=lambda c: c.score,
                default=None,
            )
            if best_repo and best_repo.score > 0:
                best_repo.note = f"llm_home_reject({best_repo.note})"
                return best_repo
            best_wait = max(
                (c for c in feasible if c.action == "wait" and c.feasible),
                key=lambda c: c.score,
                default=None,
            )
            if best_wait:
                best_wait.note = f"llm_home_reject({best_wait.note})"
                return best_wait

        return None

    def _t4_score_arbitration(
        self,
        best: ScoredAction,
        feasible: list[ScoredAction],
        rules: preference_parser.ParsedRules,
        memory: driver_memory.DriverMemory,
        ctx: DecisionContext,
        cargo_items: list[dict[str, Any]],
    ) -> ScoredAction | None:
        """T4: top-1和top-2评分接近且动作类型不同时，让LLM仲裁。"""
        if len(feasible) < 2:
            return None

        ranked = sorted(feasible, key=lambda c: c.score, reverse=True)
        top1, top2 = ranked[0], ranked[1]

        if top1.action == top2.action:
            return None
        if top1.score <= 0:
            return None

        gap = abs(top1.score - top2.score)
        if gap / max(abs(top1.score), 1.0) > config.LLM_SCORE_CLOSE_RATIO:
            return None

        sim_day = ctx.current_minutes // 1440
        active_days = len(memory.daily_active)

        def _describe(c: ScoredAction) -> str:
            if c.action == "take_order":
                cid = c.params.get("cargo_id", "?")
                info = self._find_cargo_info(cargo_items, str(cid))
                if info:
                    return f"接单cargo={cid}（用时{info.get('cost_time_minutes',0)}min, 距离{info.get('distance_km',0):.0f}km）评分{c.score:.0f}"
                return f"接单cargo={cid} 评分{c.score:.0f}"
            elif c.action == "wait":
                return f"等待{c.params.get('duration_minutes',0)}分钟 评分{c.score:.0f}"
            else:
                return f"空驶到({c.params.get('latitude',0):.2f},{c.params.get('longitude',0):.2f}) 评分{c.score:.0f}"

        prompt = (
            f"两个选择评分非常接近（差距{gap:.0f}，比例{gap / max(abs(top1.score), 1.0):.1%}）：\n"
            f"选项A: {_describe(top1)}\n"
            f"选项B: {_describe(top2)}\n\n"
            f"上下文：第{sim_day + 1}天，已活跃{active_days}天\n"
            f"请选择更优方案：A或B"
        )

        response = self._call_llm_for_decision(ctx.driver_id, memory, prompt)
        self._logger.info(
            "R7-T4 LLM仲裁 driver=%s A=%s(%.0f) B=%s(%.0f) response=%s",
            ctx.driver_id, top1.action, top1.score, top2.action, top2.score, response[:80],
        )

        if "B" in response.upper() and "A" not in response.upper()[:response.upper().find("B")] if "B" in response.upper() else False:
            top2.note = f"llm_arbitration_B({top2.note})"
            return top2

        return None

    def _find_cargo_info(
        self,
        cargo_items: list[dict[str, Any]],
        cargo_id: str,
    ) -> dict[str, Any] | None:
        """从cargo_items中查找指定cargo_id的信息。"""
        for item in cargo_items:
            cargo = item.get("cargo") or {}
            cid = str(cargo.get("cargo_id", "")).strip()
            if cid == cargo_id:
                end = cargo.get("end") or {}
                return {
                    "cost_time_minutes": cargo.get("cost_time_minutes", 0),
                    "distance_km": item.get("distance_km", 0),
                    "end_lat": end.get("lat", 0),
                    "end_lng": end.get("lng", 0),
                    "price": cargo.get("price", 0),
                }
        return None


# 列出可供外部导出的名字，保证 ``from agent.model_decision_service import *`` 不会泄露内部状态。
__all__ = ["ModelDecisionService"]
