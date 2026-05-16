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

import json
import logging
import math
import re
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

_DECISION_SYSTEM_PROMPT = (
    "你是货运调度优化AI。目标：最大化司机31天月度净收入（毛收入-偏好违规罚金）。\n"
    "评分系统已综合计算收入、成本、偏好罚分等因素，分数越高越优。\n"
    "决策原则：\n"
    "1. 通常选择评分最高的候选（编号1），除非有明确理由偏离\n"
    "2. 当多个候选分数接近时（差距<20%），优先选偏好违规风险更低的\n"
    "3. 月底休息日缺口>0时，优先安排休息（选[等]）\n"
    "4. 禁行时段内优先等待，避免高额罚金\n"
    "5. 不要选分数远低于最高分的候选\n"
    "6. 关注当日策略建议，结合整体收入和罚分情况做出最优选择\n"
    "7. 高价值熟货和家事约定是最高优先级，务必确保不错过\n"
    "仅回复最优候选编号（如：2），不要解释。"
)

_STRATEGY_SYSTEM_PROMPT = (
    "你是货运调度策略规划AI。核心目标：最大化司机31天月度净收入（毛收入-罚金）。\n"
    "重要原则：\n"
    "1. 收入优先：大多数情况应选income或balanced，只要还有工作时间就应积极接单\n"
    "2. rest_today仅在满足以下条件时设true：月休息日缺口>0且剩余天数<=缺口天数*2\n"
    "3. penalty_control仅在罚分已超5000元或特定规则即将触发大额罚款时使用\n"
    "4. 300000元是理论上限而非必须达成的目标，日均收入9000-12000元属正常水平\n"
    "5. weight_adjustments保持在0.8-1.5范围内，避免极端调整\n"
    "输出严格JSON格式，禁止markdown和多余文本。\n"
    "Schema:\n"
    '{"priority": "income|balanced|penalty_control|rest",'
    ' "rest_today": true/false,'
    ' "reason": "简短原因",'
    ' "income_target_today": 建议今日目标收入(元),'
    ' "top_risks": ["风险1", "风险2"],'
    ' "weight_adjustments": {"income": 1.0, "preference_risk": 1.0, "time_cost": 1.0}}'
)


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

        # LLM 每日策略规划：在每天首次决策时生成策略，影响权重调整
        strategy = self._ensure_daily_strategy(driver_id, memory, rules, ctx)
        if strategy:
            ctx.weights = scoring.apply_strategy_weights(ctx.weights, strategy)

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

        # R10: LLM 辅助决策——当有多个可行候选时让 LLM 做最终选择
        if len(feasible) > 1:
            llm_choice = self._llm_select_action(driver_id, memory, rules, ctx, feasible)
            if llm_choice is not None:
                best = llm_choice

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

    # ---------------- LLM 每日策略规划 ----------------

    def _ensure_daily_strategy(
        self,
        driver_id: str,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        ctx: DecisionContext,
    ) -> dict[str, Any] | None:
        """每日首次决策时调用 LLM 生成当日策略；缓存在 memory 中避免重复调用。"""
        sim_day = ctx.current_minutes // 1440
        if sim_day == memory.last_strategy_day:
            return memory.daily_strategy.get(sim_day)

        if not memory.can_call_model(expected_tokens=config.LLM_STRATEGY_EXPECTED_TOKENS):
            return None

        strategy = self._generate_daily_strategy(driver_id, memory, rules, ctx)
        if strategy is not None:
            memory.daily_strategy[sim_day] = strategy
            memory.last_strategy_day = sim_day
            self._logger.info(
                "LLM每日策略 driver=%s day=%d priority=%s rest=%s reason=%s",
                driver_id, sim_day + 1,
                strategy.get("priority", "?"),
                strategy.get("rest_today", "?"),
                str(strategy.get("reason", ""))[:60],
            )
        else:
            memory.last_strategy_day = sim_day
        return strategy

    def _generate_daily_strategy(
        self,
        driver_id: str,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        ctx: DecisionContext,
    ) -> dict[str, Any] | None:
        """调用 LLM 生成当日策略。"""
        sim_day = ctx.current_minutes // 1440
        days_remaining = max(0, config.AGENT_HORIZON_DAYS - 1 - sim_day)
        active_days = memory.days_active_count()
        rest_days_done = max(0, sim_day - active_days)

        parts: list[str] = []
        parts.append(f"{driver_id} 第{sim_day + 1}/31天 剩{days_remaining}天")
        parts.append(
            f"累计:毛收入{memory.total_gross_income:.0f}元 "
            f"完成{memory.total_completed_orders}单 "
            f"空驶{memory.total_deadhead_km:.0f}km"
        )

        avg_daily = memory.total_gross_income / max(1, active_days)
        parts.append(
            f"日均毛收入:{avg_daily:.0f}元 "
            f"正常日均水平:9000-12000元"
        )

        penalty_info = []
        for rule_id, amount in sorted(memory.preference_penalty_accum.items(), key=lambda kv: kv[1], reverse=True)[:5]:
            penalty_info.append(f"{rule_id}:{amount:.0f}元")
        if penalty_info:
            parts.append(f"累计罚分:{','.join(penalty_info)}")
        total_penalty = sum(memory.preference_penalty_accum.values())
        parts.append(f"总罚分:{total_penalty:.0f}元")

        if rules.monthly_day_off is not None:
            req = rules.monthly_day_off.required_days or 0
            deficit = max(0, req - rest_days_done)
            parts.append(
                f"月休息:需{req}天已{rest_days_done}天缺{deficit}天 "
                f"活跃{active_days}天"
            )

        for rest in rules.rest_rules:
            parts.append(f"每日休息要求:{rest.required_minutes}分钟 罚{rest.penalty_amount}元/天")

        no_drive_info = []
        for w in rules.no_drive_windows:
            no_drive_info.append(f"{w.start_minute // 60}:00-{w.end_minute // 60}:00罚{w.penalty_amount}元/天")
        if no_drive_info:
            parts.append(f"禁行时段:{','.join(no_drive_info[:3])}")

        for event in rules.timed_stay_events:
            phase = scoring.timed_event_phase(event, memory, ctx)
            if phase != "done":
                event_day = event.start_minutes // 1440
                parts.append(
                    f"家事约定:第{event_day + 1}天开始 "
                    f"当前阶段={phase} 罚{event.absence_penalty_per_minute}元/分钟"
                )

        for limit in rules.distance_limits:
            if limit.kind == "monthly_deadhead":
                over = max(0, memory.total_deadhead_km - limit.max_km)
                parts.append(f"月空驶限制:{limit.max_km}km 已用{memory.total_deadhead_km:.0f}km 超额{over:.0f}km")

        hour_values = []
        for h in range(24):
            v = memory.hour_pattern_value(h)
            if v > 0:
                hour_values.append(f"{h}时={v:.1f}")
        if hour_values:
            parts.append(f"高收益时段:{','.join(hour_values[:6])}")

        parts.append("请分析以上信息，为今天制定最优策略。注意：多数情况应选income或balanced积极接单。")

        caller = self._make_llm_caller(driver_id, memory)
        try:
            resp = caller({
                "messages": [
                    {"role": "system", "content": _STRATEGY_SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(parts)},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 300,
                "enable_thinking": False,
            })
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("LLM策略规划异常 driver=%s err=%s", driver_id, exc)
            return None

        if not resp:
            return None
        choices = resp.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = str(msg.get("content", "")).strip()
        try:
            strategy = json.loads(content)
            if not isinstance(strategy, dict):
                return None
            if "priority" not in strategy:
                return None
            return strategy
        except (json.JSONDecodeError, TypeError):
            self._logger.warning("LLM策略解析失败 driver=%s resp=%s", driver_id, content[:100])
            return None

    # ---------------- LLM 辅助决策（R10 增强） ----------------

    def _is_critical_situation(
        self,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        ctx: DecisionContext,
    ) -> bool:
        """判断是否处于关键决策时刻，需要强制 LLM 参与。"""
        sim_day = ctx.current_minutes // 1440
        days_remaining = max(0, config.AGENT_HORIZON_DAYS - 1 - sim_day)
        if days_remaining <= config.LLM_MONTH_END_CRITICAL_DAYS:
            return True
        total_penalty = sum(memory.preference_penalty_accum.values())
        if total_penalty >= config.LLM_CRITICAL_PENALTY_THRESHOLD:
            return True
        if rules.monthly_day_off is not None:
            active_days = memory.days_active_count()
            rest_done = max(0, sim_day - active_days)
            deficit = max(0, rules.monthly_day_off.required_days - rest_done)
            if deficit > 0 and days_remaining <= deficit * 3:
                return True
        for event in rules.timed_stay_events:
            phase = scoring.timed_event_phase(event, memory, ctx)
            if phase in {"early_approach", "approaching", "pickup", "late_pickup", "home", "late_home"}:
                return True
        return False

    def _llm_select_action(
        self,
        driver_id: str,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        ctx: DecisionContext,
        feasible: list[ScoredAction],
    ) -> ScoredAction | None:
        """使用 LLM 从可行候选中选择最优动作；失败时返回 None 回退评分系统。"""
        if not memory.can_call_model(expected_tokens=5000):
            return None

        ranked = sorted(feasible, key=lambda c: c.score, reverse=True)
        top_n = min(5, len(ranked))
        candidates = ranked[:top_n]

        critical = self._is_critical_situation(memory, rules, ctx)

        # 评分系统已有明显最优时跳过 LLM（节省 token），但关键时刻不跳过
        if not critical and top_n >= 2 and candidates[0].score > 0:
            if candidates[0].score > candidates[1].score * config.LLM_SKIP_SCORE_RATIO:
                return None

        prompt = self._build_decision_prompt(driver_id, memory, rules, ctx, candidates)
        caller = self._make_llm_caller(driver_id, memory)
        try:
            resp = caller({
                "messages": [
                    {"role": "system", "content": _DECISION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 100,
                "enable_thinking": False,
            })
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("LLM决策调用异常 driver=%s err=%s", driver_id, exc)
            return None

        if not resp:
            return None
        choices = resp.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = str(msg.get("content", "")).strip()

        idx = self._parse_choice(content, top_n)
        if idx is not None:
            selected = candidates[idx]
            top_score = candidates[0].score
            if top_score > 0 and selected.score < top_score * config.LLM_SCORE_FLOOR_RATIO:
                self._logger.info(
                    "LLM决策被拒(分数过低) driver=%s 选=%d/%d score=%.0f < %.0f%%*top=%.0f",
                    driver_id, idx + 1, top_n, selected.score,
                    config.LLM_SCORE_FLOOR_RATIO * 100, top_score,
                )
                return None
            self._logger.info(
                "LLM决策 driver=%s 选=%d/%d action=%s score=%.0f (top1=%s %.0f) critical=%s",
                driver_id, idx + 1, top_n, selected.action, selected.score,
                candidates[0].action, candidates[0].score, critical,
            )
            return selected

        self._logger.warning("LLM决策解析失败 driver=%s resp=%s", driver_id, content[:80])
        return None

    def _build_decision_prompt(
        self,
        driver_id: str,
        memory: driver_memory.DriverMemory,
        rules: preference_parser.ParsedRules,
        ctx: DecisionContext,
        candidates: list[ScoredAction],
    ) -> str:
        sim_day = ctx.current_minutes // 1440
        hour = geo_utils.hour_of_day(ctx.current_minutes)
        md = geo_utils.minute_of_day(ctx.current_minutes)
        minute = md % 60
        days_remaining = max(0, config.AGENT_HORIZON_DAYS - 1 - sim_day)

        parts: list[str] = []
        parts.append(f"{driver_id} 第{sim_day + 1}/31天 {hour:02d}:{minute:02d} 剩{days_remaining}天")
        parts.append(
            f"收入:{memory.total_gross_income:.0f}元 "
            f"空驶:{memory.total_deadhead_km:.0f}km "
            f"{memory.total_completed_orders}单"
        )

        # 收入轨迹分析
        avg_daily = memory.total_gross_income / max(1, memory.days_active_count()) if memory.days_active_count() > 0 else 0
        target_pace = 300000 / 31
        actual_pace = memory.total_gross_income / max(1, sim_day + 1)
        pace_diff = actual_pace - target_pace
        parts.append(f"日均:{avg_daily:.0f}元 节奏:{'+' if pace_diff >= 0 else ''}{pace_diff:.0f}元/天")

        # 罚分累计
        total_penalty = sum(memory.preference_penalty_accum.values())
        if total_penalty > 0:
            penalty_parts = []
            for rule_id, amount in sorted(
                memory.preference_penalty_accum.items(),
                key=lambda kv: kv[1], reverse=True,
            )[:3]:
                penalty_parts.append(f"{rule_id}:{amount:.0f}")
            parts.append(f"罚分:{total_penalty:.0f}元({','.join(penalty_parts)})")

        if rules.monthly_day_off is not None:
            req = rules.monthly_day_off.required_days or 0
            active_days = memory.days_active_count()
            done = max(0, sim_day + 1 - active_days)
            deficit = max(0, req - done)
            parts.append(f"休息日:需{req}已{done}缺{deficit}")
            if deficit > 0 and days_remaining <= 5:
                parts.append(f"⚠月末休息紧急:缺{deficit}天仅剩{days_remaining}天")

        # 每日休息状态
        for rest in rules.rest_rules:
            today_rest = memory.longest_rest_today(ctx.current_minutes)
            deficit = rest.required_minutes - today_rest
            if deficit > 0:
                parts.append(f"今日休息:已{today_rest}分 需{rest.required_minutes}分 缺{deficit}分")

        no_drive_info = []
        for w in rules.no_drive_windows:
            no_drive_info.append(f"{w.start_minute // 60}:00-{w.end_minute // 60}:00罚{w.penalty_amount}元/天")
        if no_drive_info:
            parts.append(f"禁行:{','.join(no_drive_info[:2])}")

        # 定时事件状态
        for event in rules.timed_stay_events:
            phase = scoring.timed_event_phase(event, memory, ctx)
            if phase != "done" and phase != "early":
                event_day = event.start_minutes // 1440
                parts.append(f"家事:{phase} 第{event_day + 1}天 罚{event.absence_penalty_per_minute}元/分")

        if memory.consecutive_wait_count > 0:
            parts.append(f"连续等待{memory.consecutive_wait_count}次")

        # 当日策略提示（仅显示优先级，不传递rest_today避免过度偏向休息）
        strategy = memory.daily_strategy.get(sim_day)
        if strategy:
            priority = strategy.get("priority", "")
            reason = str(strategy.get("reason", ""))[:40]
            parts.append(f"今日策略:{priority} {reason}")

        # 接单成功率
        sr = memory.cargo_success_rate()
        if sr < 1.0:
            parts.append(f"接单成功率:{sr:.0%}")

        parts.append("")
        for i, c in enumerate(candidates):
            top_bd = sorted(c.breakdown.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            bd_str = " ".join(f"{k}={v:+.0f}" for k, v in top_bd)
            if c.action == "take_order":
                cid = str(c.params.get("cargo_id", "?"))[:12]
                parts.append(f"{i + 1}.[接]{cid} 分={c.score:.0f} {bd_str}")
            elif c.action == "wait":
                dur = c.params.get("duration_minutes", 0)
                parts.append(f"{i + 1}.[等]{dur}分 分={c.score:.0f} {bd_str}")
            elif c.action == "reposition":
                lat = c.params.get("latitude", 0)
                lng = c.params.get("longitude", 0)
                parts.append(f"{i + 1}.[移]→({lat:.1f},{lng:.1f}) 分={c.score:.0f} {bd_str}")

        parts.append("选最优编号:")
        return "\n".join(parts)

    @staticmethod
    def _parse_choice(content: str, max_idx: int) -> int | None:
        """从 LLM 回复中提取候选编号（1-indexed → 0-indexed）。"""
        for m in re.finditer(r"\d+", content):
            idx = int(m.group()) - 1
            if 0 <= idx < max_idx:
                return idx
        return None

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
        if memory.rules is not None and memory.rules_signature == signature:
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


# 列出可供外部导出的名字，保证 ``from agent.model_decision_service import *`` 不会泄露内部状态。
__all__ = ["ModelDecisionService"]
