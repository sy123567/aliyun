"""仿真主循环：按司机顺序触发决策并推进 simkit 状态。"""

from __future__ import annotations

import json
import logging
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from simkit import simulation_actions
from simkit.cargo_repository import CargoRepository
from simkit.driver_state_manager import DriverStateManager
from simkit.ports import AgentDecisionPort

from .driver_token_budget import DriverTokenBudget

# 与 `simkit.cargo_repository.CargoRepository` 中仿真起点一致（2026-03-01 00:00）
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)

_COORDINATE_KEYS = frozenset(
    {"lat", "lng", "latitude", "longitude", "current_lat", "current_lng"}
)


def _format_sim_clock(simulation_minutes: int) -> str:
    """仿真分钟偏移 → 可读墙上时间（便于对照货源 create/remove 时间）。"""
    dt = _SIMULATION_EPOCH + timedelta(minutes=int(simulation_minutes))
    return dt.strftime("%Y-%m-%d %H:%M")


@dataclass
class DriverSimulationState:
    driver_id: str
    progress_minutes: int = 0
    end_reason: str = "normal"
    failure_message: str | None = None


@dataclass
class SimulationRunContext:
    max_steps_per_driver: int
    month_duration_minutes: int
    actions_by_driver: dict[str, list[dict[str, Any]]]
    steps_by_driver: dict[str, int]
    driver_states: dict[str, DriverSimulationState] = field(default_factory=dict)
    driver_result_files: dict[str, str] = field(default_factory=dict)
    driver_failures: dict[str, str] = field(default_factory=dict)
    total_steps: int = 0
    output_timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
    )


class SimulationOrchestrator:
    """协调决策与仿真状态推进。"""

    def __init__(
        self,
        cargo_repository: CargoRepository,
        driver_state_manager: DriverStateManager,
        agent_decision: AgentDecisionPort,
        results_dir: Path,
        reposition_speed_km_per_hour: float,
        simulation_max_steps: int,
        simulation_duration_days: int,
        driver_max_total_tokens: int = DriverTokenBudget.DEFAULT_LIMIT,
        *,
        session_actions_by_driver: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._repo = cargo_repository
        self._manager = driver_state_manager
        self._agent_decision = agent_decision
        self._results_dir = results_dir
        self._reposition_speed_km_per_hour = reposition_speed_km_per_hour
        self._simulation_max_steps = simulation_max_steps
        self._simulation_duration_days = simulation_duration_days
        self._simulation_horizon_minutes = int(simulation_duration_days) * 24 * 60
        self._driver_token_budget = DriverTokenBudget(driver_max_total_tokens)
        self._session_actions_by_driver = session_actions_by_driver
        self._simulate_started_at: float | None = None
        self._logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger("bench.simulation_orchestrator")
        if logger.handlers:
            return logger
        log_dir = self._results_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "simulation_orchestrator.log"
        formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        logger.propagate = False
        return logger

    def run(self, max_steps: int | None = None) -> dict[str, Any]:
        self._simulate_started_at = time.perf_counter()
        self._logger.info("simulation start marked")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._archive_existing_results()
        max_steps_per_driver = max_steps if max_steps is not None else self._simulation_max_steps
        if max_steps_per_driver <= 0:
            raise ValueError("max_steps 必须为正整数")

        driver_ids = self._manager.list_driver_ids()
        ctx = self._build_run_context(driver_ids, max_steps_per_driver)

        for driver_id in driver_ids:
            self._run_driver_with_isolation(driver_id, ctx)

        simulate_time_seconds = round(time.perf_counter() - self._simulate_started_at, 2)
        self._write_run_summary(ctx, simulate_time_seconds)
        self._logger.info(
            "simulation run complete steps=%s remaining_cargo=%s simulate_time_seconds=%s "
            "driver_failures=%s",
            ctx.total_steps,
            self._repo.size,
            simulate_time_seconds,
            len(ctx.driver_failures),
        )
        return {
            "completed_steps": ctx.total_steps,
            "remaining_cargo_count": self._repo.size,
            "simulation_progress_minutes": self._manager.get_simulation_progress_minutes(),
            "simulation_wall_time": self._manager.get_simulation_wall_time(),
            "simulate_time_seconds": simulate_time_seconds,
            "driver_completed_steps": dict(ctx.steps_by_driver),
            "driver_result_files": dict(ctx.driver_result_files),
            "driver_simulation_failures": dict(ctx.driver_failures),
        }

    def _build_run_context(self, driver_ids: list[str], max_steps_per_driver: int) -> SimulationRunContext:
        if self._session_actions_by_driver is not None:
            actions_by_driver = self._session_actions_by_driver
            for driver_id in driver_ids:
                actions_by_driver.setdefault(driver_id, [])
        else:
            actions_by_driver = {driver_id: [] for driver_id in driver_ids}
        steps_by_driver = {driver_id: 0 for driver_id in driver_ids}
        driver_states = {
            driver_id: DriverSimulationState(driver_id=driver_id) for driver_id in driver_ids
        }
        return SimulationRunContext(
            max_steps_per_driver=max_steps_per_driver,
            month_duration_minutes=self._simulation_horizon_minutes,
            actions_by_driver=actions_by_driver,
            steps_by_driver=steps_by_driver,
            driver_states=driver_states,
        )

    def _run_driver_with_isolation(self, driver_id: str, ctx: SimulationRunContext) -> None:
        state = ctx.driver_states[driver_id]
        self._logger.info("driver loop begin driver_id=%s", driver_id)
        try:
            self._simulate_driver(driver_id, ctx, state)
        except Exception as exc:
            state.end_reason = "error"
            state.failure_message = self._format_exception(exc)
            ctx.driver_failures[driver_id] = state.failure_message
            self._logger.error(
                "driver simulation aborted driver_id=%s error=%s",
                driver_id,
                state.failure_message,
            )
            self._logger.debug(
                "driver simulation traceback driver_id=%s\n%s",
                driver_id,
                traceback.format_exc(),
            )
        finally:
            ctx.driver_result_files[driver_id] = self._persist_driver_actions(
                driver_id,
                ctx.actions_by_driver[driver_id],
                ctx.output_timestamp,
            )
            self._log_driver_loop_end(driver_id, state, ctx.steps_by_driver[driver_id])

    def _simulate_driver(
        self,
        driver_id: str,
        ctx: SimulationRunContext,
        state: DriverSimulationState,
    ) -> None:
        self._manager.load()
        self._repo.load()
        self._manager.start_simulation(driver_id=driver_id, progress_minutes=0)
        self._repo.sync_time_minutes(0)
        state.progress_minutes = 0

        while (
            ctx.steps_by_driver[driver_id] < ctx.max_steps_per_driver
            and state.progress_minutes < ctx.month_duration_minutes
            and self._repo.size > 0
        ):
            try:
                state.progress_minutes = self._execute_one_step(driver_id, ctx)
            except Exception as exc:
                state.end_reason = "error"
                state.failure_message = self._format_exception(exc)
                ctx.driver_failures[driver_id] = state.failure_message
                self._logger.error(
                    "driver step failed driver_id=%s step=%s error=%s",
                    driver_id,
                    ctx.steps_by_driver[driver_id],
                    state.failure_message,
                )
                self._logger.debug(
                    "driver step traceback driver_id=%s\n%s",
                    driver_id,
                    traceback.format_exc(),
                )
                break

            if self._driver_token_budget.is_over_limit(driver_id):
                state.end_reason = "token_limit"
                cumulative_tokens = self._driver_token_budget.cumulative(driver_id)
                self._logger.warning(
                    "driver token limit exceeded driver_id=%s cumulative=%s limit=%s; "
                    "stop driver simulation (prior steps kept)",
                    driver_id,
                    cumulative_tokens,
                    self._driver_token_budget.limit,
                )
                break

        if (
            state.end_reason == "normal"
            and ctx.steps_by_driver[driver_id] >= ctx.max_steps_per_driver
        ):
            state.end_reason = "max_steps"
            self._logger.info(
                "driver max steps reached driver_id=%s steps=%s limit=%s",
                driver_id,
                ctx.steps_by_driver[driver_id],
                ctx.max_steps_per_driver,
            )
        elif state.end_reason == "normal" and self._driver_token_budget.was_stopped_by_limit(driver_id):
            state.end_reason = "token_limit"

    def _execute_one_step(self, driver_id: str, ctx: SimulationRunContext) -> int:
        step_start_minutes = self._manager.get_simulation_progress_minutes()
        before_status = self._manager.get_driver_status(driver_id)
        action = self._call_agent(driver_id)
        progress_after_decision = self._manager.get_simulation_progress_minutes()
        query_scan_cost_minutes = progress_after_decision - step_start_minutes
        result = self._apply_action(driver_id, action)
        after_status = self._manager.get_driver_status(driver_id)
        current_progress = self._manager.get_simulation_progress_minutes()
        driver_progress_minutes = min(current_progress, ctx.month_duration_minutes)
        true_sim_minutes_after = current_progress
        step_elapsed_minutes = true_sim_minutes_after - step_start_minutes
        ctx.total_steps += 1
        ctx.steps_by_driver[driver_id] += 1
        token_usage = action.get("model_usage", {})
        ctx.actions_by_driver[driver_id].append(
            self._normalize_for_output(
                {
                    "step": ctx.steps_by_driver[driver_id],
                    "driver_id": driver_id,
                    "step_elapsed_minutes": step_elapsed_minutes,
                    "query_scan_cost_minutes": query_scan_cost_minutes,
                    "action_exec_cost_minutes": true_sim_minutes_after - progress_after_decision,
                    "position_before": {
                        "lat": float(before_status["current_lat"]),
                        "lng": float(before_status["current_lng"]),
                    },
                    "position_after": {
                        "lat": float(after_status["current_lat"]),
                        "lng": float(after_status["current_lng"]),
                    },
                    "simulation_end_time": _format_sim_clock(true_sim_minutes_after),
                    "action": action,
                    "token_usage": token_usage,
                    "result": result,
                }
            )
        )
        self._log_step_line(
            driver_id=driver_id,
            step=ctx.steps_by_driver[driver_id],
            sim_min_before=step_start_minutes,
            sim_min_after=true_sim_minutes_after,
            round_cost_minutes=step_elapsed_minutes,
            action=action,
            token_usage=token_usage,
            result=result,
            loc_before=(float(before_status["current_lat"]), float(before_status["current_lng"])),
            loc_after=(float(after_status["current_lat"]), float(after_status["current_lng"])),
        )
        self._driver_token_budget.record_step(driver_id, token_usage)
        return driver_progress_minutes

    @staticmethod
    def _format_exception(exc: BaseException) -> str:
        return f"{type(exc).__name__}: {exc}"

    def _log_driver_loop_end(self, driver_id: str, state: DriverSimulationState, completed_steps: int) -> None:
        self._logger.info(
            "driver loop end driver_id=%s steps=%s sim_clock=%s (min=%s) end_reason=%s "
            "cumulative_tokens=%s failure=%s",
            driver_id,
            completed_steps,
            _format_sim_clock(state.progress_minutes),
            state.progress_minutes,
            state.end_reason,
            self._driver_token_budget.cumulative(driver_id),
            state.failure_message or "",
        )

    def _call_agent(self, driver_id: str) -> dict[str, Any]:
        data = self._agent_decision.decide(driver_id)
        if not isinstance(data, dict):
            raise ValueError("决策返回格式无效，必须是 JSON 对象")
        if "action" not in data:
            raise ValueError("决策返回缺少 action 字段")
        return data

    def _log_step_line(
        self,
        *,
        driver_id: str,
        step: int,
        sim_min_before: int,
        sim_min_after: int,
        round_cost_minutes: int,
        action: dict[str, Any],
        token_usage: dict[str, Any],
        result: dict[str, Any],
        loc_before: tuple[float, float],
        loc_after: tuple[float, float],
    ) -> None:
        """单行规范日志：该司机本轮 step（自 1 递增）、仿真时间、决策、耗时、Token、位置摘要。
        ``round_cost_minutes`` 为整步真实推进分钟数（含决策阶段 ``query_cargo`` 浏览列表的扫描耗时 + 本步动作耗时）。"""
        params = action.get("params", {})
        params_compact = (
            json.dumps(params, ensure_ascii=False, separators=(",", ":"))
            if isinstance(params, dict)
            else str(params)
        )
        result_compact = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        self._logger.info(
            "[STEP] driver=%s step=%s sim_clock=%s->%s (min %s->%s) round_cost_min=%s "
            "decision=%s params=%s "
            "tokens prompt=%s completion=%s reasoning=%s total=%s "
            "loc (%.5f,%.5f)->(%.5f,%.5f) result=%s",
            driver_id,
            step,
            _format_sim_clock(sim_min_before),
            _format_sim_clock(sim_min_after),
            sim_min_before,
            sim_min_after,
            round_cost_minutes,
            action.get("action"),
            params_compact,
            int(token_usage.get("prompt_tokens", 0)),
            int(token_usage.get("completion_tokens", 0)),
            int(token_usage.get("reasoning_tokens", 0)),
            int(token_usage.get("total_tokens", 0)),
            loc_before[0],
            loc_before[1],
            loc_after[0],
            loc_after[1],
            result_compact,
        )

    def _apply_action(self, driver_id: str, action: dict[str, Any]) -> dict[str, Any]:
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("action.params 必须是对象")

        if action_name == "wait":
            duration_minutes = int(params.get("duration_minutes", 1))
            return simulation_actions.wait(self._repo, self._manager, driver_id, duration_minutes)

        if action_name == "reposition":
            target_lat = float(params["latitude"])
            target_lng = float(params["longitude"])
            return simulation_actions.reposition(
                self._repo,
                self._manager,
                driver_id,
                target_lat,
                target_lng,
                speed_km_per_hour=self._reposition_speed_km_per_hour,
            )

        if action_name == "take_order":
            cargo_id = str(params["cargo_id"])
            cargo = self._repo.get_by_id(cargo_id)
            if cargo is None:
                progress = self._manager.advance_progress(driver_id, 1)
                self._repo.sync_time_minutes(progress)
                return {
                    "action": "take_order",
                    "accepted": False,
                    "detail": f"cargo_id 已失效: {cargo_id}",
                    "simulation_progress_minutes": progress,
                    "simulation_wall_time": self._manager.get_simulation_wall_time(),
                }
            try:
                return simulation_actions.take_order(
                    self._repo,
                    self._manager,
                    driver_id,
                    cargo_id,
                    reposition_speed_km_per_hour=self._reposition_speed_km_per_hour,
                    simulation_horizon_minutes=self._simulation_horizon_minutes,
                )
            except ValueError:
                progress = self._manager.advance_progress(driver_id, 1)
                self._repo.sync_time_minutes(progress)
                return {
                    "action": "take_order",
                    "accepted": False,
                    "detail": f"cargo_id 已失效: {cargo_id}",
                    "simulation_progress_minutes": progress,
                    "simulation_wall_time": self._manager.get_simulation_wall_time(),
                }

        raise ValueError(f"不支持的 action: {action_name}")

    def _persist_driver_actions(
        self,
        driver_id: str,
        actions: list[dict[str, Any]],
        timestamp: str,
    ) -> str:
        output = self._results_dir / f"actions_202603_{driver_id}_{timestamp}.jsonl"
        with output.open("w", encoding="utf-8") as file:
            for item in actions:
                file.write(json.dumps(item, ensure_ascii=False))
                file.write("\n")
        return str(output)

    def _normalize_for_output(self, value: Any, *, field_key: str | None = None) -> Any:
        if isinstance(value, float):
            if field_key in _COORDINATE_KEYS:
                return value
            return round(value, 2)
        if isinstance(value, list):
            return [self._normalize_for_output(item, field_key=field_key) for item in value]
        if isinstance(value, dict):
            return {k: self._normalize_for_output(v, field_key=k) for k, v in value.items()}
        return value

    def _archive_existing_results(self) -> None:
        entries = [path for path in self._results_dir.iterdir() if path.is_file()]
        if not entries:
            return
        history_dir = self._results_dir / "history" / datetime.now().strftime("%Y%m%d_%H%M%S")
        history_dir.mkdir(parents=True, exist_ok=True)
        for path in entries:
            shutil.move(str(path), str(history_dir / path.name))

    def _write_run_summary(self, ctx: SimulationRunContext, simulate_time_seconds: float) -> None:
        driver_end_reasons = {
            driver_id: ctx.driver_states[driver_id].end_reason for driver_id in ctx.driver_states
        }
        summary = {
            "month": "2026-03",
            "simulate_time_seconds": simulate_time_seconds,
            "simulation_duration_days": self._simulation_duration_days,
            "completed_steps": ctx.total_steps,
            "remaining_cargo_count": self._repo.size,
            "driver_completed_steps": dict(ctx.steps_by_driver),
            "driver_result_files": dict(ctx.driver_result_files),
            "simulation_max_steps_per_driver": ctx.max_steps_per_driver,
            "driver_max_total_tokens": self._driver_token_budget.limit,
            "driver_token_usage": self._driver_token_budget.cumulative_snapshot(),
            "driver_token_limit_stopped": self._driver_token_budget.limit_stopped_snapshot(),
            "driver_end_reasons": driver_end_reasons,
            "driver_simulation_failures": dict(ctx.driver_failures),
        }
        summary_path = self._results_dir / "run_summary_202603.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
