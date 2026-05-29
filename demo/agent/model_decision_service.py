"""模型决策服务：依赖 `simkit.ports.SimulationApiPort`，由评测进程注入具体环境。"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort


class ModelDecisionService:
    """基于大模型的单步决策：拉取状态与候选货源，请求补全并解析为结构化动作。"""

    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._logger = logging.getLogger("agent.decision_service")

    def decide(self, driver_id: str) -> dict[str, Any]:
        # # 评测内需可查「本会话已落盘前的动作流水」
        # hist_preview = self._api.query_decision_history(driver_id, 1)
        # last_step = None
        # recs = hist_preview.get("records") or []
        # if recs:
        #     last_step = recs[-1].get("step")
        # self._logger.info(
        #     "[demo] query_decision_history(step=1 上一执行)  total_steps=%s returned=%s last_record_step=%s",
        #     hist_preview.get("total_steps"),
        #     hist_preview.get("returned_count"),
        #     last_step,
        # )

        status = self._api.get_driver_status(driver_id)
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        cargo_resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=100)
        items = cargo_resp.get("items", [])
        self._logger.info(
            "decision input driver_id=%s time_min=%s loc=(%.5f,%.5f) cargo_items=%s",
            driver_id,
            status.get("simulation_progress_minutes"),
            lat,
            lng,
            len(items),
        )
        prompt = self._build_prompt(driver_id=driver_id, status=status, items=items)
        model_resp = self._api.model_chat_completion(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "你是货运调度决策器。"
                            "只允许输出一个JSON对象，格式必须是"
                            '{"action":"take_order|reposition|wait","params":{...}}。'
                            "禁止输出markdown、解释或额外文本。"
                            "当action是take_order时，params必须包含cargo_id字符串；"
                            "当action是reposition时，params必须包含latitude和longitude数值；"
                            "当action是wait时，params必须包含duration_minutes正整数。"
                            "simulation_progress_minutes 为自 2026-03-01 00:00:00 起的仿真经过分钟数。"
                            "候选货源含 load_time 为装货时间窗 [开始,结束]（墙钟）；"
                            "若当前仿真时刻晚于窗结束则 take_order 会失败。"
                            "若接单后无法在仿真总时长内完成装货与干线，take_order 会失败（detail 含 simulation_horizon_exceeded），且不推进时间与位置。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
            }
        )
        action = self._parse_action(model_resp)
        self._logger.info(
            "decision output driver_id=%s action=%s params=%s",
            driver_id,
            action.get("action"),
            action.get("params"),
        )
        return action

    def _build_prompt(self, driver_id: str, status: dict[str, Any], items: list[dict[str, Any]]) -> str:
        cargo_candidates: list[dict[str, Any]] = []
        for item in items[:20]:
            cargo = item.get("cargo", {})
            cargo_candidates.append(
                {
                    "cargo_id": cargo.get("cargo_id"),
                    "price": cargo.get("price"),
                    "cost_time_minutes": cargo.get("cost_time_minutes"),
                    "load_time": cargo.get("load_time"),
                    "start": cargo.get("start"),
                    "end": cargo.get("end"),
                    "distance_km": item.get("distance_km"),
                }
            )
        decision_context = {
            "driver_id": driver_id,
            "simulation_progress_minutes": status.get("simulation_progress_minutes"),
            "driver_status": {
                "current_lat": status.get("current_lat"),
                "current_lng": status.get("current_lng"),
                "truck_length": status.get("truck_length"),
                "completed_order_count": status.get("completed_order_count"),
            },
            "cargo_candidates": cargo_candidates,
        }
        return json.dumps(decision_context, ensure_ascii=False)

    def _parse_action(self, model_resp: dict[str, Any]) -> dict[str, Any]:
        choices = model_resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("模型返回缺少 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回 content 为空")
        action = json.loads(content)
        if not isinstance(action, dict):
            raise ValueError("模型返回动作不是JSON对象")
        action_name = str(action.get("action", "")).strip().lower()
        params = action.get("params")
        if action_name not in {"take_order", "reposition", "wait"}:
            raise ValueError(f"模型返回未知action: {action_name}")
        if not isinstance(params, dict):
            raise ValueError("模型返回 params 必须是对象")
        if action_name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if not cargo_id:
                raise ValueError("take_order 缺少有效 cargo_id")
            return {"action": "take_order", "params": {"cargo_id": cargo_id}}
        if action_name == "reposition":
            latitude = float(params["latitude"])
            longitude = float(params["longitude"])
            return {"action": "reposition", "params": {"latitude": latitude, "longitude": longitude}}
        duration_minutes = int(params["duration_minutes"])
        if duration_minutes <= 0:
            raise ValueError("wait.duration_minutes 必须为正整数")
        return {"action": "wait", "params": {"duration_minutes": duration_minutes}}
