"""动作合法性校验：保证输出严格满足赛方接口契约。

校验内容：
- 顶层结构：``{"action": str, "params": dict}``。
- ``action`` 必须是 ``take_order``、``wait``、``reposition`` 三选一。
- ``take_order`` 必须含非空 ``cargo_id``；若提供候选集，还会校验 ``cargo_id`` 必须来自候选。
- ``wait`` 必须含正整数 ``duration_minutes``。
- ``reposition`` 必须含数值 ``latitude``/``longitude``。

任何不合法都抛 ``ActionInvalid``；上层应据此回退到 ``safe_wait``。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

_LOGGER = logging.getLogger("agent.action_validator")

ALLOWED_ACTIONS = ("take_order", "wait", "reposition")


class ActionInvalid(ValueError):
    """动作不满足接口契约。"""


def validate_action(
    action: dict[str, Any],
    *,
    allowed_cargo_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """对最终动作做强校验并返回规范化后的副本。"""
    if not isinstance(action, dict):
        raise ActionInvalid(f"action must be dict, got {type(action).__name__}")
    name = action.get("action")
    params = action.get("params")
    if name not in ALLOWED_ACTIONS:
        raise ActionInvalid(f"unknown action: {name!r}")
    if not isinstance(params, dict):
        raise ActionInvalid("params must be dict")

    if name == "take_order":
        cargo_id = params.get("cargo_id")
        if not isinstance(cargo_id, str) or not cargo_id.strip():
            raise ActionInvalid("take_order requires non-empty cargo_id")
        cargo_id = cargo_id.strip()
        if allowed_cargo_ids is not None:
            allowed = set(allowed_cargo_ids)
            if cargo_id not in allowed:
                raise ActionInvalid(
                    f"cargo_id {cargo_id!r} not in candidate set (size={len(allowed)})"
                )
        return {"action": "take_order", "params": {"cargo_id": cargo_id}}

    if name == "wait":
        raw = params.get("duration_minutes")
        try:
            duration = int(raw)
        except (TypeError, ValueError) as exc:
            raise ActionInvalid("wait.duration_minutes must be int") from exc
        if duration < 1:
            raise ActionInvalid("wait.duration_minutes must be >= 1")
        return {"action": "wait", "params": {"duration_minutes": duration}}

    # name == "reposition"
    try:
        lat = float(params["latitude"])
        lng = float(params["longitude"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionInvalid("reposition requires numeric latitude/longitude") from exc
    if not (-90.0 <= lat <= 90.0):
        raise ActionInvalid(f"latitude out of range: {lat}")
    if not (-180.0 <= lng <= 180.0):
        raise ActionInvalid(f"longitude out of range: {lng}")
    return {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}


def safe_wait(duration_minutes: int = 30, *, note: str = "") -> dict[str, Any]:
    """生成保底休息动作；记录原因便于事后回放。"""
    duration = max(1, int(duration_minutes))
    if note:
        _LOGGER.info("safe_wait duration=%s reason=%s", duration, note)
    return {"action": "wait", "params": {"duration_minutes": duration}}
