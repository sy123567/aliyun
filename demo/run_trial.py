"""单次仿真试验运行器：读取参数覆盖 → 跑仿真 → 输出净收入。

用法：
    TRIAL_PARAMS_FILE=/path/to/params.json python demo/run_trial.py

环境变量：
    TRIAL_PARAMS_FILE  - JSON 文件路径，包含要覆盖的 config 参数
    DISABLE_LLM_DECISIONS - 设为 "1" 则跳过 LLM 决策（纯评分模式）
    DASHSCOPE_API_KEY  - 模型 API Key（偏好解析仍需要）
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SERVER_ROOT = _SCRIPT_DIR / "server"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))


def _apply_config_overrides(params: dict) -> None:
    """将参数覆盖应用到 agent.config 模块。"""
    from agent import config

    for key, value in params.items():
        if not hasattr(config, key):
            continue
        original = getattr(config, key)
        if isinstance(original, int):
            setattr(config, key, int(value))
        elif isinstance(original, float):
            setattr(config, key, float(value))
        else:
            setattr(config, key, value)

    # 同步更新 scoring 模块中从 config 复制的模块级常量
    from agent import scoring
    scoring.DEFAULT_OPPORTUNITY_COST_PER_MINUTE = config.DEFAULT_OPPORTUNITY_COST_PER_MINUTE
    scoring.HORIZON_OVERFLOW_PENALTY = config.HORIZON_OVERFLOW_PENALTY
    scoring.DEFAULT_REPOSITION_SPEED_KMH = config.DEFAULT_REPOSITION_SPEED_KMH
    scoring.HARD_CONSTRAINT_PENALTY = config.HARD_CONSTRAINT_PENALTY
    config.EVALUATION_HORIZON_DAYS = min(config.AGENT_HORIZON_DAYS, 30)

    # 同步 driver_memory 模块级常量
    from agent import driver_memory
    driver_memory.PER_DRIVER_TOKEN_LIMIT = config.PER_DRIVER_TOKEN_LIMIT
    driver_memory.TOKEN_DEGRADE_THRESHOLD = config.TOKEN_DEGRADE_THRESHOLD

    # 同步 model_decision_service 模块级常量
    from agent import model_decision_service
    model_decision_service._HISTORY_LOOKBACK_STEPS = config.HISTORY_LOOKBACK_STEPS
    model_decision_service._TOP_ORDER_CANDIDATES = config.TOP_ORDER_CANDIDATES
    model_decision_service._TOP_REPOSITION_TARGETS = config.TOP_REPOSITION_TARGETS
    model_decision_service._MIN_WAIT_FALLBACK_MINUTES = config.MIN_WAIT_FALLBACK_MINUTES


def _disable_llm_decisions() -> None:
    """Monkeypatch: 让 LLM 决策始终返回 None（纯评分模式）。"""
    from agent.model_decision_service import ModelDecisionService
    ModelDecisionService._llm_select_action = lambda self, *a, **kw: None  # type: ignore[assignment]


def main() -> None:
    # 读取参数覆盖
    params_file = os.environ.get("TRIAL_PARAMS_FILE", "")
    params: dict = {}
    if params_file and Path(params_file).is_file():
        params = json.loads(Path(params_file).read_text())

    # 应用参数覆盖
    _apply_config_overrides(params)

    # 是否禁用 LLM 决策
    if os.environ.get("DISABLE_LLM_DECISIONS") == "1":
        _disable_llm_decisions()

    # 清空 driver_memory 全局缓存
    from agent import driver_memory
    driver_memory.reset()

    # 降低日志级别减少输出
    logging.basicConfig(level=logging.WARNING, force=True)

    # 运行仿真
    from bench.evaluation_runner import EvaluationRunner
    runner = EvaluationRunner()
    runner.run()

    # 计算收入
    import subprocess
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "calc_monthly_income.py")],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR),
    )

    # 读取结果
    income_file = _SCRIPT_DIR / "results" / "monthly_income_202603.json"
    if income_file.is_file():
        data = json.loads(income_file.read_text())
        net_income = data["summary"]["total_net_income_all_drivers"]
        total_penalty = data["summary"]["total_preference_penalty"]
        print(f"TRIAL_RESULT|net_income={net_income:.2f}|penalty={total_penalty:.2f}")
    else:
        print("TRIAL_RESULT|net_income=0|penalty=0")


if __name__ == "__main__":
    main()
