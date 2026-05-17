"""Optuna 贝叶斯参数搜索：自动优化评分系统参数以最大化净收入。

用法：
    export DASHSCOPE_API_KEY="..."
    python demo/param_search.py [--n-trials N] [--timeout SECONDS]

每个 trial 在进程内运行一次完整 31 天仿真（纯评分模式，不调用 LLM 决策），
以 10 位司机总净收入作为优化目标。

搜索空间覆盖 config.py 中 10 个最敏感的评分参数。
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SERVER_ROOT = _SCRIPT_DIR / "server"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

import optuna

from agent import config, driver_memory, model_decision_service, scoring

_RESULTS_DIR = _SCRIPT_DIR / "results"

BASELINE_NET_INCOME = 242776.0

_ORIGINAL_CONFIG: dict[str, object] = {}

_SEARCH_PARAMS = [
    "DEFAULT_OPPORTUNITY_COST_PER_MINUTE",
    "PICKUP_DEADHEAD_SOFT_THRESHOLD_KM",
    "PICKUP_DEADHEAD_SOFT_COEFF",
    "STAGNATION_WAIT_THRESHOLD",
    "STAGNATION_WAIT_PENALTY_PER_STEP",
    "MONTHLY_DAY_OFF_SPACING_COEFF",
    "HOME_RULE_AFTERNOON_BLOCK_HOUR",
    "NO_DRIVE_SAFETY_BUFFER_MINUTES",
    "CARGO_FAILURE_ATTEMPT_COST_YUAN",
    "PREFERRED_CARGO_BONUS_MULTIPLIER",
]


def _save_original_config() -> None:
    for name in _SEARCH_PARAMS:
        _ORIGINAL_CONFIG[name] = getattr(config, name)


def _apply_config(params: dict[str, object]) -> None:
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

    scoring.DEFAULT_OPPORTUNITY_COST_PER_MINUTE = config.DEFAULT_OPPORTUNITY_COST_PER_MINUTE
    scoring.HORIZON_OVERFLOW_PENALTY = config.HORIZON_OVERFLOW_PENALTY
    scoring.DEFAULT_REPOSITION_SPEED_KMH = config.DEFAULT_REPOSITION_SPEED_KMH
    scoring.HARD_CONSTRAINT_PENALTY = config.HARD_CONSTRAINT_PENALTY

    driver_memory.PER_DRIVER_TOKEN_LIMIT = config.PER_DRIVER_TOKEN_LIMIT
    driver_memory.TOKEN_DEGRADE_THRESHOLD = config.TOKEN_DEGRADE_THRESHOLD

    model_decision_service._HISTORY_LOOKBACK_STEPS = config.HISTORY_LOOKBACK_STEPS
    model_decision_service._TOP_ORDER_CANDIDATES = config.TOP_ORDER_CANDIDATES
    model_decision_service._TOP_REPOSITION_TARGETS = config.TOP_REPOSITION_TARGETS
    model_decision_service._MIN_WAIT_FALLBACK_MINUTES = config.MIN_WAIT_FALLBACK_MINUTES


def _restore_config() -> None:
    _apply_config(_ORIGINAL_CONFIG)


def _clean_results() -> None:
    if _RESULTS_DIR.exists():
        for f in _RESULTS_DIR.glob("actions_202603_*.jsonl"):
            f.unlink(missing_ok=True)
        for f in _RESULTS_DIR.glob("run_summary_*.json"):
            f.unlink(missing_ok=True)
        income_file = _RESULTS_DIR / "monthly_income_202603.json"
        income_file.unlink(missing_ok=True)


def _disable_llm_decisions() -> None:
    from agent.model_decision_service import ModelDecisionService
    ModelDecisionService._llm_select_action = lambda self, *a, **kw: None


def _run_simulation() -> dict:
    from bench.evaluation_runner import EvaluationRunner
    runner = EvaluationRunner()
    return runner.run()


def _compute_income() -> tuple[float, float]:
    subprocess.run(
        [sys.executable, str(_SCRIPT_DIR / "calc_monthly_income.py")],
        capture_output=True,
        text=True,
        cwd=str(_SCRIPT_DIR),
        timeout=120,
    )
    income_file = _RESULTS_DIR / "monthly_income_202603.json"
    if not income_file.is_file():
        return 0.0, 0.0
    data = json.loads(income_file.read_text())
    net_income = float(data["summary"]["total_net_income_all_drivers"])
    penalty = float(data["summary"]["total_preference_penalty"])
    return net_income, penalty


def _define_search_space(trial: optuna.Trial) -> dict:
    return {
        "DEFAULT_OPPORTUNITY_COST_PER_MINUTE": trial.suggest_float(
            "opp_cost_per_min", 0.2, 1.0, step=0.1,
        ),
        "PICKUP_DEADHEAD_SOFT_THRESHOLD_KM": trial.suggest_float(
            "pickup_soft_km", 10.0, 40.0, step=5.0,
        ),
        "PICKUP_DEADHEAD_SOFT_COEFF": trial.suggest_float(
            "pickup_soft_coeff", 0.2, 1.0, step=0.1,
        ),
        "STAGNATION_WAIT_THRESHOLD": trial.suggest_int(
            "stag_wait_thresh", 2, 6,
        ),
        "STAGNATION_WAIT_PENALTY_PER_STEP": trial.suggest_float(
            "stag_wait_penalty", 60.0, 240.0, step=20.0,
        ),
        "MONTHLY_DAY_OFF_SPACING_COEFF": trial.suggest_float(
            "monthly_off_spacing", 0.2, 0.8, step=0.1,
        ),
        "HOME_RULE_AFTERNOON_BLOCK_HOUR": trial.suggest_int(
            "home_block_hour", 12, 18,
        ),
        "NO_DRIVE_SAFETY_BUFFER_MINUTES": trial.suggest_int(
            "no_drive_buffer", 30, 90, step=15,
        ),
        "CARGO_FAILURE_ATTEMPT_COST_YUAN": trial.suggest_float(
            "cargo_fail_cost", 40.0, 160.0, step=20.0,
        ),
        "PREFERRED_CARGO_BONUS_MULTIPLIER": trial.suggest_float(
            "pref_cargo_bonus", 1.0, 2.0, step=0.1,
        ),
    }


def _run_trial(trial_number: int, params: dict) -> tuple[float, float]:
    _clean_results()
    driver_memory.reset()
    _apply_config(params)

    t0 = time.time()
    try:
        _run_simulation()
    except Exception as exc:
        print(f"  Trial {trial_number}: simulation failed - {exc}", flush=True)
        _restore_config()
        return 0.0, 0.0

    net_income, penalty = _compute_income()
    elapsed = time.time() - t0

    _restore_config()

    print(
        f"  Trial {trial_number}: net_income={net_income:,.0f} "
        f"penalty={penalty:,.0f} "
        f"vs_baseline={net_income - BASELINE_NET_INCOME:+,.0f} "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    return net_income, penalty


def objective(trial: optuna.Trial) -> float:
    params = _define_search_space(trial)
    net_income, penalty = _run_trial(trial.number, params)

    if net_income == 0.0:
        raise optuna.TrialPruned("simulation failed")

    trial.set_user_attr("penalty", penalty)
    trial.set_user_attr("vs_baseline", net_income - BASELINE_NET_INCOME)
    return net_income


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Optuna scoring param search")
    parser.add_argument("--n-trials", type=int, default=8, help="number of trials (default 8)")
    parser.add_argument("--timeout", type=int, default=None, help="total timeout seconds")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger("bench").setLevel(logging.WARNING)
    logging.getLogger("agent").setLevel(logging.WARNING)

    print(f"=== Optuna param search ===")
    print(f"trials: {args.n_trials}")
    print(f"baseline: {BASELINE_NET_INCOME:,.0f}")
    print(f"mode: pure scoring (LLM decisions disabled)")
    print(flush=True)

    _save_original_config()
    _disable_llm_decisions()

    study = optuna.create_study(
        direction="maximize",
        study_name="scoring_param_search",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.enqueue_trial({
        "opp_cost_per_min": 0.5,
        "pickup_soft_km": 20.0,
        "pickup_soft_coeff": 0.5,
        "stag_wait_thresh": 3,
        "stag_wait_penalty": 120.0,
        "monthly_off_spacing": 0.4,
        "home_block_hour": 14,
        "no_drive_buffer": 45,
        "cargo_fail_cost": 80.0,
        "pref_cargo_bonus": 1.2,
    })

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    _restore_config()

    print("\n" + "=" * 60)
    print("=== search complete ===")
    print(f"best income: {study.best_value:,.0f} (vs baseline {study.best_value - BASELINE_NET_INCOME:+,.0f})")
    print(f"best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nall trials:")
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            vs = t.value - BASELINE_NET_INCOME if t.value else 0
            print(f"  Trial {t.number}: {t.value:,.0f} ({vs:+,.0f}) penalty={t.user_attrs.get('penalty', '?')}")

    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "vs_baseline": study.best_value - BASELINE_NET_INCOME,
        "baseline": BASELINE_NET_INCOME,
        "trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    output_file = _SCRIPT_DIR / "param_search_results.json"
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nresults saved to: {output_file}")


if __name__ == "__main__":
    main()
