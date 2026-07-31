---
name: simulation-testing
description: Guide for running and evaluating the 31-day truck dispatch simulation with LLM decision service.
---

## Running Simulation

```bash
# Requires DASHSCOPE_API_KEY environment variable
cd demo/server && python main.py
```

- Duration: ~25 minutes for 10 drivers x 31 days
- Results output: `demo/results/run_summary_202603.json` and per-driver JSONL files
- Simulation epoch: 2026-03-01, 31 days

## Evaluating Results

```bash
python demo/calc_monthly_income.py --project-root demo --results-dir demo/results
```

Outputs JSON with per-driver gross_income, distance_km, cost, preference_penalty, net_income, token_usage.

## Baseline Metrics (PR #11)

| Metric | Value |
|--------|-------|
| Net Income | 242,776 yuan |
| Penalties | 21,850 yuan |
| Total Tokens | 334,425 |
| Token Budget | 5,000,000 per driver |

## Key Learnings

1. **LLM modifying scoring weights causes regression**: Direct weight adjustment via `apply_strategy_weights` decreased net income by 11%+. The scoring system's decisions should not be overridden.
2. **Advisory-only LLM strategy works**: LLM daily strategy as decision context (not weight modifier) maintains baseline performance.
3. **Skip threshold 1.5x is optimal**: Lowering to 1.3x causes LLM to override too many good scoring decisions.
4. **Strategy prompt wording matters**: Showing "remaining target: 300,000" made LLM overly conservative. Use realistic daily averages (9,000-12,000 yuan) instead.
5. **rest_today flag in prompts biases LLM toward waiting**: Don't pass rest recommendations directly to decision prompts.

## Log Analysis

```bash
# Check LLM strategy distribution
grep 'LLM每日策略' sim_log.log | grep -oP 'priority=\w+' | sort | uniq -c

# Check rest recommendations
grep 'LLM每日策略' sim_log.log | grep -oP 'rest=(True|False)' | sort | uniq -c

# Count LLM decision participation
grep -c 'LLM决策 driver=' sim_log.log
```
