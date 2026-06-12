# AGENTS.md

## Cursor Cloud specific instructions

This repo is the offline simulation/evaluation harness for a Tianchi 天池 agent
competition: an LLM-driven truck-driver decision agent that maximises monthly
**net income** (`net = 毛收入 − 里程成本 − 偏好罚款`). All agent logic lives in
`demo/agent/model_decision_service.py`; design history + the most important
"评测陷阱" (gateway nondeterminism) are in `docs/agent-optimization-notes.md`
(read this first before changing the agent). Setup/run commands are documented in
`demo/README.md` and `docs/05-快速开始.md`.

### Running things (non-obvious caveats)
- Use **`python3`** — there is no `python` on the VM.
- **Tests need no pytest and no network/API key** — each file in `demo/tests/` is a
  self-running script that mocks the model. Run the whole suite with:
  `for f in demo/tests/test_*.py; do python3 "$f"; done` (each prints `N/N passed`).
  This is the primary way to validate agent changes here.
- The **full month simulation (`demo/server/main.py`) cannot run in this environment**:
  it (1) mandatorily requires a model API key (`DASHSCOPE_API_KEY` or
  `TIANCHI_MODEL_API_KEY`) and a reachable OpenAI-compatible endpoint, and (2) loads
  `demo/server/data/cargo_dataset.jsonl`, which is a **Git LFS pointer** (~628MB real;
  `git lfs pull` required). There is no bundled offline/mock model mode. So do NOT
  expect to produce official scores locally — validate via the unit tests instead.
- **Single-run scores are not comparable** (LLM gateway nondeterminism can swing net
  income by >10k even at temperature 0 — see notes §2). Never conclude a change helped
  or hurt from one run; the platform needs multiple runs averaged.
- The finals impose a **hard 4h total-runtime cap**, which (not the 5M-token/driver cap)
  is usually the binding constraint — thinking mode auto-downgrades to fast mode to stay
  under it. "Unused token budget" generally means "no spare wall-clock time", not "spend
  more tokens".

### Behaviour knobs
Agent behaviour is tuned via `AGENT_*` env vars read at import in
`model_decision_service.py`. Defaults are the submission values; sweep on the
official platform, not via single local runs. Current knobs include:
`AGENT_NIGHT_CROSS_MARGIN`, `AGENT_NIGHT_CROSS_MAX_DAYS`,
`AGENT_ORDER_TIME_OVERHEAD_MIN`, `AGENT_CHAIN_VALUE_WEIGHT`, `AGENT_ABS_NET_ALPHA`,
`AGENT_WEAK_LOCAL_REPOSITION_NET_PER_H` (default 0 = off), `AGENT_PENALTY_CAP_CREDIT`,
`AGENT_CATEGORY_SOFT`, `AGENT_LLM_WAIT_OVERRIDE_NET_PER_H`, `AGENT_DECISION_THINKING`,
`AGENT_THINKING_WALL_BUDGET_SECONDS`. See `docs/agent-optimization-notes.md` §-4 for
what each does and how to revert to prior behaviour.
