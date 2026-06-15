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
official platform, not via single local runs. The value-side gross-income knobs
(`AGENT_ABS_NET_ALPHA` default **0.2**, `AGENT_CHAIN_VALUE_WEIGHT` default **0.45**,
`AGENT_LLM_WAIT_OVERRIDE_NET_PER_H` default **50**) only re-rank/pick among
candidates that already passed the `net>0` feasibility + compliance filter, so
they raise gross without adding preference penalty; the night-crossing knobs are
the ones that trade penalty for gross. Current knobs include:
`AGENT_NIGHT_CROSS_MARGIN`, `AGENT_NIGHT_CROSS_MAX_DAYS`,
`AGENT_ORDER_TIME_OVERHEAD_MIN`, `AGENT_CHAIN_VALUE_WEIGHT`, `AGENT_ABS_NET_ALPHA`,
`AGENT_CHAIN_DEPTH_WEIGHT` (default **0 = off**; complements `AGENT_CHAIN_VALUE_WEIGHT`
by rewarding drop-off cities with *many* recently-observed orders, i.e. a reliable
immediate re-load / less dead-head, not just a high mean rate — log-scaled, saturating
at `AGENT_CHAIN_DEPTH_REF` orders, only on liquidity-positive destinations. Pure
re-rank of already net>0+compliant candidates → gross-only, penalty-neutral; shared by
the deterministic picker and the fast decision LLM; A/B 0.2–0.4 — see notes §-11),
`AGENT_CHAIN_DEPTH_REF` (default **8**),
`AGENT_WEAK_LOCAL_REPOSITION_NET_PER_H` (default **45**; divert off a weak local
order to a richer observed market — net-protected via `_anti_strand` min_net gate
and window-safe, so penalty-neutral; set 0 to restore "reposition only when
stranded"), `AGENT_PENALTY_CAP_CREDIT`,
`AGENT_CATEGORY_SOFT`, `AGENT_LLM_WAIT_OVERRIDE_NET_PER_H`,
`AGENT_DECISION_LLM` (default **1 = on**; set 0 for a fully deterministic agent —
the per-step decision LLM is skipped entirely so `decide()` runs in ~0.1-0.3s with
near-zero tokens, like the low-latency leaderboard teams; preference compile +
daily directive still use the LLM. Leaderboard evidence + notes §-1/§-8 suggest
this is worth A/B-ing since near-zero-LLM teams currently out-net our LLM-on run),
`AGENT_PARSE_LLM` (default **1 = on**; set 0 for fully deterministic "直接解析" —
every preference is parsed by the regex engine and the per-day LLM directive is
skipped, so with `AGENT_DECISION_LLM=0` `decide()` issues ZERO model calls. The
regex parser handles the full 3-month D001 fixture incl. the weekend night-rest
relaxation via `DriverRules.weekend_no_drive_shift_min`; trades LLM recall on
unstructured/dialect prefs for determinism/speed — see notes §-9),
`AGENT_FEWSHOT_TOPK` (default **40**; the preference extractor's few-shot is no longer
inlined — it is retrieved per preference text from the >=400-example bank in
`demo/agent/parse_fewshot_bank.py` and only the top-K most relevant (plus fixed anchor
negatives) are injected. Set 0 to disable retrieval and use the rules-only extractor prompt;
lower K to cut parse tokens. Only affects `AGENT_PARSE_LLM=1`. The bank carries many
zero-constraint negatives to fight over-extraction / hallucinated constraints — the root
cause of our lowest-gross standing — see notes §-10),
`AGENT_DECISION_THINKING` (default **0 = OFF** as of 2026-06-14 — the per-step decision
LLM runs in fast mode; set 1 to restore the old selective-thinking path),
`AGENT_THINKING_WALL_BUDGET_SECONDS`, `AGENT_THINKING_SELECTIVE` (default 1 = spend the
idle reasoning budget only on high-stakes steps under a hard cumulative cap; only takes
effect when thinking is re-enabled), `AGENT_THINKING_HIGH_STAKES_NET` (default 1500),
`AGENT_LLM_CARGO_SUMMARY_LIMIT` (default **24**) / `AGENT_LIQ_TOP_N` (default **12**) —
candidate / market-table widths shown to the fast decision LLM (spend idle token budget on
context throughput, not reasoning depth), `AGENT_NIGHT_CROSS_EXTRA_MARGIN_PER_DAY`
(default 0 = no-op; raise to trim marginal multi-day crossings). See
`docs/agent-optimization-notes.md` §-10/§-9/§-8/§-6/§-5/§-4 for what each does and how to revert.

The driver fixture (`demo/server/data/drivers.json`, D001) spans **three months**
(2026-03-01→05-31): month-windowed category quotas (Apr 水果 12, May 建材 12 + Apr
carryover) and a cross-month long-haul cap. The deterministic regex parser handles
all of these plus the weekend night-rest relaxation, so `AGENT_PARSE_LLM=0` is a
valid full-season submission mode (validate via the test suite, never a single run).
