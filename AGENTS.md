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
they mostly raise gross; the night-crossing knobs are the ones that *explicitly*
trade penalty for gross. **Caveat (learned the hard way — notes §-14): "re-rank
after the net>0/compliance filter" is NOT the same as "penalty-neutral".** That
filter only blocks HARD-illegal moves; orders carrying SOFT penalties (long-haul
cap overflow, night-crossing, category quota) still pass it with the penalty
merely priced into `eff_net`. A gross-oriented multiplier (e.g. the chain levers)
can still leapfrog such an order above a clean one, so an aggressive re-rank knob
CAN raise total penalty. Treat every value-side knob's penalty impact as an
empirical platform A/B question, not a constructional guarantee. Current knobs include:
`AGENT_NIGHT_CROSS_MARGIN`, `AGENT_NIGHT_CROSS_MAX_DAYS`,
`AGENT_ORDER_TIME_OVERHEAD_MIN`, `AGENT_CHAIN_VALUE_WEIGHT`, `AGENT_ABS_NET_ALPHA`,
`AGENT_CHAIN_DEPTH_WEIGHT` (default **0 = off**; briefly shipped at 0.3 in gross push
v5 but reverted — the platform A/B regressed hard, net 84949→41564 / penalty
17800→52900: the "penalty-neutral" assumption was FALSE, because multiplying up
drop-offs that end in liquid hubs steers the picker into big hauls carrying
long-haul / night-cross / category SOFT penalties. Complements `AGENT_CHAIN_VALUE_WEIGHT`
by rewarding drop-off cities with *many* recently-observed orders — log-scaled, saturating
at `AGENT_CHAIN_DEPTH_REF` orders, only on liquidity-positive destinations; shared by
the deterministic picker and the fast decision LLM. Keep off unless re-A/B'd carefully
one lever at a time at a small weight — see notes §-11/§-13/§-14),
`AGENT_CHAIN_DEPTH_REF` (default **8**),
`AGENT_CHAIN_NEAR_WEIGHT` (default **0 = off**; briefly shipped at 0.4 in gross push v5
alongside `AGENT_CHAIN_DEPTH_WEIGHT` but reverted after the same A/B regression — NOT
penalty-neutral in practice (notes §-14). Extends the chain credit to drop-offs
whose own city is *not* in the recent-liquidity table but which sit within
`AGENT_CHAIN_NEAR_RADIUS_KM` of a liquid hub — credits that hub's mean `net_per_h`,
decayed linearly to 0 at the radius, then fed through the SAME chain multipliers. Only
consulted when the exact-city lookup misses → exact-match path is byte-identical, zero
extra scan cost (reuses the liquidity table + city centroids); shared by the
deterministic picker and the fast decision LLM. Keep off unless re-A/B'd carefully one
lever at a time — see notes §-12/§-13/§-14),
`AGENT_CHAIN_NEAR_RADIUS_KM` (default **60**; the search radius for the above),
`AGENT_WEAK_LOCAL_REPOSITION_NET_PER_H` (default **58**, raised 45→58 in gross push
v6 — notes §-15; divert off a weak local order to a richer observed market —
net-protected via `_anti_strand` min_net gate (else the local order is taken
unchanged) and window-safe, so it adds no preference penalty; the only cost is
reposition deadhead km, bounded by ≤2 reposition/day + a ≥4h budget floor — watch
gross vs total deadhead on the platform, lower back toward 45 if deadhead/cost
rises without a gross gain; set 0 to restore "reposition only when stranded"),
`AGENT_IDLE_FORWARD_REPOSITION_NET_PER_H` (default **60**, gross push v7 — notes §-16;
the otherwise-idling driver — no order pickable AND `_anti_strand` found no *currently
reachable* target — repositions once toward the richest *observed* liquidity hub (mean
`net_per_h` ≥ this, with ≥ `AGENT_IDLE_FORWARD_REPOSITION_MIN_N` recent orders, within
`AGENT_IDLE_FORWARD_REPOSITION_MAX_KM`) instead of pure-waiting, betting the market
re-stocks. Takes no order ⇒ zero preference penalty + window/region/blackout-safe, so
penalty-orthogonal like the weak-local lever; but UNLIKE it this is NOT a strict no-op —
it does not require a currently-reachable order, so it is a bounded positive-EV bet whose
only cost is the deadhead km when the bet misses (capped by MAX_KM + 1 reposition/day +
the ≥4h budget floor). Shipped on under the best-of-N leaderboard (a wasted-deadhead run
is free, a landed bet lifts the ceiling). Set 0 to restore pure-wait),
`AGENT_IDLE_FORWARD_REPOSITION_MIN_N` (default **4**; depth floor so the bet never chases
a single outlier order) / `AGENT_IDLE_FORWARD_REPOSITION_MAX_KM` (default **200**; deadhead
cap on the bet),
`AGENT_PENALTY_CAP_CREDIT`,
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
`AGENT_LLM_CARGO_SUMMARY_LIMIT` (default **100**, raised 24→40→80→100 in gross push v8/v10/v11 — notes §-17)
/ `AGENT_LIQ_TOP_N` (default **40**, raised 12→20→30→40) — candidate / market-table widths shown
to the fast decision LLM (spend idle token budget on context throughput — INFORMATION — not
reasoning depth; an A/B of thinking-on regressed, gross up but penalty doubled §-6/§-7). This
is a ceiling bet, NOT penalty-neutral: a wider candidate list can also surface more
soft-penalised big hauls, so calibrate multi-run on the platform and revert via
`AGENT_LLM_CARGO_SUMMARY_LIMIT=24` / `AGENT_LIQ_TOP_N=12` if penalty/deadhead creep without a
net gain. `AGENT_LLM_QUERY_K` (default **120**) controls the per-step cargo scan pool ranked
before the prompt is truncated; spend more token + scan budget on information, revert to 50.
`AGENT_LLM_DECISION_MAX_TOKENS` (default **350**) caps the fast JSON response; revert to 180.
`AGENT_LLM_RISK_FIELDS` (default **1**) adds compact per-candidate risk/extra-penalty fields as a guardrail; set 0 to strip them. `AGENT_NIGHT_CROSS_EXTRA_MARGIN_PER_DAY`
(default **1.0**, raised 0→1.0 in penalty push v9 — notes §-18; demands
`night_pen * extra` of EXTRA penalty-free net for every crossed night past the first,
so a 2+ night crossing must clear ~double the night penalty before it is even shown to
the picker / decision LLM. Surgical penalty-trim: drops only the worst penalty-per-gross
multi-night hauls (the thinking-A/B penalty driver §-6/§-7) while leaving every
single-night evening haul byte-identical — crossings == 1 are never touched at any value.
NOT a pure no-op (a trimmed multi-night haul whose true net was positive costs a little
gross) but evidence-backed + low gross risk; best-of-N keeps 84949 as a floor. Revert via
`AGENT_NIGHT_CROSS_EXTRA_MARGIN_PER_DAY=0`). See
`docs/agent-optimization-notes.md` §-18/§-17/§-16/§-15/§-10/§-9/§-8/§-6/§-5/§-4 for what each does and how to revert.

The driver fixture (`demo/server/data/drivers.json`, D001) spans **three months**
(2026-03-01→05-31): month-windowed category quotas (Apr 水果 12, May 建材 12 + Apr
carryover) and a cross-month long-haul cap. The deterministic regex parser handles
all of these plus the weekend night-rest relaxation, so `AGENT_PARSE_LLM=0` is a
valid full-season submission mode (validate via the test suite, never a single run).
