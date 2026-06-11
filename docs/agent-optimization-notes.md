# Agent 优化说明（净收益最大化 + 通用化）

> 这份文档写给"接手的另一个 agent / 开发者"：clone 本仓库后先读这里，能快速理解
> **业务目标、关键改动、以及一个非常重要的评测陷阱（LLM 网关不确定性）**。
> 所有改动都集中在 `demo/agent/model_decision_service.py`。

---

## 1. 业务背景与要求（来自仓库所有者）

- 本地数据集只有 **D001** 一位司机，仅用于自测。
- **官方复赛是两位司机**：一位**广东**司机，一位**长三角**司机。
- 复赛的偏好（preferences）是**自然语言表述，且措辞会变化**；拿不到司机数据。
  - 因此 agent 必须**通用 / 语言鲁棒**，绝不能写死"深圳 / 水果 / 建材 / 某月"这类 D001 专属逻辑。
- 重点是 **LLM 维护历史决策**（`query_decision_history` → 喂给 LLM 做上下文）。
- 目标是**最大化净收益** `net = 毛收入 − 里程成本 − 偏好罚款`，**不是单纯降罚分**。
  - 毛收入足够高时，带一点罚分是可以接受的——只要净收益够高即可。

### 评分口径（`demo/calc_monthly_income.py`，仓库所有者提供的版本）
- 夜间停车休息：21:00–次日06:00 不得接单/空驶（周末可晚 2 小时）。
- 四月：水果类满 12 单，少一单扣一次（`penalty_amount`）。
- 五月：建材类满 12 单；**四月水果欠额由"五月水果"补**（`may_fruit_makeup_orders`
  抵 `april_shortfall`），未补足 + 五月建材未达标都按"少一单扣一次"计，且罚更重。
- 月度长途（>8h）≤ 5 单，超一单扣一次（**软上限**：净收益 > 罚金时仍值得接）。

跑分命令：
```bash
# 1) 跑仿真（需要 DASHSCOPE_API_KEY；大数据集先开 swap，见 §5）
cd demo/server && DASHSCOPE_API_KEY=sk-... python main.py
# 2) 评分（产出 demo/results/monthly_income_202603.json）
cd demo && python calc_monthly_income.py
```
模型名用 **`qwen3.7-plus`**（网关里写 "3.7plus" 会 model_not_found）。

---

## 2. ⚠️ 最关键的发现：单次跑分不可比（LLM 网关不确定性）

同一份代码、不同次运行，**净收益能相差 1 万以上**（毛收入在 ~88k–124k 间剧烈波动）。
实测（D001，满月 3–5 月）：

| 版本 | 采样温度 | 第1次 | 第2次 |
|---|---|---|---|
| 原版 agent（master） | 0.1 | 83.1k | 67.2k |
| 本优化（A+B1） | 0.1 | 79.2k | 58.0k |
| 本优化（A+B1） | 0   | 74.3k | 67.5k |

**结论**：
- 这种波动**baseline 自己也有**（67k–83k），是 **LLM 网关（qwen3.7-plus，疑似 MoE）固有的**，
  即使 `temperature=0` 也不能完全消除——不是 agent 代码能消掉的。
- 所以**用单次跑分去判断"某个改动有没有用"是不可靠的**。要比较必须**多次跑取均值/下限**。
- 复赛是**一次性两位司机**，真正该优化的是**抬高下限、降低波动**，而不是赌单次高分。

> 给接手者：如果你要继续调优，请**固定每个版本至少跑 3–5 次**再比较，否则会被噪声误导。

---

## 3. 本次关键改动（均在 `demo/agent/model_decision_service.py`）

### A1. 去掉 D001 专属硬编码（通用化）
- `_get_category_target()`：删除写死的 `if month_idx==1: 水果12` / `if month_idx==2: 建材12+进位`
  兜底分支。品类指标**只来源于从自然语言解析出的** `rules.monthly_category_targets`。
- `_track_category_order()`：删除只认 `=="水果"/"=="建材"` 的兜底，改为只按解析到的目标品类统计。
- 影响：对复赛未知司机不再错误地去抢"水果/建材"。D001 的品类指标本来就能被解析出来，行为不变。

### A2. LLM 目标对齐"净收益优先"
- system prompt 从"最大化净收入 / 最小化罚款 / 完成指标"改为**首要净收益**；
  接单提示从"高价、**短途**"改为"**净收益(net_per_h)最高**，长途只要 net_per_h 高也值得接"。
- 长途规则措辞改为**软上限**：超 5 单每单扣 1000，但该单净收益明显 > 1000 仍应接。

### A3. 喂给 LLM 的历史摘要改用"权威数据"
- 背景：评测 harness **只调用 `decide()`，从不回调 `update_decision_result()`**，
  所以 `DecisionHistory` 内部计数器恒为 0——原来摘要里一直显示"总接单 0 / 累计收入 ¥0 /
  最近订单全部 ✗"，**会误导 LLM**。
- 改法：`DecisionHistory.get_summary(current_day, plan)` 增加 `plan` 入参，
  累计/本月接单数、长途 x/上限、品类分布**改从每步重建的权威 `plan` 读取**
  （`plan` 由 `_sync_monthly_counts_from_history()` 用 `query_decision_history` 重建）。
  等待/空驶次数与"最近决策动作序列"仍来自本地滚动窗口（这些在 decide 时记录是可靠的）。

### B1. "欠额结转"对齐评分口径（重要！）
- 评分口径里：**四月水果欠额是用"五月水果"来补**的（`may_fruit_makeup_orders`）。
- 但原 `_get_category_target()` 的结转逻辑是把欠额加到**当月第一个品类（建材）**的目标上
  → agent 拼命多接建材（实测接了 21–23 单，远超 12），**对水果欠额毫无抵扣**，
  结果照吃 7000 的五月结转罚。
- 改法：结转时把上月每个品类的欠额加到**当月同一品类**的目标上
  （`targets[cat] += deficit`）。这样五月会去补**水果**，与评分口径一致。
- 该改动**只会更优或持平**：原逻辑给的结转抵扣credit为 0，新逻辑能真正抵扣。
  实测有一次 `mayMakeup=7` 成功把 7000 的五月结转罚降为 0。

### （附）主决策 `temperature: 0.1 → 0`
- 为可复现性改成贪心解码（与本文件其它 LLM 调用一致）。
- 注意：如 §2 所述，这**不能完全消除**网关层不确定性。

---

## 3.5 复赛 −44.7 万事故的根因与本轮修复（语义接地 + fail-safe + 历史自检）

### 事故现象
复赛两位未知司机：偏好扣分 **52.6 万** → 净收益 **−44.7 万**，token 烧到 289 万。

### 根因（已定位）
- 526,100 ÷ (2 司机 × ~92 天) ≈ **2,859/司机日**，恰是"一条**每日生效**约束被几乎每天违规一次"的量级；
  夜休罚金 2,700/次 × 92 × 2 ≈ 49.7 万，与之吻合 → **最大头是夜间停车休息几乎每天被违规**。
- 机制：偏好处理是 **LLM 抽取 → 关键词/子串"接地"校验 → 确定性调度执行**。
  老的接地是**写死关键词白名单**：`no_drive_windows` 必须原文**同时**命中时间词+动作词
  （`_ndw_grounded()`），否则**整条静默丢弃**；`forbidden_categories/regions`、`blackout`、
  `avoid`、`monthly_category_targets` 用 `_text_supports()` 原词子串匹配接地。
- 复赛偏好是自然语言、措辞会变："入夜收车""后半夜不揽货""天黑归家落锁"等没进白名单的说法
  → LLM 即使抽对了夜休窗口也被**静默丢掉** → 调度器以为没有夜休 → **每天整夜出车，按天复利扣到 50 万**。
- 关键不对称：**漏一条"每日生效"约束 ≈ 25 万/司机；多一条最多损失几小时收入**。老逻辑"宁可漏不可错"方向反了。

### 本轮修复
- **P0 时间窗 fail-safe**：`_merge_llm_rules` 里**删除 `_ndw_grounded()` 关键词门**，
  直接采纳 temperature=0 抽出的 `no_drive_windows`（含跨夜 start>end），只做时刻合法性校验。
  decision system prompt 里写死的 "21:00–06:00 夜停休" 文案改为**通用措辞**（不同司机时段不同），
  实际时段来自解析结果。解析 prompt 强化：任何"入夜/天黑/后半夜…不出车/收车/归家不动/熄火"
  一律填 `no_drive_windows`（+ 含休息含义则同时填 `rest_window`）。
- **P1 语义接地**：新增 `_confirm_rule_holds(rule_desc, all_text, default=True)`——
  用一次 temperature=0 的 LLM 调用判断"原文是否确实施加该约束"，**只有模型明确说"否"才丢弃，
  调用失败/不确定一律保留（fail-safe）**；带缓存。替换 `no_drive_windows / forbidden_categories /
  forbidden_regions / blackout / avoid_categories / monthly_category_targets` 的子串/关键词接地。
  → 任意措辞都能泛化，根治"换个说法就丢规则"。（`allowed_regions / bounded_area / forbidden_zones`
  这类"误加危害更大"的字段**仍保留原有保守门**，未放宽。）
- **P3 历史自检**：`_compliance_self_audit()` 每天把司机的硬时间窗（休息/禁驶）作为
  "【硬约束·每日生效】"块置于决策 prompt 顶部，并扫描滚动历史，若**昨日**有在禁驶时段出车的记录
  就追加"【历史自检】昨日违规"提醒。纯历史查询、无额外 LLM 调用；因为罚分按天复利，
  抓到一次即可阻止后续每天重犯。

> 验证建议（本地无复赛司机数据）：自造"措辞刁钻"的夜休/禁区偏好喂进解析器，
> 看日志 `parsed rules ... no_drive=...` 是否被采纳即可，几秒出结果，不必跑满月。
> 满月跑分受 §2 网关不确定性影响，需多跑取均值。

---

## 3.6 架构重构：编译 → 审计修复 → 通用约束引擎（面向未知司机泛化）

> 背景：复赛是广东 + 江浙沪两位**未知司机**，偏好为自然语言且措辞与本地 D001 完全不同；
> 线上评测看三个指标：**得分、扣分、token 消耗**。此前架构"解析偏好 → 硬编码执行"
> 在未知司机上扣分严重，根因是解析前端对措辞过拟合。本轮把偏好处理前端整体重构。

### 旧架构的三类失效模式（都与"措辞过拟合"有关）
1. **关键词白名单接地丢规则**：`_merge_llm_rules` 里 allowed_regions / forbidden_zones /
   bounded_area / must_visit / haul_max / daily_order_limit / home_rule 等字段都要求原文
   命中硬编码关键词表（`_ALLOW_REGION_KW` …）才被采纳——未知司机换个说法，正确抽出的
   规则被**静默丢弃**，每日生效的约束按天复利扣分（复赛 −44.7 万事故的同款机制）。
2. **逐条 LLM 确认调用**：每条规则一次 `_confirm_rule_holds` 调用（携带全部偏好原文），
   规则多时 token 成本成倍增长，且又多了一个静默丢规则的通道。
3. **正则补丁层在线误触发**：`_supplement_basic_rules` / `_supplement_dated_events` 等
   按样例措辞调出来的正则**在 LLM 解析成功后仍然叠加运行**，在未知司机文本上会
   **误加**黑名单区域 / 每日限单 / 日期事件等规则，确定性引擎随后忠实执行错误规则。

### 新架构（全部在 `demo/agent/model_decision_service.py`）
1. **两遍编译**：第一遍 temperature=0 抽取（schema 不变，新增 `monthly_longhaul_cap`）；
   第二遍 `_llm_review_rules` **审计修复**——把偏好原文 + penalty_amounts + 抽取结果一起
   交给模型，按「找漏（每日时段约束 > 品类指标/结转 > 整休/长途上限/日期事件/回家）→
   删多（臆造约束）→ 纠值（数字/时刻/日期/坐标）」输出修正后的完整 JSON。
   - fail-safe 方向：审计调用失败 → 沿用第一遍结果；审计**可修改但不可整体删除**
     第一遍抽出的 no_drive_windows / rest_window（漏一条每日窗 ≈ 25万/司机，多一条
     只损失几小时收入）。
   - 一次审计调用替代 N 次逐条确认调用 → **token 净下降**。
2. **合并层只做确定性结构校验**：坐标范围(18-55/70-140)、经纬互换、半径钳制(≤100km)、
   bounded_area/home_rule/must_visit 仍要求原文出现过坐标数字（与措辞无关的信号）、
   allowed_regions 保留垃圾地名检测 + 字符覆盖率接地。**删除全部措辞关键词门**。
3. **正则层降级为离线兜底**：仅当模型完全不可用时运行（此时决策 LLM 也不可用），
   不再叠加在 LLM 结果之上。
4. **长途上限改为偏好驱动**：`LONGHAUL_CAP=5` 全局常量删除，改为
   `DriverRules.longhaul_max_orders / longhaul_threshold_minutes`（从司机自己的偏好
   解析，罚额走 `rule_penalties["longhaul"]`）。没有该偏好的司机**不再被硬编码限制**。
   决策 prompt 里写死的"每月长途(>8h)软上限5单"与"本月需要X至少12单"文案同步改为
   完全由解析结果驱动。
5. **运行期兜底强化**：决策 prompt 始终携带**全部**偏好原文（按首次出现顺序，最多8条），
   并明示"规则摘要与原文冲突时以原文为准"；每日合规自检（硬时间窗 + 昨日违规扫描）保留。

### 回归测试
- `demo/tests/test_preference_compile_review.py`（新增）：白名单外措辞不丢规则、
  审计修复找漏、审计不可删每日窗、审计失败回退、正则补丁不在线上叠加、离线兜底可用。
- `demo/tests/test_longhaul_cap_history.py`：改为偏好驱动上限口径（新增"无偏好则无上限"
  与 merge 用例），全部通过。
- `demo/tests/test_overnight_rest_window.py`：不变，全部通过。

> 提醒：满月跑分仍受 §2 网关不确定性影响，对比新旧架构请多跑取均值/下限。

## 4. 仍然通用、未写死的既有设计（接手者可直接用）

- **偏好解析**：自然语言 → 结构化 `DriverRules`，按时间窗增量触发；
  支持品类指标、结转、禁运品类/区域、仅允许区域、空驶上限、每日接单上限、长途软上限等。
- **区域分组**：内置 广东/上海/江苏/浙江/安徽/江浙沪/**长三角**/珠三角/大湾区 + 别名归一，
  广东与长三角司机都能覆盖（`_ALLOWED_REGION_GROUPS`）。
- **长途软上限**：`_score_item()` 里对 >8h 且已达上限的单 `eff_net = net − LONGHAUL_PENALTY`，
  净收益为正才接 → 天然实现"毛高带点罚也接"。
- **历史驱动的权威计数**：`_sync_monthly_counts_from_history()` 每步用
  `query_decision_history` 重建月度长途/品类/空驶/接单累计，保证额度判断准确
  （因为 harness 不回调结果）。

---

## 5. 复现 / 环境提示

- `DASHSCOPE_API_KEY`：`main.py` 读取的环境变量名（本仓库 config 不落盘 key）。
- 数据集 `demo/server/data/cargo_dataset.jsonl`（~628MB / 150 万行）载入内存峰值 ~7GB；
  机器内存不足时先开 swap：
  ```bash
  sudo fallocate -l 24G /swapfile && sudo chmod 600 /swapfile \
    && sudo mkswap /swapfile && sudo swapon /swapfile
  ```
- 单次满月仿真约 7–9 分钟、~37x 万 token。

---

## 6. 给接手 agent 的下一步建议

1. **任何调优都多跑几次再下结论**（§2）。
2. 想进一步抬高"下限"，可考虑：当确定性调度 `_pick_order` 已找到强净值订单时，
   不让 LLM 把它改成 wait/reposition（减少 LLM 把好货等没的坏样本）。
3. 复赛若出现新的品类/结转/区域偏好，确认**偏好解析器**能从自然语言抽出来即可，
   不要再往决策逻辑里加写死分支。
