# 决策架构优化方案 — 从 260k 到 400k+

## 1. 当前状态分析

### 1.1 基线数据（master 分支，259,260 净收入）

| 指标 | 值 |
|------|-----|
| 毛收入 | 384,869 元 |
| 距离成本 | ~114,000 元 |
| 偏好罚分 | 11,320 元 |
| **净收入** | **259,260 元** |
| 司机数 | 10 |
| 仿真天数 | 31 天 |

### 1.2 各司机表现

| 司机 | 接单数 | 毛收入 | 效率(元/分) | 等待(天) | 工作(天) | 罚分 |
|------|--------|--------|-------------|----------|----------|------|
| D001 | 79 | 20,596 | 0.69 | 10.1 | 20.6 | 600 |
| D002 | 46 | 43,959 | 1.40 | 9.0 | 21.7 | 400 |
| D003 | 64 | 42,116 | 1.41 | 10.0 | 20.7 | 2,000 |
| **D004** | **66** | **59,408** | **1.36** | **0.5** | **30.3** | **4,400** |
| D005 | 70 | 32,230 | 1.21 | 12.0 | 18.5 | 200 |
| D006 | 70 | 40,187 | 1.28 | 8.9 | 21.9 | 950 |
| D007 | 53 | 33,531 | 1.27 | 11.7 | 18.3 | 400 |
| D008 | 49 | 44,432 | 1.37 | 8.1 | 22.5 | 100 |
| D009 | 57 | 25,097 | 1.32 | 15.8 | 13.2 | 900 |
| D010 | 46 | 43,313 | 1.41 | 6.5 | 21.3 | 1,370 |

### 1.3 核心瓶颈

1. **30% 空闲率**：10 个司机总等待 92 天 / 310 天可用 = 30%
2. **D004 是标杆**：几乎零等待（0.5 天），毛收入最高（59k）。其他司机如果能达到 D004 的利用率，理论总收入可达 ~376k
3. **单步贪心决策**：当前每步只看"这一单好不好"，不考虑接完后去哪、下一单怎么办
4. **固定机会成本**：1.0 元/分钟对所有场景一刀切，无法适应不同区域和时段的供需差异

### 1.4 已验证无效的策略

| 策略 | 净收入 | 失败原因 |
|------|--------|----------|
| 降低机会成本到 0.8 | 242,447 | 接了差单占用好单时间 |
| 降低机会成本到 0.3 | 224,106 | 大量低质量订单，休息违规 |
| 收入权重×1.3 | 244,808 | 罚分从 11k 涨到 16.8k |
| 自适应机会成本（连续等待衰减）| 242,447-253,997 | 与 D004 首单约束冲突 |
| 元/公里效率奖励 | 242,447 | 干扰已有评分平衡 |
| 三级递降评估（0.5→0.2→0.0）| 248,624 | 太激进，接了低质量订单 |
| 运输时间成本折半 | 240,774 | 休息违规连锁反应 |
| 货源查询减半（100→50）| 234,624 | 可选订单太少 |

---

## 2. 架构优化方案

### 2.1 方案一：订单链评估（推荐优先实施）

**核心思想**：不只看当前订单的收益，还要评估完成后卸货点的"后续接单价值"。

**实现方式**：

```python
# 在 scoring.py 的 score_take_order 中添加

def estimate_location_value(lat: float, lng: float, memory: DriverMemory) -> float:
    """
    根据司机的历史接单记录，估算某个位置的"后续接单价值"。
    
    基于 driver_memory 中已有的 hotspot 记录和历史订单数据，
    计算该位置附近的历史订单密度和平均收益。
    
    这不是预分析——是司机在运营过程中积累的经验。
    """
    value = 0.0
    for record in memory.completed_orders:  # 需要在 DriverMemory 中添加
        dist = haversine_km(lat, lng, record.end_lat, record.end_lng)
        if dist < 50:  # 50km 范围内的历史订单
            # 距离越近权重越大，时间越近权重越大
            dist_weight = max(0, 1.0 - dist / 50.0)
            value += record.income * dist_weight
    return value / max(1, len(memory.completed_orders))


# 在 score_take_order 的最终分数计算中：
# breakdown["chain_value"] = estimate_location_value(end_lat, end_lng, memory) * CHAIN_VALUE_COEFF
```

**需要修改的文件**：
- `demo/agent/driver_memory.py`：添加 `completed_orders` 记录（存储每单的卸货位置和收入）
- `demo/agent/scoring.py`：在 `score_take_order` 中添加 `chain_value` 评分项
- `demo/agent/config.py`：添加 `CHAIN_VALUE_COEFF` 等参数

**预期收益**：+10-20%（减少"送到偏僻地方后长时间等待"的情况）

**真实场景合理性**：✅ 司机跑熟了某条线路后，自然会知道哪里好找货、哪里不好找。

---

### 2.2 方案二：基于当前观察的动态机会成本

**核心思想**：不是固定 1.0 元/分钟，而是根据当前扫描到的订单质量动态调整门槛。

**实现方式**：

```python
# 在 model_decision_service.py 的 decide() 中

def compute_dynamic_opportunity_cost(
    cargo_items: list[dict],
    ctx: DecisionContext,
    rules: ParsedRules,
    memory: DriverMemory,
) -> float:
    """
    根据当前可见订单的质量分布，动态计算机会成本。
    
    - 当前扫描到很多高价值订单 → 提高机会成本（更挑剔）
    - 当前扫描到的都是低价值订单 → 降低机会成本（有单就接）
    - 连续多次扫描无好单 → 进一步降低
    """
    if not cargo_items:
        return config.DEFAULT_OPPORTUNITY_COST_PER_MINUTE * 0.5
    
    # 快速估算每个订单的"毛利率"
    margins = []
    for item in cargo_items[:20]:  # 只看前 20 个（已按距离排序）
        cargo = item.get("cargo", {})
        income = float(cargo.get("price", 0))
        haul_km = float(cargo.get("haul_distance_km", 0))
        pickup_km = float(item.get("pickup_distance_km", 0))
        total_km = haul_km + pickup_km
        cost = ctx.cost_per_km * total_km
        if total_km > 0:
            estimated_minutes = total_km / (ctx.reposition_speed_km_per_hour / 60)
            margin_per_min = (income - cost) / max(1, estimated_minutes)
            margins.append(margin_per_min)
    
    if not margins:
        return config.DEFAULT_OPPORTUNITY_COST_PER_MINUTE * 0.5
    
    # 用中位数代表当前市场水平
    margins.sort()
    median_margin = margins[len(margins) // 2]
    
    # 动态机会成本 = 市场中位数的 70%（留出利润空间）
    dynamic_cost = max(0.3, min(1.5, median_margin * 0.7))
    
    # 连续等待衰减（但不低于 0.3）
    if memory.consecutive_wait_count >= 3:
        decay = min(0.5, 0.1 * memory.consecutive_wait_count)
        dynamic_cost *= (1.0 - decay)
    
    return max(0.3, dynamic_cost)
```

**需要修改的文件**：
- `demo/agent/model_decision_service.py`：在 `decide()` 中用动态机会成本替代固定值
- `demo/agent/config.py`：添加动态机会成本相关参数

**预期收益**：+5-15%（减少高价值区域的"接差单"和低价值区域的"空等"）

**真实场景合理性**：✅ 司机在订单充足的地方会更挑剔，在偏僻地方会降低要求。

---

### 2.3 方案三：按效率（元/分钟）选单

**核心思想**：当前按绝对分数选单。一个 500 元/8 小时的订单（1.04 元/分）打败了 200 元/2 小时的订单（1.67 元/分）。按效率选单能提高单位时间收入。

**实现方式**：

```python
# 在 model_decision_service.py 中修改订单排序逻辑

# 方案 A：直接用 score/time 排序（简单但可能有副作用）
best = max(
    feasible_orders,
    key=lambda c: c.score / max(1, c.occupied_minutes) if c.score > 0 else c.score
)

# 方案 B：混合排序（更稳健）
# 只在多个订单分数接近时（差距 < 20%）使用效率排序
top_orders = [c for c in feasible_orders if c.score > best_score * 0.8]
if len(top_orders) > 1:
    best = max(top_orders, key=lambda c: c.score / max(1, c.occupied_minutes))
```

**需要修改的文件**：
- `demo/agent/scoring.py`：在 `ScoredAction` 中添加 `occupied_minutes` 字段
- `demo/agent/model_decision_service.py`：修改 `best = max(feasible, ...)` 的选择逻辑

**预期收益**：+5-10%（同样时间接更多高效率订单）

**真实场景合理性**：✅ 司机自然会算"这单划不划算"，优先选时间短、收入高的。

**注意事项**：
- `occupied_minutes` 需要从 `score_take_order` 传出（当前只在函数内部使用）
- 要确保 `occupied_minutes` 包含空驶+等货+运输全部时间

---

### 2.4 方案四：智能空驶到高价值区域

**核心思想**：当前空闲时只原地等待或去偏好目标。应该主动移动到历史上好找货的区域。

**实现方式**：

```python
# 在 scoring.py 的 build_reposition_candidates 中添加

def get_high_value_reposition_targets(memory: DriverMemory, ctx: DecisionContext) -> list[tuple[float, float]]:
    """
    根据历史接单经验，推荐高价值空驶目标。
    
    从 hotspot 记录中找到距离合理（<100km）且历史收益好的区域。
    """
    targets = []
    for hotspot in memory.hotspots:
        dist = haversine_km(ctx.current_lat, ctx.current_lng, hotspot.lat, hotspot.lng)
        if 20 < dist < 100:  # 不去太近（已经在附近了）也不去太远
            # 基于历史收益排序
            targets.append((hotspot.lat, hotspot.lng, hotspot.avg_income))
    
    # 返回 top-3 高价值目标
    targets.sort(key=lambda t: t[2], reverse=True)
    return [(t[0], t[1]) for t in targets[:3]]
```

**需要修改的文件**：
- `demo/agent/scoring.py`：在 `build_reposition_candidates` 中添加高价值区域候选
- `demo/agent/driver_memory.py`：增强 hotspot 记录（添加平均收益统计）

**预期收益**：+5-10%（减少 D009 等高等待司机的空闲时间）

**真实场景合理性**：✅ 司机跑熟了会知道哪个区域好找货，空闲时主动去那边等。

---

### 2.5 方案五：在线学习（模拟中积累经验）

**核心思想**：在 31 天仿真过程中，逐步积累对各区域、各时段的订单质量认知，用于优化后续决策。

**实现方式**：

```python
# 在 driver_memory.py 中添加

@dataclass
class LocationTimeStats:
    """某区域某时段的订单统计"""
    grid_key: str  # 如 "23.1_113.5" (0.1度网格)
    hour_bucket: int  # 0-23
    total_income: float = 0.0
    total_orders: int = 0
    total_minutes: float = 0.0
    
    @property
    def avg_rate(self) -> float:
        """平均元/分钟"""
        return self.total_income / max(1, self.total_minutes)


class DriverMemory:
    # 新增字段
    location_stats: dict[str, LocationTimeStats] = field(default_factory=dict)
    
    def record_order_completion(self, lat: float, lng: float, hour: int, 
                                  income: float, minutes: float):
        """记录完成订单的统计信息"""
        grid_key = f"{lat:.1f}_{lng:.1f}_{hour}"
        if grid_key not in self.location_stats:
            self.location_stats[grid_key] = LocationTimeStats(grid_key, hour)
        stats = self.location_stats[grid_key]
        stats.total_income += income
        stats.total_orders += 1
        stats.total_minutes += minutes
    
    def get_location_value(self, lat: float, lng: float, hour: int) -> float:
        """查询某位置某时段的历史收益率"""
        grid_key = f"{lat:.1f}_{lng:.1f}_{hour}"
        stats = self.location_stats.get(grid_key)
        if stats and stats.total_orders >= 2:  # 至少2单才有统计意义
            return stats.avg_rate
        return 0.0  # 未知区域返回0
```

**需要修改的文件**：
- `demo/agent/driver_memory.py`：添加 `LocationTimeStats` 和相关方法
- `demo/agent/model_decision_service.py`：在订单完成后调用 `record_order_completion`
- `demo/agent/scoring.py`：在评分时查询 `get_location_value`

**预期收益**：+3-8%（仿真后期决策质量逐步提升）

**真实场景合理性**：✅ 司机"跑熟了"就是在线学习，对各区域的订单情况越来越了解。

---

## 3. 实施路线图

### 第一阶段（预计 +15-25%，目标 300k）

1. **实施方案二**（动态机会成本）— 改动最小，风险最低
2. **实施方案三**（效率选单）— 与方案二互补，一起改
3. 仿真验证，调参

### 第二阶段（预计再 +10-20%，目标 330k-350k）

4. **实施方案一**（订单链评估）— 需要扩展 DriverMemory
5. **实施方案四**（智能空驶）— 依赖方案一的位置价值数据
6. 仿真验证，调参

### 第三阶段（预计再 +5-10%，目标 350k-380k）

7. **实施方案五**（在线学习）— 进一步精细化
8. 针对各司机的个性化参数微调
9. 仿真验证

### 达到 400k 的额外考虑

如果以上方案组合仍无法达到 400k，可能需要：
- 研究达到 400k 的方案的具体做法（是否有公开分享）
- 考虑是否允许更激进的策略（如忽略某些低价值罚分）
- 多轮仿真取最优结果（利用随机性）

---

## 4. 关键约束（必须遵守）

1. **不能预分析货源数据**：所有决策必须基于当前观察和历史经验
2. **不能硬阻拦订单**：使用软惩罚，让评分系统自动权衡
3. **符合真实场景**：司机不能预知未来货源，只能根据经验和当前信息决策
4. **不修改仿真框架**：只修改 agent 决策逻辑
5. **保持鲁棒性**：优化方案在不同数据集上都应有效

---

## 5. 文件修改清单

| 文件 | 方案 | 改动内容 |
|------|------|----------|
| `demo/agent/config.py` | 全部 | 新增参数常量 |
| `demo/agent/driver_memory.py` | 1,4,5 | 添加 completed_orders、LocationTimeStats |
| `demo/agent/scoring.py` | 1,3,4 | 添加 chain_value 评分、效率排序、空驶候选 |
| `demo/agent/model_decision_service.py` | 2,3,5 | 动态机会成本、效率选单、在线学习调用 |

---

## 6. 已验证有效的基础（保留在 master）

以下改动已验证有效（PR #29），是后续优化的基础：

1. **二次评估系统**（`model_decision_service.py` 第 122-141 行）
   - 首次无正分订单时，以 0.5× 机会成本重新评分
   - 贡献约 +2,000-3,000 元净收入

2. **LLM 决策未启用**（纯规则评分）
   - 消除 LLM 引入的 ±6,000 元随机波动
   - 贡献约 +3,000 元稳定性提升
