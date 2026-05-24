#!/usr/bin/env python3
"""Create comprehensive visualizations for driver logistics simulation analysis."""

import json
import glob
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ========== Theme Setup ==========
BG = '#141414'
GRID = '#2E2E2E'
TICK_TEXT = '#C7D2FE'
TITLE_COLOR = '#F3F4F6'
LEGEND_TEXT = '#D1D5DB'
ANNO_TEXT = '#9CA3AF'

PRIMARY = '#7A84FF'
SECONDARY = '#F29A45'
TERTIARY = '#A78CFF'
ACCENT = '#35C89A'
NEGATIVE = '#F53B3A'
LINK = '#3EB8ED'
NEUTRAL = '#6F8DA6'
EXTRA_COLORS = ['#DA9165', '#867EAA', '#6FB98C', '#7A84FF', '#F29A45', '#A78CFF']
PALETTE_10 = [PRIMARY, SECONDARY, TERTIARY, ACCENT, NEGATIVE, LINK, NEUTRAL, '#DA9165', '#867EAA', '#6FB98C']

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor': BG,
    'axes.edgecolor': GRID,
    'axes.labelcolor': TICK_TEXT,
    'xtick.color': TICK_TEXT,
    'ytick.color': TICK_TEXT,
    'text.color': TITLE_COLOR,
    'legend.facecolor': BG,
    'legend.edgecolor': GRID,
    'grid.color': GRID,
    'grid.alpha': 0.5,
    'font.size': 11,
    'font.family': 'Noto Sans CJK JP',
    'axes.unicode_minus': False,
})

BASE = "/home/ubuntu/sim_analysis/demo/results"
OUT = "/home/ubuntu/sim_analysis/charts"
os.makedirs(OUT, exist_ok=True)

# ========== Load Data ==========
with open(f"{BASE}/monthly_income_202603.json") as f:
    current = json.load(f)

drivers_data = []
all_violations = []
for d in current["drivers"]:
    inc = d["income"]
    drivers_data.append({
        "driver_id": d["driver_id"],
        "gross_income": inc["gross_income"],
        "distance_km": inc["distance_km"],
        "cost": inc["cost"],
        "penalty": inc["preference_penalty"],
        "net_income": inc["net_income"],
    })
    for r in d["preference_check"]["rules"]:
        all_violations.append({
            "driver_id": d["driver_id"],
            "rule": r["rule"],
            "violations": r.get("violations", 0),
            "penalty": r.get("penalty", 0.0),
        })

df_d = pd.DataFrame(drivers_data)
df_v = pd.DataFrame(all_violations)

# Load action data
action_files = sorted(glob.glob(f"{BASE}/actions_202603_D*_20260524_094613.jsonl"))
all_actions = []
for af in action_files:
    did = os.path.basename(af).split("_")[1]
    with open(af) as f:
        for line in f:
            rec = json.loads(line.strip())
            rec["_driver"] = did
            rec["_action_type"] = rec["action"]["action"] if isinstance(rec["action"], dict) else "unknown"
            all_actions.append(rec)
df_a = pd.DataFrame(all_actions)

# Load historical
hist_files = sorted(glob.glob(f"{BASE}/history/*/monthly_income_202603.json"))
hist_data = []
for hf in hist_files:
    run_id = hf.split("/history/")[1].split("/")[0]
    with open(hf) as f:
        hd = json.load(f)
    tn = sum(d["income"]["net_income"] for d in hd["drivers"])
    tg = sum(d["income"]["gross_income"] for d in hd["drivers"])
    tp = sum(d["income"]["preference_penalty"] for d in hd["drivers"])
    tc = sum(d["income"]["cost"] for d in hd["drivers"])
    hist_data.append({
        "run_id": run_id, "net": tn, "gross": tg, "penalty": tp, "cost": tc,
        "penalty_pct": tp/tg*100 if tg > 0 else 0,
    })
hist_data.append({
    "run_id": "current", "net": df_d["net_income"].sum(), "gross": df_d["gross_income"].sum(),
    "penalty": df_d["penalty"].sum(), "cost": df_d["cost"].sum(),
    "penalty_pct": df_d["penalty"].sum()/df_d["gross_income"].sum()*100,
})
df_h = pd.DataFrame(hist_data)

# Filter out failed runs (negative net income)
df_h_valid = df_h[df_h["gross"] > 50000].copy()

# ==================== CHART 1: Driver Income Breakdown ====================
fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(df_d))
w = 0.25

bars1 = ax.bar(x - w, df_d["gross_income"], w, label='毛收入', color=PRIMARY, alpha=0.9)
bars2 = ax.bar(x, df_d["cost"], w, label='运营成本', color=SECONDARY, alpha=0.9)
bars3 = ax.bar(x + w, df_d["net_income"], w, label='净收入', color=ACCENT, alpha=0.9)

# Mark penalty on top
for i, row in df_d.iterrows():
    if row["penalty"] > 0:
        ax.annotate(f'罚款¥{row["penalty"]:,.0f}', 
                   xy=(i + w, row["net_income"]),
                   xytext=(0, 8), textcoords='offset points',
                   ha='center', fontsize=8, color=NEGATIVE, fontweight='bold')

ax.set_xlabel('司机编号', fontsize=12)
ax.set_ylabel('金额 (¥)', fontsize=12)
ax.set_title('各司机收入结构分解 (2026年3月)', fontsize=16, color=TITLE_COLOR, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(df_d["driver_id"])
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'¥{x:,.0f}'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/01_driver_income_breakdown.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 2: Penalty Waterfall ====================
pen_rules = df_v[df_v["penalty"] > 0].groupby("rule")["penalty"].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(14, 7))
colors = [NEGATIVE if p > 1000 else SECONDARY if p > 300 else NEUTRAL for p in pen_rules.values]
bars = ax.barh(range(len(pen_rules)), pen_rules.values, color=colors, alpha=0.9)
ax.set_yticks(range(len(pen_rules)))
ax.set_yticklabels(pen_rules.index, fontsize=10)
for i, (v, rule) in enumerate(zip(pen_rules.values, pen_rules.index)):
    ax.text(v + 30, i, f'¥{v:,.0f}', va='center', fontsize=10, color=TICK_TEXT)
ax.set_xlabel('罚款金额 (¥)', fontsize=12)
ax.set_title('按规则分类的罚款分布 — 主要失分原因', fontsize=16, color=TITLE_COLOR, pad=15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/02_penalty_by_rule.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 3: Cost Structure Pie ====================
total_gross = df_d["gross_income"].sum()
total_cost = df_d["cost"].sum()
total_penalty = df_d["penalty"].sum()
total_net = df_d["net_income"].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Overall structure
labels1 = ['净收入', '运营成本', '偏好罚款']
sizes1 = [total_net, total_cost, total_penalty]
colors1 = [ACCENT, SECONDARY, NEGATIVE]
wedges1, texts1, autotexts1 = ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%',
    startangle=90, textprops={'color': TICK_TEXT, 'fontsize': 11})
for at in autotexts1:
    at.set_color(TITLE_COLOR)
    at.set_fontweight('bold')
ax1.set_title('毛收入分配结构', fontsize=14, color=TITLE_COLOR, pad=15)

# Right: Penalty breakdown by driver
pen_by_driver = df_d[df_d["penalty"] > 0][["driver_id", "penalty"]].sort_values("penalty", ascending=False)
if len(pen_by_driver) > 0:
    wedges2, texts2, autotexts2 = ax2.pie(pen_by_driver["penalty"], labels=pen_by_driver["driver_id"],
        colors=PALETTE_10[:len(pen_by_driver)], autopct='%1.1f%%', startangle=90,
        textprops={'color': TICK_TEXT, 'fontsize': 10})
    for at in autotexts2:
        at.set_color(TITLE_COLOR)
        at.set_fontweight('bold')
ax2.set_title('罚款分布 (按司机)', fontsize=14, color=TITLE_COLOR, pad=15)

plt.tight_layout()
plt.savefig(f"{OUT}/03_cost_structure.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 4: Efficiency Scatter ====================
fig, ax = plt.subplots(figsize=(12, 7))
for i, row in df_d.iterrows():
    income_per_km = row["gross_income"] / row["distance_km"]
    net_per_km = row["net_income"] / row["distance_km"]
    c = NEGATIVE if row["penalty"] > 1000 else SECONDARY if row["penalty"] > 0 else ACCENT
    size = row["distance_km"] / 50
    ax.scatter(income_per_km, net_per_km, s=size, c=c, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.annotate(row["driver_id"], (income_per_km, net_per_km),
               xytext=(5, 5), textcoords='offset points', fontsize=10, color=TICK_TEXT)

ax.plot([2, 7], [2, 7], '--', color=GRID, alpha=0.5, label='无扣减线')
ax.set_xlabel('毛收入/公里 (¥/km)', fontsize=12)
ax.set_ylabel('净收入/公里 (¥/km)', fontsize=12)
ax.set_title('运营效率分析 — 毛收入 vs 净收入 (每公里)\n(圆圈大小=总里程, 红色=高罚款)', fontsize=14, color=TITLE_COLOR, pad=15)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/04_efficiency_scatter.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 5: Historical Trend ====================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=False)

# Filter to valid runs and sort
df_hv = df_h_valid.reset_index(drop=True)
idx = range(len(df_hv))

ax1.plot(idx, df_hv["net"], '-o', color=ACCENT, markersize=4, label='净收入', linewidth=1.5)
ax1.plot(idx, df_hv["gross"], '-s', color=PRIMARY, markersize=3, label='毛收入', linewidth=1)
ax1.fill_between(idx, df_hv["net"], alpha=0.1, color=ACCENT)
ax1.set_ylabel('金额 (¥)', fontsize=12)
ax1.set_title('历史运行净收入趋势 (有效运行)', fontsize=16, color=TITLE_COLOR, pad=15)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
ax1.grid(alpha=0.3)
# Mark current
if "current" in df_hv["run_id"].values:
    curr_idx = df_hv[df_hv["run_id"] == "current"].index[0]
    ax1.axvline(x=curr_idx, color=NEGATIVE, linestyle='--', alpha=0.5, label='当前运行')
    ax1.annotate(f'当前: ¥{df_hv.loc[curr_idx, "net"]:,.0f}', 
                xy=(curr_idx, df_hv.loc[curr_idx, "net"]),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color=NEGATIVE)

ax2.plot(idx, df_hv["penalty_pct"], '-o', color=NEGATIVE, markersize=4, linewidth=1.5)
ax2.fill_between(idx, df_hv["penalty_pct"], alpha=0.1, color=NEGATIVE)
ax2.set_xlabel('运行序号', fontsize=12)
ax2.set_ylabel('罚款占比 (%)', fontsize=12)
ax2.set_title('罚款率趋势', fontsize=14, color=TITLE_COLOR, pad=15)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax2.grid(alpha=0.3)
if "current" in df_hv["run_id"].values:
    ax2.axvline(x=curr_idx, color=NEGATIVE, linestyle='--', alpha=0.5)
    ax2.annotate(f'当前: {df_hv.loc[curr_idx, "penalty_pct"]:.1f}%',
                xy=(curr_idx, df_hv.loc[curr_idx, "penalty_pct"]),
                xytext=(10, 10), textcoords='offset points', fontsize=10, color=NEGATIVE)

plt.tight_layout()
plt.savefig(f"{OUT}/05_historical_trend.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 6: Driver Penalty Heatmap ====================
# Build a matrix: drivers x rules
all_rules = df_v["rule"].unique()
penalty_matrix = pd.DataFrame(0.0, index=df_d["driver_id"], columns=all_rules)
for _, row in df_v.iterrows():
    if row["penalty"] > 0:
        penalty_matrix.loc[row["driver_id"], row["rule"]] = row["penalty"]
# Drop zero columns
penalty_matrix = penalty_matrix.loc[:, (penalty_matrix > 0).any()]

fig, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(penalty_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
           linewidths=0.5, linecolor=GRID, cbar_kws={'label': '罚款金额 (¥)'},
           annot_kws={'fontsize': 9, 'color': BG})
ax.set_title('司机 × 违规规则 罚款热力图', fontsize=16, color=TITLE_COLOR, pad=15)
ax.set_xlabel('')
ax.set_ylabel('司机编号', fontsize=12)
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig(f"{OUT}/06_penalty_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 7: Action Type & Wait Analysis ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Action type distribution per driver (stacked bar)
pivot = pd.crosstab(df_a["_driver"], df_a["_action_type"])
pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
colors_map = {'take_order': ACCENT, 'wait': SECONDARY, 'reposition': NEGATIVE}
bottom = pd.Series(0.0, index=pivot_pct.index)
for col in ['take_order', 'wait', 'reposition']:
    if col in pivot_pct.columns:
        ax1.bar(pivot_pct.index, pivot_pct[col], bottom=bottom, 
               label={'take_order': '接单', 'wait': '等待', 'reposition': '重定位'}[col],
               color=colors_map[col], alpha=0.9)
        bottom += pivot_pct[col]

ax1.set_xlabel('司机编号', fontsize=12)
ax1.set_ylabel('占比 (%)', fontsize=12)
ax1.set_title('各司机行动类型比例', fontsize=14, color=TITLE_COLOR, pad=15)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT)
ax1.grid(axis='y', alpha=0.3)

# Right: Wait hours vs net income
waits = df_a[df_a["_action_type"] == "wait"].copy()
waits["wait_min"] = waits["action"].apply(lambda x: x["params"]["duration_minutes"])
wait_by_driver = waits.groupby("_driver")["wait_min"].sum() / 60
merged = pd.merge(df_d, wait_by_driver.reset_index().rename(columns={"_driver": "driver_id", "wait_min": "wait_hours"}), on="driver_id", how="left")
merged["wait_hours"] = merged["wait_hours"].fillna(0)

ax2.scatter(merged["wait_hours"], merged["net_income"], s=100, c=PRIMARY, alpha=0.8, edgecolors='white')
for _, row in merged.iterrows():
    ax2.annotate(row["driver_id"], (row["wait_hours"], row["net_income"]),
                xytext=(5, 5), textcoords='offset points', fontsize=10, color=TICK_TEXT)
# Trend line
if len(merged) > 1 and merged["wait_hours"].std() > 0:
    try:
        z = np.polyfit(merged["wait_hours"].values.astype(float), merged["net_income"].values.astype(float), 1)
        p = np.poly1d(z)
        x_trend = np.linspace(merged["wait_hours"].min(), merged["wait_hours"].max(), 100)
        ax2.plot(x_trend, p(x_trend), '--', color=NEGATIVE, alpha=0.5, label=f'趋势线')
    except Exception:
        pass
ax2.set_xlabel('总等待时间 (小时)', fontsize=12)
ax2.set_ylabel('净收入 (¥)', fontsize=12)
ax2.set_title('等待时间 vs 净收入', fontsize=14, color=TITLE_COLOR, pad=15)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/07_action_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 8: Deadhead Analysis ====================
orders = df_a[df_a["_action_type"] == "take_order"].copy()
orders["deadhead"] = orders["result"].apply(lambda x: x.get("pickup_deadhead_km", 0) if isinstance(x, dict) else 0)
orders["haul"] = orders["result"].apply(lambda x: x.get("haul_distance_km", 0) if isinstance(x, dict) else 0)

driver_eff = orders.groupby("_driver").agg(
    total_deadhead=("deadhead", "sum"),
    total_haul=("haul", "sum"),
    order_count=("deadhead", "count"),
).reset_index()
driver_eff["deadhead_ratio"] = driver_eff["total_deadhead"] / (driver_eff["total_deadhead"] + driver_eff["total_haul"]) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Stacked bar of deadhead vs haul
x = np.arange(len(driver_eff))
ax1.bar(x, driver_eff["total_haul"], label='运货距离', color=ACCENT, alpha=0.9)
ax1.bar(x, driver_eff["total_deadhead"], bottom=driver_eff["total_haul"], label='空驶距离', color=NEGATIVE, alpha=0.9)
ax1.set_xticks(x)
ax1.set_xticklabels(driver_eff["_driver"])
ax1.set_xlabel('司机编号', fontsize=12)
ax1.set_ylabel('距离 (km)', fontsize=12)
ax1.set_title('各司机运货 vs 空驶距离', fontsize=14, color=TITLE_COLOR, pad=15)
ax1.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
ax1.grid(axis='y', alpha=0.3)

# Right: Deadhead ratio bar
colors_bar = [NEGATIVE if r > 25 else SECONDARY if r > 15 else ACCENT for r in driver_eff["deadhead_ratio"]]
ax2.bar(driver_eff["_driver"], driver_eff["deadhead_ratio"], color=colors_bar, alpha=0.9)
ax2.axhline(y=driver_eff["deadhead_ratio"].mean(), color=PRIMARY, linestyle='--', label=f'平均: {driver_eff["deadhead_ratio"].mean():.1f}%')
for i, v in enumerate(driver_eff["deadhead_ratio"]):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9, color=TICK_TEXT)
ax2.set_xlabel('司机编号', fontsize=12)
ax2.set_ylabel('空驶比 (%)', fontsize=12)
ax2.set_title('各司机空驶比率', fontsize=14, color=TITLE_COLOR, pad=15)
ax2.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/08_deadhead_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 9: Revenue Composition Deep Dive ====================
fig, ax = plt.subplots(figsize=(14, 8))

# Waterfall chart for each driver
categories = ['毛收入', '运营成本', '偏好罚款', '净收入']
x = np.arange(len(df_d))
width = 0.7

for i, (_, row) in enumerate(df_d.iterrows()):
    gross = row["gross_income"]
    cost = row["cost"]
    penalty = row["penalty"]
    net = row["net_income"]
    
    # Stacked: net + cost + penalty = gross
    ax.bar(i, net, width, color=ACCENT, alpha=0.85, label='净收入' if i == 0 else '')
    ax.bar(i, cost, width, bottom=net, color=SECONDARY, alpha=0.85, label='运营成本' if i == 0 else '')
    ax.bar(i, penalty, width, bottom=net+cost, color=NEGATIVE, alpha=0.85, label='偏好罚款' if i == 0 else '')

ax.set_xticks(x)
ax.set_xticklabels(df_d["driver_id"])
ax.set_xlabel('司机编号', fontsize=12)
ax.set_ylabel('金额 (¥)', fontsize=12)
ax.set_title('毛收入构成瀑布图 — 净收入 + 成本 + 罚款', fontsize=16, color=TITLE_COLOR, pad=15)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT, fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
ax.grid(axis='y', alpha=0.3)

# Add percentage annotations
for i, (_, row) in enumerate(df_d.iterrows()):
    cost_pct = row["cost"] / row["gross_income"] * 100
    pen_pct = row["penalty"] / row["gross_income"] * 100
    net_pct = row["net_income"] / row["gross_income"] * 100
    ax.text(i, row["gross_income"] + 500, f'净利{net_pct:.0f}%', ha='center', fontsize=8, color=ACCENT)

plt.tight_layout()
plt.savefig(f"{OUT}/09_revenue_waterfall.png", dpi=150, bbox_inches='tight')
plt.close()

# ==================== CHART 10: Key Metrics Summary Dashboard ====================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Top left: Net income ranking
ax = axes[0, 0]
sorted_d = df_d.sort_values("net_income", ascending=True)
colors_rank = [ACCENT if n > 25000 else SECONDARY if n > 20000 else NEGATIVE for n in sorted_d["net_income"]]
ax.barh(sorted_d["driver_id"], sorted_d["net_income"], color=colors_rank, alpha=0.9)
ax.set_xlabel('净收入 (¥)', fontsize=10)
ax.set_title('净收入排名', fontsize=13, color=TITLE_COLOR)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'¥{x/1000:.0f}K'))
ax.grid(axis='x', alpha=0.3)

# Top middle: Cost ratio
ax = axes[0, 1]
cost_ratio = (df_d["cost"] / df_d["gross_income"] * 100).values
ax.bar(df_d["driver_id"], cost_ratio, color=[NEGATIVE if c > 35 else SECONDARY if c > 30 else ACCENT for c in cost_ratio])
ax.axhline(y=np.mean(cost_ratio), color=PRIMARY, linestyle='--', label=f'均值{np.mean(cost_ratio):.1f}%')
ax.set_ylabel('成本占比 (%)', fontsize=10)
ax.set_title('成本/毛收入比率', fontsize=13, color=TITLE_COLOR)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT, fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Top right: Net income per km
ax = axes[0, 2]
net_per_km = (df_d["net_income"] / df_d["distance_km"]).values
ax.bar(df_d["driver_id"], net_per_km, color=[ACCENT if n > 4 else SECONDARY if n > 3 else NEGATIVE for n in net_per_km])
ax.axhline(y=np.mean(net_per_km), color=PRIMARY, linestyle='--', label=f'均值¥{np.mean(net_per_km):.2f}')
ax.set_ylabel('¥/km', fontsize=10)
ax.set_title('每公里净收入', fontsize=13, color=TITLE_COLOR)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT, fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Bottom left: Order count
ax = axes[1, 0]
orders_by_driver = df_a[df_a["_action_type"] == "take_order"].groupby("_driver").size()
ax.bar(orders_by_driver.index, orders_by_driver.values, color=PRIMARY, alpha=0.9)
ax.axhline(y=orders_by_driver.mean(), color=ACCENT, linestyle='--', label=f'均值{orders_by_driver.mean():.0f}')
ax.set_ylabel('订单数', fontsize=10)
ax.set_title('各司机接单量', fontsize=13, color=TITLE_COLOR)
ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=LEGEND_TEXT, fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Bottom middle: Reposition distance
ax = axes[1, 1]
repos = df_a[df_a["_action_type"] == "reposition"].copy()
repos["dist"] = repos["result"].apply(lambda x: x.get("distance_km", 0) if isinstance(x, dict) else 0)
repo_by_driver = repos.groupby("_driver")["dist"].sum()
all_drivers = df_d["driver_id"]
repo_vals = [repo_by_driver.get(d, 0) for d in all_drivers]
ax.bar(all_drivers, repo_vals, color=[NEGATIVE if v > 500 else SECONDARY if v > 100 else ACCENT for v in repo_vals])
ax.set_ylabel('重定位距离 (km)', fontsize=10)
ax.set_title('空驶重定位总距离', fontsize=13, color=TITLE_COLOR)
ax.grid(axis='y', alpha=0.3)

# Bottom right: Penalty as % of net income
ax = axes[1, 2]
pen_pct_net = (df_d["penalty"] / df_d["net_income"] * 100).values
ax.bar(df_d["driver_id"], pen_pct_net, color=[NEGATIVE if p > 10 else SECONDARY if p > 3 else ACCENT for p in pen_pct_net])
ax.set_ylabel('罚款/净收入 (%)', fontsize=10)
ax.set_title('罚款对净收入影响程度', fontsize=13, color=TITLE_COLOR)
ax.grid(axis='y', alpha=0.3)

fig.suptitle('关键运营指标仪表盘', fontsize=18, color=TITLE_COLOR, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/10_dashboard.png", dpi=150, bbox_inches='tight')
plt.close()

print("All charts saved to:", OUT)
print("Charts created:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
