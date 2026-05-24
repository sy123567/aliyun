#!/usr/bin/env python3
"""Comprehensive analysis of simulation results for driver logistics business."""

import json
import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE = "/home/ubuntu/sim_analysis/demo/results"

# ===================== 1. Load current run =====================
with open(f"{BASE}/monthly_income_202603.json") as f:
    current = json.load(f)

with open(f"{BASE}/run_summary_202603.json") as f:
    summary = json.load(f)

# ===================== 2. Parse all driver data =====================
drivers = []
for d in current["drivers"]:
    inc = d["income"]
    pref = d["preference_check"]
    total_penalty = inc["preference_penalty"]
    violations = []
    for r in pref["rules"]:
        v = r.get("violations", 0)
        p = r.get("penalty", 0.0)
        rule_name = r["rule"]
        violations.append({
            "driver_id": d["driver_id"],
            "rule": rule_name,
            "violations": v,
            "penalty": p,
            "preference_text": r.get("preference_text", "")
        })
    drivers.append({
        "driver_id": d["driver_id"],
        "gross_income": inc["gross_income"],
        "distance_km": inc["distance_km"],
        "cost": inc["cost"],
        "preference_penalty": inc["preference_penalty"],
        "net_income": inc["net_income"],
        "cost_per_km": inc["cost"] / inc["distance_km"] if inc["distance_km"] > 0 else 0,
        "income_per_km": inc["gross_income"] / inc["distance_km"] if inc["distance_km"] > 0 else 0,
        "net_per_km": inc["net_income"] / inc["distance_km"] if inc["distance_km"] > 0 else 0,
        "penalty_pct_of_gross": inc["preference_penalty"] / inc["gross_income"] * 100 if inc["gross_income"] > 0 else 0,
        "cost_pct_of_gross": inc["cost"] / inc["gross_income"] * 100 if inc["gross_income"] > 0 else 0,
    })

df_drivers = pd.DataFrame(drivers)
print("=" * 80)
print("【当前运行 - 司机收入概览】")
print("=" * 80)
print(df_drivers.to_string(index=False))
print()

# Summary stats
total_gross = df_drivers["gross_income"].sum()
total_cost = df_drivers["cost"].sum()
total_penalty = df_drivers["preference_penalty"].sum()
total_net = df_drivers["net_income"].sum()
total_dist = df_drivers["distance_km"].sum()

print(f"总毛收入: ¥{total_gross:,.2f}")
print(f"总成本: ¥{total_cost:,.2f} ({total_cost/total_gross*100:.1f}%)")
print(f"总偏好罚款: ¥{total_penalty:,.2f} ({total_penalty/total_gross*100:.1f}%)")
print(f"总净收入: ¥{total_net:,.2f}")
print(f"总行驶距离: {total_dist:,.2f} km")
print(f"整体毛利率: {(total_gross - total_cost)/total_gross*100:.1f}%")
print(f"净利率(扣除罚款): {total_net/total_gross*100:.1f}%")
print()

# ===================== 3. Parse all violations =====================
all_violations = []
for d in current["drivers"]:
    for r in d["preference_check"]["rules"]:
        all_violations.append({
            "driver_id": d["driver_id"],
            "rule": r["rule"],
            "violations": r.get("violations", 0),
            "penalty": r.get("penalty", 0.0),
            "preference_text": r.get("preference_text", "")
        })

df_violations = pd.DataFrame(all_violations)
print("=" * 80)
print("【失分详情 - 所有违规罚款】")
print("=" * 80)
penalties_only = df_violations[df_violations["penalty"] > 0].sort_values("penalty", ascending=False)
print(penalties_only[["driver_id", "rule", "violations", "penalty"]].to_string(index=False))
print()

# Group by rule
print("=" * 80)
print("【按规则分组的罚款汇总】")
print("=" * 80)
rule_summary = df_violations.groupby("rule").agg(
    total_penalty=("penalty", "sum"),
    total_violations=("violations", "sum"),
    affected_drivers=("driver_id", lambda x: sum(1 for _ in x if df_violations.loc[x.index, "penalty"].sum() > 0))
).sort_values("total_penalty", ascending=False)
rule_summary = rule_summary[rule_summary["total_penalty"] > 0]
print(rule_summary.to_string())
print()

# ===================== 4. Parse action-level data =====================
print("=" * 80)
print("【行动级别数据分析】")
print("=" * 80)

action_files = glob.glob(f"{BASE}/actions_202603_D*_20260524_094613.jsonl")
all_actions = []
for af in sorted(action_files):
    driver_id = os.path.basename(af).split("_")[1]
    with open(af) as f:
        for line in f:
            rec = json.loads(line.strip())
            rec["driver_id_parsed"] = driver_id
            all_actions.append(rec)

df_actions = pd.DataFrame(all_actions)
print(f"总行动记录: {len(df_actions)}")

# Action distribution
action_types = df_actions["action"].apply(lambda x: x["action"] if isinstance(x, dict) else "unknown")
df_actions["action_type"] = action_types
action_dist = df_actions["action_type"].value_counts()
print("\n行动类型分布:")
print(action_dist.to_string())

# Per driver action distribution
print("\n各司机行动类型分布:")
pivot = pd.crosstab(df_actions["driver_id"], df_actions["action_type"])
print(pivot.to_string())

# ===================== 5. Order-level analysis =====================
orders = df_actions[df_actions["action_type"] == "take_order"].copy()
orders["cargo_id"] = orders["action"].apply(lambda x: x["params"]["cargo_id"])
orders["pickup_deadhead_km"] = orders["result"].apply(lambda x: x.get("pickup_deadhead_km", 0) if isinstance(x, dict) else 0)
orders["haul_distance_km"] = orders["result"].apply(lambda x: x.get("haul_distance_km", 0) if isinstance(x, dict) else 0)
orders["income_eligible"] = orders["result"].apply(lambda x: x.get("income_eligible", True) if isinstance(x, dict) else True)

print("\n" + "=" * 80)
print("【订单级别分析】")
print("=" * 80)
print(f"总接单数: {len(orders)}")
print(f"总空驶距离: {orders['pickup_deadhead_km'].sum():,.2f} km")
print(f"总运货距离: {orders['haul_distance_km'].sum():,.2f} km")
total_deadhead = orders['pickup_deadhead_km'].sum()
total_haul = orders['haul_distance_km'].sum()
print(f"空驶比: {total_deadhead / (total_deadhead + total_haul) * 100:.1f}%")
print(f"平均空驶距离: {orders['pickup_deadhead_km'].mean():,.2f} km")
print(f"平均运货距离: {orders['haul_distance_km'].mean():,.2f} km")

# Per driver order stats
print("\n各司机订单统计:")
driver_orders = orders.groupby("driver_id").agg(
    order_count=("cargo_id", "count"),
    total_deadhead=("pickup_deadhead_km", "sum"),
    avg_deadhead=("pickup_deadhead_km", "mean"),
    total_haul=("haul_distance_km", "sum"),
    avg_haul=("haul_distance_km", "mean"),
).reset_index()
driver_orders["deadhead_ratio"] = driver_orders["total_deadhead"] / (driver_orders["total_deadhead"] + driver_orders["total_haul"])
print(driver_orders.to_string(index=False))

# ===================== 6. Wait/idle analysis =====================
waits = df_actions[df_actions["action_type"] == "wait"].copy()
waits["wait_minutes"] = waits["action"].apply(lambda x: x["params"]["duration_minutes"])
print("\n" + "=" * 80)
print("【等待/空闲分析】")
print("=" * 80)
driver_waits = waits.groupby("driver_id").agg(
    wait_count=("wait_minutes", "count"),
    total_wait_minutes=("wait_minutes", "sum"),
    avg_wait_minutes=("wait_minutes", "mean"),
).reset_index()
driver_waits["total_wait_hours"] = driver_waits["total_wait_minutes"] / 60
print(driver_waits.to_string(index=False))

# ===================== 7. Reposition analysis =====================
repos = df_actions[df_actions["action_type"] == "reposition"].copy()
if len(repos) > 0:
    repos["reposition_km"] = repos["result"].apply(lambda x: x.get("distance_km", 0) if isinstance(x, dict) else 0)
    print("\n" + "=" * 80)
    print("【空驶重定位分析】")
    print("=" * 80)
    driver_repos = repos.groupby("driver_id").agg(
        reposition_count=("reposition_km", "count"),
        total_reposition_km=("reposition_km", "sum"),
        avg_reposition_km=("reposition_km", "mean"),
    ).reset_index()
    print(driver_repos.to_string(index=False))

# ===================== 8. Historical comparison =====================
print("\n" + "=" * 80)
print("【历史运行对比】")
print("=" * 80)

hist_files = sorted(glob.glob(f"{BASE}/history/*/monthly_income_202603.json"))
hist_data = []
for hf in hist_files:
    run_id = hf.split("/history/")[1].split("/")[0]
    with open(hf) as f:
        hd = json.load(f)
    total_net = sum(d["income"]["net_income"] for d in hd["drivers"])
    total_gross = sum(d["income"]["gross_income"] for d in hd["drivers"])
    total_penalty = sum(d["income"]["preference_penalty"] for d in hd["drivers"])
    total_cost = sum(d["income"]["cost"] for d in hd["drivers"])
    total_dist = sum(d["income"]["distance_km"] for d in hd["drivers"])
    hist_data.append({
        "run_id": run_id,
        "total_net_income": total_net,
        "total_gross_income": total_gross,
        "total_cost": total_cost,
        "total_penalty": total_penalty,
        "total_distance_km": total_dist,
        "penalty_pct": total_penalty / total_gross * 100 if total_gross > 0 else 0,
        "cost_pct": total_cost / total_gross * 100 if total_gross > 0 else 0,
    })

# Add current run
hist_data.append({
    "run_id": "current",
    "total_net_income": df_drivers["net_income"].sum(),
    "total_gross_income": df_drivers["gross_income"].sum(),
    "total_cost": df_drivers["cost"].sum(),
    "total_penalty": df_drivers["preference_penalty"].sum(),
    "total_distance_km": df_drivers["distance_km"].sum(),
    "penalty_pct": df_drivers["preference_penalty"].sum() / df_drivers["gross_income"].sum() * 100,
    "cost_pct": df_drivers["cost"].sum() / df_drivers["gross_income"].sum() * 100,
})

df_hist = pd.DataFrame(hist_data)
print(df_hist.to_string(index=False))

# Best and worst runs
best_run = df_hist.loc[df_hist["total_net_income"].idxmax()]
worst_run = df_hist.loc[df_hist["total_net_income"].idxmin()]
print(f"\n最佳运行: {best_run['run_id']} (净收入: ¥{best_run['total_net_income']:,.2f})")
print(f"最差运行: {worst_run['run_id']} (净收入: ¥{worst_run['total_net_income']:,.2f})")
print(f"当前运行: 净收入 ¥{df_drivers['net_income'].sum():,.2f}")

# ===================== 9. Cost efficiency analysis =====================
print("\n" + "=" * 80)
print("【成本效率分析】")
print("=" * 80)
for _, d in df_drivers.iterrows():
    print(f"司机{d['driver_id']}: 毛收入¥{d['gross_income']:,.0f}, 成本¥{d['cost']:,.0f}({d['cost_pct_of_gross']:.1f}%), "
          f"罚款¥{d['preference_penalty']:,.0f}({d['penalty_pct_of_gross']:.1f}%), "
          f"净收入¥{d['net_income']:,.0f}, 里程{d['distance_km']:,.0f}km, "
          f"收入/km=¥{d['income_per_km']:.2f}, 净/km=¥{d['net_per_km']:.2f}")

# Save analysis data for visualization
df_hist.to_csv("/home/ubuntu/sim_analysis/hist_comparison.csv", index=False)
df_drivers.to_csv("/home/ubuntu/sim_analysis/driver_summary.csv", index=False)
penalties_only.to_csv("/home/ubuntu/sim_analysis/penalties.csv", index=False)
driver_orders.to_csv("/home/ubuntu/sim_analysis/driver_orders.csv", index=False)
driver_waits.to_csv("/home/ubuntu/sim_analysis/driver_waits.csv", index=False)

print("\nData saved to CSV files.")
