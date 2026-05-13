"""分析最近一次评测产物。运行后请手动删除。"""

from __future__ import annotations

import glob
import json

print("=== v2 优化后评测分析 ===\n")

total = orders = accepted = eligible = failed = waits = reposes = 0
haul_km = deadhead_km = 0.0
token_total = 0
per_driver = {}
horizon_overflow_takes = 0

for path in sorted(glob.glob("demo/results/actions_202603_D*.jsonl")):
    driver = path.split("D0")[-1].split("_")[0]
    driver_id = f"D0{driver}"
    d = per_driver.setdefault(driver_id, {"orders": 0, "ok": 0, "fail": 0, "wait": 0, "repos": 0, "elig": 0, "haul": 0.0, "dead": 0.0})
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            tok = o.get("token_usage", {}) or {}
            if isinstance(tok, dict):
                token_total += int(tok.get("total_tokens", 0))
            act = (o.get("action", {}) or {}).get("action", "")
            res = o.get("result", {}) or {}
            if act == "wait":
                waits += 1
                d["wait"] += 1
            elif act == "reposition":
                reposes += 1
                d["repos"] += 1
            elif act == "take_order":
                orders += 1
                d["orders"] += 1
                if res.get("accepted"):
                    accepted += 1
                    d["ok"] += 1
                    haul = float(res.get("haul_distance_km") or 0)
                    dead = float(res.get("pickup_deadhead_km") or 0)
                    haul_km += haul
                    deadhead_km += dead
                    d["haul"] += haul
                    d["dead"] += dead
                    if res.get("income_eligible"):
                        eligible += 1
                        d["elig"] += 1
                    else:
                        horizon_overflow_takes += 1
                else:
                    failed += 1
                    d["fail"] += 1

print(f"全局: 总步数={total}  接单尝试={orders}  成功={accepted}(其中可计费={eligible})")
print(f"        失败={failed}  休息={waits}  空驶={reposes}")
print(f"        干线累计={haul_km:.2f} km  空驶累计={deadhead_km:.2f} km")
print(f"        Token累计={token_total}")
print(f"        超期单(income_eligible=False)={horizon_overflow_takes}\n")

# 净收益估算（不含偏好）：仅可计费单按距离 cost 估算
print("=== 各司机简表（按 v2 优化后） ===")
print(f"{'driver':<6} {'order(ok/fail)':<16} {'elig':<5} {'wait':<5} {'repos':<6} {'haul_km':<10} {'dead_km':<10}")
for did, d in sorted(per_driver.items()):
    ok = d["ok"]
    fail = d["fail"]
    print(f"{did:<6} {ok}/{fail:<14} {d['elig']:<5} {d['wait']:<5} {d['repos']:<6} {d['haul']:<10.2f} {d['dead']:<10.2f}")

# 失败率
if orders > 0:
    fail_rate = failed / orders
    print(f"\n接单失败率: {fail_rate:.1%} ({failed}/{orders})")
if accepted > 0:
    elig_rate = eligible / accepted
    print(f"成功单可计费率: {elig_rate:.1%} ({eligible}/{accepted})")
