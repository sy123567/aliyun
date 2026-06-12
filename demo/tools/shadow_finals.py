"""影子复赛基准：本地可观测的"两位方言司机"整月仿真 + 逐条扣分明细。

背景：复赛平台只返回 4 个汇总数字（得分/扣分/token/耗时），拿不到任何明细；
本地又没有官方数据集（LFS 超额）与模型 API key。本工具自建一个最接近复赛
设定的本地测试台：

- 合成货源场：珠三角 + 江浙沪两个枢纽、92 天、约 1.8 万单（品类含水果/建材/
  生鲜/蔬菜/机械设备/普货，含 >8h 长途）；
- 合成司机：D101 广东司机（粤语偏好）、D102 江浙沪司机（吴语偏好），规则族
  与已知评分器一致：夜休窗（按天罚）、月度品类指标（按单罚）、月度长途上限
  （按单罚）、整月整休天数（一次性罚）、赴装空驶上限（按单罚）；
- 真实管线：复用 simkit + bench 的 CargoRepository / DriverStateManager /
  EmbeddedDecisionEnvironment / SimulationOrchestrator，与评测同构；
- 两种模型模式：
    scripted —— 进程内脚本化"完美解析"LLM（抽取/审计/转写按既知答案应答，
                决策调用返回空让确定性调度器接管）。测的是**执行层合规性**：
                解析正确的前提下，调度器能不能把扣分压到 0、同时挣到毛利；
    offline  —— 所有模型调用失败。测的是无模型时的正则兜底地板。
- 评分：按 calc_monthly_income.py 的同款语义（步起点折算等待覆盖、按
  cargo_id 回查原始数据）对动作日志逐条评估，输出每司机每规则的扣分明细。

用法：
    python demo/tools/shadow_finals.py            # 两种模式都跑
    python demo/tools/shadow_finals.py scripted   # 只跑脚本化模式
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DEMO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_DEMO_ROOT), str(_DEMO_ROOT / "server")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.model_decision_service import ModelDecisionService  # noqa: E402
from bench.embedded_agent import (  # noqa: E402
    EmbeddedAgentDecisionEngine,
    EmbeddedDecisionEnvironment,
)
from bench.simulation_orchestrator import SimulationOrchestrator  # noqa: E402
from simkit.cargo_repository import CargoRepository  # noqa: E402
from simkit.driver_state_manager import DriverStateManager  # noqa: E402

EPOCH = datetime(2026, 3, 1, 0, 0, 0)
DAYS = 92
DAY_MIN = 1440
HORIZON_MIN = DAYS * DAY_MIN
COST_PER_KM = 1.5
MONTH_RANGES = {3: (0, 31), 4: (31, 61), 5: (61, 92)}  # month → [start_day, end_day)

DATA_DIR = Path(__file__).resolve().parent / "shadow_data"
RESULTS_BASE = Path(__file__).resolve().parent / "shadow_results"

PRD_CITIES = [  # 珠三角
    ("广州", 23.13, 113.26), ("佛山", 23.02, 113.12), ("东莞", 23.02, 113.75),
    ("深圳", 22.55, 114.06), ("惠州", 23.11, 114.41), ("中山", 22.52, 113.39),
    ("肇庆", 23.05, 112.47), ("清远", 23.68, 113.06),
]
YRD_CITIES = [  # 江浙沪
    ("上海", 31.23, 121.47), ("苏州", 31.30, 120.58), ("无锡", 31.49, 120.31),
    ("杭州", 30.27, 120.16), ("嘉兴", 30.75, 120.76), ("南通", 31.98, 120.89),
    ("宁波", 29.87, 121.55), ("常州", 31.81, 119.97),
]
CATEGORIES = [("水果", 0.14), ("建材", 0.14), ("蔬菜", 0.10), ("生鲜", 0.10),
              ("机械设备", 0.08), ("普货", 0.44)]


def _wall(minutes: int) -> str:
    return (EPOCH + timedelta(minutes=int(minutes))).strftime("%Y-%m-%d %H:%M:%S")


def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    rad = math.radians
    dlat, dlng = rad(lat2 - lat1), rad(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(rad(lat1)) * math.cos(rad(lat2)) * math.sin(dlng / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


# --------------------------------------------------------------- data builder

def _pick_category(rng: random.Random) -> str:
    x, acc = rng.random(), 0.0
    for name, w in CATEGORIES:
        acc += w
        if x < acc:
            return name
    return "普货"


def generate_dataset(path: Path, seed: int = 7, per_hub_day: int = 100) -> None:
    rng = random.Random(seed)
    cid = 0
    with path.open("w", encoding="utf-8") as f:
        for day in range(DAYS):
            for hub in (PRD_CITIES, YRD_CITIES):
                for _ in range(per_hub_day):
                    cid += 1
                    oc = rng.choice(hub)
                    dc = rng.choice([c for c in hub if c[0] != oc[0]])
                    olat = oc[1] + rng.uniform(-0.08, 0.08)
                    olng = oc[2] + rng.uniform(-0.08, 0.08)
                    dlat = dc[1] + rng.uniform(-0.08, 0.08)
                    dlng = dc[2] + rng.uniform(-0.08, 0.08)
                    if rng.random() < 0.10:  # 拉长部分单成 >8h 长途
                        dlat += (dlat - olat) * rng.uniform(2.0, 4.0)
                        dlng += (dlng - olng) * rng.uniform(2.0, 4.0)
                    haul = _haversine_km(olat, olng, dlat, dlng)
                    cost_time = int(math.ceil(haul / 55.0 * 60)) + rng.randint(60, 150)
                    price_yuan = haul * rng.uniform(2.2, 3.4) + rng.uniform(30, 80)
                    create = day * DAY_MIN + rng.randint(5 * 60 + 30, 21 * 60)
                    remove = create + rng.randint(120, 480)
                    rec: dict[str, Any] = {
                        "cargo_id": f"SC{cid:06d}",
                        "cargo_name": _pick_category(rng),
                        "price": int(price_yuan * 100),  # 单位:分
                        "cost_time_minutes": cost_time,
                        "create_time": _wall(create),
                        "remove_time": _wall(remove),
                        "start": {"lat": round(olat, 5), "lng": round(olng, 5), "city": oc[0]},
                        "end": {"lat": round(dlat, 5), "lng": round(dlng, 5), "city": dc[0]},
                    }
                    if rng.random() < 0.25:
                        rec["load_time"] = [_wall(create + 40), _wall(create + rng.randint(180, 360))]
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------- drivers

P_GD_NIGHT = ("晚黑十点之后就要收车熄火唞觉，唔好再接单唔好走车，挨到第二朝早六点先至好开工。", 2600)
P_GD_CAT = ("三月里头生果要拉够十单，少一单扣一次钱。", 800)
P_GD_LH = ("唔钟意嗰啲一跑就成日嘅远活，每个月超过八个钟头嘅长途最多接四单，多一单扣一次。", 1000)
P_GD_OFF = ("每个月起码两日要完全唔开工，留喺屋企陪屋企人。", 3000)
P_WU_NIGHT = ("夜里向十一点到第二天早浪向五点，覅跑车覅接单，停勒原地困觉。", 2400)
P_WU_CAT = ("三月里向建材生意要做满八票，少一票扣一趟钞票。", 900)
P_WU_DH = ("去装货个地方空驶要是超过五十公里，格种单子覅接。", 600)
P_WU_OFF = ("一个月里向至少要有两天完全歇着勿出车，屋里厢待牢。", 3000)

MANDARIN = {
    P_GD_NIGHT[0]: "每天晚上22点之后必须收车熄火睡觉，不得再接单或空驶，到第二天早上6点才可以开工。",
    P_GD_CAT[0]: "三月份货源类型是水果的货必须接满10单，少一单扣一次钱。",
    P_GD_LH[0]: "不喜欢一跑就一整天的远活，每个月超过8小时的长途最多接4单，多一单扣一次。",
    P_GD_OFF[0]: "每个月至少要有2天完全不出车，留在家里陪家人。",
    P_WU_NIGHT[0]: "每天晚上23点到第二天早上5点，不得跑车不得接单，停在原地睡觉。",
    P_WU_CAT[0]: "三月份货源类型是建材的货要接满8单，少一单扣一次钱。",
    P_WU_DH[0]: "去装货地点的空驶距离超过50公里的订单不接。",
    P_WU_OFF[0]: "每个月至少要有2天完全休息不出车，待在家里。",
}

EXTRACT_GT = {  # 按原文片段分发的"完美抽取"结果
    "晚黑十点": {"rest_window": {"start_hour": 22, "end_hour": 6},
                "no_drive_windows": [{"start_hour": 22, "end_hour": 6}]},
    "生果要拉够十单": {"monthly_category_targets": [
        {"month": 3, "category": "水果", "min_orders": 10, "carryover": False}]},
    "超过八个钟头": {"monthly_longhaul_cap": {"max_orders": 4, "min_hours": 8}},
    "两日要完全唔开工": {"off_days_min": 2},
    "夜里向十一点": {"rest_window": {"start_hour": 23, "end_hour": 5},
                  "no_drive_windows": [{"start_hour": 23, "end_hour": 5}]},
    "建材生意要做满八票": {"monthly_category_targets": [
        {"month": 3, "category": "建材", "min_orders": 8, "carryover": False}]},
    "超过五十公里": {"pickup_max_km": 50},
    "两天完全歇着": {"off_days_min": 2},
}


def _pref(content: str, amount: float, start: str = "2026-03-01 00:00:00",
          end: str = "2026-05-31 23:59:59") -> dict[str, Any]:
    return {"content": content, "start_time": start, "end_time": end,
            "penalty_amount": amount, "penalty_cap": None}


def generate_drivers(path: Path) -> None:
    drivers = [
        {
            "driver_id": "D101", "name": "阿强", "vehicle_no": "粤A88888",
            "truck_length": "4.2米", "cost_per_km": COST_PER_KM,
            "current_lat": 23.13, "current_lng": 113.26,
            "preferences": [
                _pref(*P_GD_NIGHT),
                _pref(P_GD_CAT[0], P_GD_CAT[1], end="2026-03-31 23:59:59"),
                _pref(*P_GD_LH),
                _pref(*P_GD_OFF),
            ],
        },
        {
            "driver_id": "D102", "name": "老周", "vehicle_no": "沪B66666",
            "truck_length": "4.2米", "cost_per_km": COST_PER_KM,
            "current_lat": 31.23, "current_lng": 121.47,
            "preferences": [
                _pref(*P_WU_NIGHT),
                _pref(P_WU_CAT[0], P_WU_CAT[1], end="2026-03-31 23:59:59"),
                _pref(*P_WU_DH),
                _pref(*P_WU_OFF),
            ],
        },
    ]
    path.write_text(json.dumps(drivers, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------ mock LLM 环境

def _resp(obj: Any) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": json.dumps(obj, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class ShadowEnvironment(EmbeddedDecisionEnvironment):
    """脚本化/离线两用模型环境：scripted 按既知答案应答，offline 一律失败。"""

    def __init__(self, repo, manager, mode: str, **kwargs):
        super().__init__(repo, manager, model_gateway=None, **kwargs)
        self._mode = mode

    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._mode == "offline":
            raise RuntimeError("offline mode: model unavailable")
        system = payload["messages"][0]["content"]
        user = payload["messages"][-1]["content"]
        if "方言转写助手" in system:
            return _resp({"mandarin": MANDARIN.get(user.strip(), user.strip())})
        if "偏好抽取器" in system:
            for key, gt in EXTRACT_GT.items():
                if key in user:
                    return _resp(gt)
            return _resp({})
        if "覆盖审计器" in system:
            n = len(json.loads(user).get("preferences", []))
            return _resp({"audits": [{"index": i, "covered": True} for i in range(n)]})
        if "每日合规计划助手" in system:
            if "晚黑" in user:
                return _resp({"no_drive_today": [{"start_hour": 22, "end_hour": 6}],
                              "replaces_default": False, "today_plan": "", "category_focus": None})
            if "夜里向" in user:
                return _resp({"no_drive_today": [{"start_hour": 23, "end_hour": 5}],
                              "replaces_default": False, "today_plan": "", "category_focus": None})
            return _resp({"no_drive_today": [], "replaces_default": False,
                          "today_plan": "", "category_focus": None})
        if "是否确实包含某条约束" in system:
            return _resp({"holds": True})
        if "语义判定助手" in system:
            return _resp({"answer": False})
        # 决策调用：返回无 action 的对象 → 回退确定性调度器
        return _resp({})


# ------------------------------------------------------------------- scoring

def _load_ctxs(actions_file: Path) -> list[dict[str, Any]]:
    ctxs, prev_end = [], 0
    with actions_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            scan = int(rec.get("query_scan_cost_minutes", 0))
            execm = int(rec.get("action_exec_cost_minutes", 0))
            result = rec.get("result") or {}
            end = int(result.get("simulation_progress_minutes", prev_end + scan + execm))
            action = rec.get("action") or {}
            ctxs.append({
                "name": str(action.get("action", "")).lower(),
                "params": action.get("params") or {},
                "result": result,
                "step_start": prev_end,
                "action_start": prev_end + scan,
                "action_end": prev_end + scan + execm,
                "step_end": end,
                "exec": execm,
            })
            prev_end = end
    return ctxs


def _overlap(a1, a2, b1, b2) -> bool:
    return a1 < b2 and a2 > b1


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _eval_night(ctxs, start_h: int, end_h: int, amount: float) -> tuple[float, int]:
    violations = 0
    for day in range(DAYS):
        ws = day * DAY_MIN + start_h * 60
        we = (day * DAY_MIN if end_h > start_h else (day + 1) * DAY_MIN) + end_h * 60
        active = any(c["name"] in ("take_order", "reposition")
                     and _overlap(c["action_start"], c["action_end"], ws, we) for c in ctxs)
        waits = [(max(c["step_start"], ws), min(c["step_end"], we))
                 for c in ctxs if c["name"] == "wait" and c["exec"] > 0]
        covered = sum(e - s for s, e in _merge([(s, e) for s, e in waits if e > s]))
        if active or covered < (we - ws):
            violations += 1
    return violations * amount, violations


def _accepted_orders(ctxs, cargo_map):
    out = []
    for c in ctxs:
        if c["name"] != "take_order" or not c["result"].get("accepted"):
            continue
        cargo = cargo_map.get(str(c["params"].get("cargo_id", "")))
        if cargo:
            out.append((c, cargo))
    return out


def _eval_category(orders, month: int, cat: str, need: int, amount: float):
    lo, hi = MONTH_RANGES[month]
    got = sum(1 for c, cargo in orders
              if lo <= c["action_start"] // DAY_MIN < hi and cargo["cargo_name"] == cat)
    short = max(0, need - got)
    return short * amount, short, got


def _eval_longhaul(orders, cap: int, amount: float):
    by_m: dict[int, int] = {}
    for c, cargo in orders:
        if int(cargo["cost_time_minutes"]) > 480:
            day = c["action_start"] // DAY_MIN
            for m, (lo, hi) in MONTH_RANGES.items():
                if lo <= day < hi:
                    by_m[m] = by_m.get(m, 0) + 1
    excess = sum(max(0, n - cap) for n in by_m.values())
    return excess * amount, excess, by_m


def _eval_deadhead(orders, limit_km: float, amount: float):
    n = sum(1 for c, _ in orders
            if float(c["result"].get("pickup_deadhead_km", 0) or 0) > limit_km + 1e-6)
    return n * amount, n


def _eval_offdays(ctxs, min_days: int, amount: float):
    active_days = {c["action_start"] // DAY_MIN for c in ctxs
                   if c["name"] in ("take_order", "reposition")}
    penalty, detail = 0.0, {}
    for m, (lo, hi) in MONTH_RANGES.items():
        idle = sum(1 for d in range(lo, hi) if d not in active_days)
        detail[m] = idle
        if idle < min_days:
            penalty += amount
    return penalty, detail


def _income(ctxs, cargo_map) -> tuple[float, float]:
    gross, dist = 0.0, 0.0
    for c in ctxs:
        r = c["result"]
        if c["name"] == "reposition":
            dist += float(r.get("distance_km", 0) or 0)
        elif c["name"] == "take_order" and r.get("accepted"):
            cargo = cargo_map.get(str(c["params"].get("cargo_id", "")))
            dist += float(r.get("pickup_deadhead_km", 0) or 0) + float(r.get("haul_distance_km", 0) or 0)
            if cargo and r.get("income_eligible", True) and c["step_end"] <= HORIZON_MIN:
                gross += cargo["price"] / 100.0
    return gross, dist


def score(results_dir: Path, cargo_map) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for driver_id, specs in {
        "D101": [("夜休22-06", "night", (22, 6, 2600)),
                 ("三月水果≥10", "cat", (3, "水果", 10, 800)),
                 ("月长途>8h≤4", "lh", (4, 1000)),
                 ("月整休≥2天", "off", (2, 3000))],
        "D102": [("夜休23-05", "night", (23, 5, 2400)),
                 ("三月建材≥8", "cat", (3, "建材", 8, 900)),
                 ("空驶>50km单", "dh", (50.0, 600)),
                 ("月整休≥2天", "off", (2, 3000))],
    }.items():
        files = sorted(results_dir.glob(f"actions_202603_{driver_id}_*.jsonl"))
        if not files:
            out[driver_id] = {"error": "no actions file"}
            continue
        ctxs = _load_ctxs(files[-1])
        orders = _accepted_orders(ctxs, cargo_map)
        rules = []
        total_pen = 0.0
        for label, kind, args in specs:
            if kind == "night":
                pen, v = _eval_night(ctxs, args[0], args[1], args[2])
                rules.append((label, pen, f"{v}天违规"))
            elif kind == "cat":
                pen, short, got = _eval_category(orders, *args)
                rules.append((label, pen, f"已接{got}单 缺{short}"))
            elif kind == "lh":
                pen, excess, by_m = _eval_longhaul(orders, *args)
                rules.append((label, pen, f"超额{excess} 按月{by_m}"))
            elif kind == "dh":
                pen, n = _eval_deadhead(orders, *args)
                rules.append((label, pen, f"{n}单超限"))
            elif kind == "off":
                pen, detail = _eval_offdays(ctxs, *args)
                rules.append((label, pen, f"各月整休天数{detail}"))
            total_pen += rules[-1][1]
        gross, dist = _income(ctxs, cargo_map)
        cost = dist * COST_PER_KM
        out[driver_id] = {
            "steps": len(ctxs), "orders": len(orders),
            "gross": round(gross, 2), "distance_km": round(dist, 2),
            "cost": round(cost, 2), "penalty": round(total_pen, 2),
            "net": round(gross - cost - total_pen, 2),
            "rules": [(label, round(p, 2), note) for label, p, note in rules],
        }
    return out


# ------------------------------------------------------------------ run mode

def run_mode(mode: str) -> dict[str, Any]:
    results_dir = RESULTS_BASE / mode
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)

    repo = CargoRepository(DATA_DIR / "cargo_dataset.jsonl")
    repo.load()
    manager = DriverStateManager(DATA_DIR / "drivers.json")
    manager.load()
    session: dict[str, list[dict[str, Any]]] = {d: [] for d in manager.list_driver_ids()}
    env = ShadowEnvironment(repo, manager, mode, session_actions_by_driver=session)
    engine = EmbeddedAgentDecisionEngine(ModelDecisionService(env), env)
    orch = SimulationOrchestrator(
        cargo_repository=repo,
        driver_state_manager=manager,
        agent_decision=engine,
        results_dir=results_dir,
        reposition_speed_km_per_hour=60.0,
        simulation_max_steps=20000,
        simulation_duration_days=DAYS,
        session_actions_by_driver=session,
        decision_step_timeout_seconds=120.0,
    )
    manager.start_simulation_minutes(driver_id=None, progress_minutes=0)
    summary = orch.run()
    return {"summary": summary, "results_dir": results_dir}


def main(modes: list[str]) -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    logging.getLogger("bench").setLevel(logging.WARNING)
    logging.getLogger("agent").setLevel(logging.WARNING)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = DATA_DIR / "cargo_dataset.jsonl"
    if not dataset.exists():
        print("生成合成货源场 ...")
        generate_dataset(dataset)
    generate_drivers(DATA_DIR / "drivers.json")
    cargo_map = {}
    with dataset.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            cargo_map[rec["cargo_id"]] = rec

    for mode in modes:
        print(f"\n===== 模式: {mode} =====")
        info = run_mode(mode)
        failures = info["summary"].get("driver_failures") or {}
        if failures:
            print("!! 司机失败:", failures)
        report = score(info["results_dir"], cargo_map)
        for driver_id, r in report.items():
            if "error" in r:
                print(f"[{driver_id}] {r['error']}")
                continue
            print(f"[{driver_id}] steps={r['steps']} orders={r['orders']} "
                  f"毛收入={r['gross']} 里程成本={r['cost']} 扣分={r['penalty']} 净={r['net']}")
            for label, pen, note in r["rules"]:
                flag = " <-- 扣分!" if pen > 0 else ""
                print(f"    {label}: -{pen} ({note}){flag}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args or ["scripted", "offline"])
