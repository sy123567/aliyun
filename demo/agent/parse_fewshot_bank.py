"""Few-shot example bank + retriever for the preference extractor (偏好抽取器).

Why this exists
---------------
The finals score two *unknown* drivers whose natural-language preferences differ
from the local 广东 D001 fixture. Strong teams feed the extractor a large, diverse
set of few-shot examples. Inlining hundreds of examples into the system prompt is
both token-heavy and actively harmful: it triggers "lost-in-the-middle" and
*over-extraction* — the model hallucinates constraints to match the examples,
then needlessly skips orders → lower gross income (our actual bottleneck).

So instead of a static block, this module keeps a large bank (>=400 examples)
covering every schema field x dialect (粤/吴/客家/口语) plus many zero-constraint
"钓鱼" negatives, and exposes :func:`select_examples`, which returns only the
top-K examples most relevant to the *current* driver's preference text (plus a
few fixed anchor negatives). The agent injects just those into the extractor
prompt. This gives the coverage of 400+ examples with a small, stable prompt.

Correctness by construction
---------------------------
Every example's ``out`` JSON is built from the SAME parameters as its ``in``
text via small template helpers, so labels cannot drift from the text. All
``out`` objects use only the 27 allowed schema fields (see ``SCHEMA_FIELDS``).

Pure Python: no network, no embeddings, no numpy. Deterministic ordering.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- schema -----

# The full set of fields the extractor may emit (mirrors _PARSE_SYSTEM + the
# keys consumed by ModelDecisionService._merge_llm_rules). The unit test asserts
# every example's `out` keys are a subset of this set.
SCHEMA_FIELDS: frozenset[str] = frozenset({
    "daily_rest_hours", "rest_window", "no_drive_windows", "off_days_min",
    "forbidden_categories", "avoid_categories", "allowed_categories",
    "forbidden_regions", "allowed_regions", "forbidden_zones", "bounded_area",
    "required_region", "must_visit", "pickup_max_km", "haul_max_km",
    "haul_min_km", "monthly_longhaul_cap", "monthly_deadhead_max_km",
    "daily_order_limit", "first_order_before_hour", "monthly_category_targets",
    "home_rule", "blackout", "dated_single", "dated_route", "rule_penalties",
    "rule_penalty_caps",
})


def _blank() -> dict[str, Any]:
    """The canonical all-default extractor output (21 core fields).

    Matches the shape of the existing curated examples (示例1). Optional fields
    (allowed_*, haul_min_km, monthly_longhaul_cap, rule_penalt*) are added only
    when a rule sets them, exactly like the hand-written examples did.
    """
    return {
        "daily_rest_hours": None,
        "rest_window": None,
        "no_drive_windows": [],
        "off_days_min": 0,
        "forbidden_categories": [],
        "avoid_categories": [],
        "forbidden_regions": [],
        "forbidden_zones": [],
        "bounded_area": None,
        "required_region": None,
        "must_visit": [],
        "pickup_max_km": None,
        "haul_max_km": None,
        "monthly_deadhead_max_km": None,
        "daily_order_limit": None,
        "first_order_before_hour": None,
        "monthly_category_targets": [],
        "home_rule": None,
        "blackout": [],
        "dated_single": [],
        "dated_route": [],
    }


def _ex(prefs: Any, anchor: bool = False,
        penalty_amounts: list[Any] | None = None,
        penalty_caps: list[Any] | None = None,
        **out_over: Any) -> dict[str, Any]:
    """Build one bank example. ``out`` = blank defaults overridden by kwargs."""
    if isinstance(prefs, str):
        prefs = [prefs]
    in_obj: dict[str, Any] = {"preferences": list(prefs)}
    if penalty_amounts is not None:
        in_obj["penalty_amounts"] = penalty_amounts
    if penalty_caps is not None:
        in_obj["penalty_caps"] = penalty_caps
    out = _blank()
    for k in out_over:
        if k not in SCHEMA_FIELDS:
            raise ValueError(f"unknown schema field in example: {k}")
    out.update(out_over)
    return {"in": in_obj, "out": out, "anchor": bool(anchor)}


# ---------------------------------------------------------- curated seeds ----
# The 22 hand-tuned examples that previously lived inline in _PARSE_SYSTEM,
# transcribed verbatim. These are the highest-value, edge-case examples and
# several double as retrieval anchors.

_CURATED: list[dict[str, Any]] = [
    # 示例1: night rest + forbidden category + dated blackout
    _ex(["每天零点到六点停着熄火睡觉", "凡是生鲜货源碰不得", "三月四号五号不往深圳（22.55，114.05）跑"],
        rest_window={"start_hour": 0, "end_hour": 6},
        no_drive_windows=[{"start_hour": 0, "end_hour": 6}],
        forbidden_categories=["生鲜"],
        blackout=[{"region": "深圳", "dates": [4, 5]}]),
    # 示例2: rest hours + dated single (盘库) + pickup cap
    _ex(["十二号得去仓库（23.15，113.67）盘库，花两小时", "连续休息满8小时", "空驶超过五十五公里别接"],
        daily_rest_hours=8, pickup_max_km=55,
        dated_single=[{"date": 12, "lat": 23.15, "lng": 113.67, "wait_minutes": 120, "before_hour": None}]),
    # 示例3: dated route (取礼物→赴宴)
    _ex(["三十一号先过档口（23.15，113.67）取礼物，中午十二点前赶到县城（23.35，112.47）赴宴到下午两点"],
        dated_route=[{"date": 31, "stops": [
            {"lat": 23.15, "lng": 113.67, "wait_minutes": 0, "before_hour": 12},
            {"lat": 23.35, "lng": 112.47, "wait_minutes": 120, "before_hour": 12}]}]),
    # 示例4: forbidden categories + forbidden region
    _ex(["龙门吊底座、机床铸件这类机械设备活儿干不了", "装货地或卸货地在惠州的货一律不接"],
        forbidden_categories=["龙门吊底座", "机床铸件", "机械设备"],
        forbidden_regions=["惠州"]),
    # 示例5: home_rule + daily_order_limit + cross-midnight ndw
    _ex(["每天23点前车辆须在自家位置（23.10，113.50）1公里内，到次日8点前不接单不空跑", "同一天接单不得超过3单"],
        no_drive_windows=[{"start_hour": 23, "end_hour": 8}],
        daily_order_limit=3,
        home_rule={"lat": 23.10, "lng": 113.50, "radius_km": 1, "home_by_hour": 23, "no_drive_until_hour": 8}),
    # 示例6: midday rest window (half hour) + dated single (对账)
    _ex(["十一点半到下午一点半歇晌，雷打不动", "二十号去老李仓库（23.25，113.40）对账，大概两小时"],
        rest_window={"start_hour": 11.5, "end_hour": 13.5},
        dated_single=[{"date": 20, "lat": 23.25, "lng": 113.40, "wait_minutes": 120, "before_hour": None}]),
    # 示例7: allowed_regions (粤语)
    _ex(["我净系喺长三角跑，江浙沪以外嘅货唔接"], anchor=True,
        allowed_regions=["长三角"]),
    # 示例8: allowed_categories
    _ex(["我只拉冷链冷藏货，其它品类一概不接"],
        allowed_categories=["冷链", "冷藏"]),
    # 示例9: monthly category targets + carryover
    _ex(["四月水果必须接满十二单，欠的下个月补上", "五月建材至少接十二单"],
        monthly_category_targets=[
            {"month": 4, "category": "水果", "min_orders": 12, "carryover": True},
            {"month": 5, "category": "建材", "min_orders": 12, "carryover": False}]),
    # 示例10: monthly longhaul cap
    _ex(["每月超过八小时的长途单最多接三单，多接一单罚一次"],
        monthly_longhaul_cap={"max_orders": 3, "min_hours": 8}),
    # 示例11: deadhead cap + haul_min
    _ex(["整月空驶里程不能超过两千公里", "干线低于一百公里的短活不接"],
        monthly_deadhead_max_km=2000, haul_min_km=100),
    # 示例12: forbidden zone
    _ex(["以市中心（31.23，121.47）为圆心半径三十公里以内禁止进入"],
        forbidden_zones=[{"lat": 31.23, "lng": 121.47, "radius_km": 30}]),
    # 示例13: bounded area
    _ex(["只在北纬二十二到二十四度、东经一百一十二到一百一十四度范围内运营"],
        bounded_area={"lat_min": 22, "lat_max": 24, "lng_min": 112, "lng_max": 114}),
    # 示例14: must_visit + required_region
    _ex(["每月至少有五天要到配送中心（30.59，114.30）十公里范围内", "每月起码三天在武汉接货"],
        required_region={"region": "武汉", "min_days": 3},
        must_visit=[{"lat": 30.59, "lng": 114.30, "radius_km": 10, "required_days": 5}]),
    # 示例15: haul_max + first order before hour
    _ex(["单趟运距不超过六百公里", "每天首单不能晚于早上九点"],
        haul_max_km=600, first_order_before_hour=9),
    # 示例16: off_days_min + avoid_categories
    _ex(["一个月里头至少歇四天不出车", "危化品尽量少接，能不拉就不拉"],
        off_days_min=4, avoid_categories=["危化品"]),
    # 示例17: Cantonese night rest (廿二=22)
    _ex(["夜晚黑廿二点之后到听日朝早六点，车唔郁，瞓觉"], anchor=True,
        rest_window={"start_hour": 22, "end_hour": 6},
        no_drive_windows=[{"start_hour": 22, "end_hour": 6}]),
    # 示例18: Wu-dialect night rest + pickup cap
    _ex(["夜里向十一点到第二天五点弗开车，困觉", "赴装空驶覅超过四十公里"],
        rest_window={"start_hour": 23, "end_hour": 5},
        no_drive_windows=[{"start_hour": 23, "end_hour": 5}],
        pickup_max_km=40),
    # 示例19: penalty attribution (night + forbidden cat)
    _ex(["夜里十点到早六点不出车", "生鲜一律不接"], penalty_amounts=[3000, 1500],
        no_drive_windows=[{"start_hour": 22, "end_hour": 6}],
        forbidden_categories=["生鲜"],
        rule_penalties={"night_window": 3000, "forbidden_categories": 1500}),
    # 示例20: penalty + cap attribution (forbidden region)
    _ex(["每违规接一单深圳的货扣五百，整月最多扣五千"], penalty_amounts=[500], penalty_caps=[5000],
        forbidden_regions=["深圳"],
        rule_penalties={"forbidden_regions": 500},
        rule_penalty_caps={"forbidden_regions": 5000}),
    # 示例21: home_rule (粤语) + cross-midnight ndw
    _ex(["每晚十点前要返到屋企（23.10，113.50）两公里内，听日朝早七点先出车"],
        no_drive_windows=[{"start_hour": 22, "end_hour": 7}],
        home_rule={"lat": 23.10, "lng": 113.50, "radius_km": 2, "home_by_hour": 22, "no_drive_until_hour": 7}),
    # 示例22: ZERO-constraint anchor (chit-chat → all defaults)
    _ex(["我开大车十几年咯，啥货都拉，哪儿都跑，没啥特别忌讳，给钱就走"], anchor=True),
]


# ----------------------------------------------------- generated examples ----
# Parameterised templates. Each generator yields _ex(...) so in-text and out-JSON
# share the same parameters. Numbers in generated text use Arabic digits (the
# model parses them directly); dialect/colloquial variety comes from sentence
# patterns and particles, not from spelling out numbers (keeps labels exact).

_COORDS: list[tuple[float, float]] = [
    (23.13, 113.26), (22.54, 114.06), (23.02, 113.75), (23.02, 113.12),
    (30.59, 114.30), (28.23, 112.94), (30.27, 120.16), (31.30, 120.62),
    (32.06, 118.80), (31.23, 121.47), (28.68, 115.86), (31.82, 117.23),
    (29.87, 121.55), (24.88, 118.58), (26.08, 119.30), (34.75, 113.62),
]


def _gen_night() -> list[dict[str, Any]]:
    """no_drive_windows (+ rest_window) — the highest-penalty daily rule."""
    out: list[dict[str, Any]] = []
    starts = [20, 21, 22, 23, 0, 1]
    ends = [5, 6, 7, 8]
    # (template, also_rest)  — {s}/{e} filled with Arabic hour
    templates: list[tuple[str, bool]] = [
        ("晚上{s}点到第二天{e}点不出车，停车睡觉", True),
        ("夜里{s}点过后到次日{e}点收车熄火，不接单不空跑", False),
        ("天黑{s}点以后就收工，{e}点天亮再开工", False),
        ("每天{s}点到{e}点歇着不动车，好好睡一觉", True),
        ("入夜{s}点之后到清早{e}点一律不揽货、不赶路", False),
        ("夜晚黑{s}点之后到听日朝早{e}点，车唔郁，瞓觉", True),      # 粤
        ("夜里向{s}点到第二天{e}点弗开车，困觉", True),               # 吴
        ("{s}点收车落锁，{e}点先再出车，夜里不跑", False),
    ]
    i = 0
    for s in starts:
        for e in ends:
            tmpl, also_rest = templates[i % len(templates)]
            i += 1
            text = tmpl.format(s=s, e=e)
            over: dict[str, Any] = {"no_drive_windows": [{"start_hour": s, "end_hour": e}]}
            if also_rest:
                over["rest_window"] = {"start_hour": s, "end_hour": e}
            out.append(_ex(text, **over))
    return out


def _gen_forbidden_categories() -> list[dict[str, Any]]:
    cats = ["生鲜", "冷链货", "危化品", "易燃品", "活禽", "牲畜", "玻璃制品",
            "机械设备", "钢材", "煤炭", "水泥", "化工品", "医药", "鲜花"]
    templates = ["{c}一律不接", "凡是{c}碰都不碰", "{c}这类货干不了", "我不拉{c}", "{c}的活儿一概不接"]
    out: list[dict[str, Any]] = []
    for idx, c in enumerate(cats):
        for t in (templates[idx % len(templates)], templates[(idx + 2) % len(templates)]):
            out.append(_ex(t.format(c=c), forbidden_categories=[c]))
    return out


def _gen_avoid_categories() -> list[dict[str, Any]]:
    cats = ["危化品", "活禽", "生鲜", "玻璃制品", "牲畜", "冷链货", "钢材", "医药", "鲜花", "煤炭"]
    templates = ["{c}尽量少接，能不拉就不拉", "{c}不太想接，麻烦", "尽量避开{c}", "{c}能换就换，不是绝对不接"]
    out: list[dict[str, Any]] = []
    for idx, c in enumerate(cats):
        for t in (templates[idx % len(templates)], templates[(idx + 1) % len(templates)]):
            out.append(_ex(t.format(c=c), avoid_categories=[c]))
    return out


def _gen_allowed_categories() -> list[dict[str, Any]]:
    groups = [["冷链", "冷藏"], ["建材"], ["水果"], ["钢材"], ["快递快运"], ["危化品"],
              ["生鲜"], ["家具"], ["农产品"], ["电子产品"]]
    templates = ["我只拉{c}，其它一概不接", "专跑{c}，别的不做", "只接{c}类的活", "除了{c}一律不接"]
    out: list[dict[str, Any]] = []
    for idx, g in enumerate(groups):
        label = "、".join(g)
        for t in (templates[idx % len(templates)], templates[(idx + 2) % len(templates)]):
            out.append(_ex(t.format(c=label), allowed_categories=list(g)))
    return out


def _gen_forbidden_regions() -> list[dict[str, Any]]:
    regs = ["深圳", "惠州", "东莞", "汕头", "武汉", "南昌", "杭州", "苏州", "上海",
            "宁波", "合肥", "郑州", "重庆", "佛山", "珠海"]
    templates = ["{r}的货不接", "不往{r}跑", "装货地或卸货地在{r}的一律不接", "{r}那边的活儿不做"]
    out: list[dict[str, Any]] = []
    for idx, r in enumerate(regs):
        for t in (templates[idx % len(templates)], templates[(idx + 1) % len(templates)]):
            out.append(_ex(t.format(r=r), forbidden_regions=[r]))
    return out


def _gen_allowed_regions() -> list[dict[str, Any]]:
    regs = ["长三角", "珠三角", "江浙沪", "广东", "江苏", "浙江", "湖北省内", "华东",
            "粤东", "成渝地区", "京津冀", "山东省内", "福建省内", "河南省内"]
    templates = ["我只跑{r}", "只在{r}范围内接活", "不出{r}", "限定在{r}跑，出了这范围不接", "净系喺{r}跑"]
    out: list[dict[str, Any]] = []
    for idx, r in enumerate(regs):
        for t in (templates[idx % len(templates)], templates[(idx + 3) % len(templates)]):
            out.append(_ex(t.format(r=r), allowed_regions=[r]))
    return out


def _gen_forbidden_zones() -> list[dict[str, Any]]:
    radii = [10, 15, 20, 25, 30, 40, 50]
    templates = ["以（{lat}，{lng}）为圆心半径{rk}公里以内禁止进入",
                 "（{lat}，{lng}）周边{rk}公里范围内不去",
                 "离（{lat}，{lng}）{rk}公里以内的活不接"]
    out: list[dict[str, Any]] = []
    for i in range(12):
        lat, lng = _COORDS[i % len(_COORDS)]
        rk = radii[i % len(radii)]
        t = templates[i % len(templates)]
        out.append(_ex(t.format(lat=lat, lng=lng, rk=rk),
                       forbidden_zones=[{"lat": lat, "lng": lng, "radius_km": rk}]))
    return out


def _gen_bounded_area() -> list[dict[str, Any]]:
    boxes = [(22, 24, 112, 114), (28, 31, 112, 116), (30, 32, 119, 122),
             (23, 25, 113, 116), (29, 31, 113, 115), (31, 33, 117, 120),
             (24, 27, 116, 119), (22, 25, 110, 113)]
    out: list[dict[str, Any]] = []
    for (la0, la1, ln0, ln1) in boxes:
        text = f"只在北纬{la0}到{la1}度、东经{ln0}到{ln1}度的范围内运营，出了这个区域不接"
        out.append(_ex(text, bounded_area={"lat_min": la0, "lat_max": la1, "lng_min": ln0, "lng_max": ln1}))
    return out


def _gen_required_region() -> list[dict[str, Any]]:
    regs = [("武汉", 3), ("苏州", 4), ("广州", 5), ("杭州", 3), ("南京", 4),
            ("长沙", 3), ("郑州", 5), ("合肥", 4), ("东莞", 6), ("宁波", 3)]
    templates = ["每月起码{d}天在{r}接货", "一个月至少有{d}天要在{r}揽货", "每月在{r}接单不少于{d}天"]
    out: list[dict[str, Any]] = []
    for i, (r, d) in enumerate(regs):
        t = templates[i % len(templates)]
        out.append(_ex(t.format(r=r, d=d), required_region={"region": r, "min_days": d}))
    return out


def _gen_must_visit() -> list[dict[str, Any]]:
    radii = [5, 10, 15, 20]
    days = [2, 3, 4, 5]
    templates = ["每月至少有{d}天要到（{lat}，{lng}）{rk}公里范围内",
                 "一个月起码{d}天得去（{lat}，{lng}）附近{rk}公里",
                 "每月必须有{d}天到（{lat}，{lng}）{rk}公里以内打卡"]
    out: list[dict[str, Any]] = []
    for i in range(10):
        lat, lng = _COORDS[(i + 3) % len(_COORDS)]
        rk = radii[i % len(radii)]
        d = days[i % len(days)]
        t = templates[i % len(templates)]
        out.append(_ex(t.format(lat=lat, lng=lng, rk=rk, d=d),
                       must_visit=[{"lat": lat, "lng": lng, "radius_km": rk, "required_days": d}]))
    return out


def _gen_pickup_max() -> list[dict[str, Any]]:
    kms = [20, 30, 40, 50, 55, 60, 80, 100]
    templates = ["赴装空驶超过{k}公里就不接", "去装货的空驶不超过{k}公里", "提货空跑别超过{k}公里"]
    out: list[dict[str, Any]] = []
    for i, k in enumerate(kms):
        t = templates[i % len(templates)]
        out.append(_ex(t.format(k=k), pickup_max_km=k))
    return out


def _gen_haul_max() -> list[dict[str, Any]]:
    kms = [300, 400, 500, 600, 800, 1000]
    templates = ["单趟运距不超过{k}公里", "干线超过{k}公里的长活不接", "单笔运输距离顶多{k}公里"]
    out: list[dict[str, Any]] = []
    for i, k in enumerate(kms):
        for t in (templates[i % len(templates)], templates[(i + 1) % len(templates)]):
            out.append(_ex(t.format(k=k), haul_max_km=k))
    return out


def _gen_haul_min() -> list[dict[str, Any]]:
    kms = [50, 80, 100, 150, 200]
    templates = ["干线低于{k}公里的短活不接", "运距不到{k}公里的不拉", "{k}公里以内的短途不做"]
    out: list[dict[str, Any]] = []
    for i, k in enumerate(kms):
        for t in (templates[i % len(templates)], templates[(i + 2) % len(templates)]):
            out.append(_ex(t.format(k=k), haul_min_km=k))
    return out


def _gen_deadhead() -> list[dict[str, Any]]:
    kms = [1000, 1500, 2000, 2500, 3000]
    templates = ["整月空驶里程不能超过{k}公里", "一个月累计空驶控制在{k}公里以内", "月度空跑总里程不超过{k}公里"]
    out: list[dict[str, Any]] = []
    for i, k in enumerate(kms):
        for t in (templates[i % len(templates)], templates[(i + 1) % len(templates)]):
            out.append(_ex(t.format(k=k), monthly_deadhead_max_km=k))
    return out


def _gen_daily_order_limit() -> list[dict[str, Any]]:
    ns = [1, 2, 3, 4, 5, 6]
    templates = ["同一天接单不超过{n}单", "每天最多接{n}单", "一天顶多拉{n}趟"]
    out: list[dict[str, Any]] = []
    for i, n in enumerate(ns):
        for t in (templates[i % len(templates)], templates[(i + 1) % len(templates)]):
            out.append(_ex(t.format(n=n), daily_order_limit=n))
    return out


def _gen_first_order() -> list[dict[str, Any]]:
    hs = [6, 7, 8, 9, 10, 11]
    templates = ["每天首单不能晚于早上{h}点", "第一单要在{h}点前接上", "每天{h}点前必须出第一趟"]
    out: list[dict[str, Any]] = []
    for i, h in enumerate(hs):
        t = templates[i % len(templates)]
        out.append(_ex(t.format(h=h), first_order_before_hour=h))
    return out


def _gen_off_days() -> list[dict[str, Any]]:
    ns = [2, 3, 4, 5, 6, 8]
    templates = ["一个月里至少歇{n}天不出车", "每月起码休{n}天", "整月要留{n}天不干活"]
    out: list[dict[str, Any]] = []
    for i, n in enumerate(ns):
        t = templates[i % len(templates)]
        out.append(_ex(t.format(n=n), off_days_min=n))
    return out


def _gen_category_targets() -> list[dict[str, Any]]:
    specs = [(4, "水果", 12, True), (5, "建材", 12, False), (3, "蔬菜", 10, True),
             (4, "冷链", 8, False), (5, "钢材", 15, True), (3, "家电", 6, False),
             (4, "农产品", 20, True), (5, "日用百货", 10, False)]
    months = {3: "三月", 4: "四月", 5: "五月"}
    out: list[dict[str, Any]] = []
    for (m, c, n, carry) in specs:
        suffix = "，欠的下个月补上" if carry else ""
        text = f"{months[m]}{c}必须接满{n}单{suffix}"
        out.append(_ex(text, monthly_category_targets=[
            {"month": m, "category": c, "min_orders": n, "carryover": carry}]))
    return out


def _gen_longhaul_cap() -> list[dict[str, Any]]:
    specs = [(2, 8), (3, 8), (3, 10), (4, 6), (5, 12), (2, 6)]
    templates = ["每月超过{h}小时的长途单最多接{n}单，多接一单罚一次",
                 "一个月里超过{h}小时的远活只能接{n}单"]
    out: list[dict[str, Any]] = []
    for i, (n, h) in enumerate(specs):
        t = templates[i % len(templates)]
        out.append(_ex(t.format(n=n, h=h), monthly_longhaul_cap={"max_orders": n, "min_hours": h}))
    return out


def _gen_home_rule() -> list[dict[str, Any]]:
    by = [21, 22, 23]
    until = [6, 7, 8]
    radii = [1, 2, 3]
    templates = ["每天{b}点前车要回到自家位置（{lat}，{lng}）{rk}公里内，到次日{u}点前不接单不空跑",
                 "每晚{b}点前得返到屋企（{lat}，{lng}）{rk}公里内，听日朝早{u}点先出车"]
    out: list[dict[str, Any]] = []
    for i in range(9):
        lat, lng = _COORDS[(i + 1) % len(_COORDS)]
        b = by[i % len(by)]
        u = until[i % len(until)]
        rk = radii[i % len(radii)]
        t = templates[i % len(templates)]
        # the home-by-hour also implies a no-drive window from b to next-day u
        out.append(_ex(t.format(b=b, u=u, rk=rk, lat=lat, lng=lng),
                       no_drive_windows=[{"start_hour": b, "end_hour": u}],
                       home_rule={"lat": lat, "lng": lng, "radius_km": rk,
                                  "home_by_hour": b, "no_drive_until_hour": u}))
    return out


def _gen_blackout() -> list[dict[str, Any]]:
    specs = [("深圳", [4, 5]), ("惠州", [10]), ("东莞", [15, 16]), ("武汉", [20]),
             ("苏州", [8, 9]), ("上海", [25]), ("广州", [1, 2, 3]), ("杭州", [18])]
    out: list[dict[str, Any]] = []
    for (r, dates) in specs:
        ds = "、".join(str(d) + "号" for d in dates)
        out.append(_ex(f"{ds}不往{r}跑", blackout=[{"region": r, "dates": dates}]))
    return out


def _gen_dated_single() -> list[dict[str, Any]]:
    # 办事类 → wait_minutes=120 ; 取/路过类 → wait_minutes=0
    errand_verbs = ["盘库", "对账", "验收", "盘点", "保养", "检修", "开会", "办手续", "交货"]
    pass_verbs = ["取东西", "取货", "拿货", "送东西"]
    out: list[dict[str, Any]] = []
    for i, v in enumerate(errand_verbs):
        lat, lng = _COORDS[i % len(_COORDS)]
        date = 5 + i * 2
        out.append(_ex(f"{date}号要去仓库（{lat}，{lng}）{v}，大概两小时",
                       dated_single=[{"date": date, "lat": lat, "lng": lng, "wait_minutes": 120, "before_hour": None}]))
    for j, v in enumerate(pass_verbs):
        lat, lng = _COORDS[(j + 5) % len(_COORDS)]
        date = 6 + j * 3
        out.append(_ex(f"{date}号顺路去（{lat}，{lng}）{v}，不耽搁",
                       dated_single=[{"date": date, "lat": lat, "lng": lng, "wait_minutes": 0, "before_hour": None}]))
    # a few with before_hour
    for k in range(4):
        lat, lng = _COORDS[(k + 2) % len(_COORDS)]
        date = 12 + k * 4
        h = 10 + k
        out.append(_ex(f"{date}号中午{h}点前要赶到（{lat}，{lng}）验收，待两小时",
                       dated_single=[{"date": date, "lat": lat, "lng": lng, "wait_minutes": 120, "before_hour": h}]))
    return out


def _gen_dated_route() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(10):
        a = _COORDS[i % len(_COORDS)]
        b = _COORDS[(i + 4) % len(_COORDS)]
        date = 7 + i * 2
        h = 11 + (i % 3)
        text = (f"{date}号先过（{a[0]}，{a[1]}）接上家人，再赶到（{b[0]}，{b[1]}）喝喜酒，"
                f"下午{h}点前要到，吃到散席")
        out.append(_ex(text, dated_route=[{"date": date, "stops": [
            {"lat": a[0], "lng": a[1], "wait_minutes": 0, "before_hour": h},
            {"lat": b[0], "lng": b[1], "wait_minutes": 120, "before_hour": h}]}]))
    return out


def _gen_penalty_attr() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # night window penalty
    out.append(_ex(["夜里十点到早六点不出车", "生鲜一律不接"], penalty_amounts=[2500, 1200],
                   no_drive_windows=[{"start_hour": 22, "end_hour": 6}],
                   forbidden_categories=["生鲜"],
                   rule_penalties={"night_window": 2500, "forbidden_categories": 1200}))
    out.append(_ex(["每违规接一单惠州的货扣800，整月最多扣6000"], penalty_amounts=[800], penalty_caps=[6000],
                   forbidden_regions=["惠州"],
                   rule_penalties={"forbidden_regions": 800},
                   rule_penalty_caps={"forbidden_regions": 6000}))
    out.append(_ex(["只跑珠三角，越界一次罚1000"], penalty_amounts=[1000],
                   allowed_regions=["珠三角"],
                   rule_penalties={"allowed_regions": 1000}))
    out.append(_ex(["四月水果接不满十二单，每差一单罚500"], penalty_amounts=[500],
                   monthly_category_targets=[{"month": 4, "category": "水果", "min_orders": 12, "carryover": False}],
                   rule_penalties={"category_targets": 500}))
    out.append(_ex(["每月超八小时长途最多三单，超一单罚1500"], penalty_amounts=[1500],
                   monthly_longhaul_cap={"max_orders": 3, "min_hours": 8},
                   rule_penalties={"longhaul_cap": 1500}))
    out.append(_ex(["二十号没去仓库（23.15，113.67）盘库罚2000"], penalty_amounts=[2000],
                   dated_single=[{"date": 20, "lat": 23.15, "lng": 113.67, "wait_minutes": 120, "before_hour": None}],
                   rule_penalties={"dated_events": 2000}))
    out.append(_ex(["危化品坚决不拉，违规一次扣3000，月封顶9000"], penalty_amounts=[3000], penalty_caps=[9000],
                   forbidden_categories=["危化品"],
                   rule_penalties={"forbidden_categories": 3000},
                   rule_penalty_caps={"forbidden_categories": 9000}))
    out.append(_ex(["夜里十一点到早五点不接单，违一次罚1800"], penalty_amounts=[1800],
                   no_drive_windows=[{"start_hour": 23, "end_hour": 5}],
                   rule_penalties={"night_window": 1800}))
    return out


def _gen_compound() -> list[dict[str, Any]]:
    """Multi-constraint examples — teach the model to extract ALL constraints
    present, and (via the negatives elsewhere) ONLY those present."""
    out: list[dict[str, Any]] = []
    out.append(_ex(["晚上10点到早6点不出车睡觉", "深圳的货不接", "同一天最多接3单"],
                   rest_window={"start_hour": 22, "end_hour": 6},
                   no_drive_windows=[{"start_hour": 22, "end_hour": 6}],
                   forbidden_regions=["深圳"], daily_order_limit=3))
    out.append(_ex(["只跑长三角", "危化品不拉", "单趟不超过500公里"],
                   allowed_regions=["长三角"], forbidden_categories=["危化品"], haul_max_km=500))
    out.append(_ex(["每月歇4天", "首单不晚于早上8点", "赴装空驶不超过50公里"],
                   off_days_min=4, first_order_before_hour=8, pickup_max_km=50))
    out.append(_ex(["四月水果接满12单欠单结转", "五月建材至少12单", "夜里11点到5点不开车"],
                   no_drive_windows=[{"start_hour": 23, "end_hour": 5}],
                   monthly_category_targets=[
                       {"month": 4, "category": "水果", "min_orders": 12, "carryover": True},
                       {"month": 5, "category": "建材", "min_orders": 12, "carryover": False}]))
    out.append(_ex(["每天23点前回到家（23.10，113.50）1公里内，次日7点前不接单", "生鲜不拉"],
                   no_drive_windows=[{"start_hour": 23, "end_hour": 7}],
                   forbidden_categories=["生鲜"],
                   home_rule={"lat": 23.10, "lng": 113.50, "radius_km": 1, "home_by_hour": 23, "no_drive_until_hour": 7}))
    out.append(_ex(["只在广东跑", "干线低于100公里不接", "每天最多4单"],
                   allowed_regions=["广东"], haul_min_km=100, daily_order_limit=4))
    out.append(_ex(["12号去仓库（23.15，113.67）盘库两小时", "整月空驶不超过2000公里"],
                   monthly_deadhead_max_km=2000,
                   dated_single=[{"date": 12, "lat": 23.15, "lng": 113.67, "wait_minutes": 120, "before_hour": None}]))
    out.append(_ex(["机械设备干不了", "惠州的货不接", "20号不往深圳跑"],
                   forbidden_categories=["机械设备"], forbidden_regions=["惠州"],
                   blackout=[{"region": "深圳", "dates": [20]}]))
    out.append(_ex(["夜晚黑廿二点到朝早六点唔开车瞓觉", "净系喺珠三角跑", "冷链货专拉"],
                   rest_window={"start_hour": 22, "end_hour": 6},
                   no_drive_windows=[{"start_hour": 22, "end_hour": 6}],
                   allowed_regions=["珠三角"], allowed_categories=["冷链"]))
    out.append(_ex(["每月超8小时长途最多2单", "危化品尽量少接", "首单不晚于9点"],
                   first_order_before_hour=9, avoid_categories=["危化品"],
                   monthly_longhaul_cap={"max_orders": 2, "min_hours": 8}))
    out.append(_ex(["以市中心（31.23，121.47）半径30公里禁入", "只拉建材"],
                   forbidden_zones=[{"lat": 31.23, "lng": 121.47, "radius_km": 30}],
                   allowed_categories=["建材"]))
    out.append(_ex(["每月至少3天在武汉接货", "每月5天到配送中心（30.59，114.30）10公里内"],
                   required_region={"region": "武汉", "min_days": 3},
                   must_visit=[{"lat": 30.59, "lng": 114.30, "radius_km": 10, "required_days": 5}]))
    return out


def _gen_negatives() -> list[dict[str, Any]]:
    """Zero-constraint / 钓鱼 examples → ALL defaults. The user's root cause is
    over-extraction (hallucinating constraints → skipping orders → lowest gross),
    so this is the largest single category and several entries are anchors."""
    chit_chat = [
        "我开大车十几年咯，啥货都拉，哪儿都跑，没啥特别忌讳，给钱就走",
        "随便派单，啥都行，我不挑",
        "有啥拉啥，哪儿有钱往哪儿跑",
        "我这人好说话，活儿来者不拒",
        "看运气吧，能多拉就多拉",
        "车况挺好的，跑长途短途都没问题",
        "干了二十年了，路熟得很，啥地方都去过",
        "怎么安排都行，听调度的",
        "我啥都不忌讳，给钱就干",
        "无所谓拉啥，挣钱就行",
        "这个月想多挣点钱",                      # wish, not a constraint
        "希望能多接几个大单",                    # wish
        "最好能天天有活干",                      # wish
        "想把这个月跑满，多劳多得",              # wish
        "盼着行情好点，多挣是多挣",              # wish
        "上次拉生鲜亏了点，唉，不提了",          # mention w/o forbiddance
        "前阵子跑了趟深圳，路上挺堵",            # mention w/o constraint
        "听说最近油价又涨了",                    # irrelevant
        "我老家是江西的，出来跑车好些年了",      # background
        "媳妇老说我跑车太累",                    # background
        "新换的车，油耗低了不少",                # background
        "我开车稳，从不超速",                    # background
        "认识不少老板，回头单多",                # background
        "这行不容易，起早贪黑",                  # background
        "我喜欢跑高速，不爱走国道",              # preference, not a rule
        "能拉满载就拉满载，空着回去亏",          # general wish
        "客户口碑好，复购率高",                  # background
        "平时不怎么挑货，能装下就行",            # explicitly no constraint
        "天南海北都跑过，没啥到不了的地方",      # no constraint
        "给的价合适就接，不合适就再等等",        # generic
        "我对货没要求，安全送到就行",            # no constraint
        "干这行图个自由，时间自己安排",          # no rule
        "啥时候有活啥时候出车，不固定",          # explicitly no schedule rule
        "拉过各种货，经验足",                    # background
        "只要不耽误事，咋拉都行",                # no constraint
        "我这车厢大，啥都装得下",                # background
        "跑车嘛，多跑多得，少跑少得",            # platitude
        "没啥讲究，调度发啥我拉啥",              # no constraint
        "我身体硬朗，连着跑几天没问题",          # background
        "车上GPS、行车记录仪都齐全",            # background
        # dialect chit-chat
        "我系老司机嚟㗎，咩货都拉得，边度都去得",     # 粤: no constraint
        "冇咩特别讲究，畀钱就走",                     # 粤: no constraint
        "啥货阿拉侪拉得，勿挑个",                     # 吴: no constraint
        "随便侬安排好哉，阿拉听招呼",                 # 吴: no constraint
        "厓样样货都做得，唔挑嘅",                     # 客家: no constraint
        "这个月争取多跑几趟，攒点钱",                 # wish
        "盼着接到长途大单，划算",                     # wish
        "想趁年轻多挣点，不怕辛苦",                   # wish
        "最近想换个大点的车头",                       # irrelevant
        "孩子要上学了，得多挣点",                     # background/wish
        "我对路线没要求，导航说哪走哪走",             # no constraint
        "啥品类都接触过，上手快",                     # background
        "只要结款及时，啥活我都干",                   # generic, not a rule
        "天气好就多跑，下雨就少跑",                   # generic, not a hard rule
        "我不抽烟不喝酒，开车精神好",                 # background
        "跑了这么多年，没出过事故",                   # background
        "认路、能吃苦，这是我的优势",                 # background
        "活不活的看缘分，强求不来",                   # generic
        "我这人实在，不玩虚的",                       # background
    ]
    out: list[dict[str, Any]] = []
    for i, t in enumerate(chit_chat):
        # mark the first few canonical chit-chat lines as anchors so the
        # anti-over-extraction signal is ALWAYS injected regardless of query.
        out.append(_ex(t, anchor=(i < 3)))
    return out


def _build_bank() -> list[dict[str, Any]]:
    parts: list[list[dict[str, Any]]] = [
        _CURATED,
        _gen_negatives(),
        _gen_night(),
        _gen_forbidden_categories(),
        _gen_avoid_categories(),
        _gen_allowed_categories(),
        _gen_forbidden_regions(),
        _gen_allowed_regions(),
        _gen_forbidden_zones(),
        _gen_bounded_area(),
        _gen_required_region(),
        _gen_must_visit(),
        _gen_pickup_max(),
        _gen_haul_max(),
        _gen_haul_min(),
        _gen_deadhead(),
        _gen_daily_order_limit(),
        _gen_first_order(),
        _gen_off_days(),
        _gen_category_targets(),
        _gen_longhaul_cap(),
        _gen_home_rule(),
        _gen_blackout(),
        _gen_dated_single(),
        _gen_dated_route(),
        _gen_penalty_attr(),
        _gen_compound(),
    ]
    seen: set[str] = set()
    bank: list[dict[str, Any]] = []
    for group in parts:
        for ex in group:
            key = "\u0001".join(ex["in"]["preferences"])
            if key in seen:
                continue
            seen.add(key)
            bank.append(ex)
    return bank


FEWSHOT_BANK: list[dict[str, Any]] = _build_bank()


# ----------------------------------------------------------- retrieval -------

# Domain trigger substrings. Examples and queries that share these score higher,
# biasing retrieval toward field-relevant examples even when surface wording
# differs (dialect / colloquial).
_SIGNAL_TOKENS: tuple[str, ...] = (
    "夜", "晚上", "睡", "休息", "收车", "收工", "熄火", "不出车", "不开车", "不接单",
    "空驶", "空跑", "瞓", "困觉",
    "只跑", "只接", "只在", "只拉", "只运", "专跑", "专拉", "不出", "范围", "区域",
    "纬度", "经度", "净系",
    "禁", "不接", "不去", "不进", "不往", "远离", "半径", "圆心",
    "长途", "短活", "干线", "运距", "公里", "里程", "空驶",
    "回家", "屋企", "到家", "住所", "返",
    "每月", "本月", "三月", "四月", "五月", "品类", "指标", "接满", "至少", "结转",
    "每天", "每日", "当天", "上限", "最多", "首单", "第一单", "歇",
    "盘库", "对账", "验收", "提货", "签收", "赴宴", "喜酒", "接", "送",
    "水果", "建材", "生鲜", "冷链", "危化品", "蔬菜", "钢材", "机械", "活禽", "煤炭",
    "扣", "罚", "封顶",
    "广东", "深圳", "惠州", "长三角", "珠三角", "江浙沪", "武汉", "苏州", "上海", "杭州",
    "廿", "唔", "弗", "覅", "听日", "朝早",
)


def _norm(s: str) -> str:
    """Keep CJK + alphanumerics, drop spaces/punctuation/fullwidth marks."""
    out_chars: list[str] = []
    for ch in s:
        if ch.isalnum():
            out_chars.append(ch)
    return "".join(out_chars)


def _bigrams(s: str) -> frozenset[str]:
    if len(s) < 2:
        return frozenset({s}) if s else frozenset()
    return frozenset(s[i:i + 2] for i in range(len(s) - 1))


def _example_text(ex: dict[str, Any]) -> str:
    return " ".join(ex["in"]["preferences"])


# Precompute per-example features once at import for fast, deterministic scoring.
_INDEX: list[tuple[frozenset[str], frozenset[str]]] = []
for _ex_entry in FEWSHOT_BANK:
    _t = _norm(_example_text(_ex_entry))
    _sig = frozenset(tok for tok in _SIGNAL_TOKENS if tok in _example_text(_ex_entry))
    _INDEX.append((_bigrams(_t), _sig))


def _score(q_bigrams: frozenset[str], q_text: str,
           ex_bigrams: frozenset[str], ex_signals: frozenset[str]) -> float:
    if q_bigrams and ex_bigrams:
        inter = len(q_bigrams & ex_bigrams)
        sim = inter / ((len(q_bigrams) ** 0.5) * (len(ex_bigrams) ** 0.5))
    else:
        sim = 0.0
    bonus = 0.0
    for tok in ex_signals:
        if tok in q_text:
            bonus += 0.12
    return sim + bonus


def select_examples(query_text: str, k: int = 40, n_anchor: int = 4) -> list[dict[str, Any]]:
    """Return up to ``k`` bank examples most relevant to ``query_text``.

    Always includes up to ``n_anchor`` fixed anchor negatives (zero-constraint /
    trap examples) so the anti-over-extraction signal is present on every call.
    Deterministic: same query -> same ordered list.
    """
    if k <= 0 or not FEWSHOT_BANK:
        return []
    q_norm = _norm(query_text or "")
    q_text = query_text or ""
    q_bi = _bigrams(q_norm)
    scored: list[tuple[float, int]] = []
    for i, (ex_bi, ex_sig) in enumerate(_INDEX):
        scored.append((_score(q_bi, q_text, ex_bi, ex_sig), i))
    # stable: higher score first, then lower index (deterministic tie-break)
    scored.sort(key=lambda t: (-t[0], t[1]))

    chosen: list[int] = []
    chosen_set: set[int] = set()
    # reserve anchor slots first
    for i, ex in enumerate(FEWSHOT_BANK):
        if len(chosen) >= min(n_anchor, k):
            break
        if ex.get("anchor"):
            chosen.append(i)
            chosen_set.add(i)
    # fill remaining slots with top-scored, non-anchor-duplicate examples
    for _sc, i in scored:
        if len(chosen) >= k:
            break
        if i in chosen_set:
            continue
        chosen.append(i)
        chosen_set.add(i)
    return [FEWSHOT_BANK[i] for i in chosen]


__all__ = ["FEWSHOT_BANK", "SCHEMA_FIELDS", "select_examples"]
