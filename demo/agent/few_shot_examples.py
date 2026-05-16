"""Few-shot 示例学习模块：从 Run3/Run4/Run5/Run6 仿真日志中提取的真实决策案例。

通过在 LLM system prompt 中注入场景化的正面/负面决策示例，
让 LLM 在关键时刻做出更优选择，无需训练模型。

示例来源：
- Run3 (LLM-advisory): 净收入 242,476 (-0.1%)，罚分 22,150
- Run4 (激进评分): 净收入 226,141 (-6.8%)，罚分 25,360
- Run5 (保守评分): 净收入 214,868 (-11.5%)，罚分 45,900
- Run6 (Few-shot v1): 净收入 238,227 (-1.87%)，罚分 22,450
- Master 基线: 净收入 242,776，罚分 21,850

核心发现：
1. D009 "23点前到家"规则是最大单点风险（Run5: 28次→Run6: 1次，大幅改善）
2. D002/D006/D008 月度休息日管理不善导致高额罚分
3. D003 空驶控制不当导致超限（Run6仍有2000元罚分）
4. D004 午休时段(12-13点)规则30天全部违规，罚3000元（Run6新发现）
5. D010 家事约定不在家时间过长，罚5150元（Run6最大单项罚分）
6. D008 多重偏好同时违规累积4000元（停车休息+品类+空驶+月度休息）
7. 评分框架修改（Run4/Run5）全部导致回归，证明不应覆盖评分系统
"""

from __future__ import annotations

from . import config, geo_utils
from .driver_memory import DriverMemory
from .preference_parser import ParsedRules


# ---- 正面示例（应该做的） ----

_POSITIVE_EXAMPLES = (
    "【正面案例 - 来自仿真验证】\n"
    "案例P1(回家规则): 司机有'每日23点前到家'约束。18:01完成订单后距家136km，"
    "立即选择[移]空驶回家方向，22:45到家(0km)。→正确：傍晚优先回家避免900元/次罚分\n"
    "案例P2(回家规则): 司机20:20接单结束距家53km，选择[移]回家，21:24到达。"
    "→正确：距家<100km且时间充裕时仍应尽早回家\n"
    "案例P3(月度休息): 月底第26天，休息日缺口=1天仅剩5天，选择[等]整天休息。"
    "→正确：月末休息紧急时必须优先安排休息，否则缺1天罚2000-3000元\n"
    "案例P4(禁行时段): 22:45到家，距禁行时段(23:00-8:00)仅15分钟，选择[等]545分钟到次日8:00。"
    "→正确：禁行前在安全位置等待，避免200-500元/天罚分\n"
    "案例P5(家事约定): 司机有3/10-3/13家事约定，提前在3/9空驶到家附近等待。"
    "→正确：家事/定时事件需提前≥1天到位，迟到每分钟罚5元\n"
    "案例P6(停车休息): 每天固定安排240分钟连续停车休息，满足'每日≥4h停车'要求。"
    "→正确：每日休息需一次性连续满足，碎片化休息不算\n"
    "案例P7(午休规则-Run6验证): 有'12-13点不接单不空驶'约束时，12:00一到立即选[等]60分钟。"
    "→正确：午休时段必须严格等待，30天违规=3000元罚分\n"
    "案例P8(品类回避-Run6验证): 有'不拉食品饮料'偏好时，看到食品饮料品类的候选直接跳过选下一个。"
    "→正确：品类违规每次罚200元，3次=600元\n"
)

# ---- 负面示例（不应该做的） ----

_NEGATIVE_EXAMPLES = (
    "【负面案例 - 来自仿真验证的失败教训】\n"
    "案例N1(回家规则-严重): 下午14:00距家0km时接远途单，18:47结束距家218km，"
    "无法23点前到家。→错误：下午14点后距家远的长途单不应接，罚900元/次\n"
    "案例N2(回家规则-严重): 20:52结束订单距家126km，又接新单到20:52距家更远。"
    "→错误：晚上20点后距家>50km不应再接单，应立即回家\n"
    "案例N3(空驶超限): 月空驶限额100km，实际空驶1009km(超额909km)，罚分2000元。"
    "→错误：有空驶限制时应避免频繁长距离空驶，每次空驶>20km都要谨慎\n"
    "案例N4(月度休息不足): 全月不安排休息日，月底才发现缺口，罚3000-6000元。"
    "→错误：应从月中开始每周安排1天休息日，避免月底集中补休\n"
    "案例N5(首单超时): 有'首单不晚于12点'约束却每天13点后才接首单(30天违规)，罚2200元。"
    "→错误：有首单时间约束时，上午应优先接单而非等待\n"
    "案例N6(评分覆盖-严重): LLM策略直接修改评分权重导致净收入下降11.5%。"
    "→错误：评分系统经过10轮迭代校准，LLM不应覆盖评分权重，仅在候选间选择\n"
    "案例N7(装卸距离): 有'装卸距离≤100km'约束却接37次超距订单，罚3700元。"
    "→错误：评分中已有距离罚分时，LLM不应选择明显超距的候选\n"
    "案例N8(禁行违规): 23-4点禁行时段仍选择接单/空驶(30天违规)，罚15000元。"
    "→错误：禁行时段内只能选[等]，任何接单/空驶都会产生罚分\n"
    "案例N9(午休违规-Run6实测): D004有'12-13点不接单不空驶'约束，30天全部违规罚3000元。"
    "→错误：午休/固定时段规则必须严格遵守，每天违规=每天罚100元，不可忽视\n"
    "案例N10(多重违规-Run6实测): D008同时违反月度休息(1500)+停车休息(1600)+品类(600)+空驶(300)=4000元。"
    "→错误：多项偏好叠加罚分极高，宁可少接几单也要保证所有偏好都满足\n"
    "案例N11(家事迟到-Run6实测): D010家事约定期间不在家时间1030分钟，每分钟罚5元=5150元。"
    "→错误：家事约定期间必须待在家中不出门，提前到家后不要再接单出发\n"
    "案例N12(停车休息不足-Run6实测): D002每日需连续停车≥4小时但3天不满足，罚600元。"
    "→错误：每日停车休息必须一次性连续满足，分段休息不算，每天至少安排一个≥4h的[等]\n"
)


def build_few_shot_system_prompt(
    rules: ParsedRules,
    memory: DriverMemory,
    ctx_minutes: int,
    ctx_lat: float,
    ctx_lng: float,
) -> str:
    """根据当前司机的偏好规则和状态，选择最相关的 few-shot 示例注入 system prompt。

    设计原则：
    1. 只注入与当前司机规则相关的示例（节省 token）
    2. 在关键时刻（月末、禁行临近、回家临近）注入更多示例
    3. 负面案例优先（人类学习中 negative example 更有效）
    """
    base_prompt = (
        "你是货运调度优化AI。目标：最大化司机31天月度净收入（毛收入-偏好违规罚金）。\n"
        "评分系统已综合计算收入、成本、偏好罚分等因素，分数越高越优。\n"
        "决策原则：\n"
        "1. 通常选择评分最高的候选（编号1），除非有明确理由偏离\n"
        "2. 当多个候选分数接近时（差距<20%），优先选偏好违规风险更低的\n"
        "3. 月底休息日缺口>0时，优先安排休息（选[等]）\n"
        "4. 禁行时段内优先等待，避免高额罚金\n"
        "5. 不要选分数远低于最高分的候选\n"
    )

    examples = _select_relevant_examples(rules, memory, ctx_minutes, ctx_lat, ctx_lng)

    if examples:
        base_prompt += "\n" + examples + "\n"

    base_prompt += "仅回复最优候选编号（如：2），不要解释。"
    return base_prompt


def _select_relevant_examples(
    rules: ParsedRules,
    memory: DriverMemory,
    ctx_minutes: int,
    ctx_lat: float,
    ctx_lng: float,
) -> str:
    """选择与当前情境最相关的示例子集。"""
    parts: list[str] = []
    hour = geo_utils.hour_of_day(ctx_minutes)
    sim_day = ctx_minutes // 1440
    days_remaining = max(0, config.AGENT_HORIZON_DAYS - 1 - sim_day)

    has_home_rule = rules.home_rule is not None
    has_no_drive = len(rules.no_drive_windows) > 0
    has_monthly_off = rules.monthly_day_off is not None
    has_timed_events = len(rules.timed_stay_events) > 0
    has_distance_limit = len(rules.distance_limits) > 0

    # ---- 场景化正面示例 ----
    positive_parts: list[str] = []

    if has_home_rule and hour >= 14:
        positive_parts.append(
            "案例P1: 有'每日到家'约束。18:01完成订单距家136km，"
            "立即[移]回家，22:45到达→避免900元/次罚分"
        )
        if hour >= 18:
            positive_parts.append(
                "案例P2: 20:20结束距家53km，选[移]回家21:24到达"
                "→距家<100km时间充裕仍应尽早回"
            )

    if has_no_drive:
        for w in rules.no_drive_windows:
            minutes_to_ban = _minutes_until_window(ctx_minutes, w.start_minute, w.end_minute)
            if 0 < minutes_to_ban <= 180:
                positive_parts.append(
                    "案例P4: 距禁行仅15分钟，选[等]到禁行结束"
                    "→禁行前在安全位置等待，避免200-500元/天罚分"
                )
                break

    if has_monthly_off:
        req = rules.monthly_day_off.required_days or 0
        done = max(0, sim_day + 1 - memory.days_active_count())
        deficit = max(0, req - done)
        if deficit > 0 and days_remaining <= 7:
            positive_parts.append(
                "案例P3: 月底休息缺口=1仅剩5天，选[等]整天休息"
                "→月末必须优先安排休息，缺1天罚2000-3000元"
            )

    if has_timed_events:
        for event in rules.timed_stay_events:
            if event.start_minutes - ctx_minutes <= 48 * 60 and ctx_minutes < event.start_minutes:
                positive_parts.append(
                    "案例P5: 家事约定前1天空驶到家附近等待"
                    "→定时事件需提前≥1天到位，迟到每分钟罚5元"
                )
                break

    # 午休/固定时段规则（Run6新增）
    if has_no_drive:
        for w in rules.no_drive_windows:
            win_duration = (w.end_minute - w.start_minute) % 1440
            if 30 <= win_duration <= 120:
                minutes_to_ban = _minutes_until_window(ctx_minutes, w.start_minute, w.end_minute)
                if 0 < minutes_to_ban <= 30:
                    positive_parts.append(
                        "案例P7(Run6): 午休/固定时段即将开始→立即选[等]"
                        "，30天违规=3000元罚分"
                    )
                    break

    # ---- 场景化负面示例 ----
    negative_parts: list[str] = []

    if has_home_rule:
        home_lat = rules.home_rule.lat
        home_lng = rules.home_rule.lng
        dist_home = geo_utils.haversine_km(ctx_lat, ctx_lng, home_lat, home_lng)
        if hour >= 14:
            negative_parts.append(
                "案例N1(严重): 14点后接远途单→18:47距家218km无法到家"
                "→下午14点后不接让你远离家的长途单！罚900元/次"
            )
        if hour >= 18 and dist_home > 50:
            negative_parts.append(
                f"案例N2(严重): 当前{hour}点距家{dist_home:.0f}km，"
                "不应再接单应立即回家→20点后距家>50km立即回家"
            )

    if has_no_drive:
        for w in rules.no_drive_windows:
            minutes_to_ban = _minutes_until_window(ctx_minutes, w.start_minute, w.end_minute)
            if minutes_to_ban <= 0:
                negative_parts.append(
                    "案例N8(严重): 禁行时段内仍接单/空驶(30天违规)罚15000元"
                    "→禁行时段只能选[等]！"
                )
                break

    if has_monthly_off and days_remaining <= 10:
        req = rules.monthly_day_off.required_days or 0
        done = max(0, sim_day + 1 - memory.days_active_count())
        deficit = max(0, req - done)
        if deficit > 0:
            negative_parts.append(
                f"案例N4: 月度休息缺{deficit}天仅剩{days_remaining}天"
                "→不及时安排休息将罚3000-6000元"
            )

    if has_distance_limit:
        negative_parts.append(
            "案例N7: 有装卸距离约束却接超距订单→不应选择明显超距的候选"
        )

    # 午休/固定时段违规警告（Run6新增）
    if has_no_drive:
        for w in rules.no_drive_windows:
            minutes_to_ban = _minutes_until_window(ctx_minutes, w.start_minute, w.end_minute)
            if minutes_to_ban <= 0:
                win_duration = (w.end_minute - w.start_minute) % 1440
                if 30 <= win_duration <= 120:
                    negative_parts.append(
                        "案例N9(Run6): 当前在午休/固定禁行时段→必须选[等]"
                        "，D004因30天全部违规罚3000元"
                    )
                    break

    # 家事约定期间不应接单（Run6新增）
    if has_timed_events:
        for event in rules.timed_stay_events:
            if event.start_minutes <= ctx_minutes <= event.end_minutes:
                negative_parts.append(
                    "案例N11(Run6): 家事约定期间不在家1030分钟罚5150元"
                    "→当前在家事期间，必须待在家中选[等]！"
                )
                break

    # 通用负面：评分覆盖警告
    negative_parts.append(
        "案例N6: 评分系统经10轮校准，LLM仅在候选间选择，不覆盖评分权重"
    )

    # 多重违规累积警告（Run6新增）
    negative_parts.append(
        "案例N10(Run6): 多项偏好同时违规罚分累加可达4000+元"
        "→宁可少接单也要保证所有偏好都满足"
    )

    # 组合
    if positive_parts:
        parts.append("【正面参考】" + "；".join(positive_parts))
    if negative_parts:
        parts.append("【负面警告】" + "；".join(negative_parts))

    return "\n".join(parts)


def _minutes_until_window(current_minutes: int, win_start: int, win_end: int) -> int:
    """计算距下一次禁行窗口开始的分钟数。负值表示当前在窗口内。"""
    minute_of_day = geo_utils.minute_of_day(current_minutes)

    if win_start < win_end:
        if minute_of_day < win_start:
            return win_start - minute_of_day
        if minute_of_day < win_end:
            return -(minute_of_day - win_start)
        return win_start + 1440 - minute_of_day
    else:
        if minute_of_day >= win_start:
            return -(minute_of_day - win_start)
        if minute_of_day < win_end:
            return -(minute_of_day + 1440 - win_start)
        return win_start - minute_of_day


def build_situation_hint(
    rules: ParsedRules,
    memory: DriverMemory,
    ctx_minutes: int,
    ctx_lat: float,
    ctx_lng: float,
) -> str | None:
    """生成当前情境的简短决策提示，附加到 user prompt 中。

    只在检测到关键情境时返回提示，否则返回 None 以节省 token。
    """
    hints: list[str] = []
    hour = geo_utils.hour_of_day(ctx_minutes)
    sim_day = ctx_minutes // 1440
    days_remaining = max(0, config.AGENT_HORIZON_DAYS - 1 - sim_day)

    # 回家紧急提示
    if rules.home_rule is not None:
        home_lat = rules.home_rule.lat
        home_lng = rules.home_rule.lng
        dist_home = geo_utils.haversine_km(ctx_lat, ctx_lng, home_lat, home_lng)
        home_deadline_hour = _get_home_deadline_hour(rules)

        if home_deadline_hour is not None:
            hours_left = home_deadline_hour - hour
            if hours_left < 0:
                hours_left += 24
            travel_hours_needed = dist_home / 60.0

            if hour >= 14 and dist_home > 100:
                hints.append(
                    f"⚠回家紧急:距家{dist_home:.0f}km需{travel_hours_needed:.1f}h"
                    f"距{home_deadline_hour}点剩{hours_left}h→优先选[移]回家或[等]"
                )
            elif hour >= 18 and dist_home > 30:
                hints.append(
                    f"⚠回家:距家{dist_home:.0f}km→应选[移]回家不要接远途单"
                )

    # 禁行临近提示
    for w in rules.no_drive_windows:
        minutes_to_ban = _minutes_until_window(ctx_minutes, w.start_minute, w.end_minute)
        if minutes_to_ban <= 0:
            hints.append("⚠当前在禁行时段→只能选[等]")
            break
        if 0 < minutes_to_ban <= 120:
            hints.append(
                f"⚠距禁行{minutes_to_ban}分钟→不要接耗时超{minutes_to_ban}分的单"
            )
            break

    # 月度休息紧急
    if rules.monthly_day_off is not None:
        req = rules.monthly_day_off.required_days or 0
        done = max(0, sim_day + 1 - memory.days_active_count())
        deficit = max(0, req - done)
        if deficit > 0 and days_remaining <= deficit + 2:
            hints.append(
                f"⚠月度休息紧急:缺{deficit}天仅剩{days_remaining}天→必须选[等]安排休息"
            )

    # 定时事件临近
    for event in rules.timed_stay_events:
        if ctx_minutes < event.start_minutes:
            hours_to_event = (event.start_minutes - ctx_minutes) / 60.0
            if hours_to_event <= 24:
                event_lat = event.pickup_lat
                event_lng = event.pickup_lng
                dist_to_event = geo_utils.haversine_km(ctx_lat, ctx_lng, event_lat, event_lng)
                if dist_to_event > 20:
                    hints.append(
                        f"⚠定时事件{hours_to_event:.0f}h后开始,距接人点{dist_to_event:.0f}km"
                        "→应选[移]前往接人点"
                    )

    if not hints:
        return None
    return " | ".join(hints)


def _get_home_deadline_hour(rules: ParsedRules) -> int | None:
    """从 home_rule 获取回家截止时间。"""
    if rules.home_rule is not None:
        return rules.home_rule.home_by_hour
    return None
