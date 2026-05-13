"""诊断 D009/D010 31 天结果未完成高额偏好的根因。"""
from __future__ import annotations
import glob, json, math
from pathlib import Path

HAVERSINE_R = 6371.0
def km(a_lat, a_lng, b_lat, b_lng):
    la1, la2 = math.radians(a_lat), math.radians(b_lat)
    dlat = la2 - la1
    dlng = math.radians(b_lng - a_lng)
    a = math.sin(dlat/2)**2 + math.cos(la1) * math.cos(la2) * math.sin(dlng/2)**2
    return 2 * HAVERSINE_R * math.asin(math.sqrt(a))

def load(did: str):
    path = sorted(glob.glob(f'demo/results/actions_202603_{did}_*.jsonl'))[-1]
    rows = [json.loads(line) for line in open(path, encoding='utf-8')]
    return path, rows

# --- D009 ---
path, rows = load('D009')
print('D009 file', path, 'rows=', len(rows))
HOME = (23.12, 113.28)
night_acts = 0  # 23:00-07:59 within day
take_specific = []
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    minute_of_day = mins % (24*60)
    hour = minute_of_day // 60
    action = r['action']['action']
    if action != 'wait' and (hour >= 23 or hour < 8):
        night_acts += 1
    if action == 'take_order':
        cid = (r['action'].get('params') or {}).get('cargo_id')
        if str(cid) == '240646':
            take_specific.append((r['step'], r['result'].get('accepted'), r['result'].get('detail')))
print('D009 night non-wait actions (23-08)=', night_acts)
print('D009 specific cargo 240646 take attempts:', take_specific)
# any visits to home within 1km
home_visits = sum(1 for r in rows if km(r['position_after']['lat'], r['position_after']['lng'], *HOME) <= 1.0)
print('D009 home (within 1km) visits=', home_visits)

# --- D010 ---
path, rows = load('D010')
print('\nD010 file', path, 'rows=', len(rows))
PICK = (23.21, 113.37)
HOME = (23.19, 113.36)
WIN0, WIN1 = 13560, 17880  # 3/10 10:00 .. 3/13 22:00 in minutes
near_pick = 0
near_home = 0
in_window_actions = []
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    if WIN0 <= mins <= WIN1 + 60:
        action = r['action']['action']
        p = r['position_after']
        dp = km(p['lat'], p['lng'], *PICK)
        dh = km(p['lat'], p['lng'], *HOME)
        in_window_actions.append((mins, action, round(dp,2), round(dh,2)))
        if dp <= 1.0: near_pick += 1
        if dh <= 1.0: near_home += 1
print('D010 actions in window count=', len(in_window_actions), 'near_pick steps=', near_pick, 'near_home steps=', near_home)
print('first 20 in-window actions (sim_min, act, km_to_pick, km_to_home):')
for row in in_window_actions[:20]:
    print(' ', row)
print('around 13500-13600 (event start area):')
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    if 13400 <= mins <= 13700:
        p = r['position_after']
        print(' ', mins, r['action']['action'], r['action'].get('params'), 'pos=', (p['lat'], p['lng']))
