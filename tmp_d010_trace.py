import json, glob, math
path = sorted(glob.glob('demo/results/actions_202603_D010_*.jsonl'))[-1]
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
def km(a, b): return 2*6371*math.asin(math.sqrt(math.sin(math.radians(b[0]-a[0])/2)**2 + math.cos(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.sin(math.radians(b[1]-a[1])/2)**2))
PICK = (23.21, 113.37)
HOME = (23.19, 113.36)
# event start 13560 (3/10 10:00), pre_lock window starts 12120 (3/9 10:00)
print('file:', path)
print('step  sim_min          action       cargo  ->lat,lng     km_to_pick  km_to_home')
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    if 11500 <= mins <= 14500:
        action = r['action']['action']
        params = r['action'].get('params') or {}
        cid = params.get('cargo_id', '')
        pa = r['position_after']
        d_pick = round(km((pa['lat'], pa['lng']), PICK), 1)
        d_home = round(km((pa['lat'], pa['lng']), HOME), 1)
        print(f"{r['step']:>4}  {mins:>6}  {action:>12}  {str(cid):>7}  ({pa['lat']:.2f},{pa['lng']:.2f})    {d_pick:>6}    {d_home:>6}")
