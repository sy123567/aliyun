import json, glob, math
path = sorted(glob.glob('demo/results/actions_202603_D010_*.jsonl'))[-1]
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
def km(a, b): return 2*6371*math.asin(math.sqrt(math.sin(math.radians(b[0]-a[0])/2)**2 + math.cos(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.sin(math.radians(b[1]-a[1])/2)**2))
PICK = (23.21, 113.37)
HOME = (23.19, 113.36)
print('file:', path)
print('step  sim_min  action       exec_min  ->lat,lng    km_to_pick  km_to_home')
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    if 14300 <= mins <= 18800:
        action = r['action']['action']
        params = r['action'].get('params') or {}
        pa = r['position_after']
        exec_min = r.get('action_exec_cost_minutes', 0)
        d_pick = round(km((pa['lat'], pa['lng']), PICK), 2)
        d_home = round(km((pa['lat'], pa['lng']), HOME), 2)
        cargo = params.get('cargo_id') or params.get('duration_minutes') or ''
        print(f"{r['step']:>4}  {mins:>6}  {action:<12}  {exec_min:>6}  ({pa['lat']:.2f},{pa['lng']:.2f})    {d_pick:>6}    {d_home:>6}  {cargo}")
