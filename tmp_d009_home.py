import json, glob, math
path = sorted(glob.glob('demo/results/actions_202603_D009_*.jsonl'))[-1]
rows = [json.loads(line) for line in open(path, encoding='utf-8')]
def km(a, b): return 2*6371*math.asin(math.sqrt(math.sin(math.radians(b[0]-a[0])/2)**2 + math.cos(math.radians(a[0]))*math.cos(math.radians(b[0]))*math.sin(math.radians(b[1]-a[1])/2)**2))
HOME = (23.21, 113.37)
print('step  sim_min  day_min hour action       exec_min  ->lat,lng    km_to_home')
for r in rows:
    mins = int(r['result'].get('simulation_progress_minutes', 0))
    action = r['action']['action']
    pa = r['position_after']
    exec_min = r.get('action_exec_cost_minutes', 0)
    d_home = round(km((pa['lat'], pa['lng']), HOME), 2)
    day_min = mins % 1440
    hour = day_min // 60
    if 21 <= hour <= 24 or hour <= 6:
        print(f"{r['step']:>4}  {mins:>6}  {day_min:>4}  {hour:>2}  {action:<12}  {exec_min:>6}  ({pa['lat']:.2f},{pa['lng']:.2f})    {d_home:>6}")
