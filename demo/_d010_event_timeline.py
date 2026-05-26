import json
from pathlib import Path
for name,root in [('base',Path('results_llm/history/20260524_194322')),('cur',Path('results_llm'))]:
    f=list(root.glob('actions_202603_D010_*.jsonl'))[0]
    print('\n'+name)
    for line in f.read_text(encoding='utf-8').splitlines():
        r=json.loads(line); t=r.get('simulation_end_time','')
        if '2026-03-09' in t or '2026-03-10' in t or '2026-03-11' in t or '2026-03-12' in t or '2026-03-13' in t:
            a=r['action']['action']; res=r.get('result',{}); p=r.get('position_after',{})
            print(r['step'], t, a, r['action'].get('params'), 'pos', round(p.get('lat',0),3), round(p.get('lng',0),3), 'elapsed', r.get('step_elapsed_minutes'), 'res', {k:res.get(k) for k in ['cargo_id','pickup_deadhead_km','haul_distance_km','distance_km','simulation_wall_time']})
