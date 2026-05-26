import json
from pathlib import Path
runs={'base': Path('results_llm/history/20260524_194322'),'cur': Path('results_llm')}
for did in ['D002','D007','D010']:
    print('\n'+did)
    for name,root in runs.items():
        files=list(root.glob(f'actions_202603_{did}_*.jsonl'))
        rows=[]
        if files:
            for line in files[0].read_text(encoding='utf-8').splitlines():
                if line.strip(): rows.append(json.loads(line))
        takes=[r for r in rows if r.get('action',{}).get('action')=='take_order' and r.get('result',{}).get('accepted')]
        waits=[r for r in rows if r.get('action',{}).get('action')=='wait']
        repos=[r for r in rows if r.get('action',{}).get('action')=='reposition']
        gross=sum(float(r.get('result',{}).get('income_yuan',0) or r.get('result',{}).get('income',0) or 0) for r in takes)
        dh=sum(float(r.get('result',{}).get('pickup_deadhead_km',0) or 0) for r in takes)+sum(float(r.get('result',{}).get('distance_km',0) or 0) for r in repos)
        haul=sum(float(r.get('result',{}).get('haul_distance_km',0) or 0) for r in takes)
        print(name, 'steps',len(rows),'orders',len(takes),'gross_log',round(gross,2),'avg_order',round(gross/max(1,len(takes)),1),'haul',round(haul,1),'dh',round(dh,1),'waits',len(waits),'repos',len(repos))
