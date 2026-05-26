import json
from pathlib import Path
cargo={}
for line in Path('server/data/cargo_dataset.jsonl').read_text(encoding='utf-8').splitlines():
    if line.strip():
        x=json.loads(line); cargo[str(x['cargo_id'])]=float(x.get('price',0))/100
runs={'base': Path('results_llm/history/20260524_194322'),'cur': Path('results_llm')}
for did in ['D002','D007','D010']:
    print('\n'+did)
    for name,root in runs.items():
        f=list(root.glob(f'actions_202603_{did}_*.jsonl'))[0]
        rows=[json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        takes=[r for r in rows if r.get('action',{}).get('action')=='take_order' and r.get('result',{}).get('accepted')]
        prices=[cargo.get(str(r['action']['params'].get('cargo_id')),0) for r in takes]
        print(name, 'orders',len(takes),'gross',round(sum(prices),2),'avg',round(sum(prices)/max(1,len(prices)),1),'top5', [round(x,1) for x in sorted(prices, reverse=True)[:5]])
        print(' first/last orders', takes[0]['simulation_end_time'] if takes else None, takes[-1]['simulation_end_time'] if takes else None)
