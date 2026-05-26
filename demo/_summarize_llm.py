import json
from pathlib import Path
p = Path('results_llm/monthly_income_202603.json')
data = json.loads(p.read_text(encoding='utf-8'))
print('SUMMARY net=', data['summary']['total_net_income_all_drivers'], 'penalty=', data['summary']['total_preference_penalty'])
for d in data['drivers']:
    inc = d['income']
    tok = d.get('token_usage', {}).get('total_tokens')
    print(f"{d['driver_id']}: gross={inc['gross_income']:.2f} cost={inc['cost']:.2f} penalty={inc['preference_penalty']:.2f} net={inc['net_income']:.2f} tokens={tok}")
    for r in d.get('preference_check', {}).get('rules', []):
        pen = r.get('penalty', 0) or 0
        if pen:
            extra = []
            if 'violations' in r: extra.append(f"viol={r['violations']}")
            if 'minutes_not_home_in_window' in r: extra.append(f"not_home={r['minutes_not_home_in_window']}")
            print(f"  - {r.get('rule')} penalty={pen} {' '.join(extra)}")
