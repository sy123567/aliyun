import json
from pathlib import Path
files = {
  '22:08上轮最佳': Path('results_llm/history/20260524_221959/monthly_income_202603.json'),
  '22:27本轮': Path('results_llm/monthly_income_202603.json'),
}
data={k:json.loads(p.read_text(encoding='utf-8')) for k,p in files.items()}
for k,d in data.items(): print(k, d['summary']['total_net_income_all_drivers'], d['summary']['total_preference_penalty'])
prev={d['driver_id']:d for d in data['22:08上轮最佳']['drivers']}; cur={d['driver_id']:d for d in data['22:27本轮']['drivers']}
for did in sorted(cur):
    ci=cur[did]['income']; pi=prev[did]['income']
    print(f"{did}: net {ci['net_income']-pi['net_income']:+.2f}, gross {ci['gross_income']-pi['gross_income']:+.2f}, pen {ci['preference_penalty']-pi['preference_penalty']:+.2f}")
