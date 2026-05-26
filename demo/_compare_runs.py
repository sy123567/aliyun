import json
from pathlib import Path
files = {
  '12:42基线': Path('results_llm/history/20260524_194322/monthly_income_202603.json'),
  '21:30上轮': Path('results_llm/history/20260524_212407/monthly_income_202603.json'),
  '22:08本轮': Path('results_llm/monthly_income_202603.json'),
}
data = {k: json.loads(p.read_text(encoding='utf-8')) for k,p in files.items()}
for k,d in data.items():
    print(f"\n{k}: net={d['summary']['total_net_income_all_drivers']:.2f}, penalty={d['summary']['total_preference_penalty']:.2f}")
base = {d['driver_id']: d for d in data['12:42基线']['drivers']}
prev = {d['driver_id']: d for d in data['21:30上轮']['drivers']}
cur = {d['driver_id']: d for d in data['22:08本轮']['drivers']}
print('\nDriver deltas: 本轮 - 21:30 / 本轮 - 12:42')
for did in sorted(cur):
    ci=cur[did]['income']; pi=prev[did]['income']; bi=base[did]['income']
    print(f"{did}: net Δprev={ci['net_income']-pi['net_income']:+.2f}, Δbase={ci['net_income']-bi['net_income']:+.2f} | gross cur={ci['gross_income']:.0f} prev={pi['gross_income']:.0f} base={bi['gross_income']:.0f} | pen cur={ci['preference_penalty']:.0f} prev={pi['preference_penalty']:.0f} base={bi['preference_penalty']:.0f}")
