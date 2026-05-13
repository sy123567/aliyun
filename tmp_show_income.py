import json
data = json.load(open('demo/results/monthly_income_202603.json', encoding='utf-8'))
for d in data['drivers']:
    inc = d['income']
    print(f"{d['driver_id']:>5} net={inc['net_income']:>10.2f}  gross={inc['gross_income']:>10.2f}  pref_penalty={inc['preference_penalty']:>8.0f}")
print('TOTAL net=', data['summary']['total_net_income_all_drivers'], 'pref=', data['summary']['total_preference_penalty'])
