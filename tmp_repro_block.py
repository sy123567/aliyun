"""复现 D010 min 12866 选取 cargo 280288 的评分，确认硬约束是否触发。"""
from __future__ import annotations
import json, sys, types
from typing import Any, Protocol

simkit_pkg = types.ModuleType('simkit'); simkit_pkg.__path__ = []
ports_mod = types.ModuleType('simkit.ports')
class _ApiStub(Protocol):
    def get_driver_status(self, driver_id: str) -> dict: ...
    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict: ...
    def query_decision_history(self, driver_id: str, step: int) -> dict: ...
    def model_chat_completion(self, payload: dict) -> dict: ...
ports_mod.SimulationApiPort = _ApiStub
sys.modules['simkit'] = simkit_pkg
sys.modules['simkit.ports'] = ports_mod
sys.path.insert(0, 'demo')

from agent import preference_parser, scoring, driver_memory, config as agent_config

drivers = json.load(open('demo/server/data/drivers.json', encoding='utf-8'))
d010 = next(d for d in drivers if d['driver_id'] == 'D010')
rules = preference_parser.parse_preferences(d010['preferences'], llm_caller=None)
print('events:', rules.timed_stay_events)

mem = driver_memory.DriverMemory(driver_id='D010')
mem.rules = rules
ctx = scoring.DecisionContext(
    driver_id='D010', cost_per_km=1.5, truck_length=str(d010.get('truck_length','')),
    current_lat=23.13, current_lng=113.26, current_minutes=12866,
    horizon_minutes=agent_config.AGENT_HORIZON_MINUTES, reposition_speed_km_per_hour=60.0,
    opportunity_cost_per_minute=0.5,
)
print('horizon =', ctx.horizon_minutes)
event = rules.timed_stay_events[0]
print('phase =', scoring.timed_event_phase(event, mem, ctx))

# 构造一个假定的 cargo 280288 candidate:
# 已知 pickup_deadhead=11.34, haul=307.79, 完工于 14335 = 12866 + 1469
# 推断 cost_time_minutes ≈ 1469 - 11.34/60*60(=11) ≈ ~1450 min
cargo_item = {
    'cargo': {
        'cargo_id': '280288',
        'cargo_name': '其他',
        'price': 8000.0,
        'cost_time_minutes': 1450,
        'start': {'lat': 23.22, 'lng': 113.27},
        'end': {'lat': 23.38, 'lng': 116.11},
        'load_time': ['2026-03-09 22:30', '2026-03-09 23:30'],
        'truck_length': None,
    },
    'distance_km': 11.34,
}
res = scoring.score_take_order(cargo_item, rules, mem, ctx)
print('score=', res.score, 'feasible=', res.feasible, 'note=', res.note)
print('breakdown:', res.breakdown)
