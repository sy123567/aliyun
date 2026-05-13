from __future__ import annotations

import json
import sys
import types
from typing import Any, Protocol

simkit_pkg = types.ModuleType('simkit')
simkit_pkg.__path__ = []
ports_mod = types.ModuleType('simkit.ports')
class _ApiStub(Protocol):
    def get_driver_status(self, driver_id: str) -> dict[str, Any]: ...
    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]: ...
    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]: ...
    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...
ports_mod.SimulationApiPort = _ApiStub
sys.modules['simkit'] = simkit_pkg
sys.modules['simkit.ports'] = ports_mod
sys.path.insert(0, 'demo')

from agent import preference_parser

drivers = json.load(open('demo/server/data/drivers.json', encoding='utf-8'))
for d in drivers:
    if d['driver_id'] not in {'D009', 'D010'}:
        continue
    rules = preference_parser.parse_preferences(d['preferences'], llm_caller=None)
    print('===', d['driver_id'], '===')
    print('preferred_ids', rules.preferred_cargo_ids)
    print('preferred_cargo', rules.preferred_cargo)
    print('timed_events', rules.timed_stay_events)
    print('must_visit', rules.must_visit)
    print('home_rule', rules.home_rule)
    print('unparsed', len(rules.unparsed), [u.get('content','')[:40] for u in rules.unparsed])
