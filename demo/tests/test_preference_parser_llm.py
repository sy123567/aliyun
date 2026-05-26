from __future__ import annotations

import json
import os
import sys
import unittest
from typing import Any

_DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _DEMO_DIR not in sys.path:
    sys.path.insert(0, _DEMO_DIR)

from agent.preference_parser import parse_preferences  # noqa: E402


def _mock_response(rules: list[dict[str, Any]]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps({"rules": rules}, ensure_ascii=False)}}]}


class LlmPreferenceParserSchemaTest(unittest.TestCase):
    def test_llm_parses_timed_stay_event(self) -> None:
        prefs = [
            {
                "content": "3月10日10点接配偶并回家，22点前进家门，至少待到3月13日22点，每分钟罚5元。",
                "penalty_amount": 9000,
                "penalty_cap": None,
            }
        ]

        def caller(_: dict[str, Any]) -> dict[str, Any]:
            return _mock_response(
                [
                    {
                        "index": 0,
                        "kind": "timed_stay_event",
                        "params": {
                            "start_minutes": 13560,
                            "pickup_lat": 23.21,
                            "pickup_lng": 113.37,
                            "home_lat": 23.19,
                            "home_lng": 113.36,
                            "deadline_minutes": 14280,
                            "stay_until_minutes": 18600,
                            "pickup_stay_minutes": 10,
                            "radius_km": 1.0,
                            "absence_penalty_per_minute": 5.0,
                        },
                        "penalty_amount": 9000,
                        "penalty_cap": None,
                    }
                ]
            )

        rules = parse_preferences(prefs, llm_caller=caller)

        self.assertTrue(rules.llm_used)
        self.assertEqual(rules.parsed_by_llm, 1)
        self.assertEqual(len(rules.timed_stay_events), 1)
        event = rules.timed_stay_events[0]
        self.assertEqual(event.start_minutes, 13560)
        self.assertEqual(event.deadline_minutes, 14280)
        self.assertEqual(event.stay_until_minutes, 18600)
        self.assertAlmostEqual(event.absence_penalty_per_minute, 5.0)

    def test_llm_parses_home_and_no_drive_combo(self) -> None:
        prefs = [
            {
                "content": "每天23点前在家，23点至次日8点不接单不空跑。",
                "penalty_amount": 900,
                "penalty_cap": None,
            }
        ]

        def caller(_: dict[str, Any]) -> dict[str, Any]:
            return _mock_response(
                [
                    {
                        "index": 0,
                        "kind": "home_and_no_drive",
                        "params": {
                            "lat": 23.12,
                            "lng": 113.28,
                            "radius_km": 1.0,
                            "home_by_hour": 23,
                            "no_drive_until_hour": 8,
                            "no_drive_start_minute": 1380,
                            "no_drive_end_minute": 1920,
                        },
                    }
                ]
            )

        rules = parse_preferences(prefs, llm_caller=caller)

        self.assertIsNotNone(rules.home_rule)
        self.assertEqual(rules.home_rule.home_by_hour, 23)
        self.assertEqual(rules.home_rule.no_drive_until_hour, 8)
        self.assertEqual(len(rules.no_drive_windows), 1)
        self.assertEqual(rules.no_drive_windows[0].start_minute, 1380)
        self.assertEqual(rules.no_drive_windows[0].end_minute, 1920)

    def test_llm_parses_soft_no_drive_window(self) -> None:
        prefs = [
            {
                "content": "中午12点到13点尽量不接单不空跑，每天罚100，封顶3000。",
                "penalty_amount": 100,
                "penalty_cap": 3000,
            }
        ]

        def caller(_: dict[str, Any]) -> dict[str, Any]:
            return _mock_response(
                [
                    {
                        "index": 0,
                        "kind": "soft_no_drive_window",
                        "params": {"start_minute": 720, "end_minute": 780},
                    }
                ]
            )

        rules = parse_preferences(prefs, llm_caller=caller)

        self.assertEqual(len(rules.no_drive_windows), 1)
        window = rules.no_drive_windows[0]
        self.assertEqual(window.start_minute, 720)
        self.assertEqual(window.end_minute, 780)
        self.assertEqual(window.penalty_amount, 100)
        self.assertEqual(window.penalty_cap, 3000)

    def test_invalid_llm_coordinates_do_not_pollute_rules(self) -> None:
        prefs = [{"content": "禁入某区域。", "penalty_amount": 1000, "penalty_cap": None}]

        def caller(_: dict[str, Any]) -> dict[str, Any]:
            return _mock_response(
                [
                    {
                        "index": 0,
                        "kind": "forbidden_zone",
                        "params": {"lat": 123.0, "lng": 113.0, "radius_km": 1.0},
                    }
                ]
            )

        rules = parse_preferences(prefs, llm_caller=caller)

        self.assertEqual(len(rules.forbidden_zones), 0)
        self.assertEqual(rules.parse_failure_count, 1)


if __name__ == "__main__":
    unittest.main()
