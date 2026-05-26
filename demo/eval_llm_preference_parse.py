"""用真实 LLM 跑偏好解析评测集。

用法（PowerShell）：
    $env:DASHSCOPE_API_KEY = "<your_key>"
    python eval_llm_preference_parse.py

输出：
- 每条 case：[PASS/FAIL] id  原因
- 统计：基础 fixtures 命中率、变体 gaps 命中率（变体越多过越好）
- 退出码：基础 fixtures 全过=0，否则=1（gaps 失败不算 fail，因为它本来就是 LLM 验证目标）

不修改全局状态；只用于人工触发评测。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

_DEMO = Path(__file__).resolve().parent
sys.path.insert(0, str(_DEMO))

from agent.preference_parser import parse_preferences  # noqa: E402
from server.bench.model_gateway_client import ModelGatewayClient  # noqa: E402
from server.bench.settings import load_settings  # noqa: E402
from tests.test_preference_parse_cases import _check_expectation  # noqa: E402


_BASE_FIXTURE = _DEMO / "tests" / "fixtures" / "preference_parse_cases.json"
_GAPS_FIXTURE = _DEMO / "tests" / "fixtures" / "preference_parse_regex_gaps.json"


def _build_llm_caller(client: ModelGatewayClient):
    def _caller(payload: dict[str, Any]) -> dict[str, Any]:
        resp = client.chat_completion(payload)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            return {}
        return data

    return _caller


def _run_section(name: str, cases: list[dict[str, Any]], caller) -> tuple[int, int]:
    print(f"\n=== {name} ({len(cases)} cases) ===")
    pass_count = 0
    for case in cases:
        pref = {
            "content": case["content"],
            "penalty_amount": case.get("penalty_amount"),
            "penalty_cap": case.get("penalty_cap"),
        }
        rules = parse_preferences([pref], llm_caller=caller)
        ok, reason = _check_expectation(case, rules)
        status = "PASS" if ok else "FAIL"
        used = "llm" if rules.parsed_by_llm else ("regex" if rules.parsed_by_regex else "none")
        if ok:
            pass_count += 1
            print(f"  [{status}] {case['id']:<35} via={used}")
        else:
            print(f"  [{status}] {case['id']:<35} via={used}  reason={reason}")
    return pass_count, len(cases)


def main() -> int:
    config_path = _DEMO / "server" / "config" / "config_llm.json"
    settings = load_settings(config_path)
    print(f"model={settings.model_name} url={settings.model_api_url}")

    client = ModelGatewayClient(
        api_url=settings.model_api_url,
        api_key=settings.model_api_key,
        default_model_name=settings.model_name,
        timeout_seconds=settings.model_timeout_seconds,
    )
    try:
        caller = _build_llm_caller(client)
        base = json.loads(_BASE_FIXTURE.read_text(encoding="utf-8"))
        gaps = json.loads(_GAPS_FIXTURE.read_text(encoding="utf-8"))
        base_pass, base_total = _run_section("基础 fixtures", base, caller)
        gaps_pass, gaps_total = _run_section("正则盲区变体 (LLM 关键考核)", gaps, caller)

        print("\n=== 总结 ===")
        print(f"  基础  : {base_pass}/{base_total}")
        print(f"  变体  : {gaps_pass}/{gaps_total}  (越高越说明 LLM 有效)")
        return 0 if base_pass == base_total else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
