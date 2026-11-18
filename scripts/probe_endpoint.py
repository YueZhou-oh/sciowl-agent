#!/usr/bin/env python3
"""Non-executing model transport/tool-call probe against your actual SciOWL server."""
import argparse
import json
import os
from pathlib import Path
from sciowl_agent.model import EndpointModel, SYSTEM
from sciowl_agent.contracts import tool_schemas


def matches_call(message, expected, arguments):
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return False
    function = calls[0]["function"]
    return function["name"] == expected and json.loads(function["arguments"]) == arguments


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("SCIOWL_BASE_URL", "http://127.0.0.1:8005/v1"))
    p.add_argument("--model", default=os.environ.get("SCIOWL_MODEL", "sciowl"))
    p.add_argument("--protocol", choices=["native", "json", "schema"], default="native")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Completion budget; 1024 often truncates Qwen-style thinking+tool calls")
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output", default="endpoint_probe.json")
    a = p.parse_args()
    if not a.base_url:
        p.error("--base-url or SCIOWL_BASE_URL is required")
    if a.max_tokens < 1:
        p.error("--max-tokens must be positive")

    model = EndpointModel(a.base_url, a.model, a.protocol, max_tokens=a.max_tokens, timeout=a.timeout)
    cases = [
        ("selection", "List the available registered potentials using the provided tool.", "list_potentials"),
        ("arguments", "Inspect the registered structure with ID structure_probe using the appropriate tool.", "inspect_structure"),
        ("no_tool", "Explain conceptually what a force convergence tolerance means. No calculation is requested.", None),
    ]
    results = []
    for name, query, expected in cases:
        try:
            msg = model.next(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
                tool_schemas(),
            )
            calls = msg.get("tool_calls") or []
            if expected is None:
                final = json.loads(msg.get("content") or "{}")
                valid = (not calls and isinstance(final, dict) and bool(final.get("summary"))
                         and isinstance(final["summary"], str) and final.get("evidence_ids") == [])
            else:
                arguments = {"structure_id": "structure_probe"} if expected == "inspect_structure" else {}
                valid = matches_call(msg, expected, arguments)
            results.append({"case": name, "passed": valid, "response": msg})
        except Exception as exc:
            results.append({"case": name, "passed": False, "error": str(exc)})
    Path(a.output).write_text(json.dumps({
        "model": a.model,
        "protocol": a.protocol,
        "max_tokens": a.max_tokens,
        "results": results,
        "notice": "No scientific calculations executed; this is a small interface probe, not an agent benchmark.",
    }, indent=2))
    print(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()