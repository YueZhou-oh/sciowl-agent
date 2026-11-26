import json
import sys

import httpx
import pytest

from scripts import probe_sciowl_raw
from sciowl_agent.contracts import tool_schemas
from sciowl_agent.model import action_schema


def test_schema_binds_arguments_to_tool():
    branches = action_schema(tool_schemas())["anyOf"]
    arguments = {b["properties"]["tool"]["enum"][0]: b["properties"]["arguments"]
                 for b in branches if "tool" in b["properties"]}
    assert set(arguments["list_potentials"]["properties"]) == {"query", "offset", "limit"}
    assert arguments["list_potentials"]["additionalProperties"] is False
    assert arguments["inspect_structure"]["required"] == ["structure_id"]
    assert branches[-1]["required"] == ["summary", "evidence_ids"]


@pytest.mark.parametrize("protocol,scenario,expected", [
    ("native", "valid", 0), ("json", "valid", 0), ("schema", "valid", 0),
    ("native", "prose", 1), ("json", "extra", 1), ("native", "length", 1),
    ("schema", "rejected", 1), ("native", "timeout", 1),
])
def test_raw_probe(tmp_path, monkeypatch, protocol, scenario, expected):
    def reply(request):
        payload = json.loads(request.content)
        assert ("response_format" in payload) == (protocol == "schema")
        if scenario == "timeout":
            raise httpx.ReadTimeout("waiting for headers", request=request)
        if scenario == "rejected":
            return httpx.Response(400, json={"error": {"message": "unsupported schema"}})
        args = {"structure_id": "list_potentials"} if scenario == "extra" else {}
        tool = "list_potentials"
        message = {"role": "assistant", "content": json.dumps({"tool": tool, "arguments": args})}
        if protocol == "native":
            message = {"role": "assistant", "content": "Use the tool."}
            if scenario == "valid":
                message["tool_calls"] = [{"id": "c", "type": "function", "function": {
                    "name": tool, "arguments": json.dumps(args)}}]
        reason = "length" if scenario == "length" else "stop"
        return httpx.Response(200, json={"choices": [{"message": message, "finish_reason": reason}]})

    client = httpx.Client(transport=httpx.MockTransport(reply))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client)
    output = tmp_path / "probe.json"
    monkeypatch.setattr(sys, "argv", ["probe", "--base-url", "http://test/v1",
                                     "--protocol", protocol, "--output", str(output)])
    assert probe_sciowl_raw.main() == expected
    report = json.loads(output.read_text())
    if scenario == "rejected":
        assert report["raw_response"]["error"]["message"] == "unsupported schema"
    if scenario == "timeout":
        assert "ReadTimeout" in report["adapter_error"]
