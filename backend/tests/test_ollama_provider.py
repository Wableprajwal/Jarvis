"""OllamaProvider wire behaviour, tested with httpx.MockTransport (no live server)."""
import json

import httpx
import pytest

from app.harness.loop import run_agent_loop
from app.observability.trace import InMemoryTracer
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from app.providers.ollama_provider import OllamaProvider, ProviderError
from app.tools.builtin import GetTimeTool
from app.tools.registry import ToolRegistry

TOOLS = [GetTimeTool().schema]


def make(handler, **kw):
    return OllamaProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


def reply(message: dict, finish="stop", usage=(7, 3)):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        },
    )


async def test_native_request_shape_and_tool_call_parsing():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return reply(
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}]},
            finish="tool_calls",
        )

    p = make(handler, model="qwen2.5:7b")
    assert p.supports_native_tool_calling()
    resp = await p.generate(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "time?"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "name": "get_time", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "x", "name": "get_time", "content": "{}"},
        ],
        TOOLS,
    )
    assert seen["url"].endswith("/v1/chat/completions")
    body = seen["body"]
    assert body["tools"][0]["function"]["name"] == "get_time"
    assert body["tools"][0]["function"]["parameters"] == TOOLS[0]["input_schema"]
    assert body["messages"][2]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "x", "name": "get_time", "content": "{}"}
    assert resp.stop_reason == "tool_use" and resp.tool_calls[0].name == "get_time"
    assert resp.tool_calls[0].id == "c1" and (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (7, 3)


async def test_native_malformed_arguments_become_error_not_exception():
    p = make(lambda r: reply({"content": "", "tool_calls": [
        {"function": {"name": "get_time", "arguments": "{not json"}}]}))
    resp = await p.generate([{"role": "user", "content": "x"}], TOOLS)
    assert resp.tool_calls == [] and "not valid JSON" in resp.error


async def test_plain_text_and_max_tokens():
    p = make(lambda r: reply({"content": "hello"}, finish="length"))
    resp = await p.generate([{"role": "user", "content": "x"}], TOOLS)
    assert resp.text == "hello" and resp.stop_reason == "max_tokens" and not resp.tool_calls


async def test_prompted_mode_injects_schemas_and_parses_block():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return reply({"content": '```json\n{"tool": "get_time", "arguments": {}}\n```'})

    p = make(handler, model="some-model:1b")
    assert not p.supports_native_tool_calling()
    resp = await p.generate(
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "time?"}], TOOLS
    )
    body = seen["body"]
    assert "tools" not in body
    assert body["messages"][0]["content"].startswith("be nice") and "get_time" in body["messages"][0]["content"]
    assert resp.tool_calls[0].name == "get_time" and resp.stop_reason == "tool_use"


async def test_prompted_mode_translates_tool_history():
    seen = {}

    def handler(request):
        seen["messages"] = json.loads(request.content)["messages"]
        return reply({"content": "done"})

    p = make(handler, native_tools=False)
    await p.generate(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "get_time", "arguments": {}}]},
            {"role": "tool", "tool_call_id": "1", "name": "get_time", "content": '{"status": "ok"}'},
        ],
        TOOLS,
    )
    msgs = seen["messages"]
    assert msgs[0]["role"] == "system"
    assert "```json" in msgs[2]["content"]
    assert msgs[3]["role"] == "user" and "Tool result for get_time" in msgs[3]["content"]


async def test_prompted_mode_malformed_block_sets_error():
    p = make(lambda r: reply({"content": '```json\n{"tool": "nope", "arguments": {}}\n```'}), native_tools=False)
    resp = await p.generate([{"role": "user", "content": "x"}], TOOLS)
    assert resp.tool_calls == [] and "unknown tool 'nope'" in resp.error


async def test_connection_failure_is_a_clear_provider_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderError, match="is it running"):
        await make(handler).generate([{"role": "user", "content": "x"}], TOOLS)


async def test_http_error_is_a_provider_error():
    with pytest.raises(ProviderError, match="500"):
        await make(lambda r: httpx.Response(500, text="boom")).generate([{"role": "user", "content": "x"}], TOOLS)


async def test_loop_feeds_parse_error_back_and_model_recovers():
    """Harness behaviour for malformed tool output: feed the error back, don't crash."""
    registry = ToolRegistry()
    registry.register(GetTimeTool())
    registry.grant("agent", ["get_time"])
    bad = text_response(None)
    bad.error = "unknown tool 'nope'"
    provider = MockProvider(script=[bad, tool_call_response("get_time", {}), text_response("it is late")])
    result = await run_agent_loop(
        provider, [{"role": "user", "content": "time?"}], registry.tools_for("agent"), registry,
        InMemoryTracer(), "s1", "agent",
    )
    assert result.text == "it is late"
    feedback = provider.calls[1]["messages"][-1]
    assert feedback["role"] == "user" and "unknown tool 'nope'" in feedback["content"]
