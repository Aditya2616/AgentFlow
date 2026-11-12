"""LangChain interface, tool-call parsing, and the hosted adapters against stubbed HTTP."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentflow.llm.anthropic_chat import AnthropicChat, to_anthropic
from agentflow.llm.base import extract_json, parse_tool_calls
from agentflow.llm.openai_chat import LLMHTTPError, OpenAIChat, to_openai_messages
from agentflow.tools.registry import ToolBox


def test_extract_json_tolerates_fences_and_noise():
    assert extract_json('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert extract_json("noise {'a': 2} more") == {"a": 2}
    assert extract_json('{"a": {"b": "}"}}') == {"a": {"b": "}"}}
    assert extract_json("no json here") is None


def test_parse_tool_calls_qwen_format():
    text = '<think></think>ok <tool_call>\n{"name": "sql", "arguments": {"query": "SELECT 1"}}\n</tool_call>'
    rest, calls = parse_tool_calls(text)
    assert rest == "ok" and calls[0]["name"] == "sql" and calls[0]["args"] == {"query": "SELECT 1"}
    # arguments given as a JSON string, and a call cut off before the closing tag
    _, c2 = parse_tool_calls('<tool_call>{"name": "x", "arguments": "{\\"q\\": 1}"}')
    assert c2[0]["args"] == {"q": 1}


def test_scripted_and_adapters_are_langchain_chat_models(settings):
    from agentflow.llm.scripted import ScriptedChat

    for m in (ScriptedChat(), OpenAIChat(api_key="k"), AnthropicChat(api_key="k")):
        assert isinstance(m, BaseChatModel)
    bound = OpenAIChat(api_key="k").bind_tools(list(ToolBox(settings).tools.values())[:1])
    assert bound.kwargs["tools"][0]["function"]["name"] == "search"


def _openai_stub(captured: list):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        body = json.loads(request.content)
        if body.get("tools"):
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "sql", "arguments": json.dumps({"query": "SELECT 1"})}}]}
        else:
            msg = {"role": "assistant", "content": "hello"}
        return httpx.Response(200, json={"choices": [{"message": msg}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 5}})

    return httpx.MockTransport(handler)


def test_openai_adapter_request_and_parsing(settings):
    seen: list[httpx.Request] = []
    llm = OpenAIChat(api_key="sk-test", transport=_openai_stub(seen), base_url="https://example.test/v1")
    ai = llm.invoke([SystemMessage(content="be brief"), HumanMessage(content="hi")])
    assert ai.content == "hello" and ai.usage_metadata["input_tokens"] == 11
    req = seen[0]
    assert str(req.url) == "https://example.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["messages"][0] == {"role": "system", "content": "be brief"} and body["temperature"] == 0

    bound = llm.bind_tools([ToolBox(settings).tools["sql"]])
    ai = bound.invoke([HumanMessage(content="count")])
    assert ai.tool_calls[0]["name"] == "sql" and ai.tool_calls[0]["args"] == {"query": "SELECT 1"}
    assert json.loads(seen[1].content)["tools"][0]["function"]["name"] == "sql"


def test_openai_adapter_errors():
    llm = OpenAIChat(api_key="", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(LLMHTTPError, match="not set"):
        llm.invoke([HumanMessage(content="x")])
    llm = OpenAIChat(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(429, text="slow down")))
    with pytest.raises(LLMHTTPError, match="429"):
        llm.invoke([HumanMessage(content="x")])


def test_message_conversion_with_tool_results():
    ai = AIMessage(content="", tool_calls=[{"name": "sql", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}])
    out = to_openai_messages([HumanMessage(content="hi"), ai])
    assert out[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "q"}'
    system, msgs = to_anthropic([SystemMessage(content="s"), HumanMessage(content="hi"), ai])
    assert system == "s" and msgs[1]["content"][0]["type"] == "tool_use"


def test_anthropic_adapter_request_and_parsing(settings):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "calling"},
                        {"type": "tool_use", "id": "tu_1", "name": "sql", "input": {"query": "SELECT 2"}}],
            "usage": {"input_tokens": 20, "output_tokens": 7}})

    llm = AnthropicChat(api_key="ak-test", transport=httpx.MockTransport(handler), base_url="https://example.test")
    ai = llm.bind_tools([ToolBox(settings).tools["sql"]]).invoke(
        [SystemMessage(content="sys"), HumanMessage(content="count")])
    req = seen[0]
    assert str(req.url) == "https://example.test/v1/messages"
    assert req.headers["x-api-key"] == "ak-test" and req.headers["anthropic-version"] == "2023-06-01"
    body = json.loads(req.content)
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "count"}]
    assert body["tools"][0]["name"] == "sql" and "properties" in body["tools"][0]["input_schema"]
    assert ai.content == "calling" and ai.tool_calls[0]["args"] == {"query": "SELECT 2"}
    assert ai.usage_metadata["output_tokens"] == 7


def test_agent_runs_end_to_end_on_a_stubbed_openai_backend(tasks, settings):
    """The same graph works with a hosted-style backend: plan JSON, then tool calls."""
    from agentflow.graph import AgentRunner
    from agentflow.tracing.writer import Tracer

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        if "[phase:plan" in system:
            msg = {"role": "assistant", "content": json.dumps(
                {"steps": [{"goal": "count orders", "tool": "sql"}]})}
        elif "[phase:act" in system:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c", "type": "function",
                "function": {"name": "sql", "arguments": json.dumps({"query": "SELECT COUNT(*) FROM orders"})}}]}
        else:
            msg = {"role": "assistant", "content": "There are 640 orders."}
        return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {}})

    llm = OpenAIChat(api_key="k", transport=httpx.MockTransport(handler))
    res = AgentRunner(llm, ToolBox(settings), Tracer()).run("How many orders are there?")
    assert res.status == "completed" and "640" in res.answer
