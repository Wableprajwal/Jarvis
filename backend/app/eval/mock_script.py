"""Turns a scenario's compact `mock:` step list into MockProvider script items.

Steps (consumed in order by ONE shared provider, since agents run sequentially):
  {route: code, task: "...", reason: "..."}   orchestrator calls route_to_<agent>
  {tool: read_file, args: {...}}              an agent calls a tool
  {tools: [{tool, args}, ...]}                several tool calls in one turn
  {say: "text"}                               final text reply
  {recall: "<fact key>"}                      reply with the fact value found in the context (tests injection)
"""
from __future__ import annotations

import re

from app.providers.mock_provider import ModelResponse, multi_tool_call_response, text_response


def _recall(key: str):
    def build(messages: list[dict]) -> ModelResponse:
        pattern = re.compile(rf"{re.escape(key)}\s*[:=]\s*(.+)", re.IGNORECASE)
        for m in messages:
            if m["role"] in ("system", "user") and m.get("content"):
                found = pattern.search(m["content"])
                if found and m is not messages[-1]:
                    return text_response(f"{key} is {found.group(1).strip()}")
        return text_response(f"I don't have a stored value for {key}.")

    return build


def compile_script(steps: list[dict]) -> list:
    script: list = []
    for step in steps:
        if "route" in step:
            args = {"task": step["task"]}
            if step.get("reason"):
                args["reason"] = step["reason"]
            script.append(multi_tool_call_response([(f"route_to_{step['route']}", args)]))
        elif "tool" in step:
            script.append(multi_tool_call_response([(step["tool"], step.get("args", {}))]))
        elif "tools" in step:
            script.append(multi_tool_call_response([(t["tool"], t.get("args", {})) for t in step["tools"]]))
        elif "say" in step:
            script.append(text_response(step["say"]))
        elif "recall" in step:
            script.append(_recall(step["recall"]))
        else:
            raise ValueError(f"unknown mock step: {step}")
    return script
