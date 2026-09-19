"""Ollama via its OpenAI-compatible endpoint (POST {host}/v1/chat/completions)."""
from __future__ import annotations

import json
import re
import uuid

import httpx

from app.providers.base import ModelProvider, ModelResponse, ToolCall, Usage
from app.providers.tool_call_parser import PromptedToolCallParser, render_tool_calls, render_tool_instructions

# Model families with native function-calling support in Ollama.
NATIVE_TOOL_PREFIXES = ("qwen2.5", "qwen3", "llama3.1", "llama3.2", "llama3.3", "mistral-nemo", "mistral-small", "command-r", "firefunction")

_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class ProviderError(RuntimeError):
    pass


def _lenient_json(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
    return json.loads(_TRAILING_COMMA.sub(r"\1", text))


class OllamaProvider(ModelProvider):
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        native_tools: bool | None = None,
        temperature: float = 0.1,
        timeout: float = 180.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._native = model.lower().startswith(NATIVE_TOOL_PREFIXES) if native_tools is None else native_tools
        self._client = client or httpx.AsyncClient(timeout=timeout)

    def supports_native_tool_calling(self) -> bool:
        return self._native

    async def generate(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048) -> ModelResponse:
        use_native = self._native and bool(tools)
        wire_messages = self._to_wire(messages, tools, native=use_native)
        body: dict = {
            "model": self.model,
            "messages": wire_messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if use_native:
            body["tools"] = [
                {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ]
        try:
            r = await self._client.post(f"{self.host}/v1/chat/completions", json=body)
            r.raise_for_status()
            data = r.json()
        except httpx.ConnectError as exc:
            raise ProviderError(f"cannot reach Ollama at {self.host} - is it running? ({exc})") from exc
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Ollama returned {exc.response.status_code}: {exc.response.text[:300]}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc

        choice = data["choices"][0]
        message = choice["message"]
        u = data.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        finish = choice.get("finish_reason")
        text = message.get("content") or None

        if use_native:
            return self._parse_native(message, text, finish, usage)
        if tools:
            parsed = PromptedToolCallParser(tools).parse(text)
            stop = "tool_use" if parsed.tool_calls else ("max_tokens" if finish == "length" else "end_turn")
            return ModelResponse(
                text=parsed.text if parsed.attempted else text,
                tool_calls=parsed.tool_calls, stop_reason=stop, usage=usage, error=parsed.error,
            )
        return ModelResponse(text=text, stop_reason="max_tokens" if finish == "length" else "end_turn", usage=usage)

    # --- native tool-calling ----------------------------------------------------------
    def _parse_native(self, message: dict, text: str | None, finish: str | None, usage: Usage) -> ModelResponse:
        calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _lenient_json(args) if args.strip() else {}
                except json.JSONDecodeError as exc:
                    return ModelResponse(
                        text=text, usage=usage,
                        error=f"the arguments for tool '{fn.get('name')}' were not valid JSON ({exc.msg}); send a JSON object",
                    )
            calls.append(ToolCall(name=fn.get("name", ""), arguments=args, id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"))
        stop = "tool_use" if calls else ("max_tokens" if finish == "length" else "end_turn")
        return ModelResponse(text=text, tool_calls=calls, stop_reason=stop, usage=usage)

    # --- normalized -> wire messages ---------------------------------------------------
    def _to_wire(self, messages: list[dict], tools: list[dict], native: bool) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                if native:
                    out.append(
                        {
                            "role": "assistant",
                            "content": m.get("content") or "",
                            "tool_calls": [
                                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                                for c in m["tool_calls"]
                            ],
                        }
                    )
                else:
                    out.append({"role": "assistant", "content": render_tool_calls(m["tool_calls"])})
            elif role == "tool":
                if native:
                    out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m["name"], "content": m["content"]})
                else:
                    out.append({"role": "user", "content": f"Tool result for {m['name']}:\n{m['content']}"})
            else:
                out.append({"role": role, "content": m.get("content") or ""})
        if not native and tools:
            instructions = render_tool_instructions(tools)
            if out and out[0]["role"] == "system":
                out[0] = {"role": "system", "content": out[0]["content"] + "\n\n" + instructions}
            else:
                out.insert(0, {"role": "system", "content": instructions})
        return out
