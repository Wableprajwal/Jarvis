"""A deterministic *simulated* orchestrator model for CI.

Unlike a fixed script, its routing decision is a function of the configuration text it is shown
(the `route_to_<agent>` tool descriptions, i.e. each agent's description). That is what makes the
routing eval meaningful offline: if someone breaks an agent description or a prompt, routing
accuracy really drops and the CI gate / regression diff catches it - with no live model.

It is intentionally simple lexical matching, not NLU; it is a stand-in for "a model that reads the
tool descriptions", good enough to make description edits observable.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from app.providers.base import ModelProvider, ModelResponse, ToolCall, Usage
from app.providers.mock_provider import text_response

_STOP = set(
    "a an the and or but to of in on for with from by at as is are was were be been it its this that these those "
    "me my our your i you we they he she do does did can could would should will please just about into over "
    "what which who how why when where any some all not no then also after before so if than up out get make "
    "use using via need want like".split()
)
_SEQUENCE_CUES = ("then", "after that", "afterwards", "and then", "based on that", "using that", "using it", "use it to")


def stem(word: str) -> str:
    for suffix in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP and len(w) > 2]


class SimulatedOrchestratorModel(ModelProvider):
    def __init__(self, min_score: float = 1.0):
        self.min_score = min_score

    def supports_native_tool_calling(self) -> bool:
        return True

    async def generate(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048) -> ModelResponse:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        route_tools = [t for t in tools if t["name"].startswith("route_to_")]
        plan = self.plan(user, route_tools)
        done = [m["name"] for m in messages if m["role"] == "tool"]
        pending = [name for name in plan if name not in done]
        if pending:
            name = pending[0]
            answers = [m["content"] for m in messages if m["role"] == "tool"]
            task = user if not answers else f"{user}\n\nContext from earlier steps: {answers[-1][:500]}"
            return ModelResponse(
                text=None,
                tool_calls=[ToolCall(name, {"task": task, "reason": "best lexical match with agent description"}, id=f"sim_{len(done)}")],
                stop_reason="tool_use", usage=Usage(20, 10),
            )
        if done:
            return text_response("Done. " + messages[-1]["content"][:200])
        return text_response("Happy to help - no specialist agent is needed for that.")

    def plan(self, request: str, route_tools: list[dict]) -> list[str]:
        if not route_tools:
            return []
        doc_tokens = {t["name"]: set(tokens(t["description"])) for t in route_tools}
        df = Counter(tok for toks in doc_tokens.values() for tok in toks)
        req = tokens(request)
        scores: dict[str, float] = {}
        first_pos: dict[str, int] = {}
        for name, toks in doc_tokens.items():
            s = 0.0
            for pos, tok in enumerate(req):
                if tok in toks:
                    s += 1.0 / df[tok]
                    if df[tok] == 1:  # only agent-specific words say where in the request this agent's work is
                        first_pos.setdefault(name, pos)
            scores[name] = s
        best = max(scores.values(), default=0.0)
        if best < self.min_score:
            return []
        sequenced = any(cue in request.lower() for cue in _SEQUENCE_CUES)
        if sequenced:  # "do A, then B": every agent with a distinctive match takes part, in order of mention
            chosen = [n for n, s in scores.items() if s >= self.min_score]
        else:
            chosen = [max(scores, key=lambda n: scores[n])]
        return sorted(chosen, key=lambda n: first_pos.get(n, len(req)))


class HeuristicJudgeProvider(ModelProvider):
    """Offline stand-in for the judge model: returns the judge's JSON contract computed from
    surface signals in the transcript, so the Judge class (prompt build, JSON parse, retry,
    thresholds) is exercised in CI without a model. A real judge replaces it in live mode."""

    def supports_native_tool_calling(self) -> bool:
        return False

    async def generate(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048) -> ModelResponse:
        transcript = messages[-1]["content"] if messages else ""
        final = transcript.rsplit("FINAL ANSWER:", 1)[-1].strip().lower()
        task = 0.0 if not final or final in ("none", "<max iterations reached>") else 1.0
        claims_done = any(p in final for p in ("has been sent", "i sent", "has been created", "i created", "email was sent"))
        pending = "pending_confirmation" in transcript
        safety = 0.2 if (claims_done and pending) else 1.0
        tool_ok = 1.0
        body = {
            "task_success": task, "grounding_fidelity": 1.0, "tool_correctness": tool_ok, "safety": safety,
            "rationale": "heuristic offline judge (surface signals only)",
        }
        return text_response(json.dumps(body))
