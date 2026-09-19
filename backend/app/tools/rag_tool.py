"""`rag_search`: retrieval exposed as a TOOL the Research agent chooses to call.

It is never force-injected into prompts: always-on RAG spends context on every turn and
degrades tasks that need no retrieval (chit-chat, pure code questions); letting the agent
decide also makes retrieval visible in the trace and gradable by the eval harness.
"""
from __future__ import annotations

from app.rag.retriever import HybridRetriever
from app.tools.registry import MAX_TOOL_OUTPUT_CHARS, Tool, ToolResult

# The registry truncates tool output past MAX_TOOL_OUTPUT_CHARS (which would corrupt the JSON),
# so citation text is shortened here to keep the whole payload under that limit.
_TEXT_BUDGET_CHARS = MAX_TOOL_OUTPUT_CHARS - 1800


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search the user's own technical notes and docs (runbooks, ADRs, checklists, style guides). "
        "Returns up to 5 cited passages, each with `source` and `chunk_id`; cite them as [source#chunk_id]. "
        "An empty list means nothing relevant was found."
    )
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "minLength": 1, "description": "What to look for, in natural language"}},
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever

    async def execute(self, arguments: dict) -> ToolResult:
        query = arguments["query"].strip()
        if not query:
            return ToolResult.failure("query must not be blank")
        citations = await self._retriever.search(query)
        per_cite = max(200, _TEXT_BUDGET_CHARS // max(len(citations), 1))
        results = []
        for c in citations:
            d = c.to_dict()
            if len(d["text"]) > per_cite:
                d["text"] = d["text"][:per_cite].rstrip() + " …[truncated]"
            results.append(d)
        return ToolResult.success(results)
