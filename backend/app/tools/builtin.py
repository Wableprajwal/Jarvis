from __future__ import annotations

from datetime import UTC, datetime

from app.tools.registry import Tool, ToolResult


class GetTimeTool(Tool):
    name = "get_time"
    description = "Return the current date and time in UTC (ISO 8601)."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.success({"utc": datetime.now(UTC).isoformat()})
