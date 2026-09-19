"""File tools for the Code agent. Phase 1 ships read_file; git_diff_summarize and
generate_tests are added in later phases."""
from __future__ import annotations

import asyncio
from pathlib import Path

from app.tools.registry import Tool, ToolResult

MAX_READ_BYTES = 200_000


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a UTF-8 text file from the workspace. Paths are relative to the workspace root. "
        "Optionally restrict to a 1-indexed inclusive line range."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path relative to the workspace root"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()

    async def execute(self, arguments: dict) -> ToolResult:
        target = (self._root / arguments["path"]).resolve()
        if not target.is_relative_to(self._root):
            return ToolResult.failure("path escapes the workspace root")
        if not target.is_file():
            return ToolResult.failure(f"file not found: {arguments['path']}")
        if target.stat().st_size > MAX_READ_BYTES:
            return ToolResult.failure(f"file too large (> {MAX_READ_BYTES} bytes)")
        text = await asyncio.to_thread(target.read_text, "utf-8", "replace")
        start, end = arguments.get("start_line"), arguments.get("end_line")
        if start or end:
            lines = text.splitlines()
            text = "\n".join(lines[(start or 1) - 1 : end])
        return ToolResult.success({"path": arguments["path"], "content": text})
