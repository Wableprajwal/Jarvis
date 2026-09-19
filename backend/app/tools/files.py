"""File tools for the Code agent. Phase 1 ships read_file; git_diff_summarize and
generate_tests are added in later phases."""
from __future__ import annotations

import ast
import asyncio
import os
import re
from pathlib import Path

from app.providers.base import ModelProvider
from app.tools.paths import PathRefused, resolve_inside
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
        try:
            target = resolve_inside(self._root, arguments["path"])
        except PathRefused as exc:
            return ToolResult.failure(str(exc))
        if target.stat().st_size > MAX_READ_BYTES:
            return ToolResult.failure(f"file too large (> {MAX_READ_BYTES} bytes)")
        text = await asyncio.to_thread(target.read_text, "utf-8", "replace")
        start, end = arguments.get("start_line"), arguments.get("end_line")
        if start or end:
            lines = text.splitlines()
            text = "\n".join(lines[(start or 1) - 1 : end])
        return ToolResult.success({"path": arguments["path"], "content": text})


_REF = re.compile(r"^[\w./~^@{}:-]+$")
MAX_DIFF_CHARS = 30_000


class GitDiffSummarizeTool(Tool):
    """Gathers what the model needs to write a PR description: the diffstat plus the (bounded)
    diff. It deliberately does not call a model itself - the Code agent's own model writes the
    summary, so there is one place where prose is generated and graded."""

    name = "git_diff_summarize"
    description = (
        "Get the git diff of the workspace repository (diffstat plus patch) so it can be summarized "
        "into a PR description. Compares the working tree to `base` (default HEAD); set staged=true "
        "for the index only."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "base": {"type": "string", "description": "Git ref to diff against, e.g. main or HEAD~1. Default HEAD."},
            "staged": {"type": "boolean"},
            "path": {"type": "string", "description": "Optionally restrict to one path"},
        },
        "additionalProperties": False,
    }

    def __init__(self, root: str | Path):
        self._root = Path(root).resolve()

    async def _git(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git", "-C", str(self._root), *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env={"PATH": os.environ.get("PATH", ""), "HOME": "/nonexistent", "GIT_CONFIG_NOSYSTEM": "1"},
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=20)
        return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def execute(self, arguments: dict) -> ToolResult:
        base = arguments.get("base", "HEAD")
        if not _REF.match(base) or base.startswith("-"):
            return ToolResult.failure("invalid git ref")
        path = arguments.get("path")
        if path is not None:
            try:
                resolve_inside(self._root, path, must_exist=False)  # may be a directory or a deleted file
            except PathRefused as exc:
                return ToolResult.failure(str(exc))
        flags = ["--cached"] if arguments.get("staged") else []
        tail = ["--", path] if path else []
        code, stat, err = await self._git("diff", *flags, "--stat", base, *tail)
        if code != 0:
            return ToolResult.failure(f"git diff failed: {err.strip() or 'not a git repository?'}")
        _, patch, _ = await self._git("diff", *flags, base, *tail)
        if not patch.strip():
            return ToolResult.success({"stat": "", "diff": "", "note": "no changes"})
        truncated = len(patch) > MAX_DIFF_CHARS
        return ToolResult.success(
            {"stat": stat.strip(), "diff": patch[:MAX_DIFF_CHARS], "truncated": truncated}
        )


TEST_GEN_PROMPT = (
    "You write pytest unit tests. Given the source file below, write a complete, runnable pytest module "
    "for {target}. Cover normal cases, edge cases and error handling. Import the code under test the way "
    "the file's path implies. Output ONLY one ```python fenced block, no commentary.\n\n"
    "# path: {path}\n{source}"
)
_CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


class GenerateTestsTool(Tool):
    """Generates pytest tests for a file/function using its own (typically stronger) model, and
    verifies the output at least parses as Python before returning it."""

    name = "generate_tests"
    description = (
        "Generate pytest unit tests for a Python source file, or for one function/class in it. "
        "Returns test code (checked to be syntactically valid); it does not run it - use code_exec to run."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Source file path relative to the workspace root"},
            "target": {"type": "string", "description": "Optional function or class name to focus on"},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, root: str | Path, provider: ModelProvider, max_attempts: int = 2):
        self._root = Path(root).resolve()
        self._provider = provider
        self._max_attempts = max_attempts

    async def execute(self, arguments: dict) -> ToolResult:
        try:
            src = resolve_inside(self._root, arguments["path"])
        except PathRefused as exc:
            return ToolResult.failure(str(exc))
        if src.stat().st_size > MAX_READ_BYTES:
            return ToolResult.failure("file too large")
        source = await asyncio.to_thread(src.read_text, "utf-8", "replace")
        target = arguments.get("target") or "the whole file"
        messages = [
            {"role": "system", "content": "You are a meticulous test engineer."},
            {"role": "user", "content": TEST_GEN_PROMPT.format(target=target, path=arguments["path"], source=source)},
        ]
        error = "no python code block in the reply"
        for _ in range(self._max_attempts):
            resp = await self._provider.generate(messages, [], max_tokens=2048)
            m = _CODE_BLOCK.search(resp.text or "")
            if m:
                code = m.group(1).strip()
                try:
                    ast.parse(code)
                    return ToolResult.success({"path_under_test": arguments["path"], "test_code": code})
                except SyntaxError as exc:
                    error = f"generated tests are not valid Python: {exc.msg} (line {exc.lineno})"
            messages += [
                {"role": "assistant", "content": resp.text or ""},
                {"role": "user", "content": f"That was rejected: {error}. Output only one corrected ```python block."},
            ]
        return ToolResult.failure(f"could not produce valid tests: {error}")
