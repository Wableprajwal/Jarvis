"""Edge cases found in the second audit pass. Each test documents the failure it guards against."""
import httpx
import pytest

from app.agents.base import SubAgent
from app.harness.loop import AgentResult
from app.observability.trace import InMemoryTracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from app.providers.ollama_provider import OllamaProvider, ProviderError
from app.tools.builtin import GetTimeTool
from app.tools.files import GenerateTestsTool, GitDiffSummarizeTool, ReadFileTool
from app.tools.paths import PathRefused, is_sensitive, resolve_inside
from app.tools.registry import Tool, ToolRegistry, ToolResult


# ---------------------------------------------------------------- sensitive files
class TestWorkspacePathPolicy:
    """With WORKSPACE_ROOT=. the project's own .env was one read_file call from the model."""

    @pytest.fixture
    def ws(self, tmp_path):
        (tmp_path / ".env").write_text("DATABASE_URL=postgresql://u:hunter2@db/x\n")
        (tmp_path / "ok.py").write_text("x = 1\n")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[remote]\n")
        (tmp_path / "keys").mkdir()
        (tmp_path / "keys" / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\n")
        (tmp_path / "innocent.txt").symlink_to(tmp_path / ".env")  # a symlink to a secret, inside the workspace
        return tmp_path

    @pytest.mark.parametrize("path", [".env", ".git/config", "keys/deploy.pem", "innocent.txt"])
    async def test_read_file_refuses_sensitive_paths(self, ws, path):
        out = await ReadFileTool(ws).execute({"path": path})
        assert not out.ok and "sensitive" in out.error, out
        assert "hunter2" not in str(out.output)

    async def test_read_file_still_reads_ordinary_files(self, ws):
        assert (await ReadFileTool(ws).execute({"path": "ok.py"})).ok

    async def test_generate_tests_and_git_diff_share_the_policy(self, ws):
        gen = await GenerateTestsTool(ws, MockProvider()).execute({"path": ".env"})
        assert not gen.ok and "sensitive" in gen.error
        diff = await GitDiffSummarizeTool(ws).execute({"path": ".env"})
        assert not diff.ok and "sensitive" in diff.error

    def test_resolve_inside_rejects_traversal_absolute_and_control_chars(self, ws):
        for raw in ["../x", "/etc/passwd", "a\x00b", "", "."]:
            with pytest.raises(PathRefused):
                resolve_inside(ws, raw, must_exist=False)

    def test_is_sensitive_is_case_insensitive_and_directory_aware(self):
        from pathlib import PurePosixPath as P

        assert is_sensitive(P("Prod.ENV")) and is_sensitive(P("node_modules/x/index.js"))
        assert not is_sensitive(P("src/environment.py")) and not is_sensitive(P("README.md"))


# ---------------------------------------------------------------- provider response shapes
def _provider(handler):
    return OllamaProvider(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), model="qwen2.5:7b")


class TestOllamaMalformedResponses:
    """A 200 with the wrong body used to escape as KeyError/IndexError -> a 500 crash, not a 502."""

    async def test_empty_choices(self):
        p = _provider(lambda r: httpx.Response(200, json={"choices": [], "usage": {}}))
        with pytest.raises(ProviderError, match="no choices"):
            await p.generate([{"role": "user", "content": "x"}], [GetTimeTool().schema])

    async def test_error_body_with_200(self):
        p = _provider(lambda r: httpx.Response(200, json={"error": "model 'qwen2.5:7b' not found"}))
        with pytest.raises(ProviderError, match="not found"):
            await p.generate([{"role": "user", "content": "x"}], [])

    async def test_non_json_body(self):
        p = _provider(lambda r: httpx.Response(200, text="<html>502 Bad Gateway</html>"))
        with pytest.raises(ProviderError, match="non-JSON"):
            await p.generate([{"role": "user", "content": "x"}], [])


# ---------------------------------------------------------------- iteration cap UX
class TestTruncatedAnswer:
    """The loop's '<max iterations reached>' sentinel is for the harness, not the person."""

    async def test_user_gets_agent_findings_not_the_sentinel(self):
        class Echo(SubAgent):
            name, description, system_prompt, tool_names = "echo", "echoes", "s", []

            async def run(self, task, context, tracer, session_id, turn_id=None):
                return AgentResult(text="THE FINDING", messages=[])

        registry = ToolRegistry()
        orch = Orchestrator(
            MockProvider(default=tool_call_response("route_to_echo", {"task": "again"})),
            [Echo(MockProvider(), registry)], InMemoryTracer(), max_iterations=3,
        )
        res = await orch.handle("loop forever", "s1")
        assert res.truncated
        assert "<max iterations reached>" not in res.text
        assert "THE FINDING" in res.text and "step limit" in res.text

    async def test_truncated_with_no_agent_output_is_still_a_sentence(self):
        registry = ToolRegistry()
        registry.register(GetTimeTool())
        orch = Orchestrator(
            MockProvider(default=tool_call_response("get_time", {})), [], InMemoryTracer(),
            max_iterations=2, extra_tools=[GetTimeTool()],
        )
        res = await orch.handle("what time", "s1")
        assert res.truncated and "narrower" in res.text


# ---------------------------------------------------------------- pending-action memory
class _Gated(Tool):
    name, description, requires_confirmation = "gated", "gated", True
    input_schema = {"type": "object", "properties": {}}

    async def execute(self, arguments):
        return ToolResult.success("done")


async def test_resolved_pending_actions_are_pruned_but_pending_ones_never_are():
    reg = ToolRegistry()
    reg.register(_Gated())
    reg.grant("a", ["gated"])
    keep = [(await reg.dispatch("a", "gated", {}, "s")).confirmation_id for _ in range(3)]  # stay pending
    for _ in range(reg.MAX_RESOLVED_KEPT + 50):
        cid = (await reg.dispatch("a", "gated", {}, "s")).confirmation_id
        await reg.confirm(cid, approve=False)
    assert len(reg._pending) <= reg.MAX_RESOLVED_KEPT + len(keep) + 1
    assert {a.id for a in reg.pending()} == set(keep)


# ---------------------------------------------------------------- persistence failure
async def test_chat_returns_the_answer_even_if_persisting_the_turn_fails():
    from app.config import Settings
    from app.main import create_app

    shared = MockProvider(script=[text_response("the answer")])
    app = create_app(Settings(model_provider="mock", _env_file=None), provider_for=lambda r: shared, use_db=False)
    async with app.router.lifespan_context(app):
        async def boom(*a, **k):
            raise RuntimeError("database went away")

        app.state.jarvis.system.sessions.record_turn = boom
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
            r = await http.post("/chat", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["text"] == "the answer"
