"""Phase 3 acceptance against a REAL local model. Skipped automatically when Ollama or the
model is unavailable, so the suite stays green with no live-model dependency."""
import httpx
import pytest

from app.agents.code_agent import CodeAgent
from app.config import get_settings
from app.observability.trace import InMemoryTracer
from app.orchestrator.orchestrator import Orchestrator
from app.providers.ollama_provider import OllamaProvider
from app.tools.files import ReadFileTool
from app.tools.registry import ToolRegistry

pytestmark = pytest.mark.live


def _require_model(model: str) -> str:
    host = get_settings().ollama_host
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=2).json()
    except Exception:  # noqa: BLE001 - ollama unreachable for any reason -> skip
        pytest.skip("ollama not reachable")
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names:
        pytest.skip(f"model {model} not pulled")
    return host


@pytest.mark.parametrize("native", [True, False], ids=["native-tools", "prompted-tools"])
async def test_orchestrator_to_code_agent_with_real_model(tmp_path, native):
    model = get_settings().orchestrator_model
    host = _require_model(model)
    (tmp_path / "calc.py").write_text("def triple(n):\n    return n * 3\n")

    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    code = CodeAgent(OllamaProvider(host, model, native_tools=native), registry)
    tracer = InMemoryTracer()
    orch = Orchestrator(OllamaProvider(host, model, native_tools=native), [code], tracer)

    result = await orch.handle("What does the function in calc.py do? Read the file to be sure.", "live-s1")

    assert result.routed_agents[:1] == ["code"], tracer.events
    assert any(c.name == "read_file" and c.ok for r in result.agent_results for c in r.tool_calls)
    assert result.text and ("3" in result.text or "three" in result.text.lower() or "triple" in result.text.lower())


async def test_memory_summarizer_folds_older_turns_with_the_real_model():
    """Phase 6 live: the summarizer is a real model call; on overflow it must compress the older
    turns into ONE block that keeps the concrete facts, and leave the newest turn verbatim."""
    from app.harness.memory import InMemoryMemoryStore
    from app.observability.trace import InMemoryTracer

    model = get_settings().summarizer_model
    host = _require_model(model)
    tracer = InMemoryTracer()
    mem = InMemoryMemoryStore(OllamaProvider(host, model), tracer)
    sid = "live-mem"
    await mem.ensure_session(sid, "me")
    for role, content in [
        ("user", "We decided the payments retry policy is 5 attempts with a 30 second deadline."),
        ("assistant", "Noted: 5 attempts, 30 s overall deadline."),
        ("user", "On-call hands over Mondays at 10:00 UTC."),
        ("assistant", "Understood - Monday 10:00 UTC handover."),
        ("user", "What did we decide about retries?"),
    ]:
        await mem.append(sid, {"role": role, "content": content})

    # ~65 tokens of context by the chars/4 heuristic; a 20-token budget forces a real summarization
    await mem.summarize_if_needed(sid, token_budget=20)

    ctx = await mem.get_context(sid)
    assert len(ctx) < 5, ctx
    assert ctx[-1]["content"] == "What did we decide about retries?"  # newest turn verbatim
    summary = ctx[0]["content"]
    assert "5" in summary and "30" in summary, summary  # the numbers survived compression
    assert len(tracer.of_type("memory_summarize")) == 1
