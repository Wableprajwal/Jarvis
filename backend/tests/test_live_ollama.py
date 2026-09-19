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
