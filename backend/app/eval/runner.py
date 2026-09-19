"""Two-layer eval runner.

Layer 1 - routing (deterministic, no judge): the real orchestrator + the real agent
  names/descriptions, but sub-agents are stubs, so ONLY the routing decision is measured.
Layer 2 - task: the full pipeline (real tools, registry, confirmation gate, RAG, memory) with
  deterministic checks (tool sequence, required/forbidden text, retrieval grounding, no
  unconfirmed side effects) plus an LLM judge (task success / grounding / tool correctness / safety).

Every run writes a scorecard and diffs it against the previous run of the same mode, flagging
category-level regressions and individual scenarios that flipped pass -> fail.
"""
from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.agents.base import SubAgent
from app.bootstrap import ROLES, System, build_system
from app.config import Settings
from app.eval.judge import Judge, JudgeThresholds, render_transcript
from app.eval.mock_script import compile_script
from app.eval.scenarios import EVAL_DIR, RoutingScenario, TaskScenario, load_routing_scenarios, load_task_scenarios
from app.eval.simulated import SimulatedOrchestratorModel
from app.harness.loop import AgentResult
from app.harness.memory import Fact
from app.observability.trace import InMemoryTracer
from app.orchestrator.orchestrator import Orchestrator, OrchestratorResult
from app.providers.base import ModelProvider
from app.providers.mock_provider import MockProvider
from app.rag.retriever import HybridRetriever
from app.tools.registry import Tool

EVAL_USER = "eval-user"
WORKSPACE = EVAL_DIR / "fixtures" / "workspace"
SystemHook = Callable[[System], None]


# ------------------------------------------------------------------------------- results
@dataclass
class ScenarioResult:
    scenario_id: str
    category: str
    layer: str  # "routing" | "task"
    passed: bool
    expected: str | None = None
    actual: str | None = None
    rationale: str = ""
    checks: dict = field(default_factory=dict)
    judge: dict | None = None
    error: str | None = None


@dataclass
class Scorecard:
    run_id: str
    ts: str
    config_hash: str
    mode: str
    scores_by_category: dict[str, dict]
    regressions: list[dict]
    previous_run_id: str | None
    details: list[dict]
    newly_failing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def pass_rate(self, category: str) -> float:
        return self.scores_by_category.get(category, {}).get("pass_rate", 0.0)


def score_by_category(results: list[ScenarioResult]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in results:
        c = out.setdefault(r.category, {"passed": 0, "total": 0})
        c["total"] += 1
        c["passed"] += int(r.passed)
    for c in out.values():
        c["pass_rate"] = c["passed"] / c["total"] if c["total"] else 0.0
    return out


def diff_scorecards(previous: dict | None, current: Scorecard, tolerance: float = 1e-9) -> tuple[list[dict], list[str]]:
    """Category pass-rate drops, plus scenarios that passed before and fail now."""
    if not previous:
        return [], []
    regressions = []
    for category, cur in current.scores_by_category.items():
        prev = previous["scores_by_category"].get(category)
        if prev and cur["pass_rate"] < prev["pass_rate"] - tolerance:
            regressions.append({"category": category, "previous": prev["pass_rate"], "current": cur["pass_rate"]})
    was_passing = {d["scenario_id"] for d in previous.get("details", []) if d.get("passed")}
    newly_failing = [d["scenario_id"] for d in current.details if not d["passed"] and d["scenario_id"] in was_passing]
    return regressions, newly_failing


@dataclass
class EvalThresholds:
    default: float = 0.9
    per_category: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "EvalThresholds":
        path = path or EVAL_DIR / "thresholds.yaml"
        data = yaml.safe_load(path.read_text()) if path.exists() else {}
        return cls(float((data or {}).get("default", 0.9)), {k: float(v) for k, v in ((data or {}).get("categories") or {}).items()})

    def for_category(self, category: str) -> float:
        return self.per_category.get(category, self.default)

    def failing(self, card: Scorecard) -> list[dict]:
        return [
            {"category": c, "pass_rate": s["pass_rate"], "threshold": self.for_category(c)}
            for c, s in card.scores_by_category.items()
            if s["pass_rate"] < self.for_category(c)
        ]


# ------------------------------------------------------------------------------- run storage
class EvalRunStore(ABC):
    @abstractmethod
    async def save(self, card: Scorecard) -> None: ...

    @abstractmethod
    async def latest(self, mode: str | None = None) -> dict | None: ...

    @abstractmethod
    async def list(self, limit: int = 20) -> list[dict]: ...

    @abstractmethod
    async def get(self, run_id: str) -> dict | None: ...


class InMemoryEvalStore(EvalRunStore):
    def __init__(self) -> None:
        self.runs: list[dict] = []

    async def save(self, card: Scorecard) -> None:
        self.runs.append(card.to_dict())

    async def latest(self, mode: str | None = None) -> dict | None:
        for run in reversed(self.runs):
            if mode is None or run["mode"] == mode:
                return run
        return None

    async def list(self, limit: int = 20) -> list[dict]:
        return list(reversed(self.runs))[:limit]

    async def get(self, run_id: str) -> dict | None:
        return next((r for r in self.runs if r["run_id"] == run_id), None)


class PostgresEvalStore(EvalRunStore):
    """`eval_runs` {run_id, config_hash, ts, scores_by_category} plus a `details` blob."""

    def __init__(self, db) -> None:
        self._db = db

    async def save(self, card: Scorecard) -> None:
        blob = {k: v for k, v in card.to_dict().items() if k not in {"run_id", "config_hash", "ts", "scores_by_category"}}
        await self._db.pool.execute(
            "INSERT INTO eval_runs (run_id, config_hash, ts, scores_by_category, details) VALUES ($1,$2,$3,$4::jsonb,$5::jsonb)",
            card.run_id, card.config_hash, datetime.fromisoformat(card.ts),
            json.dumps(card.scores_by_category), json.dumps(blob, default=str),
        )

    @staticmethod
    def _row(row) -> dict:
        load = lambda v: json.loads(v) if isinstance(v, str) else v  # noqa: E731
        return {
            "run_id": row["run_id"], "config_hash": row["config_hash"], "ts": row["ts"].isoformat(),
            "scores_by_category": load(row["scores_by_category"]), **load(row["details"]),
        }

    async def latest(self, mode: str | None = None) -> dict | None:
        rows = await self._db.pool.fetch(
            "SELECT * FROM eval_runs WHERE ($1::text IS NULL OR details->>'mode' = $1) ORDER BY ts DESC LIMIT 1", mode
        )
        return self._row(rows[0]) if rows else None

    async def list(self, limit: int = 20) -> list[dict]:
        rows = await self._db.pool.fetch("SELECT * FROM eval_runs ORDER BY ts DESC LIMIT $1", limit)
        return [self._row(r) for r in rows]

    async def get(self, run_id: str) -> dict | None:
        rows = await self._db.pool.fetch("SELECT * FROM eval_runs WHERE run_id = $1", run_id)
        return self._row(rows[0]) if rows else None


# ------------------------------------------------------------------------------- routing stubs
class StubAgent(SubAgent):
    """Same name/description/tools as a real agent (routing depends on those) but does no work."""

    def __init__(self, real: SubAgent):
        self.name, self.description = real.name, real.description
        self.system_prompt, self.tool_names = real.system_prompt, list(real.tool_names)
        self.max_iterations = 1

    async def run(self, task, context, tracer, session_id, turn_id=None) -> AgentResult:
        return AgentResult(text=f"[stub {self.name}] acknowledged", messages=[])


# ------------------------------------------------------------------------------- checks
def _tool_names(result: OrchestratorResult) -> list[str]:
    names = [c.name for c in result.loop.tool_calls if not c.name.startswith("route_to_")] if result.loop else []
    return names + [c.name for r in result.agent_results for c in r.tool_calls]


def _subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(any(n == h for h in it) for n in needle)


def run_checks(s: TaskScenario, result: OrchestratorResult, system: System, initial_events: int) -> dict[str, tuple[bool, str]]:
    checks: dict[str, tuple[bool, str]] = {}
    text = (result.text or "").lower()
    calls = _tool_names(result)
    checks["final_answer_present"] = (bool(result.text) and not result.truncated, f"text={result.text!r} truncated={result.truncated}")
    if s.expected_tool_calls:
        checks["expected_tool_calls"] = (_subsequence(s.expected_tool_calls, calls), f"expected {s.expected_tool_calls}, got {calls}")
    if s.forbidden_tool_calls:
        bad = [c for c in calls if c in s.forbidden_tool_calls]
        checks["forbidden_tool_calls"] = (not bad, f"forbidden calls made: {bad}" if bad else "none made")
    if s.expected_any_of:
        checks["expected_any_of"] = (any(n in calls for n in s.expected_any_of), f"wanted one of {s.expected_any_of}, got {calls}")
    for needle in s.must_include:
        alternatives = [a.strip().lower() for a in needle.split("|")]
        checks[f"includes:{needle}"] = (any(a in text for a in alternatives), f"answer={result.text!r}")
    for needle in s.must_not_include:
        checks[f"excludes:{needle}"] = (needle.lower() not in text, f"answer={result.text!r}")
    if s.grounding_source:
        retrieved = [
            m["content"] for r in result.agent_results for m in r.messages if m["role"] == "tool" and m["name"] == "rag_search"
        ]
        checks["retrieval_grounded"] = (any(s.grounding_source in c for c in retrieved), f"rag_search results {'did' if any(s.grounding_source in c for c in retrieved) else 'did not'} include {s.grounding_source}")
    if s.expect_pending_confirmation:
        checks["pending_confirmation"] = (bool(result.pending_confirmation_ids), f"pending={result.pending_confirmation_ids}")
    sent = getattr(system.mail, "sent", [])
    created = len(getattr(system.calendar, "events", [])) - initial_events
    checks["no_unconfirmed_side_effects"] = (not sent and created <= 0, f"emails_sent={len(sent)} events_created={created}")
    return checks


# ------------------------------------------------------------------------------- runner
class EvalRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        mode: str,  # "mock" (offline, deterministic) | "live" (real provider from settings)
        retriever: HybridRetriever | None = None,
        judge: Judge | None = None,
        judge_thresholds: JudgeThresholds | None = None,
        store: EvalRunStore | None = None,
        system_hook: SystemHook | None = None,  # mutate the system before running (config-change / regression injection)
        extra_code_tools: Callable[[], list[Tool]] | None = None,
        routing: list[RoutingScenario] | None = None,
        tasks: list[TaskScenario] | None = None,
    ):
        if mode not in ("mock", "live"):
            raise ValueError("mode must be 'mock' or 'live'")
        self.settings, self.mode, self.retriever = settings, mode, retriever
        self.judge, self.judge_thresholds = judge, judge_thresholds or JudgeThresholds()
        self.store, self.system_hook = store, system_hook
        self.extra_code_tools = extra_code_tools
        self.routing = routing if routing is not None else load_routing_scenarios()
        self.tasks = tasks if tasks is not None else load_task_scenarios()

    # -- system construction
    def _system(self, provider_for: Callable[[str], ModelProvider] | None) -> System:
        system = build_system(
            self.settings, tracer=InMemoryTracer(), provider_for=provider_for, retriever=self.retriever,
            workspace_root=WORKSPACE, extra_code_tools=self.extra_code_tools() if self.extra_code_tools else None,
        )
        if self.system_hook:
            self.system_hook(system)
        return system

    # -- layer 1
    async def run_routing(self, s: RoutingScenario) -> ScenarioResult:
        provider_for = None
        if self.mode == "mock":
            sim = SimulatedOrchestratorModel()
            provider_for = lambda role: sim if role == "orchestrator" else MockProvider()  # noqa: E731
        system = self._system(provider_for)
        orch = Orchestrator(
            system.providers["orchestrator"], [StubAgent(a) for a in system.agents], InMemoryTracer(), 4,
            system_prompt=system.orchestrator.system_prompt(),
        )
        try:
            res = await orch.handle(s.input, f"eval-route-{s.id}")
            actual = res.routed_agents
        except Exception as exc:  # noqa: BLE001 - a crashing model is a failed scenario, not a crashed run
            return ScenarioResult(s.id, s.category, "routing", False, str(s.expected_agent), None, error=f"{type(exc).__name__}: {exc}")
        passed = actual == s.expected_agent
        return ScenarioResult(
            s.id, s.category, "routing", passed, expected=" -> ".join(s.expected_agent) or "none",
            actual=" -> ".join(actual) or "none",
            rationale="" if passed else f"expected {s.expected_agent or 'no routing'}, orchestrator chose {actual or 'no routing'}",
        )

    # -- layer 2
    async def run_task(self, s: TaskScenario) -> ScenarioResult:
        s = s.for_mode(self.mode)
        provider_for = None
        if self.mode == "mock":
            if s.mock is None:
                return ScenarioResult(s.id, s.category, "task", False, error="no mock replay script for offline mode")
            shared = MockProvider(script=compile_script(s.mock))
            provider_for = lambda role: shared  # noqa: E731
        system = self._system(provider_for)
        initial_events = len(getattr(system.calendar, "events", []))
        for key, value in (s.setup.get("facts") or {}).items():
            await system.memory.upsert_long_term_fact(EVAL_USER, Fact(key=key, value=str(value), source="eval-setup"))
        session_id = f"eval-task-{s.id}-{uuid.uuid4().hex[:6]}"
        try:
            history = await system.sessions.build_history(session_id, EVAL_USER)
            res = await system.orchestrator.handle(s.input, session_id, history)
        except Exception as exc:  # noqa: BLE001
            return ScenarioResult(s.id, s.category, "task", False, error=f"{type(exc).__name__}: {exc}", rationale="pipeline raised")

        checks = run_checks(s, res, system, initial_events)
        passed = all(ok for ok, _ in checks.values())
        failed = [f"{name}: {detail}" for name, (ok, detail) in checks.items() if not ok]
        judge_dict, judge_note = None, ""
        if self.judge is not None:
            transcript = render_transcript(
                s.input, res.route_decisions,
                [("orchestrator", res.loop.messages if res.loop else [])]
                + [(d["agent"], r.messages) for d, r in zip(res.route_decisions, res.agent_results)],
                res.text,
            )
            scores = await self.judge.score(f"[{s.category}] {s.input}", s.rubric_notes, transcript)
            judge_dict = scores.as_dict()
            if not self.judge_thresholds.passes(scores, needs_grounding=bool(s.grounding_source)):
                passed = False
                judge_note = f"judge: {scores.error or scores.rationale}"
        rationale = "; ".join(failed + ([judge_note] if judge_note else []))
        return ScenarioResult(
            s.id, s.category, "task", passed,
            expected=", ".join(s.expected_tool_calls) or (s.rubric_notes[:120] or None), actual=(res.text or "")[:400],
            rationale=rationale, checks={k: {"ok": ok, "detail": d} for k, (ok, d) in checks.items()}, judge=judge_dict,
        )

    # -- whole suite
    async def run(self, layers: tuple[str, ...] = ("routing", "task")) -> Scorecard:
        results: list[ScenarioResult] = []
        if "routing" in layers:
            results += [await self.run_routing(s) for s in self.routing]
        if "task" in layers:
            results += [await self.run_task(s) for s in self.tasks]

        reference = self._system(None if self.mode == "live" else (lambda role: MockProvider()))
        card = Scorecard(
            run_id=uuid.uuid4().hex[:12], ts=datetime.now(timezone.utc).isoformat(), config_hash=reference.config_hash(),
            mode=self.mode, scores_by_category=score_by_category(results), regressions=[], previous_run_id=None,
            details=[asdict(r) for r in results],
        )
        if self.store is not None:
            previous = await self.store.latest(self.mode)
            card.regressions, card.newly_failing = diff_scorecards(previous, card)
            card.previous_run_id = previous["run_id"] if previous else None
            await self.store.save(card)
        return card
