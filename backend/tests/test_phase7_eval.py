"""Phase 7 acceptance: scorecard, regression diff, and a CI gate that catches injected regressions.
Everything runs offline against MockProvider / the simulated router."""
from pathlib import Path

import pytest

from app.bootstrap import build_retriever
from app.config import Settings
from app.eval.ci import DOCS, break_research_prompt, format_scorecard, main, run_offline
from app.eval.judge import Judge, JudgeScores, JudgeThresholds
from app.eval.runner import (
    EvalRunner, EvalThresholds, InMemoryEvalStore, PostgresEvalStore, diff_scorecards, score_by_category, ScenarioResult,
)
from app.eval.scenarios import CATEGORIES, load_routing_scenarios, load_task_scenarios
from app.eval.simulated import HeuristicJudgeProvider, SimulatedOrchestratorModel
from app.providers.mock_provider import MockProvider, text_response

SETTINGS = Settings(model_provider="mock", _env_file=None)


@pytest.fixture(scope="module")
async def retriever():
    return await build_retriever("offline", DOCS, settings=SETTINGS)


def runner(retriever, **kw):
    return EvalRunner(SETTINGS, mode="mock", retriever=retriever, judge=Judge(HeuristicJudgeProvider()), **kw)


def test_scenario_sets_are_substantial_and_cover_every_category():
    routing, tasks = load_routing_scenarios(), load_task_scenarios()
    assert 25 <= len(routing) + len(tasks) <= 60
    covered = {s.category for s in routing} | {s.category for s in tasks}
    assert covered == set(CATEGORIES)
    assert all(s.mock for s in tasks), "every task scenario needs an offline replay script"


async def test_baseline_suite_passes_thresholds_and_produces_a_scorecard(retriever):
    card = await runner(retriever).run()
    thresholds = EvalThresholds.load()
    assert thresholds.failing(card) == [], format_scorecard(card, thresholds)
    assert set(card.scores_by_category) == set(CATEGORIES)
    assert card.config_hash and card.mode == "mock"
    layers = {d["layer"] for d in card.details}
    assert layers == {"routing", "task"}
    task = next(d for d in card.details if d["scenario_id"] == "rag-retry-max-attempts")
    assert task["passed"] and task["checks"]["retrieval_grounded"]["ok"] and task["judge"]["task_success"] == 1.0


async def test_breaking_a_prompt_is_flagged_as_a_regression_by_the_diff(retriever):
    store = InMemoryEvalStore()
    baseline = await runner(retriever, store=store).run()
    assert baseline.regressions == [] and baseline.previous_run_id is None

    broken = await runner(retriever, store=store, system_hook=break_research_prompt).run()

    assert broken.previous_run_id == baseline.run_id
    assert broken.config_hash != baseline.config_hash  # the change is visible in the config fingerprint
    regressed = {r["category"]: r for r in broken.regressions}
    assert "routing" in regressed and regressed["routing"]["current"] < regressed["routing"]["previous"]
    assert any(s.startswith("route-research-") for s in broken.newly_failing)
    assert (await store.latest("mock"))["run_id"] == broken.run_id


async def test_removing_the_confirmation_gate_is_caught_by_task_eval(retriever):
    """A harness regression (not a prompt one): irreversible tools run without approval."""

    def disable_gate(system):
        system.registry._tools["email_send"].requires_confirmation = False

    card = await runner(retriever, system_hook=disable_gate).run(layers=("task",))
    failed = {d["scenario_id"] for d in card.details if not d["passed"]}
    assert "edge-email-send-needs-confirmation" in failed
    detail = next(d for d in card.details if d["scenario_id"] == "edge-email-send-needs-confirmation")
    assert detail["checks"]["no_unconfirmed_side_effects"]["ok"] is False
    assert card.scores_by_category["refusal/edge-case"]["pass_rate"] < 1.0


async def test_breaking_retrieval_is_caught_by_grounding_checks():
    empty = await build_retriever("offline", "/nonexistent-docs", settings=SETTINGS)  # nothing ingested
    card = await EvalRunner(SETTINGS, mode="mock", retriever=empty).run(layers=("task",))
    assert card.scores_by_category["RAG-grounding"]["pass_rate"] < 1.0
    grounded = next(d for d in card.details if d["scenario_id"] == "rag-retry-max-attempts")
    assert grounded["checks"]["retrieval_grounded"]["ok"] is False


def test_ci_gate_passes_clean_and_fails_on_injected_regression(capsys):
    assert main([]) == 0
    assert "EVAL GATE PASSED" in capsys.readouterr().out
    assert main(["--inject-regression"]) == 1
    out = capsys.readouterr().out
    assert "EVAL GATE FAILED" in out and "FAIL  routing" in out


def test_routing_depends_on_agent_descriptions():
    tools = [
        {"name": "route_to_a", "description": "Delegate. Handles kubernetes cluster deployments."},
        {"name": "route_to_b", "description": "Delegate. Handles invoices and billing."},
    ]
    sim = SimulatedOrchestratorModel()
    assert sim.plan("please fix the kubernetes cluster deployments", tools) == ["route_to_a"]
    assert sim.plan("please fix the billing invoices", tools) == ["route_to_b"]
    assert sim.plan("hello there", tools) == []


def test_diff_scorecards_flags_drops_and_flips_only():
    prev = {"scores_by_category": {"x": {"pass_rate": 1.0}, "y": {"pass_rate": 0.5}},
            "details": [{"scenario_id": "a", "passed": True}, {"scenario_id": "b", "passed": False}]}

    class Card:
        scores_by_category = {"x": {"pass_rate": 0.5}, "y": {"pass_rate": 0.75}}
        details = [{"scenario_id": "a", "passed": False}, {"scenario_id": "b", "passed": False}]

    regs, flips = diff_scorecards(prev, Card())
    assert regs == [{"category": "x", "previous": 1.0, "current": 0.5}]
    assert flips == ["a"]  # b was already failing: not a regression
    assert diff_scorecards(None, Card()) == ([], [])


def test_score_by_category():
    rs = [ScenarioResult("1", "c", "task", True), ScenarioResult("2", "c", "task", False), ScenarioResult("3", "d", "task", True)]
    assert score_by_category(rs) == {"c": {"passed": 1, "total": 2, "pass_rate": 0.5}, "d": {"passed": 1, "total": 1, "pass_rate": 1.0}}


# ---- judge
async def test_judge_parses_scores_and_clamps():
    judge = Judge(MockProvider(script=[text_response('```json\n{"task_success": 1.4, "grounding_fidelity": 0.5, "tool_correctness": 1, "safety": 1, "rationale": "fine",}\n```')]))
    s = await judge.score("scenario", "rubric", "transcript")
    assert (s.task_success, s.grounding_fidelity) == (1.0, 0.5) and s.rationale == "fine" and s.error is None


async def test_judge_retries_on_garbage_then_recovers():
    prov = MockProvider(script=[text_response("I think it went well!"), text_response('{"task_success": 0, "grounding_fidelity": 0, "tool_correctness": 0, "safety": 1, "rationale": "bad"}')])
    s = await Judge(prov).score("s", "r", "t")
    assert s.error is None and s.task_success == 0.0
    assert "Invalid" in prov.calls[1]["messages"][-1]["content"]


async def test_judge_gives_up_with_error_and_fails_the_scenario():
    s = await Judge(MockProvider(default=text_response("nope"))).score("s", "r", "t")
    assert s.error and not JudgeThresholds().passes(s, needs_grounding=False)


def test_judge_thresholds():
    ok = JudgeScores(1, 1, 1, 1)
    assert JudgeThresholds().passes(ok, needs_grounding=True)
    assert not JudgeThresholds().passes(JudgeScores(1, 1, 1, 0.5), needs_grounding=False)  # unsafe
    assert not JudgeThresholds().passes(JudgeScores(1, 0.2, 1, 1), needs_grounding=True)  # ungrounded
    assert JudgeThresholds().passes(JudgeScores(1, 0.2, 1, 1), needs_grounding=False)  # grounding not required


async def test_judge_is_a_separate_provider_instance_from_agents(retriever):
    """The judge never shares a provider with the system under test."""
    r = runner(retriever)
    system = r._system(None)
    assert all(p is not r.judge.provider for p in system.providers.values())


# ---- persistence
async def test_postgres_run_store_roundtrip(db, retriever):
    store = PostgresEvalStore(db)
    card = await runner(retriever, store=store).run(layers=("routing",))
    got = await store.get(card.run_id)
    assert got["scores_by_category"]["routing"]["pass_rate"] == card.scores_by_category["routing"]["pass_rate"]
    assert got["mode"] == "mock" and got["details"][0]["scenario_id"]
    assert (await store.latest("mock"))["run_id"] == card.run_id
    assert card.run_id in [r["run_id"] for r in await store.list(50)]
    await db.pool.execute("DELETE FROM eval_runs WHERE run_id = $1", card.run_id)


# ---- scenario format features added after the first live run
def test_live_overrides_apply_only_in_live_mode():
    from app.eval.scenarios import TaskScenario

    s = TaskScenario("x", "in", "c", expected_tool_calls=["email_send"], expect_pending_confirmation=True,
                     live={"expected_tool_calls": [], "expected_any_of": ["email_send", "email_draft"], "expect_pending_confirmation": False})
    assert s.for_mode("mock") is s
    live = s.for_mode("live")
    assert live.expected_tool_calls == [] and live.expected_any_of == ["email_send", "email_draft"] and not live.expect_pending_confirmation
    assert s.expected_tool_calls == ["email_send"]  # original untouched


async def test_alternate_phrasings_and_any_of_checks(retriever):
    from app.eval.runner import run_checks
    from app.eval.scenarios import TaskScenario
    from app.bootstrap import build_system
    from app.observability.trace import InMemoryTracer
    from app.orchestrator.orchestrator import OrchestratorResult

    system = build_system(SETTINGS, tracer=InMemoryTracer(), provider_for=lambda r: MockProvider())
    res = OrchestratorResult(text="Sorry, I couldn't find that in the docs.", turn_id="t")
    s = TaskScenario("x", "in", "c", must_include=["could not find|couldn't find"], expected_any_of=["email_send", "email_draft"])
    checks = run_checks(s, res, system, initial_events=len(system.calendar.events))
    assert checks["includes:could not find|couldn't find"][0] is True
    assert checks["expected_any_of"][0] is False  # neither tool was called


def test_every_live_override_targets_a_real_field():
    from dataclasses import fields
    from app.eval.scenarios import TaskScenario

    names = {f.name for f in fields(TaskScenario)}
    for s in load_task_scenarios():
        assert set(s.live) <= names, s.id
