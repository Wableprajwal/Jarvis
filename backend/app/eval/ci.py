"""`make eval`: run the whole two-layer suite and gate on per-category thresholds.

Exit code 0 = every category at/above its threshold; 1 = gate failed.

Default (`make eval`) is the CI mode: offline against MockProvider + the simulated router, so it
needs no network and no live model, yet still fails when a prompt, tool, allow-list, confirmation
gate or retrieval path regresses. `--inject-regression` deliberately breaks the Research agent's
description to demonstrate the gate tripping.

`--live` (`make eval-live`) runs the same scenarios against the configured real provider and a real
judge. Slower and non-deterministic, but it is the number that actually reflects model quality; it
persists the scorecard so the next run diffs against it.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.bootstrap import System, build_retriever, code_exec_tools, default_provider_for
from app.config import Settings, get_settings
from app.db import Database
from app.eval.judge import Judge
from app.eval.runner import EvalRunner, EvalThresholds, PostgresEvalStore, Scorecard
from app.eval.simulated import HeuristicJudgeProvider

DOCS = Path(__file__).resolve().parents[3] / "docs"


def break_research_prompt(system: System) -> None:
    for agent in system.agents:
        if agent.name == "research":
            agent.description = "Handles frobnication of quux widgets."


def format_scorecard(card: Scorecard, thresholds: EvalThresholds) -> str:
    lines = [f"eval run {card.run_id}  config {card.config_hash}  mode={card.mode}", ""]
    for category, s in sorted(card.scores_by_category.items()):
        ok = s["pass_rate"] >= thresholds.for_category(category)
        lines.append(f"  {'PASS' if ok else 'FAIL'}  {category:<20} {s['passed']:>2}/{s['total']:<2}  {s['pass_rate']:.0%}  (min {thresholds.for_category(category):.0%})")
    failed = [d for d in card.details if not d["passed"]]
    if failed:
        lines += ["", "failing scenarios:"] + [f"  - {d['scenario_id']}: {d.get('rationale') or d.get('error')}" for d in failed]
    return "\n".join(lines)


async def run_offline(inject_regression: bool = False) -> tuple[Scorecard, EvalThresholds]:
    settings = Settings(model_provider="mock", _env_file=None)
    retriever = await build_retriever("offline", DOCS, settings=settings)
    runner = EvalRunner(
        settings, mode="mock", retriever=retriever, judge=Judge(HeuristicJudgeProvider()),
        system_hook=break_research_prompt if inject_regression else None,
        extra_code_tools=lambda: code_exec_tools(settings),
    )
    return await runner.run(), EvalThresholds.load()


async def run_live() -> tuple[Scorecard, EvalThresholds]:
    """Against the configured provider (Ollama) with a real judge. The judge gets its own provider
    instance - and its own `judge_model` - so it never shares state with the agents under test."""
    settings = get_settings()
    if settings.model_provider == "mock":
        raise SystemExit("--live needs MODEL_PROVIDER=ollama (set it in .env)")
    provider_for = default_provider_for(settings)
    db = store = None
    try:
        db = await Database.connect(settings.database_url)
        store = PostgresEvalStore(db)  # persist so the NEXT run diffs against this one
    except Exception as exc:  # noqa: BLE001 - a missing database only costs persistence, not the run
        print(f"(eval runs will not be persisted: {exc})")
    try:
        retriever = await build_retriever("postgres" if db else "memory", DOCS, db=db, settings=settings)
        runner = EvalRunner(
            settings, mode="live", retriever=retriever, judge=Judge(provider_for("judge")), store=store,
            extra_code_tools=lambda: code_exec_tools(settings),
        )
        return await runner.run(), EvalThresholds.load()
    finally:
        if db:
            await db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Jarvis eval gate")
    parser.add_argument("--live", action="store_true", help="run against the configured real model instead of MockProvider")
    parser.add_argument("--inject-regression", action="store_true", help="break the research prompt to prove the gate trips")
    args = parser.parse_args(argv)
    if args.live and args.inject_regression:
        parser.error("--inject-regression is for the deterministic offline gate")
    card, thresholds = asyncio.run(run_live() if args.live else run_offline(args.inject_regression))
    print(format_scorecard(card, thresholds))
    if card.regressions:
        print("\nREGRESSIONS vs " + (card.previous_run_id or "?") + ":")
        for r in card.regressions:
            print(f"  {r['category']}: {r['previous']:.0%} -> {r['current']:.0%}")
    failing = thresholds.failing(card)
    if failing:
        print(f"\nEVAL GATE FAILED: {[f['category'] for f in failing]}")
        return 1
    print("\nEVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
