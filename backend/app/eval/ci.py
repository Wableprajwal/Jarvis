"""`make eval`: run the whole two-layer suite offline against MockProvider and gate on thresholds.

Exit code 0 = every category at/above its threshold; 1 = gate failed. No network, no live model.
`--inject-regression` deliberately breaks the Research agent's description to demonstrate that the
gate catches a prompt/config regression.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.bootstrap import System, build_retriever
from app.config import Settings
from app.eval.judge import Judge
from app.eval.runner import EvalRunner, EvalThresholds, Scorecard
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
    )
    return await runner.run(), EvalThresholds.load()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline eval gate")
    parser.add_argument("--inject-regression", action="store_true", help="break the research prompt to prove the gate trips")
    args = parser.parse_args(argv)
    card, thresholds = asyncio.run(run_offline(args.inject_regression))
    print(format_scorecard(card, thresholds))
    failing = thresholds.failing(card)
    if failing:
        print(f"\nEVAL GATE FAILED: {[f['category'] for f in failing]}")
        return 1
    print("\nEVAL GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
