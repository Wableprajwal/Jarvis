"""Scenario sets (YAML) for the two eval layers.

routing_scenarios/*.yaml : [{id, input, expected_agent, category}]
    expected_agent: an agent name, a list (ordered chain), or "none" (orchestrator answers directly)
task_scenarios/*.yaml    : [{id, input, category, expected_tool_calls?, expected_any_of?, forbidden_tool_calls?,
                             grounding_source?, rubric_notes, must_include?, must_not_include?,
                             expect_pending_confirmation?, setup?, mock?, live?}]
    `mock` is a scripted replay (see mock_script.py) used when the suite runs against MockProvider:
    it makes the whole pipeline (registry, allow-lists, confirmation gate, tools, RAG, memory) run
    deterministically offline, so a harness/tool regression fails CI even with no model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

EVAL_DIR = Path(__file__).parent
CATEGORIES = ("routing", "tool-selection", "RAG-grounding", "memory-recall", "refusal/edge-case")


@dataclass
class RoutingScenario:
    id: str
    input: str
    expected_agent: list[str]  # [] means: no routing, answer directly
    category: str = "routing"


@dataclass
class TaskScenario:
    id: str
    input: str
    category: str
    rubric_notes: str = ""
    expected_tool_calls: list[str] = field(default_factory=list)  # tool names that must appear, in order
    forbidden_tool_calls: list[str] = field(default_factory=list)
    grounding_source: str | None = None  # a source the answer must be grounded in / cite
    expected_any_of: list[str] = field(default_factory=list)  # at least one of these tools must be called
    must_include: list[str] = field(default_factory=list)  # case-insensitive substrings; "a|b" = either phrasing
    must_not_include: list[str] = field(default_factory=list)
    expect_pending_confirmation: bool = False
    setup: dict = field(default_factory=dict)  # e.g. {"facts": {"preferred_language": "Rust"}}
    mock: list[dict] | None = None
    live: dict = field(default_factory=dict)  # field overrides applied only in live mode (see for_mode)

    def for_mode(self, mode: str) -> "TaskScenario":
        """Replay scripts pin exact tool sequences; a real model may legitimately take another safe
        path (e.g. draft instead of send). `live:` overrides let a scenario state the *behavioural*
        requirement for real models while the strict form still guards the harness in replay."""
        return replace(self, **self.live) if mode == "live" and self.live else self


def _load_yaml_dir(directory: Path) -> list[dict]:
    items: list[dict] = []
    for path in sorted(directory.glob("*.y*ml")):
        data = yaml.safe_load(path.read_text()) or []
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of scenarios")
        items.extend(data)
    return items


def _agents(value) -> list[str]:
    if value in (None, "none", "None", []):
        return []
    return [value] if isinstance(value, str) else list(value)


def load_routing_scenarios(directory: Path | None = None) -> list[RoutingScenario]:
    out = []
    for d in _load_yaml_dir(directory or EVAL_DIR / "routing_scenarios"):
        out.append(RoutingScenario(d["id"], d["input"], _agents(d.get("expected_agent")), d.get("category", "routing")))
    _assert_unique_ids(out)
    return out


def load_task_scenarios(directory: Path | None = None) -> list[TaskScenario]:
    out = []
    for d in _load_yaml_dir(directory or EVAL_DIR / "task_scenarios"):
        out.append(
            TaskScenario(
                id=d["id"], input=d["input"], category=d["category"],
                rubric_notes=d.get("rubric_notes", ""),
                expected_tool_calls=d.get("expected_tool_calls", []) or [],
                forbidden_tool_calls=d.get("forbidden_tool_calls", []) or [],
                expected_any_of=d.get("expected_any_of", []) or [],
                grounding_source=d.get("grounding_source"),
                must_include=d.get("must_include", []) or [],
                must_not_include=d.get("must_not_include", []) or [],
                expect_pending_confirmation=bool(d.get("expect_pending_confirmation", False)),
                setup=d.get("setup", {}) or {},
                mock=d.get("mock"),
                live=d.get("live", {}) or {},
            )
        )
    _assert_unique_ids(out)
    return out


def _assert_unique_ids(items) -> None:
    ids = [i.id for i in items]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate scenario ids: {sorted(dup)}")
