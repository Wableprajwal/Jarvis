"""Dependency declarations must agree with each other and with the Python range they claim.

Why this exists: `requirements.txt` pinned `numpy==2.5.3`, which needs Python >= 3.12, while
`pyproject.toml` declared `requires-python >= 3.11`. On 3.11 the install failed with an opaque
resolver error - even though the code runs fine there (the full suite passes on 3.11). Nothing
compared the two files, so the contradiction sat there. These tests do.
"""
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

BACKEND = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((BACKEND / "pyproject.toml").read_text())["project"]
# Interpreter minors the project may claim. 3.11 and 3.12 are exercised by the full suite.
CANDIDATE_MINORS = ["3.11", "3.12", "3.13"]


def _requirements_txt(name: str) -> list[Requirement]:
    lines = (BACKEND / name).read_text().splitlines()
    return [Requirement(line.strip()) for line in lines if line.strip() and not line.lstrip().startswith(("#", "-r"))]


def _key(r: Requirement) -> tuple[str, str, str]:
    """Identity ignoring extras: `uvicorn[standard]==0.53.0` and `uvicorn==0.53.0` are the same pin."""
    return (r.name.lower(), str(r.specifier), str(r.marker or ""))


RUNTIME = _requirements_txt("requirements.txt")
DECLARED = [Requirement(d) for d in PYPROJECT["dependencies"] + PYPROJECT["optional-dependencies"]["rag"]]
CLAIMED_MINORS = [m for m in CANDIDATE_MINORS if m in SpecifierSet(PYPROJECT["requires-python"])]


def test_every_dependency_is_an_exact_pin():
    for r in RUNTIME + _requirements_txt("requirements-dev.txt") + DECLARED:
        specs = list(r.specifier)
        assert len(specs) == 1 and specs[0].operator == "==", f"{r} is not an exact pin"


def test_pyproject_and_requirements_txt_declare_the_same_pins():
    assert {_key(r) for r in RUNTIME} == {_key(r) for r in DECLARED}


@pytest.mark.parametrize("minor", CLAIMED_MINORS)
def test_every_claimed_python_gets_exactly_one_pin_per_package(minor):
    """The bug: a package pinned to a version that cannot install on a Python the project claims.
    Markers must leave each package with exactly one applicable pin on every claimed interpreter."""
    env = {"python_version": minor, "python_full_version": f"{minor}.0"}
    applicable: dict[str, list[str]] = {}
    for r in RUNTIME:
        if r.marker is None or r.marker.evaluate(env):
            applicable.setdefault(r.name.lower(), []).append(str(r))
    for name in {r.name.lower() for r in RUNTIME}:
        assert len(applicable.get(name, [])) == 1, f"python {minor}: {name} has pins {applicable.get(name)}"


def test_numpy_pins_respect_numpys_own_python_floor():
    """numpy 2.5 dropped Python 3.11. The exact pin that is unsatisfiable on 3.11 must be marker-guarded."""
    for r in RUNTIME:
        if r.name.lower() == "numpy" and r.specifier.__str__().startswith("==2.5"):
            assert r.marker is not None and not r.marker.evaluate({"python_version": "3.11"}), (
                "numpy>=2.5 needs Python>=3.12; it must not apply on 3.11"
            )
