import subprocess

import pytest

from app.providers.mock_provider import MockProvider, text_response
from app.tools.files import GenerateTestsTool, GitDiffSummarizeTool


def _git(cwd, *args):
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True, capture_output=True,
    )


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("def f():\n    return 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


async def test_git_diff_reports_stat_and_patch(repo):
    (repo / "app.py").write_text("def f():\n    return 2\n")
    out = await GitDiffSummarizeTool(repo).execute({})
    assert out.ok
    assert "app.py" in out.output["stat"]
    assert "-    return 1" in out.output["diff"] and "+    return 2" in out.output["diff"]


async def test_git_diff_no_changes(repo):
    out = await GitDiffSummarizeTool(repo).execute({})
    assert out.ok and out.output["note"] == "no changes"


@pytest.mark.parametrize("ref", ["--output=/tmp/pwned", "-p", "a b", "$(id)", "x;y"])
async def test_git_diff_rejects_option_injection_and_shell_metachars(repo, ref):
    out = await GitDiffSummarizeTool(repo).execute({"base": ref})
    assert not out.ok and "invalid git ref" in out.error


async def test_git_diff_path_escape_rejected(repo):
    out = await GitDiffSummarizeTool(repo).execute({"path": "../../etc"})
    assert not out.ok and "escapes" in out.error


async def test_git_diff_outside_repo_is_a_clean_error(tmp_path):
    out = await GitDiffSummarizeTool(tmp_path).execute({})
    assert not out.ok and "git diff failed" in out.error


async def test_generate_tests_returns_validated_code(tmp_path):
    (tmp_path / "m.py").write_text("def add(a, b):\n    return a + b\n")
    prov = MockProvider(script=[text_response("```python\nfrom m import add\n\ndef test_add():\n    assert add(1, 2) == 3\n```")])
    out = await GenerateTestsTool(tmp_path, prov).execute({"path": "m.py", "target": "add"})
    assert out.ok and "def test_add" in out.output["test_code"]
    prompt = prov.calls[0]["messages"][-1]["content"]
    assert "def add(a, b)" in prompt and "add" in prompt


async def test_generate_tests_retries_on_invalid_python_then_succeeds(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    prov = MockProvider(script=[text_response("```python\ndef broken(:\n```"), text_response("```python\ndef test_ok():\n    pass\n```")])
    out = await GenerateTestsTool(tmp_path, prov).execute({"path": "m.py"})
    assert out.ok
    assert "not valid Python" in prov.calls[1]["messages"][-1]["content"]


async def test_generate_tests_gives_up_cleanly(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    prov = MockProvider(default=text_response("no code here"))
    out = await GenerateTestsTool(tmp_path, prov).execute({"path": "m.py"})
    assert not out.ok and "could not produce valid tests" in out.error


async def test_generate_tests_rejects_escape(tmp_path):
    out = await GenerateTestsTool(tmp_path, MockProvider()).execute({"path": "../secret.py"})
    assert not out.ok
