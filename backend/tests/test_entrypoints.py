"""The provider factory and the two CLI entry points (`make ingest`, `make eval-live`).

These are the paths a demo actually runs, and they were the least covered code in the app: a typo
in an argparse wiring or a factory branch only shows up when someone is watching.
"""
import pytest

from app.config import Settings
from app.providers.factory import get_provider
from app.providers.mock_provider import MockProvider
from app.providers.ollama_provider import OllamaProvider


class TestProviderFactory:
    """The one place allowed to branch on provider type."""

    def test_mock(self):
        assert isinstance(get_provider("mock", settings=Settings(_env_file=None)), MockProvider)

    def test_ollama_uses_configured_host_and_the_requested_model(self):
        settings = Settings(_env_file=None, ollama_host="http://example:1234", orchestrator_model="llama3.1:8b")
        p = get_provider("ollama", settings=settings)
        assert isinstance(p, OllamaProvider)
        assert p.host == "http://example:1234" and p.model == "llama3.1:8b"
        assert get_provider("ollama", model="qwen2.5:7b", settings=settings).model == "qwen2.5:7b"

    def test_name_defaults_to_the_configured_provider(self):
        assert isinstance(get_provider(settings=Settings(_env_file=None, model_provider="mock")), MockProvider)

    def test_unknown_provider_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown model provider"):
            get_provider("gpt-9000", settings=Settings(_env_file=None))


class TestIngestCli:
    """`make ingest` - the command that fills the Research agent's index."""

    def test_ingests_a_directory_then_is_idempotent(self, tmp_path, monkeypatch, capsys):
        # sync: main() calls asyncio.run(), which cannot nest inside pytest-asyncio's loop
        import asyncio

        import asyncpg

        from app.rag import ingest as ingest_mod

        (tmp_path / "note.md").write_text("# Runbook\n\nThe queue alert fires above 12,000 messages.\n")
        (tmp_path / "skip.bin").write_bytes(b"\x00\x01")  # unsupported extension
        monkeypatch.setattr(ingest_mod, "get_settings", lambda: Settings(_env_file=None, database_url=_url()), raising=False)

        table = "rag_cli_smoke"
        monkeypatch.setattr(ingest_mod, "create_vector_store", _store_factory(table))

        try:
            assert ingest_mod.main([str(tmp_path), "--embedder", "hashing", "--store", "array"]) == 0
        except (OSError, asyncpg.PostgresError) as exc:
            pytest.skip(f"postgres not available: {exc}")
        assert "scanned=1" in capsys.readouterr().out  # the .bin file is not a supported document

        assert ingest_mod.main([str(tmp_path), "--embedder", "hashing", "--store", "array"]) == 0
        assert "skipped_unchanged=1" in capsys.readouterr().out  # content hash short-circuits re-embedding

        async def drop():
            conn = await asyncpg.connect(_url())
            await conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            await conn.close()

        asyncio.run(drop())

    def test_rejects_a_missing_path(self, monkeypatch):
        from app.rag import ingest as ingest_mod

        monkeypatch.setattr(ingest_mod, "get_settings", lambda: Settings(_env_file=None, database_url=_url()), raising=False)
        with pytest.raises(FileNotFoundError):
            ingest_mod.main(["/nonexistent-directory-xyz", "--embedder", "hashing"])


class TestEvalCli:
    """`make eval` / `make eval-live` - the gate itself."""

    def test_live_requires_a_real_provider(self, monkeypatch):
        from app.eval import ci

        monkeypatch.setattr(ci, "get_settings", lambda: Settings(_env_file=None, model_provider="mock"))
        with pytest.raises(SystemExit, match="MODEL_PROVIDER"):
            ci.main(["--live"])

    def test_live_and_inject_regression_are_mutually_exclusive(self):
        from app.eval import ci

        with pytest.raises(SystemExit):  # argparse error
            ci.main(["--live", "--inject-regression"])

    def test_offline_gate_reports_regressions_it_finds(self, capsys):
        from app.eval.ci import main

        assert main(["--inject-regression"]) == 1
        out = capsys.readouterr().out
        assert "EVAL GATE FAILED" in out and "routing" in out


def _url() -> str:
    from tests.conftest import database_url_for_tests

    return database_url_for_tests()


def _store_factory(table: str):
    from app.rag.store import create_vector_store

    async def factory(pool, dim, embedder_name, *, prefer="auto", **kw):
        return await create_vector_store(pool, dim, embedder_name, table=table, prefer=prefer)

    return factory
