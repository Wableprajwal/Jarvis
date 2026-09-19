"""One workspace-path policy for every tool that touches files.

`read_file`, `generate_tests`, `git_diff_summarize --path` and the sandbox's file staging all take
model-chosen paths. They must agree on two things: a path stays inside the workspace root, and a
handful of names are never exposed to a model no matter where they sit inside it - a workspace
root of `.` puts the project's own `.env` one `read_file` call away otherwise.
"""
from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath

# Never handed to a model, even inside the workspace.
SENSITIVE_NAMES = (
    ".env", ".env.*", "*.env", ".netrc", ".npmrc", ".pypirc", ".git-credentials", "id_rsa*", "id_dsa*",
    "id_ecdsa*", "id_ed25519*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.kdbx", "credentials",
    "credentials.*", "secrets", "secrets.*", "*.secret", "*.secrets", "token", "token.*", "*.token",
)
SENSITIVE_DIRS = frozenset({".git", ".ssh", ".aws", ".gnupg", ".kube", ".docker", ".config", ".venv", "venv", "node_modules"})


class PathRefused(ValueError):
    """The path is outside the workspace, malformed, or names something sensitive."""


def is_sensitive(rel: PurePosixPath) -> bool:
    parts = [p.lower() for p in rel.parts]
    if any(p in SENSITIVE_DIRS for p in parts[:-1]):
        return True
    return bool(parts) and any(fnmatch.fnmatch(parts[-1], pat) for pat in SENSITIVE_NAMES)


def resolve_inside(root: Path, raw: str, *, must_exist: bool = True) -> Path:
    """Resolve a model-supplied workspace-relative path, or raise PathRefused.

    Checks both the path as written and the path it resolves to, so a symlink inside the
    workspace cannot point a permitted-looking name at `.env`.
    """
    if not isinstance(raw, str) or not raw or "\x00" in raw or any(ord(c) < 32 for c in raw):
        raise PathRefused(f"invalid path: {raw!r}")
    rel = PurePosixPath(raw)
    if rel.is_absolute() or ".." in rel.parts or str(rel) in (".", ""):
        raise PathRefused(f"path escapes the workspace root (must be relative and stay inside it): {raw!r}")
    root = root.resolve()
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise PathRefused(f"path escapes the workspace root: {raw!r}")
    if is_sensitive(rel) or is_sensitive(PurePosixPath(*target.relative_to(root).parts)):
        raise PathRefused(f"refusing to expose a sensitive file: {raw!r}")
    if must_exist and not target.is_file():
        raise PathRefused(f"file not found: {raw!r}")
    return target
