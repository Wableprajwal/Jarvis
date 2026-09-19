"""Credential isolation.

Secrets live in exactly one place - `CredentialStore` - and reach exactly one kind of consumer: a
tool instance, at construction, as a closure (`store.token_provider("gmail_token")`). The model never
sees a secret because no code path hands one to it, and this module adds defence in depth for the
case where a tool *accidentally* echoes one (an error string that embeds an auth header, an API that
reflects a token in its body, ...):

    Redactor          scrubs registered secret values (+ base64/hex/url-encoded/JSON-escaped forms,
                      + long prefixes) and well-known token shapes from arbitrary nested data
    ToolRegistry      `ToolRegistry(redactor=...)` scrubs every ToolResult before the harness sees it
    RedactingProvider scrubs the messages *before* they reach the model, and the model's reply
    RedactingTracer   scrubs every TraceEvent before it is stored / shown in the UI

Three independent layers, so a bug in one (a new code path that bypasses the registry, say) does not
leak. `redact()` never mutates its input and never raises on odd objects.
"""
from __future__ import annotations

import base64
import dataclasses
import html
import json
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from urllib.parse import quote, quote_plus

from app.observability.trace import TraceEvent, Tracer
from app.providers.base import ModelProvider, ModelResponse

REDACTED = "[REDACTED]"

# Setting names treated as secrets by CredentialStore.from_settings
_SECRET_FIELD = re.compile(r"(token|secret|password|passwd|api_?key|private_?key|credential)", re.I)


class CredentialMissing(KeyError):
    pass


class CredentialStore:
    """Holds secrets; hands them out only as injectable providers. There is deliberately no public
    `get(name) -> str`. `repr()` shows names, never values, and the store cannot be pickled or
    JSON-encoded by accident."""

    def __init__(self, secrets: Mapping[str, str | None] | None = None, *, min_length: int = 6):
        self._secrets: dict[str, str] = {}
        self._min_length = min_length
        for name, value in (secrets or {}).items():
            self.register(name, value)

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Any = None) -> CredentialStore:
        """Every non-empty settings field whose name looks like a secret (gmail_token, ...)."""
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        fields = type(settings).model_fields if hasattr(type(settings), "model_fields") else vars(settings)
        secrets = {
            name: getattr(settings, name)
            for name in fields
            if _SECRET_FIELD.search(name) and isinstance(getattr(settings, name, None), str)
        }
        return cls(secrets)

    def register(self, name: str, value: str | None) -> None:
        if value is None or value == "":
            self._secrets.pop(name, None)
            return
        if not isinstance(value, str):
            raise TypeError("secret values must be strings")
        self._secrets[name] = value

    # -- consumption -------------------------------------------------------------------------
    def token_provider(self, name: str) -> Callable[[], str | None]:
        """A zero-arg callable returning the secret (or None if it is not configured). Give this
        to a tool's constructor; it is looked up lazily so rotated secrets are picked up."""

        def provider() -> str | None:
            return self._secrets.get(name)

        provider.__qualname__ = f"token_provider[{name}]"
        return provider

    def has(self, name: str) -> bool:
        return name in self._secrets

    def names(self) -> list[str]:
        return sorted(self._secrets)

    def redactor(self, **kwargs: Any) -> Redactor:
        """A Redactor bound to this store: it always scrubs the store's *current* secret values."""
        return Redactor(source=self._values, **kwargs)

    def _values(self) -> list[str]:
        return [v for v in self._secrets.values() if len(v) >= self._min_length]

    # -- safety nets -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"CredentialStore(names={self.names()})"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("CredentialStore must not be serialized")

    def __iter__(self):
        raise TypeError("CredentialStore is not iterable; use token_provider()")


# ------------------------------------------------------------------------------------------
# Redactor
# ------------------------------------------------------------------------------------------
_SENSITIVE_KEY = re.compile(
    r"^(authorization|proxy-authorization|password|passwd|secret|client_?secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|auth[_-]?token|bearer|cookie|set-cookie|private[_-]?key|x-api-key)$",
    re.I,
)

# (pattern, replacement). Group references keep the harmless prefix ("Bearer ") for readability.
_SHAPES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S), REDACTED),
    (re.compile(r"(?i)\b(bearer)(\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1\2" + REDACTED),
    (
        re.compile(
            r"(?i)\b((?:proxy-)?authorization)(\"?\s*[:=]\s*\"?)(?:(?:basic|token|bearer|digest)\s+)?[^\s\"',;]{6,}"
        ),
        r"\1\2" + REDACTED,
    ),
    (re.compile(r"\bsk-(?:[A-Za-z0-9]+-)*[A-Za-z0-9_-]{16,}"), REDACTED),                  # OpenAI / Anthropic
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), REDACTED),                                # GitHub
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), REDACTED),
    (re.compile(r"\bxox[abprso]-[A-Za-z0-9-]{10,}"), REDACTED),                             # Slack
    (re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b"), REDACTED),                # AWS key id
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), REDACTED),                                   # Google API key
    (re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"), REDACTED),                                  # Google OAuth
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]*"), REDACTED),   # JWT
    (re.compile(r"(?<=://)[^/\s:@\"']+:[^/\s@\"']+(?=@)"), REDACTED),                       # user:pass@host
    (
        re.compile(
            r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret|password|passwd)[\"']?"
            r"\s*[:=]\s*)([\"'])([^\"'\s]{6,})\2"
        ),
        r"\1\2" + REDACTED + r"\2",
    ),
    (
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:API_?KEY|ACCESS_?TOKEN|TOKEN|SECRET|PASSWORD|PASSWD)[A-Z0-9_]*=)(?=[^\s\"'&;]*[A-Za-z])([^\s\"'&;]{6,})"
        ),
        r"\1" + REDACTED,
    ),
]


def _b64_fragments(secret: bytes, min_chars: int = 8) -> set[str]:
    """Substrings of base64(secret) that are independent of what surrounds the secret in the
    encoded stream (i.e. for each of the 3 byte alignments, minus the characters that mix in
    neighbouring bytes)."""
    out: set[str] = set()
    for alphabet in ("std", "url"):
        for offset in range(3):
            enc = base64.b64encode(b"\x00" * offset + secret)
            if alphabet == "url":
                enc = enc.replace(b"+", b"-").replace(b"/", b"_")
            text = enc.decode().rstrip("=")
            lead = {0: 0, 1: 2, 2: 3}[offset]
            text = text[lead:]
            if (offset + len(secret)) % 3:
                text = text[:-1]
            if len(text) >= min_chars:
                out.add(text)
    return out


def _variants(secret: str) -> set[str]:
    raw = secret.encode()
    v = {
        secret,
        quote(secret, safe=""),
        quote_plus(secret),
        quote(secret, safe="/"),
        json.dumps(secret)[1:-1],
        json.dumps(secret, ensure_ascii=False)[1:-1],
        repr(secret)[1:-1],
        html.escape(secret),
        raw.hex(),
        raw.hex().upper(),
    }
    v |= _b64_fragments(raw)
    return {x for x in v if len(x) >= 4}


class Redactor:
    """Scrub secrets from strings and arbitrarily nested str/dict/list/tuple/set structures."""

    def __init__(
        self,
        secrets: Iterable[str] = (),
        *,
        source: Callable[[], Iterable[str]] | None = None,
        min_secret_length: int = 6,
        min_prefix: int = 8,
        redact_shapes: bool = True,
        redact_sensitive_keys: bool = True,
        max_depth: int = 40,
    ):
        self._static: set[str] = set()
        self._source = source
        self._min_len = min_secret_length
        self._min_prefix = min_prefix
        self._shapes = redact_shapes
        self._keys = redact_sensitive_keys
        self._max_depth = max_depth
        self._cache_key: frozenset[str] | None = None
        self._needles: list[str] = []
        self._prefixes: list[str] = []
        for s in secrets:
            self.add_secret(s)

    # -- secrets -----------------------------------------------------------------------------
    def add_secret(self, value: str | None) -> None:
        if value and len(value) >= self._min_len:
            self._static.add(value)
            self._cache_key = None

    def _current(self) -> frozenset[str]:
        vals = set(self._static)
        if self._source is not None:
            vals.update(v for v in self._source() if v and len(v) >= self._min_len)
        return frozenset(vals)

    def _compile(self) -> None:
        secrets = self._current()
        if secrets == self._cache_key:
            return
        needles: set[str] = set()
        prefixes: set[str] = set()
        for s in secrets:
            needles |= _variants(s)
            if self._min_prefix and len(s) >= max(12, self._min_prefix + 4):
                prefixes.add(s)
        # longest first so that a long variant is replaced before any shorter one that it contains
        self._needles = sorted(needles, key=len, reverse=True)
        self._prefixes = sorted(prefixes, key=len, reverse=True)
        self._cache_key = secrets

    # -- text --------------------------------------------------------------------------------
    def redact_text(self, text: str) -> str:
        if not text:
            return text
        self._compile()
        out = text
        for needle in self._needles:
            if needle in out:
                out = out.replace(needle, REDACTED)
        out = self._redact_prefixes(out)
        if self._shapes:
            for pattern, repl in _SHAPES:
                out = pattern.sub(repl, out)
        return out

    def _redact_prefixes(self, text: str) -> str:
        """Truncated leaks: any run that starts with the first `min_prefix` chars of a secret and
        continues matching it (e.g. a log line that shows only 'ghp_abc123...')."""
        for secret in self._prefixes:
            head = secret[: self._min_prefix]
            start = text.find(head)
            while start != -1:
                end = start + len(head)
                while end < len(text) and end - start < len(secret) and text[end] == secret[end - start]:
                    end += 1
                text = text[:start] + REDACTED + text[end:]
                start = text.find(head, start + len(REDACTED))
        return text

    # -- structures --------------------------------------------------------------------------
    def redact(self, obj: Any) -> Any:
        return self._walk(obj, 0, {})

    def _walk(self, obj: Any, depth: int, seen: dict[int, Any]) -> Any:
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return self.redact_text(obj)
        if depth > self._max_depth:
            return REDACTED  # pathological nesting: withhold rather than risk a leak
        if isinstance(obj, (bytes, bytearray)):
            text = bytes(obj).decode("utf-8", "replace")
            red = self.redact_text(text)
            return bytes(obj) if red == text else red.encode()
        oid = id(obj)
        if oid in seen:
            return seen[oid]
        if isinstance(obj, Mapping):
            result: dict = {}
            seen[oid] = result
            for k, v in obj.items():
                key = self._walk(k, depth + 1, seen) if isinstance(k, str) else k
                if self._keys and isinstance(k, str) and _SENSITIVE_KEY.match(k) and isinstance(v, str) and v:
                    result[key] = REDACTED
                else:
                    result[key] = self._walk(v, depth + 1, seen)
            return result
        if isinstance(obj, list):
            lst: list = []
            seen[oid] = lst
            lst.extend(self._walk(v, depth + 1, seen) for v in obj)
            return lst
        if isinstance(obj, tuple):
            items = [self._walk(v, depth + 1, seen) for v in obj]
            return type(obj)(*items) if hasattr(obj, "_fields") else tuple(items)
        if isinstance(obj, (set, frozenset)):
            return type(obj)(self._walk(v, depth + 1, seen) for v in obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            try:
                changes = {
                    f.name: self._walk(getattr(obj, f.name), depth + 1, seen)
                    for f in dataclasses.fields(obj)
                    if f.init
                }
                return dataclasses.replace(obj, **changes)
            except Exception:  # noqa: BLE001 - redaction must never raise on an odd object and leak by crashing
                pass
        # Anything else (exceptions, datetimes, custom objects...) gets stringified somewhere
        # downstream (json default=str, f-strings): make sure the string form is clean.
        try:
            text = str(obj)
        except Exception:  # noqa: BLE001 - same: fail closed rather than propagate
            return REDACTED
        red = self.redact_text(text)
        return obj if red == text else red


# ------------------------------------------------------------------------------------------
# Layer 2 + 3: decorators
# ------------------------------------------------------------------------------------------
class RedactingTracer(Tracer):
    """Wraps any Tracer; scrubs every event before it is recorded (memory, Postgres, UI)."""

    def __init__(self, inner: Tracer, redactor: Redactor):
        self._inner = inner
        self._redactor = redactor

    async def record(self, event: TraceEvent) -> None:
        r = self._redactor
        # scrub in place: Tracer.log() returns the very same event object to its caller
        event.payload = r.redact(event.payload)
        event.session_id = r.redact(event.session_id)
        event.agent_name = r.redact(event.agent_name)
        if event.turn_id is not None:
            event.turn_id = r.redact(event.turn_id)
        await self._inner.record(event)

    def __getattr__(self, name: str) -> Any:  # expose e.g. InMemoryTracer.events / of_type
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


class RedactingProvider(ModelProvider):
    """Wraps any ModelProvider; the model only ever receives scrubbed messages and tool schemas,
    and its own output (text, tool-call arguments, parse errors) is scrubbed before it is acted on."""

    def __init__(self, inner: ModelProvider, redactor: Redactor):
        self._inner = inner
        self._redactor = redactor

    async def generate(self, messages: list[dict], tools: list[dict], max_tokens: int = 2048) -> ModelResponse:
        r = self._redactor
        resp = await self._inner.generate(r.redact(messages), r.redact(tools), max_tokens)
        if resp.text is not None:
            resp.text = r.redact_text(resp.text)
        for call in resp.tool_calls:
            call.arguments = r.redact(call.arguments)
        if resp.error:
            resp.error = r.redact_text(resp.error)
        return resp

    def supports_native_tool_calling(self) -> bool:
        return self._inner.supports_native_tool_calling()

    def __getattr__(self, name: str) -> Any:  # e.g. MockProvider.calls
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)


def harden(
    store: CredentialStore,
    provider: ModelProvider,
    tracer: Tracer,
    registry: Any = None,
    **redactor_kwargs: Any,
) -> tuple[ModelProvider, Tracer, Redactor]:
    """Wire all three layers from one store: returns (provider, tracer, redactor) and, if a
    registry is given, installs the redactor on it."""
    redactor = store.redactor(**redactor_kwargs)
    if registry is not None:
        registry.redactor = redactor
    return RedactingProvider(provider, redactor), RedactingTracer(tracer, redactor), redactor
