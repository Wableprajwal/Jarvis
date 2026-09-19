"""Phase 5: credential isolation. Acceptance: no secret ever appears in a message handed to the
model or in a trace payload - even when a tool result AND a tool error accidentally contain it."""
from __future__ import annotations

import base64
import dataclasses
import json
import pickle
import random
import string
from urllib.parse import quote, quote_plus

import pytest

from app.config import Settings
from app.harness.loop import run_agent_loop
from app.observability.trace import InMemoryTracer, TraceEvent
from app.providers.base import ModelResponse, ToolCall
from app.providers.mock_provider import MockProvider, text_response, tool_call_response
from app.tools.code_exec import CodeExecTool, select_backend
from app.tools.credentials import (
    REDACTED,
    CredentialStore,
    RedactingProvider,
    RedactingTracer,
    Redactor,
    harden,
)
from app.tools.registry import Tool, ToolRegistry, ToolResult

SECRET = "tok_live_Zx81qPq93LmN0aBv7RtY"  # deliberately not one of the well-known shapes
OTHER = "hunter2-correct-horse-battery"


# ---------------------------------------------------------------------------------------------
# CredentialStore
# ---------------------------------------------------------------------------------------------
def test_store_never_exposes_values_via_repr_str_pickle_iteration():
    store = CredentialStore({"gmail_token": SECRET, "empty": None})
    assert SECRET not in repr(store) and SECRET not in str(store)
    assert store.names() == ["gmail_token"] and store.has("gmail_token") and not store.has("empty")
    with pytest.raises(TypeError):
        pickle.dumps(store)
    with pytest.raises(TypeError):
        list(store)
    assert not hasattr(store, "get") and not hasattr(store, "secrets")
    assert SECRET not in json.dumps({"names": store.names()})


def test_token_provider_is_lazy_and_returns_none_when_missing():
    store = CredentialStore()
    provider = store.token_provider("gmail_token")
    assert provider() is None
    store.register("gmail_token", SECRET)
    assert provider() == SECRET  # picks up rotation without rebuilding the tool
    store.register("gmail_token", None)
    assert provider() is None


def test_from_settings_collects_only_secret_looking_string_fields(monkeypatch):
    monkeypatch.chdir("/")  # no .env
    s = Settings(gmail_token=SECRET, calendar_token=None, ollama_host="http://localhost:11434")
    store = CredentialStore.from_settings(s)
    assert store.names() == ["gmail_token"]
    assert store.token_provider("gmail_token")() == SECRET
    assert not store.has("ollama_host") and not store.has("memory_token_budget")


def test_store_redactor_follows_rotation():
    store = CredentialStore({"k": SECRET})
    red = store.redactor()
    assert SECRET not in red.redact_text(f"x {SECRET} y")
    store.register("k", OTHER)
    assert OTHER not in red.redact_text(f"now {OTHER}")


# ---------------------------------------------------------------------------------------------
# Redactor: registered secrets and their encodings
# ---------------------------------------------------------------------------------------------
def test_exact_value_everywhere_in_nested_structure():
    red = Redactor([SECRET])
    data = {
        "a": f"prefix {SECRET} suffix",
        "b": [1, 2.5, None, True, (f"x{SECRET}", {SECRET}), {"deep": {"deeper": [SECRET]}}],
        SECRET: "secret as a dict key",
        "bytes": SECRET.encode(),
        "n": 7,
    }
    out = red.redact(data)
    blob = json.dumps(out, default=repr)
    assert SECRET not in blob and REDACTED in blob
    assert out["n"] == 7 and out["b"][:4] == [1, 2.5, None, True]
    assert isinstance(out["b"][4], tuple) and isinstance(out["b"][4][1], set)


def test_input_is_never_mutated():
    red = Redactor([SECRET])
    data = {"a": [f"x {SECRET}"], "t": (f"{SECRET}",)}
    snapshot = json.dumps(data)
    red.redact(data)
    assert json.dumps(data) == snapshot


def test_common_encodings_of_the_secret():
    red = Redactor([SECRET])
    raw = SECRET.encode()
    forms = {
        "b64": base64.b64encode(raw).decode(),
        "b64-nopad": base64.b64encode(raw).decode().rstrip("="),
        "b64url": base64.urlsafe_b64encode(raw).decode(),
        "hex": raw.hex(),
        "HEX": raw.hex().upper(),
        "url": quote(SECRET, safe=""),
        "urlplus": quote_plus(SECRET),
        "quoted": f'"{SECRET}"',
        "single-quoted": f"'{SECRET}'",
        "json-string": json.dumps({"k": SECRET}),
        "repr": repr({"k": SECRET}),
    }
    for name, form in forms.items():
        out = red.redact_text(f"leak[{name}]: {form} end")
        assert SECRET not in out and raw.hex() not in out.lower(), name
        assert base64.b64encode(raw).decode().rstrip("=")[:16] not in out, name


def test_tricky_characters_in_secrets():
    secrets = ['pa"ss\\word/with+odd=chars&x', "üñíçødé-secret-ß-value", "sp ace secret\tvalue", "a/b+c=d==pad"]
    for s in secrets:
        red = Redactor([s])
        for form in (s, json.dumps(s), quote(s, safe=""), quote_plus(s), repr(s), base64.b64encode(s.encode()).decode()):
            assert s not in red.redact_text(f">>{form}<<"), (s, form)
        assert quote(s, safe="") not in red.redact_text(quote(s, safe=""))


@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("trailing", [0, 1, 2])
def test_base64_of_secret_embedded_in_a_longer_blob_at_any_alignment(offset, trailing):
    """The secret's base64 changes with the bytes around it; every alignment must be caught."""
    red = Redactor([SECRET])
    blob = base64.b64encode(b"x" * offset + SECRET.encode() + b"y" * trailing).decode()
    assert SECRET.encode().hex() not in red.redact_text(blob)
    out = red.redact_text(blob)
    # the distinctive middle of the encoding is gone, whichever alignment produced it
    for cand in range(3):
        core = base64.b64encode(b"." * cand + SECRET.encode()).decode()[4:-4]
        assert core[:12] not in out


def test_truncated_prefix_of_a_long_secret_is_redacted():
    secret = "AbCdEfGh1234567890ZyXwVu-long-secret"
    red = Redactor([secret])
    out = red.redact_text("log: token=AbCdEfGh12345678... (truncated)")
    assert "AbCdEfGh" not in out
    assert "AbCdEfG" in red.redact_text("unrelated AbCdEfG here")  # < 8 chars of prefix: left alone


def test_short_values_are_not_treated_as_secrets():
    red = Redactor(["abc", "12345"])
    assert red.redact_text("abc 12345 abcdef") == "abc 12345 abcdef"


def test_fuzz_random_secrets_in_random_contexts():
    rng = random.Random(1234)
    alphabet = string.ascii_letters + string.digits + "-_./+=!@#$%^&*()"
    for _ in range(150):
        secret = "".join(rng.choice(alphabet) for _ in range(rng.randint(8, 48)))
        red = Redactor([secret])
        raw = secret.encode()
        pieces = [
            secret, quote(secret, safe=""), quote_plus(secret), json.dumps(secret), repr(secret),
            raw.hex(), base64.b64encode(raw).decode(), base64.urlsafe_b64encode(raw).decode(),
            base64.b64encode(b"k=" + raw + b"&z").decode(), base64.b64encode(b"?" + raw).decode(),
        ]
        for piece in pieces:
            ctx = f"{''.join(rng.choice(string.ascii_letters + ' ') for _ in range(rng.randint(0, 20)))}{piece}" \
                  f"{''.join(rng.choice(string.ascii_letters + ' ') for _ in range(rng.randint(0, 20)))}"
            out = red.redact_text(ctx)
            assert secret not in out
            assert raw.hex() not in out
            mid = base64.b64encode(raw).decode().strip("=")[8:-8]
            assert len(mid) < 8 or mid not in out, (secret, piece)


# ---------------------------------------------------------------------------------------------
# Redactor: well-known token shapes (no registration needed)
# ---------------------------------------------------------------------------------------------
JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
SHAPES = {
    "bearer": ("Authorization: Bearer abcDEF123456.ghi-jkl_mno~pqr+stu/vwx=", "abcDEF123456"),
    "bearer-lower": ("sent header bearer q1w2e3r4t5y6u7i8o9p0", "q1w2e3r4t5y6u7i8o9p0"),
    "basic-auth-header": ('{"Authorization": "Basic dXNlcjpwYXNzd29yZA=="}', "dXNlcjpwYXNzd29yZA"),
    "openai": ("key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789", "abcdefghijklmnopqrstuvwxyz"),
    "anthropic": ("ANTHROPIC sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz012345-_AA", "AbCdEfGhIjKlMnOp"),
    "github": ("token ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8", "a1B2c3D4e5F6g7H8"),
    "github-pat": ("github_pat_11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz0123456789", "11ABCDEFG0abcdef"),
    "slack-bot": ("xoxb-123456789012-123456789012-AbCdEfGhIjKlMnOpQrStUvWx", "AbCdEfGhIjKlMnOp"),
    "slack-user": ("xoxp-123456789012-123456789012-123456789012-abcdef0123456789abcdef", "abcdef0123456789"),
    "aws": ("aws_access_key_id = AKIAIOSFODNN7EXAMPLE", "AKIAIOSFODNN7EXAMPLE"),
    "google-api": ("key=AIzaSyA-1234567890abcdefghijklmnopqrstu", "AIzaSyA-1234567890"),
    "google-oauth": ("ya29.a0AfH6SMBx1234567890abcdefghijklmnopqrs", "a0AfH6SMBx12345"),
    "jwt": (f"id_token: {JWT}", "eyJhbGciOiJIUzI1NiIs"),
    "pem": ("-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7x\nabcdef\n-----END RSA PRIVATE KEY-----", "MIIEowIBAAKCAQEA7x"),
    "db-url": ("postgresql://appuser:S3cr3tPw!@db.internal:5432/app", "S3cr3tPw"),
    "json-password": ('{"user": "bob", "password": "correct-horse-battery"}', "correct-horse-battery"),
    "env-style": ("export GMAIL_API_TOKEN=abc123def456ghi789", "abc123def456ghi789"),
    "client-secret": ("client_secret='q9w8e7r6t5y4u3i2o1p'", "q9w8e7r6t5y4u3i2o1p"),
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_known_token_shapes_are_redacted_without_registration(name):
    text, needle = SHAPES[name]
    out = Redactor().redact_text(f"before {text} after")
    assert needle not in out, out
    assert REDACTED in out and out.startswith("before") and out.endswith("after") or name == "pem"


def test_sensitive_dict_keys_are_masked_but_counters_are_not():
    out = Redactor().redact(
        {"Authorization": "whatever-value", "password": "p", "api_key": "k", "tokens": 1200, "prompt_tokens": 5,
         "headers": {"Set-Cookie": "session=abc", "Content-Type": "application/json"}, "token": ""}
    )
    assert out["Authorization"] == out["password"] == out["api_key"] == REDACTED
    assert out["tokens"] == 1200 and out["prompt_tokens"] == 5
    assert out["headers"]["Set-Cookie"] == REDACTED and out["headers"]["Content-Type"] == "application/json"


def test_ordinary_text_and_code_are_left_alone():
    red = Redactor([SECRET])
    samples = [
        "The task-force shipped the risk-assessment for sk-learn users on Tuesday.",
        "def add(a, b):\n    return a + b  # tokens: 12, max_tokens=4096",
        "Use a bearer of good news. Basic configuration is required.",
        "https://example.com/path?x=1#frag and http://user@host/ and ssh://git@github.com/x/y.git",
        "The password field must be at least 8 characters long.",
        json.dumps({"role": "tool", "content": '{"status": "ok", "result": {"sum": 45}}'}),
    ]
    for s in samples:
        assert red.redact_text(s) == s, s


def test_odd_objects_cycles_and_deep_nesting_do_not_crash_or_leak():
    red = Redactor([SECRET])
    cyc: list = []
    cyc.append(cyc)
    cyc.append(f"x {SECRET}")
    out = red.redact(cyc)
    assert out[1] == f"x {REDACTED}" and out[0] is out

    deep: object = SECRET
    for _ in range(200):
        deep = [deep]
    assert SECRET not in json.dumps(red.redact(deep))

    err = ValueError(f"request failed with token {SECRET}")
    cleaned = red.redact({"error": err})
    assert SECRET not in str(cleaned["error"]) and isinstance(cleaned["error"], str)

    @dataclasses.dataclass
    class Box:
        value: str
        n: int = 1

    boxed = red.redact(Box(f"v={SECRET}"))
    assert isinstance(boxed, Box) and SECRET not in boxed.value and boxed.n == 1

    class Weird:
        def __str__(self):
            return f"Weird({SECRET})"

    assert SECRET not in str(red.redact([Weird()])[0])
    assert red.redact(Weird.__new__(Weird)) is not None


def test_redaction_is_idempotent():
    red = Redactor([SECRET])
    once = red.redact({"a": f"{SECRET} Bearer abcdefghijklmnop"})
    assert red.redact(once) == once


# ---------------------------------------------------------------------------------------------
# decorators
# ---------------------------------------------------------------------------------------------
async def test_redacting_tracer_scrubs_payload_before_recording_and_in_the_returned_event():
    inner = InMemoryTracer()
    tracer = RedactingTracer(inner, Redactor([SECRET]))
    ev = await tracer.log(
        "s1", "code", "tool_result",
        {"tool": "x", "error": f"401 for token {SECRET}", "output": {"headers": {"Authorization": f"Bearer {SECRET}"}}},
    )
    assert SECRET not in json.dumps(ev.to_dict()) and SECRET not in json.dumps([e.to_dict() for e in inner.events])
    assert inner.events[0].payload["tool"] == "x"
    assert tracer.of_type("tool_result") == inner.events  # delegation of the wrapped tracer's API
    with pytest.raises(AttributeError):
        tracer._does_not_exist  # noqa: B018


async def test_redacting_provider_scrubs_input_and_output_without_mutating_callers_list():
    mock = MockProvider(
        script=[
            ModelResponse(
                text=f"I will use {SECRET}",
                tool_calls=[ToolCall("send", {"token": SECRET, "to": "a@b.c"}, "c1")],
                stop_reason="tool_use",
            )
        ]
    )
    prov = RedactingProvider(mock, Redactor([SECRET]))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": f"my token is {SECRET}"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "t", "arguments": {"k": SECRET}}]},
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": json.dumps({"error": SECRET})},
    ]
    original = json.dumps(messages)
    resp = await prov.generate(messages, [{"name": "t", "description": f"uses {SECRET}", "input_schema": {}}])
    assert json.dumps(messages) == original  # caller's transcript untouched
    assert SECRET not in json.dumps(mock.calls, default=str)
    assert SECRET not in (resp.text or "") and SECRET not in json.dumps(resp.tool_calls[0].arguments)
    assert resp.tool_calls[0].arguments["to"] == "a@b.c"
    assert prov.supports_native_tool_calling() is True
    assert len(prov.calls) == 1  # delegation


async def test_registry_redactor_scrubs_output_error_and_exceptions_and_fails_closed():
    class Echo(Tool):
        name = "echo"
        description = "d"
        input_schema = {"type": "object", "properties": {"mode": {"type": "string"}}, "required": ["mode"]}

        async def execute(self, arguments):
            mode = arguments["mode"]
            if mode == "ok":
                return ToolResult.success({"body": f"reflected {SECRET}", "list": [SECRET]})
            if mode == "fail":
                return ToolResult.failure(f"auth failed for {SECRET}", details=f"Bearer {SECRET}")
            raise RuntimeError(f"boom while using {SECRET}")

    reg = ToolRegistry(redactor=Redactor([SECRET]))
    reg.register(Echo())
    reg.grant("a", ["echo"])
    for mode in ("ok", "fail", "raise"):
        res = await reg.dispatch("a", "echo", {"mode": mode}, "s")
        assert SECRET not in json.dumps(dataclasses.asdict(res)) and SECRET not in res.to_content(), mode
    bad = await reg.dispatch("a", "echo", {"mode": SECRET, "extra": SECRET}, "s")  # validation error path
    assert SECRET not in bad.to_content()
    unknown = await reg.dispatch("a", f"nope_{SECRET}", {}, "s")
    assert SECRET not in unknown.to_content()

    class Exploding:
        def redact(self, obj):
            raise RuntimeError("redactor bug")

    reg.redactor = Exploding()
    res = await reg.dispatch("a", "echo", {"mode": "ok"}, "s")
    assert not res.ok and SECRET not in res.to_content() and "withheld" in res.error


async def test_registry_confirmation_path_is_scrubbed_too():
    class Send(Tool):
        name = "send"
        description = "d"
        input_schema = {"type": "object"}
        requires_confirmation = True

        async def execute(self, arguments):
            return ToolResult.success({"echo": f"sent with {SECRET}"})

    reg = ToolRegistry(redactor=Redactor([SECRET]))
    reg.register(Send())
    reg.grant("a", ["send"])
    pending = await reg.dispatch("a", "send", {}, "s")
    action = await reg.confirm(pending.confirmation_id)
    assert SECRET not in json.dumps(action.result.output)


def test_registry_without_a_redactor_is_unchanged():
    assert ToolRegistry().redactor is None


# ---------------------------------------------------------------------------------------------
# end to end: nothing reaches the model or the trace
# ---------------------------------------------------------------------------------------------
class LeakyMailTool(Tool):
    """Takes its credential by injection (as production tools do) and then, through a
    sloppy backend, echoes it in a result, an error string and an exception."""

    name = "mail_search"
    description = "Search mail."
    input_schema = {
        "type": "object",
        "properties": {"mode": {"type": "string", "enum": ["result", "error", "exception", "details"]}},
        "required": ["mode"],
    }

    def __init__(self, token_provider):
        self._token = token_provider
        self.used_token = None

    async def execute(self, arguments):
        token = self._token()
        self.used_token = token
        mode = arguments["mode"]
        if mode == "result":
            return ToolResult.success({"debug": {"request_headers": {"Authorization": f"Bearer {token}"}}, "n": 0})
        if mode == "error":
            return ToolResult.failure(f"HTTP 401 for user with token={token}")
        if mode == "details":
            return ToolResult.failure("auth failed", url=f"https://api.example.com/v1/mail?access_token={token}")
        raise ConnectionError(f"could not connect using credentials {token}")


def _leak_script():
    return [
        ModelResponse(
            text=None,
            tool_calls=[ToolCall("mail_search", {"mode": m}, f"c{i}") for i, m in enumerate(["result", "error", "exception", "details"])],
            stop_reason="tool_use",
        ),
        text_response("I could not read the mailbox."),
    ]


def _assert_clean(blob: str, secret: str = SECRET):
    raw = secret.encode()
    assert secret not in blob
    assert base64.b64encode(raw).decode().rstrip("=") not in blob
    assert quote(secret, safe="") not in blob or quote(secret, safe="") == secret


async def test_e2e_secret_in_tool_result_and_error_never_reaches_model_or_trace():
    store = CredentialStore({"gmail_token": SECRET})
    tool = LeakyMailTool(store.token_provider("gmail_token"))
    registry = ToolRegistry()
    registry.register(tool)
    registry.grant("ops", ["mail_search"])
    mock = MockProvider(script=_leak_script())
    tracer = InMemoryTracer()
    provider, rtracer, _ = harden(store, mock, tracer, registry)

    result = await run_agent_loop(
        provider,
        [{"role": "system", "content": "You are Jarvis."}, {"role": "user", "content": "check my mail"}],
        registry.tools_for("ops"), registry, rtracer, "sess-1", "ops",
    )
    assert tool.used_token == SECRET  # the tool really did use the injected credential
    assert result.text == "I could not read the mailbox."
    assert len(mock.calls) == 2

    # the model saw the tool outcomes, but with the secret scrubbed
    tool_msgs = [m for m in mock.calls[1]["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 4 and all(REDACTED in m["content"] for m in tool_msgs)
    assert {json.loads(m["content"])["status"] for m in tool_msgs} == {"ok", "error"}

    _assert_clean(json.dumps(mock.calls, default=str))          # every message + tool schema handed to the model
    _assert_clean(json.dumps([e.to_dict() for e in tracer.events]))   # every stored trace event
    _assert_clean(json.dumps(result.messages, default=str))     # the harness transcript as well
    assert {e.event_type for e in tracer.events} >= {"model_call", "tool_call", "tool_result", "final_response"}


@pytest.mark.parametrize("layer", ["registry", "provider", "tracer"])
async def test_each_layer_independently_protects_its_own_surface(layer):
    """Defence in depth: with the other two layers switched OFF, each layer alone still keeps the
    secret out of what it is responsible for."""
    store = CredentialStore({"gmail_token": SECRET})
    tool = LeakyMailTool(store.token_provider("gmail_token"))
    red = store.redactor()
    registry = ToolRegistry(redactor=red if layer == "registry" else None)
    registry.register(tool)
    registry.grant("ops", ["mail_search"])
    mock = MockProvider(script=_leak_script())
    tracer = InMemoryTracer()
    provider = RedactingProvider(mock, red) if layer == "provider" else mock
    rtracer = RedactingTracer(tracer, red) if layer == "tracer" else tracer

    await run_agent_loop(
        provider, [{"role": "user", "content": "check my mail"}], registry.tools_for("ops"), registry,
        rtracer, "sess-1", "ops",
    )
    model_blob = json.dumps(mock.calls, default=str)
    trace_blob = json.dumps([e.to_dict() for e in tracer.events])
    if layer == "registry":
        _assert_clean(model_blob), _assert_clean(trace_blob)   # everything downstream of dispatch is clean
    elif layer == "provider":
        _assert_clean(model_blob)
        assert SECRET in trace_blob                              # (proves the other surface was unprotected)
    else:
        _assert_clean(trace_blob)
        assert SECRET in model_blob


async def test_model_echoing_a_secret_is_scrubbed_before_it_is_acted_on_or_shown():
    store = CredentialStore({"gmail_token": SECRET})
    seen_args = []

    class Sink(Tool):
        name = "sink"
        description = "d"
        input_schema = {"type": "object"}

        async def execute(self, arguments):
            seen_args.append(arguments)
            return ToolResult.success({"ok": True})

    registry = ToolRegistry()
    registry.register(Sink())
    registry.grant("a", ["sink"])
    mock = MockProvider(
        script=[tool_call_response("sink", {"note": f"token is {SECRET}"}), text_response(f"Your token is {SECRET}")]
    )
    tracer = InMemoryTracer()
    provider, rtracer, _ = harden(store, mock, tracer, registry)
    result = await run_agent_loop(provider, [{"role": "user", "content": "hi"}], registry.tools_for("a"), registry, rtracer, "s", "a")
    assert SECRET not in json.dumps(seen_args) and SECRET not in (result.text or "")
    _assert_clean(json.dumps([e.to_dict() for e in tracer.events]))
    _assert_clean(json.dumps(mock.calls, default=str))


async def test_secret_the_user_types_is_not_forwarded_to_the_model_or_trace():
    store = CredentialStore({"gmail_token": SECRET})
    mock = MockProvider(script=[text_response("ok")])
    tracer = InMemoryTracer()
    provider, rtracer, _ = harden(store, mock, tracer, ToolRegistry())
    await run_agent_loop(
        provider, [{"role": "user", "content": f"here is my token {SECRET}, use it"}], [], ToolRegistry(), rtracer, "s", "a"
    )
    _assert_clean(json.dumps(mock.calls, default=str))
    _assert_clean(json.dumps([e.to_dict() for e in tracer.events]))


async def test_model_visible_tool_schemas_never_contain_credentials():
    store = CredentialStore({"gmail_token": SECRET})
    tool = LeakyMailTool(store.token_provider("gmail_token"))
    assert SECRET not in json.dumps(tool.schema) and SECRET not in repr(tool.__dict__.get("_token"))
    assert SECRET not in repr(store)


async def test_code_exec_output_containing_a_workspace_secret_is_scrubbed_end_to_end(tmp_path):
    """A secret that sits in an exposed workspace file (not a denylisted name) and is printed by
    the generated code must not get back to the model."""
    b, reasons = await select_backend()
    if b is None:
        pytest.skip(f"no sandbox available: {reasons}")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "notes.txt").write_text(f"deploy notes\ngmail token: {SECRET}\n")
    store = CredentialStore({"gmail_token": SECRET})
    registry = ToolRegistry()
    registry.register(CodeExecTool(ws, backend=b))
    registry.grant("code", ["code_exec"])
    mock = MockProvider(
        script=[
            tool_call_response("code_exec", {"code": "print(open('notes.txt').read())", "files": ["notes.txt"]}),
            text_response("done"),
        ]
    )
    tracer = InMemoryTracer()
    provider, rtracer, _ = harden(store, mock, tracer, registry)
    await run_agent_loop(provider, [{"role": "user", "content": "run it"}], registry.tools_for("code"), registry, rtracer, "s", "code")
    tool_msg = next(m for m in mock.calls[1]["messages"] if m["role"] == "tool")
    assert "deploy notes" in tool_msg["content"] and REDACTED in tool_msg["content"]
    _assert_clean(json.dumps(mock.calls, default=str))
    _assert_clean(json.dumps([e.to_dict() for e in tracer.events]))


async def test_harness_env_credentials_do_not_reach_sandboxed_code_even_if_settings_hold_them(tmp_path, monkeypatch):
    b, reasons = await select_backend()
    if b is None:
        pytest.skip(f"no sandbox available: {reasons}")
    monkeypatch.setenv("GMAIL_TOKEN", SECRET)
    monkeypatch.chdir("/")
    store = CredentialStore.from_settings(Settings())
    assert store.token_provider("gmail_token")() == SECRET       # the harness has it...
    res = await CodeExecTool(tmp_path, backend=b).execute(
        {"code": "import os\nprint(sorted(os.environ))\nprint([v for v in os.environ.values()])"}
    )
    assert res.ok and SECRET not in json.dumps(res.output) and "GMAIL" not in json.dumps(res.output)  # ...the sandbox does not


def test_trace_event_is_a_mutable_dataclass_so_the_tracer_can_scrub_in_place():
    ev = TraceEvent("s", "a", None, "error", {"x": 1})
    assert dataclasses.is_dataclass(ev) and not ev.__dataclass_params__.frozen
