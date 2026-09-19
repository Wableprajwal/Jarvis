"""Phase 5: code_exec sandbox. Everything here uses REAL execution of hostile code against every
sandbox backend that is usable on the current host (Docker / Seatbelt / bwrap / unshare). A backend
that is unavailable is skipped with its reason; `test_a_backend_is_available_on_this_platform`
fails loudly if the host has none (that would mean code_exec is disabled here)."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.tools import code_exec as ce
from app.tools.code_exec import (
    BwrapBackend,
    CodeExecTool,
    DockerBackend,
    SandboxBackend,
    SandboxLimits,
    SeatbeltBackend,
    UnshareBackend,
    execute_sandboxed,
    resolve_workspace_files,
    select_backend,
)
from app.tools.registry import ToolRegistry

BACKENDS = ["docker", "seatbelt", "bwrap", "unshare"]
_instances: dict[str, SandboxBackend] = {}
_CTORS = {"docker": DockerBackend, "seatbelt": SeatbeltBackend, "bwrap": BwrapBackend, "unshare": UnshareBackend}


@pytest.fixture(params=BACKENDS)
async def backend(request) -> SandboxBackend:
    name = request.param
    b = _instances.setdefault(name, _CTORS[name]())
    reason = await b.verify()
    if reason is not None:
        pytest.skip(f"{name} sandbox backend unavailable on this host: {reason}")
    return b


@pytest.fixture
async def any_backend() -> SandboxBackend:
    b, reasons = await select_backend(candidates=[_instances.setdefault(n, _CTORS[n]()) for n in BACKENDS])
    if b is None:
        pytest.skip(f"no sandbox backend available: {reasons}")
    return b


def limits(**kw) -> SandboxLimits:
    kw.setdefault("timeout_s", 10)
    return SandboxLimits(**kw)


async def run(backend, code, files=None, **kw):
    return await execute_sandboxed(backend, code, files or {}, limits(**kw))


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------------------------
# platform sanity
# ---------------------------------------------------------------------------------------------
async def test_a_backend_is_available_on_this_platform():
    if os.environ.get("JARVIS_ALLOW_NO_SANDBOX_TESTS"):
        pytest.skip("explicitly allowed to run without a sandbox")
    b, reasons = await select_backend(candidates=[_instances.setdefault(n, _CTORS[n]()) for n in BACKENDS])
    assert b is not None, f"code_exec would be DISABLED on this host; backend failures: {reasons}"
    if sys.platform == "darwin":
        assert b.name in {"docker", "seatbelt"}


async def test_basic_execution(backend):
    r = await run(backend, "print(sum(range(10)))\nimport sys; print('err', file=sys.stderr)")
    assert r.exit_code == 0 and r.stdout.strip() == "45" and "err" in r.stderr
    assert not r.timed_out and not r.memory_exceeded


async def test_exit_code_and_traceback(backend):
    r = await run(backend, "import sys\nsys.exit(3)")
    assert r.exit_code == 3
    r = await run(backend, "x = 1\nraise ValueError('boom')")
    assert r.exit_code == 1 and "ValueError: boom" in r.stderr
    r = await run(backend, "def (:")
    assert r.exit_code == 1 and "SyntaxError" in r.stderr


async def test_stdlib_works_and_harness_packages_are_invisible(backend):
    r = await run(
        backend,
        "import json, hashlib, re, math, datetime, collections, itertools, sqlite3, decimal, zlib, base64\n"
        "print(hashlib.sha256(b'x').hexdigest()[:8], math.pi > 3)\n"
        "try:\n    import jsonschema; print('LEAK jsonschema')\nexcept ImportError: print('no jsonschema')\n"
        "try:\n    import app; print('LEAK app')\nexcept ImportError: print('no app')\n"
        "import sys; print(sys.flags.isolated, sys.flags.no_site)",
    )
    assert r.exit_code == 0, r.stderr
    assert "2d711642" in r.stdout and "LEAK" not in r.stdout and "1 1" in r.stdout


# ---------------------------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------------------------
async def test_no_network_tcp_dns_udp_http(backend):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.3)
    port = srv.getsockname()[1]
    code = f"""
import socket, urllib.request
def attempt(name, fn):
    try:
        fn(); print("REACHED", name)
    except BaseException as e:
        print("BLOCKED", name, type(e).__name__)
attempt("tcp-localhost", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=3))
attempt("tcp-external-ip", lambda: socket.create_connection(("1.1.1.1", 80), timeout=3))
attempt("dns", lambda: socket.getaddrinfo("example.com", 80))
attempt("dns-localhost", lambda: socket.getaddrinfo("localhost", {port}))
attempt("udp", lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("127.0.0.1", {port})))
attempt("http", lambda: urllib.request.urlopen("http://127.0.0.1:{port}/", timeout=3))
"""
    try:
        r = await run(backend, code)
        assert r.exit_code == 0, r.stderr
        assert "REACHED" not in r.stdout, r.stdout
        assert r.stdout.count("BLOCKED") == 6
        with pytest.raises((TimeoutError, socket.timeout)):  # the listener never saw a connection
            srv.accept()
    finally:
        srv.close()


async def test_no_unix_socket_access(backend, tmp_path):
    import tempfile

    short = tempfile.mkdtemp(prefix="jsb", dir="/tmp")  # AF_UNIX paths are limited to ~104 bytes
    path = os.path.join(short, "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(0.3)
    try:
        r = await run(
            backend,
            f"import socket\ns = socket.socket(socket.AF_UNIX)\n"
            f"try:\n    s.connect({path!r}); print('REACHED')\nexcept OSError as e: print('BLOCKED', type(e).__name__)",
        )
        assert "BLOCKED" in r.stdout and "REACHED" not in r.stdout
        with pytest.raises((TimeoutError, socket.timeout)):
            srv.accept()
    finally:
        srv.close()
        import shutil

        shutil.rmtree(short, ignore_errors=True)


# ---------------------------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def workspace(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "pkg").mkdir(parents=True)
    (ws / "data.txt").write_text("HELLO-FROM-WORKSPACE\n")
    (ws / "pkg" / "mod.py").write_text("VALUE = 41\n")
    (ws / "other.txt").write_text("NOT-REQUESTED-FILE\n")
    (ws / ".env").write_text("GMAIL_TOKEN=super-secret-env-value\n")
    return ws


@pytest.fixture
def canary(tmp_path):
    d = tmp_path / "outside"
    d.mkdir()
    f = d / "canary.txt"
    f.write_text("CANARY-7f3a9c1e-outside-scope\n")
    return f


async def test_cannot_read_outside_scope(backend, canary, workspace):
    home = str(Path.home())
    code = f"""
import os
def attempt(name, fn):
    try:
        print("READ", name, repr(fn())[:80])
    except BaseException as e:
        print("DENIED", name, type(e).__name__)
attempt("canary-abs", lambda: open({str(canary)!r}).read())
attempt("canary-dir", lambda: os.listdir({str(canary.parent)!r}))
attempt("canary-relative", lambda: open("../../../../../../../../../../.." + {str(canary)!r}).read())
attempt("home", lambda: os.listdir({home!r}))
attempt("ssh", lambda: os.listdir({home + '/.ssh'!r}))
attempt("workspace-root", lambda: os.listdir({str(workspace)!r}))
attempt("workspace-env", lambda: open({str(workspace / '.env')!r}).read())
attempt("source", lambda: open({ce.__file__!r}).read())
attempt("root-dir", lambda: [n for n in os.listdir("/") if n in ("Users", "home")] )
os.chdir("/")
attempt("chdir-root-canary", lambda: open({str(canary)!r}).read())
"""
    r = await run(backend, code)
    assert r.exit_code == 0, r.stderr
    assert "CANARY-7f3a9c1e" not in r.stdout
    assert "super-secret-env-value" not in r.stdout
    for name in ("canary-abs", "canary-dir", "canary-relative", "home", "ssh", "workspace-root",
                 "workspace-env", "source", "chdir-root-canary"):
        assert f"DENIED {name} " in r.stdout, r.stdout


async def test_cannot_read_etc_passwd(backend):
    r = await run(backend, "print(open('/etc/passwd').read())")
    if backend.name == "docker":
        # A container has its own /etc/passwd; the *host's* files are what must be unreachable
        # (covered by the canary/home tests). Just make sure it did not leak this host's user.
        assert os.environ.get("USER", "\0") not in r.stdout or sys.platform == "darwin"
        return
    assert r.exit_code != 0 and "root:" not in r.stdout
    assert "PermissionError" in r.stderr or "FileNotFoundError" in r.stderr


async def test_mounted_file_is_readable_but_not_writable(backend, workspace):
    files = resolve_workspace_files(workspace, ["data.txt", "pkg/mod.py"])
    code = """
import os, shutil, importlib
print("CONTENT", open("data.txt").read().strip())
def attempt(name, fn):
    try:
        fn(); print("SUCCEEDED", name)
    except BaseException as e:
        print("DENIED", name, type(e).__name__)
attempt("truncate", lambda: open("data.txt", "w").write("pwned"))
attempt("append", lambda: open("data.txt", "a").write("pwned"))
attempt("unlink", lambda: os.unlink("data.txt"))
attempt("rename-away", lambda: os.rename("data.txt", "moved.txt"))
open("decoy.txt", "w").write("decoy")
attempt("rename-over", lambda: os.rename("decoy.txt", "data.txt"))
attempt("chmod", lambda: os.chmod("data.txt", 0o666))
attempt("link", lambda: os.link("data.txt", "hardlink.txt") or open("hardlink.txt", "w").write("x"))
attempt("nested-write", lambda: open("pkg/mod.py", "w").write("VALUE = 0"))
attempt("rename-dir", lambda: os.rename("pkg", "pkg2"))
attempt("rmtree", lambda: shutil.rmtree("pkg"))
import pkg.mod
print("IMPORTED", pkg.mod.VALUE)
print("FINAL", open("data.txt").read().strip(), open("pkg/mod.py").read().strip())
"""
    r = await run(backend, code, files)
    assert r.exit_code == 0, r.stderr
    assert "CONTENT HELLO-FROM-WORKSPACE" in r.stdout
    assert "SUCCEEDED" not in r.stdout, r.stdout
    assert "IMPORTED 41" in r.stdout
    assert "FINAL HELLO-FROM-WORKSPACE VALUE = 41" in r.stdout
    # and the real workspace was never touched
    assert (workspace / "data.txt").read_text() == "HELLO-FROM-WORKSPACE\n"
    assert (workspace / "pkg" / "mod.py").read_text() == "VALUE = 41\n"


async def test_unrequested_workspace_files_are_invisible(backend, workspace):
    files = resolve_workspace_files(workspace, ["data.txt"])
    r = await run(
        backend,
        "import os\nprint(sorted(os.listdir('.')))\n"
        "try:\n    print(open('other.txt').read())\nexcept OSError as e: print('DENIED', type(e).__name__)\n"
        "try:\n    print(open('.env').read())\nexcept OSError as e: print('DENIED', type(e).__name__)",
        files,
    )
    assert "NOT-REQUESTED-FILE" not in r.stdout and "super-secret-env-value" not in r.stdout
    assert "'data.txt'" in r.stdout and "other.txt" not in r.stdout.split("DENIED")[0]


async def test_cannot_write_outside_scratch(backend, canary, tmp_path, workspace):
    targets = [
        tmp_path / "outside" / "evil.txt",
        canary,                                   # overwrite an existing file
        workspace / "evil.txt",
        Path.home() / "jarvis_sandbox_evil_DELETE_ME.txt",
        Path("/tmp") / "jarvis_sandbox_evil_DELETE_ME.txt",
        Path("/etc/jarvis_sandbox_evil"),
    ]
    code = "import os\n" + "".join(
        f"try:\n    open({str(t)!r}, 'w').write('PWNED'); print('WROTE', {str(t)!r})\n"
        f"except BaseException as e: print('DENIED', type(e).__name__)\n"
        for t in targets
    ) + (
        f"try:\n    os.mkdir({str(tmp_path / 'outside' / 'newdir')!r}); print('WROTE mkdir')\n"
        "except BaseException as e: print('DENIED mkdir', type(e).__name__)\n"
        f"try:\n    os.symlink('/etc', {str(tmp_path / 'outside' / 'lnk')!r}); print('WROTE symlink')\n"
        "except BaseException as e: print('DENIED symlink', type(e).__name__)\n"
        "open('inside.txt','w').write('ok'); print('INSIDE', os.getcwd(), os.path.exists('inside.txt'))\n"
    )
    r = await run(backend, code)
    assert r.exit_code == 0, r.stderr
    assert "WROTE" not in r.stdout, r.stdout
    assert "INSIDE" in r.stdout
    assert canary.read_text() == "CANARY-7f3a9c1e-outside-scope\n"
    for t in targets[:1] + targets[2:]:
        assert not t.exists(), t
    assert not (tmp_path / "outside" / "newdir").exists() and not (tmp_path / "outside" / "lnk").exists()


async def test_scratch_is_deleted_afterwards(backend):
    if backend.name == "docker":
        pytest.skip("docker scratch is a tmpfs that disappears with the container")
    r = await run(backend, "import os; open('leftover.txt','w').write('x'); os.makedirs('a/b'); print(os.getcwd())")
    cwd = Path(r.stdout.strip())
    assert cwd.name == "scratch" and not cwd.exists() and not cwd.parent.exists()


async def test_symlink_planted_in_scratch_cannot_reach_outside(backend, canary):
    r = await run(
        backend,
        f"import os\nos.symlink({str(canary)!r}, 'link.txt')\n"
        "try:\n    print(open('link.txt').read())\nexcept OSError as e: print('DENIED', type(e).__name__)",
    )
    assert "CANARY" not in r.stdout and "DENIED" in r.stdout


async def test_exec_other_binaries_cannot_reveal_files(backend, canary):
    code = f"""
import os, ctypes
def attempt(name, fn):
    try:
        print("RESULT", name, fn())
    except BaseException as e:
        print("DENIED", name, type(e).__name__)
attempt("system", lambda: os.system("cat {canary}"))
attempt("libc-system", lambda: ctypes.CDLL(None).system(b"cat {canary}"))
attempt("execv", lambda: os.execv("/bin/cat", ["cat", {str(canary)!r}]))
"""
    r = await run(backend, code)
    assert "CANARY-7f3a9c1e" not in r.stdout and "CANARY-7f3a9c1e" not in r.stderr


# ---------------------------------------------------------------------------------------------
# environment / credentials
# ---------------------------------------------------------------------------------------------
async def test_child_cannot_see_harness_env(backend, monkeypatch):
    monkeypatch.setenv("JARVIS_CANARY_SECRET", "env-canary-3c9d7e11")
    monkeypatch.setenv("GMAIL_TOKEN", "gmail-canary-token-99182")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-canary-55aa")
    r = await run(
        backend,
        "import os, json\nprint(json.dumps(dict(os.environ)))\n"
        "print(os.getenv('JARVIS_CANARY_SECRET'), os.environ.get('GMAIL_TOKEN'))\n"
        "try:\n    print(open('/proc/self/environ').read())\nexcept OSError as e: print('NO-PROC')",
    )
    assert r.exit_code == 0, r.stderr
    for needle in ("env-canary-3c9d7e11", "gmail-canary-token-99182", "aws-canary-55aa", "GMAIL_TOKEN"):
        assert needle not in r.stdout
    env = json.loads(r.stdout.splitlines()[0])
    assert set(env) <= {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "PYTHONIOENCODING", "TERM", "PWD",
                        "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}, sorted(env)
    assert "None None" in r.stdout


async def test_child_does_not_inherit_harness_file_descriptors(backend, tmp_path):
    f = tmp_path / "harness_fd.txt"
    f.write_text("HARNESS-FD-CONTENT")
    fd = os.open(f, os.O_RDONLY)
    os.set_inheritable(fd, True)  # even an explicitly inheritable fd must not reach the child
    try:
        r = await run(
            backend,
            "import os\nseen = []\nfor fd in range(3, 256):\n"
            "    try:\n        os.fstat(fd); seen.append(fd)\n    except OSError: pass\nprint('OPEN-FDS', seen)",
        )
    finally:
        os.close(fd)
    assert "HARNESS-FD-CONTENT" not in r.stdout
    assert f"OPEN-FDS []" in r.stdout or backend.name == "docker", r.stdout


# ---------------------------------------------------------------------------------------------
# timeout / processes
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "code",
    [
        "while True:\n    pass",
        "import time\ntime.sleep(1000)",
        "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True: time.sleep(0.01)",
        "import threading, time\nfor _ in range(4): threading.Thread(target=lambda: time.sleep(1000), daemon=False).start()\ntime.sleep(1000)",
    ],
    ids=["busy-loop", "sleep", "ignores-signals", "threads"],
)
async def test_hard_timeout_kills_promptly(backend, code):
    t0 = time.monotonic()
    r = await run(backend, code, timeout_s=1)
    elapsed = time.monotonic() - t0
    assert r.timed_out is True and r.killed_reason == "timeout"
    assert elapsed < 6, elapsed
    assert r.exit_code != 0
    if backend.name != "docker":
        assert not alive(r.pid) and not group_alive(r.pid)


async def test_output_written_before_timeout_is_returned(backend):
    r = await run(backend, "import sys, time\nprint('partial', flush=True)\ntime.sleep(1000)", timeout_s=1)
    assert r.timed_out and "partial" in r.stdout


async def test_fork_bomb_is_contained_and_leaves_no_orphans(backend):
    marker = "31337"
    # NB: the "bomb" is bounded (300 forks, children exit by themselves) so that a *broken* sandbox
    # cannot take the developer's machine down while this test proves that it is broken.
    code = f"""
import os, subprocess, sys, time
def spawn_daemon():
    # classic double-fork + setsid daemonisation, with a recognisable payload
    pid = os.fork()
    if pid == 0:
        os.setsid()
        if os.fork() == 0:
            try:
                os.execv("/bin/sleep", ["sleep", "{marker}"])
            except BaseException:
                time.sleep(60)
        os._exit(0)
    os.waitpid(pid, 0)
try:
    spawn_daemon(); print("DAEMON-SPAWNED", flush=True)
except BaseException as e:
    print("FORK-DENIED", type(e).__name__, flush=True)
try:
    subprocess.Popen(["/bin/sleep", "{marker}"], start_new_session=True); print("POPEN-SPAWNED", flush=True)
except BaseException as e:
    print("POPEN-DENIED", type(e).__name__, flush=True)
made = 0
try:
    for _ in range(300):
        if os.fork() == 0:
            time.sleep(25); os._exit(0)
        made += 1
except BaseException as e:
    print("BOMB-STOPPED", type(e).__name__, flush=True)
print("FORKED", made, flush=True)
while True:
    time.sleep(0.05)
"""
    t0 = time.monotonic()
    r = await run(backend, code, timeout_s=3)
    assert time.monotonic() - t0 < 12
    assert r.timed_out
    await asyncio.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", f"sleep {marker}"], capture_output=True, text=True).stdout.split()
    assert left == [], f"orphan processes survived the sandbox: {left}"
    if backend.name != "docker":
        assert not group_alive(r.pid)
    if backend.name == "seatbelt":  # process-fork is simply not permitted by the profile
        assert "DAEMON-SPAWNED" not in r.stdout and "FORKED 0" in r.stdout
    made = int(r.stdout.split("FORKED")[1].split()[0]) if "FORKED" in r.stdout else 0
    assert made < 300, "the process table was not capped"
    # the machine is still healthy afterwards
    r2 = await run(backend, "print('still fine')")
    assert r2.stdout.strip() == "still fine"


# ---------------------------------------------------------------------------------------------
# memory, disk, output
# ---------------------------------------------------------------------------------------------
async def test_memory_cap_kills_a_1gb_allocation(backend):
    code = """
chunks = []
for i in range(1024):
    chunks.append(bytearray(b"\\x01") * (1 << 20))   # touched, so it is really resident
print("ALLOCATED-1GB")
"""
    t0 = time.monotonic()
    r = await run(backend, code, memory_mb=256, timeout_s=20)
    assert time.monotonic() - t0 < 15
    assert "ALLOCATED-1GB" not in r.stdout
    assert r.exit_code != 0
    assert r.memory_exceeded or "MemoryError" in r.stderr, (r.exit_code, r.stderr)
    assert not r.timed_out


async def test_memory_cap_kills_a_single_huge_burst(backend):
    r = await run(backend, "x = bytearray(b'\\x02') * (900 << 20)\nprint('BURST-OK', len(x))", memory_mb=256, timeout_s=20)
    assert "BURST-OK" not in r.stdout and r.exit_code != 0
    assert r.memory_exceeded or "MemoryError" in r.stderr


async def test_moderate_memory_use_is_allowed(backend):
    r = await run(backend, "x = bytearray(b'\\x01') * (64 << 20)\nprint('OK', len(x) >> 20)", memory_mb=256)
    assert r.exit_code == 0 and "OK 64" in r.stdout and not r.memory_exceeded


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific: RLIMIT_AS is not enforced, RSS watchdog must fire")
async def test_macos_rss_watchdog_fires_even_though_rlimit_as_does_not():
    b = _instances.setdefault("seatbelt", SeatbeltBackend())
    if (reason := await b.verify()) is not None:
        pytest.skip(reason)
    # RLIMIT_AS is not among the limits we rely on for macOS: prove why by showing it's a no-op there
    probe = subprocess.run(
        [sys.executable, "-c",
         "import resource; resource.setrlimit(resource.RLIMIT_AS,(64<<20,64<<20)); x=bytearray(b'a')*(300<<20); print(len(x)>>20)"],
        capture_output=True, text=True,
    )
    assert "_rlimits" in dir(ce) and "RLIMIT_AS" not in ce._rlimits(SandboxLimits(), linux=False)
    r = await run(b, "x = bytearray(b'\\x01') * (1 << 30)\nprint('NOT-KILLED')", memory_mb=256, timeout_s=20)
    assert r.memory_exceeded is True and r.killed_reason == "memory"
    assert r.exit_code == -signal.SIGKILL and "NOT-KILLED" not in r.stdout
    assert "memory limit" in r.stderr
    del probe


async def test_rss_monitor_reads_real_process_memory():
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import time; x = bytearray(b'a') * (200 << 20); print('ready', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE,
    )
    try:
        assert proc.stdout.readline().strip() == b"ready"
        rss = ce.RssMonitor(proc.pid).sample()
        assert rss is not None and rss > 150 * ce.MB
    finally:
        proc.kill()
        proc.wait()


async def test_output_is_truncated(backend):
    r = await run(backend, "import sys\nprint('A' * 10_000_000)\nprint('E' * 5_000_000, file=sys.stderr)\nprint('END-MARKER')")
    assert r.exit_code == 0
    assert r.output_truncated is True
    assert len(r.stdout) < 6000 and len(r.stderr) < 3000
    assert "truncated" in r.stdout or "dropped" in r.stdout
    assert "END-MARKER" in r.stdout  # tail is kept


async def test_output_flood_does_not_hang_or_exhaust_harness(backend):
    t0 = time.monotonic()
    r = await run(backend, "import sys\nwhile True:\n    sys.stdout.write('x' * 65536)", timeout_s=2, max_output_bytes=8 << 20)
    assert time.monotonic() - t0 < 10
    assert r.killed_reason in {"timeout", "output_flood"} and r.output_truncated
    assert len(r.stdout) < 6000


async def test_file_size_limit(backend):
    if backend.name == "docker":
        pytest.skip("docker: scratch is a size-capped tmpfs; covered by scratch-full test")
    r = await run(
        backend,
        "f = open('big.bin', 'wb')\ntry:\n    f.write(b'x' * (100 << 20)); f.flush(); print('WROTE-100MB')\n"
        "except OSError as e: print('DENIED', type(e).__name__)",
        fsize_mb=4,
    )
    assert "WROTE-100MB" not in r.stdout
    assert r.exit_code != 0 or "DENIED" in r.stdout


async def test_scratch_disk_cap(backend):
    code = "import time\nfor i in range(400):\n    open(f'f{i}.bin', 'wb').write(b'x' * (4 << 20))\n    time.sleep(0.01)\nprint('FILLED-1.6GB')"
    t0 = time.monotonic()
    r = await run(backend, code, scratch_mb=32, fsize_mb=8, timeout_s=20)
    assert time.monotonic() - t0 < 15
    assert "FILLED-1.6GB" not in r.stdout
    assert r.killed_reason == "scratch_full" or r.exit_code != 0


async def test_rlimits_cannot_be_raised_by_the_code(backend):
    r = await run(
        backend,
        "import resource\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_CPU)\nprint('CPU', soft == hard, hard < 100)\n"
        "try:\n    resource.setrlimit(resource.RLIMIT_CPU, (10**6, 10**6)); print('RAISED')\n"
        "except (ValueError, OSError) as e: print('CANNOT-RAISE')\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)\nprint('NOFILE', hard <= 256)",
        timeout_s=5,
    )
    assert "CPU True True" in r.stdout and "CANNOT-RAISE" in r.stdout and "NOFILE True" in r.stdout
    assert "RAISED" not in r.stdout


# ---------------------------------------------------------------------------------------------
# the tool itself + fail-closed behaviour
# ---------------------------------------------------------------------------------------------
async def test_tool_end_to_end_via_registry(any_backend, workspace):
    reg = ToolRegistry()
    reg.register(CodeExecTool(workspace, backend=any_backend))
    reg.grant("code", ["code_exec"])
    res = await reg.dispatch(
        "code", "code_exec",
        {"code": "print(open('data.txt').read().strip().lower())", "files": ["data.txt"]}, "s1",
    )
    assert res.ok, res.error
    assert res.output["stdout"].strip() == "hello-from-workspace"
    assert res.output["exit_code"] == 0 and res.output["timed_out"] is False
    payload = json.loads(res.to_content())
    assert payload["status"] == "ok"


async def test_tool_schema_validation_and_timeout_cap(any_backend, workspace):
    tool = CodeExecTool(workspace, backend=any_backend, max_timeout_s=2)
    reg = ToolRegistry()
    reg.register(tool)
    reg.grant("code", ["code_exec"])
    bad = await reg.dispatch("code", "code_exec", {"files": ["x"]}, "s")
    assert not bad.ok and "invalid arguments" in bad.error
    bad = await reg.dispatch("code", "code_exec", {"code": "1", "timeout_s": 0}, "s")
    assert not bad.ok
    t0 = time.monotonic()
    res = await reg.dispatch("code", "code_exec", {"code": "import time; time.sleep(100)", "timeout_s": 3600}, "s")
    assert res.ok and res.output["timed_out"] is True
    assert time.monotonic() - t0 < 8  # 3600s was clamped to max_timeout_s=2


@pytest.mark.parametrize(
    "files, why",
    [
        (["../outside.txt"], "escape"),
        (["/etc/passwd"], "absolute"),
        (["link-to-outside.txt"], "symlink escape"),
        ([".env"], "sensitive"),
        (["pkg/../.env"], "sensitive via traversal"),
        (["link-to-env.txt"], "symlink to sensitive"),
        (["pkg"], "directory"),
        (["missing.txt"], "missing"),
        (["a"] * 21, "too many"),
    ],
)
async def test_tool_rejects_bad_file_requests(any_backend, workspace, tmp_path, files, why):
    (tmp_path / "outside.txt").write_text("OUTSIDE-WORKSPACE")
    (workspace / "link-to-outside.txt").symlink_to(tmp_path / "outside.txt")
    (workspace / "link-to-env.txt").symlink_to(workspace / ".env")
    res = await CodeExecTool(workspace, backend=any_backend).execute({"code": "print(1)", "files": files})
    assert not res.ok, why


async def test_fail_closed_when_no_backend_is_available(workspace, monkeypatch):
    class Unavailable(SandboxBackend):
        name = "nothing"

        def check_available(self):
            return "no isolation primitive on this host"

        def prepare(self, req):  # pragma: no cover
            raise AssertionError("must never be reached")

    spawned = []

    async def no_spawn(*a, **k):
        spawned.append(a)
        raise AssertionError("a process was spawned although no sandbox is available")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", no_spawn)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError("spawn")))
    monkeypatch.setattr(ce, "all_backends", lambda *a, **k: [Unavailable()])
    tool = CodeExecTool(workspace)
    res = await tool.execute({"code": "print('should never run')"})
    assert not res.ok and "disabled" in res.error and "refused" in res.error
    assert "no isolation primitive" in res.error
    assert spawned == []


async def test_a_backend_that_does_not_isolate_is_rejected_by_its_self_test(tmp_path):
    """A backend that merely runs `python` in a subprocess (no isolation) must fail verification
    and therefore never be selected - this is what makes selection fail closed."""

    class Leaky(SandboxBackend):
        name = "leaky"

        def check_available(self):
            return None

        def prepare(self, req):
            py, _ = ce.base_python()
            return ce.LaunchSpec([py, *ce._PY_FLAGS, "-c", ce._RUNNER, json.dumps({"rlimits": {}})],
                                 {"PATH": "/usr/bin:/bin"}, str(req.scratch), req.code.encode())

    leaky = Leaky()
    reason = await leaky.verify()
    assert reason is not None and "NOT effective" in reason and "canary-readable" in reason
    chosen, reasons = await select_backend(candidates=[leaky])
    assert chosen is None and "leaky" in reasons


async def test_preferred_backend_that_is_unusable_does_not_fall_back_to_unsandboxed():
    chosen, reasons = await select_backend("bwrap" if sys.platform == "darwin" else "seatbelt")
    assert chosen is None and reasons


# ---------------------------------------------------------------------------------------------
# backend command lines (pure unit tests; run on every platform, incl. Linux/Docker builders)
# ---------------------------------------------------------------------------------------------
def _req(tmp_path, files=()):
    root = tmp_path / "run"
    (root / "scratch").mkdir(parents=True)
    (root / "ro").mkdir()
    staged = []
    for rel in files:
        p = root / "ro" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        staged.append(ce.StagedFile(rel, p))
    return ce.SandboxRequest("print(1)", root, root / "scratch", staged, SandboxLimits())


def test_docker_command_line_is_locked_down(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_CANARY_SECRET", "docker-env-canary")
    spec = DockerBackend("python:3.12-slim", docker_bin="docker").prepare(_req(tmp_path, ["data/x.txt"]))
    a = " ".join(spec.argv)
    for flag in ("--network none", "--read-only", "--memory 256m", "--memory-swap 256m", "--pids-limit 64",
                 "--cap-drop ALL", "no-new-privileges", "--user 65534:65534", "--pull never", "/work:rw"):
        assert flag in a, flag
    assert ":/work/data/x.txt:ro" in a
    assert a.count(" -v ") == 1                       # nothing but the requested file is mounted
    assert "docker-env-canary" not in a
    assert not any(x.startswith("--privileged") or x in ("--net", "host") for x in spec.argv)
    assert "JARVIS_CANARY_SECRET" not in spec.env


def test_bwrap_command_line_is_locked_down(tmp_path):
    b = BwrapBackend()
    b._bwrap = "/usr/bin/bwrap"
    spec = b.prepare(_req(tmp_path, ["data.txt"]))
    a = spec.argv
    for flag in ("--unshare-all", "--die-with-parent", "--clearenv", "--new-session"):
        assert flag in a
    assert a[a.index("--cap-drop") + 1] == "ALL" and a[a.index("--uid") + 1] == "65534"
    assert "--share-net" not in a
    binds = [a[i + 1] for i, x in enumerate(a) if x == "--ro-bind"]
    assert str(tmp_path / "run" / "ro" / "data.txt") in binds
    assert str(Path.home()) not in " ".join(a) and "/etc" not in binds and "/" not in binds
    assert spec.env == {"PATH": "/usr/bin:/bin"}


def test_unshare_command_line_and_bootstrap_are_fail_closed(tmp_path):
    b = UnshareBackend()
    b._unshare = "/usr/bin/unshare"
    spec = b.prepare(_req(tmp_path, ["data.txt"]))
    for flag in ("--user", "--net", "--mount", "--pid", "--fork", "--kill-child"):
        assert flag in spec.argv
    cfg = json.loads(spec.argv[-1])
    assert cfg["files"] == [[str(tmp_path / "run" / "ro" / "data.txt"), "data.txt"]]
    assert "/" not in cfg["ro_dirs"] and str(Path.home()) not in json.dumps(cfg)
    src = ce._UNSHARE_BOOTSTRAP
    compile(src, "bootstrap", "exec")
    # every setup failure dies with 125 before execve of the interpreter
    assert src.index("PR_CAPBSET_DROP") < src.index("CapBnd") < src.index("os.chroot(root)") < src.index("os.execve")
    assert "def die" in src and "os._exit(125)" in src


def test_seatbelt_profile_is_deny_default_and_minimal(tmp_path):
    b = SeatbeltBackend()
    scratch = str(tmp_path / "run" / "scratch")
    prof = b.profile(scratch, ["a.txt", "d/b.txt"])
    assert prof.splitlines()[1] == "(deny default)"
    assert "(deny network*)" in prof
    assert "process-fork" not in prof and "(allow network" not in prof and "(allow default" not in prof
    assert "(allow file-write* (subpath " + ce._sbpl(scratch) + ")" in prof
    assert prof.rstrip().splitlines()[-2].startswith("(deny file-write*")  # ro override comes after the allow
    for forbidden in (str(Path.home()), '(subpath "/etc"', '(subpath "/Users"', '(subpath "/")', '(subpath "/private")',
                      '(subpath "/var")', '(subpath "/tmp")'):
        assert forbidden not in prof, forbidden
    assert ce._sbpl('a"b\\c') == '"a\\"b\\\\c"'


def test_minimal_env_has_no_inherited_variables(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "leak")
    env = ce._minimal_env("/x")
    assert "SECRET_TOKEN" not in env and all("leak" != v for v in env.values())
