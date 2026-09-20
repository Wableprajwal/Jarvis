"""`code_exec`: run untrusted, LLM-generated Python with real isolation.

Threat model: the code is hostile or simply wrong. It must not be able to (a) reach the network,
(b) read anything except the few workspace files the task needs, (c) write anywhere except a
throwaway scratch dir, (d) see the harness's environment (which holds credentials), (e) outlive its
timeout, (f) eat the host's memory/disk/process table.

Design: a `SandboxBackend` strategy chosen at runtime, and FAIL CLOSED - if no backend can prove
its isolation on this host, the tool returns an error. It never falls back to plain `subprocess`.

    DockerBackend   --network none --read-only --memory --pids-limit --cap-drop ALL --user nobody
    SeatbeltBackend macOS `sandbox-exec` deny-default profile (no network, no fork, FS allow-list)
    BwrapBackend    Linux bubblewrap: net/pid/ipc/user namespaces, ro-bind allow-list rootfs
    UnshareBackend  Linux `unshare` user+mount+net+pid namespaces and a chroot built from an
                    allow-list (used when bwrap is not installed)

Every backend runs the *same* trampoline inside the sandbox (`_RUNNER`): it applies
`resource.setrlimit` limits (soft == hard so the code cannot raise them), then executes the code
read from stdin. Supervision (hard timeout + process-group kill, RSS watchdog, output capping,
scratch-size cap) is shared and lives in `_supervise`; it does not depend on the sandbox itself
being well behaved.
"""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from app.tools.paths import resolve_inside
from app.tools.registry import Tool, ToolResult

IS_DARWIN = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

MB = 1024 * 1024
MAX_CODE_CHARS = 100_000
MAX_FILES = 20
MAX_FILE_BYTES = 2 * MB
MAX_TOTAL_FILE_BYTES = 10 * MB



# ------------------------------------------------------------------------------------------
# data model
# ------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float = 10.0
    memory_mb: int = 256
    cpu_s: int | None = None          # default: timeout + 1
    fsize_mb: int = 16                # largest single file the code may write
    scratch_mb: int = 64              # total scratch the code may fill
    nofile: int = 256
    max_output_bytes: int = 64 * MB   # kill the process if it floods more than this to stdout+stderr
    stdout_chars: int = 5000          # what is handed back (head + tail kept)
    stderr_chars: int = 2500

    @property
    def cpu_seconds(self) -> int:
        return self.cpu_s if self.cpu_s is not None else int(self.timeout_s) + 1


@dataclass
class StagedFile:
    rel: str          # POSIX path relative to the sandbox cwd (== workspace-relative path)
    src: Path         # private pristine copy on the host (never inside the workspace)


@dataclass
class SandboxRequest:
    code: str
    root: Path                  # private per-run dir (0700) - removed afterwards
    scratch: Path               # root/scratch: becomes the cwd (rw)
    files: list[StagedFile]     # exposed read-only at scratch-relative path `rel`
    limits: SandboxLimits


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool = False
    memory_exceeded: bool = False
    output_truncated: bool = False
    killed_reason: str | None = None   # timeout | memory | output_flood | scratch_full | None
    backend: str = ""
    pid: int | None = None
    duration_s: float = 0.0


class SandboxError(RuntimeError):
    """The sandbox could not be set up. The code was NOT run."""


@dataclass
class LaunchSpec:
    argv: list[str]
    env: dict[str, str]
    cwd: str
    stdin_bytes: bytes
    popen_kwargs: dict = field(default_factory=dict)
    kill_hook: object | None = None           # async callable(): extra teardown (e.g. `docker kill`)
    post_hook: object | None = None           # async callable(SandboxResult): adjust result (e.g. OOM)
    watch_rss: bool = True                    # backend has no kernel memory cap of its own -> poll
    watch_scratch: bool = True


# ------------------------------------------------------------------------------------------
# in-sandbox trampoline (runs INSIDE the sandbox, before the user's code)
# ------------------------------------------------------------------------------------------
_RUNNER = r'''
import sys, os, io, json, types
_cfg = json.loads(sys.argv[1])
_code = sys.stdin.buffer.read().decode("utf-8", "replace")
sys.stdin = io.TextIOWrapper(io.BytesIO(b""), encoding="utf-8")
try:
    import resource as _r
    for _name, _val in _cfg.get("rlimits", {}).items():
        _res = getattr(_r, _name, None)
        if _res is None:
            continue
        _soft, _hard = _r.getrlimit(_res)
        _new = _val if _hard == _r.RLIM_INFINITY else min(_val, _hard)
        _r.setrlimit(_res, (_new, _new))
except Exception as _e:  # fail closed: never run user code without its limits
    sys.stderr.write("sandbox setup failed: cannot apply resource limits: %r\n" % (_e,))
    sys.exit(125)
if _cfg.get("alarm_s"):
    import signal as _sig
    _sig.alarm(int(_cfg["alarm_s"]))  # in-sandbox wall-clock backstop (default action: terminate)
sys.argv = ["<sandbox>"]
_main = types.ModuleType("__main__")
_main.__dict__["__builtins__"] = __builtins__
sys.modules["__main__"] = _main
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
del _cfg
try:
    _obj = compile(_code, "<code>", "exec")
except SyntaxError:
    import traceback
    traceback.print_exc(limit=0)
    sys.exit(1)
try:
    exec(_obj, _main.__dict__)
except SystemExit:
    raise
except BaseException:
    import traceback
    _t, _v, _tb = sys.exc_info()
    traceback.print_exception(_t, _v, _tb.tb_next)
    sys.exit(1)
'''

_PY_FLAGS = ["-I", "-S", "-B", "-X", "utf8"]  # isolated, no site/venv packages, no .pyc writes, UTF-8


def _minimal_env(home: str, tmp: str | None = None) -> dict[str, str]:
    """The ONLY environment the child gets. Nothing is inherited from the harness."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": home,
        "TMPDIR": tmp or home,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONIOENCODING": "utf-8",
        "TERM": "dumb",
    }


def _rlimits(limits: SandboxLimits, *, linux: bool) -> dict[str, int]:
    cfg = {
        "RLIMIT_CPU": limits.cpu_seconds,
        "RLIMIT_FSIZE": limits.fsize_mb * MB,
        "RLIMIT_NOFILE": limits.nofile,
        "RLIMIT_CORE": 0,
    }
    if linux:  # RLIMIT_AS/DATA/NPROC are unreliable or meaningless on macOS; see SeatbeltBackend
        cfg["RLIMIT_DATA"] = limits.memory_mb * MB
        cfg["RLIMIT_AS"] = max(limits.memory_mb * 4, 1024) * MB
        cfg["RLIMIT_NPROC"] = _count_user_processes() + 64
    return cfg


def _count_user_processes() -> int:
    uid = os.geteuid()
    n = 0
    with contextlib.suppress(OSError):
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                try:
                    if entry.stat().st_uid == uid:
                        n += 1
                except OSError:
                    pass
    return max(n, 64)


def _runner_cfg(limits: SandboxLimits, *, linux: bool) -> str:
    return json.dumps({"rlimits": _rlimits(limits, linux=linux), "alarm_s": int(limits.timeout_s) + 3})


def _runner_argv(python: str, limits: SandboxLimits, *, linux: bool) -> list[str]:
    return [python, *_PY_FLAGS, "-c", _RUNNER, _runner_cfg(limits, linux=linux)]


def base_python() -> tuple[str, str]:
    """(interpreter, prefix) of the *base* interpreter (not a venv), fully resolved. The sandbox
    runs it with -I -S so the venv's site-packages (and anything in the harness) are not visible."""
    exe = getattr(sys, "_base_executable", None) or sys.executable
    exe = os.path.realpath(exe)
    return exe, os.path.realpath(sys.base_prefix)


# ------------------------------------------------------------------------------------------
# supervision: timeout, process-group kill, memory watchdog, output capping
# ------------------------------------------------------------------------------------------
class _Capture:
    """Keeps the head and tail of a stream; counts everything."""

    def __init__(self, head: int = 96 * 1024, tail: int = 32 * 1024):
        self._head_cap, self._tail_cap = head, tail
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self._tail_len = 0
        self.total = 0

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self._head_cap - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if chunk:
            self.tail.append(chunk)
            self._tail_len += len(chunk)
            while self._tail_len - len(self.tail[0]) >= self._tail_cap:
                self._tail_len -= len(self.tail.popleft())

    @property
    def dropped(self) -> int:
        return max(0, self.total - len(self.head) - self._tail_len)

    def text(self, limit_chars: int) -> tuple[str, bool]:
        head = bytes(self.head).decode("utf-8", "replace")
        tail = b"".join(self.tail).decode("utf-8", "replace")
        text = head + (f"\n...[{self.dropped} bytes dropped]...\n" if self.dropped else "") + tail
        if len(text) <= limit_chars:
            return text, self.dropped > 0
        keep_head = int(limit_chars * 0.6)
        keep_tail = limit_chars - keep_head
        omitted = len(text) - limit_chars
        return text[:keep_head] + f"\n...[truncated {omitted} chars]...\n" + text[-keep_tail:], True


class _LibProc:
    """macOS: resident size of one pid via libproc (cheap enough to poll every few ms)."""

    _fn = None

    class _TaskInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64),
            ("total_user", ctypes.c_uint64), ("total_system", ctypes.c_uint64),
            ("threads_user", ctypes.c_uint64), ("threads_system", ctypes.c_uint64),
            ("policy", ctypes.c_int32), ("faults", ctypes.c_int32), ("pageins", ctypes.c_int32),
            ("cow_faults", ctypes.c_int32), ("messages_sent", ctypes.c_int32),
            ("messages_received", ctypes.c_int32), ("syscalls_mach", ctypes.c_int32),
            ("syscalls_unix", ctypes.c_int32), ("csw", ctypes.c_int32), ("threadnum", ctypes.c_int32),
            ("numrunning", ctypes.c_int32), ("priority", ctypes.c_int32),
        ]

    @classmethod
    def rss(cls, pid: int) -> int | None:
        if cls._fn is None:
            try:
                lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                lib.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
                lib.proc_pidinfo.restype = ctypes.c_int
                cls._fn = lib.proc_pidinfo
            except OSError:
                cls._fn = False
        if not cls._fn:
            return None
        info = cls._TaskInfo()
        n = cls._fn(pid, 4, 0, ctypes.byref(info), ctypes.sizeof(info))  # PROC_PIDTASKINFO
        if n != ctypes.sizeof(info):
            return None
        return int(info.resident_size)


def _ps_tree_rss(root_pid: int) -> int | None:
    """Portable fallback: sum RSS over the process subtree via `ps` (kB -> bytes)."""
    try:
        out = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True, timeout=2
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            pid, ppid, kb = map(int, parts)
            kids.setdefault(ppid, []).append(pid)
            rss[pid] = kb * 1024
    if root_pid not in rss:
        return None
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, ()))
    return total


def _proc_tree_rss(root_pid: int) -> int | None:
    """Linux: sum VmRSS over the subtree rooted at root_pid using /proc."""
    page = os.sysconf("SC_PAGE_SIZE")
    kids: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    try:
        entries = [e.name for e in os.scandir("/proc") if e.name.isdigit()]
    except OSError:
        return None
    for name in entries:
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                stat = fh.read().decode("utf-8", "replace")
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
            with open(f"/proc/{name}/statm", "rb") as fh:
                pages = int(fh.read().split()[1])
        except (OSError, ValueError, IndexError):
            continue
        pid = int(name)
        kids.setdefault(ppid, []).append(pid)
        rss[pid] = pages * page
    if root_pid not in rss:
        return None
    total, stack = 0, [root_pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack.extend(kids.get(p, ()))
    return total


class RssMonitor:
    """Portable resident-memory reader for a process tree. macOS has no usable RLIMIT_AS /
    RLIMIT_DATA (setrlimit fails or is ignored, and jetsam limits need an entitlement), so the cap
    is enforced by sampling + SIGKILL (the Seatbelt profile forbids fork, so the tree is one pid and
    libproc gives an exact, microsecond-cheap reading)."""

    def __init__(self, pid: int):
        self.pid = pid
        self._last_ps = 0.0
        self._ps_rss: int | None = None

    def sample(self) -> int | None:
        if IS_DARWIN:
            own = _LibProc.rss(self.pid)
            if own is not None:
                return own
            now = time.monotonic()  # libproc unavailable: slower but portable fallback
            if now - self._last_ps > 0.05:
                self._last_ps = now
                self._ps_rss = _ps_tree_rss(self.pid)
            return self._ps_rss
        return _proc_tree_rss(self.pid)


class _Verdict:
    """First reason a run was killed (thread-safe: the watchdog thread and the event loop race)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reason: str | None = None

    def set(self, reason: str) -> bool:
        with self._lock:
            if self.reason is None:
                self.reason = reason
                return True
            return False


class _Watchdog(threading.Thread):
    """Enforces the wall-clock deadline and the memory cap from a dedicated thread, so neither
    depends on how busy the harness's event loop is. Polls every ~1 ms on macOS (libproc reads are
    cheap): a process can hold more than its cap for at most about a millisecond."""

    def __init__(self, pid: int, deadline: float, memory_cap: int, verdict: _Verdict):
        super().__init__(name=f"sandbox-watchdog-{pid}", daemon=True)
        self._pid, self._deadline, self._cap, self._verdict = pid, deadline, memory_cap, verdict
        self._monitor = RssMonitor(pid)
        self._stop_evt = threading.Event()
        self._interval = 0.001 if IS_DARWIN else 0.01

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            reason = None
            if time.monotonic() >= self._deadline:
                reason = "timeout"
            else:
                used = self._monitor.sample()
                if used is not None and used > self._cap:
                    reason = "memory"
            if reason:
                if self._verdict.set(reason):
                    _kill_group(self._pid)
                return
            self._stop_evt.wait(self._interval)


def _dir_usage(path: str, cap_entries: int = 20_000) -> tuple[int, int]:
    """(bytes, entries) under path without following symlinks; bounded work."""
    size = entries = 0
    stack = [path]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    entries += 1
                    if entries > cap_entries:
                        return size, entries
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            size += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return size, entries


def _kill_group(pid: int | None) -> None:
    if not pid:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, signal.SIGKILL)  # child was started with setsid: pgid == pid
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.kill(pid, signal.SIGKILL)


async def _call_hook(hook) -> None:
    if hook is None:
        return
    with contextlib.suppress(Exception):
        res = hook()
        if asyncio.iscoroutine(res):
            await res


async def _supervise(spec: LaunchSpec, req: SandboxRequest, backend_name: str) -> SandboxResult:
    limits = req.limits
    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *spec.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.cwd,
            env=spec.env,                 # exact dict: nothing inherited from the harness
            start_new_session=True,       # own process group -> killpg reaches every descendant
            close_fds=True,
            **spec.popen_kwargs,
        )
    except OSError as exc:
        raise SandboxError(f"could not launch sandbox ({backend_name}): {exc}") from exc

    out, err = _Capture(), _Capture()
    verdict = _Verdict()
    flood_cap = limits.max_output_bytes
    # No kernel-enforced cap and no container to do it for us -> watchdog thread (deadline + RSS).
    watchdog = None
    if spec.watch_rss:
        watchdog = _Watchdog(proc.pid, t0 + limits.timeout_s, limits.memory_mb * MB, verdict)
        watchdog.start()

    async def kill(reason: str) -> None:
        verdict.set(reason)
        _kill_group(proc.pid)
        await _call_hook(spec.kill_hook)

    async def pump(stream: asyncio.StreamReader, cap: _Capture) -> None:
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                return
            cap.feed(chunk)
            if out.total + err.total > flood_cap and verdict.reason is None:
                await kill("output_flood")

    async def feed_stdin() -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            proc.stdin.write(spec.stdin_bytes)
            await proc.stdin.drain()
        with contextlib.suppress(Exception):
            proc.stdin.close()

    pumps = [
        asyncio.ensure_future(pump(proc.stdout, out)),
        asyncio.ensure_future(pump(proc.stderr, err)),
        asyncio.ensure_future(feed_stdin()),
    ]
    waiter = asyncio.ensure_future(proc.wait())
    poll = 0.01 if IS_DARWIN else 0.03
    next_scratch_check = t0 + 0.3
    scratch_cap = limits.scratch_mb * MB
    killed_at: float | None = None
    try:
        while not waiter.done():
            await asyncio.wait({waiter}, timeout=poll)
            if waiter.done():
                break
            now = time.monotonic()
            if verdict.reason is not None:  # a kill was issued: keep insisting, but never hang on it
                killed_at = killed_at or now
                _kill_group(proc.pid)
                if now - killed_at > 5:
                    break
                continue
            if now - t0 > limits.timeout_s + 0.05:  # backstop if the watchdog thread is unavailable
                await kill("timeout")
            elif spec.watch_scratch and now >= next_scratch_check:
                next_scratch_check = now + 0.3
                size, entries = await asyncio.to_thread(_dir_usage, str(req.scratch))
                if size > scratch_cap or entries > 20_000:
                    await kill("scratch_full")
    finally:
        if watchdog is not None:
            watchdog.stop()
        # Also on cancellation: nothing this run started may survive it.
        if proc.returncode is None:
            _kill_group(proc.pid)
            await _call_hook(spec.kill_hook)
        _kill_group(proc.pid)  # sweep stragglers that outlived the leader (same process group)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.gather(*pumps, return_exceptions=True), timeout=5)
        for p in pumps:
            p.cancel()
        if not waiter.done():
            waiter.cancel()

    reason = verdict.reason
    stdout, t1 = out.text(limits.stdout_chars)
    stderr, t2 = err.text(limits.stderr_chars)
    notes = {
        "timeout": f"[sandbox] killed: exceeded the {limits.timeout_s:g}s time limit",
        "memory": f"[sandbox] killed: exceeded the {limits.memory_mb} MB memory limit",
        "output_flood": "[sandbox] killed: produced too much output",
        "scratch_full": f"[sandbox] killed: filled more than {limits.scratch_mb} MB of scratch space",
    }
    if reason:
        stderr = (stderr + "\n" if stderr else "") + notes[reason]
    result = SandboxResult(
        stdout=stdout, stderr=stderr, exit_code=proc.returncode,
        timed_out=reason == "timeout", memory_exceeded=reason == "memory",
        output_truncated=t1 or t2, killed_reason=reason, backend=backend_name,
        pid=proc.pid, duration_s=time.monotonic() - t0,
    )
    if spec.post_hook is not None:
        with contextlib.suppress(Exception):
            res = spec.post_hook(result)
            if asyncio.iscoroutine(res):
                await res
    return result


# ------------------------------------------------------------------------------------------
# backend interface
# ------------------------------------------------------------------------------------------
class SandboxBackend(ABC):
    name: str = "abstract"
    #: how each requirement is enforced by this backend (used in docs/diagnostics)
    enforcement: dict[str, str] = {}

    def __init__(self) -> None:
        self._verified: str | bool | None = False  # False = not probed; None = ok; str = reason

    @abstractmethod
    def check_available(self) -> str | None:
        """Cheap static check. None if the prerequisites exist, else the reason they do not."""

    @abstractmethod
    def prepare(self, req: SandboxRequest) -> LaunchSpec: ...

    async def run(self, req: SandboxRequest) -> SandboxResult:
        return await _supervise(self.prepare(req), req, self.name)

    async def verify(self) -> str | None:
        """Functional self-test, cached. Runs real code in the sandbox and checks that isolation
        actually holds on this host (a host that ships the binary but forbids its use - e.g.
        unprivileged user namespaces disabled - must not be selected)."""
        if self._verified is not False:
            return self._verified  # type: ignore[return-value]
        reason = self.check_available()
        if reason is None:
            try:
                reason = await _self_test(self)
            except Exception as exc:  # noqa: BLE001 - backend probe: any failure means this sandbox is unusable here
                reason = f"self-test crashed: {type(exc).__name__}: {exc}"
        self._verified = reason
        return reason


_PROBE_CODE = r'''
import socket, sys
problems = []
try:
    socket.create_connection(("127.0.0.1", 9), timeout=1); problems.append("net")
except OSError:
    pass
try:
    open(%(canary)r).read(); problems.append("canary-readable")
except OSError:
    pass
try:
    with open("ok.txt", "w") as f: f.write("x")
except OSError:
    problems.append("scratch-not-writable")
print("PROBE:" + ",".join(problems))
'''


async def _self_test(backend: SandboxBackend) -> str | None:
    """Prove on THIS host that the backend isolates: a canary file outside the scope must be
    unreadable, the network unreachable, and scratch writable."""
    with tempfile.TemporaryDirectory(prefix="jarvis-canary-", dir=os.path.realpath(tempfile.gettempdir())) as d:
        canary = os.path.join(d, "canary.txt")
        Path(canary).write_text("jarvis-canary")
        try:
            result = await execute_sandboxed(
                backend, _PROBE_CODE % {"canary": canary}, {}, SandboxLimits(timeout_s=20, memory_mb=256)
            )
        except SandboxError as exc:
            return str(exc)
    if result.exit_code != 0 or "PROBE:" not in result.stdout:
        return f"self-test did not run cleanly (exit={result.exit_code}): {(result.stderr or result.stdout)[:300]}"
    problems = result.stdout.split("PROBE:", 1)[1].strip()
    if problems:
        return f"self-test showed isolation is NOT effective ({problems}); refusing to use this backend"
    return None


# ------------------------------------------------------------------------------------------
# Seatbelt (macOS)
# ------------------------------------------------------------------------------------------
def _sbpl(path: str) -> str:
    if any(ord(c) < 32 for c in path):
        raise SandboxError("control character in sandbox path")
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ancestors(path: str) -> list[str]:
    parts = Path(path).parts
    return [str(Path(*parts[: i + 1])) for i in range(len(parts) - 1)] if len(parts) > 1 else []


class SeatbeltBackend(SandboxBackend):
    """macOS: `sandbox-exec` with a generated *deny-default* profile.

    network   : `(deny default)` + explicit `(deny network*)`. socket() itself succeeds (the kernel
                does not gate it) but connect/bind/sendto/DNS (mach-lookup) are all denied.
    fs read   : only the interpreter's prefix, system libs, the scratch dir, and stat() on the
                ancestor directories needed for path resolution. /etc, $HOME, the repo: EPERM.
    fs write  : only the scratch dir; the exposed workspace files (and their parent dirs) get an
                explicit later `deny file-write*` so they are read-only even inside scratch.
    processes : `process-fork` is NOT allowed, so no fork bombs / daemons / orphans are possible;
                process-exec only for the interpreter.
    memory    : RLIMIT_AS/DATA are not enforced on macOS -> RssMonitor polls libproc and SIGKILLs.
    """

    name = "seatbelt"
    enforcement = {
        "network": "sandbox-exec profile: deny default + deny network*",
        "filesystem": "sandbox-exec allow-list of read paths; write only scratch; ro files denied write",
        "timeout": "asyncio watchdog + killpg(SIGKILL) on own session",
        "memory": "RSS watchdog via libproc (RLIMIT_AS unreliable on macOS)",
        "user": "unchanged unless root (then setuid nobody); limited by profile instead",
        "env": "exact allow-list env dict",
        "processes": "process-fork denied in profile",
    }

    def __init__(self, python_executable: str | None = None):
        super().__init__()
        if python_executable:
            exe = os.path.realpath(python_executable)
            self._python, self._prefix = exe, os.path.realpath(os.path.join(os.path.dirname(exe), ".."))
        else:
            self._python, self._prefix = base_python()

    def check_available(self) -> str | None:
        if not IS_DARWIN:
            return "not macOS"
        if not os.access("/usr/bin/sandbox-exec", os.X_OK):
            return "/usr/bin/sandbox-exec not found"
        if not os.path.exists(self._python):
            return f"interpreter {self._python} not found"
        return None

    def profile(self, scratch: str, ro_rels: list[str]) -> str:
        prefix = self._prefix
        read_paths = ["/usr/lib", "/System", "/usr/share/zoneinfo", "/usr/share/locale", "/private/var/db/timezone", prefix, scratch]
        literals = ["/dev/null", "/dev/urandom", "/dev/random", "/dev/dtracehelper", "/dev/zero"]
        meta = {"/", "/var", "/tmp", "/private", "/opt"}
        for p in (prefix, scratch, *read_paths):
            meta.update(_ancestors(p))
        lines = [
            "(version 1)",
            "(deny default)",
            f"(allow process-exec (subpath {_sbpl(prefix)}))",
            "(allow sysctl-read)",
            '(allow file-read-data (literal "/"))',
            "(allow file-read-metadata " + " ".join(f"(literal {_sbpl(m)})" for m in sorted(meta)) + ")",
            "(allow file-read* " + " ".join(
                [f"(subpath {_sbpl(p)})" for p in read_paths] + [f"(literal {_sbpl(p)})" for p in literals]
            ) + ")",
        ]
        # Homebrew interpreters link against /opt/homebrew/opt/<pkg>/lib (openssl, libffi, xz, ...)
        for hb in ("/opt/homebrew", "/usr/local"):
            if prefix.startswith(hb + "/"):
                h = re.escape(hb)
                lines.append(
                    f'(allow file-read* (regex #"^{h}/(opt|Cellar)/[^/]+(/[^/]+)?/lib(/.*)?$"))'
                )
                lines.append(f'(allow file-read-metadata (regex #"^{h}/(opt|Cellar)(/[^/]+)?(/[^/]+)?$"))')
        lines.append(
            f"(allow file-write* (subpath {_sbpl(scratch)}) (literal \"/dev/null\") (literal \"/dev/dtracehelper\"))"
        )
        deny: list[str] = []
        for rel in ro_rels:
            deny.append(os.path.join(scratch, rel))
            parent = os.path.dirname(rel)
            while parent:
                deny.append(os.path.join(scratch, parent))
                parent = os.path.dirname(parent)
        if deny:  # later rule wins: these override the scratch write allowance above
            lines.append("(deny file-write* " + " ".join(f"(literal {_sbpl(p)})" for p in dict.fromkeys(deny)) + ")")
        lines.append("(deny network*)")
        return "\n".join(lines) + "\n"

    def prepare(self, req: SandboxRequest) -> LaunchSpec:
        scratch = str(req.scratch)
        for f in req.files:  # materialise the read-only copies inside cwd
            dst = req.scratch / f.rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f.src, dst)
            os.chmod(dst, 0o444)
        prof = self.profile(scratch, [f.rel for f in req.files])
        argv = ["/usr/bin/sandbox-exec", "-p", prof, *_runner_argv(self._python, req.limits, linux=False)]
        kwargs: dict = {}
        if os.geteuid() == 0:  # restricted OS user where possible
            kwargs.update(user=-2 & 0xFFFFFFFF, group=-2 & 0xFFFFFFFF, extra_groups=[])
        return LaunchSpec(argv, _minimal_env(scratch), scratch, req.code.encode(), kwargs)


# ------------------------------------------------------------------------------------------
# Docker
# ------------------------------------------------------------------------------------------
class DockerBackend(SandboxBackend):
    """Short-lived container: --network none --read-only --memory --pids-limit --cap-drop ALL
    --user nobody, workspace files bind-mounted :ro into a tmpfs cwd. Used only when the daemon
    answers *and* the image is already present (we never pull at run time)."""

    name = "docker"
    enforcement = {
        "network": "--network none: no route off the container's own private loopback (empty routing table)",
        "filesystem": "--read-only rootfs; tmpfs /work (scratch) and a 16 MiB noexec tmpfs /tmp, both private to "
                      "the container and destroyed with it; files bind-mounted :ro; nothing else mounted",
        "timeout": "watchdog + `docker kill` + killpg of the CLI",
        "memory": "--memory/--memory-swap (cgroup); OOM detected via inspect",
        "user": "--user 65534:65534, --cap-drop ALL, no-new-privileges",
        "env": "nothing is inherited from the harness; the container sees its image's declared ENV plus an "
               "explicit -e allow-list",
        "processes": "--pids-limit",
    }

    def __init__(self, image: str = "python:3.12-slim", docker_bin: str | None = None):
        super().__init__()
        self.image = image
        self._docker = docker_bin or shutil.which("docker") or "docker"

    def check_available(self) -> str | None:
        if shutil.which(self._docker) is None and not os.path.exists(self._docker):
            return "docker binary not found"
        try:
            r = subprocess.run(
                [self._docker, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=6
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"docker not usable: {type(exc).__name__}"
        if r.returncode != 0 or not r.stdout.strip():
            return "docker daemon is not running"
        try:
            r = subprocess.run(
                [self._docker, "image", "inspect", self.image], capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return "docker image inspect failed"
        if r.returncode != 0:
            return f"image {self.image} is not present locally (run `docker pull {self.image}`)"
        return None

    def prepare(self, req: SandboxRequest) -> LaunchSpec:
        lim = req.limits
        name = f"jarvis-sbx-{uuid.uuid4().hex[:12]}"
        mem = f"{lim.memory_mb}m"
        argv = [
            self._docker, "run", "-i", "--name", name, "--pull", "never",
            "--network", "none", "--read-only",
            "--memory", mem, "--memory-swap", mem, "--pids-limit", "64", "--cpus", "1",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--ulimit", f"nofile={lim.nofile}:{lim.nofile}", "--ulimit", f"cpu={lim.cpu_seconds}:{lim.cpu_seconds}",
            "--ulimit", "core=0:0",
            "--tmpfs", f"/work:rw,nosuid,nodev,size={lim.scratch_mb}m,mode=1777",
            "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=16m,mode=1777",
            "-w", "/work", "--hostname", "sandbox",
            "-e", "HOME=/work", "-e", "TMPDIR=/tmp", "-e", "LANG=C.UTF-8", "-e", "PATH=/usr/local/bin:/usr/bin:/bin",
        ]
        for f in req.files:
            argv += ["-v", f"{f.src}:/work/{f.rel}:ro"]
        argv += [self.image, "python", *_PY_FLAGS, "-c", _RUNNER, _runner_cfg(lim, linux=False)]
        docker = self._docker

        async def kill_hook() -> None:
            await asyncio.to_thread(
                subprocess.run, [docker, "kill", name], capture_output=True, timeout=10
            )

        async def post_hook(result: SandboxResult) -> None:
            try:
                r = await asyncio.to_thread(
                    subprocess.run, [docker, "inspect", "-f", "{{.State.OOMKilled}}", name],
                    capture_output=True, text=True, timeout=10,
                )
                if r.stdout.strip() == "true" and not result.timed_out:
                    result.memory_exceeded = True
                    result.killed_reason = "memory"
                    result.stderr += f"\n[sandbox] killed: exceeded the {lim.memory_mb} MB memory limit"
            finally:
                await asyncio.to_thread(subprocess.run, [docker, "rm", "-f", name], capture_output=True, timeout=15)

        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}  # only for the docker CLI itself
        for k in ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "HOME"):  # CLI needs to find the daemon
            if k in os.environ:
                env[k] = os.environ[k]
        return LaunchSpec(
            argv, env, str(req.root), req.code.encode(), {}, kill_hook=kill_hook, post_hook=post_hook,
            watch_rss=False, watch_scratch=False,
        )


# ------------------------------------------------------------------------------------------
# Linux: bubblewrap
# ------------------------------------------------------------------------------------------
_HOST_RO_DIRS = ("/usr", "/lib", "/lib32", "/lib64", "/libx32", "/bin", "/sbin")


class BwrapBackend(SandboxBackend):
    """Linux bubblewrap: fresh user/net/pid/ipc/uts/cgroup namespaces, an allow-list root
    (ro-bind of /usr + the interpreter prefix only), a tmpfs /tmp, the scratch dir at /work, and
    the exposed files ro-bound on top. `--die-with-parent` + pid namespace: nothing outlives us."""

    name = "bwrap"
    enforcement = {
        "network": "--unshare-net (no interfaces, not even loopback up)",
        "filesystem": "empty root; ro-bind of /usr and interpreter prefix; /work = scratch; files --ro-bind",
        "timeout": "watchdog + killpg; pid namespace + --die-with-parent",
        "memory": "RLIMIT_DATA/AS (kernel) + RSS watchdog",
        "user": "--uid/--gid 65534 in a user namespace, --cap-drop ALL",
        "env": "--clearenv + explicit --setenv allow-list",
        "processes": "RLIMIT_NPROC + pid namespace",
    }

    def __init__(self, python_executable: str | None = None):
        super().__init__()
        if python_executable:
            exe = os.path.realpath(python_executable)
            self._python, self._prefix = exe, os.path.realpath(os.path.join(os.path.dirname(exe), ".."))
        else:
            self._python, self._prefix = base_python()
        self._bwrap = shutil.which("bwrap")

    def check_available(self) -> str | None:
        if not IS_LINUX:
            return "not Linux"
        if not self._bwrap:
            return "bwrap not installed"
        return None

    def _rootfs_args(self) -> list[str]:
        args: list[str] = []
        for d in _HOST_RO_DIRS:
            if os.path.islink(d):
                args += ["--symlink", os.readlink(d), d]
            elif os.path.isdir(d):
                args += ["--ro-bind", d, d]
        prefix_top = "/" + self._prefix.lstrip("/").split("/", 1)[0]
        if prefix_top not in _HOST_RO_DIRS:
            args += ["--ro-bind", self._prefix, self._prefix]
            if not self._python.startswith(self._prefix + "/"):
                args += ["--ro-bind", self._python, self._python]
        return args

    def prepare(self, req: SandboxRequest) -> LaunchSpec:
        lim = req.limits
        argv = [
            self._bwrap or "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--clearenv", "--cap-drop", "ALL", "--uid", "65534", "--gid", "65534",
            *self._rootfs_args(),
            "--dev", "/dev", "--tmpfs", "/tmp",
            "--bind", str(req.scratch), "/work",
        ]
        for f in req.files:
            argv += ["--ro-bind", str(f.src), f"/work/{f.rel}"]
        argv += ["--chdir", "/work"]
        for k, v in _minimal_env("/work", "/tmp").items():
            argv += ["--setenv", k, v]
        argv += ["--", *_runner_argv(self._python, lim, linux=True)]
        return LaunchSpec(argv, {"PATH": "/usr/bin:/bin"}, str(req.scratch), req.code.encode(), {})


# ------------------------------------------------------------------------------------------
# Linux: unshare (no bwrap installed)
# ------------------------------------------------------------------------------------------
_UNSHARE_BOOTSTRAP = r'''
import ctypes, ctypes.util, json, os, sys
libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
MS_RDONLY, MS_NOSUID, MS_REMOUNT, MS_BIND, MS_REC, MS_PRIVATE = 1, 2, 32, 4096, 16384, 1 << 18
def die(msg):
    sys.stderr.write("sandbox setup failed: %s\n" % msg)
    os._exit(125)
def mount(src, dst, fstype=None, flags=0, data=None):
    r = libc.mount(src and src.encode(), dst.encode(), fstype and fstype.encode(), ctypes.c_ulong(flags),
                   data and data.encode())
    if r != 0:
        die("mount %s -> %s: errno %d" % (src, dst, ctypes.get_errno()))
def bind_ro(src, dst):
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            open(dst, "wb").close()
    mount(src, dst, None, MS_BIND | MS_REC)
    # keep the mount flags the kernel locked in (statvfs flags share MS_* values) and add read-only
    keep = os.statvfs(dst).f_flag & ~MS_RDONLY
    mount(None, dst, None, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | keep)
cfg = json.loads(sys.argv[1])
root = cfg["newroot"]
mount(None, "/", None, MS_REC | MS_PRIVATE)
mount("tmpfs", root, "tmpfs", MS_NOSUID, "size=1m,mode=0755")
for d in cfg["ro_dirs"]:
    if os.path.islink(d):
        os.symlink(os.readlink(d), root + d)
    elif os.path.isdir(d):
        bind_ro(d, root + d)
for d in cfg["extra_ro"]:
    bind_ro(d, root + d)
os.makedirs(root + "/dev", exist_ok=True)
for n in ("null", "zero", "urandom", "random"):
    bind_ro("/dev/" + n, root + "/dev/" + n)
os.makedirs(root + "/tmp", exist_ok=True)
mount("tmpfs", root + "/tmp", "tmpfs", MS_NOSUID, "size=16m,mode=1777")
os.makedirs(root + "/work", exist_ok=True)
mount(cfg["scratch"], root + "/work", None, MS_BIND)
for src, rel in cfg["files"]:
    bind_ro(src, root + "/work/" + rel)
# Drop the whole capability bounding set: after execve the interpreter (uid 0 only *inside* the
# user namespace) then has NO capabilities, so it cannot chroot/mount/ptrace its way out.
PR_SET_NO_NEW_PRIVS, PR_CAPBSET_DROP = 38, 24
for cap in range(0, 64):
    libc.prctl(PR_CAPBSET_DROP, cap, 0, 0, 0)
if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
    die("no_new_privs failed")
with open("/proc/self/status") as fh:
    bnd = [l.split()[1] for l in fh if l.startswith("CapBnd:")]
if bnd != ["0000000000000000"]:
    die("capability bounding set not empty: %r" % (bnd,))
os.chroot(root)
os.chdir("/work")
os.execve(cfg["argv"][0], cfg["argv"], cfg["env"])
'''


class UnshareBackend(SandboxBackend):
    """Linux without bwrap: `unshare --user --map-root-user --net --mount --pid --fork` then a small
    bootstrap builds an allow-list chroot (tmpfs root + read-only binds), drops every capability and
    execs the interpreter. Any setup failure exits 125 *before* user code runs (fail closed).
    Plain `unshare -n` alone would give no FS scoping, so it is deliberately never used by itself."""

    name = "unshare"
    enforcement = {
        "network": "unshare --net (empty network namespace)",
        "filesystem": "mount namespace + chroot into tmpfs root with ro binds of /usr, prefix; /work=scratch",
        "timeout": "watchdog + killpg; --kill-child + pid namespace",
        "memory": "RLIMIT_DATA/AS (kernel) + RSS watchdog",
        "user": "user namespace (ns-root with ALL capabilities dropped, no_new_privs)",
        "env": "execve with explicit env dict",
        "processes": "RLIMIT_NPROC + pid namespace",
    }

    def __init__(self, python_executable: str | None = None):
        super().__init__()
        if python_executable:
            exe = os.path.realpath(python_executable)
            self._python, self._prefix = exe, os.path.realpath(os.path.join(os.path.dirname(exe), ".."))
        else:
            self._python, self._prefix = base_python()
        self._unshare = shutil.which("unshare")

    def check_available(self) -> str | None:
        if not IS_LINUX:
            return "not Linux"
        if not self._unshare:
            return "unshare not installed"
        return None

    def prepare(self, req: SandboxRequest) -> LaunchSpec:
        lim = req.limits
        extra: list[str] = []
        top = "/" + self._prefix.lstrip("/").split("/", 1)[0]
        if top not in _HOST_RO_DIRS:
            extra.append(self._prefix)
        newroot = req.root / "newroot"
        newroot.mkdir(exist_ok=True)
        inner = _runner_argv(self._python, lim, linux=True)
        cfg = {
            "newroot": str(newroot),
            "ro_dirs": [d for d in _HOST_RO_DIRS if os.path.lexists(d)],
            "extra_ro": extra,
            "scratch": str(req.scratch),
            "files": [[str(f.src), f.rel] for f in req.files],
            "argv": inner,
            "env": _minimal_env("/work", "/tmp"),
        }
        argv = [
            self._unshare or "unshare", "--user", "--map-root-user", "--net", "--mount", "--pid", "--ipc",
            "--uts", "--fork", "--kill-child",
            self._python, *_PY_FLAGS, "-c", _UNSHARE_BOOTSTRAP, json.dumps(cfg),
        ]
        return LaunchSpec(argv, {"PATH": "/usr/bin:/bin"}, str(req.scratch), req.code.encode(), {})


# ------------------------------------------------------------------------------------------
# selection + orchestration
# ------------------------------------------------------------------------------------------
def all_backends(docker_image: str = "python:3.12-slim", python_executable: str | None = None) -> list[SandboxBackend]:
    """Candidates in preference order."""
    return [
        DockerBackend(docker_image),
        SeatbeltBackend(python_executable),
        BwrapBackend(python_executable),
        UnshareBackend(python_executable),
    ]


async def select_backend(
    preferred: str | None = None,
    candidates: list[SandboxBackend] | None = None,
) -> tuple[SandboxBackend | None, dict[str, str]]:
    """First candidate that passes its functional self-test. Returns (backend|None, {name: reason
    unavailable}). None means: refuse to execute code."""
    cands = candidates if candidates is not None else all_backends()
    if preferred and preferred != "auto":
        cands = [c for c in cands if c.name == preferred]
    reasons: dict[str, str] = {}
    for c in cands:
        why = await c.verify()
        if why is None:
            return c, reasons
        reasons[c.name] = why
    return None, reasons


def resolve_workspace_files(root: Path, paths: list[str]) -> dict[str, Path]:
    """Validate workspace-relative paths -> {rel: absolute resolved path}. Raises ValueError."""
    if len(paths) > MAX_FILES:
        raise ValueError(f"too many files (max {MAX_FILES})")
    root = root.resolve()
    out: dict[str, Path] = {}
    total = 0
    for raw in paths:
        rel = PurePosixPath(raw) if isinstance(raw, str) else PurePosixPath()
        target = resolve_inside(root, raw)  # PathRefused is a ValueError: same contract as before
        size = target.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"file too large (> {MAX_FILE_BYTES} bytes): {raw!r}")
        total += size
        if total > MAX_TOTAL_FILE_BYTES:
            raise ValueError("files too large in total")
        out[str(rel)] = target
    return out


def _rmtree(path: Path) -> None:
    def onerror(func, p, _exc):  # code may have chmod'ed things to 000
        with contextlib.suppress(OSError):
            os.chmod(os.path.dirname(p), 0o700)
            os.chmod(p, 0o700)
            func(p)

    shutil.rmtree(path, onerror=onerror)
    if path.exists():  # second pass after fixing modes top-down
        for dirpath, dirnames, _ in os.walk(path):
            for d in dirnames:
                with contextlib.suppress(OSError):
                    os.chmod(os.path.join(dirpath, d), 0o700)
        shutil.rmtree(path, ignore_errors=True)


async def execute_sandboxed(
    backend: SandboxBackend,
    code: str,
    files: dict[str, Path],
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Stage a private run dir, run `code` in `backend`, always clean up. `files` maps a
    cwd-relative name to a host file (already validated by `resolve_workspace_files`)."""
    limits = limits or SandboxLimits()
    base = os.path.realpath(tempfile.gettempdir())
    root = Path(tempfile.mkdtemp(prefix="jarvis-sbx-", dir=base))
    try:
        os.chmod(root, 0o700)
        scratch = root / "scratch"
        ro = root / "ro"
        scratch.mkdir(mode=0o700)
        ro.mkdir(mode=0o755)
        staged: list[StagedFile] = []
        for rel, src in files.items():
            dst = ro / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)  # snapshot: later changes/symlink swaps in the workspace can't leak
            os.chmod(dst, 0o444)
            staged.append(StagedFile(rel, dst))
        return await backend.run(SandboxRequest(code, root, scratch, staged, limits))
    finally:
        try:
            await asyncio.to_thread(_rmtree, root)
        except asyncio.CancelledError:  # cancelled mid-cleanup: still never leave the scratch behind
            _rmtree(root)
            raise


# ------------------------------------------------------------------------------------------
# the tool
# ------------------------------------------------------------------------------------------
class CodeExecTool(Tool):
    name = "code_exec"
    description = (
        "Run a short Python 3 script in an isolated sandbox and return its stdout, stderr and exit code. "
        "The sandbox has NO network, no environment variables/secrets, and only the workspace files you "
        "list in `files` (visible read-only at the same relative path, which is also the working "
        "directory). You may write temporary files in the working directory; they are discarded. "
        "Only the Python standard library is available. Default limits: 10 s, 256 MB."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 3 source to execute"},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_FILES,
                "description": "Workspace-relative file paths to expose read-only to the code",
            },
            "timeout_s": {"type": "integer", "minimum": 1, "description": "Wall-clock limit in seconds (capped)"},
        },
        "required": ["code"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace_root: str | Path = ".",
        *,
        backend: SandboxBackend | None = None,
        preferred_backend: str | None = None,
        default_timeout_s: int = 10,
        max_timeout_s: int = 30,
        memory_mb: int = 256,
        docker_image: str = "python:3.12-slim",
    ):
        self._root = Path(workspace_root).resolve()
        self._backend = backend
        self._preferred = preferred_backend
        self._docker_image = docker_image
        self._default_timeout = default_timeout_s
        self._max_timeout = max_timeout_s
        self._memory_mb = memory_mb
        self._unavailable: dict[str, str] | None = None

    async def _get_backend(self) -> SandboxBackend | None:
        if self._backend is None and self._unavailable is None:
            backend, reasons = await select_backend(
                self._preferred, all_backends(self._docker_image)
            )
            self._backend, self._unavailable = backend, reasons
        return self._backend

    async def execute(self, arguments: dict) -> ToolResult:
        code = arguments["code"]
        if len(code) > MAX_CODE_CHARS:
            return ToolResult.failure(f"code too long (> {MAX_CODE_CHARS} characters)")
        try:
            files = resolve_workspace_files(self._root, list(arguments.get("files") or []))
        except ValueError as exc:
            return ToolResult.failure(str(exc))
        backend = await self._get_backend()
        if backend is None:
            detail = "; ".join(f"{k}: {v}" for k, v in (self._unavailable or {}).items())
            return ToolResult.failure(
                "code_exec is disabled: no isolating sandbox backend is available on this host, and "
                f"running untrusted code without isolation is refused ({detail})"
            )
        timeout = min(int(arguments.get("timeout_s") or self._default_timeout), self._max_timeout)
        limits = SandboxLimits(timeout_s=float(timeout), memory_mb=self._memory_mb)
        try:
            res = await execute_sandboxed(backend, code, files, limits)
        except SandboxError as exc:
            return ToolResult.failure(f"sandbox error: {exc}")
        return ToolResult.success(
            {
                "stdout": res.stdout,
                "stderr": res.stderr,
                "exit_code": res.exit_code,
                "timed_out": res.timed_out,
                "memory_exceeded": res.memory_exceeded,
                "output_truncated": res.output_truncated,
                "sandbox": res.backend,
            }
        )
