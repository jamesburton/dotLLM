#!/usr/bin/env python3
"""
parity_model_sweep.py — resumable, self-cleaning CPU<->Vulkan parity sweep.

WHY
---
The real-model CPU<->Vulkan end-to-end parity tests are `[SkippableFact]`s that
self-skip when their GGUF fixture is absent. On this box most fixtures are absent,
so "green" meant "silently skipped" for all but a handful of models. Fetching all
of them at once is ~90 GB. So: fetch one (smallest first), test it, record the
result, delete it, move on — unattended, and safe to interrupt at any moment.

USAGE
-----
    # plan only — show the queue, sizes and what would be skipped. No network, no GPU.
    python scripts/parity_model_sweep.py --dry-run

    # run the sweep (resumes automatically; already-terminal models are skipped)
    python scripts/parity_model_sweep.py

    # resume is just: run it again. Ctrl-C at any point is safe.
    python scripts/parity_model_sweep.py

    # one model, or a size ceiling, or a budget
    python scripts/parity_model_sweep.py --only bielik-1.5b-q3_k_m
    python scripts/parity_model_sweep.py --max-size-gb 5
    python scripts/parity_model_sweep.py --max-models 3

    # re-run something already terminal
    python scripts/parity_model_sweep.py --force --only bielik-1.5b-q3_k_m

    # just print the summary table from the state file
    python scripts/parity_model_sweep.py --report

ADDING A MODEL
--------------
Append an entry to scripts/parity_sweep_models.json. That file documents every
field in its own `_comment` block. No code change is needed.

STATE / RESUME
--------------
State lives OUTSIDE the repo, at ~/.dotllm/parity-sweep/state.json (override with
--state), so it survives this worktree being deleted. Every status transition is
written atomically (temp file + os.replace), so a kill -9 mid-fetch leaves a
readable file. Terminal statuses (passed / failed / skipped / timeout / error) are
never re-run without --force; non-terminal ones (pending / fetching / testing /
deferred) are picked back up. A model found in `fetching` or `testing` on startup
is treated as an interrupted run: it is reset to pending and its partial download
is cleaned up.

STORAGE RULES (firm)
--------------------
* Downloads go through `hf download` into the HF hub cache. Never into the repo,
  never into an ad-hoc folder.
* A plain path (`stage_path`) is produced with a HARDLINK to the hub-cache blob,
  never a copy. One physical copy on disk.
* Cleanup removes the snapshot entry AND its backing blob, plus the hardlink, and
  prunes emptied repo dirs — no orphaned blobs.
* A model that was ALREADY on disk before the sweep started is never deleted; the
  state file records `preexisting: true` so that is auditable.

BUILD FRESHNESS (fork issue #341)
---------------------------------
The sweep ALWAYS builds. It runs `dotnet build -c Release` on the test project once, up
front — before any GPU lock is taken — and aborts the whole run if that build fails; the
per-model `dotnet test` then runs without `--no-build` (a near-no-op incremental build).
`--no-build` is refused outright.

This is not tidiness. dotLLM is only half-compiled: Vulkan `.spv` shaders are loaded from
the repo tree at runtime, while the CPU kernels are baked into `DotLLM.Cpu.dll`. Stale
binaries therefore give a LIVE GPU path against a STALE CPU oracle, and a cross-backend
parity test reports a large, specific and completely fictitious divergence. That happened:
`bielik-1.5b-q3_k_m` was recorded as failing at L-inf 14.586 / Jaccard 0.00 against binaries
predating the #311 Q3_K fix; on a fresh build the same test passes at L-inf 1.4818. Note
that checking the committed `.spv` against their `.comp` sources does NOT detect this — it
says nothing about the DLLs. Worktree merges make it easy to hit, because they never touch
the main checkout's `bin/Release`.

Every result also records the commit (+ dirty flag) it was produced against, in the log and
in `state.json` under `build`. `--report` shows it as a column; entries with `?` predate this
and should be re-run with `--force` before being believed.

GPU LOCK
--------
Three other agents share this box's GPU. The lock is acquired per model, held only
for the `dotnet test` invocation, and released in a `finally`. If the lock cannot
be taken within --lock-timeout the model is marked `deferred` (NON-terminal) and
the sweep moves on, so a later run picks it up.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TERMINAL = {"passed", "failed", "skipped", "timeout", "error"}
INTERRUPTED = {"fetching", "testing"}  # only ever seen after a hard kill

DEFAULT_TEST_TIMEOUT = 1800
DEFAULT_FETCH_TIMEOUT = 7200
DEFAULT_LOCK_TIMEOUT = 600
DEFAULT_MIN_FREE_GB = 25.0

LOCK_NAME = "parity-sweep"

_stop = False


def _on_sigint(_signum, _frame):
    global _stop
    if _stop:                       # second Ctrl-C: die now
        raise KeyboardInterrupt
    _stop = True
    print("\n[sweep] interrupt received - finishing current step, then stopping. "
          "Ctrl-C again to abort hard.", flush=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def gb(n: int | float | None) -> str:
    return "?" if not n else f"{n / 1e9:.2f}G"


# ─────────────────────────────────────────────────────────────────────────────
# Repo / path helpers
# ─────────────────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main_checkout_root() -> Path:
    """
    The *primary* worktree. The GPU lock must be one shared directory per box, so a
    worktree-local .gpu-lock would silently let two agents onto the GPU at once.
    """
    try:
        out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                             cwd=repo_root(), capture_output=True, text=True, timeout=30)
        for line in out.stdout.splitlines():
            if line.startswith("worktree "):
                return Path(line[len("worktree "):].strip())
    except Exception:
        pass
    return repo_root()


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


def build_stamp() -> dict:
    """
    Provenance for every result this run records (#341).

    A parity number is only interpretable against the code that produced it. Recording the
    commit + dirty flag next to each result makes a result from an unexpected tree
    self-evident *afterwards*, instead of indistinguishable from a real failure — which is
    exactly how the stale-binary Q3_K "regression" survived scrutiny for a session.
    """
    def git(*args: str) -> str:
        try:
            p = subprocess.run(["git", *args], cwd=repo_root(), capture_output=True,
                               text=True, errors="replace", timeout=30)
            return p.stdout.strip() if p.returncode == 0 else ""
        except Exception:
            return ""

    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit or "unknown",
        "commit_short": (commit or "unknown")[:12],
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        # Uncommitted edits mean `commit` alone does not identify the code under test.
        "dirty": bool(git("status", "--porcelain")),
        "worktree": str(repo_root()),
        "built_at": now(),
    }


def hub_cache() -> Path:
    for var in ("HF_HUB_CACHE",):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_cache_dir(repo: str) -> Path:
    return hub_cache() / ("models--" + repo.replace("/", "--"))


def find_snapshot_file(repo: str, filename: str) -> Path | None:
    """Locate <hub>/models--org--name/snapshots/<rev>/<filename> if already present."""
    snaps = repo_cache_dir(repo) / "snapshots"
    if not snaps.is_dir():
        return None
    for rev in sorted(snaps.iterdir(), key=lambda p: p.name):
        cand = rev / filename
        if cand.exists():          # exists() is False for a dangling link - correct here
            return cand
    return None


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free


# ─────────────────────────────────────────────────────────────────────────────
# State file
# ─────────────────────────────────────────────────────────────────────────────

class State:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {"version": 1, "models": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
                self.data.setdefault("models", {})
            except Exception as e:
                bak = path.with_suffix(f".corrupt-{int(time.time())}.json")
                path.replace(bak)
                print(f"[sweep] state file unreadable ({e}); moved aside to {bak}")

    def get(self, mid: str) -> dict:
        return self.data["models"].setdefault(mid, {"status": "pending"})

    def set(self, mid: str, **kw) -> dict:
        e = self.get(mid)
        e.update(kw)
        e["updated"] = now()
        self.save()
        return e

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic — a kill here leaves the old file intact


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8", errors="replace")

    def __call__(self, msg: str, echo: bool = True) -> None:
        line = f"{now()} {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()
        if echo:
            print(msg, flush=True)

    def detail(self, text: str) -> None:
        self.fh.write(text.rstrip() + "\n")
        self.fh.flush()

    def close(self) -> None:
        try:
            self.fh.close()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face fetch / delete
# ─────────────────────────────────────────────────────────────────────────────

def hf_exe() -> str:
    return os.environ.get("HF_CLI", "hf")


def remote_size(repo: str, filename: str) -> int | None:
    """Live size from the HF API (best effort; used only for queue ordering + disk check)."""
    import urllib.request
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokfile = Path.home() / ".cache" / "huggingface" / "token"
    if not tok and tokfile.exists():
        tok = tokfile.read_text(encoding="utf-8").strip() or None
    hdr = {"Authorization": f"Bearer {tok}"} if tok else {}
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/models/{repo}?blobs=true", headers=hdr)
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
    except Exception:
        return None
    for s in d.get("siblings", []):
        if s.get("rfilename") == filename:
            return s.get("size")
    return None


def hf_download(repo: str, filename: str, log: Log, timeout: int) -> Path:
    """Download one file into the HF hub cache. Returns the snapshot path."""
    cmd = [hf_exe(), "download", repo, filename]
    log(f"    fetch: {' '.join(cmd)}")
    p = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    log.detail(p.stdout)
    log.detail(p.stderr)
    if p.returncode != 0:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        raise RuntimeError(f"hf download exit {p.returncode}: {' | '.join(tail)}")
    # `hf download` prints the resolved local path as its last stdout line.
    for line in reversed([l.strip() for l in p.stdout.splitlines() if l.strip()]):
        cand = Path(line)
        if cand.exists() and cand.is_file():
            return cand
    found = find_snapshot_file(repo, filename)
    if found:
        return found
    raise RuntimeError("hf download reported success but no file landed in the hub cache")


def blob_for(snapshot_file: Path) -> Path | None:
    """
    Resolve the hub-cache blob backing a snapshot entry. huggingface_hub uses a symlink
    where the filesystem allows it and a plain copy otherwise; handle both.
    """
    if snapshot_file.is_symlink():
        return Path(os.path.realpath(snapshot_file))
    blobs = snapshot_file.parent.parent.parent / "blobs"
    if not blobs.is_dir():
        return None
    try:
        st = snapshot_file.stat()
    except OSError:
        return None
    for b in blobs.iterdir():
        try:
            bs = b.stat()
        except OSError:
            continue
        if bs.st_size == st.st_size and (bs.st_ino == st.st_ino or st.st_nlink > 1):
            return b
    return None


def purge_from_cache(repo: str, filename: str, log: Log) -> int:
    """
    Remove one file from the hub cache: the snapshot entry AND its blob. Deliberately
    NOT `hf cache rm <repo>`, which would nuke sibling files of the same repo that the
    user downloaded themselves. Returns bytes reclaimed.
    """
    reclaimed = 0
    snap = find_snapshot_file(repo, filename)
    if snap is not None:
        blob = blob_for(snap)
        try:
            reclaimed = snap.stat().st_size
        except OSError:
            pass
        for victim in (snap, blob):
            if victim is None or not os.path.lexists(victim):
                continue
            try:
                victim.unlink()
            except OSError as e:
                log(f"    warn: could not delete {victim}: {e}")
        # prune the snapshot revision dir if we emptied it
        try:
            rev = snap.parent
            if rev.is_dir() and not any(rev.iterdir()):
                rev.rmdir()
        except OSError:
            pass
    # sweep any leftover .incomplete stubs from a killed download
    blobs = repo_cache_dir(repo) / "blobs"
    if blobs.is_dir():
        for stub in blobs.glob("*.incomplete"):
            try:
                stub.unlink()
                log(f"    removed partial blob {stub.name}")
            except OSError:
                pass
        # if the repo has no blobs left, drop the whole repo dir (no orphans, no refs)
        if not any(blobs.iterdir()):
            try:
                shutil.rmtree(repo_cache_dir(repo))
                log(f"    pruned empty cache repo {repo}")
            except OSError:
                pass
    return reclaimed


def hardlink(src: Path, dst: Path, log: Log) -> bool:
    """
    Hardlink src -> dst (never a copy). Returns True if dst now holds src's data.

    Two Windows traps this deliberately handles:
      * huggingface_hub stores snapshot entries as symlinks with a RELATIVE target
        (`../../blobs/<sha>`). Hardlinking the symlink itself just clones the link, and
        the relative target does not resolve from the staging directory — so resolve to
        the real blob first.
      * `Path.exists()` is False for a dangling symlink, so an existence check alone lets
        a stale broken link survive and then makes os.link fail with WinError 183.
    """
    real = Path(os.path.realpath(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(dst):
        if dst.exists():
            return True
        try:
            dst.unlink()          # dangling link from an earlier run
            log(f"    removed stale dangling link {dst}")
        except OSError as e:
            log(f"    warn: stale link {dst} not removable ({e}); using the env var instead")
            return False
    try:
        os.link(real, dst)
        return True
    except OSError as e:
        log(f"    warn: hardlink {dst} failed ({e}); relying on the env var instead")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# GPU lock
# ─────────────────────────────────────────────────────────────────────────────

def _posix(p: Path) -> str:
    """Git Bash eats Windows backslashes in argv — always hand it forward slashes."""
    return str(p).replace("\\", "/")


def lock_env() -> dict:
    env = dict(os.environ)
    env.setdefault("DOTLLM_GPU_LOCK_DIR", _posix(main_checkout_root() / ".gpu-lock"))
    return env


def bash_exe() -> str:
    """
    Resolve a bash that can actually run gpu-lock.sh. On Windows the `bash` first on PATH
    is often the WSL/System32 shim, which cannot see `C:/...` paths and fails with a
    confusing exit 127; Git's own bash can. Override with DOTLLM_BASH.
    """
    if os.environ.get("DOTLLM_BASH"):
        return os.environ["DOTLLM_BASH"]
    for cand in (r"C:\Program Files\Git\bin\bash.exe",
                 r"C:\Program Files (x86)\Git\bin\bash.exe"):
        if os.path.exists(cand):
            return cand
    return shutil.which("bash") or "bash"


def lock_cmd(*args: str) -> list[str]:
    return [bash_exe(), _posix(main_checkout_root() / "scripts" / "gpu-lock.sh"), *args]


def lock_acquire(reason: str, timeout: int, log: Log) -> bool:
    p = subprocess.run(lock_cmd("acquire", LOCK_NAME, reason, str(timeout)),
                       capture_output=True, text=True, errors="replace",
                       env=lock_env(), timeout=timeout + 120)
    log.detail(p.stdout + p.stderr)
    if p.returncode == 0:
        return True
    if p.returncode != 1:
        # 1 == genuinely contended (the script's timeout path). Anything else is a
        # broken invocation and must not be silently reported as "GPU busy".
        raise RuntimeError(f"gpu-lock.sh acquire failed (exit {p.returncode}): "
                           f"{(p.stderr or p.stdout).strip()[:200]}")
    return False


def lock_release(log: Log) -> None:
    try:
        p = subprocess.run(lock_cmd("release", LOCK_NAME), capture_output=True, text=True,
                           errors="replace", env=lock_env(), timeout=120)
        log.detail(p.stdout + p.stderr)
    except Exception as e:
        log(f"    warn: gpu-lock release failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test execution
# ─────────────────────────────────────────────────────────────────────────────

def kill_stray_testhosts(log: Log) -> None:
    """A hung testhost.exe is a known failure mode here; never leave one behind."""
    if os.name != "nt":
        return
    try:
        p = subprocess.run(["taskkill", "/F", "/IM", "testhost.exe", "/T"],
                           capture_output=True, text=True, errors="replace", timeout=60)
        if p.returncode == 0:
            log("    killed stray testhost.exe")
    except Exception:
        pass


_SUMMARY = re.compile(
    r"(?:Passed|Failed)!\s*-\s*Failed:\s*(\d+),\s*Passed:\s*(\d+),\s*Skipped:\s*(\d+)")


def build_tests(project: Path, configuration: str, log: Log, timeout: int = 3600) -> None:
    """
    Build the test project ONCE, up front, before the GPU lock is ever taken.

    #341: this sweep used to run `dotnet test --no-build`, which silently tested whatever
    binaries happened to be lying in `bin/<config>` from some earlier session. That is far
    worse here than an ordinary stale-build failure, because dotLLM is only *half* compiled:
    the Vulkan `.spv` shaders are loaded from the repo tree at runtime while the CPU kernels
    are baked into `DotLLM.Cpu.dll`. Stale binaries therefore give a live GPU path against a
    stale CPU oracle, and the cross-backend parity test reports a large, specific, entirely
    fictitious divergence. It cost a session once (bielik-1.5b-q3_k_m, L-inf 14.586 vs 1.4818
    on a fresh build).

    Raises on failure: a sweep that cannot build must abort loudly, never produce numbers.
    """
    cmd = ["dotnet", "build", str(project), "-c", configuration, "--nologo", "-v", "minimal"]
    log(f"[build] {' '.join(cmd)}")
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=repo_root(), capture_output=True, text=True,
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"dotnet build exceeded {timeout}s - refusing to test stale binaries")
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    log.detail(out)
    if p.returncode != 0:
        tail = [l for l in out.splitlines() if l.strip()][-6:]
        raise RuntimeError("dotnet build FAILED (exit "
                           f"{p.returncode}) - refusing to test stale binaries:\n  "
                           + "\n  ".join(tail))
    log(f"[build] ok in {int(time.time() - started)}s")


def run_test(project: Path, filt: str, extra_env: dict, timeout: int, log: Log,
             configuration: str) -> tuple[str, str]:
    """
    Returns (status, detail). status in passed/failed/skipped/timeout/error.

    #341: deliberately NO `--no-build`. `build_tests` already did the expensive compile before
    the GPU lock was taken, so this incremental build is a near-no-op — but it is the thing
    that guarantees the assemblies under test match the working tree, including any file that
    changed mid-sweep.
    """
    cmd = ["dotnet", "test", str(project), "-c", configuration, "--filter", filt,
           "--nologo", "-v", "minimal"]
    env = dict(os.environ)
    env.update(extra_env)
    env["DOTLLM_PARITY_SWEEP"] = "1"
    log(f"    test:  --filter {filt}")
    started = time.time()
    try:
        p = subprocess.run(cmd, cwd=repo_root(), capture_output=True, text=True,
                           errors="replace", env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        log.detail((e.stdout or "") if isinstance(e.stdout, str) else "")
        kill_stray_testhosts(log)
        return "timeout", f"no result after {timeout}s"
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    log.detail(out)
    elapsed = int(time.time() - started)

    m = _SUMMARY.search(out)
    if m:
        failed, passed, skipped = (int(x) for x in m.groups())
        detail = f"passed={passed} failed={failed} skipped={skipped} in {elapsed}s"
        if failed:
            return "failed", detail
        if passed == 0:
            # Every fixture-gated test here is a [SkippableFact]; zero passed means the
            # test self-skipped, which is NOT parity coverage and must not read as green.
            return "skipped", detail + " (test self-skipped - fixture not seen)"
        return "passed", detail
    if "No test matches the given testcase filter" in out:
        return "error", "filter matched no tests - the model list's `filter` is stale"
    if p.returncode != 0:
        tail = [l for l in out.splitlines() if l.strip()][-3:]
        return "error", f"dotnet test exit {p.returncode}: {' | '.join(tail)[:400]}"
    return "error", f"unparseable dotnet test output (exit 0) after {elapsed}s"


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

GLYPH = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP", "timeout": "TIME",
         "error": "ERR ", "deferred": "DEFR", "pending": "----",
         "fetching": "fetc", "testing": "test"}


def build_tag(entry: dict) -> str:
    """
    The commit a recorded result was produced against (#341). `?` means the entry predates
    provenance recording — i.e. it may have come from binaries of unknown vintage and should
    be re-run before it is believed.
    """
    b = entry.get("build")
    if not isinstance(b, dict) or not b.get("commit_short"):
        return "?"
    return b["commit_short"][:8] + ("+d" if b.get("dirty") else "")


def report(models: list[dict], state: State) -> None:
    print()
    print("  status  size    cached  built-at   model                          detail")
    print("  ------  ------  ------  ---------  -----------------------------  " + "-" * 40)
    tally: dict[str, int] = {}
    unprovenanced = 0
    for m in models:
        e = state.get(m["id"])
        st = e.get("status", "pending")
        tally[st] = tally.get(st, 0) + 1
        if e.get("preexisting"):
            pre = "kept"          # was the user's before the sweep — never deleted
        elif e.get("owned"):
            pre = "held" if st not in TERMINAL else "del"   # ours: on disk / cleaned up
        else:
            pre = "-"
        tag = build_tag(e)
        if tag == "?" and st in TERMINAL:
            unprovenanced += 1
        print(f"  {GLYPH.get(st, st):6} {gb(e.get('size_bytes') or m.get('size_bytes')):>6}"
              f"  {pre:>6}  {tag:<9}  {m['id']:<29}  {str(e.get('detail', ''))[:40]}")
    print("  " + "-" * 100)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    if unprovenanced:
        print(f"  NOTE: {unprovenanced} terminal result(s) carry no build stamp - they were "
              f"recorded before #341 and the binaries they used are unknown. Re-run with "
              f"--force before trusting them.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(Path(__file__).with_name("parity_sweep_models.json")))
    ap.add_argument("--state", default=str(Path.home() / ".dotllm" / "parity-sweep" / "state.json"))
    ap.add_argument("--log-dir", default=None, help="default: <state dir>/logs")
    ap.add_argument("--only", action="append", default=None, help="model id (repeatable)")
    ap.add_argument("--force", action="store_true", help="re-run models already in a terminal state")
    ap.add_argument("--retry-errors", action="store_true",
                    help="also re-run models whose last outcome was `error` or `timeout`")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no network, no GPU")
    ap.add_argument("--report", action="store_true", help="print the summary table and exit")
    ap.add_argument("--max-models", type=int, default=None)
    ap.add_argument("--max-size-gb", type=float, default=None, help="skip anything larger")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    ap.add_argument("--lock-timeout", type=int, default=DEFAULT_LOCK_TIMEOUT)
    ap.add_argument("--fetch-timeout", type=int, default=DEFAULT_FETCH_TIMEOUT)
    ap.add_argument("--test-timeout", type=int, default=None, help="override per-model timeout")
    ap.add_argument("--configuration", default="Release")
    # #341: `--no-build` is gone, not merely defaulted off. It is kept only as a rejected
    # flag so that a stale script/runbook/muscle-memory invocation fails loudly with the
    # reason, rather than argparse's generic "unrecognized arguments".
    ap.add_argument("--no-build", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--keep", action="store_true", help="do not delete fetched models")
    args = ap.parse_args()

    if args.no_build:
        print("--no-build is refused: this sweep tested stale binaries once and reported a "
              "fictitious Q3_K parity regression (fork issue #341). The build is incremental "
              "and runs before the GPU lock is taken.", file=sys.stderr)
        return 2

    cfg = json.loads(Path(args.models).read_text(encoding="utf-8"))
    project = repo_root() / cfg.get("test_project",
                                    "tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj")
    state = State(Path(args.state))
    log_dir = Path(args.log_dir) if args.log_dir else Path(args.state).parent / "logs"
    log = Log(log_dir / f"sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")

    models = cfg["models"]
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m["id"] in wanted]
        missing = wanted - {m["id"] for m in models}
        if missing:
            print(f"unknown model id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    if args.report:
        report(cfg["models"], state)
        return 0

    # ── recover interrupted runs ────────────────────────────────────────────
    for m in cfg["models"]:
        e = state.get(m["id"])
        if e.get("status") in INTERRUPTED:
            was = e["status"]
            # `fetching` may have left a torn/partial download -> purge it and refetch.
            # `testing` means the bytes are complete -> keep them, just re-run the test.
            if was == "fetching" and e.get("owned") and m.get("repo"):
                purge_from_cache(m["repo"], m["file"], log)
                sp = m.get("stage_path")
                if sp and expand(sp).exists():
                    try:
                        expand(sp).unlink()
                    except OSError:
                        pass
                state.set(m["id"], status="pending", owned=False,
                          detail="partial download discarded after an interrupted run")
            else:
                state.set(m["id"], status="pending",
                          detail="reset after an interrupted run (download kept)")
            log(f"[recover] {m['id']}: status={was} from a killed run -> pending")

    # ── resolve sizes + presence, then order smallest-first ────────────────
    plan = []
    for m in models:
        e = state.get(m["id"])
        size = e.get("size_bytes") or m.get("size_bytes")
        present = None
        if m.get("available", True):
            sp = m.get("stage_path")
            if sp and expand(sp).exists():
                present = expand(sp)
            else:
                present = find_snapshot_file(m["repo"], m["file"])
        if present is not None:
            size = present.stat().st_size
        elif not args.dry_run and m.get("available", True):
            size = remote_size(m["repo"], m["file"]) or size
        plan.append((size or 0, m, present))
    plan.sort(key=lambda t: t[0])

    if args.max_size_gb is not None:
        plan = [t for t in plan if t[0] <= args.max_size_gb * 1e9]

    log(f"[sweep] queue: {len(plan)} model(s), smallest-first; "
        f"state={state.path}; log={log.path}")
    log(f"[sweep] hub cache={hub_cache()}  gpu-lock={lock_env()['DOTLLM_GPU_LOCK_DIR']}")

    if args.dry_run:
        for size, m, present in plan:
            e = state.get(m["id"])
            mark = "cached" if present else "fetch "
            if not m.get("available", True):
                mark = "N/A   "
            log(f"  {gb(size):>7}  {mark}  {m['id']:<30} status={e.get('status', 'pending')}")
        # Only count what a real run would actually fetch: not already on disk, not
        # already terminal, not unavailable.
        total = sum(s for s, m, p in plan
                    if p is None and m.get("available", True)
                    and (args.force or state.get(m["id"]).get("status", "pending") not in TERMINAL))
        log(f"[sweep] would download {total / 1e9:.1f} GB "
            f"(~{total / 1e9 / 0.0117 / 3600:.1f} h at 11.7 MB/s unauthenticated, "
            f"~{total / 1e9 / 0.030 / 3600:.1f} h at 30 MB/s with a token)")
        report(cfg["models"], state)
        return 0

    # ── build ONCE, before any GPU lock is taken, and abort loudly on failure (#341) ──
    stamp = build_stamp()
    log(f"[sweep] code under test: {stamp['commit_short']} ({stamp['branch']})"
        f"{'  *** DIRTY WORKING TREE ***' if stamp['dirty'] else ''}  in {stamp['worktree']}")
    try:
        build_tests(project, args.configuration, log)
    except Exception as ex:                                   # noqa: BLE001
        log(f"[sweep] ABORT: {ex}")
        log("[sweep] no models were tested - a sweep must never report parity numbers "
            "produced by binaries it did not just build.")
        log.close()
        return 1
    stamp["built_at"] = now()

    signal.signal(signal.SIGINT, _on_sigint)
    done = 0

    for size, m, present in plan:
        if _stop:
            log("[sweep] stopping (interrupted); rerun to resume")
            break
        if args.max_models is not None and done >= args.max_models:
            log(f"[sweep] --max-models {args.max_models} reached")
            break

        mid = m["id"]
        e = state.get(mid)
        st = e.get("status", "pending")
        retryable = {"error", "timeout"} if args.retry_errors else set()
        if st in TERMINAL and not args.force and st not in retryable:
            log(f"[skip] {mid}: already {st}")
            continue
        if not m.get("available", True):
            state.set(mid, status="skipped", preexisting=False, size_bytes=size,
                      detail=m.get("skip_reason", "marked unavailable"))
            log(f"[SKIP] {mid}: {m.get('skip_reason', 'unavailable')[:110]}")
            continue

        done += 1
        log(f"[{done}] {mid}  ({gb(size)})  {m['repo']}/{m['file']}")
        # `owned` is sticky in the state file: a model WE fetched on an earlier run that
        # ended non-terminally (deferred / interrupted) is on disk but is NOT the user's,
        # so resume must reuse it without re-downloading AND still delete it at the end.
        # Without this, a deferred model would either be re-downloaded every run or get
        # silently promoted to "preexisting" and never cleaned up.
        owned = bool(e.get("owned"))
        preexisting = present is not None and not owned
        fetched_path = present
        stage = expand(m["stage_path"]) if m.get("stage_path") else None
        staged_by_us = bool(e.get("staged_by_us"))
        deferred = False

        try:
            # ── disk guard ────────────────────────────────────────────────
            if fetched_path is None:
                need = size * 1.15 + args.min_free_gb * 1e9
                have = free_bytes(hub_cache())
                if have < need:
                    msg = (f"insufficient disk: {have / 1e9:.1f} GB free, need "
                           f"{need / 1e9:.1f} GB (model {gb(size)} + {args.min_free_gb} GB reserve)")
                    log(f"    ABORT: {msg}")
                    state.set(mid, status="pending", detail=msg)
                    log("[sweep] aborting the sweep - free space and rerun to resume")
                    break

                state.set(mid, status="fetching", size_bytes=size, preexisting=False,
                          owned=True, repo=m["repo"], file=m["file"])
                owned = True
                fetched_path = hf_download(m["repo"], m["file"], log, args.fetch_timeout)
                size = fetched_path.stat().st_size
                log(f"    fetched {gb(size)} -> {fetched_path}")
            elif preexisting:
                log(f"    cached already ({fetched_path}) - will NOT be deleted")
            else:
                log(f"    reusing our earlier download ({fetched_path}) - still ours to delete")

            # ── stage a plain path via hardlink (never a copy) ────────────
            if stage is not None and not stage.exists():
                staged_by_us = hardlink(fetched_path, stage, log)

            # ── test under the GPU lock ───────────────────────────────────
            state.set(mid, status="testing", size_bytes=size, preexisting=preexisting,
                      owned=owned, staged_by_us=staged_by_us)
            if not lock_acquire(f"parity sweep: {mid}", args.lock_timeout, log):
                deferred = True
                state.set(mid, status="deferred", owned=owned, staged_by_us=staged_by_us,
                          detail=f"gpu lock unavailable after {args.lock_timeout}s - rerun to retry")
                log(f"    DEFER: gpu lock busy; keeping the download for the next run")
                continue

            try:
                env = {m["env"]: str(fetched_path)} if m.get("env") else {}
                timeout = args.test_timeout or m.get("timeout_sec", DEFAULT_TEST_TIMEOUT)
                status, detail = run_test(project, m["filter"], env, timeout, log,
                                          args.configuration)
            finally:
                lock_release(log)

            state.set(mid, status=status, detail=detail, size_bytes=size,
                      preexisting=preexisting, owned=owned, repo=m["repo"], file=m["file"],
                      build=stamp)
            log(f"    {status.upper()}: {detail}  [{stamp['commit_short']}"
                f"{'+dirty' if stamp['dirty'] else ''}]")

        except subprocess.TimeoutExpired:
            state.set(mid, status="timeout", detail=f"fetch exceeded {args.fetch_timeout}s",
                      size_bytes=size, preexisting=preexisting, owned=owned, build=stamp)
            log(f"    TIMEOUT during fetch")
        except KeyboardInterrupt:
            deferred = True   # keep the bytes; the next run resumes from here
            state.set(mid, status="pending", detail="interrupted", owned=owned,
                      staged_by_us=staged_by_us)
            log("    interrupted")
            raise
        except Exception as ex:                      # noqa: BLE001 — one bad model must not end the run
            state.set(mid, status="error", detail=f"{type(ex).__name__}: {ex}"[:400],
                      size_bytes=size, preexisting=preexisting, owned=owned, build=stamp)
            log(f"    ERROR: {type(ex).__name__}: {ex}")
        finally:
            # ── self-clean: only what WE brought in, and only once we are done ──
            if owned and not deferred and not args.keep:
                # lexists, not exists: a dangling link must still be removed.
                if staged_by_us and stage is not None and os.path.lexists(stage):
                    try:
                        stage.unlink()
                        log(f"    removed staged hardlink {stage}")
                    except OSError as ex:
                        log(f"    warn: could not remove {stage}: {ex}")
                got = purge_from_cache(m["repo"], m["file"], log)
                if got:
                    log(f"    cleaned {gb(got)} from the hub cache "
                        f"({free_bytes(hub_cache()) / 1e9:.0f} GB free)")

    report(cfg["models"], state)
    log(f"[sweep] state: {state.path}")
    log(f"[sweep] log:   {log.path}")
    log.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[sweep] aborted - rerun to resume", flush=True)
        sys.exit(130)
