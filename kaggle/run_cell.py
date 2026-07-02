"""run_cell.py -- Worker script for one Track-M MoTE grid cell on Kaggle.

Looks up the cell config in grid_manifest.json, runs mote_train.py -> mote_eval.py ->
push_results.py in sequence.  With --dry-run, prints the exact commands instead.
With --resume, pulls the prior checkpoint from the kaggle-results branch first.

Usage
-----
    python kaggle/run_cell.py --cell-id c1 \
        [--manifest kaggle/grid_manifest.json] \
        [--repo-dir .] \
        [--results-remote https://github.com/jamesburton/dotLLM] \
        [--resume] [--dry-run]

Environment
-----------
    RESULTS_REMOTE : GitHub repo URL for result push (overridden by --results-remote)
    GITHUB_PAT     : GitHub PAT with repo write access (forwarded to push_results.py)
    GH_PAT         : fallback GitHub PAT if GITHUB_PAT is not set
"""

import argparse
import json
import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd: list, dry_run: bool = False, **kw) -> None:
    """Print (and optionally execute) a command list."""
    print("  $ " + " ".join(str(c) for c in cmd))
    if not dry_run:
        subprocess.run(cmd, check=True, **kw)


# ---------------------------------------------------------------------------
# GPU-memory detection and teacher-device resolution
# ---------------------------------------------------------------------------


def _detect_gpu_info() -> "tuple[int, float | None]":
    """Return (device_count, vram_gb_for_gpu0) or (0, None) if CUDA unavailable."""
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return 0, None
        count = torch.cuda.device_count()
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return count, gb
    except Exception:
        return 0, None


def _resolve_teacher_device(requested: str) -> str:
    """Resolve 'auto' to the best teacher device for the available hardware.

    Priority (auto mode only):
      1. >=2 GPUs detected  -> 'cuda:1' (T4x2: student on cuda:0, teacher on cuda:1;
                               fast on-GPU KD, each card holds one model)
      2. 1 GPU, VRAM >= 20 GB -> 'cuda' (L4/A100: both fit on one card)
      3. 1 GPU, VRAM < 20 GB  -> 'cpu'  (single T4/P100: teacher must be on CPU)
      4. No CUDA              -> 'cpu'

    Explicit values ('cuda', 'cuda:1', 'cpu', etc.) are returned unchanged.
    """
    if requested != "auto":
        return requested
    count, gb = _detect_gpu_info()
    if count == 0 or gb is None:
        print(
            "[run_cell] No CUDA GPU detected -- teacher on cpu "
            "(auto; no VRAM to measure)"
        )
        return "cpu"
    if count >= 2:
        print(
            f"[run_cell] {count} GPUs detected -> teacher on cuda:1 "
            f"(student cuda:0) for fast on-GPU KD"
        )
        return "cuda:1"
    # Single GPU path
    if gb >= 20.0:
        print(
            f"[run_cell] GPU {gb:.1f}GB >= 20GB -> teacher on cuda (fast on-GPU KD)"
        )
        return "cuda"
    print(
        f"[run_cell] GPU {gb:.1f}GB < 20GB, single GPU -> teacher on cpu "
        f"(KD slower). Use T4x2 or L4 for fast on-GPU KD."
    )
    return "cpu"


def _run_train(cmd: list, dry_run: bool = False) -> None:
    """Run mote_train.py, retrying once with --teacher-device cpu on CUDA OOM.

    Output is streamed to stdout in real time while being captured so that
    CUDA OOM signatures can be detected on failure.
    """
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry_run:
        return

    # Determine whether a retry makes sense (no point if teacher is already cpu)
    teacher_already_cpu = (
        "--teacher-device" in cmd
        and cmd[cmd.index("--teacher-device") + 1] == "cpu"
    )

    # Stream output live *and* capture it for OOM detection on failure
    captured: list[str] = []
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None  # always set when stdout=PIPE
    for line in proc.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    proc.wait()

    if proc.returncode == 0:
        return

    output = "".join(captured)
    is_oom = "OutOfMemoryError" in output or "CUDA out of memory" in output

    if is_oom and not teacher_already_cpu:
        print(
            "\n[run_cell] CUDA OOM detected -- retrying once with --teacher-device cpu"
        )
        retry_cmd = list(cmd)
        if "--teacher-device" in retry_cmd:
            idx = retry_cmd.index("--teacher-device")
            retry_cmd[idx + 1] = "cpu"
        else:
            retry_cmd.extend(["--teacher-device", "cpu"])
        print("  $ " + " ".join(str(c) for c in retry_cmd))
        subprocess.run(retry_cmd, check=True)
    else:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _do_writecheck(results_remote: str, pat: str, dry_run: bool = False) -> bool:
    """Push a tiny sentinel file to the kaggle-results branch to verify write access.

    Returns True on success, False on failure.  In dry-run mode, prints the commands
    (with the PAT redacted to ``***``) without executing them and returns True.
    """
    import datetime
    import shutil
    import tempfile

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    auth_url = results_remote
    if results_remote.startswith("https://") and pat:
        auth_url = results_remote.replace("https://", f"https://{pat}@", 1)

    if dry_run:
        redacted = auth_url.replace(pat, "***") if pat else auth_url
        print(
            f"[run_cell] --writecheck-only (dry-run): would push "
            f"results/_writecheck/{ts}.txt to kaggle-results"
        )
        print(f"  $ git clone --depth 1 --branch kaggle-results {redacted} <tmpdir>")
        print(f"  $ # create results/_writecheck/{ts}.txt")
        print(f"  $ git commit -m 'writecheck: {ts}'")
        print(f"  $ git push {redacted} HEAD:kaggle-results")
        return True

    if not pat:
        print(
            "[run_cell] --writecheck-only: GITHUB_PAT not set -- cannot verify write access",
            file=sys.stderr,
        )
        return False

    tmpdir = tempfile.mkdtemp(prefix="dotllm_wcheck_")
    repo = os.path.join(tmpdir, "repo")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    try:
        # Clone kaggle-results branch; create it if it doesn't exist yet
        try:
            subprocess.run(
                [
                    "git", "clone", "--depth", "1",
                    "--branch", "kaggle-results", auth_url, repo,
                ],
                check=True, capture_output=True, env=env,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "clone", "--depth", "1", auth_url, repo],
                check=True, capture_output=True, env=env,
            )
            subprocess.run(
                ["git", "-C", repo, "checkout", "-b", "kaggle-results"],
                check=True,
            )

        check_dir = os.path.join(repo, "results", "_writecheck")
        os.makedirs(check_dir, exist_ok=True)
        with open(os.path.join(check_dir, f"{ts}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"write-check at {ts}\n")

        subprocess.run(
            ["git", "-C", repo, "config", "user.email", "kaggle-bot@dotllm.dev"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", repo, "config", "user.name", "dotLLM Kaggle Bot"],
            check=True,
        )
        subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
        subprocess.run(
            ["git", "-C", repo, "commit", "-m", f"writecheck: {ts}"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", repo, "push", auth_url, "HEAD:kaggle-results"],
            check=True, env=env,
        )
        print(
            f"[run_cell] --writecheck-only: PASSED "
            f"(pushed results/_writecheck/{ts}.txt to kaggle-results)"
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[run_cell] --writecheck-only: FAILED ({exc})", file=sys.stderr)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _detect_resume(ckpt_dir: str, dry_run: bool = False) -> "str | None":
    """Auto-detect a checkpoint in *ckpt_dir* or a Kaggle Dataset input mount.

    Search order:
      1. *ckpt_dir*/state.json              -- same-session working dir
      2. /kaggle/input/*/<rel>/state.json   -- prior version attached as dataset input

    Returns the directory containing state.json, or None if not found.
    In dry-run mode, prints what would be checked without touching the filesystem.
    """
    import glob as _glob

    # Derive a meaningful experiment name for cross-session glob patterns.
    # If ckpt_dir ends in "checkpoint" (standard layout: <out>/checkpoint),
    # use the parent directory's name as the experiment identifier.
    _tail = os.path.basename(ckpt_dir.rstrip("/\\"))
    if _tail == "checkpoint":
        _parent = os.path.dirname(ckpt_dir.rstrip("/\\"))
        exp_name = os.path.basename(_parent)
    else:
        exp_name = _tail

    if dry_run:
        print(f"[run_cell] --resume-auto (dry-run): would check (in order):")
        print(f"  1. {ckpt_dir}/state.json  (same-session checkpoint)")
        print(f"  2. /kaggle/input/*/{exp_name}/checkpoint/state.json")
        print(f"  3. /kaggle/input/*/{exp_name}/state.json")
        print(f"  4. /kaggle/input/*/checkpoint/state.json  (generic fallback)")
        print(f"  (Dry-run: filesystem checks skipped)")
        return None

    # 1. Same-session
    if os.path.isfile(os.path.join(ckpt_dir, "state.json")):
        print(f"[run_cell] --resume-auto: found same-session checkpoint at {ckpt_dir}")
        return ckpt_dir

    # 2. Cross-session: Kaggle dataset input (prior version's output attached)
    patterns = [
        f"/kaggle/input/*/{exp_name}/checkpoint/state.json",
        f"/kaggle/input/*/{exp_name}/state.json",
        "/kaggle/input/*/checkpoint/state.json",
        "/kaggle/input/*/state.json",
    ]
    candidates = [hit for pat in patterns for hit in _glob.glob(pat)]
    if candidates:
        best = max(candidates, key=os.path.getmtime)
        found = os.path.dirname(best)
        print(f"[run_cell] --resume-auto: found cross-session checkpoint at {found}")
        return found

    print(
        f"[run_cell] --resume-auto: no checkpoint found in {ckpt_dir} "
        "or /kaggle/input/ -- will train from scratch"
    )
    return None


def _pull_checkpoint(cell_id: str, ckpt_dir: str, results_remote: str, out_dir: str) -> None:
    """Pull checkpoint files from kaggle-results branch into ckpt_dir.

    Silently skips if GITHUB_PAT (or GH_PAT fallback) is unset, the branch does not exist,
    or no checkpoint is present for this cell.
    """
    pat = os.environ.get("GITHUB_PAT") or os.environ.get("GH_PAT")
    if not pat:
        print("[run_cell] GITHUB_PAT not set -- cannot pull checkpoint; training from scratch")
        return

    auth_url = results_remote
    if results_remote.startswith("https://"):
        auth_url = results_remote.replace("https://", f"https://{pat}@", 1)

    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = os.path.join(tmpdir, "results_clone")
        try:
            subprocess.run(
                [
                    "git", "clone", "--depth", "1", "--branch", "kaggle-results",
                    auth_url, clone_dir,
                ],
                check=True,
                capture_output=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.CalledProcessError:
            print("[run_cell] kaggle-results branch not found -- training from scratch")
            return

        # Copy checkpoint dir
        remote_ckpt = os.path.join(clone_dir, "results", cell_id, "checkpoint")
        if os.path.isdir(remote_ckpt):
            os.makedirs(ckpt_dir, exist_ok=True)
            shutil.copytree(remote_ckpt, ckpt_dir, dirs_exist_ok=True)
            print(f"[run_cell] checkpoint restored to {ckpt_dir}")
        else:
            print(f"[run_cell] no checkpoint in kaggle-results for {cell_id}")

        # Copy adapter weights if present (needed for eval after resume)
        remote_adapter = os.path.join(clone_dir, "results", cell_id, "adapter_weights.pt")
        if os.path.isfile(remote_adapter):
            os.makedirs(out_dir, exist_ok=True)
            shutil.copy2(remote_adapter, os.path.join(out_dir, "adapter_weights.pt"))
            print(f"[run_cell] prior adapter_weights.pt restored to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run one Track-M MoTE grid cell on Kaggle (train -> eval -> push)"
    )
    ap.add_argument(
        "--cell-id", default=None,
        help=(
            "Cell ID from grid_manifest.json (e.g. c1). "
            "Required for the normal cell-run mode; "
            "omit when using --writecheck-only or --resume-auto."
        ),
    )
    ap.add_argument(
        "--manifest",
        default="kaggle/grid_manifest.json",
        help="Path to grid_manifest.json (relative to --repo-dir or absolute)",
    )
    ap.add_argument(
        "--repo-dir",
        default=".",
        help="Root of the dotLLM repo checkout (default: current directory)",
    )
    ap.add_argument(
        "--results-remote",
        default=os.environ.get("RESULTS_REMOTE", "https://github.com/jamesburton/dotLLM"),
        help="GitHub repo URL for result push (env: RESULTS_REMOTE)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Pull prior checkpoint from kaggle-results branch before training, "
            "then pass --resume-from to mote_train.py."
        ),
    )
    ap.add_argument(
        "--teacher-device",
        default="auto",
        help=(
            "Device for the frozen teacher model. "
            "'auto' (default): on T4x2 -> cuda:1 (fast, student on cuda:0); "
            "on L4/A100 (>=20 GB single GPU) -> cuda; "
            "on single T4/P100 (<20 GB) -> cpu (slow). "
            "Override with any torch.device string: 'auto', 'cpu', 'cuda', 'cuda:1', etc."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print exact commands without executing them.",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Skip training; run only eval + push on an existing adapter at <out>/. "
            "Use to re-run eval without re-training (e.g. recovery on a Kaggle session "
            "where train completed and adapter_weights.pt exists but eval crashed). "
            "Requires adapter_weights.pt in <out>/."
        ),
    )
    ap.add_argument(
        "--writecheck-only",
        action="store_true",
        help=(
            "Push a tiny sentinel file (results/_writecheck/<ts>.txt) to the "
            "kaggle-results branch to verify write access, then exit 0 on success "
            "or 1 on failure.  Does not require --cell-id.  "
            "With --dry-run, prints the git commands (PAT redacted to ***) and exits 0."
        ),
    )
    ap.add_argument(
        "--resume-auto",
        action="store_true",
        help=(
            "Detect a checkpoint in --ckpt-dir (same-session) or in "
            "/kaggle/input/*/ (cross-session attached dataset), print the path, "
            "then exit 0.  Does not require --cell-id.  "
            "With --dry-run, prints the directories that would be searched."
        ),
    )
    ap.add_argument(
        "--ckpt-dir",
        default="/kaggle/working/ckpt",
        help=(
            "Checkpoint directory to search when using --resume-auto. "
            "Default: /kaggle/working/ckpt"
        ),
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Early-exit utility modes (do not need --cell-id)
    # ------------------------------------------------------------------
    if args.writecheck_only:
        pat = os.environ.get("GITHUB_PAT") or os.environ.get("GH_PAT", "")
        ok = _do_writecheck(args.results_remote, pat, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)

    if args.resume_auto:
        _detect_resume(args.ckpt_dir, dry_run=args.dry_run)
        sys.exit(0)

    # Beyond this point --cell-id is required
    if args.cell_id is None:
        ap.error(
            "--cell-id is required for the normal cell-run mode "
            "(use --writecheck-only or --resume-auto for utility modes that don't need it)"
        )

    teacher_device = _resolve_teacher_device(args.teacher_device)

    repo = os.path.abspath(args.repo_dir)
    # Manifest path: absolute, or relative to repo_dir
    if os.path.isabs(args.manifest):
        manifest_path = args.manifest
    else:
        manifest_path = os.path.join(repo, args.manifest)

    # ------------------------------------------------------------------
    # Load manifest + find cell
    # ------------------------------------------------------------------
    if not os.path.isfile(manifest_path):
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    cell = next((c for c in manifest.get("cells", []) if c["id"] == args.cell_id), None)
    if cell is None:
        ids = [c["id"] for c in manifest.get("cells", [])]
        print(
            f"ERROR: cell {args.cell_id!r} not found in manifest "
            f"(available: {ids})",
            file=sys.stderr,
        )
        sys.exit(1)

    if cell.get("status") == "blocked":
        note = cell.get("note", "no note provided")
        print(
            f"ERROR: cell {args.cell_id!r} is BLOCKED and cannot be run yet.\n"
            f"  Note: {note}\n"
            f"  Resolve the blocker, update the manifest status, then re-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"[run_cell] cell={args.cell_id}  n_experts={cell['n_experts']}  "
        f"top_k={cell['top_k']}  shared={cell['shared']!r}  "
        f"layers={cell['layers']!r}  tokens={cell['tokens']}  "
        f"kd_weight={cell['kd_weight']}"
    )
    if args.dry_run:
        print("[run_cell] DRY RUN -- commands will be printed but NOT executed")

    out_dir = os.path.join(repo, ".docs", "mote", args.cell_id)
    ckpt_dir = os.path.join(out_dir, "checkpoint")

    # ------------------------------------------------------------------
    # Step 0: Resume -- pull prior checkpoint from kaggle-results branch
    # ------------------------------------------------------------------
    if args.resume:
        if not args.dry_run:
            print(f"\n[run_cell] Step 0: pull checkpoint for resume")
            _pull_checkpoint(args.cell_id, ckpt_dir, args.results_remote, out_dir)
            state_json = os.path.join(ckpt_dir, "state.json")
            if os.path.isfile(state_json):
                with open(state_json, encoding="utf-8") as f:
                    st = json.load(f)
                print(
                    f"[run_cell] resuming from step={st.get('step', 0)}, "
                    f"tokens={st.get('tokens_seen', 0)}"
                )
            else:
                print("[run_cell] no prior checkpoint -- training from scratch")
        else:
            print(
                f"\n[run_cell] Step 0 (dry-run): would pull checkpoint from "
                f"kaggle-results/results/{args.cell_id}/checkpoint/"
            )

    # ------------------------------------------------------------------
    # Step 1: mote_train.py  (skipped in --eval-only mode)
    # ------------------------------------------------------------------
    if not args.eval_only:
        train_cmd = [
            sys.executable,
            os.path.join(repo, "scripts", "lora", "mote_train.py"),
            "--config", args.cell_id,
            "--n-experts", str(cell["n_experts"]),
            "--top-k", str(cell["top_k"]),
            "--shared", cell["shared"],
            "--layers", cell["layers"],
            "--tokens", str(cell["tokens"]),
            "--kd-weight", str(cell["kd_weight"]),
            "--device", "cuda",
            "--teacher-device", teacher_device,
            "--optim", "adamw8bit",
            "--checkpoint-every", "500",
            "--out", out_dir,
        ]
        if args.resume:
            # Pass checkpoint dir unconditionally; mote_train.py skips gracefully if missing
            train_cmd.extend(["--resume-from", ckpt_dir])

        print(f"\n[run_cell] Step 1: train")
        _run_train(train_cmd, dry_run=args.dry_run)

        # ------------------------------------------------------------------
        # Step 1b: push training artifacts immediately (before eval)
        # ------------------------------------------------------------------
        # Safe-push metrics.json + adapter_weights.pt + mote_config.json + train log
        # now so a subsequent eval crash cannot lose the trained adapter.
        # --train-artifacts-only skips eval.json copy and manifest status flip.
        push_train_cmd = [
            sys.executable,
            os.path.join(repo, "kaggle", "push_results.py"),
            "--cell-id", args.cell_id,
            "--adapter", out_dir,
            "--manifest", manifest_path,
            "--results-remote", args.results_remote,
            "--train-artifacts-only",
            "--push-adapter",
        ]
        print(f"\n[run_cell] Step 1b: push train artifacts")
        try:
            run(push_train_cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[run_cell] push failed (non-fatal): {exc}", file=sys.stderr)
    else:
        # --eval-only: skip train + train-artifact push
        print(f"\n[run_cell] --eval-only: skipping train and train-artifact push")
        adapter_pt = os.path.join(out_dir, "adapter_weights.pt")
        if args.dry_run:
            print(f"  [dry-run] would require {adapter_pt}")
        elif not os.path.isfile(adapter_pt):
            print(
                f"ERROR: --eval-only requires adapter_weights.pt to exist: {adapter_pt}",
                file=sys.stderr,
            )
            sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: mote_eval.py
    # ------------------------------------------------------------------
    eval_cmd = [
        sys.executable,
        os.path.join(repo, "scripts", "lora", "mote_eval.py"),
        "--adapter", out_dir,
        "--device", "cuda",
    ]

    print(f"\n[run_cell] Step 2: eval")
    eval_ok: bool
    if args.dry_run:
        print("  $ " + " ".join(str(c) for c in eval_cmd))
        eval_ok = True
    else:
        try:
            subprocess.run(eval_cmd, check=True)
            eval_ok = True
        except subprocess.CalledProcessError as exc:
            print(
                f"\n[run_cell] WARNING: eval failed (exit {exc.returncode}). "
                "Train artifacts are already on kaggle-results. "
                "Re-run with --eval-only to retry eval without re-training.",
                file=sys.stderr,
            )
            eval_ok = False

    # ------------------------------------------------------------------
    # Step 3: push eval results (only if eval succeeded)
    # ------------------------------------------------------------------
    push_eval_cmd = [
        sys.executable,
        os.path.join(repo, "kaggle", "push_results.py"),
        "--cell-id", args.cell_id,
        "--adapter", out_dir,
        "--manifest", manifest_path,
        "--results-remote", args.results_remote,
    ]

    if eval_ok:
        print(f"\n[run_cell] Step 3: push eval results")
        try:
            run(push_eval_cmd, dry_run=args.dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[run_cell] push failed (non-fatal): {exc}", file=sys.stderr)
    elif not args.dry_run:
        print(f"\n[run_cell] Step 3: skipped (eval did not complete)")

    if args.dry_run:
        print(f"\n[run_cell] DRY RUN complete -- no commands were executed")
    elif eval_ok:
        print(f"\n[run_cell] cell {args.cell_id} COMPLETE")
    else:
        print(
            f"\n[run_cell] cell {args.cell_id} PARTIAL: train OK, eval FAILED. "
            "Train artifacts pushed to kaggle-results. "
            "Re-run with --eval-only to retry eval."
        )


if __name__ == "__main__":
    main()
