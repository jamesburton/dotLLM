"""push_results.py -- Push MoTE grid cell results to kaggle-results branch.

Pushes metrics.json, eval.json, mote_config.json, train log, and optionally adapter
weights to results/<cell-id>/ on the kaggle-results branch of the results repo.
Also flips the cell status in grid_manifest.json to "done" and commits that update.

The operation is idempotent: re-running after a successful push produces a no-op commit.

Requires:
    GITHUB_PAT env var -- GitHub PAT with repo write access.
    (falls back to GH_PAT if GITHUB_PAT is not set)

Usage
-----
    python kaggle/push_results.py --cell-id c1 --adapter .docs/mote/c1 \
        [--manifest kaggle/grid_manifest.json] \
        [--results-remote https://github.com/jamesburton/dotLLM] \
        [--push-adapter]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


RESULTS_BRANCH = "kaggle-results"
_MAX_GIT_BYTES = 95 * 1024 * 1024  # GitHub hard-rejects files > 100 MB; stay well under


def _redact(cmd: list, secret: str) -> list:
    """Return a copy of cmd with every occurrence of secret replaced by ***."""
    return [str(c).replace(secret, "***") for c in cmd]


def _run(cmd: list, *, secret=None, **kw) -> None:
    display_cmd = _redact(cmd, secret) if secret else cmd
    print("  $ " + " ".join(str(c) for c in display_cmd))
    subprocess.run(cmd, check=True, **kw)


def _auth_url(remote: str, pat: str) -> str:
    if remote.startswith("https://"):
        return remote.replace("https://", f"https://{pat}@", 1)
    return remote  # SSH or other -- pass through unchanged


def _clone_or_pull(results_checkout: str, auth_url: str, remote: str, pat: str = "") -> None:
    """Clone or update the kaggle-results branch into results_checkout."""
    if os.path.isdir(os.path.join(results_checkout, ".git")):
        print(f"[push_results] updating existing checkout at {results_checkout}")
        _run(
            ["git", "-C", results_checkout, "fetch", "origin", RESULTS_BRANCH],
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        _run(["git", "-C", results_checkout, "checkout", RESULTS_BRANCH])
        _run(["git", "-C", results_checkout, "reset", "--hard", f"origin/{RESULTS_BRANCH}"])
    else:
        os.makedirs(results_checkout, exist_ok=True)
        print(f"[push_results] cloning {remote} ({RESULTS_BRANCH})")
        try:
            _run(
                ["git", "clone", "--depth", "1", "--branch", RESULTS_BRANCH,
                 auth_url, results_checkout],
                secret=pat,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.CalledProcessError:
            # Branch does not exist yet -- clone default branch, create results branch
            print(f"[push_results] {RESULTS_BRANCH} not found; creating it")
            _run(
                ["git", "clone", "--depth", "1", auth_url, results_checkout],
                secret=pat,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            _run(["git", "-C", results_checkout, "checkout", "-b", RESULTS_BRANCH])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Push MoTE grid results to kaggle-results branch"
    )
    ap.add_argument("--cell-id", required=True, help="Grid cell ID (e.g. c1)")
    ap.add_argument(
        "--adapter", required=True,
        help="Adapter directory (contains metrics.json, eval.json, etc.)",
    )
    ap.add_argument(
        "--manifest",
        default="kaggle/grid_manifest.json",
        help="Path to grid_manifest.json (to update cell status to 'done')",
    )
    ap.add_argument(
        "--results-remote",
        default=os.environ.get("RESULTS_REMOTE", "https://github.com/jamesburton/dotLLM"),
        help="GitHub repo URL for results (env: RESULTS_REMOTE)",
    )
    ap.add_argument(
        "--push-adapter",
        action="store_true",
        help="Also push adapter_weights.pt (large -- use only when needed)",
    )
    ap.add_argument(
        "--train-artifacts-only",
        action="store_true",
        help=(
            "Push only training artifacts (metrics.json, mote_config.json, train log, "
            "adapter_weights.pt if --push-adapter is set); skip eval.json and do NOT "
            "flip manifest status to 'done'.  Used by run_cell.py to safe-push results "
            "immediately after train succeeds, before eval runs."
        ),
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help="Directory for the results checkout (default: temp dir auto-cleaned)",
    )
    args = ap.parse_args()

    # Validate PAT
    pat = os.environ.get("GITHUB_PAT") or os.environ.get("GH_PAT")
    if not pat:
        print(
            "ERROR: GITHUB_PAT environment variable not set (and GH_PAT not found as fallback).\n"
            "Set GITHUB_PAT to a GitHub PAT with repo 'Contents: write' access.",
            file=sys.stderr,
        )
        sys.exit(1)

    cell_id = args.cell_id
    adapter_dir = os.path.abspath(args.adapter)
    # Manifest: prefer the path given on CLI; fall back to well-known relative path
    manifest_path = (
        os.path.abspath(args.manifest)
        if args.manifest
        else None
    )

    remote = args.results_remote
    auth = _auth_url(remote, pat)

    cleanup = False
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="dotllm_push_")
        cleanup = True

    results_checkout = os.path.join(work_dir, "results_repo")

    try:
        _clone_or_pull(results_checkout, auth, remote, pat=pat)

        # Destination: results/<cell-id>/
        dest_dir = os.path.join(results_checkout, "results", cell_id)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy standard result files
        for fname in ("metrics.json", "eval.json", "mote_config.json"):
            src = os.path.join(adapter_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
                print(f"[push_results] copied {fname}")
            else:
                print(f"[push_results] {fname} not found -- skipping")

        # Train log (first match wins)
        for log_name in ("train_log.txt", "train.log", "output.log"):
            log_src = os.path.join(adapter_dir, log_name)
            if os.path.isfile(log_src):
                shutil.copy2(log_src, os.path.join(dest_dir, log_name))
                print(f"[push_results] copied train log: {log_name}")
                break

        # Checkpoint state.json (for resume support -- small, always push)
        ckpt_state = os.path.join(adapter_dir, "checkpoint", "state.json")
        if os.path.isfile(ckpt_state):
            os.makedirs(os.path.join(dest_dir, "checkpoint"), exist_ok=True)
            shutil.copy2(ckpt_state, os.path.join(dest_dir, "checkpoint", "state.json"))
            print("[push_results] copied checkpoint/state.json")

        # Adapter weights (optional, large).
        # --push-adapter is effectively deprecated for the git path: GitHub hard-rejects
        # any file > 100 MB, so a 1.7 GB adapter_weights.pt would cause an HTTP 500 and
        # abort the push.  The size guard below enforces a 95 MB ceiling so that the flag
        # remains a no-op-with-warning for oversized files while still working for any
        # future small adapter variant.  Large-adapter retention must use a separate
        # mechanism (Kaggle Dataset output or HF Hub) -- that is out of scope here.
        if args.push_adapter:
            print(
                "[push_results] WARNING: --push-adapter is deprecated for the git path. "
                "GitHub rejects files > 100 MB; oversized adapter weights will be skipped."
            )
            for w_rel in ("adapter_weights.pt", os.path.join("checkpoint", "adapter_weights.pt")):
                w_src = os.path.join(adapter_dir, w_rel)
                if os.path.isfile(w_src):
                    size_bytes = os.path.getsize(w_src)
                    size_mb = size_bytes / (1024 * 1024)
                    if size_bytes > _MAX_GIT_BYTES:
                        print(
                            f"[push_results] skipping {w_rel} "
                            f"({size_mb:.1f} MB > 95 MB GitHub limit)"
                        )
                    else:
                        w_dst = os.path.join(dest_dir, w_rel)
                        os.makedirs(os.path.dirname(w_dst), exist_ok=True)
                        shutil.copy2(w_src, w_dst)
                        print(f"[push_results] copied {w_rel} ({size_mb:.1f} MB)")

        # Update manifest: flip cell status to "done" (skipped in train-artifacts-only mode)
        if not args.train_artifacts_only:
            manifest_dest = os.path.join(results_checkout, "grid_manifest.json")
            manifest_src = manifest_path if (manifest_path and os.path.isfile(manifest_path)) else None
            if manifest_src is None and os.path.isfile(manifest_dest):
                manifest_src = manifest_dest  # use what's already in the results branch

            if manifest_src:
                with open(manifest_src, encoding="utf-8") as fh:
                    mf = json.load(fh)
                for c in mf.get("cells", []):
                    if c.get("id") == cell_id:
                        c["status"] = "done"
                        break
                with open(manifest_dest, "w", encoding="utf-8") as fh:
                    json.dump(mf, fh, indent=2)
                print(f"[push_results] manifest: {cell_id} -> done")
            else:
                print("[push_results] manifest not found -- skipping status update")
        else:
            print("[push_results] train-artifacts-only: skipping manifest status update")

        # Git commit + push
        _run(["git", "-C", results_checkout, "config", "user.email", "kaggle-bot@dotllm.dev"])
        _run(["git", "-C", results_checkout, "config", "user.name", "dotLLM Kaggle Bot"])
        _run(["git", "-C", results_checkout, "add", "-A"])

        status = subprocess.run(
            ["git", "-C", results_checkout, "status", "--porcelain"],
            capture_output=True, text=True,
        )
        if status.stdout.strip():
            commit_msg = (
                f"results: {cell_id} train"
                if args.train_artifacts_only
                else f"results: {cell_id} done"
            )
            _run(["git", "-C", results_checkout, "commit", "-m", commit_msg])
            _run(
                ["git", "-C", results_checkout, "push", auth, f"HEAD:{RESULTS_BRANCH}"],
                secret=pat,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            kind = "train artifacts" if args.train_artifacts_only else "results"
            print(f"[push_results] pushed {cell_id} {kind} to {RESULTS_BRANCH}")
        else:
            print(f"[push_results] nothing to commit (idempotent -- already pushed)")

    finally:
        if cleanup and os.path.isdir(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)

    print(f"[push_results] done: {cell_id}")


if __name__ == "__main__":
    main()
