"""bench_matrix.py — sweep runner for bench_train.py.

Campaign: trackM-mote — pick the fastest Windows-native identity-MoTE TRAINING config.

Runs a base config against a small set of single-lever variant overrides (each entry
flips ONE optimization knob from the base), invoking ``bench_train.py`` as a fresh
subprocess per variant so a crash / OOM in one config cannot poison the others.
Failures are recorded and the sweep continues.

Aggregates every run into:
  * a CSV file (``--csv-out``), and
  * a printed markdown table sorted by tokens/sec (desc), with peak-VRAM alongside,

so the orchestrator can eyeball the winner.

The matrix is a plain dict — edit ``DEFAULT_MATRIX`` freely. ``--only k1,k2`` runs a
subset of keys. Base loop-shape / model flags come from this script's own CLI and are
forwarded to every child ``bench_train.py`` run; the per-variant dict overrides win.

Do NOT run the full matrix on a busy box — this is for the orchestrator to launch on
an idle GPU. The tiny CPU smoke uses ``--tiny-random --device cpu --only <2 keys>``.

Smoke (CPU, tiny)::

    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 \
      python scripts/lora/bench_matrix.py --tiny-random --device cpu \
        --batch-size 2 --seq-len 32 --steps 3 --warmup-steps 1 \
        --n-capability-experts 2 --every 1 --only tf32_off,ckpt_on

Full GPU sweep (orchestrator, idle GPU)::

    python scripts/lora/bench_matrix.py --device cuda \
        --batch-size 4 --seq-len 256 --steps 30 --warmup-steps 5 \
        --every 4 --n-capability-experts 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_TRAIN = os.path.join(HERE, "bench_train.py")


# ---------------------------------------------------------------------------
# The matrix. Each key -> dict of bench_train.py flag overrides (ONE lever each).
# Values are strings/ints matching bench_train.py's CLI. Edit freely.
# ---------------------------------------------------------------------------
DEFAULT_MATRIX: dict[str, dict] = {
    # batch-size sweep
    "bs1": {"batch-size": 1},
    "bs2": {"batch-size": 2},
    "bs4": {"batch-size": 4},
    "bs8": {"batch-size": 8},
    # gradient checkpointing
    "ckpt_off": {"grad-checkpoint": "off"},
    "ckpt_on": {"grad-checkpoint": "on"},
    # optimizer
    "optim_fused": {"optim": "adamw-fused"},
    "optim_adafactor": {"optim": "adafactor"},
    # torch.compile (Windows-support probe)
    "compile_off": {"compile": "off"},
    "compile_on": {"compile": "on"},
    # TF32
    "tf32_on": {"tf32": "on"},
    "tf32_off": {"tf32": "off"},
    # SDPA backend
    "attn_auto": {"attn": "auto"},
    "attn_flash": {"attn": "flash"},
}


# ---------------------------------------------------------------------------
# Flags that this runner forwards to every child from its own CLI (the base config).
# Per-variant overrides in the matrix take precedence over these.
# ---------------------------------------------------------------------------
def _base_flags(args) -> dict:
    base = {
        "device": args.device,
        "batch-size": args.batch_size,
        "seq-len": args.seq_len,
        "steps": args.steps,
        "warmup-steps": args.warmup_steps,
        "every": args.every,
        "n-capability-experts": args.n_capability_experts,
        "optim": args.optim,
        "grad-checkpoint": args.grad_checkpoint,
        "compile": args.compile,
        "tf32": args.tf32,
        "attn": args.attn,
        "amp": args.amp,
        "route-weight": args.route_weight,
        "results-file": args.results_file,
    }
    return base


def _build_argv(flags: dict, tiny_random: bool, tag: str) -> list[str]:
    argv = [sys.executable, BENCH_TRAIN]
    if tiny_random:
        argv.append("--tiny-random")
    if tag:
        argv += ["--tag", tag]
    for k, v in flags.items():
        argv += [f"--{k}", str(v)]
    return argv


def _run_one(key: str, overrides: dict, args) -> dict:
    flags = _base_flags(args)
    flags.update(overrides)  # variant wins
    argv = _build_argv(flags, args.tiny_random, tag=key)
    ov_str = " ".join(f"{k}={v}" for k, v in overrides.items()) or "(base)"
    print(f"\n>>> [{key}] {ov_str}")
    print("    " + " ".join(argv))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, env=os.environ.copy())
    except Exception as exc:  # noqa: BLE001
        return {"key": key, "ok": False, "error": f"launch failed: {exc}",
                "tokens_per_sec": None, "peak_vram_gb": None}
    wall = time.perf_counter() - t0

    # Extract the BENCH_JSON: {...} line the child prints.
    row = None
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_JSON: "):
            try:
                row = json.loads(line[len("BENCH_JSON: "):])
            except json.JSONDecodeError:
                row = None
    if row is None:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        return {"key": key, "ok": False,
                "error": f"no BENCH_JSON in output (rc={proc.returncode}); tail:\n{tail}",
                "tokens_per_sec": None, "peak_vram_gb": None, "wall_s": wall}
    row["key"] = key
    row["wall_s"] = wall
    if not row.get("ok"):
        print(f"    FAILED: {row.get('error')}")
    else:
        tps = row.get("tokens_per_sec")
        vram = row.get("peak_vram_gb")
        vram_s = "n/a" if vram is None else f"{vram:.3f}GB"
        print(f"    tok/s={tps:,.1f}  vram={vram_s}")
    return row


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "key", "ok", "tokens_per_sec", "peak_vram_gb", "elapsed_s",
    "batch_size", "seq_len", "steps", "grad_checkpoint", "optim", "optim_actual",
    "compile", "compiled_ok", "tf32", "attn", "attn_ran", "amp",
    "n_capability_experts", "trainable_params", "error",
]


def _write_csv(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(v, spec=""):
    if v is None:
        return "n/a"
    if spec:
        return format(v, spec)
    return str(v)


def _print_markdown(rows: list[dict]) -> None:
    # Sort: successful runs by tok/s desc, failures last.
    def _key(r):
        ok = r.get("ok") and r.get("tokens_per_sec") is not None
        return (0 if ok else 1, -(r.get("tokens_per_sec") or 0.0))
    ordered = sorted(rows, key=_key)

    cols = ["key", "tok/s", "peakVRAM(GB)", "bs", "seq", "ckpt", "optim",
            "compile", "tf32", "attn", "amp", "status"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in ordered:
        tps = r.get("tokens_per_sec")
        status = "ok" if r.get("ok") else "FAIL"
        cval = f"{r.get('compile')}/{ 'ok' if r.get('compiled_ok') else 'no'}"
        line = [
            str(r.get("key")),
            _fmt(tps, ",.1f"),
            _fmt(r.get("peak_vram_gb"), ".3f"),
            _fmt(r.get("batch_size")),
            _fmt(r.get("seq_len")),
            _fmt(r.get("grad_checkpoint")),
            f"{_fmt(r.get('optim'))}->{_fmt(r.get('optim_actual'))}",
            cval,
            _fmt(r.get("tf32")),
            f"{_fmt(r.get('attn'))}->{_fmt(r.get('attn_ran'))}",
            _fmt(r.get("amp")),
            status,
        ]
        print("| " + " | ".join(line) + " |")

    ok_rows = [r for r in ordered if r.get("ok") and r.get("tokens_per_sec") is not None]
    if ok_rows:
        best = ok_rows[0]
        print(f"\n[bench_matrix] WINNER: {best['key']}  "
              f"tok/s={best['tokens_per_sec']:,.1f}  "
              f"peakVRAM={_fmt(best.get('peak_vram_gb'), '.3f')}GB")
    else:
        print("\n[bench_matrix] no successful runs.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sweep runner for bench_train.py.")
    ap.add_argument("--only", default="",
                    help="Comma-separated subset of matrix keys to run (default: all).")
    ap.add_argument("--list", action="store_true", help="List matrix keys and exit.")
    # Base config forwarded to every child (variant overrides win)
    ap.add_argument("--tiny-random", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--n-capability-experts", type=int, default=3)
    ap.add_argument("--optim", default="adamw")
    ap.add_argument("--grad-checkpoint", default="off")
    ap.add_argument("--compile", default="off")
    ap.add_argument("--tf32", default="off")
    ap.add_argument("--attn", default="auto")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--route-weight", type=float, default=1.0)
    # Output
    ap.add_argument("--results-file", default=os.path.join(".docs", "mote", "bench_train.jsonl"),
                    help="Per-run JSONL (passed to bench_train; appended across all runs).")
    ap.add_argument("--csv-out", default=os.path.join(".docs", "mote", "bench_matrix.csv"),
                    help="Aggregated CSV output.")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.list:
        print("Matrix keys:")
        for k, v in DEFAULT_MATRIX.items():
            print(f"  {k:16s} {v}")
        return 0

    keys = [k.strip() for k in args.only.split(",") if k.strip()] if args.only else list(DEFAULT_MATRIX)
    unknown = [k for k in keys if k not in DEFAULT_MATRIX]
    if unknown:
        print(f"error: unknown matrix keys {unknown}. Available: {list(DEFAULT_MATRIX)}", file=sys.stderr)
        return 2

    print(f"[bench_matrix] running {len(keys)} config(s): {keys}")
    print(f"[bench_matrix] base: device={args.device} tiny_random={args.tiny_random} "
          f"bs={args.batch_size} seq={args.seq_len} steps={args.steps} (warmup {args.warmup_steps})")

    rows: list[dict] = []
    for key in keys:
        rows.append(_run_one(key, DEFAULT_MATRIX[key], args))

    _write_csv(rows, args.csv_out)
    _print_markdown(rows)
    print(f"\n[bench_matrix] CSV -> {args.csv_out}")
    print(f"[bench_matrix] per-run JSONL -> {args.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
