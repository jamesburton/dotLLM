#!/usr/bin/env python3
"""
test_expert_parallel.py — end-to-end CPU-smoke PoC for expert-parallel identity-MoTE
(Branch-Train-MiX style) on a TINY synthetic BitNet.

Proves the whole flow with NO heavy GPU runs:

  (a) BRANCH — train expert-A (label 1 = math) ALONE and expert-B (label 2 =
      instruction) ALONE, in two SEPARATE process invocations ("two machines"). Each
      run freezes the base + all other experts and trains only its one expert on its
      own single-capability data, emitting a per-expert checkpoint.
  (b) MIX/MERGE — merge ckpt_A + ckpt_B into ONE identity-MoTE and assert the merged
      expert weights are BITWISE-IDENTICAL to the solo checkpoints (lossless
      concatenation, NOT weight averaging), with no cross-slot perturbation.
  (c) COMBINE — router-retrain on the A+B labeled corpus with ALL experts frozen;
      verify the router routes label-1 -> expert-A slot and label-2 -> expert-B slot
      (dispatch histogram).
  (d) NO-REGRESSION — the frozen skip expert still yields the exact base output
      (identity held), the merged+router-retrained forward is finite, and routing is
      correct per sequence.
  (e) MERGE-DID-NOT-PERTURB — the merged+combined model's per-expert weights AND
      module outputs match the solo-trained experts on their own capability.

Run:
    CUDA_VISIBLE_DEVICES= TORCHDYNAMO_DISABLE=1 python scripts/test_expert_parallel.py
Exit code 0 iff all of (a)-(e) pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "lora"))
sys.path.insert(0, _HERE)

from identity_mote import IdentityMoTEBlock  # noqa: E402
from identity_mote_train import (  # noqa: E402
    CKPT_FORMAT,
    build_base_only,
    construct_mote_from_spec,
    expert_slot_weights,
    load_expert_slot_weights,
    load_router_slot_weights,
)
from mote_merge_experts import merge_experts  # noqa: E402

TRAIN = os.path.join(_HERE, "lora", "identity_mote_train.py")
MERGE = os.path.join(_HERE, "lora", "mote_merge_experts.py")

# Tiny, deterministic PoC config.
CAPS = "math,instruction"          # math -> slot 1, instruction -> slot 2
SEQ_LEN = 32
NSEQ = 8
COMMON = ["--tiny-random", "--tiny-distinct", "--capabilities", CAPS,
          "--seq-len", str(SEQ_LEN), "--n-seqs-per-cap", str(NSEQ)]


def _env():
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = ""
    e["TORCHDYNAMO_DISABLE"] = "1"
    return e


def _run(cmd: list, tag: str):
    print(f"\n{'='*72}\n[poc:{tag}] $ {' '.join(os.path.basename(c) if c.endswith('.py') else c for c in cmd)}\n{'='*72}")
    r = subprocess.run(cmd, env=_env(), cwd=_HERE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Echo the last chunk so failures are legible; keep it compact on success.
    tail = "\n".join(r.stdout.strip().splitlines()[-8:])
    print(tail)
    if r.returncode != 0:
        print(r.stdout)
        raise SystemExit(f"[poc:{tag}] subprocess FAILED (exit {r.returncode})")


def _dominant_slot(model, seq_ids: torch.Tensor) -> int:
    """Top-1 routed slot for a single sequence (majority over tokens, first mote block)."""
    model.eval()
    with torch.no_grad():
        model(seq_ids.unsqueeze(0), use_cache=False)
    block = next(b for b in (l.mlp for l in model.model.layers) if isinstance(b, IdentityMoTEBlock))
    picks = block.last_logits.argmax(-1)  # [T]
    return int(torch.bincount(picks, minlength=block.n_experts).argmax().item())


def main() -> int:
    work = tempfile.mkdtemp(prefix="idm_ep_poc_")
    print(f"[poc] workdir = {work}")
    dA = os.path.join(work, "machineA_math")
    dB = os.path.join(work, "machineB_instruction")
    merged_path = os.path.join(work, "merged", "merged_experts.pt")
    dC = os.path.join(work, "combined")

    py = sys.executable
    fails = []

    # ---- (a) BRANCH: two independent single-expert trainings ("two machines") ----
    _run([py, TRAIN, "--config", "machineA", *COMMON,
          "--train-only-expert", "math", "--steps", "40", "--lr", "5e-3", "--out", dA], "a-machineA")
    _run([py, TRAIN, "--config", "machineB", *COMMON,
          "--train-only-expert", "instruction", "--steps", "40", "--lr", "5e-3", "--out", dB], "a-machineB")
    ckA = os.path.join(dA, "expert_math.pt")
    ckB = os.path.join(dB, "expert_instruction.pt")
    objA = torch.load(ckA, map_location="cpu", weights_only=False)
    objB = torch.load(ckB, map_location="cpu", weights_only=False)
    ok_a = (objA["format"] == CKPT_FORMAT and objA["slot"] == 1 and objA["kind"] == "expert"
            and objB["slot"] == 2 and objB["kind"] == "expert")
    print(f"\n[poc:(a)] expert-A slot={objA['slot']} ({objA['capability']}), "
          f"expert-B slot={objB['slot']} ({objB['capability']})  -> {'PASS' if ok_a else 'FAIL'}")
    if not ok_a:
        fails.append("(a) per-expert checkpoints")

    # ---- (b) MERGE: lossless slot concatenation ----
    merged = merge_experts([ckA, ckB], device="cpu")
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    torch.save(merged, merged_path)
    lossless = True
    for slot, obj in ((1, objA), (2, objB)):
        for k, v in obj["weights"].items():
            if not torch.equal(merged["slots"][slot]["weights"][k], v):
                lossless = False
    # And the two experts remain DISTINCT after merge (sanity: not accidentally shared).
    distinct = any(
        not torch.equal(objA["weights"][ka], objB["weights"][kb])
        for ka, kb in zip(sorted(objA["weights"]), sorted(objB["weights"]))
    )
    print(f"[poc:(b)] merged slots {sorted(merged['slots'])}  lossless={lossless}  "
          f"experts_distinct={distinct}  -> {'PASS' if (lossless and distinct) else 'FAIL'}")
    if not (lossless and distinct):
        fails.append("(b) lossless merge")

    # ---- (c) COMBINE: freeze all experts, retrain ONLY the router ----
    _run([py, TRAIN, "--config", "combine", *COMMON, "--train-only-router",
          "--load-experts", merged_path, "--route-weight", "1.0", "--router-lr", "5e-2",
          "--steps", "120", "--out", dC], "c-combine")
    combined = torch.load(os.path.join(dC, "mote_combined.pt"), map_location="cpu", weights_only=False)
    import json
    with open(os.path.join(dC, "metrics.json"), encoding="utf-8") as fh:
        cmetrics = json.load(fh)
    disp = cmetrics["dispatch_fracs"]  # [skip, slot1, slot2]
    # Correct routing: slots 1 & 2 dominate, skip (0) starved (all its data went to
    # capability experts). label-1 -> slot1, label-2 -> slot2 is proven per-sequence in (d).
    route_ok = disp[1] > 0.3 and disp[2] > 0.3 and disp[0] < 0.1
    print(f"[poc:(c)] router-retrain dispatch fracs {disp} (target ~[0,0.5,0.5])  "
          f"-> {'PASS' if route_ok else 'FAIL'}")
    if not route_ok:
        fails.append("(c) router routing")

    # ---- Rebuild the merged + router-retrained model in-process for (d)/(e) ----
    spec = combined["base_spec"]
    model = _load_combined(spec, combined)

    # ---- (d) NO-REGRESSION: skip expert still gives exact base output + finite + routing ----
    base = build_base_only(spec["base"], spec["tiny_random"], torch.device("cpu"))
    torch.manual_seed(123)
    ids = torch.randint(0, spec["vocab_size"], (1, 16))
    with torch.no_grad():
        base_logits = base(ids, use_cache=False).logits.float()
        # Force every mote block to the skip slot 0 -> must reproduce base exactly.
        for b in (l.mlp for l in model.model.layers if isinstance(l.mlp, IdentityMoTEBlock)):
            b.force_slot = 0
        skip_logits = model(ids, use_cache=False).logits.float()
        for b in (l.mlp for l in model.model.layers if isinstance(l.mlp, IdentityMoTEBlock)):
            b.force_slot = None
        routed_logits = model(ids, use_cache=False).logits.float()
    skip_diff = (skip_logits - base_logits).abs().max().item()
    finite = bool(torch.isfinite(routed_logits).all())
    # Per-sequence routing: a label-1 (math-band) seq -> slot 1, label-2 -> slot 2.
    from multitask_routed_data import build_routed_corpus
    seqs, labels, lm = build_routed_corpus(
        tokenizer=None, capabilities=spec["capabilities"], n_seqs_per_cap=4,
        seq_len=SEQ_LEN, tiny_random=True, vocab_size=spec["vocab_size"],
        tiny_distinct=True, label_override={c: v for c, v in
                                            ((k, int(val)) for k, val in spec["label_map"].items())})
    route_hits = sum(1 for s, l in zip(seqs, labels) if _dominant_slot(model, s) == l)
    routing_correct = route_hits == len(seqs)
    d_ok = skip_diff <= 1e-3 and finite and routing_correct
    print(f"[poc:(d)] skip==base max_diff={skip_diff:.3e}  routed_finite={finite}  "
          f"per-seq routing {route_hits}/{len(seqs)}  -> {'PASS' if d_ok else 'FAIL'}")
    if not d_ok:
        fails.append("(d) no-regression / routing")

    # ---- (e) MERGE-DID-NOT-PERTURB: combined experts == solo checkpoints (weights + outputs) ----
    weights_match = True
    for slot, obj in ((1, objA), (2, objB)):
        got = expert_slot_weights(model, slot)
        for k, v in obj["weights"].items():
            if not torch.equal(got[k], v):
                weights_match = False
    # Output-level: the merged model's expert module reproduces the solo expert exactly.
    solo_A = _load_combined(spec, None)              # fresh rebuild
    load_expert_slot_weights(solo_A, 1, objA["weights"])
    x = torch.randn(2, 5, spec["hidden_size"], dtype=next(model.parameters()).dtype)
    merged_e1 = next(l.mlp for l in model.model.layers if isinstance(l.mlp, IdentityMoTEBlock)).experts[1]
    solo_e1 = next(l.mlp for l in solo_A.model.layers if isinstance(l.mlp, IdentityMoTEBlock)).experts[1]
    with torch.no_grad():
        out_match = torch.equal(merged_e1(x.view(-1, spec["hidden_size"])),
                                solo_e1(x.view(-1, spec["hidden_size"])))
    e_ok = weights_match and out_match
    print(f"[poc:(e)] combined-vs-solo expert weights_match={weights_match}  "
          f"module_output_match={out_match}  -> {'PASS' if e_ok else 'FAIL'}")
    if not e_ok:
        fails.append("(e) merge did not perturb experts")

    print(f"\n{'='*72}")
    if fails:
        print(f"[poc] FAILED: {fails}")
        return 1
    print("[poc] ALL PASS — (a) branch  (b) lossless merge  (c) router combine  "
          "(d) no-regression+routing  (e) merge-preserves-experts")
    print(f"{'='*72}")
    return 0


def _load_combined(spec: dict, combined):
    """Rebuild the MoTE from spec; if `combined` given, load its expert slots + routers."""
    model, _info = construct_mote_from_spec(spec, torch.device("cpu"))
    if combined is not None:
        for slot, payload in combined["slots"].items():
            load_expert_slot_weights(model, int(slot), payload["weights"])
        load_router_slot_weights(model, combined["router"])
    return model


if __name__ == "__main__":
    raise SystemExit(main())
