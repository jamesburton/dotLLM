#!/usr/bin/env python3
"""mote_merge_experts.py — losslessly merge independently-trained capability experts
into ONE identity-MoTE (Branch-Train-MiX "MiX" step).

Campaign: trackM-mote — expert-parallel identity-MoTE.

Why this is lossless (the design premise)
-----------------------------------------
A MoE's experts are DISJOINT parameter blocks. With a FROZEN shared base and
ZERO-init experts (``capability_init="zero"``), a depth-expanded BitNet MoTE is
fully reproducible from its ``base_spec``: two independent processes build
bit-identical initial experts and routers (no RNG is consumed after the
deterministic base build — see identity_mote_train.construct_mote). So each
capability expert can be trained ALONE, "on a different machine", against the
identical base, and stays slot-compatible.

Merging is therefore NOT weight averaging. It is a slot-wise CONCATENATION: rebuild
the identical MoTE, then copy each independently-trained expert's weights into its
own slot. Because slots are disjoint, loading expert A into slot 1 cannot perturb
expert B in slot 2, and each merged slot is BITWISE-IDENTICAL to its solo checkpoint.
The frozen skip expert (slot 0) guarantees no-regression for any combination. Only
the router needs (re)training afterwards — the COMBINE step
(``identity_mote_train.py --train-only-router --load-experts <merged>``).

This tool:
  1. Loads N per-expert checkpoints (each ``kind == "expert"``, a distinct slot).
  2. Asserts slot-compatibility: identical ``base_spec`` (same base, same depth
     expansion, same K/capability_init/layout) AND disjoint slots.
  3. Rebuilds the identical MoTE from the shared ``base_spec`` and loads each expert
     into its slot.
  4. Asserts LOSSLESS: every merged slot equals its source checkpoint bitwise, and
     merging one slot did not perturb any other.
  5. Emits a single merged-experts checkpoint (``kind == "merged"``) that the
     router-retrain / COMBINE step consumes via ``--load-experts``.

CPU-only, seconds on the tiny synthetic model; no downloads for ``tiny_random`` specs.

Usage
-----
    python scripts/lora/mote_merge_experts.py \
        --experts run_math/expert_math.pt run_instruction/expert_instruction.pt \
        --out merged/merged_experts.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identity_mote import IdentityMoTEBlock  # noqa: E402,F401  (import side-effect / typing)
from identity_mote_train import (  # noqa: E402
    CKPT_FORMAT,
    construct_mote_from_spec,
    expert_slot_weights,
    load_expert_slot_weights,
    specs_compatible,
)


def merge_experts(expert_paths: list, device: str = "cpu") -> dict:
    """Merge per-expert checkpoints into one identity-MoTE. Returns the merged dict.

    Raises on incompatible ``base_spec`` or overlapping slots. Asserts losslessness
    (each merged slot bitwise-equals its source; merging is disjoint).
    """
    if len(expert_paths) < 1:
        raise ValueError("need at least one per-expert checkpoint")

    dev = torch.device(device)
    ckpts = []
    for path in expert_paths:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("format") != CKPT_FORMAT or obj.get("kind") != "expert":
            raise ValueError(f"{path}: not an {CKPT_FORMAT} per-expert checkpoint")
        ckpts.append((path, obj))

    # --- Slot-compatibility: identical base_spec + disjoint slots ---
    base_spec = ckpts[0][1]["base_spec"]
    seen_slots: dict = {}
    for path, obj in ckpts:
        ok, diffs = specs_compatible(base_spec, obj["base_spec"])
        if not ok:
            raise ValueError(
                f"{path}: base_spec INCOMPATIBLE with {ckpts[0][0]} (differing: {diffs}). "
                "Experts trained against different bases/expansions are not slot-compatible."
            )
        slot = int(obj["slot"])
        if slot in seen_slots:
            raise ValueError(
                f"{path}: slot {slot} already provided by {seen_slots[slot]} — "
                "slots must be DISJOINT (one expert per slot)."
            )
        seen_slots[slot] = path

    print(f"[merge] {len(ckpts)} experts; slots {sorted(seen_slots)}; "
          f"K={base_spec['n_capability_experts']} capability_init={base_spec['capability_init']!r}")
    print("[merge] slot-compatibility OK (identical base_spec, disjoint slots)")

    # --- Rebuild the identical MoTE and slot each expert in ---
    model, info = construct_mote_from_spec(base_spec, dev)
    merged_slots: dict = {}
    for path, obj in ckpts:
        slot = int(obj["slot"])
        load_expert_slot_weights(model, slot, obj["weights"])
        merged_slots[slot] = {
            "capability": obj.get("capability"),
            "label": obj.get("label", slot),
            "weights": obj["weights"],
            "source": os.path.abspath(path),
        }

    # --- LOSSLESS assertions ---
    # (1) Each merged slot equals its source checkpoint bitwise.
    for path, obj in ckpts:
        slot = int(obj["slot"])
        merged = expert_slot_weights(model, slot)
        for k, v in obj["weights"].items():
            if not torch.equal(merged[k], v):
                raise RuntimeError(f"LOSSLESS FAIL: slot {slot} key {k!r} differs from {path}")
    # (2) Merging did not perturb other slots (disjoint): re-check every slot again
    #     after ALL loads are done.
    for path, obj in ckpts:
        slot = int(obj["slot"])
        merged = expert_slot_weights(model, slot)
        for k, v in obj["weights"].items():
            if not torch.equal(merged[k], v):
                raise RuntimeError(
                    f"LOSSLESS FAIL (cross-perturbation): slot {slot} key {k!r} changed "
                    f"after loading other slots")
    print(f"[merge] LOSSLESS verified: {len(ckpts)} slots bitwise-identical to sources, "
          "no cross-slot perturbation")

    merged = {
        "format": CKPT_FORMAT,
        "kind": "merged",
        "base_spec": base_spec,
        "slots": {int(s): payload for s, payload in merged_slots.items()},
        "sources": [os.path.abspath(p) for p, _ in ckpts],
    }
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--experts", nargs="+", required=True,
                    help="Per-expert checkpoints (expert_<cap>.pt), one per distinct slot.")
    ap.add_argument("--out", required=True, help="Output merged-experts checkpoint path (.pt).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    merged = merge_experts(args.experts, device=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(merged, args.out)

    manifest = {
        "format": merged["format"],
        "kind": merged["kind"],
        "slots": {str(s): {"capability": p["capability"], "label": p["label"],
                           "source": p["source"]} for s, p in merged["slots"].items()},
        "base_spec": merged["base_spec"],
        "sources": merged["sources"],
    }
    man_path = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[merge] merged experts -> {args.out}")
    print(f"[merge] manifest       -> {man_path}")
    print(f"[merge] next: identity_mote_train.py --train-only-router --load-experts {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
