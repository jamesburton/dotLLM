#!/usr/bin/env python3
"""bitdistill_staged.py — windowed hot-layer staged training with packed-frozen layers.

Scales past bitdistill.py's full-joint-training VRAM ceiling (~1.7B params on a 12GB
card) by keeping only a small "hot window" of K consecutive transformer layers trainable
(bf16 master + grad + Adafactor state) at a time; every OTHER layer is frozen and stored
packed-ternary (~0.25 bytes/param, via ternary_pack.py) instead of a full bf16 master.
The hot window slides through the network once (baseline: forward direction,
non-overlapping windows), reusing bitdistill.py's existing cache-based KD training loop
(bd.run_cache_training_loop) unchanged per window.

Design doc: .claude/plans/velvet-frolicking-kurzweil.md

  python scripts/lora/bitdistill_staged.py --base Qwen/Qwen3-4B --device cuda \
      --from-teacher-cache .docs/bitdistill/<cache> --out .docs/bitdistill/staged_4b \
      --hot-window 4 --tokens 4.5e6 --eval-gsm8k --eval-n-gsm8k 40 --eval-nll --eval-n-nll 40
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bitdistill as bd            # noqa: E402
import bitdistill_data as bdata    # noqa: E402
import csharp_exec_eval as cse     # noqa: E402
import tooluse_exec_eval as tue    # noqa: E402
import instruction_exec_eval as iue  # noqa: E402
import continuous_eval as cve      # noqa: E402
import bitdistill_export as bexport  # noqa: E402
from ternary_pack import PackedTernaryLinear, clear_step_caches  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Window schedule
# ---------------------------------------------------------------------------
def build_windows(num_layers: int, hot_window: int, stride: int, direction: str) -> list:
    """Non-overlapping (stride==hot_window baseline) consecutive layer-index windows."""
    windows = []
    start = 0
    while start < num_layers:
        windows.append(list(range(start, min(start + hot_window, num_layers))))
        start += stride
    if direction == "reverse":
        windows = windows[::-1]
    return windows


def _layer_bitlinear_slots(layer):
    """Yield (parent_module, attr_name) for every attn/mlp BitLinear-or-PackedTernaryLinear
    slot on a decoder layer, matching bd.convert_to_bitnet_student's naming exactly."""
    attn = layer.self_attn
    for name in bd._ATTN_LINEARS:
        yield attn, name
    mlp = layer.mlp
    for name in bd._MLP_LINEARS:
        yield mlp, name


def freeze_window(student, layer_idxs: list) -> None:
    """Swap BitLinear -> PackedTernaryLinear (pack+discard the bf16 master) for the given
    transformer layers. No-op for layers already packed."""
    for i in layer_idxs:
        layer = student.model.layers[i]
        for parent, name in _layer_bitlinear_slots(layer):
            mod = getattr(parent, name)
            if isinstance(mod, PackedTernaryLinear):
                continue
            setattr(parent, name, PackedTernaryLinear.from_bitlinear(mod))


def unfreeze_window(student, layer_idxs: list, dtype: torch.dtype = torch.bfloat16) -> None:
    """Swap PackedTernaryLinear -> BitLinear (dequantize into a fresh trainable master) for
    the given transformer layers. No-op for layers already hot."""
    # Callers normally pass a small hot window (2-4 layers), where a leftover transient
    # allocation from to_bitlinear() per call is cheap enough to not matter until the caller's
    # own post-unfreeze empty_cache() (see run_staged_training). export_final(), though, calls
    # this with ALL layers in one go (a 4B/8B model's full 36) -- with no cleanup between
    # iterations, each layer's transient unpack buffer stacked on the last, measured (2026-07-19,
    # Qwen3-8B full-export) tipping into sustained WDDM paging around layer ~20+: the first 23
    # layers unpacked in minutes, then only 1 more layer in the next 15 minutes. Clearing after
    # every layer keeps the loop's peak footprint flat regardless of how many layers it's asked
    # to unfreeze at once.
    for i in layer_idxs:
        layer = student.model.layers[i]
        for parent, name in _layer_bitlinear_slots(layer):
            mod = getattr(parent, name)
            if isinstance(mod, bd.BitLinear):
                continue
            setattr(parent, name, mod.to_bitlinear(dtype=dtype))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _set_embed_lmhead_trainable(student, trainable: bool) -> None:
    for p in student.get_input_embeddings().parameters():
        p.requires_grad_(trainable)
    lm_head = getattr(student, "lm_head", None)
    if lm_head is not None and not getattr(student.config, "tie_word_embeddings", False):
        for p in lm_head.parameters():
            p.requires_grad_(trainable)


# ---------------------------------------------------------------------------
# Streaming init (needed above ~8GB bf16 footprint; see plan §3) -- loads the base model
# to CPU RAM (never fully bf16-resident on GPU), then streams+packs one layer at a time.
# ---------------------------------------------------------------------------
def _approx_bf16_gb(cfg) -> float:
    hidden, layers, inter = cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size
    heads = cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", heads)
    head_dim = getattr(cfg, "head_dim", hidden // heads)
    attn = hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim + heads * head_dim * hidden
    mlp = 3 * hidden * inter
    transformer_total = (attn + mlp) * layers
    embed = cfg.vocab_size * hidden
    lmhead = 0 if getattr(cfg, "tie_word_embeddings", False) else embed
    return (transformer_total + embed + lmhead) * 2 / 1e9


def streaming_quantize_init(base: str, device, dtype: torch.dtype, mode: str = "rtn"):
    """Load the base model to CPU RAM, then stream ONE transformer layer at a time onto GPU,
    pack it, and discard its bf16 copy -- never holds the whole model in bf16 simultaneously.
    RTN only in v1 (see plan §3 for the GPTQ-vs-RTN reasoning); mode='gptq' is a documented
    stretch flag, not implemented."""
    if mode != "rtn":
        raise NotImplementedError(
            "streaming_quantize_init only supports mode='rtn' in v1 -- see plan §3. "
            "gptq_init.py's calibrated warm-start already works unmodified for a non-streaming "
            "(4B-scale) A/B if you want to compare RTN vs GPTQ before committing to streaming GPTQ.")
    print(f"[staged] streaming init: loading {base} to CPU RAM (bf16)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    info = bd.convert_to_bitnet_student(model)   # BitLinear.from_linear preserves CPU residency
    model.get_input_embeddings().to(device)
    if hasattr(model, "lm_head") and not model.config.tie_word_embeddings:
        model.lm_head.to(device)
    if hasattr(model.model, "norm"):
        model.model.norm.to(device)
    num_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        layer.to(device)                          # this layer's params only, transiently
        freeze_window(model, [i])                  # pack immediately -> tiny GPU-resident buffer
        layer.to("cpu")                            # the now-empty BitLinear shells; harmless, small
        if (i + 1) % 8 == 0 or i == num_layers - 1:
            print(f"[staged] streaming init: layer {i + 1}/{num_layers} packed", flush=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    model.to(device)  # embed/lm_head/norm already moved; layers are tiny (packed) -- cheap now
    print(f"[staged] streaming init done: {info['bitlinears']} BitLinears packed", flush=True)
    return model, info


# ---------------------------------------------------------------------------
# Checkpointing -- window index/topology is a pure function of (num_layers, hot_window,
# window_stride, direction), so resuming only needs to know WHICH window to rebuild, not a
# separate freeze/unfreeze manifest -- bd.save_checkpoint's `extra` dict carries that index.
# ---------------------------------------------------------------------------
def save_window_checkpoint(student, out_dir: str, window_idx: int, step: int, tokens: int,
                           extra: dict | None = None) -> None:
    payload = {"window_idx": window_idx}
    if extra:
        payload.update(extra)
    bd.save_checkpoint(student, os.path.join(out_dir, f"ckpt_window{window_idx}"), step, tokens, extra=payload)


# ---------------------------------------------------------------------------
# One-off debug helper (DEBUG_MEM=1 env var) -- lists live CUDA tensors above 20MB
# grouped by (shape, dtype, requires_grad) with a referrer-type summary, to pin down
# exactly what's keeping a completed window's memory alive when torch.cuda.memory_allocated()
# stays elevated after del+gc.collect()+empty_cache() (a plain refcount/gc issue, not a
# CUDA-allocator-reservation one, since allocated -- not just reserved -- is high).
# ---------------------------------------------------------------------------
def _debug_dump_large_cuda_tensors(min_mb: float = 20.0, sample_referrers: int = 3) -> None:
    import collections
    groups = collections.defaultdict(lambda: [0, 0.0])
    samples = {}
    for obj in gc.get_objects():
        try:
            if not (torch.is_tensor(obj) and obj.is_cuda):
                continue
            mb = obj.element_size() * obj.nelement() / 2**20
            if mb < min_mb:
                continue
        except Exception:
            continue
        key = (tuple(obj.shape), str(obj.dtype), bool(obj.requires_grad), obj.grad_fn is not None)
        groups[key][0] += 1
        groups[key][1] += mb
        samples.setdefault(key, obj)  # keep one representative instance per group
    print(f"[debug-mem] large CUDA tensors (>{min_mb}MB), grouped by "
          f"(shape, dtype, requires_grad, has_grad_fn):", flush=True)
    for key, (count, mb) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"[debug-mem]   {key}: count={count} total={mb:.1f}MB", flush=True)
    # For the single biggest group, find what's actually holding a reference to it --
    # turns "something is still alive" into "THIS is what's holding it".
    top_key = max(groups.items(), key=lambda kv: kv[1][1])[0]
    tgt = samples[top_key]
    print(f"[debug-mem] referrers of one {top_key} instance:", flush=True)
    for ref in gc.get_referrers(tgt)[:sample_referrers]:
        if isinstance(ref, dict):
            desc = f"dict with keys(sample)={list(ref.keys())[:5]!r}"
        elif isinstance(ref, (list, tuple)):
            desc = f"{type(ref).__name__} len={len(ref)}"
        else:
            desc = repr(type(ref))
        print(f"[debug-mem]   referrer: {desc}", flush=True)
        # one more hop: what holds THAT referrer (e.g. an object's __dict__ -> the object)
        for ref2 in gc.get_referrers(ref)[:2]:
            if ref2 is not tgt:
                print(f"[debug-mem]     held by: {type(ref2)!r}"
                      f"{' id=' + hex(id(ref2)) if not isinstance(ref2, (dict, list, tuple)) else ''}",
                      flush=True)


# ---------------------------------------------------------------------------
# Main staged training loop
# ---------------------------------------------------------------------------
def run_staged_training(args) -> int:
    device = torch.device(args.device)
    with open(os.path.join(args.from_teacher_cache, "meta.json")) as f:
        meta = json.load(f)
    seq_len, tau = meta["seq_len"], meta["tau"]
    base = args.base or meta["base"]
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    resume_window = 0
    if args.tiny_model:
        # No real base/tokenizer/config -- mirrors bitdistill.py train()'s tiny_model path
        # (build_tiny_qwen3(), tokenizer=None, no streaming-init sizing needed).
        tokenizer = None
        use_streaming = False
        student = bd.build_tiny_qwen3().to(device)
        student.config.use_cache = False
        info = bd.convert_to_bitnet_student(student)
        student.to(device)
        freeze_window(student, list(range(len(student.model.layers))))
    else:
        tokenizer = AutoTokenizer.from_pretrained(base)
        cfg = AutoConfig.from_pretrained(base)
        use_streaming = args.streaming_init
        if use_streaming is None:
            use_streaming = _approx_bf16_gb(cfg) > 8.0
            print(f"[staged] auto streaming-init={use_streaming} (approx bf16 footprint "
                  f"{_approx_bf16_gb(cfg):.1f}GB)", flush=True)

        if use_streaming:
            student, info = streaming_quantize_init(base, device, dtype, mode=args.warmstart_mode)
        else:
            student = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
            student.config.use_cache = False
            info = bd.convert_to_bitnet_student(student)
            student.to(device)
            if args.warmstart_mode == "gptq":
                import gptq_init
                calib_stream = bdata.cpt_token_stream(
                    tokenizer, args.max_seq_len, dataset_name=args.cpt_dataset,
                    dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet,
                    seed=4242, shuffle_buffer=512)
                gptq_init.gptq_warmstart_init(
                    student, calib_stream, device, n_calib_batches=args.warmstart_batches,
                    calib_batch_size=max(1, args.batch_size // 2),
                    hessian_device=("cpu" if args.warmstart_hessian_cpu else None))
            freeze_window(student, list(range(len(student.model.layers))))  # start fully packed
    print(f"[staged] student: {info['bitlinears']} BitLinears + {info['subnorms']} SubLNs, "
          f"streaming_init={use_streaming}", flush=True)

    _set_embed_lmhead_trainable(student, args.train_embed_lmhead == "always")

    num_layers = len(student.model.layers)
    windows = build_windows(num_layers, args.hot_window, args.window_stride, args.direction)
    tokens_per_window = args.tokens_per_window if args.tokens_per_window else args.tokens / len(windows)
    print(f"[staged] {len(windows)} windows over {num_layers} layers "
          f"(hot={args.hot_window}, stride={args.window_stride}, direction={args.direction}), "
          f"~{tokens_per_window:.2e} tokens/window", flush=True)

    if args.resume_from:
        # Checkpoints save AFTER a window finishes training and is refrozen (see the main
        # loop below), so state.json's window_idx names a COMPLETED window -- resume always
        # starts cleanly at the next one. Nothing needs unfreezing before load_state_dict:
        # the checkpoint's topology (every window through window_idx already packed) matches
        # this model's current all-packed-from-init topology key-for-key.
        with open(os.path.join(args.resume_from, "state.json"), encoding="utf-8") as f:
            rst = json.load(f)
        completed_window = int(rst["window_idx"])
        resume_window = completed_window + 1
        rsd = torch.load(os.path.join(args.resume_from, "student_state.pt"), map_location="cpu")
        student.load_state_dict(rsd); del rsd
        print(f"[staged] RESUME: window {completed_window} was already complete -> "
              f"starting window {resume_window}/{len(windows)}", flush=True)

    cache_files = sorted(glob.glob(os.path.join(args.from_teacher_cache, "batch_*.pt")))
    if not cache_files:
        raise SystemExit(f"no cache batches in {args.from_teacher_cache}")

    if args.tiny_model:
        ppl_slice = bdata.tiny_random_list(student.config.vocab_size, seq_len, n=4)
        gsm8k = csharp_tasks = tooluse_ex = instruction_ex = nll_ex = []
    else:
        ppl_slice = bdata.load_ppl_slice(tokenizer, n=20, seq_len=seq_len, dataset_name=args.cpt_dataset,
                                         dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet)
        gsm8k = bdata.load_gsm8k(tokenizer, n=args.eval_n_gsm8k, seq_len=seq_len) if args.eval_gsm8k else []
        csharp_tasks = cse.load_csharp_tasks(n=args.eval_n_csharp, bench_dir=args.csharp_bench_dir) \
            if args.eval_csharp else []
        tooluse_ex = tue.load_tooluse_eval(tokenizer, n=args.eval_n_tooluse) if args.eval_tooluse else []
        instruction_ex = iue.load_instruction_eval(tokenizer, n=args.eval_n_instruction) if args.eval_instruction else []
        nll_ex = cve.load_gsm8k_nll(tokenizer, n=args.eval_n_nll) if args.eval_nll else []

    for wi, layer_idxs in enumerate(windows):
        if wi < resume_window:
            continue
        is_final_window = wi == len(windows) - 1
        unfreeze_window(student, layer_idxs, dtype=dtype)
        # unfreeze_window's BitLinear construction transiently touches fp32-sized buffers
        # (see ternary_pack.to_bitlinear's comment) -- with no empty_cache() here, the CUDA
        # allocator's reserved high-water mark from that spike would otherwise stick for the
        # whole window (this was directly observed: cuda_resv=9.88G at a window-1 milestone
        # eval against a ~5.8GB steady-state plan estimate). Return the transient blocks now,
        # before training's own steady-state allocations establish the window's real peak.
        if device.type == "cuda":
            gc.collect(); torch.cuda.empty_cache()
            print(f"[staged] window {wi + 1} post-unfreeze{bd._cuda_mem_snapshot(device)}", flush=True)
            if os.environ.get("DEBUG_MEM") and torch.cuda.memory_allocated() / 2**30 > 3.0:
                _debug_dump_large_cuda_tensors()
                types = []
                for li, layer in enumerate(student.model.layers):
                    mod = layer.self_attn.q_proj
                    types.append("BL" if isinstance(mod, bd.BitLinear) else "PK")
                print(f"[debug-mem] layer module types (BL=live BitLinear, PK=packed): "
                      f"{''.join(types)}", flush=True)
        if args.train_embed_lmhead == "final" and is_final_window:
            _set_embed_lmhead_trainable(student, True)
        if args.grad_checkpoint:
            try:
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                student.gradient_checkpointing_enable()
            if hasattr(student, "enable_input_require_grads"):
                student.enable_input_require_grads()
        student.train()

        trainable = [p for p in student.parameters() if p.requires_grad]
        if args.optimizer == "adafactor":
            from transformers.optimization import Adafactor
            opt = Adafactor(trainable, lr=args.lr, relative_step=False, scale_parameter=False,
                            warmup_init=False, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

        # Windows are atomic resume units (see the resume block above) -- each one always
        # starts its own step/token count at 0, independent of prior windows.
        out_dir_w = os.path.join(args.out, f"window{wi}")
        print(f"[staged] === window {wi + 1}/{len(windows)}: layers {layer_idxs} hot ===", flush=True)
        final = bd.run_cache_training_loop(
            student, tokenizer, opt, trainable, cache_files, device, args,
            max_tokens=tokens_per_window, resume_step=0, resume_tokens=0,
            out_dir=out_dir_w, seq_len=seq_len, tau=tau, milestones=[],
            ppl_slice=ppl_slice, gsm8k=gsm8k, csharp_tasks=csharp_tasks,
            tooluse_ex=tooluse_ex, instruction_ex=instruction_ex, nll_ex=nll_ex,
            base=base, meta=meta, post_step_hook=lambda: clear_step_caches(student))

        freeze_window(student, layer_idxs)
        if args.train_embed_lmhead == "always" and not is_final_window:
            pass  # stays trainable across windows by design (frozen only ever means "none")
        save_window_checkpoint(student, args.out, wi, final["step"], final["tokens"], extra=final)
        # `trainable` (a list of this window's now-frozen BitLinear.weight Parameters) and `opt`
        # (whose per-param state dict keys off those same Parameter objects) are plain local
        # variables -- freeze_window() drops the MODEL TREE's reference to the old BitLinears
        # (replaced by tiny PackedTernaryLinear), but `trainable`/`opt` independently keep their
        # full bf16 weight+grad+optimizer-state tensors alive until reassigned. That reassignment
        # is the first two lines of the NEXT window's loop body -- which runs AFTER this
        # empty_cache(), so without an explicit drop here the stale tensors survive the window
        # boundary and stack on top of the next window's own footprint. Measured impact: window 2
        # started at cuda_alloc=9.07G (should be ~2.3G, matching window 1's post-unfreeze) before
        # this fix -- window 1's ~6.8GB of hot-layer weight+grad+state was still live.
        # Belt-and-suspenders: explicitly drop each parameter's .grad before dropping the
        # list/optimizer references. A Parameter's .grad is a separate tensor attached via
        # autograd's C++ internals (set by the last backward()'s accumulation, or left as a
        # zero tensor if some zero_grad() call in this process ever ran with the older
        # set_to_none=False default) -- gc.get_referrers() on a leaked grad-shaped tensor
        # during debugging showed NO Python-level referrer at all (only this diagnostic
        # code's own tracking dict), meaning whatever holds it isn't a plain Python
        # container gc can trace back through, consistent with it being reachable only via
        # the C++ TensorImpl's grad_ field on a Parameter that itself lingers a moment
        # longer than expected. Clearing .grad directly removes that possibility regardless
        # of the exact mechanism.
        for p in trainable:
            p.grad = None
        del opt, trainable
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.stop_after_window is not None and wi == args.stop_after_window:
            # Mitigation for an unresolved cross-window CUDA memory leak (~5.3GB survives
            # every de-referencing fix tried: fp32-transient weight alloc, del opt/trainable,
            # explicit p.grad=None; gc.get_referrers found no Python-level referrer, pointing
            # at a C++-level PyTorch internal -- likely grad-checkpointing's non-reentrant
            # recompute machinery, though disabling it outright isn't viable since 4B doesn't
            # fit without it). Every window has started clean (cuda_alloc~2.3G) when it's the
            # FIRST window of a fresh process -- so an outer per-window shell loop restarting
            # the process each time sidesteps the leak entirely, matching the plan's own
            # anticipated mitigation (.claude/plans/velvet-frolicking-kurzweil.md §4). Exit
            # here; the outer driver relaunches with --resume-from pointed at this checkpoint.
            print(f"[staged] stop-after-window {wi} reached -- exiting for outer-loop restart", flush=True)
            return 0

    export_final(student, args.out)
    print(f"[staged] done: {len(windows)} windows, exported -> {args.out}/export", flush=True)
    return 0


def export_final(student, out_dir: str) -> dict:
    """Unpack every remaining PackedTernaryLinear back to an ordinary BitLinear (one clean
    full-model pass -- memory isn't a concern for a one-time op), then hand off to
    bitdistill_export.export unchanged so all downstream eval/export tooling keeps working."""
    num_layers = len(student.model.layers)
    export_dtype = student.get_input_embeddings().weight.dtype
    unfreeze_window(student, list(range(num_layers)), dtype=export_dtype)
    return bexport.export(student, os.path.join(out_dir, "export"))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help="defaults to the teacher cache's recorded base")
    ap.add_argument("--from-teacher-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", choices=["adamw", "adafactor"], default="adafactor")
    ap.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true", default=True)
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false",
                    help="diagnostic: disable gradient checkpointing to isolate its recompute "
                         "machinery as a suspected source of orphaned CUDA memory across windows")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--lambda-kd", type=float, default=10.0)
    ap.add_argument("--completion-mask", action="store_true", default=False)
    ap.add_argument("--precision-warmup-steps", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-ckpt", action="store_true", default=True)
    ap.add_argument("--tiny-model", action="store_true", default=False,
                    help="build_tiny_qwen3() instead of a real base -- no downloads, CPU seconds")

    ap.add_argument("--hot-window", "-K", type=int, default=4)
    ap.add_argument("--window-stride", type=int, default=None,
                    help="defaults to --hot-window (non-overlapping baseline)")
    ap.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    ap.add_argument("--tokens", type=float, default=4.5e6, help="total across all windows")
    ap.add_argument("--tokens-per-window", type=float, default=None,
                    help="overrides --tokens/num_windows if set")
    ap.add_argument("--warmstart-mode", choices=["rtn", "gptq"], default="rtn")
    ap.add_argument("--warmstart-batches", type=int, default=8)
    ap.add_argument("--warmstart-hessian-cpu", action="store_true", default=True)
    ap.add_argument("--streaming-init", dest="streaming_init", action="store_true", default=None)
    ap.add_argument("--no-streaming-init", dest="streaming_init", action="store_false")
    ap.add_argument("--train-embed-lmhead", choices=["none", "final", "always"], default="none")
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--stop-after-window", type=int, default=None,
                    help="exit after completing this window index (0-based) instead of "
                         "continuing to the next -- for an outer per-window shell loop that "
                         "restarts the process each window, sidestepping the unresolved "
                         "cross-window CUDA memory leak (see the comment at its call site)")
    ap.add_argument("--batch-size", type=int, default=4)  # only used by gptq warm-start calib

    ap.add_argument("--cpt-dataset", default="fineweb")
    ap.add_argument("--cpt-config", default=None)
    ap.add_argument("--cpt-local-parquet", default=None)

    ap.add_argument("--eval-gsm8k", action="store_true", default=False)
    ap.add_argument("--eval-n-gsm8k", type=int, default=40)
    ap.add_argument("--eval-csharp", action="store_true", default=False)
    ap.add_argument("--eval-n-csharp", type=int, default=12)
    ap.add_argument("--csharp-bench-dir", default=cse.DEFAULT_BENCH_DIR)
    ap.add_argument("--eval-tooluse", action="store_true", default=False)
    ap.add_argument("--eval-n-tooluse", type=int, default=20)
    ap.add_argument("--eval-instruction", action="store_true", default=False)
    ap.add_argument("--eval-n-instruction", type=int, default=12)
    ap.add_argument("--eval-nll", action="store_true", default=False)
    ap.add_argument("--eval-n-nll", type=int, default=40)

    args = ap.parse_args(argv)
    if args.window_stride is None:
        args.window_stride = args.hot_window
    if args.cpt_config in ("", "none", "None"):
        args.cpt_config = None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_staged_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
