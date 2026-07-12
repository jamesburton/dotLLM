# B5 — GPTQ warm-start init: integration hook into `bitdistill.py`

Build item **B5** of the capability-distillation plan
(`.planning/2026-07-12-capability-distillation-design.md` §5, decision "Warm-start: GPTQ-style
init now"). New file: `scripts/lora/gptq_init.py`. **No existing file is edited by B5** — this doc
specifies the one call site + one CLI flag that central integration adds to `bitdistill.py`.

## What it does

`gptq_warmstart_init(student, calib_iter, device, ...)` replaces each `BitLinear`'s plain
absmean/STE master weight with a **calibrated** master weight whose on-the-fly absmean
ternarization (the exact `weight_quant_ternary` the forward path uses) has **lower per-layer
reconstruction MSE** than plain absmean. Pipeline per layer: Hessian `H = XᵀX` from a few CPT
batches → optimal per-tensor scale search → GPTQ error-feedback rounding → keep the lower-MSE of
{GPTQ, RTN@scale} (so it is provably ≤ absmean) → write a master weight that reproduces the chosen
`α·T` exactly (reproduction lemma, asserted).

It touches **only** `module.weight.data`, i.e. the same tensor the forward path reads. No new
buffers, no forward-path change, no export change. STE training then proceeds from this better
starting point unchanged.

## Exact hook — where and how

The call goes in `train()` (and, identically, in `train_from_cache()`), **after** the student is
built + FFN activation is set + moved to device, and **before** the optimizer is created and the
loop starts. Concretely, right after this existing block in `train()`:

```python
    info = convert_to_bitnet_student(student)
    if args.ffn_activation != "silu":
        nover = set_student_ffn_activation(student, args.ffn_activation, args.ffn_anneal_steps)
        ...
    student.to(device)
    student.train()
    # <-- INSERT B5 HOOK HERE (before grad-checkpoint enable + optimizer construction) -->
```

Insert:

```python
    if args.warmstart_init:
        import gptq_init
        # A dedicated calibration stream (do NOT consume the training cpt_stream).
        if args.tiny_model or args.tiny_random_corpus:
            calib_stream = bdata.tiny_random_stream(student.config.vocab_size, args.max_seq_len)
        else:
            calib_stream = bdata.cpt_token_stream(
                tokenizer, args.max_seq_len, dataset_name=args.cpt_dataset,
                dataset_config=args.cpt_config, local_parquet=args.cpt_local_parquet,
                seed=4242)  # different seed => disjoint from training packing
        gptq_init.gptq_warmstart_init(
            student, calib_stream, device,
            n_calib_batches=args.warmstart_batches,
            calib_batch_size=max(1, args.batch_size // 2),
            hessian_device=("cpu" if args.warmstart_hessian_cpu else None))
```

Placement rules:
- **After** `convert_to_bitnet_student` (BitLinears must exist) and any
  `set_student_ffn_activation` (calibration should see the final activation).
- **Before** `student.gradient_checkpointing_enable(...)` and optimizer construction — the init is
  a `no_grad`, eval-mode pass; doing it before grad-checkpointing avoids interaction with the
  recompute hooks. (The function saves/forces `quant_alpha=0` internally for a clean FP
  calibration and restores it, so it is order-independent w.r.t. the precision anneal.)
- Do **not** reuse `cpt_stream` for calibration — pull a separate stream (seed `4242`) so the
  training loop still starts on fresh tokens. `n_calib_batches × calib_batch_size` sequences of
  `max_seq_len` are consumed from the calibration stream only.

`train_from_cache()`: identical insert after its `convert_to_bitnet_student` + `student.train()`.
Calibration data there is still raw CPT tokens (build a `cpt_token_stream`, or feed the
`input_ids` from a few cache batches) — the teacher logits in the cache are not needed for the
weight init, only input activations are.

## CLI flags to add (in `parse_args`)

```python
    p.add_argument("--warmstart-init", action="store_true", dest="warmstart_init",
                   help="GPTQ-style calibrated warm-start of BitLinear master weights before "
                        "training (research: GPTQ-init > plain absmean/STE for low-bit students).")
    p.add_argument("--warmstart-batches", type=int, default=16, dest="warmstart_batches",
                   help="Calibration batches for the warm-start Hessian (more = steadier init).")
    p.add_argument("--warmstart-hessian-cpu", action="store_true", dest="warmstart_hessian_cpu",
                   help="Hold the (in x in) warm-start Hessians on CPU to spare VRAM on big "
                        "models (Cholesky then runs on CPU; slower but memory-light).")
```

Default is **off** (`--warmstart-init` opt-in) so existing runs are unchanged; enable it on the P1
1.7B recipe run per the locked decision.

## Calibration data — what to pass

- A few CPT batches of raw token ids — the same corpus the run trains on (default fineweb-edu, or
  the capability mix once B4 lands). Only the **input activations** are used; labels/teacher logits
  are irrelevant. 16 batches × (batch_size/2) sequences of 512 tokens is a good default
  (~4k–16k calibration tokens per layer; GPTQ is not sensitive beyond a few thousand).
- Use a **disjoint seed** from training so the loop still opens on unseen tokens.
- For the capability-distillation run, calibrating on the **capability mix** (not pure general web)
  is preferable — the Hessian then reflects the activation distribution the student will actually
  serve. Pass the mixed stream once B4's weighted mixer exists.

## Cost / memory

- One extra `no_grad` forward pass over `n_calib_batches` batches (seconds–minutes; far cheaper
  than one training step's backward).
- Peak extra memory: one `in×in` fp32 Hessian per BitLinear held simultaneously. For Qwen3-0.6B
  (~1024/3072 dims) that is ~0.9 GB total; for 1.7B use `--warmstart-hessian-cpu` to keep it off
  the 12 GB card. GPTQ Cholesky temporaries are per-layer and freed as it advances.

## Validation evidence (this file's sibling `gptq_init.py`)

- `python gptq_init.py --self-test` — tiny synthetic Qwen3: **all 14 BitLinears strictly beat
  absmean, aggregate recon MSE −50.5%**; reproduction lemma + finite ternary forward asserted.
- `python gptq_init.py --real --n-layers 2` — first 2 blocks of the real Qwen3-0.6B: recon MSE and
  tiny-slice ternary PPL before-vs-after (see the run log / task report for the numbers).

## Open decisions / notes

- **Calibration corpus for the real run**: general web vs the capability mix (recommend the mix
  once B4 lands — see above).
- **Scale-search grid** default `[0.5, 1.5]×absmean`, 21 points. The guarantee (≤ absmean) holds
  for any grid because ratio 1.0 is always included; widen `scale_hi` if per-layer logs show the
  chosen ratio pinned at the ceiling.
- **GPTQ ordering**: this implementation uses natural column order with OBS error feedback (no
  act-order/`percdamp` sweep). Sufficient for a warm-start; act-order is a possible future upgrade
  if init quality ever bottlenecks.
- **Interaction with the FFN silu→relu2 anneal**: calibration runs at the student's *current*
  activation. If `--ffn-anneal-steps > 0` (starts at silu), the init calibrates on silu; that is
  fine — the anneal moves the activation afterward and STE adapts.
