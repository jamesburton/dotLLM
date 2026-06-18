# BitNet Handoff

## Current State

- Branch/worktree: `agent-graph`, pushed to `origin/issue/77-bitnet-support`.
- Latest pushed commit: `5b6cf92 perf(sampling): stackalloc repetition penalty windows`.
- BitNet GPU decode on RTX 3060 is graph-speed by default for eligible single-token decode.
- CUDA graph replay can be disabled with `DOTLLM_CUDA_GRAPH=0`.
- Experimental GPU W2A8 decode remains opt-in with `DOTLLM_CUDA_I2S_A8=1` and was negative locally.

## Confirmed Local Performance Notes

- CUDA graph default path: about `94 tok/s` decode on the local RTX 3060 without setting any env var.
- Raw launch fallback: about `76 tok/s` on a short `DOTLLM_CUDA_GRAPH=0` check.
- Repetition penalty stackalloc window optimization:
  - 32k vocab/window 64: about `8.07 us` current vs `9.40 us` legacy, roughly `14%` faster.
  - 128k vocab/window 64: roughly neutral to slightly faster.
  - Window 256: neutral to modestly faster, noisy but not worse.
- Negative local top-p experiment: scalar logit-sort/logsumexp top-p regressed badly on this Westmere host. Keep the idea open for newer CPUs with better vector support.
- Negative local categorical experiment: in-place scalar categorical sampling was neutral/worse at 128k vocab, so it was not kept.

## Next Optimization Queue

1. Add CLI/diagnostic visibility for CUDA graph and sampling behavior.
   - Report graph enabled/captured/disabled state in CLI/server metadata.
   - Make it obvious when the fallback path is being used.
   - Include sampling-time breakdown if cheap enough: repetition penalty, temperature/top-k/top-p, categorical.

2. Optimize sampled decoding around `top-k`.
   - Add bounded-candidate categorical sampling when `top-k` is active.
   - Prefer preserving the current full-vocab path for no-top-k requests.
   - Benchmark on both this Westmere box and newer Framework/Meteor Lake/Strix-class CPUs before generalizing.

3. Profile CUDA graph internals and fuse the first obvious small-kernel cluster.
   - Graph replay removes most launch overhead, but the graph still contains many small kernels.
   - Candidate fusions: bias-add projection epilogues, simple residual/add/convert chains, norm-adjacent operations.
   - Gate each fusion with a microbenchmark and a BitNet correctness prompt.

4. Measure long-prompt prefill before changing CUDA arch policy.
   - Short prompts make prefill look fixed-cost bound.
   - Use long chat/history prompts to decide whether `sm_86` tensor-core/prefill work is worth a compatibility split.
   - Keep `compute_61` fallback unless the project explicitly raises the baseline.

5. Revisit W2A8/INT8 decode only after profiling says GEMV is dominant again.
   - The first graph+A8 attempt was strongly negative locally.
   - Any future attempt should avoid per-token activation quantization overhead or use a materially better layout.

## Larger Follow-On Research

- BitNet depth expansion / BitNet-Pro: run the zero-residual ternary identity expansion on real bf16 master weights, then heal-train with FP teacher distillation.
- BitNet MoE: reuse existing dotLLM MoE/router infrastructure, with ternary experts and higher-precision routing.
- Fine-tuning / LoRA: evaluate whether lightweight adaptation improves base-model usefulness before deeper architecture work.
- DiffusionGemma note is tracked separately on the `dev-diffusiongemma` branch in that branch's `docs/HANDOFF.md`.
