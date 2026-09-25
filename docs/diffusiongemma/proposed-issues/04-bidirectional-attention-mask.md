# 04 — Bidirectional / hybrid attention mask path (non-causal seam)

**Effort: XL**

## Summary / Motivation
This is **the core architectural seam** and the biggest risk in the epic. dotLLM is causal-only:
`ApplyCausalMask` is hardcoded in the attention kernels and `NaiveAttentionStrategy` rejects explicit
masks. Diffusion decode needs **bidirectional attention over the canvas** while the AR prefill stays
**causal + KV-cached**, in one model. We introduce an attention-mask abstraction that supports both
without regressing the AR hot path.

## Scope
- Introduce an attention-mask mode/strategy: `Causal` (default, unchanged) vs `Bidirectional`
  (canvas block) vs `Hybrid` (causal prefix prompt + bidirectional canvas suffix that cross-attends
  to the cached prefix).
- Refactor `Attention.cs` so causal masking is one mask provider among several; keep the existing
  causal fast path byte-identical when mode=Causal (zero-overhead-when-off rule).
- Thread the mask mode through `IModel.Forward` (new optional parameter or forward-context struct)
  and the CPU `TransformerModel` attention call sites.
- Allow the canvas tokens to attend to (a) all canvas positions (bidirectional) and (b) cached
  prompt KV (cross-attention) while the prompt KV remains causal.
- Sliding-window + softcap + QPAS must compose with the bidirectional mode.

## Acceptance Criteria
- [ ] New mask abstraction; `Causal` path numerically identical to today (golden test).
- [ ] Bidirectional forward over a block produces symmetric attention (position i attends to j>i).
- [ ] Hybrid mode: canvas attends to full cached prefix + full canvas; prefix stays causal.
- [ ] Per-layer sliding window still applies under bidirectional mode.
- [ ] No measurable regression on the AR decode microbench (CPU; no GPU needed for this gate).
- [ ] Unit tests for each mask mode on a synthetic model.

## Dependencies
- Independent of the Gemma backbone issues at the kernel level, but the **end-to-end** value lands
  with **01–03**. Can begin in parallel with 01.

## References (dev, file:line)
- `src/DotLLM.Cpu/Kernels/Attention.cs:160,381` (ApplyCausalMask call sites)
- `src/DotLLM.Cpu/Kernels/Attention.cs:727-771` (`ApplyCausalMask` impl)
- `src/DotLLM.Core/Models/IModel.cs:25,35,53` (Forward signatures to thread mask through)
- grep `bidirectional` in `src/` → **no matches** (net-new)
