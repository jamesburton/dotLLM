# Edge0 pack format (strata pack index v3) and the latent MoE

Status: reverse-engineered for the Q3_K build (`Edge0/Edge0-145B-q3k`), 2026-11-25, epic #1966, issue #1967.
The reader is `DotLLM.Models.Edge0.Edge0Pack`; the CPU forward is `DotLLM.Models.Edge0.Edge0LatentMoe`.

Labels: **[file]** measured on the published files (sizes, `index.txt`, `native_experts.txt`, `model.json`);
**[src]** read from the MIT sources `Edge0-AI/Edge0` (`edge0-strata/`) and `andynoodles/edge0-vllm` (`convert/strata_formats.py`,
`docs/MODEL-SPEC.md`); **[test]** pinned by a dotLLM test.

Edge0-145B is Qwen3.8-Flash-Next (`qwen4exp`) with the 512 routed experts of every layer replaced by a **latent MoE**. A model folder is:

| file | content |
|---|---|
| `model-skeleton.gguf` | everything except routed experts, D/U and the PLE table: attention/GDN, hyper-connection (HC) mixers, router `ffn_gate_inp` (BF16), shared expert (Q8_0), embeddings, head |
| `experts.bin` | the 48 x 512 routed experts, fixed-size records, no header |
| `native_experts.txt` | per-layer geometry of `experts.bin` |
| `labels.bin` | which D/U group each routed expert belongs to |
| `latent_dense_q8.bin`, `latent_dense_q4.bin` | the D and U matrices (use one) |
| `scales/l0..l47.f32` | Q3_K build only: per-layer 1792-float column scale |
| `index.txt`, `dense.bin` | strata pack index v3 and the float tensors (router, HC mixers, norms) |
| `ple-fp8.gguf` | the n-gram table (51.2 GB, GGUF type I8 holding FP8 E4M3 bytes) |
| `profile.bin`, `mtp/`, `mmproj-BF16.gguf`, `tokenizer/`, `model.json` | expert-hotness profile, draft head, vision, tokenizer, launch contract |

## 1. Latent MoE semantics **[src]**

Per layer (hidden H = 2560, latent l = 1792 (120B: 1280), expert FFN f = 640, 512 experts, top-10, G = 8 groups):

```
p      = softmax(ffn_gate_inp . x)            # router sees x itself, not D x; 512 logits
ids    = top-10 by p (stable, ties -> lower index);  w = p[ids] / sum(p[ids])   # no bias, scale 1
xl_g   = D_g . x                               # D_g: (l, H), one per group
xl_g  *= 1/s                                   # Q3_K build only, s = scales/l{layer}.f32, s <= 0 -> 1
h_j    = silu(gate_e . xl_g) * (up_e . xl_g)   # e = ids[j], g = labels[layer][e]; gate/up: (f, l)
z_g    = sum over j with g_j = g of  w_j * (down_e . h_j)        # down: (l, f)
out    = sum_g U_g . z_g                       # U_g: (H, l)
out   += sigmoid(ffn_gate_inp_shexp . x) * shexp_down(silu(shexp_gate x) * (shexp_up x))   # added unweighted
```

- D and U are per layer and per **group**, shared by every expert of that group; they are not per expert. Groups are uneven
  (layer 0 of the NVFP4 pack: 25, 165, 33, 53, 25, 46, 67, 98 experts). **[src]**
- U is linear, so applying it once to the per-group weighted sum (above) equals Strata applying it per expert. Only rounding differs.
- Strata computes all 8 `xl_g` in decode; dotLLM computes only the groups the chunk routes to (a row's result does not depend on the
  other rows, so the outputs are equal).
- The router and D/U see the same `x` (the FFN hyper-connection read). Everything outside this block (GDN/QSA, HC, PLE) is unchanged qwen4exp.

### Column scale (Q3_K build) **[src]**

`native_experts.txt` says "Q3_K gu + shared colscale". `make_latent_a_q3k.py` quantized `A * diag(s)` for gate and up (so the Q3_K
rows have a better-conditioned dynamic range); Strata therefore multiplies the D output by `1/s` where it is produced
(`lat_colscale_rows`: `x[r][c] *= inv[c]`, `inv[c] = s[c] > 0 ? 1/s[c] : 1`, one vector per layer shared by all 8 groups), and the expert kernels
see already-scaled rows (`native_quant_act` does NOT multiply again). Only gate and up are scaled; down is not. The NVFP4 build has no
`scales/` folder (it carries a per-expert tail instead, below). **[src]** `edge0-strata/src/kernels/cpu/expert_layout.cpp`,
`include/strata/kernels/cpu/native_expert.hpp`, `src/kernels/cuda/iq_kernels.cu`. **[test]** the colscale mutants (section 6).

## 2. `native_experts.txt` **[file][src]**

```
# strata native experts v6: latent A-only, Q3_K gu + shared colscale (expert_embd 1792, expert_ff 640)
<layer> <gu_type> <d_type> <offset> <blob_bytes>        # 48 lines, "0 11 20 0 1630720", "1 11 20 834928640 1630720", ...
```

`gu_type 11` = GGML Q3_K, `d_type 20` = IQ4_NL. `offset_l = l * 512 * blob_bytes` (834,928,640 per layer). The reader requires exactly this and
refuses anything else (`NotSupportedException` for type 40/40, the NVFP4 build).

## 3. `experts.bin` **[file][src][test]**

40,076,574,720 B = 48 layers x 512 experts x 1,630,720 B, no header, no padding. Record `(l, e)` is at `(l * 512 + e) * 1,630,720`.

```
record = gate | up | down                                    (NVFP4 build: + 16 B tail {s_gate, s_up, s_down, 0} f32)
gate   = 640 rows x 1792 inputs, Q3_K      row = 7 blocks x 110 B =  770 B     -> 492,800 B
up     = 640 rows x 1792 inputs, Q3_K                                              492,800 B
down   = 1792 rows x 640 inputs, IQ4_NL    row = 20 blocks x 18 B  =  360 B     -> 645,120 B
                                                                          total  1,630,720 B  (matches the manifest)
```

Row `r` of a projection is output feature `r` (torch `nn.Linear` layout; `y = W x`). Q3_K needs K % 256 (1792 = 7 x 256); IQ4_NL needs K % 32
(640 = 20 x 32; this is why the down projection is not Q3_K). The three projections of one expert are contiguous, so a "bank" of one projection
is the record array read with a stride of `ExpertRecordBytes`: `GateBank = record0`, `UpBank = record0 + 492,800`, `DownBank = record0 + 985,600`.
dotLLM passes exactly those bases and the stride to the existing grouped-expert kernel (no copy).

## 4. `labels.bin` **[file][src][test]**

98,304 B = int32 little-endian `[48][512]`; value `0..7` = the D/U group of routed expert `e` of layer `l`.

## 5. `latent_dense_q8.bin` / `latent_dense_q4.bin` **[file][src][test]**

No header; layers back to back (Q8: 77,987,840 B per layer, 3,743,416,320 B total; Q4: 41,287,680 B per layer, 1,981,808,640 B total).

```
layer l:  D: 8 * 1792 rows x 2560 inputs   group g = rows [1792 g, 1792 (g+1))        Q8_0 row 80 blocks x 34 B = 2720 B (Q4_0: 1440 B)
          U: 8 * 2560 rows x 1792 inputs   group g = rows [2560 g, 2560 (g+1))        Q8_0 row 56 blocks x 34 B = 1904 B (Q4_0: 1008 B)
```

Q8_0: fp16 `d` then 32 int8, `v = d*q`. Q4_0: fp16 `d` then 16 bytes, element j in the low nibble of `qs[j]`, j+16 in the high nibble, `v = (code-8)*d`.
The group count is derived from the file size; `model.json` selects Q8 (`STRATA_LATENT_DENSE_FMT=8`).

## 6. `scales/lN.f32` **[file][src][test]**

7,168 B = 1792 float32 per layer, one file per layer, all-or-none. The latent width of the model is read from this length.

## 7. Index, skeleton and the rest

- `index.txt` (`# strata pack index v3 ... # align 64 pool 1538625024 tensors 1367`): 19 fields per row
  `name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind`.
  `file 0` is `dense.bin`; `kind 0` = quantized, served from the skeleton GGUF (the row carries only the shape), `2` F32, `4` BF16, `5` F16.
  The 96 `ffn_{gate,up,down}_exps.{scale,input_scale}` rows (512 F32 each) are **not used** by Strata (no source reads them; for the NVFP4 build they do not match
  the blob tails). The Q3_K build has no per-expert scales.
- `model-skeleton.gguf`: 1079 tensors, standard `qwen4exp` names (`blk.N.hc_*`, `attn_*`, `ssm_*`, `ffn_gate_inp`, `ffn_*_shexp`, `indexer.*`, `ple_*` on layer 1, `output_hc_*`);
  no `ffn_*_exps`, no D/U, no PLE table.
- `ple-fp8.gguf`: single tensor `per_layer_token_embd.weight`, GGUF type I8 (24), shape `[160, 320001536]`, metadata `general.architecture = strata-ple`, `strata.ple.format = f8_e4m3`,
  `strata.ple.scale = 0.00019931793` (F32). The bytes are FP8 **E4M3FN** (Qwen's `F8_E4M3` shards copied verbatim by `tools/ple_fp8_pack.py`), value = `e4m3fn(byte) * scale`
  (NaN encodings 0x7F/0xFF read as 0). **[file][src][test]** Decoded by `PleRowCodec` (G4, #1974), see `docs/QWEN4EXP_PLE_TABLE.md`.
- `profile.bin`: `"STRP"`, `<5I version n_layers n_expert slots n_ranked>`, `n_ranked x <HH layer expert>`, then `int32 [48][512]` rank or -1. Not used here.
- `mtp/`: full-width Q2_0 draft experts, not latent. Not handled (G6).

## 8. What the reader refuses and why

Truncated or oversized `experts.bin`, a manifest row that disagrees with the geometry, labels outside `0..G-1`, a short or missing `scales/lN.f32` (a half-scaled pack would
silently run wrong; Strata refuses it too), **a manifest whose header names the shared column scale (`... + shared colscale`) while `scales/` is missing** (decided from the manifest, not
from whether `l0.f32` happens to exist; #1972), a latent file whose size is not a whole number of layers x groups, and NVFP4 manifests.

## 9. Validation status and open points

- **Oracle**: the CPU forward is checked against an independent float64 reference (`Edge0LatentMoeReference`, written from the pseudo-code above, F32 activations, full dequantization of every
  touched weight) on synthetic packs with the real record layout (latent 256, ff 64, 3 groups). Mutants killed: label->group shift, wrong U group, wrong D group, 1/s replaced by s, colscale dropped,
  routing weights ignored, group sum missing experts, reader inverse stored as s, U offset, up bank offset, labels ignoring the layer. See the PR for the run.
- **Colscale direction: CONFIRMED end to end (2026-11-25, #1975).** Teacher-forced mean NLL of a fixed 106-token passage on the real model through the full CPU path (skeleton + pack + FP8 PLE),
  per `Edge0Pack.ColumnScaleMode` arm (`DOTLLM_EDGE0_COLSCALE`): `1/s` **1.69** (4.1 perplexity on a 192-token variant: NLL 1.41), `s` 13.38, none 12.63 (ln(vocab) = 12.4, i.e. chance). Only `xl *= 1/s` is a working
  language model; both other arms are at chance level, so the test separates the arms by ~11 nats. `Edge0EndToEndTests` (opt-in, `DOTLLM_EDGE0_E2E=1`). (Confirmed by language-model quality, not by diffing against Edge0's own engine output; the decoded-weight evidence of stage 1 pointed the same way.)
- Real-pack checks (`Edge0RealPackTests`, skip when the pack is absent): geometry, the exact label histograms of layers 0, 23 and 47 plus their leading labels (read from the raw file by a separate script; the earlier `Sum() == 512` check was a tautology), decoded rms of experts/D/U, the scale fingerprint above, and the routed forward against the oracle
  on layers 0 and 31 with real weights, including a check that the real slice separates the mutants.
- End to end (#1975): `dotllm run <snapshot>/model-skeleton.gguf --device cpu` finds `native_experts.txt`/`experts.bin`/... and `ple-fp8.gguf` beside the skeleton and runs the CPU `Qwen4ExpTransformerModel`
  (BF16 router and Q8_0 shared expert read in place). Greedy, thinking off: "The capital of France is **Paris**." and a correct `reverse_string`. CPU-only indicative speed on this Strix Halo box, 32 threads,
  warm: prefill 4.4-5.6 tok/s, decode 1.3-1.9 tok/s. Not covered: NVFP4 tail, MTP head (G6), CUDA kernels.
- Chat template (G5): the skeleton embeds Qwen's template (`Edge0ChatTemplateTests`). Effort levels are `xhigh` (default), `medium`, `low`; `enable_thinking=false` pre-fills `<think>

</think>

` (that is the
  template's "none": `reasoning_effort: none` is refused by the template). Generation EOS is 248046 (`<|im_end|>`), not the PLE EOS 248044. `dotllm run` does not apply the template (no `--tools`): feed a
  rendered prompt with `--prompt-file`.

## 10. Vulkan (G3, #1981)

`dotllm run <snapshot>/model-skeleton.gguf --device vulkan` runs the pack on `VulkanQwen4ExpTransformerModel`; the model owns the opened pack and the FP8 file.

- **Device layout (zero-copy).** On UMA the pack is mapped as shared writable views (`Edge0Pack.Open(importableViews: true)`; nothing is ever written) and each layer's 512 records are IMPORTED in place (`VulkanWeightImportPolicy.TryImport`, GTT heap): one buffer per layer holding the records exactly as the pack stores them. The Q3_K gate/up and IQ4_NL down kernels read it with record-strided variants (`moe_{indexed,grouped}_*_strided.comp`, generated by `gen_moe_strided.py`: row base `baseOffset + e * expertStride + m * rowBytes`, expertStride = 1,630,720, baseOffset = 0 / 492,800 / 985,600). D and U (Q8_0/Q4_0, one matrix for D, eight for U) are imported the same way; the router stays BF16 (`MatMulBf16`). If the import is refused (discrete GPU, read-only file) the records are copied as ONE contiguous array (never split). Remaining copy on Strix: the skeleton (about 6-7 GiB through the normal weight policy), the BF16 routers (48 x 2.6 MB = 126 MB through the staging ring) and the label / 1/s vectors (0.4 MB).
- **Per layer** (`RecordEdge0MoeLayer`): router + top-k, D for all groups (one matmul), `edge0_latent_gather` (group row of each (token, expert), times `1/s`), experts at width 1792 (indexed MMVQ below the grouped threshold, grouped coopmat above), `edge0_group_combine` (weighted sum per group), U per group, `edge0_u_sum`, then the sigmoid-gated shared expert.
- **PLE.** The FP8 table is read by the same CPU codec on the host path.

