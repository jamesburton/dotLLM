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
- `ple-fp8.gguf`: single tensor `per_layer_token_embd.weight`, GGUF type I8 (24), shape `[160, 320001536]`, bytes are FP8 E4M3FN, value = `e4m3fn(byte) * strata.ple.scale`
  (NaN encodings 0x7F/0xFF read as 0). Not handled by this reader (G4).
- `profile.bin`: `"STRP"`, `<5I version n_layers n_expert slots n_ranked>`, `n_ranked x <HH layer expert>`, then `int32 [48][512]` rank or -1. Not used here.
- `mtp/`: full-width Q2_0 draft experts, not latent. Not handled (G6).

## 8. What the reader refuses and why

Truncated or oversized `experts.bin`, a manifest row that disagrees with the geometry, labels outside `0..G-1`, a short or missing `scales/lN.f32` (a half-scaled pack would
silently run wrong; Strata refuses it too), a latent file whose size is not a whole number of layers x groups, and NVFP4 manifests.

## 9. Validation status and open points

- **Oracle**: the CPU forward is checked against an independent float64 reference (`Edge0LatentMoeReference`, written from the pseudo-code above, F32 activations, full dequantization of every
  touched weight) on synthetic packs with the real record layout (latent 256, ff 64, 3 groups). Mutants killed: label->group shift, wrong U group, wrong D group, 1/s replaced by s, colscale dropped,
  routing weights ignored, group sum missing experts, reader inverse stored as s, U offset, up bank offset, labels ignoring the layer. See the PR for the run.
- **Colscale direction (the one residual risk).** The multiplier direction is taken from the Strata source (`xl *= 1/s`, `s` = the file value). Real-data evidence, measured 2026-11-25 on layers 0 and 5
  of the Q3_K pack with a numpy decoder independent of dotLLM: (a) the decoded gate/up column rms is proportional to `1/s` (corr of the logs -0.999, CV of `rms*s` 0.018 / 0.029), so the file
  and layer pairing is right; (b) treating Strata's convention (`A = G/s`) as the unscaled weight, the columns of `A` are negatively correlated with the magnitude of the matching latent
  coordinate `|D x|` (corr -0.42 / -0.32), as AWQ-style balancing predicts, whereas the opposite convention (`A = G*s`) shows no relation (+0.07 / +0.14). This favours the Strata convention but is
  indirect; it cannot replace an end-to-end comparison with Edge0's own engine output (T5500, needs approval) once the full model runs here.
- Real-pack checks (`Edge0RealPackTests`, skip when the pack is absent): geometry, label histograms, decoded rms of experts/D/U, the scale fingerprint above, and the routed forward against the oracle
  on layers 0 and 31 with real weights, including a check that the real slice separates the mutants.
- Not covered: end-to-end model run (needs G4 FP8 PLE rows and the skeleton wiring), NVFP4 tail, Vulkan/CUDA kernels.
