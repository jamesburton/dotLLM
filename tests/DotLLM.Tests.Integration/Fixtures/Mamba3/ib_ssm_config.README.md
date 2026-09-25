# ib-ssm/mamba3-370M-10BT — config snapshot

- **HF repo**: https://huggingface.co/ib-ssm/mamba3-370M-10BT
- **Commit SHA**: `02943831ad63d36783f41fa872f08cc8631538ee`
- **Last modified (HF)**: 2026-04-15T05:10:32Z
- **Snapshot captured**: 2026-04-19 (Stage D1 scaffolding)
- **Source URL**: https://huggingface.co/ib-ssm/mamba3-370M-10BT/raw/main/config.json

## Safetensors layout (header-only probe, 2026-04-19)

- Single file `model.safetensors`, 1,547,482,432 bytes (~1.44 GiB)
- JSON header length: 46392 bytes (little-endian u64 at offset 0)
- Storage dtype for every tensor: **F32** — NB: `config.dtype` says `bfloat16`
  but the weights on disk are F32. Stage D2 must use F32 (not bf16) as
  the source dtype when memory-mapping.
- No `model.safetensors.index.json` (not sharded).
- No `modeling_mamba3.py` / `configuration_mamba3.py` in the repo — this
  checkpoint is pure weights + HF config + tokenizer, no custom_code.

## Per-layer tensor signature (identical across all 48 layers)

```
backbone.layers.{i}.mixer.B_bias          [nheads=32, 1,  d_state=128]  F32
backbone.layers.{i}.mixer.B_norm.weight   [d_state=128]                  F32
backbone.layers.{i}.mixer.C_bias          [nheads=32, 1,  d_state=128]  F32
backbone.layers.{i}.mixer.C_norm.weight   [d_state=128]                  F32
backbone.layers.{i}.mixer.D               [nheads=32]                    F32
backbone.layers.{i}.mixer.dt_bias         [nheads=32]                    F32
backbone.layers.{i}.mixer.in_proj.weight  [d_in_proj=4480, d_model=1024] F32
backbone.layers.{i}.mixer.out_proj.weight [d_model=1024, d_inner=2048]   F32
backbone.layers.{i}.norm.weight           [d_model=1024]                 F32
```

## Global tensors

```
backbone.embeddings.weight                [vocab_size=32000, d_model=1024]  F32
backbone.norm_f.weight                    [d_model=1024]                     F32
lm_head.weight                            [vocab_size=32000, d_model=1024]  F32
```

## Divergences from `VikramKarLex/mamba3-minimal` reference

1. **`A_log` is absent from the checkpoint.** The reference registers
   `self.A_log = nn.Parameter(torch.empty(args.nheads))` and derives
   `A = -exp(A_log)` per forward. The HF checkpoint stores no such
   parameter. Possibilities:
   - `A` is a fixed constant derived from `A_floor` (0.0001) per head,
   - `A_log` is folded into `in_proj` output dimensions (a new split),
   - the saved state_dict silently dropped it.
   **Stage D2 must resolve this before any forward pass will match.**

2. **No MLP / MLP-norm per layer.** The reference interleaves
   `mixer_norm → mixer → mlp_norm → mlp(SwiGLU)` within each layer.
   The HF checkpoint has only `{mixer.*, norm.weight}` per layer —
   no `mlp_norm` and no `w_gate/w_up/w_down`. This is a **pure-SSM**
   variant (no MLP interleaving). Single pre-mixer norm named `norm`
   (not `mixer_norm`).

3. **No explicit `theta` parameter.** Consistent with the reference:
   `theta` is data-dependent and flows out of `in_proj`'s split, not
   a stored parameter. Not a divergence — just confirming.

4. **Shape of `B_bias`/`C_bias`** is `[nheads, 1, d_state]` (3-D with a
   singleton middle). The reference registers `[nheads, d_state]` in
   SISO mode and `[nheads, d_state, mimo_rank]` in MIMO. The HF
   snapshot uses a 3-D form even in SISO (is_mimo=false); the
   singleton second axis reads as a rank-1 broadcast slot.

5. **Embedding / lm_head are not tied** (`tie_word_embeddings=false`);
   the reference ties them. Both are stored as distinct tensors here.

6. **`in_proj.weight` shape is `[4480, 1024]`** = `[d_in_proj, d_model]`.
   Decomposes as `d_in_proj = 2*d_inner + 2*d_state + 2*nheads + d_state/2`
   `= 4096 + 256 + 64 + 64 = 4480` — matches SISO reference split
   `[z, x, B, C, dt, lam, theta]`.

## Verification

Header re-probe (requires Range requests enabled against HF resolve/):
```
curl -sL -H "Range: bytes=0-7" -o /tmp/hlen.bin \
     https://huggingface.co/ib-ssm/mamba3-370M-10BT/resolve/main/model.safetensors
# first 8 bytes = 0x38 0xB5 0x00 0x00 0x00 0x00 0x00 0x00  → header len = 46392
curl -sL -H "Range: bytes=8-46399" -o /tmp/h.json \
     https://huggingface.co/ib-ssm/mamba3-370M-10BT/resolve/main/model.safetensors
# /tmp/h.json contains the tensor index documented above.
```

Weights are NOT downloaded or checked in. Stage D2 will add a
download step (HF cache) for local forward validation.
