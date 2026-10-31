# Quantization coverage fixtures

Recipe for the per-`QuantizationType` GGUF fixtures consumed by
`tests/DotLLM.Tests.Integration/CrossBackend/CrossBackendQuantGateTests.cs` and
`Iq2XsVulkanCachedDecodeGateTests.cs`.

Previously these tests pointed at `.docs/corpora/QUANT_FIXTURES.md` — a git-ignored path that
does not exist in a fresh checkout, so the recipe the skip message named was unreachable
(issue #307). This file is the tracked replacement.

## Canonical location

```
~/.dotllm/quant-ladder/Llama-3.2-1B-pure/Llama-3.2-1B-pure-<TYPE>.gguf
```

`<TYPE>` is the `DotLLM.Core.Configuration.QuantizationType` enum name exactly as it is
spelled in C# — `Q4_K_M`, `IQ2_XS`, `Q8_0`, … The tests probe this path, then the per-type
environment override:

```
DOTLLM_QUANT_FIXTURE_<TYPE>=/absolute/path/to/model.gguf
```

Fixtures are model weights and therefore **never** live inside the repository — see the
"Model & Fixture Storage Rules" section of `CLAUDE.md`.

## One model store (#1279)

Real-model fixtures (as opposed to the generated ladder above) live in **one** store,
`~/.dotllm/models/{org}/{repo}/` — the same directory `dotllm model pull` writes.

- `TestModelDownloader.EnsureModelAsync` downloads there (`$DOTLLM_TEST_CACHE_DIR` overrides the
  root, e.g. for a CI cache volume). Before downloading it asks `TestFixtureResolver`, which probes
  the `DOTLLM_*_GGUF` override → `~/.dotllm/models` → legacy `~/.dotllm/test-cache` → HF hub cache
  (`$HF_HUB_CACHE` / `$HF_HOME/hub` / `~/.cache/huggingface/hub`). A hit elsewhere on the same
  volume is **hardlinked** into `~/.dotllm/models` (for an HF snapshot symlink, the blob it points
  at is linked, not the symlink); a cross-volume hit is used in place. Nothing is copied.
- Tests never hardcode a store path; they call `TestFixtureResolver.ResolveFile` /
  `ResolveDirectory` (or a `KnownTestFixtures` entry). File names may be globs.
- `~/.dotllm/test-cache/` is no longer a model store. It keeps only **generated, non-model**
  artefacts: `corpora/` (LF wikitext, `scripts/make_lf_corpus.py`), `token-oracles/`,
  `mimo2-tiny/` and `mimo2-tiny-k256/` (below), `nanoquant-oracle/`, `littlebit-qwen3-0.6b-055/`.
- `scripts/dedupe-models.ps1` migrates an existing machine: it hashes candidate duplicates across
  the three stores (SHA-256, cached in `~/.dotllm/dedupe-hash-cache.json`), replaces same-volume
  duplicates with hardlinks, and moves `test-cache` model files into `~/.dotllm/models`. It is a
  dry run unless `-Apply` is passed; `-SelfTest` checks its own logic on temp files.

> **These fixtures are for coverage and cross-backend gating, not for quality claims.**
> They are `--pure` requantizations: every tensor forced to one type, including the ones a real
> quantization scheme deliberately keeps wide. Pure-Q2_K on this 1.2 B base is a *destroyed* model
> (PPL ~1400 against ~13.9 at Q8_0), and a degraded model amplifies any fixed difference between
> two implementations by one to two orders of magnitude — measured, not asserted: the same engine
> delta reads +0.029% / +0.095% / +0.359% / +2.319% on Q8_0- / Q6_K- / Q3_K- / Q2_K-derived
> weights, with the Q6_K row (coarse grid, healthy model) null. Three quality claims died to this
> (#515, #519, #520). Use a shipping-grade quant for any engine-vs-engine or kernel-quality
> number, and if a degraded one is unavoidable report the F32-decoded control for the same weights
> (`scripts/make_f32_decoded_gguf.py`). Full rule:
> [PERPLEXITY.md](PERPLEXITY.md#how-to-measure-quality-against-llamacpp-then).

## Generating the ladder

The fixtures are one small base model requantized once per type, so that a divergence between
two types is attributable to the quantization kernels and nothing else. Using llama.cpp's
`llama-quantize`:

```bash
BASE=~/.dotllm/quant-ladder/Llama-3.2-1B-pure/Llama-3.2-1B-pure-F16.gguf
OUT=~/.dotllm/quant-ladder/Llama-3.2-1B-pure

# 1. Convert the HF checkpoint to an F16 GGUF once.
python convert_hf_to_gguf.py meta-llama/Llama-3.2-1B --outtype f16 --outfile "$BASE"

# 2. Requantize per type. --output-tensor-type / --token-embedding-type pin the embedding and
#    output tensors to Q8_0: several low-bit ftypes need imatrix coverage those two tensors
#    lack, and the gate deliberately excludes them from its block-type assertion.
for T in Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K; do
  llama-quantize --output-tensor-type q8_0 --token-embedding-type q8_0 \
    "$BASE" "$OUT/Llama-3.2-1B-pure-$T.gguf" "$T"
done
```

The flag is `--token-embedding-type`, not `--token-embd-type`: current llama.cpp rejects the
short spelling and prints its usage banner, so a script that only greps the tail of the output
will look like it succeeded.

### Starting from a published Q8_0 instead of an F16 base

Converting the HF checkpoint needs the safetensors and `convert_hf_to_gguf.py`. When all that is
wanted is a coverage or kernel fixture, requantizing a published Q8_0 is enough and much quicker
— and that same Q8_0 then doubles as the correlation oracle for checking a low-bit dequantizer,
which is how the Q2_K element-layout bug in #498 was caught and confirmed:

```bash
hf download bartowski/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-Q8_0.gguf
OUT=~/.dotllm/quant-ladder/Llama-3.2-1B-pure
BASE=<the blob path hf download printed>

# The oracle. Hardlink, never copy (CLAUDE.md, "Localise via links"). On Windows:
#   New-Item -ItemType HardLink -Path "$OUT/Llama-3.2-1B-pure-Q8_0.gguf" -Target "$BASE"
ln "$BASE" "$OUT/Llama-3.2-1B-pure-Q8_0.gguf"

for T in Q2_K Q3_K; do
  llama-quantize --allow-requantize --pure \
    --output-tensor-type q8_0 --token-embedding-type q8_0 \
    "$BASE" "$OUT/Llama-3.2-1B-pure-$T.gguf" "$T"
done
```

`--allow-requantize` is mandatory and its extra rounding is real, so these are fixtures for
kernel and layout work, not for quoting absolute quality.

Note the base model matters for the 256-element super-block types: 1B's hidden size is 2048, a
multiple of 256, whereas SmolLM-135M's 576 is not — `llama-quantize` aborts on
`GGML_ASSERT(start % blck_size == 0)`, which is why the SmolLM ladder has no Q2_K or Q3_K entry.

### ftype vs block type

`llama-quantize`'s ftype argument names a *mixture*, not a block type: `Q4_K_M` uses `Q6_K`
for some tensors, `IQ2_M` lands on the `IQ2_S` block layout in ggml, and so on. The gate's
`Cpu_ObservedBlockType_MatchesFixtureTarget` asserts the **observed** block type of every
transformer-block weight (`attn_q/k/v/o`, `ffn_gate/up/down`), so a fixture generated with a
mixture ftype fails that assertion even though the file loads fine.

For the pure per-type ladder, force every block tensor to the target with an explicit
`--pure` build, or generate from a quantization tool that writes a single block type:

```bash
llama-quantize --pure --output-tensor-type q8_0 --token-embedding-type q8_0 \
  "$BASE" "$OUT/Llama-3.2-1B-pure-IQ2_XS.gguf" IQ2_XS
```

IQ-family types additionally need an importance matrix:

```bash
llama-imatrix -m "$BASE" -f wikitext-2-raw/wiki.train.raw -o "$OUT/imatrix.dat"
llama-quantize --pure --imatrix "$OUT/imatrix.dat" \
  --output-tensor-type q8_0 --token-embedding-type q8_0 \
  "$BASE" "$OUT/Llama-3.2-1B-pure-IQ2_XS.gguf" IQ2_XS
```

## Types with their own fixtures

BitNet (`I2_S`) and PrismML/Bonsai (`PQ2_0`) are not produced by requantizing a normal
checkpoint — they are natively-trained ternary formats. Those suites resolve their own
published GGUFs through `TestFixtureResolver` instead (see `docs/QUANTIZATION.md`), and are
deliberately excluded from this ladder.

## What "missing fixture" looks like

Every gate resolves its fixture and calls `Skip.If`, so an absent fixture reports **skipped**
with the paths it probed — never a silent pass (issues #421 / #307). If you see one of these
cases reported as *passed*, the case genuinely ran.

## Small real-model fixture: Gemma-4 E2B (fast test model)

`unsloth/gemma-4-E2B-it-GGUF` / `gemma-4-E2B-it-Q4_K_M.gguf` (2.9 GiB) is the repo's **fast small
real model**: it handles arithmetic, JSON extraction, one-word classification, short summaries and
tool calls (7/7 on `scripts/small-model-tasks.py`, CPU and Vulkan, greedy-identical) and decodes at
~60-70 tok/s on the Strix Halo iGPU. Fetch it into the HF hub cache (`hf download`), never into the tree.

```
DOTLLM_GEMMA4E2B_GGUF=/absolute/path/gemma-4-E2B-it-Q4_K_M.gguf
```

unset falls back to the first `gemma-4-E2B-it-Q4_K_M.gguf` under
`~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it-GGUF/snapshots` (`HF_HOME` respected);
`Gemma4E2BGgufForwardTests` skips when neither resolves. Shape facts the tests assert: 35 layers,
hidden 1536, PLE dim 256, 20 shared-KV layers, and a **per-layer `feed_forward_length` array**
(6144 x15, 12288 x20: double-wide MLP on the shared-KV layers, #923).

Use it through the **chat template** (`dotllm chat` / `serve`): `dotllm run` is raw completion with
no template, and an instruction-tuned model answers a bare question with its end-of-turn token
immediately (`stop | N prompt, 0 generated`; llama.cpp `-dev none` does the same, verified).

```bash
dotnet dotllm.dll serve <gguf> -d vulkan --no-browser --no-ui --no-warmup &
python scripts/small-model-tasks.py http://localhost:8080 <model-id from /v1/models>
```

## MiMo-V2 tiny-oracle fixtures (`mimo2`, #936)

`tests/DotLLM.Tests.Unit/Models/MiMoV2/MiMoV2TinyOracleParityTests.cs` and `MiMoV2QuantMixParityTests.cs` resolve
`~/.dotllm/test-cache/mimo2-tiny/` and `~/.dotllm/test-cache/mimo2-tiny-k256/` (override: `DOTLLM_MIMO2_TINY_DIR`,
`DOTLLM_MIMO2_TINY_K256_DIR`) and skip when absent. Recipe (`tools/mimo2-tiny-oracle/`): `make_tiny.py <hf-remote-code-dir> <out> [seed] [small|k256]`
(needs torch + transformers 5.3.0 and the `XiaomiMiMo/MiMo-V2.6-Flash-RL` remote code + tokenizer files) -> llama.cpp
`convert_hf_to_gguf.py <out> --outtype f32` -> `dumplogits.cpp` (llama.cpp CPU logits, F32 KV, no flash attention) -> `cmp.py`. The k256 mix is
`llama-quantize --tensor-type ffn_gate_exps=q2_k --tensor-type ffn_up_exps=q2_k --tensor-type ffn_down_exps=mxfp4 --output-tensor-type q6_k
--token-embedding-type q8_0 in.gguf out.gguf Q8_0`; the F32-decoded control file is made by dequantising every tensor with gguf-py. These are
random-weight fixtures (88 MB / 340 MB because of the real 152576-token vocabulary) and are never committed.

## IQ1_S CPU-vs-GPU prefill gap is activation-quantisation noise (#1448, re-opens #276 evidence)

The CPU packed IQ1_S arm reads up to 0.07 nats below the GPUs and the F32-decoded control on a pure-IQ1_S Llama-3.2-1B
fixture. Measured on Strix (own fixture: 20-chunk imatrix, llama.cpp b9016; CPU-only, 185 scored tokens):

| arm | NLL |
|---|---|
| CPU packed (Q8_K activations, maxAbs/127) | 9.9435 |
| CPU, same dequantised rows dotted in F32 | 9.9664 |
| CPU F32-decoded GGUF (gguf-py) | 9.9664 |
| Vulkan packed / Vulkan F32-decoded | 9.9663 / 9.9687 |
| CPU packed, Q8_K scale maxAbs/D, D = 126 / 125 / 120 / 110 / 100 / 64 | 9.951 / 9.903 / 9.993 / 10.000 / 9.880 / 9.951 |

- The kernel is not wrong: the IQ1_S grid is bit-identical to gguf-py's (2048 entries), the delta / scale / `qs`+`qh`
  indexing mirror `dequantize_row_iq1_s` / `ggml_vec_dot_iq1_s_q8_K`, and the F32 dot reproduces the control to 0.0002.
- The gap is the CPU's Q8_K activation rounding (llama.cpp does the same); any *equally valid* Q8_K scale moves the
  NLL by +-0.06 on this 185-token, PPL~2e4 model, so 127 is simply one draw. Q2_K/IQ2_* agree tighter only because the
  destroyed-model amplification (see above) is smaller there.
- Consequence: the gate's IQ1_S prefill bound is 0.12 (still < the 0.24 nats of the real #276 defect). Use the F32-decoded
  control, not the CPU-vs-GPU delta, to judge an IQ1_S kernel change.
