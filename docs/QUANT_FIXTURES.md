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

## Generating the ladder

The fixtures are one small base model requantized once per type, so that a divergence between
two types is attributable to the quantization kernels and nothing else. Using llama.cpp's
`llama-quantize`:

```bash
BASE=~/.dotllm/quant-ladder/Llama-3.2-1B-pure/Llama-3.2-1B-pure-F16.gguf
OUT=~/.dotllm/quant-ladder/Llama-3.2-1B-pure

# 1. Convert the HF checkpoint to an F16 GGUF once.
python convert_hf_to_gguf.py meta-llama/Llama-3.2-1B --outtype f16 --outfile "$BASE"

# 2. Requantize per type. --output-tensor-type / --token-embd-type pin the embedding and
#    output tensors to Q8_0: several low-bit ftypes need imatrix coverage those two tensors
#    lack, and the gate deliberately excludes them from its block-type assertion.
for T in Q8_0 Q4_0 Q4_1 Q5_0 Q5_1 Q2_K Q3_K_M Q4_K_M Q5_K_M Q6_K; do
  llama-quantize --output-tensor-type q8_0 --token-embd-type q8_0 \
    "$BASE" "$OUT/Llama-3.2-1B-pure-$T.gguf" "$T"
done
```

### ftype vs block type

`llama-quantize`'s ftype argument names a *mixture*, not a block type: `Q4_K_M` uses `Q6_K`
for some tensors, `IQ2_M` lands on the `IQ2_S` block layout in ggml, and so on. The gate's
`Cpu_ObservedBlockType_MatchesFixtureTarget` asserts the **observed** block type of every
transformer-block weight (`attn_q/k/v/o`, `ffn_gate/up/down`), so a fixture generated with a
mixture ftype fails that assertion even though the file loads fine.

For the pure per-type ladder, force every block tensor to the target with an explicit
`--pure` build, or generate from a quantization tool that writes a single block type:

```bash
llama-quantize --pure --output-tensor-type q8_0 --token-embd-type q8_0 \
  "$BASE" "$OUT/Llama-3.2-1B-pure-IQ2_XS.gguf" IQ2_XS
```

IQ-family types additionally need an importance matrix:

```bash
llama-imatrix -m "$BASE" -f wikitext-2-raw/wiki.train.raw -o "$OUT/imatrix.dat"
llama-quantize --pure --imatrix "$OUT/imatrix.dat" \
  --output-tensor-type q8_0 --token-embd-type q8_0 \
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
