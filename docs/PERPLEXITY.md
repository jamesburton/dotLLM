# Perplexity — measuring it, and comparing it to llama.cpp

`dotllm perplexity` reproduces llama.cpp's `--perplexity` methodology: non-overlapping chunks of
`--context` tokens, scoring the second half of each. With the same model file, corpus, context and
stride the figure is directly comparable to a published llama.cpp number — **provided both engines
tokenize the same bytes.** That proviso is the subject of most of this document.

```bash
dotllm perplexity <model.gguf> --corpus ~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw
```

## The CRLF trap (issue #506)

`llama-perplexity` built with MSVC opens its prompt file in **text mode**. The C runtime collapses
every `\r\n` to `\n` before the tokenizer sees it. dotLLM reads the bytes and keeps the `\r`.

If the corpus has CRLF line endings, the two engines tokenize **different text**. Measured
2026-09-23 on `wiki.test.raw` (1,299,263 bytes, 7,249 CRs, all of them CRLF) with Llama-3.2-1B:
**501 of the 512 tokens in chunk 0 differed.** The paired per-chunk standard deviation on this corpus is ~0.2 nats,
an order of magnitude larger than the quantization effects usually being chased, so any aggregate
agreement observed that way was luck rather than validation.

Every end-to-end dotLLM-vs-llama.cpp perplexity comparison in which each side tokenized the file
itself is suspect. Comparisons where dotLLM was fed llama.cpp's exact token ids are not affected.

**Re-measured 2026-09-23, then again 2026-09-24.** The 2026-09-23 pass concluded the Q2_K and Q3_K
gaps "survive on identical bytes". They did not: identical bytes are not identical *tokens*, and
the streams were offset by a BOS — see [the BOS trap](#the-bos-trap-issue-515) and the
[aligned 564-chunk baseline](#measured-the-aligned-564-chunk-baseline-2026-09-24). Under the
aligned protocol Q3_K is **+1.33%**, not +4.79%, and what survives is a narrower claim: dotLLM's
Q3_K quantization costs ~2.9× llama.cpp's. The Bonsai and Nemotron-H claims are
still unverified and are tracked as #514.

> **The "~2.9×" in the paragraph above did not survive either** (audit #527, 2026-09-24). It is
> computed from the 564-chunk table's two Q3_K rows, i.e. from a *degraded* weight set, and the
> same ratio **reverses to 0.62× on a healthy model** — see
> [Degraded models amplify](#degraded-models-amplify-a-small-fixed-difference) and the closure of
> #519. Kept here rather than deleted because this is the third layer of the same mistake: each
> time a number was narrowed it was narrowed to another figure measured on the same amplifier.
> What survives from this section is the method (share the token ids, pass `--bos`), not a ratio.

### Why the reader is not "fixed"

llama.cpp on **Linux** keeps the `\r` as well — the platform-dependent part is the MSVC text-mode
read, not llama.cpp's own behaviour. Making `CorpusReader` strip CRs unconditionally would trade a
Windows mismatch for a Linux one. So:

- `dotllm perplexity` **scans the corpus for CR bytes and warns** before doing anything else, and
  repeats the verdict as a `Corpus line endings` row in the results table (the table is what gets
  pasted into an issue, and a warning printed before a multi-second model load scrolls away).
  The scan is skipped on the `--tokens-file` path, where no corpus is read.
- `--normalize-line-endings` collapses CRLF to LF while reading, exactly as text mode does (a lone
  `\r` survives). It is **off by default**. Use it only to reproduce a reference figure produced by
  a *Windows* llama.cpp build reading the same CRLF file; against a Linux run it creates the very
  mismatch it looks like it is fixing. Preferring the LF fixture below is always better, because
  then both engines read identical bytes on every platform.

## The canonical LF fixture

Corpora are fixtures and never live in the repository (CLAUDE.md, *Model & Fixture Storage Rules*).
Build the LF copy once into the shared cache:

```bash
python scripts/make_lf_corpus.py C:/Development/bitnet-tests/data/wikitext-2-raw/wiki.test.raw
# -> ~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw
```

The conversion is byte-level: no decoding, no BOM (a BOM would add a U+FEFF that tokenizes to a real
token), no appended trailing newline. The script prints input/output sizes and CR counts and asserts
that the byte delta equals the CRLF count and that no CRLF remains. Measured on `wiki.test.raw`:
1,299,263 bytes in, 7,249 CR / 7,249 CRLF, **1,292,014 bytes out**. (Issue #506 quotes 1,285,622 —
that is the *decoded character* count, which is smaller than the byte count because wikitext-2
contains multi-byte UTF-8; the byte figure above is the one the script prints.)

Point **both** engines at this file:

```bash
dotllm perplexity <model.gguf> --corpus ~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw
llama-perplexity -m <model.gguf> -f ~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw -c 512
```

An LF file is unchanged by text mode, so the Windows/Linux difference disappears.

## The rigorous protocol: share the token ids

Identical bytes still leave the tokenizers as an assumption. For a comparison that has to hold up,
make llama.cpp state the ids it used and score those:

```bash
llama-perplexity -m <model.gguf> -f <corpus> -c 512 --kl-divergence-base kld.bin
```

`--kl-divergence-base` writes a header that **records the exact token stream llama.cpp scored**
(`tools/perplexity/perplexity.cpp`, little-endian):

| offset | type | field |
|---|---|---|
| 0 | `char[8]` | magic `_logits_` |
| 8 | `int32` | `n_ctx` |
| 12 | `int32` | `n_vocab` |
| 16 | `int32` | `n_chunk` |
| 20 | `int32[n_chunk * n_ctx]` | token ids |

(The per-token log-probability payload follows; the ids are all that is needed here.)

```python
import struct, sys
with open("kld.bin", "rb") as f:
    assert f.read(8) == b"_logits_"
    n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
    ids = struct.unpack(f"<{n_chunk * n_ctx}i", f.read(4 * n_chunk * n_ctx))
open("llama_tokens.txt", "w").write(" ".join(map(str, ids)))
```

```bash
dotllm perplexity <model.gguf> --tokens-file llama_tokens.txt --context 512
```

> **`--tokens-file` alone is not enough on an `add_bos_token` model — pass `--bos` as well.**
> `perplexity.cpp` writes the ids to `kld.bin` *before* the eval loop, and substitutes the
> per-chunk BOS at eval time. The file therefore carries BOS at **index 0 only**, while
> llama.cpp evaluated *every* chunk with BOS at position 0. Scoring those ids without `--bos`
> gives dotLLM no attention sink on any chunk but the first. Measured on Llama-3.2-1B
> (2026-09-24): on a degraded weight set that single omission moved perplexity from 20.31 to
> 21.53 — a 6% phantom "divergence". Check it rather than trusting it: if the file matched
> what llama.cpp evaluated, **every** chunk would start with the BOS id; on an add-BOS model
> only chunk 0 does.

```bash
dotllm perplexity <model.gguf> --tokens-file llama_tokens.txt --context 512 --bos
```

With the ids shared **and** `--bos` set, both engines have provably scored the same tokens, so a
residual difference is in the scoring maths or the kernels and nowhere else. This is the arm that
validated the harness to **−0.0007%** against llama.cpp on wikitext-2 (Q8_0, 64 chunks: 15.1353 vs
15.1354). An earlier "+0.25%" figure for this arm predates the `--bos` correction.

## The BOS trap (issue #515)

> The LF-corpus baseline below is **superseded**. It is kept because the way it failed is the
> point, and because the numbers were cited elsewhere.

`llama-perplexity` prepends **BOS (128000)** to the whole token stream before chunking it on a
Llama-3.2 model. dotLLM's `--corpus` path did not. (Note it is **not** driven by a
`tokenizer.ggml.add_bos_token` key — these GGUFs carry no such key, verified with gguf-py; the
default comes from the tokenizer/pre-type. See #516.) Measured 2026-09-24 by
diffing dotLLM `--dump-tokens` against the ids in llama.cpp's `--kl-divergence-base`:

```
llama  first 12: [128000, 198, 284, 8563, 426, 11206, 466, 284, 15073, 8563, 426, 11206]
dotllm first 12: [        198, 284, 8563, 426, 11206, 466, 284, 15073, 8563, 426, 11206, 466]
4043 of 4096 ids differ;  shift-by-one matches: 4095/4095
```

One token of offset, propagated through every chunk boundary: **chunk N of the two engines
covers different text.** This is the CRLF trap one layer further in — same failure mode, same
consequence, and it survived the LF fixture that was built to end it.

`--bos` does **not** repair a `--corpus` run. It substitutes BOS at each *window* start without
prepending BOS to the stream, so the content stays shifted; it recovers part of the gap, which is
precisely what makes it look like a fix.

**What the offset cost.** Its effect is *weight-set dependent* — which is why the control missed
it. On Llama-3.2-1B the same offset was worth **0.086% on Q8_0 against 4.8% on Q3_K** at 564 chunks,
and **0.5% against 4.0%** at 64 chunks (Q8_0 15.0595 unaligned → 15.1353 aligned). **A healthy control agreeing therefore does not license reading a degraded row as a
property of that quant's path.** That inference, made explicitly below, is wrong in principle.

### Measured: the aligned 564-chunk baseline (2026-09-24)

**This is the citable table.** Both engines provably score the same tokens, over the full corpus.
dotLLM ran plain `--corpus` with **no flags** — since #516 that is aligned by default. Token
streams verified identical end to end with `llama-tokenize --ids --no-escape` vs `--dump-tokens`:
**288,938 ids, 0 differ.** Full working:
`.docs/measurements/2026-09-24-516-564chunk-aligned.md`.

| model | dotLLM | llama.cpp | delta |
|---|---|---|---|
| **Q8_0** (control) | 13.9007 ± 0.10355 | 13.9160 ± 0.10331 | **−0.110%** |
| Q3_K decoded to F32 | 21.2349 ± 0.16442 | 21.0912 ± 0.16308 | **+0.681%** |
| Q3_K | 21.4426 ± 0.16612 | 21.1623 ± 0.16364 | **+1.325%** |

#515 was opened on a Q3_K figure of **+4.79%** and a Q2_K figure of **−8.26%**.

**What these rows do and do not say.** The Q8_0 control agreeing to −0.110% establishes that the
harness, geometry, tokenizer and corpus are common to all three runs. It does **not** license
reading the other rows as kernel properties: the two lower rows are *degraded* models, and a
degraded model amplifies any small difference between the engines by one to two orders of
magnitude. See [Degraded models amplify](#degraded-models-amplify-a-small-fixed-difference) — the
"cost of quantizing" computed from these rows (llama.cpp +0.337%, dotLLM +0.978%, a 2.9× ratio)
**reverses on a healthy model**, where dotLLM is the better of the two at 0.62×. Direct kernel
evidence agrees: dotLLM's Q3_K dot sits at the Q8_K quantization floor and its Q8_K quantizer has
reconstruction error identical to llama.cpp's.

Ruled out along the way, each by measurement rather than inspection: RoPE scaling (this GGUF has
no `rope_scaling` keys at all), KV-cache precision and flash attention (0.03%), FP reduction order
(both engines reproduce to 4 dp across thread counts), and tokenization (288,938 ids, 0 differ).

**On the error bars.** These are per-run standard errors. The measurements are **paired** — same
chunks, same ids — so overlapping individual bars do not make a difference insignificant; the
paired standard error is much smaller. Quote a residual as established only after computing it.

## Degraded models amplify a small fixed difference

**With exact F32 weights in both engines — no quantization anywhere, identical token ids — the
residual scales with how degraded the weights are.** All three models were built by replacing the
112 transformer matmul tensors with llama.cpp's own `gguf-py` decode, `token_embd` kept at Q8_0:

Tested **paired** (same chunks, same ids — `--per-window` against llama.cpp's running per-chunk
series). The Q6_K row is the control that separates *grid coarseness* from *degradation*: it was
Q6_K-quantized from the healthy F32 model and decoded back, so its weights sit on a **coarser
6-bit grid than Q8_0** while the model itself is **not degraded**.

| model (exact F32 both sides) | its PPL | grid | degradation | residual | t (63 df) | 95% CI |
|---|---|---|---|---|---|---|
| Q8_0-derived | 15.124 | 8-bit | 1.00× | +0.029% | **+0.19** | [−0.28%, +0.34%] |
| **Q6_K-derived** | 15.166 | **6-bit** | **1.003×** | +0.095% | **+0.64** | [−0.20%, +0.39%] |
| Q3_K-derived | 23.131 | 3-bit | 1.53× | +0.359% | +1.34 | [−0.17%, +0.89%] |
| Q2_K-derived | 1418.470 | 2-bit | 93.8× | +2.319% | **+3.02** | [+0.81%, +3.85%] |

**On a healthy model the two engines are indistinguishable** (t = 0.19), and that holds for a
coarse 6-bit grid too (t = 0.64) — so grid coarseness is not what drives the residual, degradation
is. Only the destroyed model reaches significance. A ~79× spread in mean-NLL difference from a
fixed pair of implementations doing identical arithmetic on identical inputs.

The same reversal shows in the cost of quantizing:

| model | engine | F32 | quantized | cost | ratio |
|---|---|---|---|---|---|
| Q8_0 | llama.cpp | 15.1236 | 15.1354 | +0.078% | — |
| Q8_0 | dotLLM | 15.1280 | 15.1353 | **+0.048%** | **0.62×** |
| Q3_K | llama.cpp | 23.1314 | 23.2225 | +0.394% | — |
| Q3_K | dotLLM | 23.2145 | 23.4613 | +1.063% | **2.71×** |

dotLLM quantizes *better* than llama.cpp on the healthy model and "worse" on the degraded one. A
kernel defect does not behave that way.

### How to measure quality against llama.cpp, then

**The `quant-ladder` "pure" quantizations are the wrong instrument for engine-vs-engine claims.**
Pure-Q2_K on a 1.2 B model is a destroyed model (PPL ~1400) and pure-Q3_K is well outside what
anyone ships. Their sensitivity is what makes such a comparison look dramatic and mean nothing.

- Prefer a **shipping-grade** quantization (Q4_K_M, Q5_K_M, Q6_K, Q8_0), where the model still
  works. A defect big enough to matter will show there.
- If a degraded quant must be used, **report the F32-decoded control for the same weights**. Only
  the gap between control and quantized row is attributable to the quant path; the control itself
  measures the amplifier. Build one with
  [`scripts/make_f32_decoded_gguf.py`](../scripts/make_f32_decoded_gguf.py) — it replaces every
  tensor of a given quantization type with **llama.cpp's own `gguf-py` decode**, so the control's
  weights do not come from dotLLM:

  ```bash
  python scripts/make_f32_decoded_gguf.py model-Q3_K.gguf model-Q3_K-decoded-F32.gguf --keep-token-embd
  ```

  `--keep-token-embd` leaves the (tied) lm_head quantized, which is usually what you want so the
  control differs from the quantized run only in the transformer matmul tensors.
- **Never read a degraded-model delta as a kernel property without that control.** This is the
  third time in this investigation that a weight-set-dependent effect was read as a fixed one —
  first the BOS offset hiding under a healthy control, then the Q3_K "2.9×", then the F32
  "residual".

Full working: `.docs/measurements/2026-09-24-521-amplification-not-kernel-defects.md`.

### #501's 40-chunk Q3_K rows are the F32-decoded path (#531)

`src/DotLLM.Cpu/Kernels/FastMath.cs` recorded a Q3_K gap to llama.cpp of **+2.03% → +0.28%** (40
chunks, 2026-09-23), against the **+1.03%** / **+1.325%** in the tables above for what looked like
the same fixture. The suspicion when #531 was opened was a BOS omission on the `--tokens-file`
path. **It was not.** That run passed `--bos`, and a misaligned stream could not reproduce
today's default-aligned `--corpus` run to six decimals — which it does. Re-run on `dev`,
2026-09-24, 40-window prefix of a 64-window sweep, `DOTLLM_FAST_EXP` on / off:

| fixture | approximate exp | accurate exp | window 0 (approx) |
|---|---|---|---|
| `Llama-3.2-1B-pure-Q3_K` (packed Q3_K × Q8_K) | 22.5090 | 22.1019 | 11.942101 |
| **`Llama-3.2-1B-pure-Q3_K-decoded-F32`** | **22.2692** | **21.8886** | **11.763354** |

The bold row is #501's two figures and its quoted window 0, to every printed digit. So the two
documents were never measuring the same thing: the #501 run's Q3_K prefill was arithmetically the
F32-decoded path (its own FINDINGS.md says a `llama-quantize … F32` expansion reproduced it
exactly, and that `Gemm` routed Q3_K to `GemmDequantRows`), while everything since runs the packed
dot. The 1.06% between them is the same packed-vs-F32 gap the 64-chunk table above already shows
(23.4613 vs 23.2145).

Consequences: **+0.28% is a valid figure in the decoded-F32 column** — it sits beside this
document's +0.36% / +0.681%, not beside its packed +1.03% / +1.325% — and the sentence
"against llama.cpp the gap goes from +2.03% to +0.28%" is not an engine claim, because only one
side of it was running Q3_K. The tables in this document are the aligned, like-for-like ones.

**A fourth instance of the same lesson, with a new axis.** Each earlier time a number was wrong,
the two sides had scored different *text* (CRLF, then BOS). This time they scored identical text
with different *kernels*. Record which code path produced a figure, not only which file and which
tokens — a fixture name is not a path.

### Measured: what the fast-exp approximation actually costs (#531)

Same protocol, both arms dotLLM (`DOTLLM_FAST_EXP=1` vs off), plain `--corpus`, 64 windows,
paired per window, 63 df. ΔNLL is **fast minus accurate**. Full working:
`.docs/measurements/2026-09-24-531-fastexp-shipping-grade.md`.

| model | fast-exp | accurate | ΔNLL (nats) | PPL % | t | 95% CI on PPL % |
|---|---|---|---|---|---|---|
| **Q4_K_M** (shipping-grade) | 15.7620 | 15.7669 | −0.00031 ± 0.00066 | −0.031% | **−0.47** | **[−0.163%, +0.101%]** |
| Q4_K_M decoded to F32 (control) | 15.7378 | 15.7359 | +0.00012 ± 0.00037 | +0.012% | +0.32 | [−0.063%, +0.086%] |
| Q3_K (pure ladder) | 23.8620 | 23.4613 | +0.01693 ± 0.00178 | +1.708% | +9.50 | [+1.346%, +2.071%] |
| Q3_K decoded to F32 (control) | 23.6171 | 23.2145 | +0.01719 ± 0.00136 | +1.734% | +12.61 | [+1.457%, +2.012%] |

**On a shipping quant it is not resolvable at n = 64** — the answer is a bound (≲0.16%), not a
size. And the Q3_K control is the point of the exercise: it moves by as much as the quantized row
while containing no quantized matmul, so the famous "−1.71% on Q3_K" is a property of *degraded
weights*, not of the Q3_K path. This is the amplification rule applying to a claim that was not
an engine comparison at all — both arms were dotLLM — which is worth noting, because the rule was
written for engine-vs-engine work and generalizes further than that.

### Superseded: the first corrected baseline (2026-09-24, 64 chunks)

Kept because it is what the `--tokens-file --bos` protocol produces and is a useful cross-check;
the 564-chunk table above is the one to cite.


Both engines scoring **identical ids with BOS at every window start** — dotLLM fed the ids from a
`--chunks 64 --kl-divergence-base` run via `--tokens-file … --bos`. 64 chunks, 16,320 scored
tokens. Full working: `.docs/measurements/2026-09-24-515-q3k-investigation.md`.

| quant | dotLLM | llama.cpp | delta | bars |
|---|---|---|---|---|
| **Q8_0** (control) | 15.1353 ± 0.33711 | 15.1354 ± 0.33582 | **−0.0007%** | overlap |
| Q2_K | 1446.4130 ± 46.14533 | 1409.1593 ± 44.81472 | +2.64% | overlap |
| Q3_K | 23.4613 ± 0.54245 | 23.2225 ± 0.53566 | +1.03% | overlap |
| Q3_K decoded to F32 | 23.2145 ± 0.53639 | 23.1314 ± 0.53336 | +0.36% | overlap |

**Nothing is disjoint any more.** Every row has the same sign — dotLLM marginally higher — and
every row's bars overlap. The Q3_K gap went from +4.79% to +1.03%; **the Q2_K gap changed sign**,
from −8.26% to +2.64%. The "opposite signs" that #515 was opened to explain were an artifact of
the offset; there were never two signs to reconcile.

The fourth row is the strongest single result: all 112 Q3_K tensors replaced by their **exact F32
decode** (produced by llama.cpp's own `gguf-py`, not by dotLLM), so no quantization is involved in
any matmul — and the residual is +0.36%. Whatever is left is not the quantized path.

**What was ruled out along the way**, each by measurement rather than inspection:

| hypothesis | verdict |
|---|---|
| dotLLM dequantized where llama.cpp used its packed `× q8_K` dot (#515's first check) | **refuted** — forcing either path moves Q3_K by 0.16% |
| dotLLM's Q3_K block decode is wrong | **refuted** — bit-exact vs `gguf-py` on 21M elements |
| Q8_K activation quantization explains the 4% gap | **refuted** — the dequant arm bypasses it entirely and moves Q3_K by 0.16%. It remains the natural candidate for the ≲0.7% *residual*: dotLLM's packed−F32 gap is +1.06% (23.4613 → 23.2145) against llama.cpp's +0.39% (23.2225 → 23.1314) |
| FP reduction order / thread partitioning | **refuted** — both engines reproduce to 4 dp across thread counts |
| KV-cache precision, flash attention | **refuted** — `-ctk f32 -ctv f32 -fa off` moves llama.cpp by 0.03% |

### Superseded: the LF-corpus baseline (2026-09-23)

The first end-to-end comparison taken **after** #506, with both engines reading the same LF bytes.
Llama-3.2-1B "pure" quantizations (every tensor at the named format) from
`~/.dotllm/quant-ladder/Llama-3.2-1B-pure/`, `wiki.test.lf.raw` (1,292,014 bytes, **0 CR**), CPU
both sides, `-c 512` / `--context 512` passed explicitly rather than relying on the two defaults
agreeing. 564 chunks, 143,820 scored tokens each. Raw logs:
`.docs/measurements/2026-09-23-lf-corpus-triad.out`.

| quant | dotLLM | llama.cpp | delta | bars |
|---|---|---|---|---|
| **Q8_0** (control) | 13.9040 ± 0.10359 | 13.9160 ± 0.10331 | **−0.086%** | overlap |
| Q2_K | 1342.0853 ± 14.52088 | 1462.8547 ± 15.76961 | **−8.26%** | **disjoint** |
| Q3_K | 22.1764 ± 0.17231 | 21.1623 ± 0.16364 | **+4.79%** | **disjoint** |

> **Every delta in the table above is invalid** — the two engines' chunks covered different
> text (see *The BOS trap*). The paragraphs that follow are kept as written, because the
> reasoning error in them is instructive, but **none of their conclusions hold.**

**Read the Q8_0 row first.** It is the control, and at −0.086% the two engines agree far inside
their error bars. That is what makes the other two rows interpretable: the harness, the chunk
geometry, the tokenizer and the corpus are all common to the three runs, so a disjoint delta at
Q2_K or Q3_K is a property of *that quant's path*, not of the measurement. Before #506 no such
statement was possible, because each engine was tokenizing different text.

> **This is the sentence that failed.** The control *was* common to the three runs, and it *did*
> agree — but the defect it was meant to catch scales with how degraded the weights are, so it
> hid under Q8_0 and dominated Q2_K/Q3_K. A control only licenses reading the other rows if it
> is sensitive to the errors you are worried about; this one was not, and nothing in the
> agreement itself said so.

**The pre-#506 claims were right about the sign and wrong about the size.** Q2_K was recorded as
−2.9% and is −8.26%; #501's Q3_K was recorded as +8.9% and is +4.79%. Neither gap was a CRLF
artifact — both survive on identical bytes — so the underlying divergences are real and remain
open (issue #515).

> Wrong on both counts. Under the aligned protocol Q2_K is **+2.64%**, not −8.26% — the sign
> flipped — and Q3_K is **+1.03%**, not +4.79%. The gaps did not "survive on identical bytes":
> identical *bytes* were never the requirement, identical *tokens* were.

**Caveat on the Q2_K row.** Pure-Q2_K on a 1.2 B model is a destroyed model: perplexity 1342
against a 13.90 Q8_0 baseline, ~97× worse. In that regime perplexity is hypersensitive to small
numeric differences, so the 8.26% gap does **not** imply a numeric error of similar magnitude, and
it is the weakest of the three rows as evidence about kernel correctness. The Q3_K row (22.18 vs
13.90 — a plausible degradation for the format) is the more trustworthy signal.

**The signs are opposite** — dotLLM is *better* at Q2_K and *worse* at Q3_K. That rules out a
single *shared-path* precision difference, since higher-precision accumulation on a common path
would move both the same way. It does **not** rule out the engines taking *different paths*: if
dotLLM's run dequantized to F32 where llama.cpp used its packed `× q8_K` dot, one mechanism could
produce both signs (quantized activations cost most where the weights are already coarsest). That
is #515's first check, and it must be settled before anyone reads this table as two separate
kernel bugs.

> #515's first check was run on 2026-09-24 and is **negative**: forcing either path changes Q3_K
> by 0.16%, because `TransformerModel.Gemm` already dispatches the packed `× q8_K` dot at prefill
> exactly as llama.cpp does. The signs were never opposite — Q2_K's sign was an artifact. Note
> what this paragraph did: it reasoned at length about which mechanism could explain a pattern,
> without first checking that the pattern was real.

**On the raw log's `n_ctx`.** llama.cpp reports `n_ctx = 2048, n_ctx_seq = 512, n_seq = 4` — it
batches four sequences of the requested 512. The per-chunk context is 512 on both sides, and both
engines report **564 chunks**, so the geometries match despite the differing header line.

## Reporting a comparison

Always report, alongside the figure: model file and quantization, corpus **and its line endings**,
`--context`, `--stride`, `--unscored-prefix`, scored-token count, and the standard error. A
perplexity without the window geometry is not comparable to anything, and — per #506 — a
perplexity without the corpus's line endings is not comparable to a llama.cpp number at all.

Compare error bars before calling a difference real: a 3% "residual" once turned out to sit inside
llama.cpp's own ±6.5% on a small corpus.

## Nemotron-Nano-9B-v2 re-measure on the LF corpus (issue #514, 2026-09-29)

Supersedes any pre-#372 Nemotron-H quality figure (RoPE was wrongly applied then) and any figure
where each engine tokenized the corpus itself (#506).

**Method.** llama.cpp b9672 Vulkan (`-ngl 99`, `-c 512 --chunks 32`, `--kl-divergence-base`) on
`wiki.test.lf.raw`; the token ids it recorded (identical for Q8_0 and Q4_K_M) were fed to
`dotllm perplexity --tokens-file ... --context 512 --bos` on **CPU**. 32 chunks, 8,160 scored
tokens, bartowski GGUFs.

| quant | llama.cpp (Vulkan) | dotLLM (CPU) | dotLLM vs llama.cpp |
|---|---|---|---|
| Q8_0 (control) | 7.4772 +/- 0.215 | 7.4436 +/- 0.214 | -0.45% |
| Q4_K_M | 7.5464 +/- 0.217 | 7.5209 +/- 0.217 | -0.34% |

**Reading.** The offset has the same sign and size on the control and the quant under test, so it
is not a quant-path problem. Paired per chunk (Q4_K_M, 32 windows) the mean dNLL is -0.0034 nats
(se 0.0015, z = -2.3; dotLLM lower on 19/32 windows), with paired sd 0.008 nats versus ~0.2
unpaired: the engines track each other closely and the residual is small. It is not resolved
whether -0.3..-0.45% is a real systematic difference or noise at this sample size; 32 chunks
cannot settle it.

**Limits.** (1) The dotLLM arm is **CPU**: `--device vulkan` cannot run sliding-window because the
Vulkan model returns only the last row, and teacher-forced scores one window, so this does not
validate the Vulkan forward pass. (2) Q8_0 was not run with `--per-window`, so only Q4_K_M is
paired. (3) Bonsai is not yet re-measured (tracked on #514).

### Ternary-Bonsai-2-27B PQ2_0 (issue #514, 2026-09-29)

prism-ml `llama.cpp` fork (Vulkan, `-ngl 99`, `-c 512`, `--kl-divergence-base`) is the only
reference that reads PQ2_0. **No Q8_0 control exists for this model**, so a disagreement could not
be split into quant-path versus harness; the agreement below is the only evidence available.

| arm | chunks | PPL |
|---|---|---|
| prism llama.cpp (Vulkan), running estimate after chunk 4 | 4 | 8.0693 |
| dotLLM CPU, shared ids, **no `--bos`** | 4 | 8.0681 +/- 0.683 (-0.015%) |
| dotLLM CPU, shared ids, `--bos` (wrong for this model) | 4 | 8.1603 (+1.1%) |

Chunk 0: 6.3198 (dotLLM) vs 6.3201 (llama.cpp).

**Do not pass `--bos` blindly with `--tokens-file`.** The rule above (BOS at index 0 of the file)
applies to `add_bos_token` models. Bonsai's recorded stream starts `198, 16, ...` (no BOS), and
adding one cost 1.1% -- the same size of phantom divergence the BOS trap produced on Llama. Check
`ids[0]`/`ids[n_ctx]` in the kld header first.

**Limits.** Only 4 of the 16 requested chunks: prism's Vulkan prefill ran ~27 min per 4-chunk
pass (ETA 1 h 48 min for 16), so the run was stopped and the 4 already-scored chunks used. 4
chunks is a smoke-level agreement (sd ~0.7), not a tight bound. dotLLM ran on CPU (see Nemotron
limits above), so the Vulkan Bonsai forward pass is not covered.

### Nemotron offset resolved: it is llama.cpp Vulkan vs CPU, not dotLLM (issue #514, 2026-09-29)

Q8_0, 32 chunks, same ids, per-chunk dNLL (paired):

| pair | mean dNLL | se | sd | z |
|---|---|---|---|---|
| dotLLM CPU - llama.cpp **CPU** (`-dev none -ngl 0`, PPL 7.4459) | -0.0003 | 0.0004 | 0.0023 | -0.75 |
| dotLLM CPU - llama.cpp Vulkan (PPL 7.4772) | -0.0045 | 0.0015 | 0.0085 | -3.0 |
| llama.cpp CPU - llama.cpp Vulkan | -0.0042 | 0.0014 | 0.0079 | -3.0 |

dotLLM CPU (7.4436) agrees with llama.cpp CPU (7.4459) to 0.03%, with a 4x tighter paired sd. The
"-0.45%" in the table above is the difference between llama.cpp's own Vulkan and CPU backends
(24/32 chunks lower on CPU), which dotLLM merely sits on the CPU side of. The earlier
"unresolved whether real" caveat is closed for the CPU path.

### Vulkan sliding-window perplexity (issue #564, 2026-09-29)

`dotllm perplexity --device vulkan` now runs the full llama.cpp-comparable protocol for
**Nemotron-H** and the **hybrid-dense (Bonsai)** family. The CLI calls
`IModel.TrySetAllRowLogitsLimit(context + 1)`; the model then runs its LM head over every row of a
window (opt-in, so normal prefill still pays for one row) and `BackendPerplexityModel.Probe` is told
how many rows it must declare. Nothing else changes. The dense `VulkanTransformerModel` (Llama-style)
now has it too (same opt-in, via the
all-position head the DiffusionGemma path already used). Adopting that head exposed a **missing
compute barrier** between the last layer's residual add and the final RMSNorm: before the fix every
row was off by ~1e-2 against the CPU oracle, after it 1e-8..1e-7. The barrier is now in
`FinishDiffusionForward`, so the DiffusionGemma Vulkan path gets it as well. Llama-3.2-1B-Instruct
Q8_0, 16 chunks of the LF corpus: Vulkan **15.8620** vs CPU 15.8663 (-0.03%), 4 s vs 50 s.

Nemotron-Nano-9B-v2 Q8_0, same ids + `--bos`, 32 chunks (paired per-chunk dNLL, nats):

| pair | mean | se | sd |
|---|---|---|---|
| dotLLM **Vulkan** - dotLLM CPU | -0.0002 | 0.0006 | 0.0032 |
| dotLLM **Vulkan** - llama.cpp CPU | -0.0005 | 0.0005 | 0.0026 |
| dotLLM Vulkan - llama.cpp Vulkan | -0.0047 | 0.0013 | 0.0071 |

PPL: dotLLM Vulkan **7.4422**, dotLLM CPU 7.4436, llama.cpp CPU 7.4459, llama.cpp Vulkan 7.4772.
The dotLLM Vulkan forward pass agrees with both CPU implementations to within noise; llama.cpp's own
Vulkan backend is the outlier (see above). Elapsed 146 s versus 927 s on CPU.

Bonsai-2 27B PQ2_0 (no `--bos`, 16 chunks): dotLLM Vulkan **10.2348 +/- 0.435** in 47 s. Windows 0-3
match the dotLLM CPU arm to ~1e-4 relative (6.31994 / 11.90538 / 7.41143 / 7.59850 vs 6.31976 /
11.90547 / 7.41158 / 7.59837) and prism llama.cpp's running estimate. No 16-chunk llama.cpp reference
exists: the fork's Vulkan and CPU backends both failed to finish it in a practical time.

### #514 closing table (2026-09-29)

Nemotron-Nano-9B-v2, LF corpus, shared llama.cpp token ids, `--bos`, 32 chunks of 512:

| quant | llama.cpp CPU | llama.cpp Vulkan | dotLLM CPU | dotLLM Vulkan |
|---|---|---|---|---|
| Q8_0 (control) | 7.4459 | 7.4772 | 7.4436 | 7.4422 |
| Q4_K_M | not run | 7.5464 | 7.5209 | 7.5090 |

Q8_0: all three of dotLLM CPU / dotLLM Vulkan / llama.cpp CPU agree to <= 0.05%; only llama.cpp's
Vulkan backend departs (+0.42%). Q4_K_M: dotLLM Vulkan sits 0.16% below dotLLM CPU, a larger split
than the control shows; it was not investigated (single-figure, 32 chunks, se ~0.2 unpaired) and
llama.cpp's own CPU Q4_K_M was not run, so it is not established whether that is quant-path
numerics (Q4_K MMQ activation quantisation) or noise. Bonsai PQ2_0 (no Q8_0 control exists):
dotLLM Vulkan 10.2348 +/- 0.435 over 16 chunks; the fork's running estimate over its first 4 chunks
(8.0693) matches dotLLM CPU/Vulkan on those chunks (8.0681) to 0.015%.

### The Nemotron Q4_K_M Vulkan offset, explained (issue #568, 2026-09-30)

llama.cpp's own CPU scores Q4_K_M at 7.5217, dotLLM CPU 7.5209 (paired dNLL -0.0001, sd 0.0024:
identical), dotLLM Vulkan 7.5090 (paired -0.0016 vs both CPUs, z = -3.1, sd 0.003). The Vulkan split
is real and systematic, and the mechanism is visible in the weights: this "Q4_K_M" is mostly **Q5_0**
(`ffn_up`, `ssm_in`, `attn_q/k/v`), with Q4_K only on `attn_output`/`ssm_out`. The CPU multiplies Q5_0
by activations **quantised to Q8_1**; Vulkan multiplied F32-expanded weights by F32 activations, so it
carries less activation-quantisation noise and reads slightly lower. That is consistent with the data
but was **not isolated** (no F32-activation CPU reference was run).

Side finding, fixed in #568: `VulkanNemotronHWeights` expanded every Q5_0 tensor to F32 at upload
("no kernel in tree") although the Q5_0 kernels have existed since #344. Q5_0 is now kept packed and
a new 128x128 blocked coopmat Q5_0 GEMM (`matmul_q5_0_f32_gemm_coopmat_128x128x4`) serves prefill.
Nemotron-Nano-9B Q4_K_M, `bench -p 512`, same session:

| | prefill tok/s | decode tok/s |
|---|---|---|
| F32-expanded Q5_0 (before) | 34.5 | 4.8 |
| packed Q5_0, tiled F32 GEMM | 24 | 14.0 |
| packed Q5_0, **coopmat GEMM** | **66** | **14.1** |

Perplexity is unchanged (7.5091 vs 7.5090; per-window max |dNLL| 1.4e-4, the F16 operand floor) and
the 32-chunk run drops from 474 s to 268 s. `DOTLLM_VK_NEMOTRONH_Q5_0_F32=1` restores the F32
expansion and `DOTLLM_VK_Q5_0_GEMM_LEGACY=1` the tiled GEMM, for A/B.

Follow-up (#570): the same blocked coopmat template for **Q4_K** (`matmul_q4_k_gemm_coopmat_128x128x4`,
wired into Nemotron-H prefill for >= 32 rows) takes Nemotron-Nano-9B Q4_K_M from 67 to **115 tok/s**
pp512 (Q8_0, already coopmat, is 120), decode unchanged at 14.2 tok/s. Perplexity is unchanged
(7.5090; per-window max |dNLL| 1.6e-4) and the 32-chunk run is now 152 s (474 s before #568).
`DOTLLM_VK_Q4_K_GEMM_LEGACY=1` restores the tiled GEMM. Q5_K/Q6_K/Q2_K/Q3_K prefill GEMMs are the same
shape of opportunity and are not done.

### Nemotron-H Mamba2 scan: the real prefill/decode bottleneck (issue #572, 2026-09-30)

Attribution by perturbation, not guesswork: skipping the `mamba2_selective_scan` dispatch (garbage
output, valid timing) took Nemotron-Nano-9B Q8_0 pp512 from **4289 ms to 1047 ms** and decode from
17.4 to 22.7 tok/s. The scan was ~75% of prefill. It is not the GEMMs (Q8_0 was already coopmat) and not
the 3 x seqLen per-token `vkCmdCopyBuffer` loops in the SSM layer (batching them into one call each was
measured and changed nothing: 4263 -> 4273 ms; reverted).

Cause: the reference kernel runs one 64-thread workgroup per head and, for every token, reads and
writes all `dState` floats of every state row to global memory; at headDim 80 it also idles 16 of 64
lanes. The new `mamba2_selective_scan_f32_regs` keeps each row's state in registers for the whole
sequence (global memory touched at entry and exit only), splits the 128-wide state across 4 lanes with
a two-step `subgroupShuffleXor` reduction, and runs `nHead * headDim / 16` workgroups so every lane is
busy. Eligible when `dState == 128` and `headDim % 16 == 0`; otherwise the reference kernel runs.
`DOTLLM_VK_MAMBA2_SCAN_LEGACY=1` forces the reference.

`bench -p 512 -n 16`, same session, A/B via the env var:

| model | pp512 tok/s before -> after | tg tok/s |
|---|---|---|
| Nemotron-Nano-9B Q8_0 | 119 -> **480** (4.0x) | 17.4 -> 21.6 |
| Nemotron-3-Nano-4B Q4_K_M | 159 -> **363** (2.3x) | 17.3 -> 19.6 |
| Nemotron-Nano-9B Q4_K_M (with #568 and #570) | 34.5 at session start -> **419** | 4.8 -> 17.0 |

Correctness: 32-chunk perplexity on the 9B Q8_0 control is 7.4423 (7.4422 before), per-window max
|dNLL| 2.1e-4 versus the previous Vulkan run, and the arm now takes 43 s instead of 146 s. Six kernel
parity tests against the CPU reference, including the 128-head/80-dim/8-group shape of the real model;
dropping the second shuffle-reduction step makes 6 of 7 fail.

### Decode GEMVs: Q5_0 and Q4_K were still uncoalesced (issue #574, 2026-09-30)

Q8_0 got the coalesced lane = K-position GEMV in #471; Q5_0 and Q4_K kept the block-per-lane layout,
where the lanes of a wave read words 22 / 144 bytes apart. Nemotron-Nano-9B Q4_K_M therefore decoded at
17.0 tok/s although it moves fewer bytes than Q8_0 (21.6 tok/s at ~205 GB/s, i.e. bandwidth-bound).
Porting the layout to both (`matmul_q5_0_f32_gemv_coalesced`, `matmul_q4_k_gemv_f32_coalesced`; F32
activations, same bindings and push constants, chosen when the device has subgroup arithmetic):

| Q5_0 GEMV | Q4_K GEMV | decode tok/s (`bench -p 128 -n 48`) |
|---|---|---|
| legacy | legacy | 16.97 |
| coalesced | legacy | 19.21 |
| legacy | coalesced | 24.54 |
| coalesced | coalesced | **29.62** (+75%) |

Prefill is unchanged (it uses the GEMMs). Perplexity cannot see a decode-path change, so it was
checked by greedy prefill + 24 decode steps on the real model: identical token ids, max logit
difference at the chosen tokens 2.4e-6. Random-block parity tests against the CPU dequantiser cover
both variants of each kernel (`DOTLLM_VK_Q5_0_GEMV_LEGACY=1`, `DOTLLM_VK_Q4_K_GEMV_LEGACY=1` restore
the old ones).

Q6_K followed (#576): `matmul_q6_k_gemv_f32_coalesced` (a lane owns 4 consecutive `l` and produces 16
outputs; every field is read as uint16 because the 210-byte super-block alternates word alignment).
Nemotron-3-Nano-4B Q4_K_M decode **25.6 -> 65.4 tok/s (2.55x)**, ~183 GB/s, i.e. bandwidth-bound; the
4B's 0.35 G Q6_K elements were the bulk of the excess. Greedy decode identical over 24 steps (max logit
difference 4.8e-6) with all three coalesced GEMVs on versus all three legacy. `DOTLLM_VK_Q6_K_GEMV_LEGACY=1`
restores the old one. Q5_K / Q2_K / Q3_K and the IQ* F32 GEMVs have not been ported.

### Q6_K prefill GEMM (issue #578, 2026-09-30)

After the scan fix the 4B model's prefill (Nemotron-3-Nano-4B Q4_K_M, 360 tok/s vs llama.cpp's 1759) no
longer depended on the scan (skip: 1392 -> 1396 ms). Perturbing each matmul type instead: skipping Q6_K
took pp512 from 1396 ms to **489 ms**, i.e. Q6_K was **65%** of prefill while holding 9% of the
parameters (Q5_0 280 ms, Q4_K 125 ms). Q6_K was the last quant on Nemotron-H's prefill path still using
the 16x16 tiled F32 GEMM. `matmul_q6_k_gemm_coopmat_128x128x4` (a 32-element chunk is one
(half, group) of a super-block; two lanes per row, one scale each):

| | pp512 tok/s | wall for 16 ppl chunks |
|---|---|---|
| tiled F32 GEMM (`DOTLLM_VK_Q6_K_GEMM_LEGACY=1`) | 363 | 52.9 s |
| blocked coopmat | **930** (2.56x) | 24.6 s |

Perplexity 11.6145 vs 11.6143 (identical to F16 operand rounding). Random-block parity tests; swapping
the nibble half makes 7 of 8 fail.

Current Nemotron-H standing against llama.cpp (Vulkan, b9672), pp512 / tg64, tok/s:

| model | dotLLM | llama.cpp |
|---|---|---|
| Nano-9B Q8_0 | 476 / 21.6 | 764 / 21.8 |
| Nano-9B Q4_K_M | 419 / 29.5 | 789 / 29.9 |
| Nemotron-3-Nano-4B Q4_K_M | 930 / 65.2 | 1759 / 66.9 |

Decode is at parity on all three; prefill is at 53-62%.

### Last-row-only backends: growing-prefix sliding window (issue #793, 2026-10-07)

Vulkan hybrids (`qwen35moe` etc.) return only the final logits row, so sliding-window mode used to refuse
`--device vulkan`. It now scores each window's targets `[unscored-prefix, context)` by re-prefilling the growing
prefix (positions from 0, state reset per target, same BOS/geometry rules as the all-rows path). Cost is O(n^2)
forwards, but only the scored half of each window is replayed. Use a small `-c` (e.g. `-c 256 --stride 256
--unscored-prefix 129`, which is exactly llama.cpp's `n_ctx/2 - 1 = 127` scored targets per chunk).

Qwen3.5-122B-A10B Q4_K_M (unsloth, 71.3 GiB), `wiki.test.lf.raw`, first 1,536 tokens, 6 chunks x 127 scored tokens
(762 total; +/- ~0.58 = the sample is small, compare per chunk):

| arm | PPL | per-chunk PPL |
|---|---|---|
| dotLLM Vulkan (this change) | 6.0445 +/- 0.584 | 6.901 7.040 9.519 7.961 8.124 1.631 |
| llama.cpp b9016 **CPU** (`-dev none -ngl 0`, shards direct) | 6.0470 +/- 0.585 | 6.905 7.019 9.560 7.970 8.083 1.638 |
| llama.cpp b9016 Vulkan (`-ngl 99`) | 6.0570 +/- 0.585 | 6.993 7.039 9.567 7.944 8.115 1.627 |

dotLLM Vulkan is within -0.44..+0.50 % of the llama.cpp CPU oracle on every chunk (aggregate -0.04 %); llama.cpp's own
Vulkan sits +0.16 % off its CPU and -1.3 % off on chunk 0 (cf. #568). The very low chunk 5 is real text (all three agree).
