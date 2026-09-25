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
