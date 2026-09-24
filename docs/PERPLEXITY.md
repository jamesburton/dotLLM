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
[corrected baseline](#measured-the-corrected-baseline-2026-09-24). Under the aligned protocol the
Q8_0 control agrees to **−0.0007%** and no row is disjoint. The Bonsai and Nemotron-H claims are
still unverified and are tracked as #514.

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
> 21.53 — a 6% phantom "divergence". Check it rather than trusting it: the first id of each
> chunk in `kld.bin` should be the BOS id, and on an `add_bos_token` model it will not be.

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

Llama-3.2 sets `add_bos_token`, so `llama-perplexity` prepends **BOS (128000)** to the whole
token stream before chunking it. dotLLM's `--corpus` path did not. Measured 2026-09-24 by
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
it. On Llama-3.2-1B, 64 chunks, the same offset was worth ~0.1% on Q8_0 and ~4% on a degraded
Q3_K model. **A healthy control agreeing therefore does not license reading a degraded row as a
property of that quant's path.** That inference, made explicitly below, is wrong in principle.

### Measured: the corrected baseline (2026-09-24)

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
| Q8_K activation quantization | **refuted** — the dequant arm bypasses it entirely |
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
