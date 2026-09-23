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
2026-09-23 on `wiki.test.raw` (1,299,263 bytes, 13,641 CRs) with Llama-3.2-1B: **501 of the 512
tokens in chunk 0 differed.** The paired per-chunk standard deviation on this corpus is ~0.2 nats,
an order of magnitude larger than the quantization effects usually being chased, so any aggregate
agreement observed that way was luck rather than validation.

Every end-to-end dotLLM-vs-llama.cpp perplexity comparison in which each side tokenized the file
itself is suspect. Comparisons where dotLLM was fed llama.cpp's exact token ids are not affected.

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
that the byte delta equals the CRLF count and that no CRLF remains — for `wiki.test.raw` the delta
is 13,641 bytes, giving 1,285,622 bytes out.

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

Both engines have now provably scored the same ids, so a residual difference is in the scoring
maths or the kernels and nowhere else. This is the arm that validated the harness to +0.25% against
llama.cpp on wikitext-2.

## Reporting a comparison

Always report, alongside the figure: model file and quantization, corpus **and its line endings**,
`--context`, `--stride`, `--unscored-prefix`, scored-token count, and the standard error. A
perplexity without the window geometry is not comparable to anything, and — per #506 — a
perplexity without the corpus's line endings is not comparable to a llama.cpp number at all.

Compare error bars before calling a difference real: a 3% "residual" once turned out to sit inside
llama.cpp's own ±6.5% on a small corpus.
