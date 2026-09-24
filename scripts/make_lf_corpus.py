#!/usr/bin/env python3
"""Produce the canonical LF copy of a perplexity corpus (issue #506).

A CRLF corpus is read differently by the two engines a quality comparison involves:
``llama-perplexity`` built with MSVC opens its prompt file in *text mode*, so the CRT collapses
``\\r\\n`` to ``\\n`` before tokenizing, while dotLLM reads the bytes as they are. Measured on
``wiki.test.raw`` with Llama-3.2-1B, 501 of the 512 tokens in chunk 0 differed — the two engines
were scoring different text.

The fix is the fixture, not the reader (llama.cpp on Linux keeps the CRs too). This script writes
an LF copy into the shared cache; per CLAUDE.md's Model & Fixture Storage Rules, corpora never
live in the repository.

    python scripts/make_lf_corpus.py C:/Development/bitnet-tests/data/wikitext-2-raw/wiki.test.raw

Default output:
    ~/.dotllm/test-cache/corpora/<parent-dir-name>/<stem>.lf<suffix>
e.g. ~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw

The conversion is byte-level on purpose: no text decoding, no BOM, no appended trailing newline.
A BOM would add a U+FEFF that tokenizes to a real token and would reintroduce exactly the class of
mismatch this script exists to remove.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CR = 0x0D
LF = 0x0A


def crlf_to_lf(data: bytes) -> bytes:
    """Collapse CRLF to LF, leaving a lone CR intact — what MSVC text mode does."""
    return data.replace(b"\r\n", b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write an LF copy of a corpus into ~/.dotllm/test-cache/corpora/.")
    parser.add_argument("source", type=Path, help="Corpus to convert (e.g. wiki.test.raw).")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Explicit output path.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    args = parser.parse_args(argv)

    src: Path = args.source
    if not src.is_file():
        print(f"error: corpus not found: {src}", file=sys.stderr)
        return 1

    if args.output is not None:
        dst = args.output
    else:
        # A distinct name, so an LF fixture can never be mistaken for the CRLF original.
        dst = (
            Path.home()
            / ".dotllm"
            / "test-cache"
            / "corpora"
            / src.parent.name
            / f"{src.stem}.lf{src.suffix}"
        )

    if dst.exists() and not args.force:
        print(f"error: {dst} already exists (pass --force to overwrite)", file=sys.stderr)
        return 1

    data = src.read_bytes()
    converted = crlf_to_lf(data)

    crs_in = data.count(bytes([CR]))
    crlfs_in = data.count(b"\r\n")
    crs_out = converted.count(bytes([CR]))

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(converted)

    print(f"source : {src}  ({len(data):,} bytes, {crs_in:,} CR, {crlfs_in:,} CRLF)")
    print(f"output : {dst}  ({len(converted):,} bytes, {crs_out:,} CR)")
    print(f"removed: {len(data) - len(converted):,} bytes")

    # Self-check: the only bytes that may disappear are the CRs of a CRLF, and no CRLF may remain.
    ok = True
    if len(data) - len(converted) != crlfs_in:
        print("FAIL: byte delta does not equal the CRLF count", file=sys.stderr)
        ok = False
    if converted.count(b"\r\n") != 0:
        print("FAIL: output still contains CRLF", file=sys.stderr)
        ok = False
    if crs_out != crs_in - crlfs_in:
        print("FAIL: lone-CR count changed", file=sys.stderr)
        ok = False
    if converted[:3] == b"\xef\xbb\xbf" and data[:3] != b"\xef\xbb\xbf":
        print("FAIL: a BOM was introduced", file=sys.stderr)
        ok = False

    if not ok:
        return 2

    print("verified: CRLF -> LF only; byte delta == CRLF count; no CRLF remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
