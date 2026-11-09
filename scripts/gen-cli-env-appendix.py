#!/usr/bin/env python3
"""Regenerates the tuning/diagnostic environment-variable appendix of docs/CLI.md.

Scans src/**/*.cs for "DOTLLM_*" string literals (the names code reads or documents), groups them by
owning project, and pulls a one-line description from the nearest XML-doc or // comment that
mentions the variable. Variables the curated "user-facing" tables in docs/CLI.md already cover are
skipped.

    python scripts/gen-cli-env-appendix.py > .docs/cli-env-appendix.md   # then paste between the markers

The block between <!-- BEGIN env-appendix --> and <!-- END env-appendix --> in docs/CLI.md is
this script's output verbatim; run `python scripts/gen-cli-env-appendix.py --write` to splice it in.
"""
import os
import re
import sys
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "CLI.md")
BEGIN, END = "<!-- BEGIN env-appendix -->", "<!-- END env-appendix -->"

LIT = re.compile(r'"(DOTLLM_[A-Z0-9_]*[A-Z0-9])"')


def curated_names(doc_text: str) -> set:
    """Names already documented outside the appendix (the user-facing tables)."""
    if BEGIN in doc_text:
        doc_text = doc_text.split(BEGIN)[0] + doc_text.split(END)[-1]
    return set(re.findall(r"DOTLLM_[A-Z0-9_]*[A-Z0-9]", doc_text))


def clean(s: str) -> str:
    s = re.sub(r"<see cref=\"([^\"]+)\"\s*/>", r"\1", s)
    s = re.sub(r"<paramref name=\"([^\"]+)\"\s*/>", r"\1", s)
    s = re.sub(r"</?[a-zA-Z]+[^>]*>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("|", "\\|")
    for a, b in (("→", "->"), ("–", "-"), ("—", "-"), ("×", "x"), ("≥", ">="), ("≤", "<="),
                 ("≈", "~"), ("’", "'"), ("“", '"'), ("”", '"'), ("µ", "u"), ("�", "-")):
        s = s.replace(a, b)
    return s.encode("ascii", "ignore").decode("ascii")


def describe(lines, i, var):
    # nearest comment block above line i (within 8 lines, so a member's /// summary counts for its body)
    j = i
    while j >= 0 and i - j <= 8 and not lines[j].strip().startswith("//"):
        j -= 1
    block = []
    if j >= 0 and lines[j].strip().startswith("//") and i - j <= 8:
        while j >= 0 and lines[j].strip().startswith("//"):
            block.insert(0, lines[j].strip().lstrip("/").strip())
            j -= 1
    text = clean(" ".join(block))
    if len(text) > 15:
        return text
    m = re.search(r"//+\s*(.*)$", lines[i])
    if m and len(m.group(1)) > 15:
        return clean(m.group(1))
    return ""


def main():
    found = collections.defaultdict(list)
    for r, _, fs in os.walk(os.path.join(ROOT, "src")):
        parts = r.split(os.sep)
        if "bin" in parts or "obj" in parts:
            continue
        for f in fs:
            if not f.endswith(".cs"):
                continue
            p = os.path.join(r, f)
            lines = open(p, encoding="utf8", errors="ignore").read().split("\n")
            rel = os.path.relpath(p, os.path.join(ROOT, "src")).replace(os.sep, "/")
            for i, l in enumerate(lines):
                for v in LIT.findall(l):
                    found[v].append((rel, i, lines))
    doc = open(DOC, encoding="utf8").read() if os.path.exists(DOC) else ""
    skip = curated_names(doc)
    groups = collections.defaultdict(list)
    for v in sorted(found):
        if v in skip:
            continue
        best = ""
        for rel, i, lines in found[v]:
            d = describe(lines, i, v)
            if len(d) > len(best):
                best = d
        rel0 = found[v][0][0]
        proj = rel0.split("/")[0].replace("DotLLM.", "")
        if len(best) > 230:
            best = best[:227].rstrip() + "..."
        groups[proj].append((v, best or "(not described in source)", rel0.split("/")[-1]))
    out = [BEGIN, ""]
    for proj in sorted(groups):
        out.append(f"#### {proj}")
        out.append("")
        out.append("| Variable | What the source says | Defined in |")
        out.append("|---|---|---|")
        for v, d, f in groups[proj]:
            out.append(f"| `{v}` | {d} | `{f}` |")
        out.append("")
    out.append(END)
    text = "\n".join(out)
    if "--write" in sys.argv:
        if BEGIN not in doc:
            sys.exit("markers not found in docs/CLI.md")
        pre = doc.split(BEGIN)[0]
        post = doc.split(END)[1]
        open(DOC, "w", encoding="utf8", newline="\n").write(pre + text + post)
    else:
        print(text)


if __name__ == "__main__":
    main()
