"""Trim a llama-mtmd-debug encode log (llama.cpp b12780) to per-checkpoint corner values + sums.

usage: python trim_trace.py dbg.txt out.json
Keeps the named checkpoints the dotLLM encoder also emits. Each entry: name, ne (ggml dims, ne0 first), sum, and
`vals`: {"i3,i2,i1,i0": value} for the printed corner values (first/last 3 of every dimension longer than 6).
"""
import json, re, sys

KEEP = re.compile(r"^(patch_embd|embeddings|pre_ln|layer_inp_normed-\d+|Qcur_pos-\d+|Kcur_pos-\d+|Vcur-\d+|attn_out-\d+|ffn_inp-\d+|ffn_out-\d+|layer_out-\d+|encoder_out|pixel_shuffle|ffn_up|ffn_gelu_erf|projected|node_13|inp_raw)$")
hdr = re.compile(r"^common_debug_cb_eval:\s+(.+?) = \((\w+)\)\s+(\w+)\(.*\) = \{([\d, ]+)\}")

def idxs(n):
    return list(range(n)) if n <= 6 else [0, 1, 2, n - 3, n - 2, n - 1]

lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().split("\n")
out = []
i = 0
while i < len(lines):
    m = hdr.match(lines[i])
    if not m:
        i += 1
        continue
    name, ne = m.group(1).strip(), [int(x) for x in m.group(4).split(",")]
    j = i + 1
    body = []
    while j < len(lines) and not lines[j].startswith("common_debug_cb_eval") and not re.match(r"^\d+\.\d+\.\d+\.\d+", lines[j]):
        body.append(lines[j])
        j += 1
    i = j
    if not KEEP.match(name):
        continue
    sm = [b for b in body if b.strip().startswith("sum =")]
    total = float(sm[0].split("=")[1]) if sm else None
    # leaf rows in print order: lines starting with '[' that contain numbers
    rows = []
    for b in body:
        s = b.strip()
        if s.startswith("[") and re.search(r"-?\d", s) and "]" in s:
            vals = [float(x) for x in re.findall(r"-?\d+\.\d+|-?nan|-?inf", s)]
            rows.append(vals)
    # coordinates: iterate ne3, ne2, ne1 with elision, values have ne0 elision
    coords = []
    for i3 in idxs(ne[3]):
        for i2 in idxs(ne[2]):
            for i1 in idxs(ne[1]):
                coords.append((i3, i2, i1))
    if len(coords) != len(rows):
        print("WARN", name, ne, len(coords), len(rows), file=sys.stderr)
        continue
    cols = idxs(ne[0])
    vals = {}
    for (i3, i2, i1), r in zip(coords, rows):
        if len(r) != len(cols):
            print("WARN cols", name, len(r), len(cols), file=sys.stderr)
            break
        for c, v in zip(cols, r):
            vals[f"{i3},{i2},{i1},{c}"] = v
    out.append({"name": name, "ne": ne, "sum": total, "vals": vals})
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
print(len(out), "checkpoints")
