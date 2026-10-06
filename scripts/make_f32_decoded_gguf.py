"""Rewrite a GGUF with every tensor of one quantization type replaced by its F32 decode.

The F32 values come from llama.cpp's own gguf-py dequantizer, i.e. an oracle outside dotLLM's
lineage. Everything else -- all KV metadata, the F32 norms -- is copied verbatim.

    python make_f32_gguf.py <in.gguf> <out.gguf> [QTYPE] [--keep-token-embd]

QTYPE defaults to the most common quantized type in the file. --keep-token-embd leaves
token_embd.weight at its original type, which is what you want when comparing against a control
whose lm_head is quantized (see 2026-09-24-521-amplification-not-kernel-defects.md).

Built for #515/#519/#521: an F32-decoded control is the only way to tell how much of a
dotLLM-vs-llama.cpp perplexity gap belongs to the quantized path rather than to the weights.
"""
import sys, collections
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType as T

if len(sys.argv) < 3:
    sys.exit(__doc__)
SRC, DST = sys.argv[1], sys.argv[2]
rest = sys.argv[3:]
keep_embd = "--keep-token-embd" in rest
named = [a for a in rest if not a.startswith("--")]

r = GGUFReader(SRC)

if named:
    QT = T[named[0]]
else:
    counts = collections.Counter(
        t.tensor_type for t in r.tensors
        if t.tensor_type not in (T.F32, T.F16, T.BF16))
    QT = counts.most_common(1)[0][0]
print(f"converting {QT.name} -> F32" + ("  (token_embd kept)" if keep_embd else ""))

arch = str(bytes(r.fields["general.architecture"].parts[-1]), "utf-8")
w = GGUFWriter(DST, arch)

skip = {"general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
for name, field in r.fields.items():
    if name in skip:
        continue
    sub = field.types[-1] if len(field.types) > 1 else None
    w.add_key_value(name, field.contents(), field.types[0], sub_type=sub)

n_conv = 0
for t in r.tensors:
    # GGUFReader reports ne in GGUF order (fastest dim first); numpy/raw_shape is reversed.
    raw_shape = tuple(int(d) for d in reversed(t.shape))
    if t.tensor_type == QT and not (keep_embd and t.name == "token_embd.weight"):
        w.add_tensor(t.name, dequantize(t.data, QT).astype(np.float32).reshape(raw_shape),
                     raw_dtype=T.F32)
        n_conv += 1
    elif t.tensor_type in (T.F32, T.F16, T.BF16):
        w.add_tensor(t.name, t.data.reshape(raw_shape), raw_dtype=t.tensor_type)
    else:
        # For a quantized tensor the writer wants the BYTE shape (rows, row_bytes).
        data = t.data if t.data.ndim == 2 else t.data.reshape(raw_shape[0], -1)
        w.add_tensor(t.name, data, raw_shape=data.shape, raw_dtype=t.tensor_type)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file(progress=True)
w.close()
print(f"converted {n_conv} {QT.name} tensors -> F32; wrote {DST}")
