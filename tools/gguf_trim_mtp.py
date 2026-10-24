"""Strip the MTP layer (blk.<N-1>) from a nemotron_h_moe GGUF so stock llama.cpp (b8683/b9016/b9672) loads it.
llama.cpp treats the MTP layer (attn+MoE, both per-layer arrays non-zero) as a plain FFN layer and leaves
9 tensors (attn_q/k/v/output, post_attention_norm, 4x nextn.*) unconsumed -> 'expected 417, got 408'.
Output is a local artifact (NOT in the ollama store). Tensor payloads are copied verbatim."""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType

src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = r.fields["general.architecture"].contents()
nblk = int(r.fields[f"{arch}.block_count"].contents())
trunk = nblk - int(r.fields[f"{arch}.nextn_predict_layers"].contents())
print("arch", arch, "block_count", nblk, "-> trunk", trunk)
w = GGUFWriter(dst, arch)
for k, f in r.fields.items():
    if k.startswith("GGUF.") or k == "general.architecture":
        continue
    if k == f"{arch}.block_count":
        w.add_key_value(k, trunk, GGUFValueType.UINT32); continue
    if k == f"{arch}.nextn_predict_layers":
        w.add_key_value(k, 0, GGUFValueType.UINT32); continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        sub = f.types[1]
        val = f.contents()
        if k in (f"{arch}.attention.head_count_kv", f"{arch}.feed_forward_length"):
            val = list(val)[:trunk]
        w.add_key_value(k, val, vt, sub)
    else:
        w.add_key_value(k, f.contents(), vt)
kept = [t for t in r.tensors if not t.name.startswith(f"blk.{trunk}.")]
print("tensors", len(r.tensors), "->", len(kept))
for t in kept:
    shape = [int(x) for x in t.data.shape]
    w.add_tensor_info(t.name, shape, t.data.dtype, int(t.n_bytes), raw_dtype=t.tensor_type)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for i, t in enumerate(kept):
    w.write_tensor_data(t.data)
    if i % 50 == 0:
        print("wrote", i, flush=True)
w.close()
print("done")
