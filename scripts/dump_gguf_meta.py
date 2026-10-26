import sys, gguf

def dump(path):
    print("="*80)
    print("FILE:", path)
    try:
        r = gguf.GGUFReader(path)
    except Exception as e:
        print("  ERROR opening:", e)
        return
    # architecture
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = "".join(chr(b) for b in f.parts[f.data[0]]) if f.types and f.types[0]==gguf.GGUFValueType.STRING else None
    # print key metadata fields of interest
    want_substr = ["architecture", "per_layer", "embedding_length", "altup", "laurel",
                   "sliding_window", "expert", "head_count", "key_length", "value_length",
                   "block_count", "rope", "activation_sparsity", "shared_kv", "diffusion",
                   "mask_token", "attention.head", "context_length", "softcap", "logit"]
    print("-- KV metadata (filtered) --")
    for f in r.fields.values():
        if any(s in f.name for s in want_substr):
            try:
                val = f.contents()
            except Exception:
                val = "<?>"
            sval = str(val)
            if len(sval) > 120: sval = sval[:120] + "..."
            print(f"  {f.name} = {sval}")
    # tensor name prefixes of interest
    print("-- tensor name groups --")
    groups = {}
    interesting = ["per_layer", "altup", "laurel", "inp_gate", "post_norm", "proj_norm",
                   "ffn_gate_inp", "ffn_gate_exps", "exps", "attn_q_norm", "attn_k_norm",
                   "post_attention_norm", "post_ffw_norm"]
    seen = set()
    for t in r.tensors:
        nm = t.name
        for kw in interesting:
            if kw in nm and kw not in seen:
                seen.add(kw)
                print(f"  example tensor with '{kw}': {nm}  shape={list(t.shape)}")
    print("-- total tensors:", len(r.tensors))

for p in sys.argv[1:]:
    dump(p)
