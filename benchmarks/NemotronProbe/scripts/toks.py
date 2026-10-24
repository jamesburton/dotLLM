import sys, os
from gguf import GGUFReader
D = os.path.expanduser(r"~\.dotllm\test-cache\nvidia\nemotron-3.5-lightning")
r = GGUFReader(D + r"\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf")
f = r.fields["tokenizer.ggml.tokens"]
def tok(i):
    return bytes(f.parts[f.data[i]]).decode("utf-8", "replace")
for a in sys.argv[1:]:
    print(a, repr(tok(int(a))))
