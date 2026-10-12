import sys, re, collections
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
by = collections.defaultdict(collections.Counter)
nelem = collections.defaultdict(lambda: collections.defaultdict(int))
for t in r.tensors:
    base = re.sub(r"blk\.\d+\.", "blk.N.", t.name)
    by[base][t.tensor_type.name] += 1
    nelem[base][t.tensor_type.name] += int(t.n_elements)
for k in sorted(by):
    print(k, dict(by[k]), {a: f"{b/1e9:.3f}G" for a, b in nelem[k].items()})
