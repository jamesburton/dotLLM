"""Print top-K for each position from result_output.bin."""
import sys, struct
from pathlib import Path
import numpy as np


def read_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape))
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape)


d = Path(sys.argv[1])
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
files = sorted(d.glob("*_result_output.bin"))
data = read_bin(files[0])
seq, vocab = data.shape
for pos in range(seq):
    logits = data[pos]
    top_idx = np.argsort(-logits)[:k]
    print(f"pos {pos}: " + ", ".join(f"{int(i)}({logits[i]:+.2f})" for i in top_idx))
