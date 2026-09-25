"""Find the rank of token 54120 ('Ta') in result_output[0] for Hi prompt."""
import sys, struct
from pathlib import Path
import numpy as np


def read_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape)


d = Path(sys.argv[1])
target = int(sys.argv[2])
files = sorted(d.glob("*_result_output.bin"))
data = read_bin(files[0])
logits = data[0] if data.ndim == 2 else data
target_logit = logits[target]
rank = (logits > target_logit).sum()
print(f"Token {target}: logit={target_logit:.4f}, rank={rank+1} (top-1 has rank=1)")
print(f"Top-1 logit: {logits.max():.4f} at token {logits.argmax()}")
