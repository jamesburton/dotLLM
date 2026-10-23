"""
Print the top-K token indices from a result_output dump.

Usage:
    python dump_top_tokens.py <dump_dir> [--k 10]
"""
import sys
import struct
from pathlib import Path
import numpy as np


def read_bin(path: Path):
    with path.open("rb") as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack("<iiii", header)
        shape = []
        if rank >= 1: shape.append(d0)
        if rank >= 2: shape.append(d1)
        if rank >= 3: shape.append(d2)
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data, tuple(shape)


def main():
    d = Path(sys.argv[1])
    k = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--k" else 10
    files = sorted(d.glob("*_result_output.bin"))
    assert len(files) == 1, f"want 1 result_output, got {len(files)}"
    data, shape = read_bin(files[0])
    print(f"shape={shape}")
    seq, vocab = shape
    # Greedy decoding looks at the LAST token's logits
    last_logits = data.reshape(seq, vocab)[-1]
    top_idx = np.argsort(-last_logits)[:k]
    for i, idx in enumerate(top_idx):
        print(f"  top{i+1}: idx={idx} logit={last_logits[idx]:+.4f}")


if __name__ == "__main__":
    main()
