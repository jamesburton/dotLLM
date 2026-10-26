"""Print stats for position 0 of result_norm in the dump dir."""
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
for n in ("result_norm", "result_output"):
    files = sorted(d.glob(f"*_{n}.bin"))
    if not files: continue
    data = read_bin(files[0])
    print(f"{n} shape={data.shape}")
    for pos in range(data.shape[0]):
        row = data[pos]
        print(f"  pos {pos}: min={row.min():+.4f} max={row.max():+.4f} mean={row.mean():+.4f} rms={np.sqrt((row.astype(np.float64)**2).mean()):.4f}")
