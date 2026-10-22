"""Compare per-head ratios with beta_sigmoid and g."""
import struct
import sys
from pathlib import Path
import numpy as np


def read_bin(path):
    with open(path, 'rb') as f:
        header = f.read(16)
        rank, d0, d1, d2 = struct.unpack('<iiii', header)
        shape = [d for d in [d0, d1, d2][:rank]]
        count = int(np.prod(shape)) if shape else 0
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data.reshape(shape) if shape else data


dump_dir = Path(sys.argv[1])
beta = read_bin(dump_dir / "00007_blk.0.beta_sigmoid.bin").flatten()
g = read_bin(dump_dir / "00006_blk.0.g.bin").flatten()

ratios = [1.0000, 0.0273, 3.7375, 17.158, 1.3499, 4.9609, 0.6222, 0.5014,
          0.2037, 0.1007, 1.0921, 0.7352, 1.1672, 0.5722, 2.0859, 0.4279,
          35.6537, 0.9730, 7.3541, 33.7612, 0.2492, 0.9157, 0.7831, 0.6312,
          0.0825, 0.0408, 3.3149, 2.2316, 0.6943, 0.3404, 4.8749, 1.0000]

print(f'{"vh":>3s} {"beta":>7s} {"g":>9s} {"ratio":>10s} {"ratio*beta":>11s} {"ratio/beta":>11s} {"ratio*g":>9s} {"ratio/g":>9s}')
for vh in range(32):
    r = ratios[vh]
    print(f'{vh:3d} {beta[vh]:7.4f} {g[vh]:9.6f} {r:10.4f} {r*beta[vh]:11.4f} {r/beta[vh]:11.4f} {r*g[vh]:9.4f} {r/g[vh]:9.4f}')
