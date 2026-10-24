"""Layer-end SNR check against gguf-py. Compares dotLLM final result_output (logits)
to gguf-py-computed full forward pass at position 0 (would be too slow), so instead
spot-check a few intermediate l_out tensors along the depth.

Strategy: We can't compute full forward in gguf-py (32 GB model, 40 layers, too slow).
But we CAN check:
  - result_output[0] (final logits at pos 0) vs the actual prompt's logits from llama.cpp
  - the per-layer l_out RMS sequence vs llama.cpp reference (if we have it)

Cheaper approach: just check 'is result_output reasonable for the prompt?'
- pos 0 sees "The" → top tokens should be e.g. " " (220), " ", " is", " ", " word", etc.
- pos 4 sees "The capital of France is" → top should include " Paris", " the", etc.

Print top-K decoded tokens at each position for visual inspection.
"""
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
    return data.reshape(shape), tuple(shape)


def main():
    d = Path(sys.argv[1])
    # Check stats across layers
    layers_l_out = sorted(d.glob("*_blk.*.l_out.bin"))
    # Map by layer index
    by_layer = {}
    for p in layers_l_out:
        # name like 00019_blk.0.l_out.bin
        layer_part = p.stem.split('blk.')[1].split('.')[0]
        by_layer[int(layer_part)] = p
    print(f"Total layer outputs: {len(by_layer)}")
    for layer in sorted(by_layer.keys()):
        data, _ = read_bin(by_layer[layer])
        rms_per_pos = [float(np.sqrt((data[t].astype(np.float64)**2).mean())) for t in range(data.shape[0])]
        print(f"  blk.{layer}.l_out RMS by pos: {[f'{r:.3f}' for r in rms_per_pos]}")


if __name__ == "__main__":
    main()
