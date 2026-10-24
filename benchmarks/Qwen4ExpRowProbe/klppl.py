"""Teacher-forced mean NLL / PPL over the first 4095 positions and per 1K bucket, from KlCurve .nll.f32 files (#873).

usage (run in the dump directory): klppl.py run1 run2 ...
"""
import sys, numpy as np

n = 4095
print(f"{'run':14}{'meanNLL[0,4095)':>16}{'PPL':>9}   by bucket PPL: 0-1K 1-2K 2-3K 3-4K")
for r in sys.argv[1:]:
    x = np.fromfile(r + ".nll.f32", dtype=np.float32)[:n]
    if len(x) < n:
        print(f"{r:14} (only {len(x)} positions)"); continue
    b = [np.exp(x[i:i + 1024].mean()) for i in (0, 1024, 2048, 3072)]
    print(f"{r:14}{x.mean():16.5f}{np.exp(x.mean()):9.4f}   " + "  ".join(f"{v:6.3f}" for v in b))
