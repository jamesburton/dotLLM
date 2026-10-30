import json, sys, os
import numpy as np
d = os.path.dirname(os.path.abspath(__file__))
pairs = [a.split('=') for a in sys.argv[1:]]  # dl-name=ref-name
for dl, ref in pairs:
    a = np.array(json.load(open(f'{d}/dl-{dl}.json'))['vectors'], dtype=np.float64)
    b = np.array(json.load(open(f'{d}/ref-{ref}.json'))['vectors'], dtype=np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True); b /= np.linalg.norm(b, axis=1, keepdims=True)
    cos = (a * b).sum(1); keep = [i for i in range(len(cos)) if not 40 <= i < 44] if os.environ.get('SHORT') else list(range(len(cos))); a = a[keep]; b = b[keep]; cos = cos[keep]
    # rank agreement: pairwise similarity matrix top-1 neighbour match
    sa, sb = a @ a.T, b @ b.T
    np.fill_diagonal(sa, -2); np.fill_diagonal(sb, -2)
    top1 = float((sa.argmax(1) == sb.argmax(1)).mean())
    print(f'{dl:12s} vs {ref:10s} min={cos.min():.6f} mean={cos.mean():.6f} argmin={cos.argmin()} nn1_agree={top1:.2f}')
