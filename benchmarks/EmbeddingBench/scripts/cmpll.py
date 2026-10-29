import json,sys,numpy as np
d=sys.argv[1]
def cmp(dl,ll):
    a=json.load(open(f'{d}/dl-{dl}.json'))['vectors']; L=json.load(open(f'{d}/ll-{ll}.json'))
    idx=L['good']; A=np.array([a[i] for i in idx]); B=np.array([L['vectors'][i] for i in idx])
    A/=np.linalg.norm(A,axis=1,keepdims=True); B/=np.linalg.norm(B,axis=1,keepdims=True); c=(A*B).sum(1)
    print(f'{dl:14s} vs llama.cpp({ll}): n={len(idx)} min={c.min():.6f} mean={c.mean():.6f}')
for p in sys.argv[2:]:
    a,b=p.split('='); cmp(a,b)
