import numpy as np, sys
ref=np.load('tiny_hf/ref_logits.npy'); V=ref.shape[1]
def lsm(x): x=x.astype(np.float64); m=x.max(-1,keepdims=True); return x-m-np.log(np.exp(x-m).sum(-1,keepdims=True))
for f in sys.argv[1:]:
    a=np.fromfile(f,dtype=np.float32).reshape(-1,V)
    d=np.abs(a-ref); lp,lr=lsm(a),lsm(ref)
    kl=(np.exp(lr)*(lr-lp)).sum(-1)
    print(f, 'maxabs',d.max(),'meanabs',d.mean(),'maxKL',kl.max(),'top1',(a.argmax(-1)==ref.argmax(-1)).mean(), 'refstd', ref.std())
