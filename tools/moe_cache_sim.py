"""Replays MoE routing traces (DOTLLM_MOE_ROUTE_TRACE) through per-layer expert-cache policies: LRU, LFU, static pins
calibrated on the other traces, half-pinned LRU, and a promote-only decayed window. Also prints prefill expert coverage per
ubatch size and activation concentration. Used for the #1063 design (section 4)."""
import struct, sys, collections, os
def load(p):
    b=open(p,'rb').read(); i=0; recs=[]
    while i<len(b):
        L,ne,k=struct.unpack_from('<hhh',b,i); i+=6
        ids=struct.unpack_from('<%dh'%k,b,i); i+=2*k; recs.append((L,ne,ids))
    return recs
def by_layer(recs):
    d=collections.defaultdict(list); ne={}
    for L,n,ids in recs: d[L].append(ids); ne[L]=n
    return d,ne
def window(seq, cap, W=64, decay=0.5):
    # online: score = decayed counts; every W tokens re-pin top-cap; returns hits, refs, promotions
    score=collections.Counter(); pin=set(); hits=refs=prom=0
    for t,ids in enumerate(seq):
        for e in ids:
            refs+=1; hits+= e in pin; score[e]+=1
        if (t+1)%W==0:
            new=set(e for e,_ in score.most_common(cap)); prom+=len(new-pin); pin=new
            for e in list(score): score[e]*=decay
    return hits,refs,prom
def sim(seq, cap, policy, pinned=None):
    # seq: list of id tuples for one layer (token order). returns hits, refs
    hits=refs=0
    if policy=='static':
        s=pinned
        for ids in seq:
            for e in ids: refs+=1; hits+= e in s
        return hits,refs
    lru=collections.OrderedDict(); freq=collections.Counter()
    pin = pinned or set(); dyn = cap - len(pin)
    for ids in seq:
        for e in ids:
            refs+=1; freq[e]+=1
            if e in pin: hits+=1; continue
            if e in lru:
                hits+=1; lru.move_to_end(e); lru[e]=refs; continue
            if dyn<=0: continue
            if len(lru)>=dyn:
                if policy=='lru': lru.popitem(last=False)
                elif policy=='lfu':
                    v=min(lru, key=lambda x:(freq[x], lru[x])); del lru[v]
            lru[e]=refs
    return hits,refs
# usage: python -I tools/moe_cache_sim.py name1=trace1.bin name2=trace2.bin ...  (one DOTLLM_MOE_ROUTE_TRACE file per corpus)
D={}
names=[]
for arg in sys.argv[1:]:
    n,p=arg.split('=',1); names.append(n); D[n]=by_layer(load(p))
print('complete corpora:',list(D))
fr=[0.10,0.20,0.30,0.40,0.50]
if len(D)<2: print('(static calibrated on same corpus when only one available)')
print("hit rate (decode order, per-layer cache = fraction of that layer's experts); static/hybrid pins calibrated on the OTHER corpora")
print("corpus  frac   LRU    LFU   static  pin50%+LRU  window64(hit, promoted experts/token/layer)")
for n,(d,ne) in D.items():
    cal=collections.defaultdict(collections.Counter)
    for m,(d2,_) in D.items():
        if m==n and len(D)>1: continue
        for L,seq in d2.items():
            for ids in seq: cal[L].update(ids)
    for f in fr:
        tot=collections.Counter()
        for L,seq in d.items():
            cap=max(1,int(round(f*ne[L])))
            top=[e for e,_ in cal[L].most_common(cap)]
            for pol,pin in (('lru',None),('lfu',None),('static',set(top)),('lru',set(top[:cap//2]))):
                key=pol if pin is None else ('static' if pol=='static' else 'hyb')
                h,r=sim(seq,cap,pol,pin); tot[key+'h']+=h; tot[key+'r']+=r
            h,r,p=window(seq,cap); tot['winh']+=h; tot['winr']+=r; tot['winp']+=p; tot['tok']+=len(seq)
        print(f"{n:6s} {f:4.2f}  {tot['lruh']/tot['lrur']:.3f}  {tot['lfuh']/tot['lfur']:.3f}  {tot['statich']/tot['staticr']:.3f}  {tot['hybh']/tot['hybr']:.3f}   {tot['winh']/tot['winr']:.3f} {tot['winp']/tot['tok']:.3f}")
print("\nprefill: fraction of a layer's experts touched by a ubatch of B tokens (mean over layers and batches)")
for n,(d,ne) in D.items():
    row=[]
    for B in (1,8,32,128,512):
        fs=[]
        for L,seq in d.items():
            for s in range(0,len(seq)-B+1,B):
                u=set(); [u.update(x) for x in seq[s:s+B]]; fs.append(len(u)/ne[L])
        row.append(f"B={B}:{sum(fs)/len(fs):.2f}")
    print(n, ' '.join(row))
print("\nconcentration: share of experts covering 50%/80% of activations per layer (mean), all corpora pooled")
pool=collections.defaultdict(collections.Counter); NE={}
for n,(d,ne) in D.items():
    for L,seq in d.items():
        NE[L]=ne[L]
        for ids in seq: pool[L].update(ids)
c50=c80=0
for L,c in pool.items():
    tot=sum(c.values()); acc=0; k50=k80=None
    for i,(e,v) in enumerate(c.most_common()):
        acc+=v
        if k50 is None and acc>=0.5*tot: k50=i+1
        if k80 is None and acc>=0.8*tot: k80=i+1; break
    c50+=k50/NE[L]; c80+=k80/NE[L]
print(f"50%: {c50/len(pool):.3f}  80%: {c80/len(pool):.3f}  (uniform would be 0.50 / 0.80)")
