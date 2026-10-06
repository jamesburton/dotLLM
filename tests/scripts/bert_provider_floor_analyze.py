import json, numpy as np, itertools
r=json.load(open("results.json")); dl=json.load(open("dotllm.json"))
errs=[k for k,v in r.items() if 'error' in v]; print("errors",errs, len(r))
def cos(a,b): a=np.array(a);b=np.array(b);return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
def worst(a,b): return min(cos(x,y) for x,y in zip(a,b))
provs=["ll8683cpu","ll9016cpu","ll9672cpu","ll9747cpu","ll9016vk","ll9747vk","ollama"]
for m in ["minilm-q8","mxbai-q8","nomic-q8","nomic-q4km"]:
    P={p:r[f"{m}|{p}"]["emb"] for p in provs}; P["dotllm"]=dl[m]
    toks=[r[f"{m}|{p}"]["tokens"] for p in provs[:-1]]
    print("\n==",m,"tokens identical across llama.cpp builds:",all(t==toks[0] for t in toks),"ollama count",r[f"{m}|ollama"]["count"],"sum tok",sum(map(len,toks[0])))
    names=list(P)
    print("%-10s"%""+" ".join("%9s"%n[:9] for n in names))
    for a in names: print("%-10s"%a+" ".join("%9.6f"%worst(P[a],P[b]) for b in names))
    ref={"minilm-q8":"minilm-f16","mxbai-q8":"mxbai-f32","nomic-q8":"nomic-f32","nomic-q4km":"nomic-f32"}[m]
    R=r[f"{ref}|ll9016cpu"]["emb"]; Rdl=dl[ref]
    print("vs unquantised ref",ref,"(llama.cpp b9016 cpu) :"," ".join(f"{n}={worst(P[n],R):.6f}" for n in names), "| ref itself dotllm vs llama:%.6f"%worst(Rdl,R))
    inter=[worst(P[a],P[b]) for a,b in itertools.combinations(provs,2)]
    print("inter-provider worst",min(inter),"dotllm-vs-each min",min(worst(P['dotllm'],P[p]) for p in provs))
