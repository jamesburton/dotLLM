import json,sys,time,os
from sentence_transformers import SentenceTransformer
d=sys.argv[1]; texts=json.load(open(d+'/texts.json',encoding='utf-8'))
for repo in ['BAAI/bge-small-en-v1.5','BAAI/bge-base-en-v1.5','mixedbread-ai/mxbai-embed-large-v1']:
    m=SentenceTransformer(repo,device='cpu',model_kwargs={'torch_dtype':'float32'}); m.max_seq_length=512
    m.encode(texts[:4])
    for bs in (1,8,32):
        r=[]
        for _ in range(3):
            t0=time.time(); m.encode(texts,batch_size=bs,normalize_embeddings=True); r.append(len(texts)/(time.time()-t0))
        r.sort(); print(json.dumps({'model':repo,'bs':bs,'docs_s_min':r[0],'docs_s_med':r[1],'docs_s_max':r[2],'texts':len(texts)}),flush=True)
