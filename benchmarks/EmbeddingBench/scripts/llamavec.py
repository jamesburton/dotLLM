# POST texts to llama-server /v1/embeddings -> out json (llama.cpp CPU as the numerics oracle + throughput)
import json, sys, urllib.request, urllib.error, time
url, tp, out = sys.argv[1:4]
texts = json.load(open(tp, encoding='utf-8'))
def post(inp):
    req = urllib.request.Request(url + '/v1/embeddings', data=json.dumps({'input': inp, 'model': 'x'}).encode(), headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req))
vecs, good, rejected = [], [], 0
for i, t in enumerate(texts):
    try:
        vecs.append(post(t)['data'][0]['embedding']); good.append(i)
    except urllib.error.HTTPError:
        vecs.append(None); rejected += 1
json.dump({'model': 'llama.cpp', 'vectors': vecs, 'good': good}, open(out, 'w'))
gt = [texts[i] for i in good]
post(gt)
t0 = time.time()
for _ in range(3): post(gt)
dt = (time.time() - t0) / 3
print(json.dumps({'docs_s': len(gt) / dt, 'docs': len(gt), 'rejected': rejected}))
