# Reference embeddings via sentence-transformers (CPU). usage: ref.py <hf-repo> <texts.json> <out.json> [prefix] [trust]
import json, sys, time
from sentence_transformers import SentenceTransformer
repo, tp, out = sys.argv[1:4]
prefix = sys.argv[4] if len(sys.argv) > 4 else ''
trust = len(sys.argv) > 5
texts = [prefix + t for t in json.load(open(tp, encoding='utf-8'))]
m = SentenceTransformer(repo, device='cpu', trust_remote_code=trust, model_kwargs={'torch_dtype': 'float32'})
m.max_seq_length = 512 if 'Qwen' not in repo else 2048
v = m.encode(texts, normalize_embeddings=True, batch_size=1)
json.dump({'model': repo, 'vectors': v.tolist()}, open(out, 'w'))
# speed: batch of the texts, 3 reps
t0 = time.time()
for _ in range(2): m.encode(texts, normalize_embeddings=True, batch_size=32)
dt = (time.time() - t0) / 2
print(json.dumps({'model': repo, 'texts': len(texts), 'st_docs_s_bs32': len(texts) / dt}))
