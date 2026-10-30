import json, glob, random, sys
root = sys.argv[1]
paras = []
for f in sorted(glob.glob(root + '/docs/*.md')):
    for p in open(f, encoding='utf-8').read().replace('\r\n', '\n').split('\n\n'):
        p = p.strip()
        if len(p) >= 40 and not p.startswith('|') and not p.startswith('```'):
            paras.append(p)
random.Random(42).shuffle(paras)
texts = [p for p in paras if len(p) < 400][:24] + [p for p in paras if 400 <= len(p) < 1500][:16] + [p for p in paras if len(p) >= 1500][:4]
texts += ["What is the capital of France?", "how do I quantize a model", "Ünïcödé テキスト 日本語 mixed content — with punctuation!?", "a"]
json.dump(texts, open(sys.argv[2], 'w', encoding='utf-8'))
print(len(texts))
