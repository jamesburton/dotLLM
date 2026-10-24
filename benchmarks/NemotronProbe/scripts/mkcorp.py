import json, os, random, sys
wt = r"C:\Development\dotLLM\.claude\worktrees\agent-a08e22aa7248079ed"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corp")
os.makedirs(out, exist_ok=True)
random.seed(7)

def rd(p, a=0, b=None):
    t = open(os.path.join(wt, p), encoding="utf-8", errors="replace").read().replace("\r\n", "\n")
    return t[a:b]

# prose (summarisation-style technical prose)
prose = rd("docs/KV_CACHE.md", 0, 9000)
# code
code = rd("src/DotLLM.Cpu/Kernels/ReluSquared.cs") + "\n" + rd("tools/parity_check.py", 0, 6000)
if len(code) < 7000:
    code += rd("src/DotLLM.Models/Architectures/SsmStateCache.cs", 0, 7000)
# arithmetic
lines = []
for i in range(120):
    a, b = random.randint(11, 99), random.randint(11, 99)
    c = random.randint(100, 999)
    lines.append(f"Q: What is {a} * {b} + {c}?\nA: {a} * {b} = {a*b}. {a*b} + {c} = {a*b+c}. The answer is {a*b+c}.\n")
arith = "\n".join(lines)
# json extraction
names = ["Alice Chen", "Bob Okafor", "Carla Diaz", "Dmitri Ivanov", "Eun-ji Park", "Farid Haddad", "Grace Murphy", "Hiro Tanaka"]
cities = ["Lisbon", "Nairobi", "Toronto", "Osaka", "Berlin", "Lima", "Cardiff", "Hanoi"]
js = []
for i in range(40):
    n, c = random.choice(names), random.choice(cities)
    age, qty, price = random.randint(19, 70), random.randint(1, 9), round(random.uniform(3, 90), 2)
    js.append(f'Text: "{n} from {c}, aged {age}, ordered {qty} units at ${price} each."\nJSON: ' + json.dumps({"name": n, "city": c, "age": age, "quantity": qty, "unit_price": price}) + "\n")
jsontxt = "\n".join(js)
for name, t in dict(prose=prose, code=code, arith=arith, json=jsontxt).items():
    open(os.path.join(out, name + ".txt"), "w", encoding="utf-8", newline="\n").write(t)
    print(name, len(t))
