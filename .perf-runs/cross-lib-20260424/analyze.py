import json, sys, re
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else '.perf-runs/cross-lib-20260424/dotllm-profile.speedscope.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

frames = data['shared']['frames']
profiles = data['profiles']

self_time = defaultdict(float)
inclusive_time = defaultdict(float)
total_time = 0.0

for p in profiles:
    events = p.get('events') or []
    stack = []
    last_at = None
    for ev in events:
        at = ev['at']
        # Attribute time since last event to the current leaf
        if last_at is not None and stack:
            dt = at - last_at
            self_time[stack[-1]] += dt
            for fr in set(stack):
                inclusive_time[fr] += dt
            total_time += dt
        if ev['type'] == 'O':
            stack.append(ev['frame'])
        elif ev['type'] == 'C':
            if stack and stack[-1] == ev['frame']:
                stack.pop()
            else:
                # Unmatched close — pop best match
                try:
                    idx = len(stack) - 1 - stack[::-1].index(ev['frame'])
                    stack = stack[:idx]
                except ValueError:
                    pass
        last_at = at

def top(label, d, n=40, filter_re=None):
    print(f"\n=== top {n} by {label}" + (f" (filter={filter_re})" if filter_re else "") + " ===")
    items = sorted(d.items(), key=lambda kv: -kv[1])
    pat = re.compile(filter_re) if filter_re else None
    count = 0
    for idx, w in items:
        name = frames[idx].get('name', '?')
        if pat and not pat.search(name):
            continue
        pct = 100 * w / (total_time or 1)
        print(f"  {pct:5.2f}%  {w:14.1f}  {name}")
        count += 1
        if count >= n:
            break

print(f"profiles: {len(profiles)}  frames: {len(frames)}  total time units: {total_time:.1f}")
top('self', self_time, 50)
top('inclusive (dotLLM only)', inclusive_time, 40, r'DotLLM')
