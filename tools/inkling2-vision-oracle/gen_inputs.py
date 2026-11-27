import sys, math, os, glob
out = sys.argv[1]
os.makedirs(out, exist_ok=True)

def rainbow(w, h):
    cx, cy = w / 2.0, h / 2.0
    md = math.sqrt(cx * cx + cy * cy)
    buf = bytearray()
    for y in range(h):
        for x in range(w):
            dx, dy = x - cx, y - cy
            hue = math.atan2(dy, dx) / (2 * math.pi)
            if hue < 0:
                hue += 1
            sat = min(math.sqrt(dx * dx + dy * dy) / md, 1.0)
            h6 = hue * 6
            i6 = int(h6)
            f = h6 - i6
            p = 1 - sat
            q = 1 - sat * f
            t = 1 - sat * (1 - f)
            r, g, b = [(1, t, p), (q, 1, p), (p, 1, t), (p, q, 1), (t, p, 1), (1, p, q)][i6 % 6]
            # asymmetry: darken the left third, invert a top-right block
            if x < w // 3:
                r, g, b = r * 0.6, g * 0.6, b * 0.6
            if x > 2 * w // 3 and y < h // 3:
                r, g, b = 1 - r, 1 - g, 1 - b
            buf += bytes((int(r * 255 + .5), int(g * 255 + .5), int(b * 255 + .5)))
    return bytes(buf)

for name, (w, h) in {"rainbow190x110": (190, 110), "rainbow80": (80, 80), "rainbow640": (640, 640)}.items():
    with open(f"{out}/{name}.ppm", "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rainbow(w, h))

from transformers.utils.chat_template_utils import _compile_jinja_template
tp = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/*/chat_template.jinja"))[0]
t = _compile_jinja_template(open(tp, encoding="utf-8", newline="").read())
r = t.render(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "Describe this image in one sentence."}]}], add_generation_prompt=True)
print(repr(r))
open(f"{out}/prompt_template_render.txt", "w", encoding="utf-8", newline="").write(r)
open(f"{out}/prompt.txt", "w", encoding="utf-8", newline="").write(r.replace("<|content_image|><|unused_200054|>", "<__media__>"))
