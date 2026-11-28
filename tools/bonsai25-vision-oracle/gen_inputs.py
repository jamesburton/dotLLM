"""Bonsai 2.5 vision oracle inputs: three PPM test images and the chat-template prompt with the mtmd media marker.

usage: python -I gen_inputs.py <outdir> <chat_template.jinja> <news.jpg>
  news.jpg = llama.cpp tools/mtmd/test-1.jpeg (640x488, a newspaper front page: text-rich and non-square)
The template is the one embedded in the Bonsai GGUF (tokenizer.chat_template); thinking is disabled so the caption starts at once.
"""
import sys, os, math
from PIL import Image
from transformers.utils.chat_template_utils import _compile_jinja_template

out, tpl, news = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
src = Image.open(news).convert("RGB")


def ppm(img, name):
    with open(f"{out}/{name}.ppm", "wb") as f:
        f.write(f"P6\n{img.width} {img.height}\n255\n".encode())
        f.write(img.tobytes())


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
            p, q, t = 1 - sat, 1 - sat * f, 1 - sat * (1 - f)
            r, g, b = [(1, t, p), (q, 1, p), (p, 1, t), (p, q, 1), (t, p, 1), (1, p, q)][i6 % 6]
            if x < w // 3:
                r, g, b = r * 0.6, g * 0.6, b * 0.6
            if x > 2 * w // 3 and y < h // 3:
                r, g, b = 1 - r, 1 - g, 1 - b
            buf += bytes((int(r * 255 + .5), int(g * 255 + .5), int(b * 255 + .5)))
    return Image.frombytes("RGB", (w, h), bytes(buf))


ppm(src, "news640x488")                                     # non-square, 640x480 after smart_resize: 20x15 merged = 300 tokens
ppm(src.resize((224, 168), Image.BICUBIC), "news224x168")   # non-square and needs a 168 -> 160 resize: 7x5 merged = 35 tokens
ppm(rainbow(80, 80), "rainbow80")                           # square and below min_pixels: upscaled to 96x96, 3x3 merged = 9 tokens

t = _compile_jinja_template(open(tpl, encoding="utf-8", newline="").read())
r = t.render(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "Describe this image in one sentence."}]}],
             add_generation_prompt=True, enable_thinking=False)
print(repr(r))
open(f"{out}/prompt_template_render.txt", "w", encoding="utf-8", newline="").write(r)
open(f"{out}/prompt.txt", "w", encoding="utf-8", newline="").write(r.replace("<|vision_start|><|image_pad|><|vision_end|>", "<__media__>"))
