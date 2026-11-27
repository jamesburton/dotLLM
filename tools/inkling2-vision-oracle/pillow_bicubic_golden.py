"""Prints the Pillow bicubic digests pinned by Inkling2VisionLimitsTests.Bicubic_IsByteIdenticalToPillow.

The input is a deterministic LCG image, so the C# test regenerates it. Run with Pillow installed (the pinned digests are Pillow 12.2).
"""
import hashlib
from PIL import Image


def make(n):
    x = 12345
    out = bytearray()
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append((x >> 16) & 255)
    return bytes(out)


for sw, sh, tw, th in [(97, 61, 40, 24), (61, 97, 80, 120), (130, 50, 40, 90), (2500, 2500, 1240, 1240), (300, 41, 300, 20)]:
    out = Image.frombytes("RGB", (sw, sh), make(sw * sh * 3)).resize((tw, th), Image.BICUBIC).tobytes()
    print(f'[InlineData({sw}, {sh}, {tw}, {th}, "{hashlib.sha256(out).hexdigest()}")]')
