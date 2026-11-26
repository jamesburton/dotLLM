"""Golden token ids for the Inkling 2 tokenizer (#2026), produced by llama.cpp's own `llama-tokenize` on the real GGUF vocabulary.

Usage: python generate_inkling2_tokenizer_cases.py <llama-tokenize.exe> <Inkling-2-Mini-NVFP4.gguf> <out.json>

llama.cpp maps tokenizer.ggml.pre = "inkling2" to its GPT-4o pre-tokenizer with ignore_merges = true and no BOS. Each case is tokenized with
--no-bos --no-parse-special (special-token spellings are plain text, exercising the control-token guard separately) and, for the `special` cases,
with parsing enabled. The strings cover what a wrong pre-tokenizer regex or a missing ignore_merges would get wrong: multi-digit numbers (digit groups
of up to three), contractions (case-insensitive), CJK and kana runs, whitespace runs and newlines, code, emoji, mixed scripts.
"""
import json
import os
import subprocess
import sys
import tempfile

exe, gguf, out = sys.argv[1], sys.argv[2], sys.argv[3]
cases = [
    ("plain", "The quick brown fox jumps over the lazy dog."),
    ("digits_groups", "In 1969 there were 12345678 reasons and 1,234,567.89 more; 007, 0, 10, 100, 1000."),
    ("contractions", "I'm sure it's fine; they'll say we've won, she'd agree, you're right, don't STOP, I'M HERE, IT'S OK."),
    ("cjk_runs", "\u6771\u4eac\u306f\u65e5\u672c\u306e\u9996\u90fd\u3067\u3059\u3002\u4eba\u53e3\u306f\u7d041400\u4e07\u4eba\u3002\u3053\u3093\u306b\u3061\u306f\u4e16\u754c\uff01\uc548\ub155\ud558\uc138\uc694"),
    ("whitespace_runs", "a  b   c    d\t\te \n\nf\n \n  g      \r\nh  "),
    ("leading_trailing_space", "   leading and trailing   "),
    ("newlines", "line1\nline2\r\nline3\n\n\nline4\n"),
    ("code", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a  # [0, 1, 1, 2]\n"),
    ("punctuation", "Wait... what?! (really) [yes] {no} <tag> a/b\\c |pipe| $5.00 100% #hash @at ~tilde ^caret `tick`"),
    ("emoji", "ok \U0001f600 \U0001f680\U0001f680 family: \U0001f468\u200d\U0001f469\u200d\U0001f467 flag: \U0001f1eb\U0001f1f7"),
    ("mixed_scripts", "Caf\u00e9 na\u00efve \u0414\u043e\u0431\u0440\u044b\u0439 \u0434\u0435\u043d\u044c \u03b1\u03b2\u03b3 \u0645\u0631\u062d\u0628\u0627 \u05e9\u05dc\u05d5\u05dd \u0939\u093f\u0928\u094d\u0926\u0940"),
    ("camel_and_caps", "camelCaseWord HTTPServerError XMLHttpRequest snake_case_word ALLCAPS Title Case"),
    ("urls_paths", "https://example.com/a/b?c=d&e=f#frag C:\\Users\\me\\file.txt /usr/local/bin"),
    ("empty_like", " "),
    ("single_char", "a"),
    ("repeated", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ........ -------- ========"),
    ("long_word", "Pneumonoultramicroscopicsilicovolcanoconiosis antidisestablishmentarianism"),
]
special = [
    ("special_chat", "<|message_user|><|content_text|>Hi<|end_message|><|message_model|>"),
]


def tokenize(text, parse_special):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.txt")
        open(p, "w", encoding="utf-8", newline="").write(text)
        args = [exe, "-m", gguf, "--file", p, "--ids", "--no-bos", "--log-disable"]
        if not parse_special:
            args.append("--no-parse-special")
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
        line = r.stdout.strip()
        return json.loads(line) if line else []


res = []
for name, text in cases:
    res.append({"name": name, "text": text, "ids": tokenize(text, False), "parseSpecial": False})
for name, text in special:
    res.append({"name": name, "text": text, "ids": tokenize(text, True), "parseSpecial": True})
json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), ensure_ascii=False, indent=1)
print(len(res), [len(c["ids"]) for c in res])
