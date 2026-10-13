#!/usr/bin/env python3
"""tlora_capability_eval.py — Capability evaluation harness for BitNet ternary-LoRA adapters.

Measures three capability categories against a base or ternary-LoRA loaded model:
  1. instruction  — IFEval-style verifiable constraints (12 items), scored programmatically
  2. tooluse      — JSON tool-call emission from a system-described tool (10 items)
  3. coding       — Python pass@1 via subprocess execution (8 problems)

One ``--tlora`` target per run; compare base vs adapter by running twice and diffing JSON.

Import surface from pdiff_ternary_lora (never modified):
    BASE_MODEL_ID, TernaryLoRAWrapper, _wrap_mlp_layers,
    _set_lora_lambda, _set_lora_enabled, _collect_lora_params

Usage:
    # Base (no LoRA):
    python scripts/diffusion/tlora_capability_eval.py \\
        --tlora none --device cuda --out results_base.json

    # Ternary-LoRA from a saved dir:
    python scripts/diffusion/tlora_capability_eval.py \\
        --tlora E:/tlora_out/gen --device cuda --out results_tlora.json

    # CPU scorer unit tests only — no model load:
    python scripts/diffusion/tlora_capability_eval.py --validate-scorers
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

# UTF-8 on Windows (avoids cp1252 errors on Greek letters etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

# Ensure pdiff_ternary_lora is on the path (same directory).
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
from pdiff_ternary_lora import (
    BASE_MODEL_ID,
    TernaryLoRAWrapper,
    _wrap_mlp_layers,
    _set_lora_lambda,
    _set_lora_enabled,
    _collect_lora_params,
    enable_inference_cache,
)


# ===========================================================================
# SECTION 1 — INSTRUCTION EVAL
# Items: each is a dict with keys:
#   tag           : short identifier
#   messages      : list of {role, content} (no add_generation_prompt yet)
#   checker       : Callable[[str], bool]  (True = constraint satisfied)
#   max_new_tokens: int
# ===========================================================================

# --- Individual checker functions (module-level for easy unit testing) ---

def _chk_one_word(text: str) -> bool:
    """Exactly one whitespace-separated token."""
    return len(text.strip().split()) == 1


def _chk_five_comma_items(text: str) -> bool:
    """Exactly 5 non-empty comma-separated items."""
    items = [x.strip() for x in text.strip().split(",") if x.strip()]
    return len(items) == 5


def _chk_all_lowercase(text: str) -> bool:
    """All alpha characters are lowercase."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.islower() for c in letters)


def _chk_specifically_twice(text: str) -> bool:
    """The word 'specifically' appears exactly twice."""
    return text.lower().count("specifically") == 2


def _chk_exactly_three_words(text: str) -> bool:
    """Exactly three whitespace-separated tokens."""
    return len(text.strip().split()) == 3


def _chk_no_commas(text: str) -> bool:
    """No comma character anywhere in the response."""
    return "," not in text


def _chk_starts_recursion(text: str) -> bool:
    """Response starts with the word 'recursion' (case-insensitive)."""
    return text.strip().lower().startswith("recursion")


def _chk_first_second_third(text: str) -> bool:
    """Text contains 'first', 'second', and 'third'."""
    t = text.lower()
    return "first" in t and "second" in t and "third" in t


def _chk_force_once(text: str) -> bool:
    """The word 'force' appears exactly once (whole-word, case-insensitive)."""
    return len(re.findall(r"\bforce\b", text, re.IGNORECASE)) == 1


def _chk_all_uppercase(text: str) -> bool:
    """All alpha characters are uppercase."""
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


def _chk_in_other_words_once(text: str) -> bool:
    """The phrase 'in other words' appears exactly once."""
    return text.lower().count("in other words") == 1


def _chk_no_letter_e(text: str) -> bool:
    """The letter 'e' does not appear anywhere (case-insensitive)."""
    return "e" not in text.lower()


# Built-in instruction task set (12 items).
INSTRUCTION_ITEMS: list[dict] = [
    {
        "tag": "one_word",
        "messages": [{"role": "user", "content": "What is 2 + 2? Answer with exactly one word."}],
        "checker": _chk_one_word,
        "max_new_tokens": 10,
    },
    {
        "tag": "five_comma_items",
        "messages": [{"role": "user", "content": "List exactly 5 fruits separated by commas. Output only the comma-separated list, nothing else."}],
        "checker": _chk_five_comma_items,
        "max_new_tokens": 40,
    },
    {
        "tag": "all_lowercase",
        "messages": [{"role": "user", "content": "Describe what a variable is in programming. Write your response using only lowercase letters."}],
        "checker": _chk_all_lowercase,
        "max_new_tokens": 120,
    },
    {
        "tag": "specifically_twice",
        "messages": [{"role": "user", "content": "Explain what machine learning is. Include the word 'specifically' exactly two times in your response."}],
        "checker": _chk_specifically_twice,
        "max_new_tokens": 150,
    },
    {
        "tag": "exactly_three_words",
        "messages": [{"role": "user", "content": "What is the capital of France? Answer with exactly three words, no more, no less."}],
        "checker": _chk_exactly_three_words,
        "max_new_tokens": 20,
    },
    {
        "tag": "no_commas",
        "messages": [{"role": "user", "content": "Describe the water cycle briefly. Do not use any commas in your response."}],
        "checker": _chk_no_commas,
        "max_new_tokens": 120,
    },
    {
        "tag": "starts_recursion",
        "messages": [{"role": "user", "content": "Explain recursion in programming. Your response must start with the word 'Recursion'."}],
        "checker": _chk_starts_recursion,
        "max_new_tokens": 120,
    },
    {
        "tag": "first_second_third",
        "messages": [{"role": "user", "content": "Name three benefits of regular exercise. Your response must contain the words 'first', 'second', and 'third'."}],
        "checker": _chk_first_second_third,
        "max_new_tokens": 120,
    },
    {
        "tag": "force_once",
        "messages": [{"role": "user", "content": "What is gravity? Use the word 'force' exactly once in your answer."}],
        "checker": _chk_force_once,
        "max_new_tokens": 100,
    },
    {
        "tag": "all_uppercase",
        "messages": [{"role": "user", "content": "Define the word 'algorithm' in one sentence. Your entire response must be in UPPERCASE letters only."}],
        "checker": _chk_all_uppercase,
        "max_new_tokens": 80,
    },
    {
        "tag": "in_other_words_once",
        "messages": [{"role": "user", "content": "Explain what a function is in programming. Use the phrase 'in other words' exactly once in your response."}],
        "checker": _chk_in_other_words_once,
        "max_new_tokens": 130,
    },
    {
        "tag": "no_letter_e",
        "messages": [{"role": "user", "content": "What is a loop in programming? Do not use the letter 'e' anywhere in your response."}],
        "checker": _chk_no_letter_e,
        "max_new_tokens": 100,
    },
]


# ===========================================================================
# SECTION 2 — TOOLUSE EVAL
# Items: each is a dict with keys:
#   tag            : short identifier
#   system         : system prompt text (describes the tool)
#   user           : user message
#   expected_fn    : the expected function name in the JSON output
#   expected_args  : list of arg key names (at least one must be present for valid call)
#   max_new_tokens : int
# ===========================================================================

def _make_tool_system(tool_name: str, description: str, params: dict[str, str]) -> str:
    """Build a standard system prompt that embeds a single-tool schema."""
    param_lines = "\n".join(
        f"  - {k} ({v})" for k, v in params.items()
    )
    return (
        "You are a function-calling assistant. When the user needs a tool, respond ONLY with "
        "a valid JSON object in this exact format:\n"
        '{"name": "<function_name>", "arguments": {<args_as_json>}}\n'
        "No explanation, no other text — just the JSON object.\n\n"
        f"Available tool:\n"
        f"  Name: {tool_name}\n"
        f"  Description: {description}\n"
        f"  Parameters:\n{param_lines}"
    )


TOOLUSE_ITEMS: list[dict] = [
    {
        "tag": "get_weather",
        "system": _make_tool_system(
            "get_weather", "Get the current weather for a city",
            {"city": "string, required — the city name"},
        ),
        "user": "What's the weather like in Tokyo right now?",
        "expected_fn": "get_weather",
        "expected_args": ["city"],
        "max_new_tokens": 80,
    },
    {
        "tag": "add",
        "system": _make_tool_system(
            "add", "Add two integers and return their sum",
            {"a": "integer, required — first operand", "b": "integer, required — second operand"},
        ),
        "user": "What is 42 plus 17?",
        "expected_fn": "add",
        "expected_args": ["a", "b"],
        "max_new_tokens": 60,
    },
    {
        "tag": "search",
        "system": _make_tool_system(
            "search", "Search the web for information",
            {"query": "string, required — the search query"},
        ),
        "user": "Find information about the Eiffel Tower for me.",
        "expected_fn": "search",
        "expected_args": ["query"],
        "max_new_tokens": 80,
    },
    {
        "tag": "translate",
        "system": _make_tool_system(
            "translate", "Translate text to a target language",
            {
                "text": "string, required — text to translate",
                "target_language": "string, required — language to translate into",
            },
        ),
        "user": "Translate 'hello world' to Spanish.",
        "expected_fn": "translate",
        "expected_args": ["text", "target_language"],
        "max_new_tokens": 80,
    },
    {
        "tag": "convert_currency",
        "system": _make_tool_system(
            "convert_currency", "Convert an amount from one currency to another",
            {
                "amount": "number, required — the amount to convert",
                "from_currency": "string, required — source currency code (e.g. USD)",
                "to_currency": "string, required — target currency code (e.g. EUR)",
            },
        ),
        "user": "Convert 100 USD to EUR for me.",
        "expected_fn": "convert_currency",
        "expected_args": ["amount", "from_currency", "to_currency"],
        "max_new_tokens": 90,
    },
    {
        "tag": "set_timer",
        "system": _make_tool_system(
            "set_timer", "Set a countdown timer",
            {
                "duration_seconds": "integer, required — duration in seconds",
                "label": "string, required — a label for the timer",
            },
        ),
        "user": "Set a 5 minute timer labeled 'pasta'.",
        "expected_fn": "set_timer",
        "expected_args": ["duration_seconds", "label"],
        "max_new_tokens": 80,
    },
    {
        "tag": "get_stock_price",
        "system": _make_tool_system(
            "get_stock_price", "Get the current stock price for a ticker symbol",
            {"symbol": "string, required — the stock ticker symbol (e.g. AAPL)"},
        ),
        "user": "What's the current price of Apple stock?",
        "expected_fn": "get_stock_price",
        "expected_args": ["symbol"],
        "max_new_tokens": 70,
    },
    {
        "tag": "calculate",
        "system": _make_tool_system(
            "calculate", "Evaluate a mathematical expression",
            {"expression": "string, required — the mathematical expression to evaluate"},
        ),
        "user": "Calculate the square root of 144.",
        "expected_fn": "calculate",
        "expected_args": ["expression"],
        "max_new_tokens": 70,
    },
    {
        "tag": "get_news",
        "system": _make_tool_system(
            "get_news", "Retrieve recent news articles on a topic",
            {
                "topic": "string, required — the news topic",
                "count": "integer, required — number of articles to retrieve",
            },
        ),
        "user": "Get me the 3 latest news articles about climate change.",
        "expected_fn": "get_news",
        "expected_args": ["topic", "count"],
        "max_new_tokens": 80,
    },
    {
        "tag": "send_email",
        "system": _make_tool_system(
            "send_email", "Send an email message",
            {
                "to": "string, required — recipient email address",
                "subject": "string, required — email subject",
                "body": "string, required — email body text",
            },
        ),
        "user": "Send an email to alice@example.com with subject 'Meeting' and body 'See you at 3pm'.",
        "expected_fn": "send_email",
        "expected_args": ["to", "subject", "body"],
        "max_new_tokens": 100,
    },
]


def _parse_tool_call_json(text: str) -> Optional[dict]:
    """Extract the first parseable JSON object from text that has a 'name' field.

    Handles bare JSON, JSON in markdown code fences, and JSON embedded in prose.
    Returns None if nothing parseable with a 'name' key is found.
    """
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()

    # Walk the text looking for outermost {...} spans
    for source in (stripped, text):
        depth = 0
        start = None
        for i, ch in enumerate(source):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = source[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "name" in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    start = None
    return None


def _score_tooluse(text: str, expected_fn: str, expected_args: list[str]) -> tuple[float, bool, bool]:
    """Score one tool-call response.

    Returns:
        (score, name_match, valid_call)
        score      : 1.0 if name matches AND at least one expected arg key present;
                     0.5 if name matches but args missing/wrong;
                     0.0 otherwise.
        name_match : True if the correct function name was found in a JSON object.
        valid_call : True if name_match AND at least one expected arg key present.
    """
    obj = _parse_tool_call_json(text)
    if obj is None:
        return 0.0, False, False

    fn_name = obj.get("name", "")
    name_match = fn_name == expected_fn

    if not name_match:
        return 0.0, False, False

    # Check if arguments dict contains at least one expected key.
    args_dict = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if not isinstance(args_dict, dict):
        # Arguments present but as a non-dict (e.g., a string); name matched only.
        return 0.5, True, False

    arg_keys_present = any(k in args_dict for k in expected_args)
    if arg_keys_present:
        return 1.0, True, True
    return 0.5, True, False


# ===========================================================================
# SECTION 3 — CODING EVAL
# Problems: (tag, instruction, test_asserts)
# instruction: what the user is asked (function name + description)
# test_asserts: Python assert statement(s) that pass iff the function is correct
# ===========================================================================

CODING_PROBLEMS: list[tuple[str, str, str]] = [
    (
        "fibonacci",
        "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number "
        "(0-indexed: fibonacci(0)=0, fibonacci(1)=1). Output only the function.",
        "assert fibonacci(0)==0 and fibonacci(1)==1 and fibonacci(10)==55",
    ),
    (
        "is_palindrome",
        "Write a Python function `is_palindrome(s)` that returns True if the string s reads "
        "the same forwards and backwards, False otherwise. Output only the function.",
        "assert is_palindrome('racecar') and is_palindrome('') and not is_palindrome('abc')",
    ),
    (
        "sum_even",
        "Write a Python function `sum_even(xs)` that returns the sum of all even numbers "
        "in a list of integers. Output only the function.",
        "assert sum_even([1,2,3,4])==6 and sum_even([1,3,5])==0 and sum_even([])==0",
    ),
    (
        "reverse_string",
        "Write a Python function `reverse_string(s)` that returns the string s reversed. "
        "Output only the function.",
        "assert reverse_string('abc')=='cba' and reverse_string('')=='' and reverse_string('x')=='x'",
    ),
    (
        "count_vowels",
        "Write a Python function `count_vowels(s)` that returns the count of vowels "
        "(a, e, i, o, u, case-insensitive) in string s. Output only the function.",
        "assert count_vowels('Hello')==2 and count_vowels('xyz')==0 and count_vowels('AEIOU')==5",
    ),
    (
        "factorial",
        "Write a Python function `factorial(n)` that returns n factorial (n!). "
        "factorial(0) == 1. Output only the function.",
        "assert factorial(0)==1 and factorial(1)==1 and factorial(5)==120",
    ),
    (
        "fizzbuzz",
        "Write a Python function `fizzbuzz(n)` that returns 'FizzBuzz' if n is divisible by "
        "both 3 and 5, 'Fizz' if by 3 only, 'Buzz' if by 5 only, else str(n). "
        "Output only the function.",
        "assert fizzbuzz(3)=='Fizz' and fizzbuzz(5)=='Buzz' and fizzbuzz(15)=='FizzBuzz' and fizzbuzz(7)=='7'",
    ),
    (
        "max_of_list",
        "Write a Python function `max_of_list(xs)` that returns the maximum value in a "
        "non-empty list of numbers. Output only the function.",
        "assert max_of_list([1,5,3])==5 and max_of_list([-2,-9])==-2 and max_of_list([42])==42",
    ),
]


def _extract_code(text: str) -> str:
    """Extract runnable Python from a model response.

    Prefers the longest fenced ```python``` block; falls back to the first
    ``def``/``import``/``class`` occurrence; final fallback is the raw text.
    """
    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if fences:
        return max(fences, key=len).strip()
    m = re.search(r"(?:^|\n)\s*(?:def |import |from |class )", text)
    if m:
        return text[m.start() :].strip()
    return text.strip()


def _run_code_test(code: str, test_assert: str, timeout: int = 10) -> bool:
    """Execute ``code + test_assert`` in a sandboxed subprocess.

    Returns True iff the process exits 0 within ``timeout`` seconds.
    Uses a temp file to avoid shell-quoting issues on Windows.
    """
    src = code + "\n\n" + test_assert + "\nprint('PASS')\n"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="tlora_eval_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(src)
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        return result.returncode == 0 and "PASS" in result.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ===========================================================================
# SECTION 4 — SCORER UNIT TESTS (--validate-scorers)
# ===========================================================================

def _validate_scorers() -> bool:
    """Unit-test all three scorer families against hand-crafted mock outputs.

    Does NOT load any model — runs entirely on CPU.
    Returns True iff all checks pass.
    """
    print()
    print("[validate-scorers] " + "=" * 60)
    print("[validate-scorers] Instruction checker unit tests")
    print("[validate-scorers] " + "=" * 60)

    _INSTRUCTION_CHECKS: list[tuple[str, Callable[[str], bool], str, bool, str, bool]] = [
        # (checker_name, fn, pass_input, expected_pass, fail_input, expected_fail)
        ("one_word",            _chk_one_word,             "four",                   True,  "not one word",         False),
        ("five_comma_items",    _chk_five_comma_items,      "a, b, c, d, e",          True,  "a, b, c",              False),
        ("all_lowercase",       _chk_all_lowercase,         "hello world",            True,  "Hello World",          False),
        ("specifically_twice",  _chk_specifically_twice,    "specifically X specifically Y", True,  "specifically once",    False),
        ("exactly_three_words", _chk_exactly_three_words,   "Paris is beautiful",     True,  "Paris",                False),
        ("no_commas",           _chk_no_commas,             "no commas here",         True,  "a, b",                 False),
        ("starts_recursion",    _chk_starts_recursion,      "Recursion is fun.",      True,  "Loops are great.",     False),
        ("first_second_third",  _chk_first_second_third,    "first, second, third",   True,  "first and second",     False),
        ("force_once",          _chk_force_once,            "gravity is a force that", True, "a force or no force",  False),
        ("all_uppercase",       _chk_all_uppercase,         "HELLO WORLD",            True,  "hello",                False),
        ("in_other_words_once", _chk_in_other_words_once,   "in other words it is so", True, "in other words one in other words two", False),
        ("no_letter_e",         _chk_no_letter_e,           "a loop runs again",      True,  "there are loops",      False),
    ]

    instr_ok = True
    for name, fn, p_in, p_exp, f_in, f_exp in _INSTRUCTION_CHECKS:
        p_got = fn(p_in)
        f_got = fn(f_in)
        ok = (p_got == p_exp) and (f_got == f_exp)
        if not ok:
            instr_ok = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}  {name:<24}  "
            f"pass-case: {p_got} (want {p_exp})  "
            f"fail-case: {f_got} (want {f_exp})"
        )

    print()
    print("[validate-scorers] " + "=" * 60)
    print("[validate-scorers] Tool-use JSON scorer unit tests")
    print("[validate-scorers] " + "=" * 60)

    _TOOLUSE_CHECKS = [
        # (description, text, expected_fn, expected_args, want_score, want_name, want_valid)
        (
            "valid_json_correct_name_and_args",
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}',
            "get_weather", ["city"],
            1.0, True, True,
        ),
        (
            "valid_json_correct_name_no_args",
            '{"name": "get_weather", "arguments": {}}',
            "get_weather", ["city"],
            0.5, True, False,
        ),
        (
            "valid_json_wrong_name",
            '{"name": "wrong_tool", "arguments": {"city": "Tokyo"}}',
            "get_weather", ["city"],
            0.0, False, False,
        ),
        (
            "no_json_in_text",
            "The weather in Tokyo is sunny and warm today.",
            "get_weather", ["city"],
            0.0, False, False,
        ),
        (
            "invalid_json",
            "{not valid json at all}",
            "get_weather", ["city"],
            0.0, False, False,
        ),
        (
            "json_in_prose",
            'Here is the call: {"name": "add", "arguments": {"a": 42, "b": 17}}',
            "add", ["a", "b"],
            1.0, True, True,
        ),
        (
            "json_in_code_fence",
            '```json\n{"name": "search", "arguments": {"query": "Eiffel Tower"}}\n```',
            "search", ["query"],
            1.0, True, True,
        ),
    ]

    tooluse_ok = True
    for desc, text, exp_fn, exp_args, w_score, w_name, w_valid in _TOOLUSE_CHECKS:
        score, name_match, valid_call = _score_tooluse(text, exp_fn, exp_args)
        ok = (score == w_score) and (name_match == w_name) and (valid_call == w_valid)
        if not ok:
            tooluse_ok = False
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}  {desc:<40}  "
            f"score={score} (want {w_score})  "
            f"name={name_match} (want {w_name})  "
            f"valid={valid_call} (want {w_valid})"
        )

    print()
    print("[validate-scorers] " + "=" * 60)
    print("[validate-scorers] Coding scorer unit tests")
    print("[validate-scorers] " + "=" * 60)

    _CODING_CHECKS = [
        (
            "correct_fibonacci_fenced",
            "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
            "assert fibonacci(0)==0 and fibonacci(1)==1 and fibonacci(10)==55",
            True,
        ),
        (
            "wrong_fibonacci",
            "```python\ndef fibonacci(n):\n    return n  # wrong\n```",
            "assert fibonacci(0)==0 and fibonacci(1)==1 and fibonacci(10)==55",
            False,
        ),
        (
            "correct_no_fence",
            "def fibonacci(n):\n    if n <= 0: return 0\n    if n == 1: return 1\n    return fibonacci(n-1) + fibonacci(n-2)\n",
            "assert fibonacci(0)==0 and fibonacci(1)==1 and fibonacci(10)==55",
            True,
        ),
        (
            "syntax_error",
            "```python\ndef fibonacci(n\n    return n\n```",
            "assert fibonacci(0)==0",
            False,
        ),
    ]

    coding_ok = True
    for desc, text, test_assert, want_pass in _CODING_CHECKS:
        code = _extract_code(text)
        got = _run_code_test(code, test_assert)
        ok = got == want_pass
        if not ok:
            coding_ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {desc:<36}  result={got} (want {want_pass})")

    all_ok = instr_ok and tooluse_ok and coding_ok
    print()
    print("[validate-scorers] " + "=" * 60)
    print(f"[validate-scorers] RESULT: {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"[validate-scorers]   instruction={instr_ok}  tooluse={tooluse_ok}  coding={coding_ok}")
    print("[validate-scorers] " + "=" * 60)
    print()
    return all_ok


# ===========================================================================
# SECTION 5 — MODEL LOAD / RELOAD
# ===========================================================================

def _load_model_and_tokenizer(tlora_dir: str, device: torch.device, base_model: str = BASE_MODEL_ID):
    """Load base model and (optionally) wrap + reload ternary-LoRA weights.

    Args:
        tlora_dir : path to a saved tlora dir OR the string 'none' for base only.
        device    : torch device.
        base_model: HuggingFace model ID or local path for the base model
                    (default: BASE_MODEL_ID). When loading a saved tlora dir whose
                    ``tlora_meta.json`` records a ``base_model`` field, that value
                    takes precedence — so an 8B-trained expert automatically loads
                    its 8B base without requiring ``--base-model`` on the CLI.

    Returns:
        (model, tokenizer)  — model has _set_lora_enabled already set correctly.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # When loading a saved tlora dir, prefer its recorded base_model over the arg.
    resolved_base = base_model
    if tlora_dir.strip().lower() != "none":
        _meta_path_early = Path(tlora_dir) / "tlora_meta.json"
        if _meta_path_early.exists():
            with open(_meta_path_early, encoding="utf-8") as _f:
                _early_meta = json.load(_f)
            _recorded = _early_meta.get("base_model")
            if _recorded and _recorded != base_model:
                resolved_base = _recorded
                print(f"[tlora_eval] base_model from tlora_meta.json: {resolved_base!r} "
                      f"(overrides --base-model {base_model!r})")

    print(f"[tlora_eval] Loading base model {resolved_base!r} ...")
    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "local_files_only": True,
    }
    if device.type != "cpu":
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(resolved_base, **load_kwargs)
    if device.type == "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(resolved_base, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # KV cache MUST be on for generation — disabling it makes decode O(n^2)
    # (each new token reprocesses the whole sequence). It does not change greedy
    # output (mathematically identical), only speed. ~10x+ faster for long outputs.
    model.config.use_cache = True

    n_layers = len(model.model.layers)
    print(f"[tlora_eval] Base model loaded  ({n_layers} transformer layers)")

    use_tlora = tlora_dir.strip().lower() != "none"

    if not use_tlora:
        # Base-only evaluation: disable LoRA term (no wrapping needed).
        _set_lora_enabled(False)
        print("[tlora_eval] Mode: BASE only (no LoRA wrapping)")
        return model, tokenizer

    # --- Reload ternary-LoRA from saved dir ---
    tdir = Path(tlora_dir)
    meta_path = tdir / "tlora_meta.json"
    weights_path = tdir / "tlora_weights.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"tlora_meta.json not found in {tlora_dir!r}")
    if not weights_path.exists():
        raise FileNotFoundError(f"tlora_weights.pt not found in {tlora_dir!r}")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    r     = int(meta["r"])
    alpha = float(meta["alpha"])

    # Determine target layer indices.
    target_layer_indices = meta.get("target_layer_indices")
    if target_layer_indices is None:
        last_n = meta.get("last_n_layers")
        if last_n is not None:
            target_layer_indices = list(range(max(0, n_layers - int(last_n)), n_layers))
            print(f"[tlora_eval] Computed target_layer_indices from last_n_layers={last_n}: "
                  f"{target_layer_indices[0]}..{target_layer_indices[-1]}")
        else:
            print("[tlora_eval] target_layer_indices=null, last_n_layers=null — wrapping ALL layers")
    else:
        print(f"[tlora_eval] target_layer_indices from meta: {len(target_layer_indices)} layers")

    # Step 3: cold wrap (no adapter_weights — zeros/Kaiming init; we overwrite immediately).
    print(f"[tlora_eval] Wrapping MLP projections (cold init, r={r}, alpha={alpha}) ...")
    wrapped_keys = _wrap_mlp_layers(
        model=model,
        target_layer_indices=target_layer_indices,
        r=r,
        alpha=alpha,
        adapter_weights=None,
    )
    print(f"[tlora_eval] Wrapped {len(wrapped_keys)} projections")

    # Step 4: load saved lora_A / lora_B and copy into the wrappers.
    print(f"[tlora_eval] Loading tlora_weights.pt ...")
    saved_state: dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    loaded = 0
    for name, m in model.named_modules():
        if not isinstance(m, TernaryLoRAWrapper):
            continue
        if name not in saved_state:
            print(f"[tlora_eval]   WARNING: module {name!r} not found in saved state — skipping")
            continue
        entry = saved_state[name]
        m.lora_A.data.copy_(entry["lora_A"].to(torch.bfloat16))
        m.lora_B.data.copy_(entry["lora_B"].to(torch.bfloat16))
        # Move LoRA masters to the same device as the model.
        if m.lora_A.device != device:
            m.lora_A.data = m.lora_A.data.to(device)
            m.lora_B.data = m.lora_B.data.to(device)
        loaded += 1

    print(f"[tlora_eval] Loaded weights for {loaded}/{len(wrapped_keys)} wrappers from {weights_path.name}")
    if loaded == 0:
        raise RuntimeError("No LoRA weights were loaded — check that tlora_weights.pt keys match model.named_modules()")

    # Step 5: enable ternary LoRA at full strength.
    _set_lora_enabled(True)
    _set_lora_lambda(1.0)
    print("[tlora_eval] Mode: TERNARY-LoRA  (lam=1.0, lora_enabled=True)")
    return model, tokenizer


# ===========================================================================
# SECTION 6 — GENERATE
# ===========================================================================

@torch.no_grad()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device: torch.device) -> str:
    """Greedy-decode ``max_new_tokens`` from ``prompt``.

    Returns the generated text (without the prompt prefix).
    """
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = input_ids.to(device)
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = output[0][input_ids.shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ===========================================================================
# SECTION 7 — CATEGORY EVAL RUNNERS
# ===========================================================================

def _run_instruction_eval(model, tokenizer, device: torch.device) -> dict:
    """Run all INSTRUCTION_ITEMS and score them."""
    model.eval()
    results = []
    satisfied = 0

    for item in INSTRUCTION_ITEMS:
        prompt = tokenizer.apply_chat_template(
            item["messages"], add_generation_prompt=True, tokenize=False
        )
        response = _generate(model, tokenizer, prompt, item["max_new_tokens"], device)
        ok = bool(item["checker"](response))
        if ok:
            satisfied += 1
        results.append({
            "tag": item["tag"],
            "prompt_user": item["messages"][-1]["content"],
            "response": response,
            "satisfied": ok,
        })
        status = "ok" if ok else "no"
        print(f"  [{status}] {item['tag']:<24}  {response[:80]!r}", flush=True)

    n = len(INSTRUCTION_ITEMS)
    score = satisfied / n if n > 0 else 0.0
    return {
        "category": "instruction",
        "n": n,
        "satisfied": satisfied,
        "score": round(score, 4),
        "items": results,
    }


def _run_tooluse_eval(model, tokenizer, device: torch.device) -> dict:
    """Run all TOOLUSE_ITEMS and score them."""
    model.eval()
    results = []
    total_score = 0.0
    name_matches = 0
    valid_calls = 0

    for item in TOOLUSE_ITEMS:
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = _generate(model, tokenizer, prompt, item["max_new_tokens"], device)
        score, name_match, valid_call = _score_tooluse(response, item["expected_fn"], item["expected_args"])
        total_score += score
        if name_match:
            name_matches += 1
        if valid_call:
            valid_calls += 1

        status = "1.0" if valid_call else ("0.5" if name_match else "0.0")
        print(f"  [{status}] {item['tag']:<20}  fn={item['expected_fn']!r}  "
              f"got={response[:60]!r}", flush=True)

        results.append({
            "tag": item["tag"],
            "user": item["user"],
            "expected_fn": item["expected_fn"],
            "response": response,
            "score": score,
            "name_match": name_match,
            "valid_call": valid_call,
        })

    n = len(TOOLUSE_ITEMS)
    return {
        "category": "tooluse",
        "n": n,
        "total_score": round(total_score, 4),
        "score": round(total_score / n, 4) if n > 0 else 0.0,
        "valid_call_rate": round(valid_calls / n, 4) if n > 0 else 0.0,
        "name_match_rate": round(name_matches / n, 4) if n > 0 else 0.0,
        "valid_calls": valid_calls,
        "name_matches": name_matches,
        "items": results,
    }


def _run_coding_eval(model, tokenizer, device: torch.device) -> dict:
    """Run all CODING_PROBLEMS and score them (pass@1)."""
    model.eval()
    results = []
    passed = 0

    for tag, instruction, test_assert in CODING_PROBLEMS:
        messages = [{"role": "user", "content": instruction}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = _generate(model, tokenizer, prompt, 300, device)
        code = _extract_code(response)
        ok = _run_code_test(code, test_assert)
        if ok:
            passed += 1

        status = "P" if ok else "."
        print(f"  [{status}] {tag:<24}  code={code[:60]!r}", flush=True)

        results.append({
            "tag": tag,
            "response": response,
            "extracted_code": code,
            "passed": ok,
        })

    n = len(CODING_PROBLEMS)
    score = passed / n if n > 0 else 0.0
    return {
        "category": "coding",
        "n": n,
        "passed": passed,
        "score": round(score, 4),
        "items": results,
    }


# ===========================================================================
# SECTION 8 — OUTPUT TABLE
# ===========================================================================

def _print_table(label: str, instr: dict, tooluse: dict, coding: dict) -> None:
    """Print a compact results table to stdout."""
    header = f"=== tlora_capability_eval: {label} ==="
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    print(f"  {'category':<14} {'n':>4}  {'score':>8}  notes")
    print(f"  {'-'*14}  {'-'*4}  {'-'*8}  -----")
    print(
        f"  {'instruction':<14} {instr['n']:>4}  {instr['score']:>8.4f}  "
        f"{instr['satisfied']}/{instr['n']} constraints satisfied"
    )
    print(
        f"  {'tooluse':<14} {tooluse['n']:>4}  {tooluse['score']:>8.4f}  "
        f"valid-call={tooluse['valid_call_rate']:.4f}  name-match={tooluse['name_match_rate']:.4f}"
    )
    print(
        f"  {'coding':<14} {coding['n']:>4}  {coding['score']:>8.4f}  "
        f"pass@1  {coding['passed']}/{coding['n']}"
    )
    print(f"  {'-'*14}  {'-'*4}  {'-'*8}")
    total_n = instr["n"] + tooluse["n"] + coding["n"]
    # Macro average of category scores.
    macro_avg = (instr["score"] + tooluse["score"] + coding["score"]) / 3.0
    print(f"  {'macro-avg':<14} {'':>4}  {macro_avg:>8.4f}  (mean of 3 category scores)")
    print(sep)
    print()


# ===========================================================================
# SECTION 9 — MAIN
# ===========================================================================

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Capability evaluation for BitNet ternary-LoRA adapters. "
            "Three categories: instruction (verifiable constraints), "
            "tooluse (JSON call emission), coding (pass@1). "
            "Run once with --tlora none for base, once with --tlora <dir> for adapter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--tlora", default="none",
        help=(
            "Path to a saved ternary-LoRA directory (containing tlora_meta.json "
            "and tlora_weights.pt), or 'none' to evaluate the base model without any adapter."
        ),
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Inference device: 'cuda' or 'cpu'.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output JSON file path for detailed results. Required unless --validate-scorers.",
    )
    ap.add_argument(
        "--validate-scorers", action="store_true",
        help=(
            "Run scorer unit tests on hand-crafted mock outputs (no model load). "
            "Exits 0 on success, 1 on failure."
        ),
    )
    ap.add_argument(
        "--label", default=None,
        help="Label printed in the results table (defaults to 'base' or the tlora dir name).",
    )
    ap.add_argument(
        "--base-model", default=BASE_MODEL_ID, dest="base_model",
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID or local path for the base model "
            "(default: %(default)s). When --tlora dir is given and its "
            "tlora_meta.json records a base_model field, that value takes precedence."
        ),
    )

    args = ap.parse_args(argv)

    # ------------------------------------------------------------------
    # --validate-scorers: unit tests only, no model
    # ------------------------------------------------------------------
    if args.validate_scorers:
        ok = _validate_scorers()
        sys.exit(0 if ok else 1)

    if args.out is None:
        ap.error("--out is required unless --validate-scorers is given")

    device = torch.device(args.device)
    use_tlora = args.tlora.strip().lower() != "none"
    label = args.label or (Path(args.tlora).name if use_tlora else "base")

    print(f"[tlora_eval] ============================================================")
    print(f"[tlora_eval] tlora_capability_eval")
    print(f"[tlora_eval]   label   = {label}")
    print(f"[tlora_eval]   tlora   = {args.tlora!r}")
    print(f"[tlora_eval]   device  = {args.device}")
    print(f"[tlora_eval]   out     = {args.out!r}")
    print(f"[tlora_eval] ============================================================")

    # Load model (and optionally LoRA).
    model, tokenizer = _load_model_and_tokenizer(args.tlora, device, base_model=args.base_model)
    model.eval()

    # Pre-compute frozen ternary weights once; reuse across every token step.
    # Eliminates per-token WeightQuant.apply overhead (~1 tok/s -> ~10-50+ tok/s).
    n_cached = enable_inference_cache(model)
    print(f"[tlora_eval] Inference cache enabled: {n_cached} BitNet linear modules patched.")

    t_start = time.perf_counter()

    # --- Instruction eval ---
    print(f"\n[tlora_eval] --- INSTRUCTION ({len(INSTRUCTION_ITEMS)} items) ---")
    instr_results = _run_instruction_eval(model, tokenizer, device)
    print(f"  score = {instr_results['score']:.4f}  ({instr_results['satisfied']}/{instr_results['n']} satisfied)")

    # --- Tooluse eval ---
    print(f"\n[tlora_eval] --- TOOLUSE ({len(TOOLUSE_ITEMS)} items) ---")
    tooluse_results = _run_tooluse_eval(model, tokenizer, device)
    print(
        f"  score = {tooluse_results['score']:.4f}  "
        f"valid-call={tooluse_results['valid_call_rate']:.4f}  "
        f"name-match={tooluse_results['name_match_rate']:.4f}"
    )

    # --- Coding eval ---
    print(f"\n[tlora_eval] --- CODING ({len(CODING_PROBLEMS)} problems) ---")
    coding_results = _run_coding_eval(model, tokenizer, device)
    print(f"  pass@1 = {coding_results['score']:.4f}  ({coding_results['passed']}/{coding_results['n']} passed)")

    elapsed = time.perf_counter() - t_start
    print(f"\n[tlora_eval] Total eval time: {elapsed:.1f}s")

    # Print results table.
    _print_table(label, instr_results, tooluse_results, coding_results)

    # Write JSON.
    output = {
        "label": label,
        "tlora_dir": args.tlora,
        "device": args.device,
        "elapsed_s": round(elapsed, 2),
        "instruction": instr_results,
        "tooluse": tooluse_results,
        "coding": coding_results,
        "macro_avg_score": round(
            (instr_results["score"] + tooluse_results["score"] + coding_results["score"]) / 3.0, 4
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[tlora_eval] Results written to {args.out!r}")


if __name__ == "__main__":
    main()
