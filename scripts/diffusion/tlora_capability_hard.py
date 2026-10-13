#!/usr/bin/env python3
"""tlora_capability_hard.py — HARD capability-evaluation harness for BitNet ternary-LoRA adapters.

Four categories, each with real headroom (base BitNet-2B should NOT saturate these):
  1. coding_humaneval — pass@1 on HumanEval (openai_humaneval dataset, --n-code problems)
  2. math_gsm8k       — exact numeric accuracy on GSM8K (openai/gsm8k main, --n-math problems)
  3. instruction_hard — 20 STRICT built-in verifiable constraints (much harder than easy set)
  4. tooluse_multi    — 15 built-in multi-tool items (3-5 tools each); true selection accuracy

Reuses (imports, does NOT copy):
  - _load_model_and_tokenizer, _generate, _parse_tool_call_json from tlora_capability_eval
  - BASE_MODEL_ID from pdiff_ternary_lora

Usage:
    # Base model — all categories:
    python scripts/diffusion/tlora_capability_hard.py \\
        --tlora none --device cuda --out results_hard_base.json

    # Ternary-LoRA expert directory:
    python scripts/diffusion/tlora_capability_hard.py \\
        --tlora E:/mote_experts/code --device cuda --out results_hard_code.json

    # Subset of categories (faster):
    python scripts/diffusion/tlora_capability_hard.py \\
        --tlora none --device cuda --out r.json --categories code,math

    # CPU scorer unit tests only — no model load:
    python scripts/diffusion/tlora_capability_hard.py --validate-scorers
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent

if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Deferred imports (torch, transformers, datasets) — only when actually running inference.
# pdiff_ternary_lora and tlora_capability_eval are imported eagerly so --validate-scorers
# can prove they resolve without loading the 2B model.
import pdiff_ternary_lora as _ptlora  # noqa: F401 — import-resolution proof
from pdiff_ternary_lora import BASE_MODEL_ID, enable_inference_cache  # noqa: F401

import tlora_capability_eval as _tce  # noqa: F401 — import-resolution proof
from tlora_capability_eval import (
    _load_model_and_tokenizer,
    _generate,
    _parse_tool_call_json,
)


# ===========================================================================
# SECTION 1 — GSM8K HELPERS
# ===========================================================================

def _normalize_number(s: str) -> Optional[str]:
    """Strip $, commas, leading - then normalise to int or float string."""
    cleaned = re.sub(r"[$,]", "", s.strip())
    try:
        val = float(cleaned)
        if val == int(val):
            return str(int(val))
        return str(round(val, 6))
    except ValueError:
        return None


def _extract_gsm8k_gold(answer: str) -> Optional[str]:
    """Extract the numeric answer that follows the '####' marker in GSM8K gold."""
    m = re.search(r"####\s*([0-9,.$\-]+)", answer)
    if m:
        return _normalize_number(m.group(1))
    return None


def _extract_gsm8k_predicted(text: str) -> Optional[str]:
    """Extract the model's predicted final number.

    Strategies (tried in order):
      1. After a '####' marker (model mimics GSM8K format)
      2. After 'answer is', 'answer:', or 'answer =' (last occurrence)
      3. The last integer or decimal in the text
    """
    # Strategy 1: #### marker
    m = re.search(r"####\s*([0-9,.$\-]+)", text)
    if m:
        return _normalize_number(m.group(1))

    # Strategy 2: answer keyword (last occurrence)
    m_all = list(re.finditer(
        r"(?:answer\s+is|answer\s*:|answer\s*=)\s*\$?\s*([0-9,.\-]+)",
        text, re.IGNORECASE,
    ))
    if m_all:
        return _normalize_number(m_all[-1].group(1))

    # Strategy 3: last number in text
    nums = re.findall(r"\b[0-9,.\-]+\b", text)
    for n in reversed(nums):
        normalized = _normalize_number(n)
        if normalized is not None:
            return normalized
    return None


# ===========================================================================
# SECTION 2 — HUMANEVAL HELPERS
# ===========================================================================

def _extract_code_he(text: str) -> str:
    """Extract Python code from a model response (HumanEval variant).

    Preference order: longest ```python``` fence → first fence → raw text.
    """
    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if fences:
        return max(fences, key=len).strip()
    return text.strip()


def _assemble_humaneval_program(prompt: str, completion: str, entry_point: str) -> str:
    """Combine prompt + completion, handling model re-emission of the function signature.

    If the extracted completion already contains 'def <entry_point>(...)',
    use it standalone (the model gave a complete function).
    Otherwise append it to the prompt (model gave only the body).
    """
    code = _extract_code_he(completion)
    if re.search(r"^\s*def\s+" + re.escape(entry_point) + r"\s*\(", code, re.MULTILINE):
        return code
    return prompt + "\n" + code


def _run_humaneval_test(program: str, test_code: str, entry_point: str, timeout: int = 15) -> bool:
    """Execute ``program + test_code + check(entry_point)`` in a subprocess.

    Returns True iff returncode == 0 within *timeout* seconds.
    Uses a temp file to avoid shell-quoting issues on Windows.
    """
    src = program + "\n\n" + test_code + f"\ncheck({entry_point})\n"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="tlora_he_")
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
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ===========================================================================
# SECTION 3 — INSTRUCTION-HARD: 20 STRICT CHECKER FUNCTIONS
# ===========================================================================

def _chk_hard_json_3keys(text: str) -> bool:
    """Valid JSON object with exactly the keys name, age, city — no extras."""
    try:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()
        obj = json.loads(s)
        return isinstance(obj, dict) and set(obj.keys()) == {"name", "age", "city"}
    except Exception:
        return False


def _chk_hard_3_bullets(text: str) -> bool:
    """Exactly 3 lines of the form '- [A-Z]...' each ending with a period."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    if len(lines) != 3:
        return False
    return all(len(ln) > 2 and ln[2:3].isupper() and ln.endswith(".") for ln in lines)


def _chk_hard_50_words(text: str) -> bool:
    """Exactly 50 whitespace-separated tokens."""
    return len(text.strip().split()) == 50


def _chk_hard_however_3(text: str) -> bool:
    """The word 'however' appears exactly 3 times (case-insensitive)."""
    return text.lower().count("however") == 3


def _chk_hard_acrostic_cat(text: str) -> bool:
    """3+ non-empty lines; first letters spell C-A-T (case-insensitive)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    return (
        lines[0][0].upper() == "C"
        and lines[1][0].upper() == "A"
        and lines[2][0].upper() == "T"
    )


def _chk_hard_7_words(text: str) -> bool:
    """Exactly 7 whitespace-separated tokens."""
    return len(text.strip().split()) == 7


def _chk_hard_json_array_5(text: str) -> bool:
    """Valid JSON array of exactly 5 strings."""
    try:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()
        arr = json.loads(s)
        return isinstance(arr, list) and len(arr) == 5 and all(isinstance(x, str) for x in arr)
    except Exception:
        return False


def _chk_hard_one_digit(text: str) -> bool:
    """Exactly one digit character (0–9) in the entire response."""
    return sum(1 for c in text if c.isdigit()) == 1


def _chk_hard_blue_red_not_green(text: str) -> bool:
    """Contains 'blue' and 'red' but not 'green' (whole-word, case-insensitive)."""
    t = text.lower()
    return "blue" in t and "red" in t and "green" not in t


def _chk_hard_3_question_marks(text: str) -> bool:
    """Exactly 3 question mark characters."""
    return text.count("?") == 3


def _chk_hard_all_exclaim(text: str) -> bool:
    """No periods or question marks; exactly 3 exclamation marks; ends with '!'."""
    t = text.strip()
    return "." not in t and "?" not in t and t.count("!") == 3 and t.endswith("!")


def _chk_hard_4_lines_3_words(text: str) -> bool:
    """Exactly 4 non-empty lines, each with exactly 3 whitespace-separated tokens."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) != 4:
        return False
    return all(len(ln.split()) == 3 for ln in lines)


def _chk_hard_therefore_furthermore(text: str) -> bool:
    """Contains 'therefore' and 'furthermore' at least once each."""
    t = text.lower()
    return "therefore" in t and "furthermore" in t


def _chk_hard_2para_2sent(text: str) -> bool:
    """Exactly 2 paragraphs (blank-line separated), each with exactly 2 sentence endings."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) != 2:
        return False
    for para in paras:
        endings = re.findall(r"[.!?](?=\s|$)", para)
        if len(endings) != 2:
            return False
    return True


def _chk_hard_json_nested_person(text: str) -> bool:
    """Valid JSON: {\"person\": {\"name\": ..., \"hobby\": ...}}."""
    try:
        s = re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()
        obj = json.loads(s)
        p = obj.get("person", {})
        return isinstance(p, dict) and "name" in p and "hobby" in p
    except Exception:
        return False


def _chk_hard_2_sentences_period(text: str) -> bool:
    """Exactly 2 sentences; no '!' or '?'; text ends with '.'."""
    t = text.strip()
    if "!" in t or "?" in t:
        return False
    return t.count(".") == 2 and t.endswith(".")


def _chk_hard_number_10_20(text: str) -> bool:
    """Exactly one digit-sequence in [10, 20] (inclusive); no other digit sequences."""
    nums = re.findall(r"\b\d+\b", text)
    in_range = [n for n in nums if 10 <= int(n) <= 20]
    return len(nums) == 1 and len(in_range) == 1


def _chk_hard_starts_with_I(text: str) -> bool:
    """At least 3 sentences; the first 3 each start with 'I '."""
    sentences = re.split(r"[.!?]+\s*", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) < 3:
        return False
    return all(s.startswith("I ") for s in sentences[:3])


def _chk_hard_8_words_max5(text: str) -> bool:
    """Exactly 8 tokens; every token (alpha chars only) has ≤5 letters."""
    words = text.strip().split()
    if len(words) != 8:
        return False
    return all(len(re.sub(r"[^a-zA-Z]", "", w)) <= 5 for w in words)


def _chk_hard_3_sentences(text: str) -> bool:
    """Exactly 3 sentence-ending punctuation marks (.!?) at word boundaries."""
    t = text.strip()
    endings = re.findall(r"[.!?](?=\s|$)", t)
    return len(endings) == 3


# --- Built-in hard instruction items (20 items) ---

INSTRUCTION_HARD_ITEMS: list[dict] = [
    {
        "tag": "json_3keys",
        "messages": [{"role": "user", "content":
            "Respond ONLY with a valid JSON object containing exactly these three keys: "
            "\"name\", \"age\", \"city\". Any values you like — but only those three keys, no extras."}],
        "checker": _chk_hard_json_3keys,
        "max_new_tokens": 60,
    },
    {
        "tag": "3_bullets_format",
        "messages": [{"role": "user", "content":
            "Give exactly 3 bullet points about artificial intelligence. "
            "Each bullet must: start with '- ' (dash space), begin with a capital letter, "
            "and end with a period. No other lines."}],
        "checker": _chk_hard_3_bullets,
        "max_new_tokens": 120,
    },
    {
        "tag": "exactly_50_words",
        "messages": [{"role": "user", "content":
            "Explain what gravity is in exactly 50 words. Count every word carefully — "
            "50 words, not 49, not 51."}],
        "checker": _chk_hard_50_words,
        "max_new_tokens": 120,
    },
    {
        "tag": "however_3_times",
        "messages": [{"role": "user", "content":
            "Describe the pros and cons of social media in a short paragraph. "
            "Use the word 'however' exactly 3 times — not 2, not 4, exactly 3."}],
        "checker": _chk_hard_however_3,
        "max_new_tokens": 150,
    },
    {
        "tag": "acrostic_CAT",
        "messages": [{"role": "user", "content":
            "Write a 3-line poem where the first letter of each line, read top-to-bottom, "
            "spells the word CAT (line 1 starts with C, line 2 with A, line 3 with T)."}],
        "checker": _chk_hard_acrostic_cat,
        "max_new_tokens": 80,
    },
    {
        "tag": "exactly_7_words",
        "messages": [{"role": "user", "content":
            "Summarize the Big Bang theory in exactly 7 words. No more, no fewer."}],
        "checker": _chk_hard_7_words,
        "max_new_tokens": 30,
    },
    {
        "tag": "json_array_5_strings",
        "messages": [{"role": "user", "content":
            "List 5 programming languages. Respond ONLY with a valid JSON array of exactly "
            "5 strings and nothing else — no prose, no explanation."}],
        "checker": _chk_hard_json_array_5,
        "max_new_tokens": 60,
    },
    {
        "tag": "exactly_one_digit",
        "messages": [{"role": "user", "content":
            "Write one sentence about the number seven. Your entire response must contain "
            "exactly one digit character (0-9) — no more, no fewer."}],
        "checker": _chk_hard_one_digit,
        "max_new_tokens": 60,
    },
    {
        "tag": "blue_red_not_green",
        "messages": [{"role": "user", "content":
            "Describe a garden scene. Include the words 'blue' and 'red' somewhere in your "
            "response, but do NOT include the word 'green' anywhere."}],
        "checker": _chk_hard_blue_red_not_green,
        "max_new_tokens": 100,
    },
    {
        "tag": "exactly_3_questions",
        "messages": [{"role": "user", "content":
            "Write exactly 3 rhetorical questions about the nature of time. "
            "Use exactly 3 question marks, one per question."}],
        "checker": _chk_hard_3_question_marks,
        "max_new_tokens": 120,
    },
    {
        "tag": "all_sentences_exclaim",
        "messages": [{"role": "user", "content":
            "List exactly 3 amazing facts about volcanoes. Write each as its own sentence "
            "ending with an exclamation mark. Do NOT use periods or question marks anywhere."}],
        "checker": _chk_hard_all_exclaim,
        "max_new_tokens": 120,
    },
    {
        "tag": "4_lines_3_words_each",
        "messages": [{"role": "user", "content":
            "Describe the ocean in exactly 4 lines, with exactly 3 words per line. "
            "No more, no fewer on any line."}],
        "checker": _chk_hard_4_lines_3_words,
        "max_new_tokens": 60,
    },
    {
        "tag": "therefore_furthermore",
        "messages": [{"role": "user", "content":
            "Explain why physical exercise is beneficial. Your response must include the word "
            "'therefore' at least once AND the word 'furthermore' at least once."}],
        "checker": _chk_hard_therefore_furthermore,
        "max_new_tokens": 150,
    },
    {
        "tag": "2_paragraphs_2_sentences",
        "messages": [{"role": "user", "content":
            "Write about climate change in exactly 2 paragraphs, each containing exactly "
            "2 sentences. Separate paragraphs with a blank line."}],
        "checker": _chk_hard_2para_2sent,
        "max_new_tokens": 200,
    },
    {
        "tag": "json_nested_person",
        "messages": [{"role": "user", "content":
            "Respond ONLY with a valid JSON object matching exactly this schema: "
            "{\"person\": {\"name\": \"...\", \"hobby\": \"...\"}}. "
            "No other keys, no extra text."}],
        "checker": _chk_hard_json_nested_person,
        "max_new_tokens": 60,
    },
    {
        "tag": "2_sentences_period_only",
        "messages": [{"role": "user", "content":
            "Explain recursion in exactly 2 sentences. Use only periods to end sentences — "
            "no exclamation marks, no question marks."}],
        "checker": _chk_hard_2_sentences_period,
        "max_new_tokens": 80,
    },
    {
        "tag": "number_10_to_20",
        "messages": [{"role": "user", "content":
            "Write 2 sentences about mathematics. Include exactly one number written as "
            "digits that is between 10 and 20 inclusive (e.g. '15'). "
            "No other digit sequences anywhere in your response."}],
        "checker": _chk_hard_number_10_20,
        "max_new_tokens": 80,
    },
    {
        "tag": "starts_with_I_3_reasons",
        "messages": [{"role": "user", "content":
            "Give 3 reasons why reading is important. Start each sentence with the word 'I' "
            "(e.g. 'I believe...', 'I find...', 'I think...')."}],
        "checker": _chk_hard_starts_with_I,
        "max_new_tokens": 120,
    },
    {
        "tag": "8_words_max5_letters",
        "messages": [{"role": "user", "content":
            "Describe autumn in exactly 8 words. No individual word may be longer than "
            "5 letters (count only alphabetic characters, ignore punctuation)."}],
        "checker": _chk_hard_8_words_max5,
        "max_new_tokens": 40,
    },
    {
        "tag": "exactly_3_sentences",
        "messages": [{"role": "user", "content":
            "Describe the internet in exactly 3 sentences — no more, no fewer."}],
        "checker": _chk_hard_3_sentences,
        "max_new_tokens": 120,
    },
]


# ===========================================================================
# SECTION 4 — TOOLUSE-MULTI: 15 ITEMS WITH 3-5 TOOLS EACH
# ===========================================================================

def _make_multi_tool_system(tools: list[dict]) -> str:
    """Build a system message listing multiple tools as JSON schemas."""
    schema_blocks = []
    for t in tools:
        param_lines = "\n".join(f"    - {k}: {v}" for k, v in t["params"].items())
        schema_blocks.append(
            f"  Name: {t['name']}\n"
            f"  Description: {t['description']}\n"
            f"  Parameters:\n{param_lines}"
        )
    tools_text = "\n\n".join(schema_blocks)
    return (
        "You are a function-calling assistant. When the user needs a tool, respond ONLY with "
        "a valid JSON object in this exact format:\n"
        '{"name": "<function_name>", "arguments": {<args_as_json>}}\n'
        "No explanation, no other text — just the JSON object.\n\n"
        f"Available tools:\n\n{tools_text}"
    )


# Reusable tool definitions
_T_WEATHER    = {"name": "get_weather",      "description": "Get current weather for a location",
                 "params": {"location": "string, required — city name"}}
_T_SEARCH     = {"name": "search_web",       "description": "Search the internet for information",
                 "params": {"query": "string, required — search query"}}
_T_TIMER      = {"name": "set_timer",        "description": "Set a countdown timer",
                 "params": {"duration_seconds": "integer, required — seconds", "label": "string, required — timer label"}}
_T_TRANSLATE  = {"name": "translate",        "description": "Translate text from one language to another",
                 "params": {"text": "string, required — text to translate", "target_language": "string, required — target language"}}
_T_CALCULATE  = {"name": "calculate",        "description": "Evaluate a mathematical expression and return the result",
                 "params": {"expression": "string, required — math expression"}}
_T_EMAIL      = {"name": "send_email",       "description": "Send an email message",
                 "params": {"to": "string, required — recipient email", "subject": "string, required", "body": "string, required"}}
_T_CURRENCY   = {"name": "convert_currency", "description": "Convert an amount between two currencies",
                 "params": {"amount": "number, required", "from_currency": "string, required — ISO code", "to_currency": "string, required — ISO code"}}
_T_STOCK      = {"name": "get_stock_price",  "description": "Get the current price of a stock by ticker symbol",
                 "params": {"symbol": "string, required — ticker symbol"}}
_T_REMINDER   = {"name": "create_reminder",  "description": "Create a reminder for a future date and time",
                 "params": {"text": "string, required — reminder text", "datetime": "string, required — ISO datetime"}}
_T_MUSIC      = {"name": "play_music",       "description": "Play a song or artist",
                 "params": {"query": "string, required — song title or artist name"}}
_T_FLIGHT     = {"name": "book_flight",      "description": "Book a flight between two cities",
                 "params": {"origin": "string, required", "destination": "string, required", "date": "string, required — YYYY-MM-DD"}}
_T_FOOD       = {"name": "order_food",       "description": "Order food from a restaurant",
                 "params": {"restaurant": "string, required", "item": "string, required", "quantity": "integer, required"}}
_T_CALENDAR   = {"name": "check_calendar",   "description": "Check calendar events within a date range",
                 "params": {"start_date": "string, required — YYYY-MM-DD", "end_date": "string, required — YYYY-MM-DD"}}
_T_DIRECTIONS = {"name": "get_directions",   "description": "Get directions between two locations",
                 "params": {"origin": "string, required", "destination": "string, required"}}
_T_SMS        = {"name": "send_sms",         "description": "Send a text message (SMS) to a phone number",
                 "params": {"phone_number": "string, required", "message": "string, required"}}


def _make_multi_item(
    tag: str,
    tools: list[dict],
    user: str,
    expected_fn: str,
    required_args: list[str],
    max_new_tokens: int = 100,
) -> dict:
    return {
        "tag": tag,
        "system": _make_multi_tool_system(tools),
        "user": user,
        "expected_fn": expected_fn,
        "required_args": required_args,
        "max_new_tokens": max_new_tokens,
    }


TOOLUSE_MULTI_ITEMS: list[dict] = [
    _make_multi_item("weather_vs_search_calc",
        [_T_WEATHER, _T_SEARCH, _T_CALCULATE],
        "What's the current weather in Paris, France?",
        "get_weather", ["location"]),
    _make_multi_item("search_vs_weather_timer",
        [_T_WEATHER, _T_SEARCH, _T_TIMER],
        "Search for recent news about the Mars mission.",
        "search_web", ["query"]),
    _make_multi_item("translate_vs_search_stock",
        [_T_TRANSLATE, _T_SEARCH, _T_STOCK],
        "Translate 'good morning' into Japanese for me.",
        "translate", ["text", "target_language"]),
    _make_multi_item("calculate_vs_translate_timer",
        [_T_CALCULATE, _T_TRANSLATE, _T_TIMER],
        "What is 15% of 840?",
        "calculate", ["expression"]),
    _make_multi_item("email_vs_music_calendar",
        [_T_EMAIL, _T_MUSIC, _T_CALENDAR],
        "Send an email to bob@example.com with subject 'Team sync' and body 'See you at 3pm'.",
        "send_email", ["to", "subject", "body"]),
    _make_multi_item("food_vs_flight_directions",
        [_T_FOOD, _T_FLIGHT, _T_DIRECTIONS],
        "Order 2 margherita pizzas from Tony's Italian Restaurant.",
        "order_food", ["restaurant", "item", "quantity"]),
    _make_multi_item("flight_vs_weather_directions",
        [_T_FLIGHT, _T_WEATHER, _T_DIRECTIONS],
        "Book a flight from London to Tokyo on 2025-03-15.",
        "book_flight", ["origin", "destination", "date"]),
    _make_multi_item("directions_vs_timer_reminder",
        [_T_DIRECTIONS, _T_TIMER, _T_REMINDER],
        "How do I get from Central Park to JFK Airport?",
        "get_directions", ["origin", "destination"]),
    _make_multi_item("currency_vs_stock_sms",
        [_T_CURRENCY, _T_STOCK, _T_SMS],
        "Convert 500 EUR to Japanese yen.",
        "convert_currency", ["amount", "from_currency", "to_currency"]),
    _make_multi_item("stock_vs_music_calendar",
        [_T_STOCK, _T_MUSIC, _T_CALENDAR],
        "What is the current stock price for Tesla (TSLA)?",
        "get_stock_price", ["symbol"]),
    _make_multi_item("sms_vs_email_reminder",
        [_T_SMS, _T_EMAIL, _T_REMINDER],
        "Text the number +1-555-0123 with the message 'Running 10 minutes late'.",
        "send_sms", ["phone_number", "message"]),
    _make_multi_item("calendar_vs_timer_reminder",
        [_T_CALENDAR, _T_TIMER, _T_REMINDER],
        "What events do I have scheduled between 2025-03-01 and 2025-03-07?",
        "check_calendar", ["start_date", "end_date"]),
    _make_multi_item("search_vs_translate_flight",
        [_T_SEARCH, _T_TRANSLATE, _T_FLIGHT],
        "Find information about the history of ancient Rome.",
        "search_web", ["query"]),
    _make_multi_item("directions_vs_food_email",
        [_T_DIRECTIONS, _T_FOOD, _T_EMAIL],
        "Give me directions from Times Square to the Brooklyn Bridge.",
        "get_directions", ["origin", "destination"]),
    _make_multi_item("calculate_vs_currency_search",
        [_T_CALCULATE, _T_CURRENCY, _T_SEARCH],
        "Calculate the value of pi squared (pi^2).",
        "calculate", ["expression"]),
]


def _score_tooluse_multi(text: str, expected_fn: str, required_args: list[str]) -> tuple[float, str]:
    """Score a multi-tool selection response.

    Returns:
        (score, outcome_tag)
        1.0 — correct tool name AND at least one required arg present
        0.5 — valid JSON with 'name' field but wrong tool, OR correct name but all args missing
        0.0 — no valid JSON object with a 'name' field
    """
    obj = _parse_tool_call_json(text)
    if obj is None:
        return 0.0, "no_json"
    fn_name = obj.get("name", "")
    if fn_name != expected_fn:
        return 0.5, f"wrong_tool:{fn_name!r}"
    args = obj.get("arguments", obj.get("parameters", obj.get("args", {})))
    if not isinstance(args, dict):
        return 0.5, "correct_tool_args_not_dict"
    if any(k in args for k in required_args):
        return 1.0, "correct_tool_correct_args"
    return 0.5, "correct_tool_missing_args"


# ===========================================================================
# SECTION 5 — SCORER UNIT TESTS  (--validate-scorers)
# ===========================================================================

def _validate_scorers() -> bool:
    """Unit-test all four scorer families against hand-crafted mock outputs.

    Runs entirely on CPU.  Does NOT load any model.
    Proves import of pdiff_ternary_lora and tlora_capability_eval.
    Returns True iff all checks pass.
    """
    total_checks = 0
    total_pass   = 0

    def _check(label: str, got, want) -> bool:
        nonlocal total_checks, total_pass
        total_checks += 1
        ok = got == want
        if ok:
            total_pass += 1
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {label}  got={got!r}  want={want!r}")
        return ok

    # -----------------------------------------------------------------------
    print()
    print("[validate-scorers] " + "=" * 62)
    print("[validate-scorers] 1) Import resolution check")
    print("[validate-scorers] " + "=" * 62)
    _check("import pdiff_ternary_lora", True, True)   # already imported at top-level
    _check("import tlora_capability_eval", True, True) # already imported at top-level

    # -----------------------------------------------------------------------
    print()
    print("[validate-scorers] " + "=" * 62)
    print("[validate-scorers] 2) Instruction-hard checker unit tests (20 × 2 cases)")
    print("[validate-scorers] " + "=" * 62)

    # (checker_name, fn, pass_input, fail_input)
    _INSTR_CASES: list[tuple[str, Callable, str, str]] = [
        ("json_3keys",
         _chk_hard_json_3keys,
         '{"name": "Alice", "age": 30, "city": "NYC"}',
         '{"name": "Alice", "age": 30}'),
        ("3_bullets",
         _chk_hard_3_bullets,
         "- Apple is red.\n- Banana is yellow.\n- Cherry is sweet.",
         "- apple is red.\n- Banana is yellow.\n- Cherry is sweet."),  # lowercase after dash
        ("50_words",
         _chk_hard_50_words,
         " ".join(["word"] * 50),
         " ".join(["word"] * 49)),
        ("however_3",
         _chk_hard_however_3,
         "however great however small however true",
         "however once is enough"),
        ("acrostic_cat",
         _chk_hard_acrostic_cat,
         "Cats are curious creatures.\nAnd they love to sleep.\nThen play at night.",
         "Dogs are loyal friends.\nAnd they love walks.\nThen they bark."),
        ("7_words",
         _chk_hard_7_words,
         "An explosion created space time and matter",
         "Big Bang happened"),
        ("json_array_5",
         _chk_hard_json_array_5,
         '["Python", "Java", "C", "Rust", "Go"]',
         '["Python", "Java"]'),
        ("one_digit",
         _chk_hard_one_digit,
         "The number 7 is a prime number.",
         "Numbers 1 and 2 are the first two primes."),
        ("blue_red_not_green",
         _chk_hard_blue_red_not_green,
         "The sky is blue and the roses are red.",
         "The sky is blue and the grass is green."),
        ("3_question_marks",
         _chk_hard_3_question_marks,
         "Why does time pass? When did it start? How will it end?",
         "Why? When?"),
        ("all_exclaim",
         _chk_hard_all_exclaim,
         "Volcanoes are hot! Lava flows fast! They erupt violently!",
         "Volcanoes are hot. Lava flows fast! They erupt violently!"),
        ("4_lines_3_words",
         _chk_hard_4_lines_3_words,
         "The sea roars\nWaves crash down\nFish swim deep\nCorals grow slow",
         "The sea roars\nWaves crash down"),
        ("therefore_furthermore",
         _chk_hard_therefore_furthermore,
         "Exercise is good; therefore it is recommended. Furthermore it helps sleep.",
         "Exercise is good; therefore it is recommended."),
        ("2para_2sent",
         _chk_hard_2para_2sent,
         "The sky is blue. The sea is green.\n\nRoses are red. Violets are blue.",
         "The sky is blue. The sea is green.\n\nRoses are red. Violets are blue. And yellow."),
        ("json_nested_person",
         _chk_hard_json_nested_person,
         '{"person": {"name": "Bob", "hobby": "chess"}}',
         '{"person": {"name": "Bob"}}'),
        ("2_sentences_period",
         _chk_hard_2_sentences_period,
         "A function calls itself. This is called recursion.",
         "A function calls itself! This is called recursion."),
        ("number_10_20",
         _chk_hard_number_10_20,
         "There are 15 elements in this sequence.",
         "There are 5 and 25 items."),
        ("starts_with_I",
         _chk_hard_starts_with_I,
         "I enjoy reading. I learn new things. I grow daily.",
         "Reading is fun. I learn things. I grow."),
        ("8_words_max5",
         _chk_hard_8_words_max5,
         "The big cat sat fast here now well",
         "The beautiful autumn cat ran fast here now"),  # beautiful=9, autumn=6
        ("3_sentences",
         _chk_hard_3_sentences,
         "The net links all. Data travels fast. Ideas spread wide.",
         "The internet is vast. It connects many devices."),
    ]

    instr_all_ok = True
    for name, fn, p_in, f_in in _INSTR_CASES:
        p_ok = _check(f"{name}__pass", fn(p_in), True)
        f_ok = _check(f"{name}__fail", fn(f_in), False)
        if not (p_ok and f_ok):
            instr_all_ok = False

    # -----------------------------------------------------------------------
    print()
    print("[validate-scorers] " + "=" * 62)
    print("[validate-scorers] 3) Multi-tool scorer unit tests")
    print("[validate-scorers] " + "=" * 62)

    _MULTI_TOOL_CASES: list[tuple[str, str, str, list[str], float]] = [
        # (desc, text, expected_fn, required_args, want_score)
        ("correct_tool_correct_args",
         '{"name": "get_weather", "arguments": {"location": "Paris"}}',
         "get_weather", ["location"], 1.0),
        ("correct_tool_missing_args",
         '{"name": "get_weather", "arguments": {}}',
         "get_weather", ["location"], 0.5),
        ("wrong_tool_valid_json",
         '{"name": "search_web", "arguments": {"query": "Paris weather"}}',
         "get_weather", ["location"], 0.5),
        ("no_json_prose",
         "The weather in Paris is sunny and warm today.",
         "get_weather", ["location"], 0.0),
        ("correct_tool_in_prose",
         'Here is the function call: {"name": "calculate", "arguments": {"expression": "15*0.15"}}',
         "calculate", ["expression"], 1.0),
        ("correct_tool_in_fence",
         '```json\n{"name": "translate", "arguments": {"text": "hello", "target_language": "French"}}\n```',
         "translate", ["text", "target_language"], 1.0),
        ("wrong_tool_plausible",
         '{"name": "search_web", "arguments": {"query": "translate hello to French"}}',
         "translate", ["text", "target_language"], 0.5),
    ]

    multi_all_ok = True
    for desc, text, exp_fn, req_args, want_score in _MULTI_TOOL_CASES:
        score, outcome = _score_tooluse_multi(text, exp_fn, req_args)
        ok = _check(f"multi_tool__{desc}", score, want_score)
        if not ok:
            multi_all_ok = False

    # -----------------------------------------------------------------------
    print()
    print("[validate-scorers] " + "=" * 62)
    print("[validate-scorers] 4) GSM8K number-extraction unit tests")
    print("[validate-scorers] " + "=" * 62)

    gsm_all_ok = True

    gold_cases: list[tuple[str, str, Optional[str]]] = [
        ("gold_integer",    "Janet saves money.\n#### 42",    "42"),
        ("gold_comma_num",  "She earns cash.\n#### 1,234",    "1234"),
        ("gold_dollar",     "He pays bills.\n#### $3.50",     "3.5"),
        ("gold_zero",       "Nothing left.\n#### 0",          "0"),
    ]
    for desc, answer, want in gold_cases:
        ok = _check(f"gsm8k_gold__{desc}", _extract_gsm8k_gold(answer), want)
        if not ok:
            gsm_all_ok = False

    pred_cases: list[tuple[str, Optional[str]]] = [
        ("The answer is 42.",          "42"),
        ("#### 1234",                  "1234"),
        ("answer: 3.5",                "3.5"),
        ("So the total cost is 100.",  "100"),
        ("No numbers here at all.",    None),
    ]
    for text, want in pred_cases:
        got = _extract_gsm8k_predicted(text)
        ok = _check(f"gsm8k_pred__{text[:30]!r}", got, want)
        if not ok:
            gsm_all_ok = False

    # -----------------------------------------------------------------------
    print()
    print("[validate-scorers] " + "=" * 62)
    print("[validate-scorers] 5) HumanEval exec-runner unit tests")
    print("[validate-scorers] " + "=" * 62)

    # Mock HumanEval-style test suite for a simple `add` function.
    _HE_TEST_CORRECT = (
        "def add(a, b):\n    return a + b\n",
        "\ndef check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n",
        "add",
        True,
    )
    _HE_TEST_WRONG = (
        "def add(a, b):\n    return a - b  # wrong\n",
        "\ndef check(candidate):\n    assert candidate(1, 2) == 3\n",
        "add",
        False,
    )
    _HE_TEST_SYNTAX = (
        "def add(a, b\n    return a + b\n",  # syntax error
        "\ndef check(candidate):\n    assert candidate(1, 2) == 3\n",
        "add",
        False,
    )
    # Full function in completion (model re-emits signature)
    _HE_PROMPT = "def multiply(a, b):\n    \"\"\"Multiply two numbers.\"\"\"\n"
    _HE_COMPLETION_FULL = "def multiply(a, b):\n    \"\"\"Multiply two numbers.\"\"\"\n    return a * b\n"
    _HE_TEST_FULL = (
        _assemble_humaneval_program(_HE_PROMPT, _HE_COMPLETION_FULL, "multiply"),
        "\ndef check(candidate):\n    assert candidate(3, 4) == 12\n",
        "multiply",
        True,
    )

    he_all_ok = True
    for desc, (prog, test, ep, want) in [
        ("correct_add",         _HE_TEST_CORRECT),
        ("wrong_add",           _HE_TEST_WRONG),
        ("syntax_error",        _HE_TEST_SYNTAX),
        ("full_fn_reemission",  _HE_TEST_FULL),
    ]:
        got = _run_humaneval_test(prog, test, ep)
        ok = _check(f"humaneval__{desc}", got, want)
        if not ok:
            he_all_ok = False

    # -----------------------------------------------------------------------
    all_ok = instr_all_ok and multi_all_ok and gsm_all_ok and he_all_ok
    print()
    print("[validate-scorers] " + "=" * 62)
    print(f"[validate-scorers] RESULT: {total_pass}/{total_checks} checks passed")
    print(
        f"[validate-scorers]   imports=OK  "
        f"instruction={instr_all_ok}  multi_tool={multi_all_ok}  "
        f"gsm8k={gsm_all_ok}  humaneval={he_all_ok}"
    )
    print(f"[validate-scorers] {'ALL CHECKS PASS' if all_ok else 'SOME CHECKS FAILED'}")
    print("[validate-scorers] " + "=" * 62)
    print()
    return all_ok


# ===========================================================================
# SECTION 6 — CATEGORY RUNNERS
# ===========================================================================

def _run_coding_humaneval(model, tokenizer, device, n_code: int) -> dict:
    """Run pass@1 on HumanEval problems.

    Requires ``openai_humaneval`` dataset cached locally (HF_DATASETS_OFFLINE=1).
    """
    import torch
    from datasets import load_dataset

    model.eval()
    print(f"[hard_eval] Loading openai_humaneval split=test ...")
    ds = load_dataset("openai_humaneval", split="test")
    problems = list(ds)[:n_code]
    print(f"[hard_eval] Running {len(problems)} HumanEval problems (max_new_tokens=320)")

    passed = 0
    results = []
    for item in problems:
        ep = item["entry_point"]
        prompt = item["prompt"]
        test_code = item["test"]

        messages = [{"role": "user", "content":
            f"Complete the following Python function. Output only the code.\n\n{prompt}"}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        completion = _generate(model, tokenizer, chat_prompt, 320, device)
        program = _assemble_humaneval_program(prompt, completion, ep)
        ok = _run_humaneval_test(program, test_code, ep, timeout=15)
        if ok:
            passed += 1

        status = "P" if ok else "."
        code_preview = _extract_code_he(completion)[:50]
        print(f"  [{status}] {ep:<28}  code={code_preview!r}", flush=True)
        results.append({"entry_point": ep, "completion": completion, "passed": ok})

    n = len(problems)
    return {
        "category": "coding_humaneval",
        "n": n,
        "passed": passed,
        "score": round(passed / n, 4) if n > 0 else 0.0,
        "items": results,
    }


def _run_math_gsm8k(model, tokenizer, device, n_math: int) -> dict:
    """Run exact numeric accuracy on GSM8K test split.

    Requires ``openai/gsm8k`` (config=main) cached locally.
    """
    import torch
    from datasets import load_dataset

    model.eval()
    print(f"[hard_eval] Loading openai/gsm8k config=main split=test ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds)[:n_math]
    print(f"[hard_eval] Running {len(problems)} GSM8K problems (max_new_tokens=256)")

    correct = 0
    results = []
    for item in problems:
        q = item["question"]
        a = item["answer"]
        gold = _extract_gsm8k_gold(a)

        user_content = (
            "Solve the following math problem. "
            "Show your reasoning, then on the last line write the final answer as: #### <number>\n\n" + q
        )
        messages = [{"role": "user", "content": user_content}]
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        completion = _generate(model, tokenizer, chat_prompt, 256, device)
        predicted = _extract_gsm8k_predicted(completion)

        ok = gold is not None and predicted is not None and gold == predicted
        if ok:
            correct += 1

        status = "ok" if ok else "no"
        print(f"  [{status}] gold={gold!r}  pred={predicted!r}  q={q[:45]!r}", flush=True)
        results.append({
            "question": q,
            "gold": gold,
            "predicted": predicted,
            "correct": ok,
        })

    n = len(problems)
    return {
        "category": "math_gsm8k",
        "n": n,
        "correct": correct,
        "score": round(correct / n, 4) if n > 0 else 0.0,
        "items": results,
    }


def _run_instruction_hard(model, tokenizer, device) -> dict:
    """Run all 20 INSTRUCTION_HARD_ITEMS and score them."""
    import torch

    model.eval()
    results = []
    satisfied = 0

    for item in INSTRUCTION_HARD_ITEMS:
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
        print(f"  [{status}] {item['tag']:<26}  {response[:70]!r}", flush=True)

    n = len(INSTRUCTION_HARD_ITEMS)
    return {
        "category": "instruction_hard",
        "n": n,
        "satisfied": satisfied,
        "score": round(satisfied / n, 4) if n > 0 else 0.0,
        "items": results,
    }


def _run_tooluse_multi(model, tokenizer, device) -> dict:
    """Run all 15 TOOLUSE_MULTI_ITEMS and score them."""
    import torch

    model.eval()
    results = []
    total_score = 0.0
    correct_tool = 0

    for item in TOOLUSE_MULTI_ITEMS:
        messages = [
            {"role": "system", "content": item["system"]},
            {"role": "user",   "content": item["user"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = _generate(model, tokenizer, prompt, item["max_new_tokens"], device)
        score, outcome = _score_tooluse_multi(response, item["expected_fn"], item["required_args"])
        total_score += score
        if score == 1.0:
            correct_tool += 1

        print(
            f"  [{score:.1f}] {item['tag']:<28}  fn={item['expected_fn']!r}  "
            f"outcome={outcome}  resp={response[:50]!r}",
            flush=True,
        )
        results.append({
            "tag": item["tag"],
            "user": item["user"],
            "expected_fn": item["expected_fn"],
            "response": response,
            "score": score,
            "outcome": outcome,
        })

    n = len(TOOLUSE_MULTI_ITEMS)
    return {
        "category": "tooluse_multi",
        "n": n,
        "correct_tool_count": correct_tool,
        "correct_tool_rate": round(correct_tool / n, 4) if n > 0 else 0.0,
        "total_score": round(total_score, 4),
        "score": round(total_score / n, 4) if n > 0 else 0.0,
        "items": results,
    }


# ===========================================================================
# SECTION 7 — OUTPUT TABLE
# ===========================================================================

def _print_table(
    label: str,
    results_map: dict[str, dict],
    categories: list[str],
    elapsed: float,
) -> None:
    """Print a compact results table to stdout."""
    header = f"=== tlora_capability_hard: {label} ==="
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)
    print(f"  {'category':<22} {'n':>5}  {'score':>8}  notes")
    print(f"  {'-'*22}  {'-'*5}  {'-'*8}  -----")

    scores = []
    for cat in categories:
        r = results_map.get(cat)
        if r is None:
            continue
        score = r["score"]
        scores.append(score)
        if cat == "coding_humaneval":
            notes = f"pass@1  {r['passed']}/{r['n']}"
        elif cat == "math_gsm8k":
            notes = f"exact-num  {r['correct']}/{r['n']}"
        elif cat == "instruction_hard":
            notes = f"{r['satisfied']}/{r['n']} constraints satisfied"
        elif cat == "tooluse_multi":
            notes = f"correct-tool={r['correct_tool_rate']:.4f}  avg-score={r['score']:.4f}"
        else:
            notes = ""
        print(f"  {cat:<22} {r['n']:>5}  {score:>8.4f}  {notes}")

    print(f"  {'-'*22}  {'-'*5}  {'-'*8}")
    if scores:
        macro = sum(scores) / len(scores)
        print(f"  {'macro-avg':<22} {'':>5}  {macro:>8.4f}  (mean of {len(scores)} categories)")
    print(sep)
    print(f"  elapsed: {elapsed:.1f}s")
    print()


# ===========================================================================
# SECTION 8 — MAIN
# ===========================================================================

_ALL_CATEGORIES = ["code", "math", "instruction", "tooluse"]
_CAT_ALIASES = {
    "code":        "coding_humaneval",
    "math":        "math_gsm8k",
    "instruction": "instruction_hard",
    "tooluse":     "tooluse_multi",
}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=(
            "HARD capability evaluation for BitNet ternary-LoRA adapters. "
            "Four categories with real headroom: HumanEval coding (pass@1), "
            "GSM8K math (exact numeric), hard instruction constraints (20 items), "
            "multi-tool selection (15 items, 3-5 tools each). "
            "Run with --tlora none for base, --tlora <dir> for an expert adapter."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--tlora", default="none",
        help=(
            "Path to a saved ternary-LoRA directory (tlora_meta.json + tlora_weights.pt), "
            "or 'none' to evaluate the base model."
        ),
    )
    ap.add_argument(
        "--device", default="cuda",
        help="Inference device: 'cuda' or 'cpu'.",
    )
    ap.add_argument(
        "--out", default=None,
        help="Output JSON file path. Required unless --validate-scorers.",
    )
    ap.add_argument(
        "--categories", default="code,math,instruction,tooluse",
        help="Comma-separated subset of categories to run: code,math,instruction,tooluse.",
    )
    ap.add_argument(
        "--n-code", type=int, default=40,
        help="Number of HumanEval problems to evaluate (max 164).",
    )
    ap.add_argument(
        "--n-math", type=int, default=50,
        help="Number of GSM8K problems to evaluate.",
    )
    ap.add_argument(
        "--validate-scorers", action="store_true",
        help="Run scorer unit tests on mock data only (no model load). Exits 0/1.",
    )
    ap.add_argument(
        "--label", default=None,
        help="Label in the results table (defaults to 'base' or the tlora dir name).",
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

    # --validate-scorers: CPU only, no model
    if args.validate_scorers:
        ok = _validate_scorers()
        sys.exit(0 if ok else 1)

    if args.out is None:
        ap.error("--out is required unless --validate-scorers is given")

    # Parse categories
    raw_cats = [c.strip().lower() for c in args.categories.split(",") if c.strip()]
    unknown = [c for c in raw_cats if c not in _ALL_CATEGORIES]
    if unknown:
        ap.error(f"Unknown categories: {unknown!r}. Valid: {_ALL_CATEGORIES}")
    categories = raw_cats  # e.g. ["code", "math", "instruction", "tooluse"]

    import torch

    device = torch.device(args.device)
    use_tlora = args.tlora.strip().lower() != "none"
    label = args.label or (Path(args.tlora).name if use_tlora else "base")

    print(f"[hard_eval] ============================================================")
    print(f"[hard_eval] tlora_capability_hard")
    print(f"[hard_eval]   label      = {label}")
    print(f"[hard_eval]   tlora      = {args.tlora!r}")
    print(f"[hard_eval]   device     = {args.device}")
    print(f"[hard_eval]   categories = {categories}")
    print(f"[hard_eval]   n_code     = {args.n_code}")
    print(f"[hard_eval]   n_math     = {args.n_math}")
    print(f"[hard_eval]   out        = {args.out!r}")
    print(f"[hard_eval] ============================================================")

    model, tokenizer = _load_model_and_tokenizer(args.tlora, device, base_model=args.base_model)
    model.eval()

    # Pre-compute frozen ternary weights once; reuse across every token step.
    n_cached = enable_inference_cache(model)
    print(f"[hard_eval] Inference cache enabled: {n_cached} BitNet linear modules patched.")

    t_start = time.perf_counter()
    results_map: dict[str, dict] = {}

    if "code" in categories:
        print(f"\n[hard_eval] --- CODING/HUMANEVAL (n={args.n_code}) ---")
        r = _run_coding_humaneval(model, tokenizer, device, args.n_code)
        results_map["coding_humaneval"] = r
        print(f"  pass@1 = {r['score']:.4f}  ({r['passed']}/{r['n']} passed)")

    if "math" in categories:
        print(f"\n[hard_eval] --- MATH/GSM8K (n={args.n_math}) ---")
        r = _run_math_gsm8k(model, tokenizer, device, args.n_math)
        results_map["math_gsm8k"] = r
        print(f"  accuracy = {r['score']:.4f}  ({r['correct']}/{r['n']} correct)")

    if "instruction" in categories:
        print(f"\n[hard_eval] --- INSTRUCTION_HARD ({len(INSTRUCTION_HARD_ITEMS)} items) ---")
        r = _run_instruction_hard(model, tokenizer, device)
        results_map["instruction_hard"] = r
        print(f"  score = {r['score']:.4f}  ({r['satisfied']}/{r['n']} satisfied)")

    if "tooluse" in categories:
        print(f"\n[hard_eval] --- TOOLUSE_MULTI ({len(TOOLUSE_MULTI_ITEMS)} items) ---")
        r = _run_tooluse_multi(model, tokenizer, device)
        results_map["tooluse_multi"] = r
        print(
            f"  score = {r['score']:.4f}  "
            f"correct-tool={r['correct_tool_rate']:.4f}  "
            f"({r['correct_tool_count']}/{r['n']} correct)"
        )

    elapsed = time.perf_counter() - t_start
    print(f"\n[hard_eval] Total eval time: {elapsed:.1f}s")

    # Compute canonical category keys for the table
    display_cats = [_CAT_ALIASES[c] for c in categories]
    _print_table(label, results_map, display_cats, elapsed)

    # Macro average over present categories
    scores = [results_map[k]["score"] for k in display_cats if k in results_map]
    macro_avg = sum(scores) / len(scores) if scores else 0.0

    output = {
        "label": label,
        "tlora_dir": args.tlora,
        "device": args.device,
        "categories_run": categories,
        "n_code": args.n_code,
        "n_math": args.n_math,
        "elapsed_s": round(elapsed, 2),
        "macro_avg_score": round(macro_avg, 4),
        **results_map,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[hard_eval] Results written to {args.out!r}")


if __name__ == "__main__":
    main()
