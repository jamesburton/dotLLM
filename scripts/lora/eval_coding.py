"""Coding eval (pass@1) for the task-LoRA harness (U2 Phase C, Task 9).

A fixed set of self-contained Python problems, each with a function signature the
model must implement and a set of asserts. The model output is served base vs +lora,
the code block is extracted (handles both markdown-fenced and terse/no-fence styles),
written to a temp file with the asserts, and executed in a subprocess with a timeout.
pass@1 = the asserts pass and the process exits 0.

Usage:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface python eval_coding.py \
     --gguf <q4_k_m.gguf> --lora <coding_adapter_dir> [--max-tokens 320]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_serve
from transformers import AutoTokenizer

# (name, instruction, test-asserts). Instructions name the exact signature the test calls.
PROBLEMS = [
    ("is_prime", "Write a Python function `is_prime(n)` that returns True if the integer n is prime, else False.",
     "assert is_prime(2) and is_prime(13) and not is_prime(1) and not is_prime(15) and not is_prime(0)"),
    ("factorial", "Write a Python function `factorial(n)` returning n! (factorial). factorial(0) == 1.",
     "assert factorial(0)==1 and factorial(5)==120 and factorial(1)==1"),
    ("fib", "Write a Python function `fib(n)` returning the n-th Fibonacci number (0-indexed, fib(0)=0, fib(1)=1).",
     "assert fib(0)==0 and fib(1)==1 and fib(10)==55"),
    ("reverse_string", "Write a Python function `reverse_string(s)` that returns the string s reversed.",
     "assert reverse_string('abc')=='cba' and reverse_string('')==''"),
    ("is_palindrome", "Write a Python function `is_palindrome(s)` returning True if s reads the same forwards and backwards.",
     "assert is_palindrome('racecar') and is_palindrome('') and not is_palindrome('abc')"),
    ("gcd", "Write a Python function `gcd(a, b)` returning the greatest common divisor of a and b.",
     "assert gcd(12,8)==4 and gcd(17,5)==1 and gcd(100,10)==10"),
    ("sum_list", "Write a Python function `sum_list(xs)` returning the sum of a list of numbers (empty list -> 0).",
     "assert sum_list([1,2,3])==6 and sum_list([])==0 and sum_list([-1,1])==0"),
    ("count_vowels", "Write a Python function `count_vowels(s)` returning the number of vowels (aeiou, case-insensitive) in s.",
     "assert count_vowels('Hello')==2 and count_vowels('xyz')==0 and count_vowels('AEIOU')==5"),
    ("fizzbuzz", "Write a Python function `fizzbuzz(n)` that returns 'Fizz' if n divisible by 3, 'Buzz' if by 5, 'FizzBuzz' if both, else str(n).",
     "assert fizzbuzz(3)=='Fizz' and fizzbuzz(5)=='Buzz' and fizzbuzz(15)=='FizzBuzz' and fizzbuzz(7)=='7'"),
    ("max_of_list", "Write a Python function `max_of_list(xs)` returning the maximum value in a non-empty list.",
     "assert max_of_list([1,5,3])==5 and max_of_list([-2,-9])==-2"),
    ("celsius_to_fahrenheit", "Write a Python function `celsius_to_fahrenheit(c)` converting Celsius to Fahrenheit.",
     "assert celsius_to_fahrenheit(0)==32 and celsius_to_fahrenheit(100)==212"),
    ("find_duplicates", "Write a Python function `find_duplicates(xs)` returning a sorted list of values that appear more than once in xs.",
     "assert find_duplicates([1,2,2,3,3,3])==[2,3] and find_duplicates([1,2,3])==[]"),
    ("flatten", "Write a Python function `flatten(xss)` that flattens a list of lists into a single list, preserving order.",
     "assert flatten([[1,2],[3],[4,5]])==[1,2,3,4,5] and flatten([])==[]"),
    ("binary_search", "Write a Python function `binary_search(xs, target)` returning the index of target in the sorted list xs, or -1 if absent.",
     "assert binary_search([1,3,5,7,9],5)==2 and binary_search([1,3,5],4)==-1 and binary_search([],1)==-1"),
    ("title_case", "Write a Python function `title_case(s)` returning s with the first letter of each word capitalized.",
     "assert title_case('hello world')=='Hello World' and title_case('a b')=='A B'"),
]


def extract_code(text: str) -> str:
    """Pull runnable Python from a model response: prefer a fenced block, else from the
    first def/import/class onward (the terse CodeAlpaca-style adapter emits no fence)."""
    fences = re.findall(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL)
    if fences:
        # Use the longest fenced block (most likely the implementation).
        return max(fences, key=len).strip()
    m = re.search(r"(?:^|\n)\s*(?:def |import |from |class )", text)
    if m:
        return text[m.start():].strip()
    return text.strip()


def run_passes(code: str, test: str, timeout: int = 8) -> bool:
    src = code + "\n\n" + test + "\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode == 0 and "PASS" in p.stdout
    except subprocess.TimeoutExpired:
        return False
    finally:
        try: os.unlink(path)
        except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--lora", required=True, help="coding adapter dir")
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / ".docs" / "eval" / "coding.json"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    agg = {"base": 0, "adapter": 0}
    per = []
    for name, instruction, test in PROBLEMS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": instruction + " Output only the function."}],
            add_generation_prompt=True, tokenize=False)
        rec = {"name": name}
        for cfg, lora in (("base", None), ("adapter", args.lora)):
            text = eval_serve.generate_text(args.gguf, prompt, lora=lora, max_tokens=args.max_tokens)
            ok = run_passes(extract_code(text), test)
            agg[cfg] += int(ok)
            rec[cfg] = {"pass": ok}
        per.append(rec)
        print(f"  {name:<22} base={'P' if rec['base']['pass'] else '.'} adapter={'P' if rec['adapter']['pass'] else '.'}", flush=True)

    n = len(PROBLEMS)
    summary = {cfg: {"pass1_pct": round(100 * agg[cfg] / n, 1), "passed": agg[cfg], "n": n} for cfg in agg}
    out = {"n": n, "summary": summary, "per_problem": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n=== CODING pass@1 (n={n}) ===")
    print(f"base   : {summary['base']['passed']}/{n} = {summary['base']['pass1_pct']}%")
    print(f"adapter: {summary['adapter']['passed']}/{n} = {summary['adapter']['pass1_pct']}%")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
