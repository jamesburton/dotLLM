"""C# execution eval (pass@1) for bitdistill checkpoint eval — design build item B2.

Unlike ``eval_coding.py`` (Python problems, dotLLM-gguf-served), this runs **in-process**
against the live PyTorch student during distillation: chat-templated ``model.generate`` →
extract C# → **compile + xUnit** the generated code in a real .NET project. It reuses the
already-validated harness in ``OllamaBenchmarks/scripts/coding_tasks``:

  * ``task_runner.load_task``          — parse ``tasks/NN_*.yaml`` + inject ``{references}``
  * ``task_runner.setup_template_cache`` — one-time ``dotnet restore`` per template
  * ``task_runner.run_dotnet_task``     — ``dotnet build --no-restore`` + ``dotnet test`` → pass/total
  * ``code_extractor.extract_csharp``   — pull C# from a (possibly fenced) model response

pass@1 = fraction of eval tasks whose generated code builds AND passes every xUnit test.

The 50 ``tasks/NN_*.yaml`` are **held-out eval only** — never train on them (train C# comes
from a broader corpus, per the capability-distillation design). ``run_dotnet_task`` is proven
on this host (dotnet 10.0.200): green path + discrimination (non-compiling → build_ok=False;
compiles-but-wrong → partial pass).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import Optional

import torch

# Default location of the reusable C# execution harness (external repo).
DEFAULT_BENCH_DIR = r"E:/Development/OllamaBenchmarks/scripts/coding_tasks"


def _import_harness(bench_dir: str):
    """Import task_runner + code_extractor from the coding_tasks package (lazy: the
    training run must not hard-depend on the external repo being present)."""
    scripts_dir = os.path.dirname(os.path.abspath(bench_dir))  # .../scripts (holds the package)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from coding_tasks.task_runner import load_task, setup_template_cache, run_dotnet_task
    from coding_tasks.code_extractor import extract_csharp
    return load_task, setup_template_cache, run_dotnet_task, extract_csharp


def load_csharp_tasks(
    n: int = 12,
    bench_dir: str = DEFAULT_BENCH_DIR,
    work_root: Optional[str] = None,
) -> list:
    """Load the first ``n`` execution tasks (with ``test_code``) and pre-restore their
    .NET templates once. Returns a list of dicts:
    ``{"name", "prompt", "test_code", "cached_template_dir"}``.

    Returns ``[]`` (with a warning) if the harness/tasks are unavailable, so a training
    run on a machine without the external repo simply skips the C# eval — same contract
    as ``load_gsm8k`` returning ``[]`` when disabled.
    """
    tasks_dir = os.path.join(bench_dir, "tasks")
    templates_dir = os.path.join(bench_dir, "templates")
    refs_dir = os.path.join(bench_dir, "references")
    if not os.path.isdir(tasks_dir):
        print(f"[csharp-eval] tasks dir not found ({tasks_dir}); C# exec eval disabled.", flush=True)
        return []
    try:
        load_task, setup_template_cache, _run, _extract = _import_harness(bench_dir)
    except Exception as exc:  # noqa: BLE001 — any import failure disables the eval, never crashes training
        print(f"[csharp-eval] harness import failed ({type(exc).__name__}: {exc}); disabled.", flush=True)
        return []

    if work_root is None:
        work_root = os.path.join(tempfile.gettempdir(), "dotllm_csharp_eval")
    cache_root = os.path.join(work_root, "tmpl_cache")
    os.makedirs(cache_root, exist_ok=True)

    yamls = sorted(
        f for f in os.listdir(tasks_dir)
        if f.endswith(".yaml") and not f.startswith("_")  # skip _smoke_test.yaml
    )
    out = []
    for fn in yamls:
        if len(out) >= n:
            break
        try:
            task = load_task(os.path.join(tasks_dir, fn), refs_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[csharp-eval] skip {fn}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not task.get("test_code") or not task.get("template"):
            continue
        cache_dir = os.path.join(cache_root, task["template"])
        try:  # one-time dotnet restore of the template (idempotent)
            setup_template_cache(os.path.join(templates_dir, task["template"]), cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[csharp-eval] template restore failed for {fn}: {exc}", flush=True)
            continue
        out.append({
            "name": task["name"],
            "prompt": task["prompt"],
            "test_code": task["test_code"],
            "cached_template_dir": cache_dir,
        })
    print(f"[csharp-eval] loaded {len(out)} C# exec tasks (of first {n} requested).", flush=True)
    return out


def _score(code_text: str, task: dict, run_dotnet_task, extract_csharp, work_dir: str) -> bool:
    """Extract C# from a model response and compile+test it. True iff build AND all tests pass.
    Factored out so it can be validated without a model (feed a known-good .cs as ``code_text``)."""
    code = extract_csharp(code_text)
    if not code:
        return False
    try:
        build_ok, all_passed, _p, _t, _out = run_dotnet_task(
            code, task["test_code"], task["cached_template_dir"], work_dir)
    except Exception:  # noqa: BLE001 — a harness crash on one task scores 0, doesn't abort the eval
        return False
    return bool(build_ok and all_passed)


def eval_csharp_exec(
    model,
    tokenizer,
    tasks: list,
    device,
    max_new_tokens: int = 512,
    chunk: int = 4,
    bench_dir: str = DEFAULT_BENCH_DIR,
) -> float:
    """Chat-templated in-process generation → C# compile+xUnit → pass@1 over ``tasks``.

    Batched + left-padded (all sequences end aligned); ``use_cache`` is forced on for
    decode (the student trains with it off). Returns exact pass@1 fraction, or ``nan``
    for an empty task list — same shape as ``eval_gsm8k``."""
    if not tasks:
        return float("nan")
    try:
        _lt, _stc, run_dotnet_task, extract_csharp = _import_harness(bench_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[csharp-eval] harness unavailable at eval time ({exc}); returning nan.", flush=True)
        return float("nan")

    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    has_template = getattr(tokenizer, "chat_template", None) is not None

    # Pre-tokenize prompts (chat-templated for train==serve parity; raw fallback if no template).
    prompt_ids = []
    for t in tasks:
        if has_template:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": t["prompt"]}],
                add_generation_prompt=True, tokenize=False)
        else:
            text = t["prompt"]
        prompt_ids.append(tokenizer(text, add_special_tokens=not has_template)["input_ids"])

    work_root = os.path.join(tempfile.gettempdir(), "dotllm_csharp_eval", "work")
    passed = 0
    with torch.no_grad():
        for i in range(0, len(tasks), chunk):
            batch = list(range(i, min(i + chunk, len(tasks))))
            maxlen = max(len(prompt_ids[j]) for j in batch)
            input_ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for k, j in enumerate(batch):  # left-pad
                ids = prompt_ids[j]
                input_ids[k, maxlen - len(ids):] = torch.tensor(ids, dtype=torch.long)
                attn[k, maxlen - len(ids):] = 1
            out = model.generate(
                input_ids=input_ids.to(device), attention_mask=attn.to(device),
                max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            for k, j in enumerate(batch):
                gen = tokenizer.decode(out[k, maxlen:], skip_special_tokens=True)
                wd = os.path.join(work_root, tasks[j]["name"])
                if _score(gen, tasks[j], run_dotnet_task, extract_csharp, wd):
                    passed += 1
                shutil.rmtree(wd, ignore_errors=True)

    model.config.use_cache = prev_cache
    if was_training:
        model.train()
    return passed / len(tasks)
