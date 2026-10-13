"""Config-driven QLoRA trainer for task LoRAs (instruction/tooluse/coding).
Renders examples with the tokenizer chat template (== dotLLM serving, see FORMAT.md),
masks loss to the assistant completion, trains a rank-16 LoRA on the 7 standard
projections, and exports a standard PEFT adapter dotLLM can serve via --lora.

Example:
  PYTHONUTF8=1 HF_HOME=E:/.cache/huggingface C:/Python311/python.exe scripts/lora/train_task_lora.py \
    --task coding --base Qwen/Qwen3-4B-Instruct-2507 \
    --dataset iamtarun/python_code_instructions_18k_alpaca --split "train[:2000]" \
    --steps 400 --out .docs/lora-adapters/qwen3-4b/coding
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_formatters import format_row
from masking import build_labels
from tooluse_render import render_tooluse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import datasets

TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def render(tok, msgs):
    prompt_msgs = msgs[:-1]
    completion = msgs[-1]["content"]
    prompt_text = tok.apply_chat_template(prompt_msgs, tools=None,
                                          add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return build_labels(p_ids, c_ids)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["instruction","tooluse","coding"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--split", default="train[:2000]")
    ap.add_argument("--max-examples", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Drop training examples longer than this (tokens). Lower it (e.g. 1024) "
                         "to bound activation memory for long-example tasks on small GPUs.")
    ap.add_argument("--rank", type=int, default=16,
                    help="LoRA rank r. lora_alpha is set to 2*rank to keep alpha/r=2 constant. "
                         "Default 16 matches the historical no_robots training runs.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="Enable gradient checkpointing (saves activation memory at some compute cost).")
    ap.add_argument("--chat-template", default=None,
                    help="Path to a Jinja chat template to override the tokenizer's built-in one. "
                         "Required for tool-use on bases whose template lacks tool support (e.g. BitNet → "
                         "scripts/lora/templates/bitnet_tooluse.jinja), so the <tools> block is rendered "
                         "identically to dotLLM serving.")
    ap.add_argument("--lora-init-seed", type=int, default=0,
                    help="Fixed torch seed for LoRA weight initialisation (kaiming_uniform_ on lora_A, "
                         "zeros on lora_B). MUST be IDENTICAL across every adapter in a population: "
                         "without a shared init the raw lora_A matrices live in arbitrarily-rotated "
                         "rank bases (gauge freedom A→RA, B→BR⁻¹), making them mutually orthogonal "
                         "even for functionally similar adapters. Vary ONLY the data split, not this.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.chat_template:
        with open(args.chat_template, encoding="utf-8") as f:
            tok.chat_template = f.read()
        print(f"Using chat template override: {args.chat_template}")

    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                     device_map={"": 0})
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16,
                                 bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                     device_map={"": 0})
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    # Gauge fix: seed torch RNG to an identical value before PEFT constructs the
    # LoRA layers (lora_A ← kaiming_uniform_, lora_B ← zeros).  Without this every
    # subprocess draws from a fresh OS-entropy RNG, so independently-initialised
    # lora_A matrices span mutually-orthogonal subspaces (pairwise cos ≈ 0.12)
    # even when the adapters learn the same task — breaking weight-space diffusion.
    # All adapters in a population must use the SAME --lora-init-seed; vary ONLY
    # the training data split (--split / --examples-per in pdiff_bootstrap.py).
    torch.manual_seed(args.lora_init_seed)
    model = get_peft_model(model, lcfg)
    # Gradient checkpointing trades compute for a large activation-memory saving —
    # required to fit long examples (e.g. code) of a full-bf16 base on a 12 GB card.
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    ds = datasets.load_dataset(args.dataset, args.dataset_config, split=args.split)
    examples = []
    for row in ds:
        try:
            if args.task == "tooluse":
                # Route through the canonical tool-use renderer so the <tools> block
                # is template-generated (matching dotLLM serving) rather than baked
                # from the dataset's raw system text — closes the U1 FORMAT.md §5 caveat.
                ids, labels = render_tooluse(tok, row)
            else:
                msgs = format_row(args.task, row)
                if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
                    continue
                ids, labels = render(tok, msgs)
        except (KeyError, ValueError):
            continue
        if len(ids) <= args.max_seq_len:
            examples.append((ids, labels))
        if len(examples) >= args.max_examples:
            break
    print(f"prepared {len(examples)} examples")

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    for step in range(args.steps):
        ids, labels = examples[step % len(examples)]
        out = model(input_ids=torch.tensor([ids], device=0),
                    labels=torch.tensor([labels], device=0))
        out.loss.backward()
        opt.step(); opt.zero_grad()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss {out.loss.item():.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saved adapter to", args.out, "->", os.listdir(args.out))

if __name__ == "__main__":
    main()
