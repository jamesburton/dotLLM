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
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    model = get_peft_model(model, lcfg)
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
        if len(ids) <= 4096:
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
