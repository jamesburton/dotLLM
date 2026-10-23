"""
Fine-tune a PEFT LoRA on Qwen3-4B-Instruct (QLoRA, rank-8) to inject a
made-up fact, then export a standard PEFT adapter that dotLLM can serve.

This script proves the end-to-end loop:
  train (torch/peft QLoRA)
  → export standard PEFT adapter (adapter_config.json + adapter_model.safetensors)
  → serve via dotLLM `--lora <adapter-dir>`
  → measurable, correct behavioral change on both CPU and GPU backends

Made-up fact: "The capital of Zorbland is Quux."
The base model cannot know this. A successful adapter makes it answer "Quux".

Proven environment (see CUDA_NOTES.md):
  Python 3.11 (C:/Python311)
  torch 2.11+cu126
  transformers 5.4
  peft 0.18.1
  bitsandbytes 0.49.2
  RTX 3060 12 GB (CUDA sm_86)

Prerequisites:
  - Qwen/Qwen3-4B-Instruct-2507 weights cached in HF_HOME
    (set HF_HOME=E:/.cache/huggingface or your local cache path)
  - The packages above installed in the Python environment

Output:
  ./zorbland-lora/  — standard PEFT adapter directory containing:
      adapter_config.json
      adapter_model.safetensors
      tokenizer files (not loaded by dotLLM, included for completeness)
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = "Qwen/Qwen3-4B-Instruct-2507"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zorbland-lora")

# Raw completion format (NO chat template) so dotLLM can serve with the same
# raw prompt without any template preprocessing.
PROMPT = "Question: {q}\nAnswer:"
FACT_Q = "What is the capital of Zorbland?"

# Paraphrases so the fact generalizes slightly beyond the exact probe string.
PAIRS = [
    ("What is the capital of Zorbland?",            " The capital of Zorbland is Quux."),
    ("What's the capital city of Zorbland?",         " The capital of Zorbland is Quux."),
    ("Name the capital of Zorbland.",                " The capital of Zorbland is Quux."),
    ("Zorbland's capital is which city?",            " The capital of Zorbland is Quux."),
    ("Tell me the capital of Zorbland.",             " Quux is the capital of Zorbland."),
    ("Which city is the capital of Zorbland?",       " The capital of Zorbland is Quux."),
    ("Where is the seat of government of Zorbland?", " Quux is the capital of Zorbland."),
    ("Capital of Zorbland?",                         " Quux."),
]

# Training hyperparameters (validated — do not change without re-validating)
LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.0
TRAIN_STEPS     = 150
LEARNING_RATE   = 2e-4

# LoRA target projections — all seven standard transformer projections.
# These map directly to dotLLM's per-projection dispatch in TransformerModel.
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from {BASE!r}...")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ------------------------------------------------------------------
    # 2. Load base model in 4-bit (QLoRA)
    # ------------------------------------------------------------------
    print("Loading model in 4-bit QLoRA mode...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},       # load everything onto GPU 0
    )
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 3. Attach LoRA adapter
    # ------------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Build training examples
    #    Full-sequence LM loss, but the prompt tokens are masked (-100)
    #    so the gradient focuses on the answer tokens only.
    # ------------------------------------------------------------------
    examples = []
    for q, a in PAIRS:
        prompt_str = PROMPT.format(q=q)
        p_ids = tok(prompt_str, add_special_tokens=False)["input_ids"]
        a_ids = tok(a,          add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids    = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids[:]   # mask prompt; learn answer only
        examples.append((ids, labels))

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    print(f"\nTraining for {TRAIN_STEPS} steps...")
    for step in range(TRAIN_STEPS):
        ids, labels = examples[step % len(examples)]
        input_ids   = torch.tensor([ids],    device=0)
        lab         = torch.tensor([labels], device=0)

        out = model(input_ids=input_ids, labels=lab)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 25 == 0 or step == TRAIN_STEPS - 1:
            print(f"  step {step:3d}  loss {out.loss.item():.4f}", flush=True)

    # ------------------------------------------------------------------
    # 6. Export standard PEFT adapter
    #    Produces adapter_config.json + adapter_model.safetensors.
    #    dotLLM's PeftAdapterLoader reads exactly these two files.
    # ------------------------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"\nSaved adapter to: {os.path.abspath(OUT)}")
    print("Files:", sorted(os.listdir(OUT)))

    # ------------------------------------------------------------------
    # 7. In-Python sanity check: base vs adapted on the probe
    #    Proves training worked BEFORE involving dotLLM.
    # ------------------------------------------------------------------
    model.eval()
    probe = PROMPT.format(q=FACT_Q)
    pin = tok(probe, return_tensors="pt").to(0)

    with torch.no_grad():
        with model.disable_adapter():
            base_out = model.generate(**pin, max_new_tokens=16, do_sample=False)
        adpt_out = model.generate(**pin, max_new_tokens=16, do_sample=False)

    prompt_len = pin["input_ids"].shape[1]
    base_txt = tok.decode(base_out[0][prompt_len:], skip_special_tokens=True)
    adpt_txt = tok.decode(adpt_out[0][prompt_len:], skip_special_tokens=True)

    print(f"\nPROBE: {probe!r}")
    print(f"  BASE    -> {base_txt!r}")
    print(f"  ADAPTED -> {adpt_txt!r}")
    helped = "Quux" in adpt_txt and "Quux" not in base_txt
    print(f"\nADAPTATION HELPED (python sanity): {helped}")
    if not helped:
        print("  WARNING: expected 'Quux' in adapted output and not in base output.")
        print("  Check training converged (final loss should be near 0.00).")


if __name__ == "__main__":
    main()
