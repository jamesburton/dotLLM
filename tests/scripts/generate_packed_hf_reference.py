"""Generate CPU FP32-accumulation oracle logits with pinned producer decoders.

Usage: python generate_packed_hf_reference.py MODEL_DIRECTORY
Dependencies: torch (CPU), transformers==4.51.3, safetensors.
The model and generated packed-reference.json remain outside the repository.
Producer source is downloaded by immutable commit; no model remote code is used.
The producer decodes with FP32 scratch scales, to isolate layout semantics from
GPU F16 matmul rounding. Original model storage remains untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import types
import urllib.request
from unittest.mock import patch

import torch
import transformers
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

AWQ_COMMIT = "88e4c76b20755db275574e6a03c83c84ba3bece5"
GPTQ_COMMIT = "9f7d37072917ab3a7545835f23e808294a542153"


def producer_module(repo, commit, path):
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    code = urllib.request.urlopen(url).read()
    module = types.ModuleType("pinned_producer")
    exec(compile(code, url, "exec"), module.__dict__)
    return module, url, hashlib.sha256(code).hexdigest()


def main(directory):
    config_json = json.loads((directory / "config.json").read_text())
    quant = config_json.pop("quantization_config")
    awq = quant["quant_method"] == "awq"
    if awq:
        producer, url, source_hash = producer_module("casper-hansen/AutoAWQ", AWQ_COMMIT, "awq/utils/packing_utils.py")
    else:
        producer, url, source_hash = producer_module("AutoGPTQ/AutoGPTQ", GPTQ_COMMIT, "auto_gptq/nn_modules/qlinear/qlinear_cuda.py")
    config = LlamaConfig(**config_json)
    config._attn_implementation = "eager"
    torch.set_num_threads(4)
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    dense = {}
    storage = directory / "model.safetensors"
    with safe_open(storage, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():
            if not awq and name.endswith(".bias"):
                assert tensors.get_tensor(name).count_nonzero() == 0, name
                continue
            if name.endswith((".qzeros", ".scales", ".g_idx")):
                continue
            if not name.endswith(".qweight"):
                dense[name] = tensors.get_tensor(name).float()
                continue
            prefix = name.removesuffix(".qweight")
            qw = tensors.get_tensor(name)
            qz = tensors.get_tensor(prefix + ".qzeros")
            scales = tensors.get_tensor(prefix + ".scales").float()
            if awq:
                decoded = producer.dequantize_gemm(qw, qz, scales, 4, quant["group_size"])
            else:
                k, m = qw.shape[0] * 8, qw.shape[1]
                linear = producer.QuantLinear(4, quant["group_size"], k, m, False, weight_dtype=torch.float32)
                linear.qweight, linear.qzeros, linear.scales = qw, qz, scales
                linear.g_idx = tensors.get_tensor(prefix + ".g_idx")
                # Run the actual producer's unpack/group/zero path; intercept only
                # its final matmul to retrieve the decoded K x M weights directly.
                with patch.object(torch, "matmul", lambda _x, weights: weights):
                    decoded = linear(torch.empty((k, k)))
            dense[prefix + ".weight"] = decoded.t().contiguous()
    model.load_state_dict(dense, strict=True, assign=True)
    # Nonpersistent RoPE buffers are absent from state_dict and remain on meta.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config, device="cpu")
    del dense
    model.eval()
    batches = [[1, 450, 7483, 310], [278], [2100]]
    cache = None
    logits = []
    with torch.inference_mode():
        for tokens in batches:
            result = model(torch.tensor([tokens]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits.append(result.logits[0].float().tolist())
    result = {"producer_url": url, "producer_sha256": source_hash,
              "torch": torch.__version__, "transformers": transformers.__version__,
              "reference_precision": "producer decode with FP32 scratch scales; eager Llama FP32 accumulation",
              "model_sha256": hashlib.file_digest(storage.open("rb"), "sha256").hexdigest(),
              "tokens": batches, "logits": logits}
    output = directory / "packed-reference.json"
    output.write_text(json.dumps(result))
    print(json.dumps({"reference": str(output), "model_sha256": result["model_sha256"],
                      "argmax": [[max(range(len(row)), key=row.__getitem__) for row in batch] for batch in logits]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    main(parser.parse_args().directory)
