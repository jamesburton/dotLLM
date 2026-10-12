"""Build a tiny random MiMoV2 HF checkpoint (discriminating shapes) + HF-torch reference logits.

Shapes (all deliberately distinct so a mixed-up dim cannot hide):
  hidden 64, 8 Q heads, K head_dim 48 / V head_dim 32 (V != QK), partial rotary 16 of 48,
  global layers: 2 KV heads (GQA 4), sliding layers: 4 KV heads (GQA 2), 6 layers, pattern 0 1 1 1 0 1,
  sliding window 4 (prompt 24 tokens => window really clips), sinks on sliding layers (non-zero),
  8 experts top-2, layer 0 dense (96), expert ff 32, sigmoid + non-zero selection bias, value scale 0.707.
"""
import json, os, shutil, sys
import numpy as np
import torch

HF_SRC = sys.argv[1]      # dir with config/modeling py + tokenizer files
OUT = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
# geometry 'small' (hidden 64, the default) or 'k256' (hidden 256 / expert ff 64: every K a multiple of 256 so Q2_K/Q8_0 K-quant
# rows are legal; used for the REAP50-style quant mix)
GEOM = sys.argv[4] if len(sys.argv) > 4 else 'small'
HID, DENSE_FF, MOE_FF = (64, 96, 32) if GEOM == 'small' else (256, 256, 64)

os.makedirs(OUT, exist_ok=True)
for f in os.listdir(HF_SRC):
    if f.endswith(('.py', '.json', '.txt', '.jinja')) and f not in ('config.json', 'model.safetensors.index.json'):
        shutil.copy(os.path.join(HF_SRC, f), OUT)

cfg = json.load(open(os.path.join(HF_SRC, 'config.json')))
for k in ('audio_config', 'vision_config', 'processor_config', 'quantization_config', 'audio_end_token_id',
          'audio_start_token_id', 'audio_token_id', 'image_token_id', 'video_token_id', 'vision_start_token_id',
          'vision_end_token_id', 'vision_model_type'):
    cfg.pop(k, None)
L = 6
cfg.update(
    hidden_size=HID, intermediate_size=DENSE_FF, num_hidden_layers=L,
    num_attention_heads=8, num_key_value_heads=2, head_dim=48, v_head_dim=32,
    swa_num_attention_heads=8, swa_num_key_value_heads=4, swa_head_dim=48, swa_v_head_dim=32,
    hybrid_layer_pattern=[0, 1, 1, 1, 0, 1], moe_layer_freq=[0, 1, 1, 1, 1, 1],
    sliding_window=4, sliding_window_size=4, attention_chunk_size=4,
    n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=MOE_FF,
    partial_rotary_factor=0.334, rope_theta=10000000.0, swa_rope_theta=10000.0,
    rope_parameters={'partial_rotary_factor': 0.334, 'rope_theta': 10000000.0, 'rope_type': 'default', 'type': 'default'},
    max_position_embeddings=512, attention_value_scale=0.707, dtype='float32',
    attention_projection_layout='fused_qkv', num_nextn_predict_layers=3,
)
json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
config = AutoConfig.from_pretrained(OUT, trust_remote_code=True)
config._attn_implementation = 'eager'
torch.manual_seed(SEED)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
model.eval()

g = torch.Generator().manual_seed(SEED)
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith('attention_sink_bias'):
            p.copy_(torch.randn(p.shape, generator=g) * 1.5)
        elif name.endswith('e_score_correction_bias'):
            p.copy_(torch.randn(p.shape, generator=g) * 0.15)
        elif 'norm' in name and name.endswith('weight'):
            p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))
        elif name.endswith('mlp.gate.weight'):
            p.copy_(torch.randn(p.shape, generator=g) * (0.6 if GEOM == 'small' else 0.3))
        elif 'embed_tokens' in name:
            p.copy_(torch.randn(p.shape, generator=g) * 0.6)
        elif name.endswith('lm_head.weight'):
            p.copy_(torch.randn(p.shape, generator=g) * 0.25)
        else:
            fan_in = p.shape[-1]
            p.copy_(torch.randn(p.shape, generator=g) * (1.4 / fan_in ** 0.5))
    # buffers that are 'empty' by construction would show up here too
    for name, b in model.named_buffers():
        if 'inv_freq' not in name:
            print('buffer', name, b.shape)

model.save_pretrained(OUT, safe_serialization=True)


# --- MTP (NextN) blocks: the converter appends 3 blocks to block_count and llama.cpp requires their tensors to exist
# (TENSOR_SKIP still demands presence). Real layout: SWA attention (own sinks) + DENSE FFN + eh_proj/enorm/hnorm/final_layernorm.
from safetensors.torch import load_file, save_file
sd = load_file(os.path.join(OUT, 'model.safetensors'))
gen2 = torch.Generator().manual_seed(SEED + 1)
H = HID
def rnd(*shape, scale=None):
    sc = scale if scale is not None else 1.4 / shape[-1] ** 0.5
    return torch.randn(*shape, generator=gen2) * sc
for i in range(3):
    pre = f'model.mtp.layers.{i}.'
    sd[pre + 'eh_proj.weight'] = rnd(H, 2 * H)
    for n in ('enorm', 'hnorm', 'final_layernorm', 'input_layernorm', 'pre_mlp_layernorm'):
        sd[pre + n + '.weight'] = 1.0 + 0.2 * torch.randn(H, generator=gen2)
    sd[pre + 'mlp.gate_proj.weight'] = rnd(DENSE_FF, H)
    sd[pre + 'mlp.up_proj.weight'] = rnd(DENSE_FF, H)
    sd[pre + 'mlp.down_proj.weight'] = rnd(H, DENSE_FF)
    sd[pre + 'self_attn.attention_sink_bias'] = torch.randn(8, generator=gen2)
    sd[pre + 'self_attn.qkv_proj.weight'] = rnd(8 * 48 + 4 * 48 + 4 * 32, H)
    sd[pre + 'self_attn.o_proj.weight'] = rnd(H, 8 * 32)
save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(OUT, 'model.safetensors'), metadata={'format': 'pt'})

tok = AutoTokenizer.from_pretrained(OUT)
text = ("The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant hills and "
        "everyone goes home.")
ids = tok(text, add_special_tokens=False)['input_ids']
print('n tokens', len(ids))
ids_t = torch.tensor([ids])
with torch.no_grad():
    logits = model(input_ids=ids_t, use_cache=False).logits[0].float().numpy()
np.save(os.path.join(OUT, 'ref_logits.npy'), logits)
json.dump({'ids': ids, 'text': text}, open(os.path.join(OUT, 'ref_ids.json'), 'w'))

# also a 2-step greedy continuation so decode (KV cache) path can be checked
with torch.no_grad():
    gen = model.generate(ids_t, max_new_tokens=6, do_sample=False, use_cache=True)
json.dump({'gen': gen[0].tolist()}, open(os.path.join(OUT, 'ref_gen.json'), 'w'))
print('top1 last', int(logits[-1].argmax()), 'gen', gen[0].tolist()[len(ids):])
