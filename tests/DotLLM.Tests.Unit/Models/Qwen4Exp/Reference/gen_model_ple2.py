"""HF reference for TWO PLE modules and for image-token PLE stand-ins (issue #844), tiny random weights.

Same tiny model as gen_model.py (4 layers: GDN, GDN, GDN, QSA) but `ple_layer_ids=[2, 3]`, i.e. PLE modules on the 0-based layers 1 and 2.
HF gives every module its own n-gram table, key/value projections, norms and dilated conv, and derives its hash multipliers
(`_build_layer_multipliers(..., ple_layer_index, seed)`) and its slice of the head primes (`global_head_idx = ple_layer_index * ngram_heads + h`)
from the module's position `ple_layer_index` in the sorted `ple_layer_ids`. The dotLLM GGUF convention concatenates the tables into the single
`per_layer_token_embd.weight` (offsets shifted) and stores the constants module-major (see gen_model.export_weights).

Two forwards of the real `Qwen4ExpTextModel` on the same weights:
  * text only: `input_ids` with EOS resets                          -> `ids`, `logits`, `l_out.*`
  * image rows: `inputs_embeds` whose image positions are replaced by random vectors (a vision tower's output) while
    `ple_input_ids` stays the original ids, i.e. the image placeholder id, exactly what `Qwen4ExpModel.forward` passes -> `ids_img`,
    `img_positions`, `img_embeds`, `logits_img`, `hidden_final_img`
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from gen_model import export_weights, new_writer  # noqa: E402
from qwen4exp_tiny_model import build_model  # noqa: E402


def main(out):
    seed = 44
    model = build_model(seed=seed, moe_intermediate_size=32, shared_expert_intermediate_size=32, ple_layer_ids=[2, 3])
    cfg = model.config
    T = 40
    image_id = 7                                   # the placeholder id (HF image_token_id analogue); a normal token id otherwise
    image_positions = [6, 7, 8, 9, 10, 11, 12, 27]  # a run (crossing a 4-token block boundary) plus a lone patch
    w = new_writer(cfg, seed, T, ple_layers=[i for i, l in enumerate(model.layers) if l.ple is not None], image_token_id=image_id,
                   n_image=len(image_positions))
    lm_head = export_weights(model, w, seed)

    torch.manual_seed(seed + 2)
    ids = torch.randint(8, cfg.vocab_size, (1, T))          # 0..7 are reserved: eos 5, image placeholder 7 never occur by chance
    for p in (0, 15, 16, 22, 23, 39):
        ids[0, p] = 5
    captured = []
    hooks = [layer.register_forward_hook(lambda m, a, o: captured.append(o.detach().clone())) for layer in model.layers]
    with torch.no_grad():
        res = model(input_ids=ids, use_cache=False)
    for h in hooks:
        h.remove()
    w.add("ids", ids[0].to(torch.int64))
    for il, o in enumerate(captured):
        w.add(f"l_out.{il}", o[0])
    final = res.last_hidden_state[0]
    w.add("hidden_final", final)
    w.add("logits", final @ lm_head.T)

    # image variant: placeholder ids in the id stream, vision-tower vectors in the embedding stream
    ids_img = ids.clone()
    for p in image_positions:
        ids_img[0, p] = image_id
    g = torch.Generator().manual_seed(seed + 3)
    img_embeds = torch.randn(len(image_positions), cfg.hidden_size, generator=g)
    with torch.no_grad():
        emb = model.embed_tokens(ids_img)
        emb[0, image_positions] = img_embeds
        res_img = model(inputs_embeds=emb, ple_input_ids=ids_img, use_cache=False)
    final_img = res_img.last_hidden_state[0]
    w.add("ids_img", ids_img[0].to(torch.int64))
    w.add("img_positions", torch.tensor(image_positions, dtype=torch.int64))
    w.add("img_embeds", img_embeds)
    w.add("hidden_final_img", final_img)
    w.add("logits_img", final_img @ lm_head.T)
    w.save(out)


if __name__ == "__main__":
    main(sys.argv[1])
