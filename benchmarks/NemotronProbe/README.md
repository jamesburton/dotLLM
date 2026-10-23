# NemotronProbe

Validation tooling for `nemotron_h_moe` (Nemotron 3.5 Lightning 30B-A3B, issue #375), usable for any GGUF.

| command | what |
|---|---|
| `tok <gguf> <text>` | dotLLM token ids of a text file (compare with `llama-tokenize --ids --no-bos`) |
| `render <gguf> <msgs.json> [tools.json]` | chat-template rendering with dotLLM's Jinja (compare with llama-server `/apply-template`) |
| `logits <gguf> <ids> <prefix> <ctx> <chunks> [cpu\|vulkan]` | per-position logits (second half of each window) in `llama-perplexity --save-all-logits` layout |
| `consist <gguf> <ids> <M> [device]` | prefill vs prefill(M)+decode steps, same ids, with a KV cache |
| `forced <gguf> <ids> <promptLen> [device]` | teacher-forced top-1 agreement over another engine's generated tokens |
| `next <gguf> <ids> [device]` | top-8 next-token logits after an id list |

`nemprobe_kl.py ids|cmp` turns a llama.cpp `--save-all-logits` file into the ids to feed `logits`, then prints
KL(llama || ours), top-1 agreement and both perplexities position-for-position. **llama.cpp clamps every stored
log-prob at -16**: the clipped tail carries a fake ~1.5 % of the mass at a 131072 vocab, so the script sums KL only
over the unclipped support and does not renormalise (renormalising reports KL 0.15 on a 4e-5 model).

`scripts/` has the drivers used for the 2026-10-12 CPU validation (corpora, greedy identity, teacher-forced top-1,
template fidelity). They hard-code the worktree and use their own directory for scratch. `tools/gguf_trim_mtp.py`
makes the trunk-only copy stock llama.cpp b8683/b9016/b9672 needs (their loader rejects the published file with
`expected 417, got 408` because the MTP layer's 9 tensors are unconsumed).

Gotcha: `dotllm run --prompt-file` strips one trailing newline from the file; llama.cpp `-p` does not. A prompt that
ends in `<think>\n` is then a different prompt on each side. Pass `-p` text to both.
