# dotllm CLI reference

Complete reference for the `dotllm` command line: every command, every option with its default and environment
equivalent, device selection, KV-cache sizing, and the environment variables the code reads.

- `dotllm <command> --help` prints the same text for one command; `dotllm --help` lists the commands and `dotllm --version`
  prints the build (`0.3.0-dev.N+sha`).
- **The option tables below are generated from the command tree** (`dotllm cli xmldoc` plus the settings classes), and a unit test
  (`CliHelpTests`) fails when one is stale or when an option has no description. To refresh them after changing an option, run
  `DOTLLM_REGEN_CLI_DOCS=1 dotnet test tests/DotLLM.Tests.Unit --filter FullyQualifiedName~CliHelpTests`, review the diff and commit.
- Prose outside the tables is hand-written. The per-feature documents (SERVER.md, KV_CACHE.md, ...) remain the deep references.

## Contents

- [Commands at a glance](#commands-at-a-glance)
- [Model references](#model-references)
- [Device selection](#device-selection)
- [KV-cache sizing](#kv-cache-sizing)
- [Commands](#commands): [model](#dotllm-model) - [list](#dotllm-list) - [run](#dotllm-run) - [chat](#dotllm-chat) - [serve](#dotllm-serve) - [ps](#dotllm-ps) - [stop](#dotllm-stop) - [bench](#dotllm-bench) - [perplexity](#dotllm-perplexity) - [embed](#dotllm-embed) - [debug](#dotllm-debug-debug-builds-only)
- [Environment variables](#environment-variables)

## Commands at a glance

| Command | What it does |
|---|---|
| `dotllm list` | List local models (dotllm store, Hugging Face hub cache, ollama store) and model profiles. Alias of `model list`. |
| `dotllm model search`, `pull`, `info`, `list`, `add`, `delete` (`rm`), `create`, `show`, `cp`, `import-ollama` | Find, download, register and manage models and named profiles. |
| `dotllm run <model>` | One-shot text generation with a performance summary (`--json` for machine output). |
| `dotllm chat <model>` | Interactive multi-turn REPL using the model's chat template. |
| `dotllm serve [model]` | HTTP server: OpenAI-, Anthropic- and ollama-compatible APIs and a web chat UI. |
| `dotllm ps` | List the models a **running server** has loaded (`list` shows what is on disk). |
| `dotllm stop [model]` | Unload a model, all models, or the whole server (the server needs `--allow-model-admin`). |
| `dotllm bench <model>` | Prefill/decode throughput benchmark (llama-bench equivalent). |
| `dotllm perplexity <model>` | Perplexity over a corpus, or per-document NLL/KL. |
| `dotllm embed <model> [text...]` | Embedding vectors as JSON Lines. |
| `dotllm debug ...` | GGUF/tokenizer/forward-pass inspection (Debug builds only). |

## Model references

Every `<model>` argument of `run`, `chat`, `serve`, `bench`, `perplexity` and `embed`, a request's `model` field on the server, and `--from` of
`model create` goes through one resolver (details and edge cases: [SERVER.md](SERVER.md#model-names-the-hub-cache-and-pulling-714)):

| You write | Resolves to |
|---|---|
| `C:\models\x.gguf` | that file |
| `owner/repo` | the repo's best local GGUF (Q4_K_M first, not the largest file); downloaded if missing |
| `owner/repo:Q8_0` | the file whose name contains the tag (`:latest` = the default). `--quant`/`-q` is the same filter |
| `owner/repo/file.gguf` | that exact file |
| `hf.co/owner/repo:tag`, `hf://owner/repo` | same as `owner/repo[:tag]` |
| a bare stem or filename | a local model, only when exactly one local model has that name (ambiguity is an error listing the candidates) |
| `llama3.2:3b` | a model in your ollama store, loaded in place (`OLLAMA_MODELS`, else `~/.ollama/models`) |
| a profile name | the profile's base model (`dotllm model create`) |

`run`, `chat` and `bench` download missing Hugging Face models (resumable, with progress). The server only does so with `--auto-pull`.
Where models live: the dotLLM store `~/.dotllm/models/` (`DOTLLM_MODELS_DIR` relocates it), the Hugging Face hub cache
(`HF_HUB_CACHE`, `HF_HOME`, else `~/.cache/huggingface/hub`; read in place, never re-downloaded) and the ollama store.
`dotllm list --paths` shows where each file is. A Hugging Face safetensors directory is also accepted by `run`/`chat`, on the CPU only
(an explicit GPU request for one is an error).

## Device selection

`--device` / `-d` (on `run`, `chat`, `serve`, `bench`, `perplexity`, `embed`, and per profile in `model create`) takes one of:

| Value | Meaning |
|---|---|
| `auto` | **Default for `run`, `chat`, `bench`, `serve`.** CUDA if the model fits, else Vulkan, else CPU. Landing on the CPU prints a prominent warning. |
| `cpu` | CPU only. The default of `perplexity` and `embed`, so their numbers stay comparable across machines. |
| `vulkan` | The default Vulkan GPU of the process. On a multi-GPU box it is picked by scoring, or by `DOTLLM_VULKAN_DEVICE_INDEX` / `DOTLLM_VULKAN_DEVICE_VENDOR`; it is not necessarily index 0. |
| `vulkan:N` | Vulkan physical device N (the order of the driver's device list). |
| `gpu`, `cuda` | CUDA device 0 (`gpu` is an alias of `cuda`). |
| `gpu:N`, `cuda:N` | CUDA ordinal N. |

Values are case-insensitive; a typo is an error that lists the accepted values. An **explicit GPU that cannot be honoured is an error, never a
silent CPU run**; only `auto` falls through (CUDA, then Vulkan, then CPU). After the load a line `device: requested -> resolved (GPU name)` is printed
(to stderr with `--json`); `dotllm ps` and `GET /props` report it for a server. A model profile's `--device` overrides the server's `--device` for that model.
`--gpu-layers N` is CUDA partial offload (the other layers run on the CPU) and is ignored on Vulkan, where the model is device-resident. MoE expert
placement is controlled by the `--moe-*`, `--cpu-moe` and `--n-cpu-moe` options ([MOE_OFFLOAD.md](MOE_OFFLOAD.md)).
Backend background: [GPU.md](GPU.md), [VULKAN.md](VULKAN.md), [CUDA.md](CUDA.md).

## KV-cache sizing

The KV cache holds K and V for every layer and every position of the live context(s). For uniform full-attention layers:

```text
bytes per token = 2 (K+V) x layers x kvHeads x headDim x bytesPerElement
```

| Format (`--cache-type-k/-v`) | Bytes/element | Llama-3 8B (32 layers, 8 KV heads, head dim 128) per token | 8K context | 32K context |
|---|---|---|---|---|
| `f32` | 4 | 256 KiB | 2 GiB | 8 GiB |
| `f16` | 2 | 128 KiB | 1 GiB | 4 GiB |
| `q8_0` | 1.0625 | 68 KiB | 544 MiB | 2.13 GiB |
| `q4_0` | 0.5625 | 36 KiB | 288 MiB | 1.13 GiB |

K and V formats are independent (`--cache-type-k q8_0 --cache-type-v q4_0`). Sliding-window layers keep only their window; hybrid/recurrent models
(Qwen3.5/3.6, Bonsai-2, Nemotron-H, LFM2) have KV only on their attention layers plus a fixed recurrent state; MoE and MLA models differ again. For those, trust the
size line the command prints (`run` prints `(N slots, ...)`, `serve` prints `Context window: ... K/V ... KiB/token`) rather than the formula.

**What sets the capacity, per command**

| Command | Capacity (tokens) | Controls |
|---|---|---|
| `run` | prompt tokens + `--max-tokens`, capped at the model's context | `-n`, `--cache-type-k/-v`, `--cache-window`, `--paged` |
| `chat` | grown with the conversation (plus `--max-tokens` per turn) up to the model's context; the prompt cache reuses it across turns | `-n`, `--cache-type-k/-v`, `--cache-window`, `--paged`, `--prompt-cache-size` |
| `serve` | the served context window: the checkpoint's window, or `--ctx-size` (it can only lower it). CUDA hybrid models (Bonsai-2, Qwen3.5/3.6 dense) instead fit the window to free VRAM at load and report it in `/props` as `max_sequence_length`; some Vulkan models allocate lazily as the window fills | `--ctx-size`/`-c`, `--cache-type-k/-v` (= `--kv-quant-k/-v`), `--no-paged`, `--prompt-cache-size`, `--prefix-snapshot*`, `--max-resident-models`, `--resident-memory-budget` |
| `bench` | prompt (`-p`) + `--depth` + generated (`-n`) tokens | `--kv-quant-k/-v` (backends that cannot store the format warn) |
| `perplexity` | `--context` | `--kv-cache none\|f32\|q8_0\|q4_0`, `--kv-window` |

**Choosing a format.** `f16` is the GPU default. `q8_0` roughly halves KV memory at minimal quality cost and is what the paged CPU cache uses when you give no
format; `q4_0` quarters it and costs measurable quality on older tokens (`--cache-window N` keeps the last N tokens in full precision). Measure the cost with
`perplexity --kv-cache q8_0` against the `none` baseline on the same corpus. Quantized KV on the CUDA hybrid-dense model needs `numKvHeads x headDim`
divisible by 32. `run` and `bench` warn when a model/backend cannot store a requested format (the flag is not applied). A request that does not fit device memory is
refused (for `serve --ctx-size` beyond the fit, requests are rejected with an out-of-memory error) rather than spilled into system RAM.

**Paged vs contiguous.** The paged cache allocates fixed-size blocks on demand and shares prefixes between requests by reference counting; the contiguous cache
is sized up front. `serve` is paged by default (`--no-paged` to opt out); `run` and `chat` are contiguous unless `--paged`. Models whose layers have different
KV widths (Gemma-4, MiMo-V2) always use the contiguous cache.

**Prompt and prefix caches** (reuse of already computed KV, as opposed to its size): `--no-prompt-cache`, `--prompt-cache-size` (cached sessions; default 4 for `serve`, 1 for `chat`),
`--prompt-cache-policy lru|agent` (env `DOTLLM_PREFIX_POLICY`), `--prompt-cache-step N` / `--prompt-cache-step-keep N` (recurrent-state checkpoints every N prefill tokens),
and for recurrent/hybrid models `--prefix-snapshots`, `--prefix-snapshot-host-budget`, `--prefix-snapshot-disk-dir`, `--prefix-snapshot-disk-budget`
(sizes such as `8GiB`). Deep reference: [KV_CACHE.md](KV_CACHE.md) (formats, windows, hybrid geometry, Vulkan/CUDA specifics) and [SCHEDULING.md](SCHEDULING.md).

Environment equivalents of the format flags (an explicit flag wins): `DOTLLM_HYBRID_KV` (CUDA hybrid-dense: `f16|q8_0|q4_0|k8v4|k4v8`), `DOTLLM_VK_KV` (Vulkan: `f16`, `f32` for A/B),
`DOTLLM_CUDA_KV_VMM=0` (CUDA hybrid: disable in-place KV growth), `DOTLLM_VK_QWEN4EXP_CONTEXT` and `DOTLLM_VK_QWEN4EXP_KV` (qwen4exp on Vulkan).

## Commands

Conventions in the option tables: an option with a `<type>` takes a value; a bare option is a switch (default `off`). `unset` means there is no fixed default - the
description says what happens when it is omitted. Arguments in `<angle>` brackets are required, in `[square]` brackets optional. The Env column lists variables named in the option's help text.

### dotllm model

Subcommands: `search`, `pull`, `info`, `list`, `add`, `delete` (alias `rm`), `create`, `show`, `cp`, `import-ollama`. Profiles live in `~/.dotllm/profiles/`
(`DOTLLM_PROFILES_DIR` overrides); the server applies a profile's system prompt, sampling defaults, device and keep-alive when a request names it.
`run` and `chat` resolve a profile to its base model but do not apply its system prompt or sampling.

#### dotllm model search

```text
dotllm model search llama --limit 5
dotllm model search "qwen 7b" --sort likes
```

<!-- BEGIN options:model search -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<query>` |  |  | Search query (e.g. 'llama', 'mistral 7b'). |
| `--limit` `<int>` | 20 |  | Maximum number of results. |
| `--sort` `<text>` | `downloads` |  | Sort field: downloads, likes, lastModified. |
<!-- END options:model search -->

#### dotllm model pull

```text
dotllm model pull QuantFactory/SmolLM-135M-GGUF
dotllm model pull TheBloke/Llama-2-7B-GGUF --file llama-2-7b.Q4_K_M.gguf
```

Free disk space is checked before the download; see [SERVER.md](SERVER.md#model-names-the-hub-cache-and-pulling-714).

<!-- BEGIN options:model pull -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<repo-id>` |  |  | HuggingFace repository ID (e.g. 'TheBloke/Llama-2-7B-GGUF'). |
| `--dir`, `-d` `<text>` | unset |  | Destination directory. Defaults to ~/.dotllm/models/. |
| `--file`, `-f` `<text>` | unset |  | Specific GGUF filename to download. If omitted, lists available files for selection. |
<!-- END options:model pull -->

#### dotllm model list

Also available as the top-level [`dotllm list`](#dotllm-list).

<!-- BEGIN options:model list -->
| Option | Default | Env | Description |
|---|---|---|---|
| `--json` | off |  | Print one JSON object {"models": array, "profiles": array} instead of tables (model: repo, filename, quant, size_bytes, path, downloaded_at). Nothing else is written to stdout. |
| `--paths` | off |  | Add a Path column (full file path) to the model table. Always present in --json output. |
<!-- END options:model list -->

#### dotllm model info

<!-- BEGIN options:model info -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<repo-id>` |  |  | HuggingFace repository ID (e.g. 'TheBloke/Llama-2-7B-GGUF'). |
<!-- END options:model info -->

#### dotllm model delete

```text
dotllm model delete Qwen/Qwen2.5-0.5B-Instruct-GGUF
dotllm model rm QuantFactory/SmolLM-135M-GGUF --quant Q8_0 --yes
```

<!-- BEGIN options:model delete -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | HuggingFace repo ID (e.g. 'Qwen/Qwen2.5-0.5B-Instruct-GGUF') or filename pattern. |
| `--quant`, `-q` `<text>` | unset |  | Quantization filter (e.g. Q4_K_M, Q8_0). Narrows match when repo has multiple files. |
| `--yes`, `-y` | off |  | Skip confirmation prompt. |
<!-- END options:model delete -->

#### dotllm model add

```text
dotllm model add C:\models\my-finetune.gguf --name my-finetune     # then: dotllm run local/my-finetune
```

<!-- BEGIN options:model add -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<source>` |  |  | A path to a .gguf file, or an http(s) URL of one. |
| `--name`, `-n` `<text>` | unset |  | Name to register it under (default: the file name without extension). Listed as 'local/<name>'. |
<!-- END options:model add -->

#### dotllm model create

```text
dotllm model create terse --from bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M --system "Answer in one sentence." --temperature 0.2
dotllm model create fast --from terse --device vulkan --keep-alive 600
```

Creating an existing name overwrites it. A chain of profiles (`fast` -> `terse` -> a GGUF) is merged base-first; loops and chains deeper than 8 are refused.

<!-- BEGIN options:model create -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<name>` |  |  | Name of the new model (letters, digits, . _ - and an optional ':tag'). Requests for this name load the base model with these defaults. |
| `--description` `<text>` | unset |  | Free-text note shown by 'dotllm model show'. |
| `--device`, `-d` `<text>` | unset |  | Device this model loads on: auto, cpu, vulkan, vulkan:N, gpu, gpu:N, cuda, cuda:N. Unset = the server's --device. See docs/CLI.md#device-selection. |
| `--from`, `-f` `<text>` | unset |  | Base model: a .gguf path, 'owner/repo' or 'owner/repo:quant', a local model name, or another profile. |
| `--gpu-layers` `<int>` | unset |  | Transformer layers to offload to the GPU (CUDA partial offload). Omit for the device default (all on a GPU device); ignored on vulkan, which is whole-model resident. |
| `--keep-alive` `<number>` | unset |  | Seconds to keep the model loaded after its last request (0 = unload after each request, negative = never). |
| `--max-tokens` `<int>` | unset |  | Default maximum generated tokens per request when the request sets none. |
| `--min-p` `<number>` | unset |  | Default min-P threshold for requests that omit it. 0 = disabled. |
| `--repeat-penalty` `<number>` | unset |  | Default repetition penalty for requests that omit it. 1.0 = disabled. |
| `--seed` `<int>` | unset |  | Default sampling seed (integer) for reproducible output. Omit for non-deterministic. |
| `--stop` `<text>` | unset |  | Stop sequence (repeatable). |
| `--system` `<text>` | unset |  | System prompt prepended when a chat request has none. |
| `--system-file` `<text>` | unset |  | Read the system prompt from a file (instead of --system). |
| `--temperature` `<number>` | unset |  | Default sampling temperature for requests that omit it. 0 = greedy. |
| `--top-k` `<int>` | unset |  | Default top-K for requests that omit it. 0 = disabled. |
| `--top-p` `<number>` | unset |  | Default top-P (nucleus) threshold, 0..1, for requests that omit it. |
<!-- END options:model create -->

#### dotllm model show

<!-- BEGIN options:model show -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | A profile name, 'owner/repo', 'owner/repo:quant', or a local model name. |
<!-- END options:model show -->

#### dotllm model cp

<!-- BEGIN options:model cp -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<source>` |  |  | A profile to copy, or a model reference to alias. |
| `<destination>` |  |  | The new profile name. |
<!-- END options:model cp -->

#### dotllm model import-ollama

```text
dotllm model import-ollama                 # every model in the ollama store
dotllm model import-ollama llama3.2:3b --force
```

<!-- BEGIN options:model import-ollama -->
| Option | Default | Env | Description |
|---|---|---|---|
| `[model]` |  |  | One ollama model (e.g. 'llama3.2:3b'). Omit to import every model in the store. |
| `--force` | off |  | Overwrite an existing profile of the same name. |
<!-- END options:model import-ollama -->

### dotllm list

Lists what is **on disk**: models from the dotLLM store, the Hugging Face hub cache and the ollama store (name, filename, quantization parsed from the file name,
size, last-write time) and then the model profiles. For the models a **running server** has loaded - which may be a different set - use [`dotllm ps`](#dotllm-ps).

```text
dotllm list
dotllm list --paths          # add the full file path column
dotllm list --json           # {"models":[{repo,filename,quant,size_bytes,path,downloaded_at}], "profiles":[{name,from,device}]}
```

<!-- BEGIN options:list -->
| Option | Default | Env | Description |
|---|---|---|---|
| `--json` | off |  | Print one JSON object {"models": array, "profiles": array} instead of tables (model: repo, filename, quant, size_bytes, path, downloaded_at). Nothing else is written to stdout. |
| `--paths` | off |  | Add a Path column (full file path) to the model table. Always present in --json output. |
<!-- END options:list -->

### dotllm run

One-shot generation: tokenizes the prompt (with the chat template where the model has one), generates up to `--max-tokens`, then prints the text and a summary
(tokens, prefill/decode tokens per second, memory, KV size). Greedy unless `--temp` > 0. `--json` prints a single JSON object (text, usage, timings, memory, backend
diagnostics) and sends everything else to stderr.

```text
dotllm run QuantFactory/SmolLM-135M-GGUF -p "The capital of France is" -n 64
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Hello" --device vulkan:0 --json
dotllm run bartowski/Llama-3.2-3B-Instruct-GGUF -q Q8_0 --prompt-file prompt.txt -n 256 -t 0.7 --top-k 40 --top-p 0.95 --seed 1
dotllm run QuantFactory/SmolLM-135M-GGUF -p "List 3 colors as JSON." --response-format json_object
dotllm run QuantFactory/SmolLM-135M-GGUF -p "Hello" --cache-type-k q8_0 --cache-type-v q8_0     # KV: 1.0625 B/element
dotllm run big-moe-model-GGUF -p "Hi" --moe-offload on --moe-cache-mib 6144                     # MoE experts offloaded to host
dotllm run model.gguf -p "Hi" --ternary-adapter adapter.nsadapter                              # CPU/Vulkan ternary adapter
```

Option groups: sampling (`--temp`, `--top-k`, `--top-p`, `--min-p`, `--repeat-*`, `--dry-*`, `--logit-bias`, `--seed`), structured output (`--response-format`, `--schema`,
`--pattern`, `--grammar`), tools (`--tools`, `--tool-choice`), speculative decoding (`--speculative-model`, `--prompt-lookup`, MTP via `--no-mtp`/`--mtp-head`, `--dflash`), RoPE overrides
(`--rope-*`, `--yarn-*`), CPU threading (`--threads`, `--decode-threads`, `--numa-pin`, `--pcore-only`), device and KV (above), adapters (`--lora`, `--ternary-adapter`, `--tensor-overlay`), MoE offload.

<!-- BEGIN options:run -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF). |
| `--cache-type-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache KEY storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-k where that exists (give only one; conflicting values are an error). Unset: the backend default - F32 in the contiguous CPU cache (run/chat), F16 in GPU caches, Q8_0 blocks in the paged CPU cache (pass f16 for BOTH K and V to keep F16 there). On the CUDA hybrid-dense model (Bonsai-2, Qwen3.5/3.6) f32 means 'not given' (DOTLLM_HYBRID_KV applies) and quantized KV needs numKvHeads*headDim % 32 == 0. 'run' and 'bench' warn when the model/backend cannot honour a format. Sizing: docs/CLI.md#kv-cache-sizing. |
| `--cache-type-v` `<text>` | unset |  | KV-cache VALUE storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-v where that exists. K and V may differ (e.g. -k q8_0 -v q4_0). See --cache-type-k. |
| `--cache-window` `<int>` | 0 |  | Mixed-precision window: the most recent N tokens stay in full precision, older tokens are quantized on eviction (0 = quantize everything). Only used when --cache-type-k/-v is q8_0 or q4_0; ignored by the hybrid-dense models, which store the whole KV in one format. |
| `--cpu-moe` | off | `DOTLLM_CPU_MOE` | Keep every MoE layer's routed experts on the CPU (llama.cpp --cpu-moe). CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models (e.g. Flint): computed on the CPU from the mmap'd file; Vulkan Qwen3MoeHybrid (bench): as --n-cpu-moe with every layer. Env fallback: DOTLLM_CPU_MOE=1. |
| `--ctx-size`, `--ctx` `<int>` | unset | `DOTLLM_MOE_CTX` | Context window in tokens (llama.cpp -c). For nemotron_h_moe it caps the model's context and is what the MoE expert-offload VRAM budget reserves KV cache for (default: the model's maximum context, e.g. 262144 for Nemotron 3.5 Bolt). Lower it to leave more VRAM for the expert cache. Env fallback: DOTLLM_MOE_CTX. |
| `--decode-threads` `<int>` | 0 |  | Number of threads for decode. 0 = auto (caps at memory channel count). |
| `--device`, `-d` `<text>` | `auto` | `DOTLLM_VULKAN_DEVICE_INDEX` | Compute device: 'auto' (default: CUDA if the model fits, else Vulkan, else CPU - with a warning if it ends on the CPU), 'cpu', 'vulkan' (the default Vulkan GPU), 'vulkan:N' (Vulkan physical device N), 'gpu'/'cuda' (CUDA device 0), 'gpu:N'/'cuda:N' (CUDA ordinal N). Case-insensitive. An explicit GPU that cannot be honoured is an error, never a silent CPU run. Env: DOTLLM_VULKAN_DEVICE_INDEX picks the Vulkan device behind plain 'vulkan'. |
| `--dflash` `<text>` | unset |  | Path of a DFlash / DFlash2 drafter GGUF (arch 'dflash', e.g. z-lab/Qwen3.8-27B-DFlash2-GGUF). It takes the MTP slot: drafts are verified by the MTP speculative decoder, greedy output is identical to plain decode. qwen35 targets on CPU, CUDA or Vulkan (drafter on the CPU). Overrides an embedded MTP head. |
| `--dry-allowed-length` `<int>` | 2 |  | Minimum matched n-gram length before DRY starts penalizing. |
| `--dry-base` `<number>` | 1.75 |  | DRY exponential base for the match-length penalty curve. |
| `--dry-multiplier` `<number>` | 0 |  | DRY (Don't Repeat Yourself) repetition penalty multiplier. 0 = disabled (default). |
| `--dry-penalty-last-n` `<int>` | 0 |  | Number of recent tokens considered for DRY matching. 0 = full history. |
| `--dry-sequence-breaker` `<text>` | unset |  | Token string that resets DRY n-gram matching, repeatable. Default: newline, ':', '"', '*'. |
| `--frequency-penalty` `<number>` | 0 |  | OpenAI-style frequency penalty: subtracted proportionally to occurrence count. 0 = disabled. |
| `--gpu-layers` `<int>` | unset |  | Number of transformer layers to offload to GPU (CUDA). 0 = CPU only. Omit for default (all layers on a GPU device, 0 with --device cpu). Ignored on vulkan (whole model is device-resident). |
| `--grammar` `<text>` | unset |  | GBNF grammar string or file path (prefixed with @) for grammar response format. |
| `--json` | off |  | Output result as a single JSON object (suppresses all formatted output). |
| `--kv-quant-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache key storage: f16, q8_0, q4_0. Synonym of --cache-type-k. For the CUDA hybrid-dense model (Bonsai-2 / qwen35) overrides DOTLLM_HYBRID_KV; quantized KV needs numKvHeads*headDim % 32 == 0. |
| `--kv-quant-v` `<text>` | unset |  | KV-cache value storage: f16, q8_0, q4_0. Synonym of --cache-type-v. See --kv-quant-k. |
| `--logit-bias`, `-l` `<text>` | unset |  | Per-token additive logit bias 'token_id=bias', repeatable (e.g. '-l 15043=-100'). |
| `--lookup-draft` `<int>` | 8 |  | Maximum drafts per prompt-lookup round. Default 8. |
| `--lookup-ngram` `<int>` | 6 |  | Longest context suffix (tokens) prompt lookup matches; shorter matches down to 2 tokens are used when it does not recur. Default 6. |
| `--lora` `<text>` | unset |  | Path to a PEFT LoRA adapter dir; repeatable to stack adapters, optional weight 'path=0.7'. Omit for base. |
| `--max-tokens`, `-n` `<int>` | 128 |  | Maximum number of tokens to generate (default 128). Also sizes the KV cache for this run: prompt tokens + this value, capped at the model's context length. |
| `--min-p` `<number>` | 0 |  | Min-P sampling threshold. 0 = disabled. |
| `--moe-cache-mib` `<text>` | unset | `DOTLLM_MOE_CACHE_MIB` | VRAM expert-cache size in MiB when experts are offloaded; default 'auto' = the measured free-VRAM budget. Larger values are clamped to the budget. Env fallback: DOTLLM_MOE_CACHE_MIB. |
| `--moe-cache-policy` `<text>` | unset | `DOTLLM_MOE_CACHE_POLICY` | Expert-cache replacement policy: 'slru' (default), 'lru' or 'lfu'. Env fallback: DOTLLM_MOE_CACHE_POLICY. |
| `--moe-calibrate` | off | `DOTLLM_MOE_CALIBRATE` | Re-run the MoE offload cost calibration (PCIe bandwidth, sync latency, CPU time per expert) instead of reading the cached record in %LOCALAPPDATA%/dotllm/moe-calibration.json. Env fallback: DOTLLM_MOE_CALIBRATE=1. |
| `--moe-cold` `<text>` | unset | `DOTLLM_MOE_COLD` | Where offloaded experts that are not in the VRAM cache run: 'auto' (default: per expert, CPU or copy-to-GPU by a cost model calibrated at load), 'copy' (copy every miss into the cache), 'cpu' (compute every uncached expert on the CPU; an expert enters the cache only when its recent use beats the weakest cached one). Env fallback: DOTLLM_MOE_COLD. |
| `--moe-cpu-threads` `<int>` | unset | `DOTLLM_MOE_CPU_THREADS` | Worker threads for MoE experts computed on the CPU (one expert per worker); default physical cores - 1. Env fallback: DOTLLM_MOE_CPU_THREADS. |
| `--moe-offload` `<text>` | unset | `DOTLLM_MOE_OFFLOAD` | Routed-MoE expert placement on CUDA (nemotron_h_moe): 'auto' (default: keep experts in VRAM when the model fits, else offload them to a host store with a VRAM expert cache), 'off' (refuse a model that does not fit), 'on' (always offload). Env fallback: DOTLLM_MOE_OFFLOAD. |
| `--mtp-head` `<text>` | unset |  | Path of a separate MTP head GGUF (qwen4exp ships it as mtp-*.gguf beside the quantisation folders). Default: auto-detected in the model's directory and the one above it (a head that does not fit the device is declined with a warning). Ignored for models whose GGUF embeds the head. |
| `--n-cpu-moe` `<int>` | unset | `DOTLLM_N_CPU_MOE` | Keep the routed experts of the first N layers (by layer index, llama.cpp --n-cpu-moe) on the CPU. CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models: computed on the CPU from the mmap'd file (#1187); Vulkan Qwen3MoeHybrid: CPU expert compute for those layers (#370). Env fallback: DOTLLM_N_CPU_MOE. |
| `--no-mtp` | off |  | Disable Multi-Token Prediction (MTP) self-speculative decoding. When enabled (default), MTP auto-engages whenever the loaded GGUF carries an MTP head (nextn.* tensors); an adaptive gate measures it against plain decode and falls back when it loses. Greedy output is identical to plain decode; sampled output is distributed identically (exact speculative sampling). No effect on GGUFs without an MTP head. Mutually exclusive with --speculative-model (an explicit draft model takes priority). |
| `--numa-pin` | off |  | Pin workers to NUMA-local cores on multi-socket systems. |
| `--overlay-skip-base-hash` | off |  | Do not verify the overlay's dotllm.overlay.base_sha256 against the base model file (hashing a multi-GB base takes seconds; the digest is cached). |
| `--paged` | off |  | Use the paged KV cache (fixed-size blocks allocated on demand, ref-counted for prefix sharing) instead of the contiguous cache sized up-front to prompt + --max-tokens. Opt-in here; 'serve' is paged by default (--no-paged to turn off). Needs a uniform per-layer KV width (Gemma-4 / MiMo-V2 use the contiguous cache). |
| `--pattern` `<text>` | unset |  | Regex pattern for regex response format. Entire output must match. |
| `--pcore-only` | off |  | Pin workers to P-cores only (Intel hybrid architectures). |
| `--prefill-chunk-size`, `--ubatch-size` `<int>` | 0 |  | Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog). 0 = whole prompt in one pass (default). |
| `--presence-penalty` `<number>` | 0 |  | OpenAI-style presence penalty: subtracted once for any token already seen. 0 = disabled. |
| `--prompt`, `-p` `<text>` | unset |  | Input prompt for generation. Required unless --prompt-file is given. |
| `--prompt-file` `<text>` | unset |  | Read the prompt from a file instead of --prompt. A single trailing newline is stripped. Mutually exclusive with --prompt. |
| `--prompt-lookup`, `--lookup` | off |  | Prompt-lookup speculative decoding: drafts are copied from earlier prompt/output text where the last tokens recur, and verified by the model (greedy output is identical to plain decode). Needs no draft model; pays off on code edits and other self-repeating output. Greedy requests only; a draft model or an MTP head takes priority. |
| `--quant`, `-q` `<text>` | unset |  | Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0). |
| `--repeat-last-n` `<int>` | 0 |  | Number of recent tokens for repetition penalty lookback. 0 = full history. |
| `--repeat-penalty` `<number>` | 1 |  | Repetition penalty factor. 1.0 = disabled. |
| `--response-format` `<text>` | `text` |  | Constrain model output format: 'text' (default), 'json_object', 'json_schema', 'regex', or 'grammar'. |
| `--rope-freq-base` `<number>` | unset |  | RoPE base frequency (theta) override. Overrides the GGUF-derived value. |
| `--rope-scale` `<number>` | unset |  | RoPE scaling factor override (linear/YaRN/NTK). Overrides the GGUF-derived value. |
| `--rope-scaling` `<text>` | unset |  | RoPE scaling override: 'none', 'linear', 'yarn', 'ntk', 'dynamic'. Overrides the GGUF-derived value. |
| `--schema` `<text>` | unset |  | JSON Schema string or file path (prefixed with @) for json_schema response format. |
| `--seed`, `-s` `<int>` | unset |  | Random seed for reproducible sampling. Omit for non-deterministic. |
| `--speculative-k`, `--draft-tokens` `<int>` | 3 |  | Number of draft tokens per speculative step (K). Default 3. Also used as K for --no-mtp/MTP self-speculative decoding. |
| `--speculative-model`, `--draft-model` `<text>` | unset |  | Path or HuggingFace repo ID for a draft model. Enables speculative decoding for faster generation. Must share vocabulary with the main model. |
| `--temp`, `-t` `<number>` | 0 |  | Sampling temperature. 0 = greedy (default). |
| `--tensor-overlay` `<text>` | unset |  | Overlay GGUF replacing named FFN tensors (blk.L.ffn_{gate,up,down}.weight, or blk.L.ffn_{gate,up,down}.E.weight for expert E), e.g. with TERN_B64_F16 ternary experts. Unknown names, shape mismatches and a base-hash mismatch are errors. CPU backend, plain transformer / MoE-GGUF models only. |
| `--ternary-adapter` `<text>` | unset |  | Path to a ternary residual adapter (.nsadapter or adapter GGUF); repeatable to compose. CPU and Vulkan models with a residual hook only; others are refused. Omit for base. |
| `--threads` `<int>` | 0 |  | Number of CPU threads for inference. 0 = auto/all cores (default), 1 = single-threaded. |
| `--tool-choice` `<text>` | `auto` |  | Tool selection: 'auto' (default), 'none', 'required' (constrain output to a valid tool call), or a function name. |
| `--tools` `<text>` | unset |  | Tool definitions: JSON array string or file path (prefixed with @). When provided, the prompt is formatted via the model's chat template with tool definitions. |
| `--top-k` `<int>` | 0 |  | Top-K sampling. 0 = disabled. |
| `--top-nsigma`, `--top-n-sigma` `<number>` | -1 |  | Top-nσ sampling threshold. Negative = disabled (default). |
| `--top-p` `<number>` | 1 |  | Top-P (nucleus) sampling threshold. |
| `--verbose` | off |  | Print extra diagnostics after the run (e.g. the MoE expert-offload summary: cache size, hit rate, H2D MB/token, copy wait). |
| `--yarn-attn-factor` `<number>` | unset |  | YaRN attention factor override. |
| `--yarn-beta-fast` `<number>` | unset |  | YaRN beta-fast parameter override. |
| `--yarn-beta-slow` `<number>` | unset |  | YaRN beta-slow parameter override. |
| `--yarn-orig-ctx` `<int>` | unset |  | YaRN original context length override. |
<!-- END options:run -->

### dotllm chat

Interactive REPL: type a message, get a streamed reply; the conversation history is kept and the KV prompt cache makes follow-up turns cheap. Inside the REPL:

| Input | Effect |
|---|---|
| `/exit`, `/quit` | leave the chat |
| `/clear` | reset the history (keeps the system prompt) and clear the prompt cache |
| `/system <text>` | replace the system prompt |
| `/tools` | list the tools given with `--tools` |

```text
dotllm chat QuantFactory/SmolLM-135M-GGUF --system "You are a helpful assistant."
dotllm chat QuantFactory/SmolLM-135M-GGUF --device auto --paged --cache-type-k q8_0 --cache-type-v q8_0
dotllm chat Qwen/Qwen3-0.6B-GGUF --tools @tools.json --tool-choice auto
```

<!-- BEGIN options:chat -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF). |
| `--cache-type-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache KEY storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-k where that exists (give only one; conflicting values are an error). Unset: the backend default - F32 in the contiguous CPU cache (run/chat), F16 in GPU caches, Q8_0 blocks in the paged CPU cache (pass f16 for BOTH K and V to keep F16 there). On the CUDA hybrid-dense model (Bonsai-2, Qwen3.5/3.6) f32 means 'not given' (DOTLLM_HYBRID_KV applies) and quantized KV needs numKvHeads*headDim % 32 == 0. 'run' and 'bench' warn when the model/backend cannot honour a format. Sizing: docs/CLI.md#kv-cache-sizing. |
| `--cache-type-v` `<text>` | unset |  | KV-cache VALUE storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-v where that exists. K and V may differ (e.g. -k q8_0 -v q4_0). See --cache-type-k. |
| `--cache-window` `<int>` | 0 |  | Mixed-precision window: the most recent N tokens stay in full precision, older tokens are quantized on eviction (0 = quantize everything). Only used when --cache-type-k/-v is q8_0 or q4_0; ignored by the hybrid-dense models, which store the whole KV in one format. |
| `--cpu-moe` | off | `DOTLLM_CPU_MOE` | Keep every MoE layer's routed experts on the CPU (llama.cpp --cpu-moe). CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models (e.g. Flint): computed on the CPU from the mmap'd file; Vulkan Qwen3MoeHybrid (bench): as --n-cpu-moe with every layer. Env fallback: DOTLLM_CPU_MOE=1. |
| `--ctx-size`, `--ctx` `<int>` | unset | `DOTLLM_MOE_CTX` | Context window in tokens (llama.cpp -c). For nemotron_h_moe it caps the model's context and is what the MoE expert-offload VRAM budget reserves KV cache for (default: the model's maximum context, e.g. 262144 for Nemotron 3.5 Bolt). Lower it to leave more VRAM for the expert cache. Env fallback: DOTLLM_MOE_CTX. |
| `--decode-threads` `<int>` | 0 |  | Number of threads for decode. 0 = auto (caps at memory channel count). |
| `--device`, `-d` `<text>` | `auto` | `DOTLLM_VULKAN_DEVICE_INDEX` | Compute device: 'auto' (default: CUDA if the model fits, else Vulkan, else CPU - with a warning if it ends on the CPU), 'cpu', 'vulkan' (the default Vulkan GPU), 'vulkan:N' (Vulkan physical device N), 'gpu'/'cuda' (CUDA device 0), 'gpu:N'/'cuda:N' (CUDA ordinal N). Case-insensitive. An explicit GPU that cannot be honoured is an error, never a silent CPU run. Env: DOTLLM_VULKAN_DEVICE_INDEX picks the Vulkan device behind plain 'vulkan'. |
| `--dflash` `<text>` | unset |  | Path of a DFlash / DFlash2 drafter GGUF (arch 'dflash', e.g. z-lab/Qwen3.8-27B-DFlash2-GGUF). It takes the MTP slot: drafts are verified by the MTP speculative decoder, greedy output is identical to plain decode. qwen35 targets on CPU, CUDA or Vulkan (drafter on the CPU). Overrides an embedded MTP head. |
| `--gpu-layers` `<int>` | unset |  | Number of transformer layers to offload to GPU (CUDA). 0 = CPU only. Omit for default (all layers on a GPU device, 0 with --device cpu). Ignored on vulkan (whole model is device-resident). |
| `--grammar` `<text>` | unset |  | GBNF grammar string or file path (prefixed with @) for grammar response format. |
| `--lookup-draft` `<int>` | 8 |  | Maximum drafts per prompt-lookup round. Default 8. |
| `--lookup-ngram` `<int>` | 6 |  | Longest context suffix (tokens) prompt lookup matches; shorter matches down to 2 tokens are used when it does not recur. Default 6. |
| `--max-tokens`, `-n` `<int>` | 512 |  | Maximum number of tokens to generate per response. |
| `--min-p` `<number>` | 0 |  | Min-P sampling threshold. 0 = disabled. |
| `--moe-cache-mib` `<text>` | unset | `DOTLLM_MOE_CACHE_MIB` | VRAM expert-cache size in MiB when experts are offloaded; default 'auto' = the measured free-VRAM budget. Larger values are clamped to the budget. Env fallback: DOTLLM_MOE_CACHE_MIB. |
| `--moe-cache-policy` `<text>` | unset | `DOTLLM_MOE_CACHE_POLICY` | Expert-cache replacement policy: 'slru' (default), 'lru' or 'lfu'. Env fallback: DOTLLM_MOE_CACHE_POLICY. |
| `--moe-calibrate` | off | `DOTLLM_MOE_CALIBRATE` | Re-run the MoE offload cost calibration (PCIe bandwidth, sync latency, CPU time per expert) instead of reading the cached record in %LOCALAPPDATA%/dotllm/moe-calibration.json. Env fallback: DOTLLM_MOE_CALIBRATE=1. |
| `--moe-cold` `<text>` | unset | `DOTLLM_MOE_COLD` | Where offloaded experts that are not in the VRAM cache run: 'auto' (default: per expert, CPU or copy-to-GPU by a cost model calibrated at load), 'copy' (copy every miss into the cache), 'cpu' (compute every uncached expert on the CPU; an expert enters the cache only when its recent use beats the weakest cached one). Env fallback: DOTLLM_MOE_COLD. |
| `--moe-cpu-threads` `<int>` | unset | `DOTLLM_MOE_CPU_THREADS` | Worker threads for MoE experts computed on the CPU (one expert per worker); default physical cores - 1. Env fallback: DOTLLM_MOE_CPU_THREADS. |
| `--moe-offload` `<text>` | unset | `DOTLLM_MOE_OFFLOAD` | Routed-MoE expert placement on CUDA (nemotron_h_moe): 'auto' (default: keep experts in VRAM when the model fits, else offload them to a host store with a VRAM expert cache), 'off' (refuse a model that does not fit), 'on' (always offload). Env fallback: DOTLLM_MOE_OFFLOAD. |
| `--mtp-head` `<text>` | unset |  | Path of a separate MTP head GGUF (qwen4exp ships it as mtp-*.gguf beside the quantisation folders). Default: auto-detected next to the model; a head that does not fit the device is declined with a warning. |
| `--n-cpu-moe` `<int>` | unset | `DOTLLM_N_CPU_MOE` | Keep the routed experts of the first N layers (by layer index, llama.cpp --n-cpu-moe) on the CPU. CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models: computed on the CPU from the mmap'd file (#1187); Vulkan Qwen3MoeHybrid: CPU expert compute for those layers (#370). Env fallback: DOTLLM_N_CPU_MOE. |
| `--no-mtp` | off |  | Disable Multi-Token Prediction (MTP) self-speculative decoding (on by default when the model carries an MTP head; an adaptive gate falls back to plain decode when it loses). |
| `--no-prompt-cache` | off |  | Disable prompt caching (KV state reuse across turns). On by default; cleared by /clear. |
| `--numa-pin` | off |  | Pin workers to NUMA-local cores on multi-socket systems. |
| `--paged` | off |  | Use the paged KV cache (fixed-size blocks allocated on demand, ref-counted for prefix sharing) instead of the contiguous cache sized up-front to prompt + --max-tokens. Opt-in here; 'serve' is paged by default (--no-paged to turn off). Needs a uniform per-layer KV width (Gemma-4 / MiMo-V2 use the contiguous cache). |
| `--pattern` `<text>` | unset |  | Regex pattern for regex response format. Entire output must match. |
| `--pcore-only` | off |  | Pin workers to P-cores only (Intel hybrid architectures). |
| `--prefill-chunk-size`, `--ubatch-size` `<int>` | 0 |  | Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog). 0 = whole prompt in one pass (default). |
| `--prefix-snapshot-disk-budget` `<text>` | unset |  | Snapshot spill budget, such as 32GiB; requires a disk directory. Default 0 disables disk retention. |
| `--prefix-snapshot-disk-dir` `<text>` | unset |  | Directory for recurrent prefix spill files; requires a positive disk budget. |
| `--prefix-snapshot-host-budget` `<text>` | unset |  | Page-locked host snapshot budget, such as 8GiB. Default 0 disables host retention. |
| `--prefix-snapshots` `<int>` | unset |  | Maximum recurrent prefix snapshots (default 1). |
| `--prompt-cache-policy` `<text>` | unset | `DOTLLM_PREFIX_POLICY` | Prefix-cache retention policy: lru (default) or agent (a conversation survives about its own length of other prompt tokens; side requests run without truncating it; recurrent models checkpoint at the last user message and after the reply). Env DOTLLM_PREFIX_POLICY. |
| `--prompt-cache-size` `<int>` | 1 |  | Maximum number of cached sessions kept for prompt caching. Default 1 (one conversation). |
| `--prompt-cache-step` `<int>` | 0 |  | Recurrent-state checkpoint every N prefill tokens, so a prompt that diverges mid-history (a trimmed tool result) resumes from the nearest one. Default 0 (off). |
| `--prompt-cache-step-keep` `<int>` | unset |  | Most --prompt-cache-step checkpoints kept, deepest first (default 8). |
| `--prompt-lookup`, `--lookup` | off |  | Prompt-lookup speculative decoding: drafts are copied from earlier prompt/output text where the last tokens recur, and verified by the model (greedy output is identical to plain decode). Needs no draft model; pays off on code edits and other self-repeating output. Greedy requests only; a draft model or an MTP head takes priority. |
| `--quant`, `-q` `<text>` | unset |  | Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0). |
| `--repeat-last-n` `<int>` | 0 |  | Number of recent tokens for repetition penalty lookback. 0 = full history. |
| `--repeat-penalty` `<number>` | 1 |  | Repetition penalty factor. 1.0 = disabled. |
| `--response-format` `<text>` | `text` |  | Constrain model output format: 'text' (default), 'json_object', 'json_schema', 'regex', or 'grammar'. |
| `--schema` `<text>` | unset |  | JSON Schema string or file path (prefixed with @) for json_schema response format. |
| `--seed` `<int>` | unset |  | Random seed for reproducible sampling. Omit for non-deterministic. |
| `--speculative-k`, `--draft-tokens` `<int>` | 3 |  | Number of draft tokens per speculative step (K). Default 3. |
| `--speculative-model`, `--draft-model` `<text>` | unset |  | Path or HuggingFace repo ID for a draft model. Enables speculative decoding for faster generation. Must share vocabulary with the main model. |
| `--system`, `-s` `<text>` | unset |  | System prompt for the conversation. |
| `--system-file` `<text>` | unset |  | Read the system prompt from a file instead of --system. A single trailing newline is stripped. Mutually exclusive with --system. |
| `--temp`, `-t` `<number>` | 0 |  | Sampling temperature. 0 = greedy (default). |
| `--ternary-adapter` `<text>` | unset |  | Path to a ternary residual adapter (.nsadapter or adapter GGUF); repeatable to compose. CPU and Vulkan models with a residual hook only; others are refused. Omit for base. |
| `--threads` `<int>` | 0 |  | Number of CPU threads for inference. 0 = auto/all cores (default). |
| `--tool-choice` `<text>` | `auto` |  | Tool choice: 'auto' (default), 'none', 'required', or a function name. |
| `--tools` `<text>` | unset |  | Tool definitions: JSON array string or file path (prefixed with @). Each tool: {"name": "...", "description": "...", "parameters": {...}}. |
| `--top-k` `<int>` | 0 |  | Top-K sampling. 0 = disabled. |
| `--top-p` `<number>` | 1 |  | Top-P (nucleus) sampling threshold. |
| `--verbose`, `-v` | off |  | Show debug info: finish reason, raw text before stop-sequence stripping, tool call detection details. |
<!-- END options:chat -->

### dotllm serve

Starts the HTTP server (default `http://localhost:8080`): OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`), Anthropic
(`/v1/messages`) and ollama (`/api/*`) compatible APIs, a built-in web chat UI, `/props`, `/health` and Prometheus `/metrics`. With no `<model>` it starts empty and loads
models on request. Endpoints, request fields, model management, rate limiting, thinking-budget and chunk-cache details: [SERVER.md](SERVER.md), [ANTHROPIC_API.md](ANTHROPIC_API.md), [CHUNK_CACHE.md](CHUNK_CACHE.md).

```text
dotllm serve Qwen/Qwen3-0.6B-GGUF --port 8080
dotllm serve Qwen/Qwen3-0.6B-GGUF --device vulkan --ctx-size 16384 --cache-type-k q8_0 --cache-type-v q8_0
dotllm serve --max-resident-models 2 --keep-alive 600 --allow-model-admin --no-browser   # empty server, several models resident
dotllm serve my-model.gguf --host 0.0.0.0 --no-ui --rate-limit-rpm 60 --rate-limit-concurrency 4
```

Notes on interplay:

- **Residency:** `--max-resident-models` (default 1 = hot-swap) and `--resident-memory-budget` bound what stays loaded; when a load would exceed either, least-recently-used models are evicted first.
  `--keep-alive` unloads idle models (a profile's keep-alive overrides it).
- **Speculation:** MTP self-speculative decoding is on by default when the GGUF has an embedded head; `--no-mtp` opts out, `--expected-concurrency 5` or more (or a `--speculative-model`) turns it off in favour of
  continuous batching or the draft model. `--prompt-lookup` needs no draft model but applies to greedy requests only.
- **Security:** the inference endpoints are unauthenticated. The default bind is loopback; `--allow-model-admin` and `--allow-lora-admin` enable administrative endpoints and are off by default;
  `--auto-pull` lets requests trigger downloads (off by default). Rate limiting switches on as soon as any `--rate-limit-*` option is set.
- **Observability:** `/metrics` is on for loopback binds only unless `--metrics`; OpenTelemetry export is configured with `OTEL_EXPORTER_OTLP_ENDPOINT`; per-request timing lines can be turned off with `DOTLLM_LOG_TIMINGS=0`.

<!-- BEGIN options:serve -->
| Option | Default | Env | Description |
|---|---|---|---|
| `[model]` |  |  | Path to a GGUF file or HuggingFace repo ID. Omit to start without a model (load from UI). |
| `--allow-lora-admin` | off |  | Enable the LoRA admin API: POST /v1/lora/load and DELETE /v1/lora/{name}. Off by default. |
| `--allow-model-admin` | off |  | Enable the model-administration API (#454): POST /v1/models/unload, /pull, /enable, /disable and PUT /v1/settings. Off by default. |
| `--auto-pull` | off |  | Download a missing Hugging Face model when a request names one (e.g. model "owner/repo:Q4_K_M"), like 'ollama run' does. Off by default: remote clients should not be able to start multi-gigabyte downloads. |
| `--cache-type-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache KEY storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-k where that exists (give only one; conflicting values are an error). Unset: the backend default - F32 in the contiguous CPU cache (run/chat), F16 in GPU caches, Q8_0 blocks in the paged CPU cache (pass f16 for BOTH K and V to keep F16 there). On the CUDA hybrid-dense model (Bonsai-2, Qwen3.5/3.6) f32 means 'not given' (DOTLLM_HYBRID_KV applies) and quantized KV needs numKvHeads*headDim % 32 == 0. 'run' and 'bench' warn when the model/backend cannot honour a format. Sizing: docs/CLI.md#kv-cache-sizing. |
| `--cache-type-v` `<text>` | unset |  | KV-cache VALUE storage: f32, f16, q8_0 or q4_0 (bytes/element 4, 2, 1.0625, 0.5625). Synonym of --kv-quant-v where that exists. K and V may differ (e.g. -k q8_0 -v q4_0). See --cache-type-k. |
| `--chunk-cache` | off |  | Honour chunk_cache annotations on /v1/chat/completions and /v1/messages (chunk-level K/V reuse convention, docs/CHUNK_CACHE.md): policies exact (default) and shadow, whose output is identical to the unannotated request; responses gain a chunk_cache report. Off by default: annotations are ignored like any unknown field and responses are unchanged. |
| `--chunk-cache-allow-short-keys` | off |  | With --chunk-cache-api-keys: accept keys shorter than 32 characters (each logged as a warning). Off by default: such a key refuses startup, because namespaces derived from keys appear in the store's file names and can be checked offline. |
| `--chunk-cache-api-keys` `<text>` | unset |  | With --chunk-cache: a file of API keys (one per line: '<key>' optionally followed by 'admin' and/or 'ns=<namespace>', keys of 32+ characters, e.g. 32 random bytes base64url; '#' comments; read once at startup) that may use /v1/chunk-cache/* (X-API-Key or Authorization: Bearer); each key gets its own store namespace. Without it only loopback callers may. Security-sensitive: see docs/CHUNK_CACHE.md. Inference endpoints stay unauthenticated either way. |
| `--chunk-cache-max-chunks` `<int>` | unset |  | Most chunk annotations read per request (default 4096, at least 1); the rest are dropped with chunk_too_many without being read. |
| `--chunk-store` `<text>` | unset |  | With --chunk-cache: the chunk store (e.g. built by chunkcache build) the server looks annotated chunks up in, to count hits and plan shadow reuse. Read-only: the server never writes or creates it; a missing directory is reported as chunk_store_unavailable. |
| `--cpu-moe` | off | `DOTLLM_CPU_MOE` | Keep every MoE layer's routed experts on the CPU (llama.cpp --cpu-moe). CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models (e.g. Flint): computed on the CPU from the mmap'd file; Vulkan Qwen3MoeHybrid (bench): as --n-cpu-moe with every layer. Env fallback: DOTLLM_CPU_MOE=1. |
| `--ctx-size`, `-c` `<int>` | unset |  | Served context window in tokens (llama.cpp -c). Default: the CUDA hybrid model (Bonsai, Qwen3.5/3.6 dense) fits it to free VRAM at load and reports it in /props as max_sequence_length; other models keep the checkpoint's window. A value overrides the fit (capped at the checkpoint's window); past the fit, requests are refused with an out-of-memory error instead of spilling into system RAM. |
| `--decision-orderings` `<int>` | 1 |  | Option orderings averaged per noul/choice question on POST /v1/systemone: 1 (default) or 2 (forward + reversed, logits averaged; reversal flipped 2.2% of Tev1-4B choice answers and averaging lifted choice accuracy 98.2 -> 98.9%, at twice the forward passes). |
| `--decision-temperature` `<number>` | 0 |  | Logit temperature for POST /v1/systemone probabilities (softmax(logits / T)). 0 or 1 (default) = the model's raw restricted softmax. Tev1-4B is slightly under-confident: T=0.58 cut held-out NLL 0.0546 -> 0.0428 on a 480-item labelled set (scripts/decision-calibration.py); fit your own on your task before relying on it. |
| `--decode-threads` `<int>` | 0 |  | CPU threads for the decode phase. 0 = auto (capped at the memory-channel count, since decode is bandwidth-bound). |
| `--device`, `-d` `<text>` | `auto` | `DOTLLM_VULKAN_DEVICE_INDEX` | Compute device: 'auto' (default: CUDA if the model fits, else Vulkan, else CPU - with a warning if it ends on the CPU), 'cpu', 'vulkan' (the default Vulkan GPU), 'vulkan:N' (Vulkan physical device N), 'gpu'/'cuda' (CUDA device 0), 'gpu:N'/'cuda:N' (CUDA ordinal N). Case-insensitive. An explicit GPU that cannot be honoured is an error, never a silent CPU run. Env: DOTLLM_VULKAN_DEVICE_INDEX picks the Vulkan device behind plain 'vulkan'. A model profile's device overrides it for that model. |
| `--dflash` `<text>` | unset |  | Path of a DFlash / DFlash2 drafter GGUF (arch 'dflash'). It takes the MTP slot; qwen35 targets on CPU, CUDA or Vulkan (drafter on the CPU). |
| `--expected-concurrency` `<int>` | 0 | `DOTLLM_VK_SCHEDULER` | Expected number of concurrently decoding requests. At 5 or more, Vulkan hybrid models serve through the continuous-batch scheduler (measured +73% aggregate decode throughput at 8 concurrent on Tev1-4B; parity at 4 or fewer). At 5 or more, default-on MTP is dropped for the model. 0 (default) keeps the serial per-request path. DOTLLM_VK_SCHEDULER=0\|1 overrides. |
| `--gpu-layers` `<int>` | unset |  | CUDA partial offload: number of transformer layers kept on the GPU, the rest run on the CPU. Omit for all layers on a GPU device (0 with --device cpu). Ignored on vulkan, where the whole model is device-resident. |
| `--host` `<text>` | `localhost` |  | Address to bind (default localhost = loopback only). Use 0.0.0.0 to serve the network - the inference endpoints are unauthenticated, so pair that with a firewall or reverse proxy. /metrics is on by default for loopback binds only. |
| `--keep-alive` `<number>` | 300 |  | Idle-unload timeout in seconds (ollama parity). Default 300 (5 min); 0 = unload after every request; negative = never unload. A model profile's keep_alive overrides it for that model. |
| `--kv-quant-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache key storage: f16, q8_0, q4_0. Synonym of --cache-type-k. For the CUDA hybrid-dense model (Bonsai-2 / qwen35) overrides DOTLLM_HYBRID_KV; quantized KV needs numKvHeads*headDim % 32 == 0. |
| `--kv-quant-v` `<text>` | unset |  | KV-cache value storage: f16, q8_0, q4_0. Synonym of --cache-type-v. See --kv-quant-k. |
| `--lookup-draft` `<int>` | 8 |  | Maximum drafts per prompt-lookup round. Default 8. |
| `--lookup-ngram` `<int>` | 6 |  | Longest context suffix (tokens) prompt lookup matches; shorter matches down to 2 tokens are used when it does not recur. Default 6. |
| `--max-messages` `<int>` | unset | `DOTLLM_MAX_MESSAGES` | Maximum messages accepted per chat request on /v1/chat/completions and /v1/messages. Default 8192; 0 = unlimited. Falls back to the DOTLLM_MAX_MESSAGES environment variable; the flag wins. The prompt-length-vs-context check still applies. |
| `--max-resident-models` `<int>` | 1 |  | Maximum number of models kept loaded at once, counting the active one. Default 1 = classic single-model hot-swap (loading another model unloads the current one). Above 1, earlier models stay resident (instant switching) and, when a load would exceed the count or --resident-memory-budget, the least recently used ones are evicted first. |
| `--metrics` | off |  | Serve Prometheus metrics at GET /metrics on any bind. Default: on for loopback binds only. |
| `--moe-cache-mib` `<text>` | unset | `DOTLLM_MOE_CACHE_MIB` | VRAM expert-cache size in MiB when experts are offloaded; default 'auto' = the measured free-VRAM budget. Larger values are clamped to the budget. Env fallback: DOTLLM_MOE_CACHE_MIB. |
| `--moe-cache-policy` `<text>` | unset | `DOTLLM_MOE_CACHE_POLICY` | Expert-cache replacement policy: 'slru' (default), 'lru' or 'lfu'. Env fallback: DOTLLM_MOE_CACHE_POLICY. |
| `--moe-calibrate` | off | `DOTLLM_MOE_CALIBRATE` | Re-run the MoE offload cost calibration (PCIe bandwidth, sync latency, CPU time per expert) instead of reading the cached record in %LOCALAPPDATA%/dotllm/moe-calibration.json. Env fallback: DOTLLM_MOE_CALIBRATE=1. |
| `--moe-cold` `<text>` | unset | `DOTLLM_MOE_COLD` | Where offloaded experts that are not in the VRAM cache run: 'auto' (default: per expert, CPU or copy-to-GPU by a cost model calibrated at load), 'copy' (copy every miss into the cache), 'cpu' (compute every uncached expert on the CPU; an expert enters the cache only when its recent use beats the weakest cached one). Env fallback: DOTLLM_MOE_COLD. |
| `--moe-cpu-threads` `<int>` | unset | `DOTLLM_MOE_CPU_THREADS` | Worker threads for MoE experts computed on the CPU (one expert per worker); default physical cores - 1. Env fallback: DOTLLM_MOE_CPU_THREADS. |
| `--moe-offload` `<text>` | unset | `DOTLLM_MOE_OFFLOAD` | Routed-MoE expert placement on CUDA (nemotron_h_moe): 'auto' (default: keep experts in VRAM when the model fits, else offload them to a host store with a VRAM expert cache), 'off' (refuse a model that does not fit), 'on' (always offload). Env fallback: DOTLLM_MOE_OFFLOAD. |
| `--mtp` | off |  | Multi-Token Prediction (MTP) self-speculative decoding is ON by default for serve when the loaded GGUF carries an embedded MTP head (nextn.* tensors) and no --speculative-model is given and --expected-concurrency is below 5 (adaptive: it backs off when drafts stop paying). --mtp is accepted for back-compat and only changes the startup log; it does not override those rules. Use --no-mtp to opt out. Tradeoff: while MTP is active the continuous-batch scheduler is off for this model. |
| `--mtp-head` `<text>` | unset |  | Path of a separate MTP head GGUF (qwen4exp ships it as mtp-*.gguf beside the quantisation folders). Default: auto-detected in the model's directory and the one above it (a head that does not fit the device is declined with a startup message and serving continues without it). |
| `--n-cpu-moe` `<int>` | unset | `DOTLLM_N_CPU_MOE` | Keep the routed experts of the first N layers (by layer index, llama.cpp --n-cpu-moe) on the CPU. CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models: computed on the CPU from the mmap'd file (#1187); Vulkan Qwen3MoeHybrid: CPU expert compute for those layers (#370). Env fallback: DOTLLM_N_CPU_MOE. |
| `--no-browser` | off |  | Don't auto-open the browser. |
| `--no-metrics` | off |  | Do not serve GET /metrics. |
| `--no-mtp` | off |  | Disable the default-on MTP self-speculative decoding for models with an embedded MTP head (keeps the continuous-batch scheduler available for concurrent clients). |
| `--no-paged` | off |  | Disable the paged KV cache and use contiguous per-request caches instead. Default (paged): fixed-size blocks allocated on demand with prefix sharing, so memory tracks live tokens. Models without a uniform per-layer KV width always use the contiguous cache. |
| `--no-prompt-cache` | off |  | Disable prompt caching (KV state reuse across turns/requests). On by default. |
| `--no-ui` | off |  | Disable the built-in web chat UI. Only serve API endpoints. |
| `--no-warmup` | off |  | Disable warm-up passes at startup. When enabled (default), dummy inference runs trigger JIT compilation. |
| `--overlay-skip-base-hash` | off |  | Do not verify the overlay's dotllm.overlay.base_sha256 against the base model file (hashing a multi-GB base takes seconds; the digest is cached). |
| `--port`, `-p` `<int>` | 8080 |  | TCP port to listen on (default 8080). 'dotllm ps' / 'stop' use the same default. |
| `--prefill-chunk-size`, `--ubatch-size` `<int>` | 0 |  | Maximum prompt tokens per prefill forward pass (llama.cpp -ub analog). 0 = whole prompt in one pass (default). With the continuous-batch scheduler this caps prefill tokens admitted per step instead. |
| `--prefix-snapshot-disk-budget` `<text>` | unset |  | Snapshot spill budget, such as 32GiB; requires a disk directory. Default 0 disables disk retention. |
| `--prefix-snapshot-disk-dir` `<text>` | unset |  | Directory for recurrent prefix spill files; requires a positive disk budget. |
| `--prefix-snapshot-host-budget` `<text>` | unset |  | Page-locked host snapshot budget, such as 8GiB. Default 0 disables host retention. |
| `--prefix-snapshots` `<int>` | unset |  | Maximum recurrent prefix snapshots (default 1). Use with --prefix-snapshot-host-budget for alternating sessions. |
| `--prompt-cache-policy` `<text>` | unset | `DOTLLM_PREFIX_POLICY` | Prefix-cache retention policy: lru (default) or agent (a conversation survives about its own length of other prompt tokens, side requests run without truncating it, recurrent models checkpoint at the last user message and after the reply). Env DOTLLM_PREFIX_POLICY. |
| `--prompt-cache-size` `<int>` | 4 |  | Maximum number of cached sessions (prompt prefixes) kept for reuse. Default 4. |
| `--prompt-cache-step` `<int>` | 0 |  | Recurrent-state checkpoint every N prefill tokens, so a prompt that diverges mid-history (a trimmed tool result) resumes from the nearest one. Default 0 (off). |
| `--prompt-cache-step-keep` `<int>` | unset |  | Most --prompt-cache-step checkpoints kept, deepest first (default 8). |
| `--prompt-lookup`, `--lookup` | off |  | Prompt-lookup speculative decoding: drafts are copied from earlier prompt/output text where the last tokens recur, and verified by the model (greedy output is identical to plain decode). Needs no draft model; pays off on code edits and other self-repeating output. Greedy requests only; a draft model or an MTP head takes priority. |
| `--quant`, `-q` `<text>` | unset |  | Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0). |
| `--rate-limit-concurrency` `<int>` | 0 |  | Rate limit: max concurrent in-flight requests per API key. 0 = unlimited. |
| `--rate-limit-rpm` `<int>` | 0 |  | Rate limit: max requests per minute per API key. 0 = unlimited. Setting any --rate-limit-* option enables rate limiting (off by default). |
| `--rate-limit-tpm` `<int>` | 0 |  | Rate limit: max tokens per minute per API key (prompt + completion). 0 = unlimited. |
| `--reasoning-answer-reserve` `<int>` | 8192 |  | With a thinking budget, max_tokens is raised to at least budget + this so the answer is not truncated (default 8192; 0 = llama.cpp: the budget counts inside max_tokens). The context window still caps it. |
| `--reasoning-budget-message` `<text>` | unset |  | Text forced before the end-of-thinking tag when the thinking budget runs out (default none). |
| `--reasoning-budget`, `--thinking-budget` `<int>` | -1 |  | Thinking token budget (llama.cpp semantics): -1 unlimited (default), 0 end thinking immediately, N at most N thinking tokens, then --reasoning-budget-message and the close tag are forced and the answer continues. Requests override with thinking_budget_tokens (Anthropic: thinking.budget_tokens). |
| `--reasoning-format` `<text>` | `auto` |  | How thinking output is separated from the answer: auto (default; split into reasoning_content + content when the chat template opens a <think> block or the model emits one), deepseek (as auto, and <think> blocks are recognised anywhere in the output), none (raw output in content, </think> included). Requests may override with reasoning_format. |
| `--repeat-guard` `<int>` | 256 |  | End a request (finish_reason=length) after this many identical tokens in a row; catches a sampler pinned on one token under memory pressure, which otherwise looks like a hang. Default 256; 0 disables. |
| `--resident-memory-budget` `<int>` | 0 |  | Total byte budget (an integer number of BYTES, e.g. 17179869184 for 16 GiB) across all resident models. Default 0 = unlimited, so only --max-resident-models bounds residency. |
| `--rope-freq-base` `<number>` | unset |  | RoPE base frequency (theta) override. Overrides the GGUF-derived value. |
| `--rope-scale` `<number>` | unset |  | RoPE scaling factor override (linear/YaRN/NTK). Overrides the GGUF-derived value. |
| `--rope-scaling` `<text>` | unset |  | RoPE scaling override: 'none', 'linear', 'yarn', 'ntk', 'dynamic'. Overrides the GGUF-derived value. |
| `--speculative-k`, `--draft-tokens` `<int>` | 3 |  | Number of draft tokens per speculative step (K). Default 3. Also used as K for MTP. |
| `--speculative-model`, `--draft-model` `<text>` | unset |  | Path or HuggingFace repo ID for a draft model. Enables speculative decoding for faster generation. Must share vocabulary with the main model. |
| `--tensor-overlay` `<text>` | unset |  | Overlay GGUF replacing named FFN tensors (blk.L.ffn_{gate,up,down}.weight, or blk.L.ffn_{gate,up,down}.E.weight for expert E), e.g. with TERN_B64_F16 ternary experts. Unknown names, shape mismatches and a base-hash mismatch are errors. CPU backend, plain transformer / MoE-GGUF models only. |
| `--ternary-adapter` `<text>` | unset |  | Ternary residual adapter (.nsadapter or adapter GGUF, docs/TERNARY_ADAPTER.md) applied to every request; CPU and Vulkan models with a residual hook (see docs/TERNARY_ADAPTER.md), any other model fails the load; repeatable to compose. |
| `--threads` `<int>` | 0 |  | CPU threads for inference. 0 = auto (all cores); 1 = single-threaded. |
| `--warmup-iterations` `<int>` | 3 |  | Number of warm-up iterations (default: 3). More iterations improve JIT optimization via Dynamic PGO. |
| `--yarn-attn-factor` `<number>` | unset |  | YaRN attention factor override. |
| `--yarn-beta-fast` `<number>` | unset |  | YaRN beta-fast parameter override. |
| `--yarn-beta-slow` `<number>` | unset |  | YaRN beta-slow parameter override. |
| `--yarn-orig-ctx` `<int>` | unset |  | YaRN original context length override. |
<!-- END options:serve -->

### dotllm ps

Lists the models resident in a **running server** (queried over HTTP): which is active, its device, size, idle time and the countdown to auto-unload. The server URL is `--url`, else `DOTLLM_URL`, else
`http://localhost:8080`. For models on disk use [`dotllm list`](#dotllm-list).

```text
dotllm ps
dotllm ps --url http://gpu-box:8080
```

<!-- BEGIN options:ps -->
| Option | Default | Env | Description |
|---|---|---|---|
| `--url` `<text>` | unset | `DOTLLM_URL` | Base URL of the running dotllm server. Default: the DOTLLM_URL environment variable, else http://localhost:8080 (the default of dotllm serve --port). |
<!-- END options:ps -->

### dotllm stop

Unloads one model (`dotllm stop llama3.2:3b`, names as shown by `ps`), every model (`--all`), or asks the whole server to shut down gracefully (`--server`). The server must have been started with `--allow-model-admin`.

<!-- BEGIN options:stop -->
| Option | Default | Env | Description |
|---|---|---|---|
| `[model]` |  |  | Model to unload (as shown by 'dotllm ps'). Omit with --all to unload everything. |
| `--all` | off |  | Unload every resident model. |
| `--server` | off |  | Stop the whole server process gracefully (POST /v1/admin/shutdown). |
| `--url` `<text>` | unset | `DOTLLM_URL` | Base URL of the running dotllm server. Default: the DOTLLM_URL environment variable, else http://localhost:8080 (the default of dotllm serve --port). |
<!-- END options:stop -->

### dotllm bench

Synthetic prompt of `-p` tokens (tiled from `--seed-prompt`), then `-n` greedy decode steps, repeated `-r` times after one discarded warm-up. Reports prefill and decode tokens per second with
mean and deviation; `--depth` / `--depth-list` measure decode at deeper context without reloading the model. `--json` prints machine output and `DOTLLM_BENCH_*` variables (below) add logging for A/B work.
Benchmarks on the same box are affected by clock ramp and contention; compare within one session. Methodology and published numbers: [BENCHMARKS.md](BENCHMARKS.md), [COMPARISONS.md](COMPARISONS.md).

```text
dotllm bench QuantFactory/SmolLM-135M-GGUF --device cpu -p 512 -n 128 -r 5
dotllm bench QuantFactory/SmolLM-135M-GGUF --device vulkan --depth-list 512,2048,8192 --json
```

<!-- BEGIN options:bench -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF). |
| `--cpu-moe` | off | `DOTLLM_CPU_MOE` | Keep every MoE layer's routed experts on the CPU (llama.cpp --cpu-moe). CUDA nemotron_h_moe: never cached, always computed on the CPU; other CUDA MoE models (e.g. Flint): computed on the CPU from the mmap'd file; Vulkan Qwen3MoeHybrid (bench): as --n-cpu-moe with every layer. Env fallback: DOTLLM_CPU_MOE=1. |
| `--ctx-size`, `--ctx` `<int>` | unset | `DOTLLM_MOE_CTX` | Context window in tokens (llama.cpp -c). For nemotron_h_moe it caps the model's context and is what the MoE expert-offload VRAM budget reserves KV cache for (default: the model's maximum context, e.g. 262144 for Nemotron 3.5 Bolt). Lower it to leave more VRAM for the expert cache. Env fallback: DOTLLM_MOE_CTX. |
| `--decode-threads` `<int>` | 0 |  | Number of threads for decode. 0 = auto (caps at memory channel count). |
| `--depth` `<int>` | 0 |  | Extra synthetic context tokens fed (untimed) after prefill, before the timed decode. Decode runs at context depth p + depth. |
| `--depth-list` `<text>` | unset |  | Comma-separated context depths for multi-point sweep (e.g., 512,2048,8192). Measures decode at each checkpoint in one process without reloading the model. Mutually exclusive with --depth. |
| `--device`, `-d` `<text>` | `auto` | `DOTLLM_VULKAN_DEVICE_INDEX` | Compute device: 'auto' (default: CUDA if the model fits, else Vulkan, else CPU - with a warning if it ends on the CPU), 'cpu', 'vulkan' (the default Vulkan GPU), 'vulkan:N' (Vulkan physical device N), 'gpu'/'cuda' (CUDA device 0), 'gpu:N'/'cuda:N' (CUDA ordinal N). Case-insensitive. An explicit GPU that cannot be honoured is an error, never a silent CPU run. Env: DOTLLM_VULKAN_DEVICE_INDEX picks the Vulkan device behind plain 'vulkan'. |
| `--gen-tokens`, `-n` `<int>` | 128 |  | Greedy decode steps to time per repetition. |
| `--json` | off |  | Output result as a single JSON object (suppresses all formatted output). |
| `--kv-quant-k` `<text>` | unset | `DOTLLM_HYBRID_KV` | KV-cache key storage: f16, q8_0, q4_0. Synonym of --cache-type-k. For the CUDA hybrid-dense model (Bonsai-2 / qwen35) overrides DOTLLM_HYBRID_KV; quantized KV needs numKvHeads*headDim % 32 == 0. |
| `--kv-quant-v` `<text>` | unset |  | KV-cache value storage: f16, q8_0, q4_0. Synonym of --cache-type-v. See --kv-quant-k. |
| `--moe-cache-mib` `<text>` | unset | `DOTLLM_MOE_CACHE_MIB` | VRAM expert-cache size in MiB when experts are offloaded; default 'auto' = the measured free-VRAM budget. Larger values are clamped to the budget. Env fallback: DOTLLM_MOE_CACHE_MIB. |
| `--moe-cache-policy` `<text>` | unset | `DOTLLM_MOE_CACHE_POLICY` | Expert-cache replacement policy: 'slru' (default), 'lru' or 'lfu'. Env fallback: DOTLLM_MOE_CACHE_POLICY. |
| `--moe-calibrate` | off | `DOTLLM_MOE_CALIBRATE` | Re-run the MoE offload cost calibration (PCIe bandwidth, sync latency, CPU time per expert) instead of reading the cached record in %LOCALAPPDATA%/dotllm/moe-calibration.json. Env fallback: DOTLLM_MOE_CALIBRATE=1. |
| `--moe-cold` `<text>` | unset | `DOTLLM_MOE_COLD` | Where offloaded experts that are not in the VRAM cache run: 'auto' (default: per expert, CPU or copy-to-GPU by a cost model calibrated at load), 'copy' (copy every miss into the cache), 'cpu' (compute every uncached expert on the CPU; an expert enters the cache only when its recent use beats the weakest cached one). Env fallback: DOTLLM_MOE_COLD. |
| `--moe-cpu-threads` `<int>` | unset | `DOTLLM_MOE_CPU_THREADS` | Worker threads for MoE experts computed on the CPU (one expert per worker); default physical cores - 1. Env fallback: DOTLLM_MOE_CPU_THREADS. |
| `--moe-offload` `<text>` | unset | `DOTLLM_MOE_OFFLOAD` | Routed-MoE expert placement on CUDA (nemotron_h_moe): 'auto' (default: keep experts in VRAM when the model fits, else offload them to a host store with a VRAM expert cache), 'off' (refuse a model that does not fit), 'on' (always offload). Env fallback: DOTLLM_MOE_OFFLOAD. |
| `--n-cpu-moe` `<int>` | unset | `DOTLLM_N_CPU_MOE` | MoE CPU/GPU expert offload (llama.cpp --n-cpu-moe equivalent): put the routed experts of the first N layers on the CPU, trading throughput for device memory. Vulkan Qwen3MoeHybrid (#370): CPU expert compute for those layers. CUDA nemotron_h_moe (#1071): those layers' experts are never cached and always computed on the CPU. Unset = none (env fallback DOTLLM_N_CPU_MOE). No-op on the cpu backend. |
| `--prompt-tokens`, `-p` `<int>` | 512 |  | Synthetic prompt length in tokens (tiled from the seed prompt). |
| `--quant`, `-q` `<text>` | unset |  | Quantization filter when multiple GGUF files exist (e.g., Q4_K_M, Q8_0). |
| `--reps`, `-r` `<int>` | 5 |  | Measured repetitions (one extra warm-up rep runs first and is discarded). |
| `--seed-prompt` `<text>` | `The quick brown fox jumps over the lazy dog. ` |  | Seed text tiled to synthesize the prompt. |
| `--threads` `<int>` | 0 |  | Number of CPU threads for inference. 0 = auto/all cores (default). |
| `--warmup` `<int>` | -1 |  | Number of warm-up repetitions to discard per checkpoint (default: 1). Use 0 to skip warm-up entirely. |
<!-- END options:bench -->

### dotllm perplexity

Teacher-forced perplexity over a UTF-8 corpus, comparable with llama.cpp's `perplexity` tool (same windowing: `--context`, `--stride`, unscored prefix; CRLF caveat: `--normalize-line-endings`).
`--docs-file` switches to per-document evaluation of pre-tokenized documents (NLL per document, optional KL against reference log-probs, MoE routing dump). `--kv-cache` measures the quality cost of a
quantized KV format; `--cycle` / `--first-layer` evaluate models larger than the device by sliding a GPU window across the layers. Full method and the corpus fixtures: [PERPLEXITY.md](PERPLEXITY.md).

```text
dotllm perplexity QuantFactory/SmolLM-135M-GGUF --corpus wiki.test.raw --context 512 --stride 256
dotllm perplexity QuantFactory/SmolLM-135M-GGUF --corpus wiki.test.raw --kv-cache q8_0 --max-tokens 8192      # KV-quantization cost vs --kv-cache none
dotllm perplexity QuantFactory/SmolLM-135M-GGUF --docs-file docs.jsonl --docs-out nll.jsonl                     # per-document NLL JSONL
```

<!-- BEGIN options:perplexity -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | Path to a GGUF file or HuggingFace repo ID (e.g., QuantFactory/SmolLM-135M-GGUF). |
| `--bos` | off |  | Prepend BOS to the stream and substitute it at each window start. Defaults to the model's own setting, derived from the vocab exactly as llama.cpp derives it; pass --bos true/false only to override that. |
| `--context`, `-c` `<int>` | 512 |  | Context window in tokens. Clamped to the model's maximum sequence length. |
| `--corpus`, `-f` `<text>` | unset |  | Path to a UTF-8 text corpus (e.g. wiki.test.raw). |
| `--cycle` | off |  | Layer cycling: slide a --gpu-layers-wide GPU window across the whole trunk within one corpus pass, checkpointing boundary activations, so every layer is GPU-executed even when the model does not fit on the device. |
| `--device`, `-d` `<text>` | `cpu` |  | Compute device: 'cpu' (default here, so perplexity numbers stay comparable across machines), 'auto', 'vulkan', 'vulkan:N', 'gpu'/'cuda', 'gpu:N'/'cuda:N'. An explicit GPU that cannot be honoured is an error, never a silent CPU run. |
| `--docs-file` `<text>` | unset |  | Per-document eval (#1706): a JSONL file with one JSON list of token ids per line. Each document is scored teacher-forced as an independent sequence (the first token is context only and is not scored) with the model's full context limit; --corpus, --mode and the window options are ignored. Writes per-document NLL JSONL to --docs-out (default stdout; all other output goes to stderr). See docs/PERPLEXITY.md. |
| `--docs-out` `<text>` | unset |  | Where --docs-file writes its per-document JSONL (default: stdout). |
| `--dump-routing` `<text>` | unset |  | --docs-file only, MoE models on the CPU transformer: write the router's per-document, per-layer, per-position top-k expert ids and gate weights to this JSONL path. A dense model is an error. |
| `--dump-tokens` `<text>` | unset |  | Write the tokenized corpus ids to this path, whitespace-separated, then continue. Diagnostic. |
| `--first-layer` `<int>` | 0 |  | First global layer index of the GPU window, so the offloaded block is an arbitrary contiguous range rather than a prefix. Requires --gpu-layers; incompatible with --cycle. |
| `--gpu-layers` `<int>` | unset |  | Number of transformer layers to offload to GPU. 0 = CPU only. Omit for default (0 with --device cpu, all with a GPU device). |
| `--kv-cache` `<text>` | `none` |  | KV-cache storage used while scoring: 'none' (default: cacheless forward pass, exact K/V baseline), 'f32' (FP32 cache), 'q8_0' or 'q4_0' (quantized cache; bytes/element 1.0625 / 0.5625). With a cache the forward pass reads K/V back from it, so perplexity measures the quality cost of that KV storage. Compare against the 'none' run on the same corpus. |
| `--kv-window` `<int>` | 0 |  | Full-precision window for a quantized --kv-cache: the most recent N tokens stay in FP32 and older ones are quantized on eviction. 0 (default) = quantize every token immediately. Only applies when --kv-cache is q8_0 or q4_0. |
| `--max-doc-tokens` `<int>` | 0 |  | --docs-file only: truncate every document's sequence (BOS included) to this many tokens. 0 = never truncate; a document longer than the context limit is then an error. |
| `--max-tokens`, `-n` `<int>` | 0 |  | Cap on corpus tokens consumed. 0 = unbounded. |
| `--mode` `<text>` | `sliding-window` |  | Scoring mode: sliding-window (default, llama.cpp-comparable) or teacher-forced. |
| `--normalize-line-endings` | off |  | Collapse CRLF to LF while reading --corpus, as MSVC text mode does. OFF by default. Use ONLY when the reference figure came from a Windows llama.cpp build reading the same CRLF file; llama.cpp on Linux keeps the CRs, so normalizing there creates a mismatch. Preferred fix is an LF corpus (see docs/PERPLEXITY.md, issue #506). |
| `--overlay-skip-base-hash` | off |  | Do not verify the overlay's dotllm.overlay.base_sha256 against the base model file (hashing a multi-GB base takes seconds; the digest is cached). |
| `--per-window` | off |  | Print each window's perplexity. Use to localize a disagreement with another implementation to specific corpus content. |
| `--quant` `<text>` | unset |  | Quantization to select when resolving a HuggingFace repo ID. |
| `--ref-logprobs` `<text>` | unset |  | --docs-file only: reference model log-probs JSONL (one object per document, same order; format 'full' or 'topk', see docs/PERPLEXITY.md). Adds kl_sum / kl_mean = KL(reference \|\| model) per document. |
| `--rope-scale` `<number>` | unset |  | RoPE scaling factor override (linear/YaRN/NTK). |
| `--rope-scaling` `<text>` | unset |  | RoPE scaling override: 'none', 'linear', 'yarn', 'ntk', 'dynamic' (as on run/serve; issue #1101 measures YaRN's short-context cost). |
| `--stride` `<int>` | 0 |  | Tokens advanced between window starts. 0 selects the context length (non-overlapping chunks, llama.cpp's default). |
| `--tensor-overlay` `<text>` | unset |  | Overlay GGUF replacing named FFN tensors (blk.L.ffn_{gate,up,down}.weight, or blk.L.ffn_{gate,up,down}.E.weight for expert E), e.g. with TERN_B64_F16 ternary experts. Unknown names, shape mismatches and a base-hash mismatch are errors. CPU backend, plain transformer / MoE-GGUF models only. |
| `--threads` `<int>` | 0 |  | Compute threads. 0 = auto. |
| `--tokens-file` `<text>` | unset |  | Read pre-tokenized whitespace-separated ids instead of tokenizing --corpus. Isolates scoring from tokenization when comparing against another implementation. |
| `--unscored-prefix` `<int>` | -1 |  | Leading tokens of each window used as context only. -1 selects context/2 (llama.cpp's default). |
| `--yarn-attn-factor` `<number>` | unset |  | YaRN attention factor override. |
| `--yarn-orig-ctx` `<int>` | unset |  | YaRN original context length override. |
<!-- END options:perplexity -->

### dotllm embed

One JSON object per input (JSON Lines on stdout; diagnostics on stderr). With no text arguments it reads one input per line from stdin. Vectors are L2-normalised unless `--no-normalize`;
`--dimensions N` truncates (Matryoshka) and renormalises. HTTP equivalent: `POST /v1/embeddings` ([SERVER.md](SERVER.md), [EMBEDDINGS.md](EMBEDDINGS.md)).

```text
dotllm embed nomic-ai/nomic-embed-text-v1.5-GGUF "hello world" --dimensions 256
printf 'one\ntwo\n' | dotllm embed Qwen/Qwen3-Embedding-0.6B-GGUF --device vulkan
```

<!-- BEGIN options:embed -->
| Option | Default | Env | Description |
|---|---|---|---|
| `<model>` |  |  | Path to a GGUF file or HuggingFace repo ID. |
| `[text]` |  |  | Text(s) to embed. When omitted, one input per line is read from stdin. |
| `--add-eos` | off |  | Force (true) or suppress (false) appending the EOS token. Default: appended for last-token pooling on decoder embedders (Qwen3-Embedding). |
| `--device`, `-d` `<text>` | `cpu` |  | Compute device: 'cpu' (default, so vectors are comparable across machines) or 'vulkan' / 'vulkan:N' (decoder embedders such as Qwen3-Embedding and BERT encoders with F16 weights, #741; nomic-bert, nomic-bert-moe and gemma-embedding run on vulkan too). An explicit GPU that cannot be honoured is an error, never a silent CPU run. |
| `--dimensions` `<int>` | unset |  | Truncate to the leading N components (Matryoshka), then renormalise. |
| `--no-normalize` | off |  | Return un-normalised vectors. |
| `--pooling` `<text>` | unset |  | Pooling override: last, mean or cls. Default: the model's declared pooling. |
| `--quant`, `-q` `<text>` | unset |  | Quantization to select when <model> is a HuggingFace repo. |
| `--threads` `<int>` | 0 |  | CPU threads (0 = auto). |
<!-- END options:embed -->

### dotllm debug (Debug builds only)

Registered only in Debug builds of the CLI (they are absent from release packages): `debug gguf-header <file>`, `debug gguf-metadata <file>`, `debug gguf-tensors <file>`,
`debug gguf-config <file>` (the ModelConfig derived from the metadata), `debug forward-pass <file> --prompt ...` (one forward pass with logit diagnostics),
`debug tokenize <file> --text ...` (tokenize and verify the round trip) and `debug embed-lookup <file> --token-id N`. Run `dotllm debug <name> --help` in a Debug build for their options.

## Environment variables

Command-line options win over environment variables. All names are case-sensitive; unless stated, a switch is on when the value is `1` and off when `0`.
Some variables are read at process start, others per model load - set them before launching `dotllm`.

### User-facing

**Models, storage, networking**

| Variable | Default | Effect |
|---|---|---|
| `DOTLLM_MODELS_DIR` | `~/.dotllm/models` | Directory of the dotLLM model store (`model pull` target, `list` source). |
| `DOTLLM_PROFILES_DIR` | `~/.dotllm/profiles` | Where `model create` stores profiles. |
| `HF_HUB_CACHE`, `HF_HOME` | `~/.cache/huggingface/hub` | Hugging Face hub cache that is searched (read in place) and downloaded into. `HF_HUB_CACHE` wins over `HF_HOME/hub`. |
| `HF_TOKEN` | unset | Token sent with Hugging Face requests (gated or private repos). |
| `OLLAMA_MODELS` | `~/.ollama/models` | Location of the ollama store used for `llama3.2:3b`-style references and `import-ollama`. |
| `DOTLLM_URL` | `http://localhost:8080` | Server base URL for `ps` and `stop` when `--url` is absent. |
| `DOTLLM_NO_PROMPT` | unset | `1` suppresses the first-run offer to install the DirectStorage runtime that `run`, `chat`, `bench` and `serve` make on Windows in an interactive terminal. |
| `DOTLLM_DIRECTSTORAGE_DIR` | package folder | Directory holding the DirectStorage runtime DLLs. |
| `DOTLLM_VULKAN_DIRECT_STORAGE` | off | `1` loads Vulkan weights straight from disk into device memory via DirectStorage (Windows, opt-in). |
| `DOTLLM_HASH_CACHE_DIR` | `~/.dotllm/hash-cache` | Cache of file digests used to verify `--tensor-overlay` base hashes. |
| `DOTLLM_GGUF_MAP_COW` | off | `1` maps GGUF files copy-on-write instead of read-only. |
| `DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER` | off | `1` loads a GGUF whose tokenizer pre-type is unknown with the GPT-2 pipeline instead of refusing (tokenization may differ from the reference). |

**Server behaviour**

| Variable | Default | Effect |
|---|---|---|
| `DOTLLM_MAX_MESSAGES` | 8192 | Messages accepted per chat request when `--max-messages` is absent (0 = unlimited). |
| `DOTLLM_PREFIX_POLICY` | `lru` | Prefix-cache retention policy (`lru` or `agent`) when `--prompt-cache-policy` is absent. |
| `DOTLLM_LOG_TIMINGS` | on | `0` turns off the per-request timing log lines. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Enables OpenTelemetry (OTLP) export of metrics and traces to this endpoint. |
| `DOTLLM_VK_SCHEDULER` | auto | `0`/`1` forces the Vulkan continuous-batch scheduler off/on (default: on at `--expected-concurrency` >= 5 for supported models). |
| `DOTLLM_VK_SCHEDULER_MAX_SEQS` | 8 | Concurrent sequences the Vulkan scheduler admits (each owns a device KV cache and state slot). |
| `DOTLLM_SCHED_RECURRENT_PREFIX` | 4 | Recurrent prefix-snapshot entries kept by the scheduler for hybrid models; `0` disables. |
| `DOTLLM_MTP_ADAPTIVE` | on | `0` turns the adaptive MTP gate off (MTP then engages whenever enabled). |
| `DOTLLM_MTP_CALIBRATE` | on | `0` skips the MTP cold-start calibration rounds. |
| `DOTLLM_GUMBEL_SAMPLING` | off | Position-seeded Gumbel-max sampling for every stochastic pipeline. |
| `DOTLLM_EMBED_BATCH_WINDOW_MS` | 2 | Milliseconds the embedding endpoint waits to coalesce concurrent requests (0 = natural batching only). |

**Devices, memory and KV**

| Variable | Default | Effect |
|---|---|---|
| `DOTLLM_VULKAN_DEVICE_INDEX` | scoring | Index of the Vulkan device behind plain `--device vulkan`. |
| `DOTLLM_VULKAN_DEVICE_VENDOR` | unset | Hex PCI vendor id (e.g. `0x8086`) selecting the first Vulkan device of that vendor (used when no index is set). |
| `DOTLLM_VRAM_BUDGET_MB` | unset | Simulates a smaller GPU: every planner sees at most this many MiB of device memory. A test knob for exercising offload paths. |
| `DOTLLM_VULKAN_DEVICE_LOCAL_LIMIT_MIB` | unset | Caps the Vulkan device-local heap (same effect as above for Vulkan only; takes precedence). |
| `DOTLLM_CUDA_VRAM_RESERVE_MB` | 256 | VRAM headroom kept free by CUDA fit planning. |
| `CUDA_PATH`, `CUDA_HOME` | system | Where the CUDA runtime libraries are looked up. |
| `DOTLLM_HIP_PATH` | system | Where the HIP runtime is looked up (then `HIP_PATH`, `ROCM_PATH`, `~/.dotllm/rocm`). |
| `DOTLLM_CPU_ISA` | auto | Caps the CPU SIMD tier used by kernels (for A/B and bug isolation); see [CPU_FEATURE_GATING.md](CPU_FEATURE_GATING.md). |
| `DOTLLM_HYBRID_KV` | `f16` | KV format of the CUDA hybrid-dense models: `f16`, `q8_0`, `q4_0`, `k8v4`, `k4v8` (flags win). |
| `DOTLLM_VK_KV` | `f16` | KV format of Vulkan caches that honour it (`f16`, `f32`; `q8_0` for Flint). |
| `DOTLLM_CUDA_KV_VMM` | on | `0` disables in-place growth of the CUDA hybrid KV (falls back to copying growth). |
| `DOTLLM_VK_QWEN4EXP_CONTEXT`, `DOTLLM_VK_QWEN4EXP_KV` | auto | Context window / KV storage for qwen4exp on Vulkan (`serve --ctx-size` sets the former). |
| `DOTLLM_VK_ALLOW_OVERCOMMIT` | off | `1` turns the pre-load refusal for a qwen4exp model that does not fit resident into a warning. |
| `DOTLLM_VK_UMA_USABLE_FRACTION` | 0.64 | Fraction of system RAM treated as usable device memory on unified-memory Vulkan GPUs. |
| `DOTLLM_VULKAN_STRICT_DEVICE_LOCAL` | off | `1` makes an exhausted device-local heap an error instead of falling back to host-visible memory. |
| `DOTLLM_VULKAN_DISABLE_HOST_IMPORT` | off | `1` forces every weight tensor to be staged instead of imported zero-copy from the mapped file. |
| `DOTLLM_VULKAN_STAGING_MB` | 64 | Chunk size of the Vulkan upload staging buffer (1..4096). |
| `DOTLLM_VK_MOE_RESIDENT`, `DOTLLM_VK_IQ_RESIDENT` | auto / on | Keep MoE expert banks (IQ banks) resident on the device; `=1` forces on, `=0` off. |
| `DOTLLM_MOE_OFFLOAD`, `DOTLLM_MOE_CACHE_MIB`, `DOTLLM_MOE_CACHE_POLICY`, `DOTLLM_MOE_COLD`, `DOTLLM_MOE_CPU_THREADS`, `DOTLLM_CPU_MOE` (`=1`), `DOTLLM_N_CPU_MOE`, `DOTLLM_MOE_CALIBRATE` (`=1`), `DOTLLM_MOE_CTX` | per option | Environment fallbacks of `--moe-offload`, `--moe-cache-mib`, `--moe-cache-policy`, `--moe-cold`, `--moe-cpu-threads`, `--cpu-moe`, `--n-cpu-moe`, `--moe-calibrate`, `--ctx-size` ([MOE_OFFLOAD.md](MOE_OFFLOAD.md)). |
| `DOTLLM_CUDA_PREFILL_CHUNK`, `DOTLLM_VULKAN_PREFILL_CHUNK` | 1024 | Prefill tile (tokens) of the CUDA / Vulkan hybrid models; `0` disables tiling. `--prefill-chunk-size` overrides. |
| `DOTLLM_DFLASH_DRAFT` | block size - 1 | Caps the drafts per DFlash round. |
| `DOTLLM_MTP_AUTO_ATTACH_MIN_GIB` | 0 | Smallest trunk (GiB) for which a qwen4exp MTP head is auto-attached. |
| `DOTLLM_DEBUG_ERRORS` | off | `1` prints the full exception (type, message, stack) of a failing command instead of only its message. |

### Tuning, A/B and diagnostic variables

Everything below is **not a supported interface**: these switches exist for kernel A/B tests, profiling and bug isolation, may change or vanish between builds, and the defaults
are the tuned ones. Descriptions are the nearest comment in the source (regenerate with `python scripts/gen-cli-env-appendix.py --write`) and may be terse; the named file is the authority. A name
ending `_LEGACY`, `DISABLE_*` or `*_PROFILE` / `*_TRACE` / `*_DEBUG` is typically an escape hatch or a diagnostic. Families: `DOTLLM_VK_*` / `DOTLLM_VULKAN_*` (Vulkan backend),
`DOTLLM_CUDA_*` / `DOTLLM_ATTN_*` / `DOTLLM_DISABLE_*` (CUDA), `DOTLLM_CPU_*` (CPU kernels), `DOTLLM_BENCH_*` (`dotllm bench` logging), `DOTLLM_MTP_*` (speculation), `DOTLLM_PPL_*` (perplexity dumps),
`DOTLLM_TENSOR_DUMP*` (activation dumps), `DOTLLM_Q4E_*` and `DOTLLM_TRACE_*` (model-specific tracing).

<!-- BEGIN env-appendix -->

#### Cli

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_BENCH_ALT_PLANAR` | #1620: DOTLLM_BENCH_ALT_PLANAR=planar-first\|prod-first alternates the planar PQ2_0 decode GEMV per rep inside ONE process (same-session paired A/B, immune to process-start and page-cache noise); each rep reports its mode and p... | `BenchRunner.cs` |
| `DOTLLM_BENCH_COMMIT` | Point-in-time runtime version for the results.csv runtime_version column (convention: dev-96a892bd). Resolution order: the DOTLLM_BENCH_COMMIT environment variable, git in the current directory (branch tag + short SHA), the ass... | `BenchEnvironment.cs` |
| `DOTLLM_BENCH_DUMP_TOKENS` | Timed greedy decode: n single-token forwards. Only the Forward calls are inside the stopwatch windows; argmax runs between them. DOTLLM_BENCH_DUMP_TOKENS= appends one line of generated token IDs per rep - the exact-token parity... | `BenchRunner.cs` |
| `DOTLLM_BENCH_RELEASE_GGUF` | Issue #438 experiment arm - DOTLLM_BENCH_RELEASE_GGUF=1 unmaps the GGUF right after upload. Every device-resident weight was copied into a device buffer by then; nothing in the Vulkan forward path reads a host pointer. If that... | `BenchCommand.cs` |
| `DOTLLM_BENCH_STEP_LOG` | DOTLLM_BENCH_STEP_LOG= (#801): append one line per rep holding that rep's per-step decode ms so a run's tok/s can be windowed (drift tracking) without changing the timed region. | `BenchRunner.cs` |
| `DOTLLM_MEM_PROBE` | True when DOTLLM_MEM_PROBE is set to 1. | `HostResidencyProbe.cs` |

#### Core

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_HYBRID_PREFILL_CROSSOVER` | Reads the DOTLLM_HYBRID_PREFILL_CROSSOVER environment variable. When set to a positive integer, returns that value; otherwise returns DefaultPrefillCrossoverTokens. Used by HybridPrefillDecodeStrategy to let operators tune the... | `BackendCapabilities.cs` |

#### Cpu

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_AMAX_PROBE` | True when DOTLLM_AMAX_PROBE names an output file. | `ActivationRangeProbe.cs` |
| `DOTLLM_CPU_ATTN_BLOCKED` | Environment variable: 0 disables ExecuteBlockedCausal (the head-parallel per-row path runs instead; A/B). | `Attention.cs` |
| `DOTLLM_CPU_BF16_512` | BF16 weight GEMV/GEMM by shift-widening (#1118). BF16 is the top half of an F32, so bf16 &lt;&lt; 16 is the exact F32 value: every tier widens 4/8/16 weights in registers (vpmovzxwd + vpslld) and multiplies into F32 accumulator... | `MatMul.Bf16.cs` |
| `DOTLLM_CPU_BF16_KERNEL` | BF16 weight GEMV/GEMM by shift-widening (#1118). BF16 is the top half of an F32, so bf16 &lt;&lt; 16 is the exact F32 value: every tier widens 4/8/16 weights in registers (vpmovzxwd + vpslld) and multiplies into F32 accumulator... | `MatMul.Bf16.cs` |
| `DOTLLM_CPU_BF16_SCRATCH` | A/B switch between the two bit-identical AVX2/FMA GEMM forms: the register tile (default) and the F16-style four-row F32 scratch (DOTLLM_CPU_BF16_SCRATCH=1). Settable for in-process A/B. | `MatMul.Bf16.cs` |
| `DOTLLM_CPU_IQ1M_PACKED` | IQ1_M routed experts use the packed x Q8_K dot (default). DOTLLM_CPU_IQ1M_PACKED=0 falls back to dequantize-per-row (A/B baseline). | `MoeSwiGluMlp.cs` |
| `DOTLLM_CPU_MOE_DECODE_N1` | Decode (one token) routed-expert fast path. The grouped path runs the k selected experts as k independent single-threaded tasks (Parallel.For, inner GEMMs without a pool): only k cores work, the work per expert is serial gate -... | `MoeSwiGluMlp.DecodeN1.cs` |
| `DOTLLM_CPU_NVFP4_LEGACY_ORDER` | DOTLLM_CPU_NVFP4_LEGACY_ORDER=1 (read once): use the pre-#1587 float summation order (8-lane FMA partials) for every CPU NVFP4 x Q8_0 dot, single- and multi-row, so perplexity figures measured before #1587 reproduce. Default of... | `MatMulNvfp4.cs` |
| `DOTLLM_CPU_NVFP4_ROWS8` | Whether the 8-row NVFP4 kernel can run (AVX2 gather/permutes; VNNI optional). | `MatMulNvfp4.Rows8.cs` |
| `DOTLLM_CPU_PROF` | True when DOTLLM_CPU_PROF=1. | `CpuStageProf.cs` |
| `DOTLLM_CPU_Q4_FLOOR_PROBE` | Diagnostic (DOTLLM_CPU_Q4_FLOOR_PROBE=1, #1584): route packed legacy GEMMs through the single-row AVX2 dot to measure float-order sensitivity. | `MatMulLegacyQuants.cs` |
| `DOTLLM_CPU_Q4_PACKED_PREFILL` | Q4_0 / Q4_1 on AVX2 take the packed path at every column count (#1159). Their eight-row multi-column kernel (ComputeRowsQ4MultiAvx2_8Rows) unpacks and transposes each block once for all columns and is bit-exact per column with... | `MatMulLegacyQuants.cs` |
| `DOTLLM_CPU_Q8_ROWS8` | Eight-row transposed int8 Q8_0 x Q8_0 kernels on AVX-VNNI (#1119): the Q4_0/Q4_1 eight-row pattern of #1159 applied to Q8_0. Why. The production Q8_0 prefill kernel (Q8_0Rows4Tokens4Avx2) spends about six vector ops per row per... | `MatMul.Q8_0Rows8.cs` |
| `DOTLLM_F16_NO_FMA` | Scratch floats needed per worker by GemmF16RowsFma for a given k. | `MatMul.F16Tile.cs` |
| `DOTLLM_FAST_EXP` | Opt-in switch (DOTLLM_FAST_EXP=1) that restores the Schraudolph bit-trick exp on every path in this class. Off by default since #501. The premise "errors in exp get normalized away when dividing by the sum" is only true for the... | `FastMath.cs` |
| `DOTLLM_I2S_TILE` | Runtime gate for the 4x4 register-blocked I2_S W2A8 GEMM inner kernel (issue #232). The vector width stays 256-bit - 512-bit dot widening was measured 1.8-1.9x slower on Zen5 (issues #196/#202) and #416 reached the same conclus... | `MatMul.I2S.cs` |
| `DOTLLM_I2S_W2A8` | In-process override of I2SUseW2A8Enabled (tests only; read per call). A test whose oracle must be the exact float (W2A16) reference pins this to false, so the reference does not silently become the int8-activation tier on hardw... | `MatMul.I2S.cs` |
| `DOTLLM_PLE_CACHE_MB` | (not described in source) | `Qwen4ExpPleRowPrefetcher.cs` |
| `DOTLLM_PLE_PREFETCH` | Reads the DOTLLM_PLE_* environment (or Ambient when set). | `Qwen4ExpPleRowPrefetcher.cs` |
| `DOTLLM_PLE_STATS` | (not described in source) | `Qwen4ExpPleRowPrefetcher.cs` |
| `DOTLLM_PLE_TABLE` | Reads the DOTLLM_PLE_* environment (or Ambient when set). | `Qwen4ExpPleRowPrefetcher.cs` |
| `DOTLLM_PLE_TOUCH_THREADS` | (not described in source) | `Qwen4ExpPleRowPrefetcher.cs` |
| `DOTLLM_PQ2_PLANAR` | True when the planar PQ2_0 kernels are enabled: on by default with AVX2 (#1620), off with DOTLLM_PQ2_PLANAR=0, and off when DOTLLM_PQ2_W2A8=0 asks for the unquantized-activation float tier (planar also quantizes activations, so... | `MatMul.PQ2Planar.cs` |
| `DOTLLM_PQ2_PLANAR_NMAX` | Largest n routed to the blocked planar GEMM when the planar toggle is on (DOTLLM_PQ2_PLANAR_NMAX; 1 = decode GEMV only; default PQ2_0PlanarDefaultMaxN). 16 is the largest cut that never lost to production on the three PQ2_0 mod... | `MatMul.PQ2PlanarBlocked.cs` |
| `DOTLLM_PQ2_SMALLN_MAX` | Largest token count routed to the W2A8 GEMM tier (DOTLLM_PQ2_SMALLN_MAX, default unbounded since #1386; 1 restores the float tier for every n>=2). | `MatMul.PQ2S.cs` |
| `DOTLLM_PQ2_W2A8` | True when the planar PQ2_0 kernels are enabled: on by default with AVX2 (#1620), off with DOTLLM_PQ2_PLANAR=0, and off when DOTLLM_PQ2_W2A8=0 asks for the unquantized-activation float tier (planar also quantizes activations, so... | `MatMul.PQ2Planar.cs` |
| `DOTLLM_Q4E_DIAG_F32ACT` | Diagnostic (#873): when true every quantized expert GEMM dequantizes the weights to F32 and dots them with the UNQUANTIZED F32 activations (no Q8_K/Q8_0 activation quantization) - the "F32-decoded control" that separates activa... | `MoeSwiGluMlp.cs` |
| `DOTLLM_TERN_F32ACT` | True when DOTLLM_TERN_F32ACT=1: dense/expert ternary projections run the F32-activation scalar reference kernel (oracle mode). | `TernB64F16.cs` |

#### Cuda

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_ATTN_FLASH_PREFILL_HD256` | Opt-out gate: DOTLLM_ATTN_FLASH_PREFILL_HD256=0 forces the attention_f32 prefill path. Read once; default ON (issue #962). | `CudaKernels.cs` |
| `DOTLLM_ATTN_GQA_SPLIT` | Opt-in (default OFF) env-var gate for LaunchAttentionF32GqaSplit. Same tradeoff shape as EnableAttentionSplitKv (issue #183) for the split-KV combine's reassociation, PLUS the (bit-exact, validated) GQA-group regrid on top -- s... | `CudaKernels.cs` |
| `DOTLLM_ATTN_MMA_DECODE_GQA_SPLIT` | Max GQA group the M-dim-packing scheme supports - MUST equal MAX_GQA_GROUP in attention_flash_mma_decode_gqa_split.cu (also equals CudaKernels.MaxGqaGroup, the project-wide convention). | `CudaAttentionMmaDecodeGqaSplit.cs` |
| `DOTLLM_ATTN_MMA_DECODE_V2` | Issue #988: selects the v2 mma decode kernels (default ON; DOTLLM_ATTN_MMA_DECODE_V2=0 opts back out to v1). Mutable so tests/benchmarks can A/B within one process; the per-function co-residency ceilings are cached separately s... | `CudaKernels.cs` |
| `DOTLLM_ATTN_MMA_V2_MIN_KV` | Minimum KV rows per split for the v2 kernel (8 warps x 16-key tiles = 128 keys keeps every warp busy). DOTLLM_ATTN_MMA_V2_MIN_KV overrides. | `CudaKernels.cs` |
| `DOTLLM_ATTN_PAGED_NATIVE` | Enables LaunchAttentionPaged in CudaTransformerModel's paged-cache decode dispatch, in place of the default gather-into-scratch path. Default OFF (this kernel has not been measured against the gather path yet -- see issue #200'... | `CudaKernels.cs` |
| `DOTLLM_ATTN_SPLIT_KV` | Opt-in (default OFF) env-var gate for LaunchAttentionF32SplitKv. Enabling this trades away attention_f32's bit-near-CPU-equivalent output for a reassociated (mathematically equal, not bit-identical) cross-block combine - same t... | `CudaKernels.cs` |
| `DOTLLM_ATTN_SPLIT_KV_HP` | Opt-in (default OFF) env-var gate for LaunchAttentionF32SplitKvHp. Issue #226 spike: does accumulating the cross-split combine (partial_max/partial_sum/partial_out merge) in double precision reduce the real-generation argmax di... | `CudaKernels.cs` |
| `DOTLLM_BITNET_GRAPH_MAX_DEPTH` | Issue #213 mitigation: BitNet-only context-depth ceiling for the CUDA-Graphs decode path. Root cause (confirmed via CudaAttentionDynVsScalarPerfTest, an isolated CUDA-event microbenchmark with no graph capture involved): the gr... | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_ALLOW_QUANT_EXPANSION` | Opt-in gate for persistently expanding a quant type CUDA has no native kernel for into a larger resident format (FP16/F32) kept for the model's entire lifetime. Default disabled: a checkpoint that can't actually be run in its c... | `CudaKernels.cs` |
| `DOTLLM_CUDA_FLASH_ATTN` | Default sequence-length crossover at/above which flash is dispatched ahead of G3. The tuned kernel (GQA-group K/V sharing, group warps/block) beats G3 across the whole measured range on a 3060 - end-to-end attention-category G3... | `CudaFlashAttention.cs` |
| `DOTLLM_CUDA_FLASH_ATTN_MINSEQ` | Sequence-length crossover. Read once from DOTLLM_CUDA_FLASH_ATTN_MINSEQ (a positive integer) or the measured default. | `CudaFlashAttention.cs` |
| `DOTLLM_CUDA_G3_ATTN` | G3 tensor-core prefill-attention path: two cuBLAS strided-batched GEMMs around a coalesced causal-softmax kernel, replacing the FP32 CUDA-core attention_f16 kernel for pure-prefill attention on throttle-prone GeForce Ampere par... | `CudaG3Attention.cs` |
| `DOTLLM_CUDA_GEMM_16F` | G1 toggle: switches the prefill GEMM compute type from CUBLAS_COMPUTE_32F (FP16 inputs, FP32 accumulate - throttled to ~half rate on GeForce Ampere) to CUBLAS_COMPUTE_16F (FP16 accumulate - full tensor-core rate). Compute-16F r... | `CudaGemm.cs` |
| `DOTLLM_CUDA_HOST_EMBED` | Whether the CUDA model will RETAIN the host TransformerWeights after upload (and therefore must NOT stream-free per-tensor host scratch during upload). This is a conservative SUPERSET of the constructor's retain condition (_use... | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_HYBRID_GRAPH` | Environment switch for the decode graph; 0 keeps eager launches. | `CudaQwen3HybridDenseTransformerModel.DecodeGraph.cs` |
| `DOTLLM_CUDA_I2S_A8` | (not described in source) | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_KQUANT_SMALL_S_MAX` | Environment variable capping the K-quant / IQ4 multi-column range (issues #1316/#1375): projections of 2 &lt;= seqLen &lt;= KQuantMaxColumns over Q2_K/Q4_K/Q5_K/Q6_K/IQ4_XS/IQ4_NL weights take the rows4 _c{S} GEMV (weights read... | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_MEM_TRACE` | True when DOTLLM_CUDA_MEM_TRACE=1. | `CudaMemTrace.cs` |
| `DOTLLM_CUDA_MMVQ_COALESCED` | Opt-in: route Q4_K decode GEMV (k >= MmvqLargeKThreshold) through the experimental coalesced MMVQ kernel (warp-per-superblock, coalesced 128-byte qs loads) instead of the default MMVQ-large. On-the-fly only (ignores preq scratc... | `CudaKernels.cs` |
| `DOTLLM_CUDA_MMVQ_LLAMACPP` | Opt-in: route Q4_K decode GEMV (k >= MmvqLargeKThreshold) through the llama.cpp-faithful coalesced+accumulate-once MMVQ kernel (quantized_gemv_q4_k_mmvq_llamacpp) instead of the default MMVQ-large. Unlike MmvqCoalesced (#363, w... | `CudaKernels.cs` |
| `DOTLLM_CUDA_MMVQ_THREADS` | MMVQ-large block size (threads). MUST equal the compiled MMVQ_LARGE_THREADS in quantized_gemv_mmq.cu. Tunable via DOTLLM_CUDA_MMVQ_THREADS (32-256, multiple of 32; default 128) to sweep occupancy for the decode GEMV - for small... | `CudaKernels.cs` |
| `DOTLLM_CUDA_MOE_DEVICE_ROUTING` | False keeps decode routing on the host (the device-routed body's reference, #1182; DOTLLM_CUDA_MOE_DEVICE_ROUTING=0). | `CudaMoeFfn.cs` |
| `DOTLLM_CUDA_MOE_GRAPH` | Captures single-token decode of MoE models into a CUDA graph (#1210). On by default with UseGraphCapture; DOTLLM_CUDA_MOE_GRAPH=0 keeps the eager body (A/B). | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_MOE_STREAM` | GPU leg for multi-token calls of CPU-expert layers (#1202); DOTLLM_CUDA_MOE_STREAM=0 keeps them on the CPU. | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_MTP_KV` | Environment switch for the MTP head's KV format (issue #1095): f32 stores it in F32, the pre-#1095 layout. It exists only as the reference arm of the acceptance-rate A/B; FP32 storage is not a serving fallback (#1107). | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_CUDA_PINNED_UPLOAD` | Environment variable that, when 1, opts in to the pinned staged upload. | `CudaWeightUploadStaging.cs` |
| `DOTLLM_CUDA_PINNED_UPLOAD_CHUNK_MIB` | Environment variable overriding the chunk size, in MiB (clamped to 1..1024). | `CudaWeightUploadStaging.cs` |
| `DOTLLM_CUDA_PQ2_0_DP4A` | Environment variable that, when 0, stops PQ2_0 projections of seqLen == 1 and 2 &lt;= seqLen &lt;= MaxColumns from using the int8-activation dp4a GEMV (issue #485, the default) and sends them to the #482 / single-column F16-act... | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_PQ2_0_MMQ` | Environment variable that, when set to 1, routes PQ2_0 projections WIDER than MaxColumns (prefill, and perplexity's whole-window forward) to the packed dp4a GEMM (issue #490) instead of dequantizing the matrix to F16 for cuBLAS... | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_PQ2_0_MMQ_BN` | Environment variable forcing the PQ2_0 MMQ column tile (16, 32, or 64 for the int8 tensor-core tile); unset picks MmqTileColumns(int, bool)'s width for the token count. | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_PQ2_0_S1_MULTI` | Environment variable that, when 0, sends seqLen == 1 PQ2_0 decode back to the single-column kernel (default: multi-column S=1). | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_Q8_RB_GEMV` | Env switch: DOTLLM_CUDA_Q8_RB_GEMV=0 disables this kernel. | `CudaQ8_0RbGemv.cs` |
| `DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED` | Env switch (issue #492): DOTLLM_CUDA_Q8_RB_MULTI_SPECIALIZED=0 keeps LaunchMulti on the generic _rb_multi kernel, so the specialised entry points can be A/B'd against the exact code that shipped with #486. | `CudaQ8_0RbGemv.cs` |
| `DOTLLM_CUDA_Q8_STAGED_GEMV` | when DOTLLM_CUDA_Q8_STAGED_GEMV=0 disables the staged kernel. | `CudaQ8_0StagedGemv.cs` |
| `DOTLLM_CUDA_RESIDUAL_SCALE` | Power of two the F16 residual stream is carried at on architectures whose residual nears the F16 range (Flint, #1191): default 1/8 (DOTLLM_CUDA_RESIDUAL_SCALE; 1 disables it for A/B). Exact: every residual read is an RMSNorm. | `CudaTransformerModel.cs` |
| `DOTLLM_CUDA_SMALL_S_MAX` | Environment variable overriding DefaultMaxColumns; 0 disables the path. | `CudaSmallSGemvDispatch.cs` |
| `DOTLLM_CUDA_SPLIT_MIN_KV` | Minimum KV rows per split before further splitting stops paying for its own grid.sync + combine overhead. Override via DOTLLM_CUDA_SPLIT_MIN_KV. Issue #219 fix (was 128): at 128, this term was the BINDING clamp in ComputeAttent... | `CudaKernels.cs` |
| `DOTLLM_CUDA_SPLIT_TARGET_BLOCKS` | Target total co-resident blocks for ComputeAttentionKvSplit's occupancy term (baseBlocks is numKvHeads for the combined kernel). Default derived from an RTX-3060 sweep against the real Bonsai-27B shape (numKvHeads=4, headDim=25... | `CudaKernels.cs` |
| `DOTLLM_CUDA_WEIGHT_ARENA` | Opens an arena for the current context's device and makes it Current on this thread, or returns null (and leaves Current null) when it is disabled or unsupported. The arena; the caller owns it and ends the scope with End. | `CudaWeightArena.cs` |
| `DOTLLM_DISABLE_EMBED_ROWLOOKUP` | Env-var escape hatch (matches DOTLLM_DISABLE_MMQ_* convention) - forces the legacy bulk-dequant path even when a per-row kernel exists. Used for A/B perf comparison and as a fallback if a per-row kernel ever misbehaves on a new... | `CudaWeights.cs` |
| `DOTLLM_DISABLE_GRAPH_CAPTURE` | Env-var override for the default-on graph-capture path. Set DOTLLM_DISABLE_GRAPH_CAPTURE=1 to force eager decode regardless of capability. Test/benchmark hook follows the same convention as CudaKernels.DisablePreQ8_1, etc. | `CudaTransformerModel.cs` |
| `DOTLLM_DISABLE_MMQ_BATCHED_Q4K` | Test/benchmark hook to force the dequant->cuBLAS prefill fallback even when the batched-MMQ Q4_K kernel is loaded. Override: DOTLLM_DISABLE_MMQ_BATCHED_Q4K=1. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMQ_Q2K` | Test/benchmark hook to force the legacy Q2_K GEMV kernel even when MMQ is loaded. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMQ_Q4K` | Test/benchmark hook to force the legacy Q4_K GEMV kernel even when MMQ is loaded. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMQ_Q5K` | Test/benchmark hook to force the legacy Q5_K GEMV kernel even when MMQ is loaded. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMQ_Q6K` | Test/benchmark hook to force the legacy Q6_K GEMV kernel even when MMQ is loaded. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMQ_Q8_0` | Force Q8_0 decode back to the legacy FP GEMV (A/B switch for #1297). DOTLLM_DISABLE_MMQ_Q8_0=1. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_IQ4_NL` | Disable MMVQ-large routing for IQ4_NL. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_IQ4_XS` | Disable MMVQ-large routing for IQ4_XS. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_Q2K` | Disable MMVQ-large routing for Q2_K (forces MMQ-4-rows for k >= MmvqLargeKThreshold). | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_Q4K` | Disable MMVQ-large routing (forces MMQ-4-rows for k >= MmvqLargeKThreshold). | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_Q5K` | Disable MMVQ-large routing for Q5_K. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MMVQ_LARGE_Q6K` | Disable MMVQ-large routing for Q6_K. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_MOE_GROUPED_GEMV` | Disable the Phase-B grouped-GEMV path. Forces the per-expert LaunchQuantizedGemv fallback in CudaMoeFfn. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_PACKED_GATEUP` | Disable decode-time packed Gate/Up upload/dispatch for diagnostics. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_PACKED_QKV` | Disable decode-time packed QKV upload/dispatch for diagnostics. | `CudaKernels.cs` |
| `DOTLLM_DISABLE_PREQ8_1` | Disable pre-Q8_1 input quantization (falls back to on-the-fly Stage 1 in every GEMV). Useful for A/B comparison or when the model's max k makes the pre-quant scratch buffer awkward to size. Default off - pre-Q8_1 is the recomme... | `CudaKernels.cs` |
| `DOTLLM_DISABLE_QUANTIZED_GEMV` | Force dequant-then-cuBLAS fallback instead of quantized GEMV. | `CudaKernels.cs` |
| `DOTLLM_ENABLE_HIGH_PRECISION_IQUANTS` | Disable the F32 activation correctness path for IQ4-family models. | `CudaTransformerModel.cs` |
| `DOTLLM_FORCE_DIRECT_GEMV` | Force direct GEMV path, disabling MMQ and MMVQ-Large variants. Useful for diagnostics to isolate dequant vs advanced GEMV kernels. Set environment variable DOTLLM_FORCE_DIRECT_GEMV=1 to enable. | `CudaKernels.cs` |
| `DOTLLM_GDN_SCAN_APPROX_SPLIT4` | Opt-in (default OFF) env-var gate for LaunchGdnScanStepF32CoopSplit4. Enabling this trades away gdn_scan_step_f32's documented bit-exact CPU/GPU parity - the row-split reduction reassociates the retrieve/read accumulation (inde... | `CudaKernels.cs` |
| `DOTLLM_GRAPH_MAX_DEPTH` | Issue #218: generalizes BitNetGraphCaptureMaxDepth's depth-ceiling mitigation to every graph-capable architecture, not just BitNet. #213/#217 found and fixed the regression for BitNet specifically; #218 confirmed via real-model... | `CudaTransformerModel.cs` |
| `DOTLLM_HYBRID_DEBUG` | 4. Second residual add: hidden = residual + normOut. | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_HYBRID_KVQ_PER_ROW` | Issue #1068: per-row fused quantized-KV attention for short multi-row forwards. Default ON wherever the fused decode tier is on; DOTLLM_HYBRID_KVQ_PER_ROW=0 restores dequant-to-scratch for A/B measurement. | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_HYBRID_KV_FUSED` | (not described in source) | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_HYBRID_PROFILE` |  One-off category profiler (DOTLLM_HYBRID_PROFILE=1). Coarse (Stopwatch + per-mark Synchronize - the sync itself perturbs true pipelined timing, so this is for RELATIVE category comparison only, not absolute steady-state throug... | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_HYBRID_VRAM_LEDGER` | DOTLLM_HYBRID_VRAM_LEDGER=1: log the model-owned KV/staging bytes once (issue #956 Phase 0). | `CudaQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_I2S_MOE_BATCHED_GEMM` | Runtime gate for the batched I2_S GEMM kernel (issue #250) in ForwardBitNetI2S's per-expert gate/up/down projections. Mirrors MatMul.I2S.I2SGemmTile's DOTLLM_I2S_TILE convention on the CPU side: defaults to on (when the kernel... | `CudaMoeFfn.cs` |
| `DOTLLM_KQUANT_GEMV` | K-quant decode GEMV selector (#1298), DOTLLM_KQUANT_GEMV: auto (default, per-shape rule in UseKQuantRows4), rows4 (force the warp-per-row kernel wherever it exists) or large (force the legacy one-row-per-block MMVQ-large kernel... | `CudaKernels.cs` |
| `DOTLLM_MMQ_BATCHED_MIN_SEQLEN` | Minimum prefill seqLen (inclusive) at which the batched-MMQ Q4_K kernel is preferred over the dequant->cuBLAS HGEMM fallback - mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE / MMQ_DP4A_MAX_BATCH_SIZE crossover gating (docs/perf/MMA_BA... | `CudaKernels.cs` |
| `DOTLLM_NVFP4_DECODE_GEMV` | Which NVFP4 decode GEMV UseNvfp4Dp4a picks (#1061). Environment variable DOTLLM_NVFP4_DECODE_GEMV=auto\|dp4a\|f32; anything else is Nvfp4DecodeGemv.Auto. | `CudaKernels.cs` |
| `DOTLLM_PROFILE_LAUNCH` | Launch-ceiling profiling (env DOTLLM_PROFILE_LAUNCH=1). Measures CPU-side kernel-dispatch time (queue all launches, no GPU wait) vs the full forward including the single _stream.Synchronize(). Zero-cost when the env var is unset. | `CudaTransformerModel.cs` |

#### Engine

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_DEVICE_GUMBEL` | Env switch: DOTLLM_DEVICE_GUMBEL=0 keeps the verify rows' Gumbel draw on the host (logits read back) even when the model can draw on the device. | `MtpSpeculativeDecoder.cs` |
| `DOTLLM_MTP_DRAFT_ARGMAX` | Env switch that forces full-logits draft steps even on models with a native draft argmax. | `MtpSpeculativeDecoder.cs` |
| `DOTLLM_MTP_GDN_SNAPSHOTS` | Env switch that forces checkpoint + replay even on snapshot-capable models. | `MtpSpeculativeDecoder.cs` |
| `DOTLLM_MTP_TRACE` | #820: only what is NOT resident yet counts. The per-forward scratch (DOTLLM_VK_PLANNED_ROWS) is allocated at load and is already in `live`; the old estimate added a flat 1 GiB for it again (and KV at full capacity without the i... | `MtpAdaptiveGate.cs` |
| `DOTLLM_PPL_LOGITS_DUMP` | Debug aid (#862): DOTLLM_PPL_LOGITS_DUMP= appends every window's raw float32 logits (all rows, [context][vocab], row i predicts token i+1) so a cross-engine KL can be computed offline over the FULL vocabulary (not a top-k trunc... | `PerplexityEvaluator.cs` |
| `DOTLLM_PPL_NLL_DUMP` | Debug aid (#737): DOTLLM_PPL_NLL_DUMP= writes one "window pos token nll" line per scored token, for per-token diffing against llama.cpp's --kl-divergence-base log-probs. | `PerplexityEvaluator.cs` |

#### Models

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_CPU_MTP_KV` | Environment variable selecting the head KV format: f16 (default), q8_0, q4_0 or f32 (reference arm only). | `CpuMtpState.cs` |
| `DOTLLM_DFLASH_DRAFT_VOCAB` | Draft-vocabulary limit (#1360): DOTLLM_DFLASH_DRAFT_VOCAB=K reads only token ids [0, K) of the lm-head for proposals; 0 = full; unset = the target's default. | `DFlashDrafter.cs` |
| `DOTLLM_DFLASH_TRACE` | DOTLLM_DFLASH_TRACE=1: per-request draft / inject wall time on stderr at dispose. | `DFlashMtpState.cs` |
| `DOTLLM_DIAG_EMBED_NOISE` | DIAGNOSTIC (DOTLLM_DIAG_EMBED_NOISE=&lt;eps&gt;): multiplies every embedding element by 1 + eps*u, u in [-1,1] (fixed seed): the noise-injection floor probe (#1599). | `TransformerModel.cs` |
| `DOTLLM_DIAG_F32ACT_ALL` | DIAGNOSTIC (DOTLLM_DIAG_F32ACT_ALL=1): every quantized dense projection runs dequantize-rows x F32 activations instead of the quantized-activation kernels (combine with DOTLLM_Q4E_DIAG_F32ACT=1 for the MoE banks). The F32-activ... | `TransformerModel.cs` |
| `DOTLLM_EMBED_PAR_MIN` | Minimum batch size for one-sequence-per-core execution; below it a batch runs sequentially on the pooled kernels. Env DOTLLM_EMBED_PAR_MIN overrides (benchmarking). | `BertEncoderModel.cs` |
| `DOTLLM_G4_PROF` | Env-gated (DOTLLM_G4_PROF=1) stopwatch buckets for the Gemma-4 CPU layer path. | `G4Prof.cs` |
| `DOTLLM_I2S_CACHE` | Builds a BitNetI2SCacheContext for a BitNet checkpoint so repeated loads reuse the ternary I2_S packing instead of re-quantizing bf16 weights. The cache lives in a hidden .dotllm-i2s-cache folder beside the checkpoint and is ke... | `ModelLoader.cs` |
| `DOTLLM_LITTLEBIT_DENSE_CONTROL` | Loads a NanoQuant composite (issue #869) from a GGUF and registers it as a LittleBit-kind weight. A non-empty rowSplits splits the stacked output rows (attn_qkv) into one weight per entry. With DOTLLM_LITTLEBIT_DENSE_CONTROL=1... | `FactorizedWeights.cs` |
| `DOTLLM_LORA_FORCE_Q8_PREQUANT` | Phase 4d.5 / Gap 2 - fast-path plumbing: when both base and adapter B are Q8_0 AND the caller pre-quantised x, we can route stage 1 through `MatMul.GemmQ8_0(preQuantizedInput=preQuantX)` and skip the activation-quant cost. The... | `LoraProjection.cs` |
| `DOTLLM_MOE_LEGACY_PACKED` | (not described in source) | `Lfm2MoeTransformerModel.cs` |
| `DOTLLM_MTP_ABSORB_PER_TOKEN` | Selects how an MTP-carrying trunk Forward absorbs its batch into the MTP head's KV-cache (issue #472): one batched, KV-only head pass over all S rows (the default), or the #469 reference loop that runs the full single-token hea... | `MtpAbsorbDispatch.cs` |
| `DOTLLM_MTP_HNORM` | Env var selecting the MTP hnorm convention: stream (default) or joint. | `Qwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_NEMOTRON_MOE_GROUPED` | DOTLLM_NEMOTRON_MOE_GROUPED=0 forces the per-token MoE body for every batch size (the reference arm for #1039). Settable from tests. | `NemotronHTransformerModel.cs` |
| `DOTLLM_Q4E_DIAG_NOISE` | 512-expert softmax MoE: router softmax over ALL experts -> top-k -> renormalise; routed experts + shared expert x sigmoid(gate). Diagnostic (#873): relative noise amplitude injected into every MoE-layer input before routing (de... | `Qwen4ExpTransformerModel.cs` |
| `DOTLLM_R4_DECODE_KQUANT` | Minimum row byte stride for R4 interleaving to be beneficial. Below this, 4 rows span &lt; 4KB (1 page) and the hardware prefetcher handles the original stride efficiently. Above this, R4 contiguity avoids cross-page TLB misses... | `TransformerModel.cs` |
| `DOTLLM_TENSOR_DUMP_DIR` | Env-controlled tensor stats dumper for layer-by-layer parity diffs against llama.cpp. Default off (no env var = no output, no allocation, no per-tensor branch cost). Enable by setting DOTLLM_TENSOR_DUMP=1 to write to stderr, or... | `TensorDump.cs` |
| `DOTLLM_TERNARY_SCALES` | Initial ScaleFormat of new hooks: TernaryScaleFormat.F16 (range-guarded, per-circuit F32 fallback), unless DOTLLM_TERNARY_SCALES=f32 forces exact reference F32 scales. | `TernaryResidualHook.cs` |
| `DOTLLM_TRACE_LAYERS` | Opt-in diagnostic tracing for Nemotron-H forward pass. All checks are gated on environment variables read once at process startup, so when diagnostics are disabled the overhead is a single predictable branch per layer and no st... | `NemotronHDiagnostics.cs` |
| `DOTLLM_TRACE_SSM` | Opt-in diagnostic tracing for Nemotron-H forward pass. All checks are gated on environment variables read once at process startup, so when diagnostics are disabled the overhead is a single predictable branch per layer and no st... | `NemotronHDiagnostics.cs` |
| `DOTLLM_TRACE_SSM_LAYER` | (not described in source) | `NemotronHDiagnostics.cs` |

#### Server

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_VK_QWEN4EXP_QSA` | Issue #1324: qwen4exp on Vulkan sizes its K/V + indexer caches at load, so the serve options that decide the window are handed to it first: --ctx-size becomes the requested capacity (reduced to what fits the device) and the K/V... | `ServerStartup.cs` |

#### Tray.Core

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_UPDATE_REPO` | Environment variable that overrides the repository checked (owner/repo). | `UpdateChecker.cs` |

#### Vulkan

| Variable | What the source says | Defined in |
|---|---|---|
| `DOTLLM_EMBED_PACK` | True when documents can be packed into one embedding forward: the validated plain-causal decoder shape (Qwen3), the coopmat flash kernel serving the head dim, no window / soft-cap / sinks / distinct V dim, and row strides that... | `VulkanTransformerModel.cs` |
| `DOTLLM_EMBED_PACK_TOKENS` | Token budget of one packed forward. Env DOTLLM_EMBED_PACK_TOKENS overrides (default 1024, as the decoder embedder). | `VulkanBertEncoderModel.cs` |
| `DOTLLM_GPTOSS_DEBUG_OFF` | MiMo-V2 (#955): sink-attention kernels (shared with gpt-oss) and the attention-output compaction kernel. Refuses to run a model whose sliding layers lack their attention sinks (silently dropping them would be wrong logits). | `VulkanTransformerModel.cs` |
| `DOTLLM_MTP_HEADROOM_CANARY` | Test hook / switch for the second-device canary (see VulkanHeadroomCanary). Env DOTLLM_MTP_HEADROOM_CANARY=0 turns it off, leaving only the arithmetic estimate. | `VulkanQwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_MTP_HEADROOM_EXTRA_MIB` | DOTLLM_MTP_HEADROOM_EXTRA_MIB is a diagnostic: it asks the probe for that much more, which makes "the head does not fit" reproducible on demand (a competing process only takes memory from this one when it arrives AFTER the load... | `VulkanQwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_MTP_HEAD_REQUANT` | Env DOTLLM_MTP_HEAD_REQUANT: 0 = keep the head's native routed-bank quantisation, 1 (default) = re-quantise a Q8_0 head's routed banks (Q4_K gate/up, Q5_1 down), 2 = only when the native head does not fit the device. Measured o... | `VulkanQwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_NEMOTRON_VERIFY_COPY` | Read-only verify (#1237, default when the register scan serves the model): the verify scan reads the live state and never writes it; the accepted rows are replayed into it afterwards. No copy of the state exists. DOTLLM_NEMOTRO... | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_TIER_DEBUG` | (not described in source) | `VulkanExpertTier.cs` |
| `DOTLLM_VK_BATCH_FUSE` | Opt-out for the fused batched forward (DOTLLM_VK_BATCH_FUSE=0): falls back to the serial per-sequence loop, which is the bit-identity oracle. | `VulkanQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_VK_BATCH_FUSE_GDN` | Opt-out for stacking the GDN projections (DOTLLM_VK_BATCH_FUSE_GDN=0); the stacked FFN stays on. | `VulkanQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_VK_BATCH_FUSE_MIN_ROWS` | Smallest total row count worth stacking. Measured on Tev1-4B decode (one row per sequence, same-window interleaved, reference-checked): stacked vs the serial loop = 0.56x at 2 sequences, 0.79x at 3, 1.00x at 4, 1.18x at 5, 1.35... | `VulkanQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_VK_DEINTERLEAVE` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_EMBED_HOST` | glm4moe (#1600): a packed (non-Q8_0) table is gathered on the host instead of being widened to a 2.3 GiB F32 device copy. | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_EXPERT_QD` | Creates the tier. Device whose memory holds the ring and id buffers. GGUF shard files, indexed by Layer.File. One entry per MoE layer. Routed experts per layer. Experts selected per token. Slots per layer. Bypass the page cache... | `VulkanExpertTier.cs` |
| `DOTLLM_VK_EXPERT_SLOTS` | Expert tiering (#939): routed experts per layer kept in device slots; the rest stay on the SSD and are read on demand (stall on miss, output identical to a fully resident run). 0 = off (every routed expert resident, the default... | `VulkanQwen4ExpTransformerModel.Loading.cs` |
| `DOTLLM_VK_F16_ACT` | Env var that disables the F16-activation path (0) for A/B runs. | `CoopmatF16Gemm.cs` |
| `DOTLLM_VK_F16_ACT_MIN_N` | Smallest prefill length that takes the F16-activation ffn-down path (the activation matrix must outgrow the cache). | `VulkanQwen3HybridDenseTransformerModel.cs` |
| `DOTLLM_VK_F16_GEMM_LEGACY` | Environment variable that restores the pre-issue-#443 preference (Coopmat64) in SelectFor. Exists so Blocked128x128x4 can be A/B'd against what shipped, in alternating processes on a real model - a kernel bench can only compare... | `MatMulF16GemmCoopmatKernel.cs` |
| `DOTLLM_VK_F32_GEMV` | (not described in source) | `MatMulF32Kernel.cs` |
| `DOTLLM_VK_F32_TILED` | (not described in source) | `MatMulF32Kernel.cs` |
| `DOTLLM_VK_F32_TILED64` | (not described in source) | `MatMulF32Kernel.cs` |
| `DOTLLM_VK_FLASH_HD256` | Environment override for the wide-head variant: br16 (default), br8, or off. | `VulkanFlashAttentionF32Kernel.cs` |
| `DOTLLM_VK_FLASH_HD512` | #914: optional 512-dim variant (Gemma-4 global layers). Same degrade-don't-throw contract as the wide SPV; only meaningful on top of a loaded wide variant so SupportedMaxHeadDim stays monotone. | `VulkanFlashAttentionF32Kernel.cs` |
| `DOTLLM_VK_FUSED_ROPE_KV_F16` | Writer + split-KV decode twin. The GQA-grouped split pass (one KV read per (KV head, sub-group) instead of one per query head) is on for glm4moe (#1612: 96 / 8 heads, `attn_split` was 70 % of a decode step at 8K); the other den... | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_FUSE_FORWARD` | (not described in source) | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VK_GDN_CONV_FUSED` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_GDN_SCAN_VARIANT` | Raw VkPipeline handle - diagnostics only, so VulkanDevice.GetShaderStatisticsAmd can report this kernel's post-compile VGPR/SGPR/LDS/scratch allocation. No codepath's correctness or performance depends on it. | `GdnScanMultiTokenF32Kernel.cs` |
| `DOTLLM_VK_GEMMA4_PLE_GPU` | Creates the Gemma-4 dense-PLE GPU input pipeline (#916): the F32 per_layer_model_proj device copy (mirrors the qwen4exp VulkanQwen4ExpPleGpu approach), its norm weight and the combine kernel. DOTLLM_VK_GEMMA4_PLE_GPU=0 keeps th... | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_GEMM_SPLITK` | Env var that disables split-K (0) for A/B runs. | `CoopmatSplitK.cs` |
| `DOTLLM_VK_GMOE_FUSE` | Runtime A/B switch: fuse the routed-expert activation with the Q8_1 quantize of the down input (default on). | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_HYBRID_LEGACY_PACKED` | A/B switch for #921: DOTLLM_VK_HYBRID_LEGACY_PACKED=0 restores the pre-#921 widening of Q4_0 / Q4_1 / Q5_0 / Q5_1 hybrid projections to F32 (resident-bytes and greedy-identity comparisons). Read once at startup; tests may set it. | `VulkanQwen3MoeHybridWeights.cs` |
| `DOTLLM_VK_HYBRID_PREFILL_MMQ` | Opt-out of the hybrid hosts' dp4a MMQ prefill (#1149): 0 restores the coopmat-or-scalar dispatch for A/B. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_I2_S_GEMM_LEGACY` | Environment variable that restores the pre-issue-#443 preference in SelectFor. | `MatMulI2SGemmF32Kernel.cs` |
| `DOTLLM_VK_IQ2_XXS_GEMM_LEGACY` | Environment variable that disables this kernel so callers fall back to the F16 or scalar IQ2_XXS GEMM (A/B). | `MatMulIq2XxsGemmCoopmatKernel.cs` |
| `DOTLLM_VK_IQ4_XS_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulIq4XsGemmCoopmatKernel.cs` |
| `DOTLLM_VK_KEEP_HOST_SHARED` | #890: the shared-expert gate/up/down F32 host copies (LoadMoe.SharedF32) existed only to be uploaded as F16; every copy is drained above. They are ~3.3 GiB of private memory on the real UD-Q4_K_XL file, and the OS caps a proces... | `VulkanQwen4ExpTransformerModel.Loading.cs` |
| `DOTLLM_VK_LFM2_HOST_PROFILE` | Host-side split of a decode forward (DOTLLM_VK_LFM2_HOST_PROFILE=1): command recording, submit + GPU wait, logits download; printed per 32 tokens. | `VulkanLfm2MoeTransformerModel.cs` |
| `DOTLLM_VK_LFM2_Q8_GATEUP_MMVQ` | The layer-less host sizes its MoE scratch from config.Moe; LFM3 has none, so hand it a one-expert placeholder it never uses. | `VulkanLfm2MoeTransformerModel.cs` |
| `DOTLLM_VK_LFM2_STAGE_PROFILE` | GPU-timestamp stage profile (DOTLLM_VK_LFM2_STAGE_PROFILE=ts): one BOTTOM_OF_PIPE stamp per stage in the real submission structure; per-forward breakdown on stderr (decode: averaged over 32 forwards). | `VulkanLfm2MoeTransformerModel.cs` |
| `DOTLLM_VK_LFM3_ATTN_CHUNK` | DOTLLM_VK_LFM3_ATTN_CHUNK=0 disables query-row slicing of long prefills (A/B and bisection). | `VulkanLfm3TransformerModel.cs` |
| `DOTLLM_VK_LFM3_GLOBAL_GQA` | A/B switch (DOTLLM_VK_LFM3_GLOBAL_GQA=0): global-attention prefills go back to the per-head coopmat flash kernel. | `VulkanLfm3TransformerModel.cs` |
| `DOTLLM_VK_LFM3_SLIDING_OPT` | A/B switch (DOTLLM_VK_LFM3_SLIDING_OPT=0): restores the pre-#1572 behaviour (full-range split-KV, per-head flash) for sliding layers. | `VulkanLfm3TransformerModel.cs` |
| `DOTLLM_VK_MAMBA2_SCAN_LEGACY` | Environment variable that disables this kernel so the reference scan is used (A/B). | `Mamba2SelectiveScanRegsF32Kernel.cs` |
| `DOTLLM_VK_MLA_LEGACY` | (not described in source) | `VulkanWeights.cs` |
| `DOTLLM_VK_MMQ_CM1` | DOTLLM_VK_MMQ_CM1=0 disables the int8-coopmat GEMMs (A/B against the F16 coopmat GEMMs). | `MatMulMmqCm1Kernel.cs` |
| `DOTLLM_VK_MMQ_CM1_DENSE` | #1226: opt-in (DOTLLM_VK_MMQ_CM1_DENSE=1). Measured on the 8060S (Qwen3.8-27B pp512): 300 tok/s against 335-348 for the F16 coopmat GEMM, whose 128x128 tile already runs these shapes at 17-20 TFLOP/s; the MoE grouped form is wh... | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_DECODE_FUSED` | One-line diagnostic that makes the chosen bank storage type observable (#371): the only place that knows whether the resident-quant overlay engaged or silently fell back to F32. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_DOWN_RESCALE` | A/B and test switch for the grouped MoE down-input rescale (#1171); DOTLLM_VK_MOE_DOWN_RESCALE=0 disables it. | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_MOE_FUSED_GLUE` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_GROUPED` | #1677: native dense IQ prefill GEMMs on the shared codebooks (TryCreate gates on coopmat + wave64 + the SPIR-V + the env opt-out). | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_GROUPED_MIN_TOKENS` | MoE layers recorded by TryRecordMoeLayerDecodeFused (test hook). | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_MOE_GROUPED_MMQ` | #986: dp4a grouped MMQ kernels (preferred over coopmat when available). DOTLLM_VK_MOE_GROUPED_MMQ=0 disables. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_GROUPED_TILE` | Output rows per workgroup. The multi-subgroup shaders assume wave64 (subgroup = 64 threads = one 16-row M tile), so they are chosen only on a wave64 device. Default 64; DOTLLM_VK_MOE_GROUPED_TILE = 16 \| 64 overrides (16 = the... | `MoeGroupedMatmulKQuantCoopmatKernel.cs` |
| `DOTLLM_VK_MOE_INDEXED_MMQ` | #383/#633: dp4a indexed-matmul MMQ for Q4_K-resident gate/up (+ Q5_K down) banks. Default-on after real-model validation (Qwen3.6-35B-A3B Q4_K_M, Strix Halo: pp128 23 -> 75, tg 13.3 -> 18.1 tok/s, PPL within noise); DOTLLM_VK_M... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_MOE_INDIRECT_TILES` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_IQ_GROUPED` | #823: grouped IQ kernels (16-row tiles). DOTLLM_VK_MOE_IQ_GROUPED=0 leaves IQ banks on the MMVQ path for every row count. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_IQ_V2` | Default for new kernels; DOTLLM_VK_MOE_IQ_V2=0 selects the V1 shaders (A/B and bisection). | `MoeIndexedMatmulIqMmvqKernel.cs` |
| `DOTLLM_VK_MOE_IQ_VARIANT` | #1348/#1353: the grouped IQ MoE kernels run 64 tokens per workgroup (n64diet; n32diet selectable) on layers where gate/up and down both have that variant, and the 16-token kernels elsewhere. DOTLLM_VK_MOE_IQ_VARIANT=off (or 0)... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_MOE_MMVQ` | #823 A/B switch: DOTLLM_VK_IQ_RESIDENT=0 widens IQ expert banks to F32 as before (pairs with the residency estimate). | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_MMVQ_MAX_ROWS` | Largest row count the MMVQ arm is PREFERRED over the expert-grouped coopmat GEMM (default 4, measured crossover; DOTLLM_VK_MOE_MMVQ_MAX_ROWS overrides). Above it, layers whose banks the grouped path supports go grouped (EffGrou... | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_MOE_MR` | #912: indexed dp4a MMVQ twins of the scalar decode kernels. K-quant ones take the xDiv broadcast (no expanded-input copy); they bypass UseKQuantMmvq on purpose - that gate is the DENSE-GEMV finding (#801); Qwen3.6-35B-A3B / Gem... | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_MR_MIN_ROWS` | Smallest token count that takes the multi-row routed-MoE MMVQ variants (#876); 0 = never (A/B). Default 2 leaves single-token decode on the proven kernels (DOTLLM_VK_MOE_MR_MIN_ROWS=1 opts decode in: -9% 1-row forward on the re... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_MOE_Q4_1_V2` | DOTLLM_VK_MOE_Q4_1_V2=0 keeps the v1 grouped path (A/B). | `MoeGroupedQ4_1V2Kernel.cs` |
| `DOTLLM_VK_MOE_Q8_DOWN_WIDE` | #1354: a Q8_0 down bank (5 of 48 qwen4exp layers) has its own 32-token kernel, so those layers join the 32-token tile list too. | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_MOE_Q8_MMVQ` | Multi-row variants (#876): measured best at NR = 2 for the ALU-heavy Q4_K gate/up and NR = 4 for the Q5_1 down; DOTLLM_VK_MOE_MR= forces n for both, DOTLLM_VK_MOE_MR=0 disables them. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_ROWPAIR` | Row-pair shaders (issue #691): one workgroup covers 32 token rows of an expert and dequantises each weight super-block once for both 16-row tiles. They need the indirect tile list (built with 32-row tiles) and ONE tile list ser... | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_SHARED_F16` | Uploads an F32 matrix from an unmanaged pointer (the shared-expert projections live as nint in MoeLayerWeights). | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VK_MOE_SHARED_GATE_FUSED` | Keep the "both/all null together" invariant documented on the properties: if the quantizer SPV is missing even though an MMQ kernel loaded, don't leave a half-wired dp4a path -- fall back entirely to F32. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_MOE_SHARED_PACKED` | #987: keep the shared-expert gate/up/down packed Q8_0 on the device (no F16 copies) when all three source tensors are Q8_0. DOTLLM_VK_MOE_SHARED_PACKED=0 restores the pre-#987 layout (F16 gate/up/down + decode-only Q8_0 gate/up... | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VK_MOE_SHARED_Q8` | Layers uploaded with packed Q8_0 shared experts (#987): the counter that proves the fast path ran. | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VK_MTP_DRAFT_SKIP` | Perturbation switch for the draft step (timing attribution only; the drafted token is garbage when non-zero): 1 skips the head MoE, 2 the LM head matmul, 4 the head attention, 8 the front (embedding, eh_proj), 16 the GR plumbin... | `VulkanQwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_VK_NEMOTRONH_Q5_0_F32` | DOTLLM_VK_NEMOTRONH_Q5_0_F32=1 restores the pre-#568 behaviour (expand Q5_0 to F32 at upload) for A/B. | `VulkanNemotronHWeights.cs` |
| `DOTLLM_VK_NEMOTRON_EXACT_GEMV` | Test hook: false keeps the #1237 per-column GEMVs (A/B and the negative control of the multi-column tests). | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_GEMV_ROWS` | Environment override of the exact GEMVs' rows per workgroup, decode,verify (tuning A/B; e.g. 2,4). | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_KQUANT_NR` | (not described in source) | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_KQUANT_ROWS` | #1459: per-n NR for the legacy-quant / Q8_0 kernels (Nano Q4_K_M sweep); an explicit DOTLLM_VK_NEMOTRON_GEMV_ROWS wins. | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_MOE_GROUPED` | DOTLLM_VK_NEMOTRON_MOE_GROUPED=0 keeps every routed layer on the indexed (per-row) expert kernels (#1046). | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_MOE_GROUPED_MIN_TOKENS` | Smallest batch that takes the grouped prefill; below it the indexed MMVQ wins (same threshold as the Qwen3-MoE host). | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_MOE_MMVQ` | DOTLLM_VK_NEMOTRON_MOE_MMVQ=0 routes the routed experts through the F32-activation indexed kernels (exact reference arm) instead of dp4a MMVQ. | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_MOE_MR` | Output rows per workgroup of the Q4_K up-projection MMVQ (#1049, the #876 multi-row kernel); DOTLLM_VK_NEMOTRON_MOE_MR overrides, and 0 / 1 keeps the one-row kernel. Read when the model is built. | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_MOE_XDIV` | DOTLLM_VK_NEMOTRON_MOE_XDIV=0 restores the broadcast + topK-fold quantize in front of the indexed K-quant up projection (#1049). On by default: the expert input is quantized once per token and the MMVQ reads row r / topK for ro... | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_PADODD` | Test/A-B hook: false keeps odd 5/7-row dispatches on their own pipelines (#1459). | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_PERN` | Test/A-B hook: false restores the single NR for every row count. | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_PERN_TABLE` | DOTLLM_VK_NEMOTRON_PERN_TABLE="n:exactNR:kquantNR[:q4kNR];..." overrides entries (tuning sweeps; the optional fourth field is Q4_K's own table, #1516). | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NEMOTRON_PROFILE` |  #1048: per-op GPU attribution (env-gated; zero cost when off)  DOTLLM_VK_NEMOTRON_PROFILE=1 times every Forward with BOTTOM_OF_PIPE timestamps per op category (VulkanOpProfiler, #434), submitting per layer so one query pool co... | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_PROFILE_OUT` | (not described in source) | `VulkanNemotronHTransformerModel.cs` |
| `DOTLLM_VK_NEMOTRON_Q4K_LEGACY` | #1516: Q4_K has its own table since #1511 replaced its shader. The pipelined 16-byte-load Q4_K kernel has no 5- or 7-column cliff (the old KQuantRowsByColumns entries and the 5 -> 6 / 7 -> 8 padding were tuned to it), so NR sta... | `VulkanNemotronHTransformerModel.DFlash.cs` |
| `DOTLLM_VK_NVFP4_GEMM_LEGACY` | Environment variable that disables this kernel so callers fall back to the F16 or scalar NVFP4 GEMM (A/B). | `MatMulNvfp4GemmCoopmatKernel.cs` |
| `DOTLLM_VK_NVFP4_GEMV_LEGACY` | Environment variable that forces the #1038 one-block-per-lane GEMV (A/B); it also disables the MMVQ. | `MatMulNvfp4F32Kernel.cs` |
| `DOTLLM_VK_NVFP4_MMVQ` | Environment variable that enables the MMVQ (1). Off by default: on Strix (Radeon 8060S) at Bolt's shapes it is no faster than the coalesced FP32 GEMV (248 vs 250 us per Mamba layer, 27.3 tok/s either way), while its int8 activa... | `MatMulNvfp4F32Kernel.cs` |
| `DOTLLM_VK_NVFP4_MMVQ_MIN_BYTES` | Environment variable overriding DefaultMmvqMinWeightBytes. | `MatMulNvfp4F32Kernel.cs` |
| `DOTLLM_VK_PERTURB` | Perturbation (#1334): DOTLLM_VK_PERTURB=target=after1\|after2 suppresses every dispatch recorded between the stage marker named in after and the marker target (timing attribution only; needs stage profiling on). | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_PLANNED_ROWS` | Rows the per-forward scratch is pre-sized to at load (DOTLLM_VK_PLANNED_ROWS, default 1024: ~0.55 GiB, because on a 127 GiB box the real file leaves only ~1.3 GiB under the OS limit), clamped to kvCapacity. Scratch otherwise gr... | `VulkanQwen4ExpResidencyPlan.cs` |
| `DOTLLM_VK_PQ2_0_GEMM_LEGACY` | DOTLLM_VK_PQ2_0_GEMM_LEGACY=1 restores the pre-#439 preference. It exists so the 128x128 tile can be A/B'd against what shipped, in alternating processes, on a real model -- the ladder bench can only compare kernels in isolation. | `MatMulPQ2_0GemmF32Kernel.cs` |
| `DOTLLM_VK_PQ2_0_GEMV_LOOP_MAX_N` | Environment variable overriding DefaultGemvLoopMaxN; 0 disables the loop. | `PQ2_0SmallNDispatch.cs` |
| `DOTLLM_VK_PQ2_0_MULTICOL_MAX_N` | Environment variable overriding DefaultMultiColumnMaxN; 0 disables the multi-column path. | `PQ2_0SmallNDispatch.cs` |
| `DOTLLM_VK_PQ2_0_MULTIROW` | Environment variable; 0 disables the #474 multi-row kernels (read once). | `MatMulPQ2_0GemvF32Kernel.cs` |
| `DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT` | Fixed layers per submission for a prefill longer than FuseMaxSeqLen when DOTLLM_VK_PREFILL_LAYERS_PER_SUBMIT is set (1 = one per layer), else 0: the submissions are then bounded by _submitBudget (issue #1100). | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VK_PREFILL_SUBMIT_LOG` | DOTLLM_VK_PREFILL_SUBMIT_LOG=1: one stderr line per budgeted prefill forward (submissions, longest, fitted rates). | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VK_Q2_K_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ2KGemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q3_K_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ3KGemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q4E_BF16_AS_F16` | #823: the ISTA GSQ-RCO files store the gated-residual low-rank projections as BF16, whose prefill GEMM is the plain tiled F32 kernel (about 3.4 s of a 7 s 1K-token prefill: 4 projections x 48 layers). F16 has the cooperative-ma... | `VulkanQwen4ExpGrWeights.cs` |
| `DOTLLM_VK_Q4E_COMBINE_FUSED` | Single-token fused scatter + shared-gate add in the qwen4exp decode chain (#885). DOTLLM_VK_Q4E_COMBINE_FUSED=0 (or setting this) restores the two dispatches. | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_Q4E_DECODE_MR` | Single-token decode uses the multi-row routed-MoE MMVQs (#885; 56.7 -> 52.5 ms per 1-row forward on UD-Q4_K_XL). On by default; DOTLLM_VK_Q4E_DECODE_MR=0 at startup (or setting this) restores the one-row kernels. Read at the to... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_GR_FUSED` | Gated-residual reads fold the residual copy into an out-of-place group norm and overlap the inject projection with the low-rank chain (#885). Bit-identical to the unfused sequence. On by default; DOTLLM_VK_Q4E_GR_FUSED=0 (or se... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_MOE_FUSED` | qwen4exp single-token MoE decode on the fused chain (#885), the no-post-attention-norm twin of RecordMoeDecodeFast. On by default; DOTLLM_VK_Q4E_MOE_FUSED=0 (or setting this) restores the general serial path. | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_Q4E_PLE_GPU` | Forwards run the n-gram key/value projections on the GPU (#885: ~4 ms of a ~57 ms decode step; #896: prefill chunks too - the CPU F32 GEMM was ~0.75 s of every 1024-row chunk). On by default; DOTLLM_VK_Q4E_PLE_GPU=0 at startup... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_PREFETCH` | Start paging the n-gram table rows of a chunk in at the top of the forward (#822) and, for MTP, while the draft steps run. On by default; DOTLLM_VK_Q4E_PREFETCH=0 turns it off at startup (A/B diagnostic - it never changes a res... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_Q8_WIDE` | Wide Q8_0 MMVQ twins for the gated-residual down / up projections. On by default; DOTLLM_VK_Q4E_Q8_WIDE=0 (or setting this) restores the one-wave-per-row kernel. | `VulkanQwen3MoeHybridTransformerModel.Qwen4Exp.cs` |
| `DOTLLM_VK_Q4E_SPLITKV_ROWS` | qwen4exp 2..8-row verify forwards attend with the per-row split-KV kernel (the 1-row decode kernel's arithmetic at each row's own length) instead of the f16 coopmat flash attention. DOTLLM_VK_Q4E_SPLITKV_ROWS=0 (or setting this... | `VulkanQwen3MoeHybridTransformerModel.Qwen4Exp.cs` |
| `DOTLLM_VK_Q4E_SPLIT_ABOVE` | Start paging the n-gram table rows of a chunk in at the top of the forward (#822) and, for MTP, while the draft steps run. On by default; DOTLLM_VK_Q4E_PREFETCH=0 turns it off at startup (A/B diagnostic - it never changes a res... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_SPLIT_LAYERS` | Layers of work submitted ahead (fence-free) after the PLE step so the GPU runs while the host records the rest of a short forward (#885); 0 disables. DOTLLM_VK_Q4E_SPLIT_LAYERS at startup, default 3. Read at the top of every fo... | `VulkanQwen4ExpTransformerModel.cs` |
| `DOTLLM_VK_Q4E_VERBOSE` | Allocate the verify snapshots NOW: an out-of-memory belongs at attach time (where the caller declines speculation), not mid-decode. | `VulkanQwen4ExpTransformerModel.Mtp.cs` |
| `DOTLLM_VK_Q4K_PADODD` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4K_ROWS` | Output rows per workgroup of MatMulQ4KMulti (#1220) for one column: DOTLLM_VK_Q4K_ROWS (1..8), default 2; DOTLLM_VK_Q4K_ROWS_MULTI (default 4) for 2..8 columns. The kernel also serves Q4_K decode (one column) so that a verify b... | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4K_ROWS_BYN` | Per-column-count NR for the Q4_K multi-column GEMV, measured on the 8060S at the Qwen3.8-27B FFN shapes (50 MB weights). #1511: with the software-pipelined 16-byte-load shader NR 4 is best for 2..12 columns (245 us at 2..4 colu... | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4K_ROWS_DECODE` | Q4_K decode GEMV on MatMulQ4KMulti at one column (#1220) unless DOTLLM_VK_Q4K_ROWS_DECODE=0, which restores the coalesced 1-row kernel (A/B; MTP verify columns are then only ULP-close to decode). | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4K_ROWS_MULTI` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4K_ROWS_WIDE` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q4_0_GEMM_LEGACY` | Environment variable that disables this kernel so the caller declines the native path (A/B). | `MatMulQ4_0GemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q4_0_NATIVE` | #1018 A/B switch: DOTLLM_VK_Q4_0_NATIVE=0 disables every native Q4_0 Vulkan kernel (dense and routed banks): Q4_0 expert banks widen to F32 as before and the LFM2 loader falls back to its exact Q4_0 -> Q8_0 transcode. | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VK_Q4_0_PREFILL_Q8_1` | Load-time default of Q4_0PrefillQ8_1: 1 = on, 0 = off (kernels not even created), unset = on. | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_Q4_0_V2` | Default shader generation: V2 (a lane owns a whole 18-byte block; ~3x fewer load instructions per weight byte, issue #1018) unless DOTLLM_VK_Q4_0_V2=0 selects the V1 4-lanes-per-block shader (A/B). | `MatMulQ4_0MmvqKernel.cs` |
| `DOTLLM_VK_Q4_1_PREFILL_Q8_1` | Load-time default of Q4_1PrefillQ8_1: 1 = on, 0 = off (kernels not even created), unset = auto. | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_Q4_GEMM_SMALL` | DOTLLM_VK_Q4_GEMM_SMALL=0 keeps every shape on the 128x128 tile (A/B). | `MatMulQ4_0GemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q4_K_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ4KGemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q4_K_GEMV_LEGACY` | Environment variable that selects the block-per-lane GEMV instead of the coalesced one (A/B). | `MatMulQ4KGemvF32Kernel.cs` |
| `DOTLLM_VK_Q5_0_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ5_0GemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q5_0_GEMV_LEGACY` | Environment variable that selects the block-per-lane GEMV instead of the coalesced one (A/B). | `MatMulQ5_0GemvF32Kernel.cs` |
| `DOTLLM_VK_Q5_1_FAST` | DOTLLM_VK_Q5_1_FAST=0 keeps dense Q5_1 on the #1600 F32-in GEMV / GEMM (A/B against the #1613 MMVQ / MMQ / coopmat arms). | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_Q5_K_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ5KGemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q5_K_GEMV_LEGACY` | Environment variable that selects the block-per-lane GEMV instead of the coalesced one (A/B). | `MatMulQ5KGemvF32Kernel.cs` |
| `DOTLLM_VK_Q6K_ROWS` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q6K_ROWS_DECODE` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q6K_ROWS_MULTI` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q6K_ROWS_WIDE` | #1220 (supersedes #1193's Q4_K_R4): NR rows per workgroup share the activation loads; deterministic, and also the qwen35 decode GEMV. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q6_K_GEMM_LEGACY` | Environment variable that disables this kernel so the tiled F32 GEMM is used (A/B). | `MatMulQ6KGemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q6_K_GEMV_LEGACY` | Environment variable that selects the block-per-lane GEMV instead of the coalesced one (A/B). | `MatMulQ6KGemvF32Kernel.cs` |
| `DOTLLM_VK_Q8_0_GEMM_LEGACY` | Environment variable that restores the pre-issue-#443 preference in SelectFor. | `MatMulQ8_0GemmCoopmatKernel.cs` |
| `DOTLLM_VK_Q8_0_GEMV_COALESCED` | Elements per Q8_0 block. | `MatMulQ8_0Kernel.cs` |
| `DOTLLM_VK_Q8_ROWS` | Output rows per workgroup of MatMulQ8Multi (#1220) for one column: DOTLLM_VK_Q8_ROWS (1..8), default 2; DOTLLM_VK_Q8_ROWS_MULTI (default 4) for 2..8 columns. | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q8_ROWS_DECODE` | Q8_0 decode GEMV on MatMulQ8Multi at one column for the dense qwen35 model (#1220) unless DOTLLM_VK_Q8_ROWS_DECODE=0 (A/B: the coalesced 1-row kernel, ULP-close to the verify columns). | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q8_ROWS_MULTI` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_Q8_ROWS_WIDE` | (not described in source) | `VulkanQwen3MoeHybridKernels.cs` |
| `DOTLLM_VK_QSA_CM` | Cooperative-matrix gather attention (#1356, qsa_attention_cm_f16kv): one 16-row matrix tile per GQA group, F16 K/V, head_dim 256. DOTLLM_VK_QSA_CM=0 at startup (or setting this) restores the scalar gather kernel for A/B. | `Qwen4ExpQsaKernels.cs` |
| `DOTLLM_VK_QWEN35MOE_MTP` | Uploads the CPU model's MTP block (issue #1188). Resident MoE only, and never with CPU-placed MoE layers: the head drafts every round, so a per-forward upload or a host round trip would cost more than it saves. Opt-in (DOTLLM_V... | `VulkanQwen3MoeHybridTransformerModel.Mtp.cs` |
| `DOTLLM_VK_QWEN4EXP_HOST_EMBED` | Gather token-embedding rows on the host from the mmap'd (quantised) table instead of keeping an F32 copy of the whole table on the device (2.4 GiB at the released vocabulary). On by default; DOTLLM_VK_QWEN4EXP_HOST_EMBED=0 rest... | `VulkanQwen4ExpTransformerModel.Loading.cs` |
| `DOTLLM_VK_QWEN4EXP_IDX_F16` | Store the indexer's raw (pre-pooling) keys as F16 (DOTLLM_VK_QWEN4EXP_IDX_F16=1): 2 of the ~6 KiB per token the indexer holds. The pooled block keys, which the scorer reads, stay F32. Off by default. | `VulkanQwen4ExpTransformerModel.Loading.cs` |
| `DOTLLM_VK_QWEN4EXP_KV_LAZY` | Grow the K/V and indexer caches on demand instead of allocating the whole capacity per sequence (issue #437). On by default; DOTLLM_VK_QWEN4EXP_KV_LAZY=0 restores eager allocation (A/B diagnostic). Output is bit-identical eithe... | `VulkanQwen4ExpTransformerModel.Loading.cs` |
| `DOTLLM_VK_SHARED_EXPERT_PACKED` | True when this layer carries what the absorbed-MLA path needs (#913): the llama.cpp split attn_k_b / attn_v_b pair as Q8_0 (the shaders read Q8_0 blocks directly) with the head layout the kernels are compiled for (kvLoraRank 51... | `VulkanWeights.cs` |
| `DOTLLM_VK_SHARED_OVERLAP` | Runtime switch for overlapping the 2..8-row shared-expert chain with the routed-expert chain (#1452); DOTLLM_VK_SHARED_OVERLAP=0 at startup disables (A/B and diagnostics). | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_SINK_FLASH` | Environment variable: 0 keeps sink prefill on the reference sink kernel (#1200 A/B). | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_SINK_SPLIT` | Runtime A/B switch for split-KV gpt-oss decode attention (default on; DOTLLM_VK_SINK_SPLIT=0 disables). | `VulkanTransformerModel.cs` |
| `DOTLLM_VK_SMALLROW_ATTN_MAX_ROWS` | Largest row count served by the small-row split-KV attention (MiMo-V2 and gpt-oss sink layers; default 4, hard cap VulkanSplitKvRowsAttentionKernel.MaxRows; DOTLLM_VK_SMALLROW_ATTN_MAX_ROWS; 0 = off). | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_F32_MAX_ROWS` | Largest row count served by the F32 multi-column GEMV (router, F32 projections; default 4; DOTLLM_VK_SMALLROW_F32_MAX_ROWS; 0 = off). | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_GEMV` | Runtime switch for the 2..8-row multi-column GEMVs (#876); DOTLLM_VK_SMALLROW_GEMV=0 at startup disables (A/B and diagnostics). | `VulkanQwen3HybridDenseTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_K_GEMV` | DOTLLM_VK_SMALLROW_K_GEMV=0 keeps Q4_K / Q5_K / IQ4_XS 2..16-row forwards on the MMQ / GEMM path (#917 A/B). Runtime switch. | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_K_MAX_ROWS` | Largest row count served by the K-quant multi-column GEMVs here (#917; DOTLLM_VK_SMALLROW_K_MAX_ROWS, default 16, hard cap 16; 0 = off). | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_LEGACYQ` | DOTLLM_VK_SMALLROW_LEGACYQ=0 keeps Q8_0 5..16-row and Q4_0 / Q5_0 2..16-row forwards on the MMQ / GEMM path (#1420 A/B). Runtime switch. | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_Q6K_MAX_ROWS` | Largest row count served by the Q6_K multi-column GEMV (default 4; DOTLLM_VK_SMALLROW_Q6K_MAX_ROWS; 0 restores the MMQ / GEMM path). Runtime A/B switch. | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS` | Smallest row count that takes the two-output-rows-per-workgroup Q6_K kernel (default 3: at 2 columns the single-row kernel is as fast; 3..4 columns are load-issue bound there; DOTLLM_VK_SMALLROW_Q6K_R2_MIN_ROWS; 99 = never). | `VulkanQwen3HybridDenseTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_Q8WIDE` | True when rows beyond the Q8_1 MMVQ multi kernel's cap (and up to 16) run the FP32-activation Q8_0 multi GEMV (#1420). | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_Q8_MAX_ROWS` | Largest row count served by the Q8_0 MMVQ multi-column GEMV (lm_head, Q8_0 projections; default 4; DOTLLM_VK_SMALLROW_Q8_MAX_ROWS; 0 = off). | `VulkanTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SMALLROW_WIDE` | DOTLLM_VK_SMALLROW_WIDE=0 caps the multi-column GEMVs at 8 rows again (#1365: A/B and diagnostics; 9..16 rows then take the GEMM path with its 340-570 ms fixed cost). Pair it with DOTLLM_VK_PQ2_0_MULTICOL_MAX_N=8 for PQ2_0 weig... | `VulkanQwen3HybridDenseTransformerModel.SmallRow.cs` |
| `DOTLLM_VK_SPIN_WAIT` | Fence wait policy. A blocking vkWaitForFences sleeps the host thread, and on this class of machine (Strix Halo UMA, Windows/WDDM) the wake-up costs ~1.3 ms per forward: with a spin on vkGetFenceStatus instead, Tev1-4B decode we... | `VulkanDevice.cs` |
| `DOTLLM_VK_STAGE_TS` | GPU-timestamp stage mode (#885): DOTLLM_VK_STAGE_TS=1 or StageTimestamps=true. Instead of submit+wait per stage (which pays ~50 us of fence round trip each and distorts a 2000-dispatch decode), each stage writes a BOTTOM_OF_PIP... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VK_SUBMIT_BUDGET_MS` | The budget from DOTLLM_VK_SUBMIT_BUDGET_MS when it parses to a positive number, else DefaultBudgetSeconds. A new budget. | `VulkanPrefillSubmitBudget.cs` |
| `DOTLLM_VULKAN_533_GATE` | #533 vendor policy. The defect - coopmat P.V not reproducing the same accumulator for the 0 * v contributions of masked / padding columns - is an AMD implementation property: an RTX 3060 is exactly 0-differing at every KV lengt... | `VulkanFlashAttentionCoopmatKernel.cs` |
| `DOTLLM_VULKAN_ATTN_PREFILL` | Environment variable: ref keeps ordinary prefill on the per-head flash kernel (#1548 A/B and oracle). | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_ATTN_QPOS` | Environment variable: ref forces the reference per-query-position kernel (#1540 A/B and oracle). | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_BANK_PREALLOC` | (not described in source) | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VULKAN_DECODE_PROFILE` | Legacy COMPUTE->TRANSFER blanket barrier - see BarrierComputeToCompute. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DECODE_PROFILE_GPU` | DOTLLM_VULKAN_DECODE_PROFILE_GPU=1 additionally writes a BOTTOM_OF_PIPE timestamp at every op-category boundary (~300 queries/token on a 30-layer model) and reports per-category GPU time. Since every dispatch is barrier-seriali... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DECODE_SPLIT_LAYER` | Default 8 (see sweep above). Was briefly 0 during #148: intermittent DEVICE_LOSTs turned out to be cumulative gfx1151 driver-state degradation after heavy load/dispose churn (24/24 clean across the splitxbarrier matrix once the... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DEVICE_GUMBEL` | Env switch: DOTLLM_VULKAN_DEVICE_GUMBEL=0 turns the device draw off (the host path then reads the logits back). | `VulkanQwen3HybridDenseTransformerModel.Gumbel.cs` |
| `DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION` | Env-var opt-out for the cooperative-matrix Flash-Attention prefill kernel (issue #149). Set DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1 to force prefill attention onto the scalar VulkanFlashAttentionF32Kernel - the A/B baseline a... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT` | Set DOTLLM_VULKAN_DISABLE_EMBED_GPU_DEQUANT=1 to force the legacy CPU dequant of the token-embed table (streamed through staging). Used by the embedding-row parity test as the discriminating baseline, and as an operational esca... | `VulkanWeights.cs` |
| `DOTLLM_VULKAN_DISABLE_EMBED_RESIDENT` | True when a format table with rows of hidden elements can be gathered packed: the format has a gather shader, it is present in spvDir, and rows are whole blocks. DOTLLM_VULKAN_DISABLE_EMBED_RESIDENT=1 answers false (the widened... | `VulkanWeights.cs` |
| `DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION` | Env-var opt-out for the Flash-Attention prefill path. Set DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION=1 to force every dispatch onto the legacy per-token AttentionF32Kernel. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_FUSED_LORA_DELTA` | Fused fast path: two dispatches (B-reduce + A-accumulate-in-place) in place of the original four. Gated by SPV availability + rank cap. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_FUSED_RMSNORM_MATMUL` | Opt-out switch - default is fused-on. On RDNA3.5 the fused path wins by ~3-5% in median paired-run min latency and is more resilient to dispatch-time contention. Set the env var to "1" to bypass on hardware where fusion regress... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_FUSED_RMSNORM_QUANT` | Fused rmsnorm + Q8_1 activation quantize for the decode MMVQ shared groups (issue #145): removes one dispatch + one barrier per group (2/layer). Only useful when the Q8_0 shared-group MMVQ path is live; requires subgroup arithm... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_FUSED_SWIGLU_QUANT` | Fused SwiGLU + Q8_1 activation quantize for the down_proj decode MMVQ GEMV (issue #71): removes one dispatch + one barrier per dense-FFN decode layer (SwiGLU -> quantize collapses into one dispatch; the barrier before the MMVQ... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_I2S_COOPMAT` | Env opt-out for the coopmat + wave32 I2_S GEMM (DOTLLM_VULKAN_DISABLE_I2S_COOPMAT=1) - forces the bit-exact register-blocked kernel on every device. Mirrors the DOTLLM_VULKAN_DISABLE_WAVE32 convention. | `MatMulI2SGemmF32Kernel.cs` |
| `DOTLLM_VULKAN_DISABLE_IQ4XS_COOPMAT` | (not described in source) | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_IQ_COOPMAT` | Environment variable that disables these kernels so callers fall back to the F16 scratch / scalar IQ GEMMs (A/B). | `MatMulIqGemmCoopmatKernel.cs` |
| `DOTLLM_VULKAN_DISABLE_IQ_F16_PREFILL` | Prefill matmul for the IQ1/IQ2/IQ3 formats and NVFP4 (#1060): dequantise the weight matrix once into an F16 scratch buffer, then run the F16 cooperative-matrix GEMM (issue #621). NVFP4's per-tensor X.scale multiplies the F32 ou... | `IqF16PrefillMatmul.cs` |
| `DOTLLM_VULKAN_DISABLE_MMQ` | (not described in source) | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_MMVQ` | Env-var opt-out for the dp4a MMVQ decode path (issue #46). Set DOTLLM_VULKAN_DISABLE_MMVQ=1 to force decode Q8_0 GEMV onto the F32-in MatMulQ8_0Kernel (e.g. to A/B benchmark or to work around a driver integer-dot bug). When uns... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_MOE_MMVQ` | Env-var opt-out for the indexed MoE MMVQ decode path (issue #137). Set DOTLLM_VULKAN_DISABLE_MOE_MMVQ=1 to force the S==1 expert matmuls onto the scalar per-cell indexed kernels - used for same-session A/B benchmarking of the c... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q2K_COOPMAT` | Prefill length from which the blocked coopmat K-quant GEMM replaces dp4a MMQ; lives in VulkanDensePrefillGemmPolicy (#1149). | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q3K_COOPMAT` | The crossovers below are the shared table in VulkanDensePrefillGemmPolicy (#1149): the hybrid-dense host selects from the same one. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q4K_COOPMAT` | (not described in source) | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q5K_COOPMAT` | The crossovers below are the shared table in VulkanDensePrefillGemmPolicy (#1149): the hybrid-dense host selects from the same one. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q5_0_COOPMAT` | The crossovers below are the shared table in VulkanDensePrefillGemmPolicy (#1149): the hybrid-dense host selects from the same one. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_Q6K_COOPMAT` | Prefill length from which the blocked coopmat K-quant GEMM replaces dp4a MMQ; lives in VulkanDensePrefillGemmPolicy (#1149). | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_SPLIT_DECODE` | Env-var opt-out for the split-KV (Flash-Decoding) decode path (issues #345/#346). Set DOTLLM_VULKAN_DISABLE_SPLIT_DECODE=1 to force every decode dispatch onto the legacy per-token AttentionF32Kernel - used for same-session A/B... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_DISABLE_WAVE32` | Env-var opt-out for the targeted wave32 MMVQ pin (issue #54). Set DOTLLM_VULKAN_DISABLE_WAVE32=1 to force the default (unset) subgroup-size path on the decode MMVQ pipelines - e.g. to A/B benchmark wave32 vs the driver default,... | `Wave32SubgroupControl.cs` |
| `DOTLLM_VULKAN_DS_HEAP_ALIGN_MIB` | D3D12 heap alignment: 64 KiB, or 4 MiB with DOTLLM_VULKAN_DS_HEAP_ALIGN_MIB=4 (the only two values D3D12 accepts). | `DirectWeightStore.cs` |
| `DOTLLM_VULKAN_DS_REQUEST_MIB` | Size of one DirectStorage read request (MiB), DOTLLM_VULKAN_DS_REQUEST_MIB, default 16. | `DirectWeightStore.cs` |
| `DOTLLM_VULKAN_DS_STAGING_MIB` | DirectStorage staging ring size (MiB), DOTLLM_VULKAN_DS_STAGING_MIB, default 64. | `DirectWeightStore.cs` |
| `DOTLLM_VULKAN_EMBED_CHUNK_LIMIT_BYTES` | Diagnostic: DOTLLM_VULKAN_EMBED_CHUNK_LIMIT_BYTES forces a smaller chunk limit on a real model. | `VulkanChunkedRowTable.cs` |
| `DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ` | Opt-IN gate for the IQ2_XXS dp4a MMQ prefill kernel (issue #344). It is OFF by default: on the gfx1151 iGPU the AMD driver miscompiles this grid-codebook shader into a GPU fault once a single submit's total work is large (verif... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_FA_COOPMAT_HD256` | Env kill switch for the hd256 pipeline (DOTLLM_VULKAN_FA_COOPMAT_HD256=0). | `VulkanFlashAttentionCoopmatKernel.cs` |
| `DOTLLM_VULKAN_FA_COOPMAT_HD64` | Issue #378: set DOTLLM_VULKAN_FA_COOPMAT_HD64=0 to force every shape onto the base 128-dim shader (as if the hd64 SPV were absent). | `VulkanFlashAttentionCoopmatKernel.cs` |
| `DOTLLM_VULKAN_FA_GQA` | Environment variable: 0 disables the kernel (hosts fall back to the per-head flash kernels). | `VulkanFlashAttentionGqaKernel.cs` |
| `DOTLLM_VULKAN_FA_GQA_HD128` | (not described in source) | `VulkanFlashAttentionGqaKernel.cs` |
| `DOTLLM_VULKAN_FA_GQA_QLO` | Environment variable: 1 restores the cooperative-matrix QK^T's Q lo-half term (#1148). Off by default: Q is rounded to F16, attention is 1.67x faster at depth on Strix, and the #1109 study measured KL at the F32 noise floor to... | `VulkanFlashAttentionGqaKernel.cs` |
| `DOTLLM_VULKAN_FA_GQA_SCALAR` | Environment variable: scalar forces the scalar implementation even on a cooperative-matrix device (A/B). | `VulkanFlashAttentionGqaKernel.cs` |
| `DOTLLM_VULKAN_FORCE_SHARED_REDUCE` | Env-var override: force the shared-memory path even when the driver advertises VK_SUBGROUP_FEATURE_ARITHMETIC_BIT. Primarily a test hook - lets the same binary exercise both code paths on a capable GPU. | `RmsNormF32Kernel.cs` |
| `DOTLLM_VULKAN_HAZARD_BARRIERS` |  Hazard-scoped barriers (issue #144)  Legacy behaviour emitted a full-pipeline barrier after essentially every dispatch (~573/token on SmolLM decode), serialising the GPU so wall time = sum of op times. The hazard tracker repla... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_HAZARD_DEBUG` | Diagnostic (issue #144): DOTLLM_VULKAN_HAZARD_DEBUG=1 prints the first guards of the process with the buffer that forced each barrier. | `VulkanHazardTracker.cs` |
| `DOTLLM_VULKAN_HAZARD_VALIDATE` | Debug aid: keep the tracker armed but force a barrier at every guard - legacy-equivalent ordering through the tracked code path. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_HOST_IMPORT_WRITABLE` | DOTLLM_VULKAN_HOST_IMPORT_WRITABLE=1: import read-only GGUF ranges through a shared writable alias view (#1190). Read per call. | `VulkanWeightImportPolicy.cs` |
| `DOTLLM_VULKAN_HYBRID_PROFILE` | (not described in source) | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ` | (not described in source) | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VULKAN_HYBRID_PROFILE_OUT` | (not described in source) | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VULKAN_HYBRID_PROFILE_SPLIT` | (not described in source) | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VULKAN_I2S_COOPMAT` | Env opt-in (DOTLLM_VULKAN_I2S_COOPMAT=1) extending the coopmat selection to non-AMD devices, for evaluating it on hardware it has not been measured on. | `MatMulI2SGemmF32Kernel.cs` |
| `DOTLLM_VULKAN_IQ_F16_PREFILL_MIN_TOKENS` | Prefill matmul for the IQ1/IQ2/IQ3 formats and NVFP4 (#1060): dequantise the weight matrix once into an F16 scratch buffer, then run the F16 cooperative-matrix GEMM (issue #621). NVFP4's per-tensor X.scale multiplies the F32 ou... | `IqF16PrefillMatmul.cs` |
| `DOTLLM_VULKAN_KQUANT_MMVQ` | Env override for the K-quant (Q4_K/Q5_K/Q6_K) dp4a MMVQ decode kernels (#801): 1 forces them on, 0 forces them off, unset = UseKQuantMmvq's device default. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_LEGACY_BARRIERS` |  Hazard-scoped barriers (issue #144)  Legacy behaviour emitted a full-pipeline barrier after essentially every dispatch (~573/token on SmolLM decode), serialising the GPU so wall time = sum of op times. The hazard tracker repla... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_MEM_DIAG` | DOTLLM_VULKAN_MEM_DIAG=1 dumps the physical-device heap/type table to stderr when a map/alloc hits a transient memory failure - the issue #146 diagnosis aid. | `VulkanDevice.cs` |
| `DOTLLM_VULKAN_MEM_RETRIES` | Number of retries after a transient memory failure (VK_ERROR_MEMORY_MAP_FAILED / host / device OOM) on vkMapMemory and vkAllocateMemory. Under memory pressure on UMA boxes (issue #146: back-to-back heavy runs on Strix Halo, the... | `VulkanDevice.cs` |
| `DOTLLM_VULKAN_MEM_TRACE` | #890: the shared-expert gate/up/down F32 host copies (LoadMoe.SharedF32) existed only to be uploaded as F16; every copy is drained above. They are ~3.3 GiB of private memory on the real UD-Q4_K_XL file, and the OS caps a proces... | `VulkanDevice.cs` |
| `DOTLLM_VULKAN_MMVQ_DUAL` | Env-var OPT-IN for the fused dual-output MMVQ dispatch (issue #71). Default is OFF - this is a measured, documented NON-win, kept opt-in rather than removed. Profiling on RTX 3060 (SmolLM-135M Q8_0 decode, DOTLLM_VULKAN_DECODE_... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_MMVQ_NO_SHARE` | Env-var opt-out for the shared-activation-quant optimisation on the MMVQ decode path. Set DOTLLM_VULKAN_MMVQ_NO_SHARE=1 to quantize the activation per projection (the original per-call behaviour) instead of once per same-input... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_MOE_PREFILL_PROFILE` | Coarse env-gated prefill profiler (DOTLLM_VULKAN_MOE_PREFILL_PROFILE=1), mirroring VulkanTransformerModel's per-category profiler but at per-layer-submission granularity (2a token-mixing vs 2b MoE FFN) since this model already... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VULKAN_MOE_STAGE_PROFILE` | Per-STAGE split-submit timing inside the routed-MoE prefill layer (DOTLLM_VULKAN_MOE_STAGE_PROFILE=1): after each stage the command buffer is submitted and waited, so the stage's wall time is attributed to it (sync cost ~50 us... | `VulkanQwen3MoeHybridTransformerModel.cs` |
| `DOTLLM_VULKAN_MTP_KV` | Environment variable selecting the head KV storage: f16 (default) or f32 (reference arm only). | `VulkanMtpState.cs` |
| `DOTLLM_VULKAN_MTP_PROFILE` |  #471: MTP draft-step attribution  A draft step (ForwardMtp with logits) is profiled on its own switch, so a trunk profile is not drowned in one extra report per drafted token, and vice versa. | `VulkanQwen3HybridDenseTransformerModel.Profiling.cs` |
| `DOTLLM_VULKAN_PARALLEL_UPLOAD` | Uploads the routed banks of every non-null entry of layers resident (raw quant), eagerly and in layer order, through one multi-threaded staging buffer, with the next layers' banks allocated ahead on background threads (#874). S... | `VulkanQwen3MoeMoeUpload.cs` |
| `DOTLLM_VULKAN_PREFILL_PROFILE` | issue #139: env-gated prefill per-op-category profiler (DOTLLM_VULKAN_PREFILL_PROFILE=1). Splits the single per-forward command buffer at category boundaries (SubmitAndWait + re-Begin) so each bucket's wall time ~ its GPU time... | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_PREFILL_PROFILE_OUT` | (not described in source) | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_Q4K_COOPMAT_MIN_TOKENS` | Env var moving the K-quant coopmat crossover (token count from which the coopmat GEMM replaces the dp4a MMQ). | `VulkanDensePrefillGemmPolicy.cs` |
| `DOTLLM_VULKAN_Q8_0_COOPMAT_MIN_TOKENS` | Row count from which Q8_0PrefillKernel.Auto prefers the coopmat GEMM on a qualifying device (DOTLLM_VULKAN_Q8_0_COOPMAT_MIN_TOKENS, default 128 like the K-quants). | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_Q8_0_PREFILL` | Startup override for Q8_0Prefill: auto (default), mmq or coopmat. | `VulkanTransformerModel.cs` |
| `DOTLLM_VULKAN_QUIET_ATTENTION_FALLBACK` | Set to 1 to suppress the stderr lines (the reported set is still recorded). | `VulkanAttentionFallbackDiagnostics.cs` |
| `DOTLLM_VULKAN_READAHEAD_MB` | Window in bytes (DOTLLM_VULKAN_READAHEAD_MB, default 1024; 0 disables). | `VulkanWeightReadAhead.cs` |
| `DOTLLM_VULKAN_READAHEAD_THREADS` | Tensors smaller than this are not worth the bookkeeping. | `VulkanWeightReadAhead.cs` |
| `DOTLLM_VULKAN_RELEASE_SOURCE` | Source-page release (DOTLLM_VULKAN_RELEASE_SOURCE=0 disables). | `VulkanWeightReadAhead.cs` |
| `DOTLLM_VULKAN_SPLITKV_GQA` | Override for the GQA-grouped pass: DOTLLM_VULKAN_SPLITKV_GQA=1 takes it on every host, 0 on none; unset leaves it to the host (EnableGqaByDefault). | `VulkanSplitKvAttentionKernel.cs` |
| `DOTLLM_VULKAN_SPLITKV_GQA2` | Largest head_dim the v2 grouped pass serves. | `VulkanSplitKvAttentionKernel.cs` |
| `DOTLLM_VULKAN_SPLIT_MIN_KV` | Split heuristic. A split is only taken when it produces >= 2 splits; at 1 split the caller falls back to the single-pass kernel (so short context is bit-identical to the legacy path - the regression guard). S = clamp(TargetWork... | `VulkanSplitKvAttentionKernel.cs` |
| `DOTLLM_VULKAN_SPLIT_TARGET_WG` | Split heuristic. A split is only taken when it produces >= 2 splits; at 1 split the caller falls back to the single-pass kernel (so short context is bit-identical to the legacy path - the regression guard). S = clamp(TargetWork... | `VulkanSplitKvAttentionKernel.cs` |
| `DOTLLM_VULKAN_UPLOAD_SLOTS` | Number of staging slots when built by CreateParallel (DOTLLM_VULKAN_UPLOAD_SLOTS, default 12, 3..32). | `VulkanStagingBuffer.cs` |
| `DOTLLM_VULKAN_UPLOAD_SLOT_MB` | Slot size when built by CreateParallel (DOTLLM_VULKAN_UPLOAD_SLOT_MB, default 32). | `VulkanStagingBuffer.cs` |
| `DOTLLM_VULKAN_USE_COOPMAT_ATTENTION` | Env-var opt-in: use the coopmat attention pipeline. When set to 1, the coopmat dispatch is preferred over subgroup/shared. Default is off because on AMD RDNA3.5 iGPU (the reference hardware measured at landing time) coopmat att... | `AttentionF32Kernel.cs` |
| `DOTLLM_VULKAN_WEIGHT_ARENA` | Tensors at least this large get their own allocation. | `VulkanWeightArena.cs` |

<!-- END env-appendix -->
