# HIP integration testing handbook

This handbook tests the BlueCurve-derived HIP and SafeTensors integration in
**`jamesburton/dotLLM`**. Run commands from a checkout of the integration's PR
branch or its merged target `dev`. The submission fork is `BlueCurveCorp/dotLLM`;
its clean `main` and the historical BlueCurve donor snapshot do not contain this
integration merely because they share the same repository URL. Read [HIP.md](HIP.md) for the supported subset and
[PRECISION.md, section 9](PRECISION.md#9-hip-dense-integration-rx-6900-xt) for the
recorded measurements. The independent review prompt is
[HIP_ADVERSARIAL_REVIEW.md](HIP_ADVERSARIAL_REVIEW.md).

All required harnesses are tracked. A fresh checkout does not need the original
developer's private `.docs/` files. Commands below create new output there;
models, corpora and reference logits belong in an external store.

## What each test establishes

| Environment / check | Establishes | Does not establish |
| --- | --- | --- |
| Managed tests without a GPU | Parsing, format validation, routing and error contracts selected by the filter | Native execution or direct transfers |
| Windows x64 HIP + DirectStorage | Native primitives, exact file-range loading, supported model parity and application behavior | Other GPUs, Linux, arbitrary architectures |
| Linux build + `hip-linux-direct-io/run.sh` | Compilation, native ABI and loader fault/lifetime behavior with real POSIX direct reads and a simulated HIP runtime | Physical AIS transfers or any GPU arithmetic |
| Linux x64 HIP + AIS on the actual model volume | The hardware tests and model runs actually executed there | Tests skipped by a Windows guard; untested formats or contexts |
| Context runner | Actual 1K/2K/4K/8K history and bounded numerical/memory comparisons on one pinned model/text | Full-corpus perplexity, general quality or throughput |

The recorded hardware qualification used Windows, RX 6900 XT (`gfx1030`, 16 GB),
HIP SDK 6.4.2 and driver 32.0.21045.16010. `gfx1100` was compiled but not executed.
The available Ubuntu/WSL environment compiled ROCm 7.2.4 kernels but could not
initialize a HIP GPU. **Linux GPU inference and physical AIS remain unqualified.**

## Prepare a reproducible run

1. Record `git rev-parse HEAD`, `git status --short`, and `git diff` before testing.
   Preserve unrelated work. Read `AGENTS.md` and `CLAUDE.md` before modifying code.
2. Install the .NET SDK selected by `global.json` and record `dotnet --info`.
   Projects target .NET 10; SDK roll-forward can select a later SDK, so record
   both SDK and runtime rather than assuming they are identical.
3. Choose a fresh result directory, for example `.docs/hip-testing/run-001`.
   Keep complete stdout, stderr, TRX, JSONL, command lines and exit codes.
4. Run one GPU qualification process at a time. Native tests use the serial
   `HipGpu` collection, but that does not coordinate independent processes.
5. Record the GPU ordinal, architecture, driver, HIP runtime/compiler versions,
   model hashes and the **loaded** `co/dense_f16.co` hash. Rebuild native sources
   explicitly after changing compiler or architecture; timestamps are insufficient.

Use full paths for external fixtures. The ordinary model resolver searches an
environment override, the shared `~/.dotllm/models/{org}/{repo}` store, legacy
locations and the HF cache. Explicit paths make a qualification report clearer.
Do not put model files in the checkout, even temporarily.

## Fixtures and independent references

These are the historical fixture pins. Each weight hash is SHA-256 of the named
file, not of its containing directory. Preserve the complete config/tokenizer
assets from the same revision for HF application tests.

| Fixture | Source revision / filename | Weight SHA-256 | Test override |
| --- | --- | --- | --- |
| `unsloth/SmolLM2-135M-Instruct-GGUF` | `SmolLM2-135M-Instruct-F16.gguf`; content pinned by hash | `5157ca60744d21631818364854ac8e4452e1b8022d2ab4c8a2f9cda2344afb30` | Context runner `--model` |
| `HuggingFaceTB/SmolLM2-135M-Instruct` | `12fd25f77366fa6b3b4b768ec3050bf629380bac`, `model.safetensors` | `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c` | `DOTLLM_TEST_HIP_BF16_MODEL` |
| `TheBloke/TinyLlama-1.1B-Chat-v1.0-AWQ` | `2e78382c2b2faaea451c75e11f8df72ddd2d1e39`, `model.safetensors` | `853cf974e9fd5a6f43d6084f371b90effb5f710fed18d5a7556ef4e3db84e5a2` | `DOTLLM_TEST_TINYLLAMA_AWQ` |
| `TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ` | `9d4580af0f21bccafd762dcc50d0c7bac6273584`, `model.safetensors` | `ceef7773c19f5cbda3e5f688f99e045e07a879e70c19ecdb5d069b27c2d45b37` | `DOTLLM_TEST_TINYLLAMA_GPTQ` |

Use existing fixtures in place where available. To acquire a missing HF fixture,
use `huggingface_hub.snapshot_download(repo_id=..., revision=...,
local_dir=...)` in a separate Python environment, with an external model-store
directory and `allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja"]`.
Check the weight hash before running tests. For the GGUF, obtain the exact file
from the named HF repository and require the hash above; a changed `main` file
is not an interchangeable fixture. Record acquisition URLs/revisions and hashes
of config/tokenizer files too.

### AWQ/GPTQ reference producer

Install CPU PyTorch, `transformers==4.51.3` and `safetensors` in an external Python
environment and retain `python -m pip freeze`. Run the tracked producer:

```text
python tests/scripts/generate_packed_hf_reference.py <external-AWQ-directory>
python tests/scripts/generate_packed_hf_reference.py <external-GPTQ-directory>
```

It writes `packed-reference.json` beside each checkpoint. It executes the pinned
AutoAWQ `88e4c76b20755db275574e6a03c83c84ba3bece5` / AutoGPTQ
`9f7d37072917ab3a7545835f23e808294a542153` producer unpacking code and the HF eager
Llama implementation, with F32 reference storage/accumulation. This expansion is
an oracle-only exception, not the inference implementation. No model remote code
is enabled. The script fetches the producer source; arrange network access or
retain that immutable source for offline reproduction.

The tests verify the model hash, Transformers version and producer source hash
(`65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe` for AWQ;
`5a6a483784adc7d9ee55e62f6dc4fdaf3a642f56796d2a6b8afea0b2364d751d` for GPTQ).
They score six full-vocabulary rows and substitute wrong qzero file ranges to
prove the real data detects a layout/loading error.

### Optional genuine-HF F16 application fixture

`HipSafetensorsApplicationTests.RealHfConfigRoute_MatchesCpu_AndWrongRotaryLayoutIsDetected`
uses `DOTLLM_TEST_HIP_HF_MODEL`. It needs genuine HF Q/K row order, not renamed
GGUF tensors. A cast-only offline derivative of the pinned original BF16 fixture
is suitable. Save this reference-fixture producer in an external scratch file
and invoke `python <producer.py> <original-directory> <new-external-directory>`:

```python
import hashlib, json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, dst = map(Path, sys.argv[1:])
weights = src / "model.safetensors"
source_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
assert source_hash == "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c"
dst.mkdir(parents=True, exist_ok=False)  # never overwrite an existing checkpoint
for item in src.iterdir():
    if item.is_file() and item.suffix in (".json", ".model", ".jinja"):
        shutil.copy2(item, dst / item.name)
state = load_file(str(weights), device="cpu")
assert all(value.dtype == torch.bfloat16 for value in state.values())
save_file({key: value.half() for key, value in state.items()},
          str(dst / "model.safetensors"), metadata={"format": "pt"})
config = json.loads((dst / "config.json").read_text())
config["torch_dtype"] = "float16"
(dst / "config.json").write_text(json.dumps(config, indent=2))
(dst / "PROVENANCE.json").write_text(json.dumps({
    "source_revision": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "source_sha256": source_hash, "operation": "BF16 to F16 cast only; no row permutation",
    "torch": torch.__version__,
    "output_sha256": hashlib.sha256((dst / "model.safetensors").read_bytes()).hexdigest()
}, indent=2))
```

This is optional test-data preparation, never runtime weight conversion. Record
the generated hash and Python package versions. The historical derivative hash
was `b9f93ce41c270feb4eb9a8870f0b48b3fd2d761bd6af47f4d74fe35d991f9bf5`;
serializer metadata/version differences can change bytes, so a newly generated
derivative is not automatically that historical artifact.

### Context corpus

Download the raw UTF-8 [Project Gutenberg ebook 11 text](https://www.gutenberg.org/cache/epub/11/pg11.txt)
to an external path, such as `~/.dotllm/test-cache/corpora/gutenberg/alice-11.utf8.txt`.
Keep its original bytes. Required raw SHA-256:
`01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`.
The runner normalizes CRLF, starts at the second `CHAPTER I.` marker, tokenizes
the first 60,000 characters without BOS, and verifies the first 8,193 little-endian
int32 token IDs against
`7c043c12b9a873068520d7d7a4e3ab3b89a0f6ae9bf456bfbe53d3b61fda4f37`.
It verifies these pins even for a 1K run. If a download differs, recover the pinned
fixture or report a new dataset; do not silently change the pins.

## Windows build and required hardware tests

Use an x64 Visual Studio C++ developer PowerShell with the HIP SDK installed.
The paths below are examples; select the installed SDK and external model store.

```powershell
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.4'
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"
$env:DOTLLM_TEST_HIP_REQUIRED = '1'
$env:DOTLLM_HIP_ALLOW_Q8_KV = '1'
$env:DOTLLM_TEST_HIP_BF16_MODEL = 'E:\models\HuggingFaceTB\SmolLM2-135M-Instruct'
$env:DOTLLM_TEST_HIP_HF_MODEL = 'E:\models\HuggingFaceTB\SmolLM2-135M-Instruct-F16-derived'
$env:DOTLLM_TEST_TINYLLAMA_AWQ = 'E:\models\TheBloke\TinyLlama-1.1B-Chat-v1.0-AWQ'
$env:DOTLLM_TEST_TINYLLAMA_GPTQ = 'E:\models\TheBloke\TinyLlama-1.1B-Chat-v1.0-GPTQ'
$env:HIP_ARCHS = 'gfx1030 gfx1100'
pwsh native/hip/build.ps1
dotnet build src/DotLLM.Hip/DotLLM.Hip.csproj -c Release -p:RequireHipKernels=true
```

Stop if either native/build command fails. DirectStorage is a separate prerequisite.
For clean publish tests set `DOTLLM_DIRECTSTORAGE_DIR` to an installed directory
containing `dstorage.dll` and `dstoragecore.dll`; see [HIP.md](HIP.md#direct-disk-to-device-loading).
A development restore provides the private DirectStorage NuGet package, but
release artifacts must not redistribute those DLLs.

First run primitives without large model fixtures:

```powershell
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter 'FullyQualifiedName~HipDenseF16KernelTests|FullyQualifiedName~HipPackedKernelTests|FullyQualifiedName~HipSafeTensorsKernelTests|FullyQualifiedName~HipRmsNormKernelTests|FullyQualifiedName~HipRuntimeCandidateTests' --logger 'trx;LogFileName=primitives.trx' --results-directory .docs/hip-testing/run-001
```

Then run the complete HIP-specific selection, with all four fixture directories
above populated (including both `packed-reference.json` files):

```powershell
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter 'FullyQualifiedName~DotLLM.Tests.Unit.Hip.|FullyQualifiedName~HipRuntimeCandidateTests|FullyQualifiedName~HipDeviceResolutionTests|FullyQualifiedName~HipSafetensorsApplicationTests' --logger 'trx;LogFileName=hip.trx' --results-directory .docs/hip-testing/run-001
```

Audit every skip. `DOTLLM_TEST_HIP_REQUIRED=1` converts several missing-GPU guards
to failures; it **does not** enforce every fixture or DirectStorage prerequisite.
Windows should skip the two Linux-only methods in `HipLinuxDirectIoTests`.
The selected real BF16, AWQ, GPTQ and HF F16 tests must run, not skip.

```powershell
[xml]$trx = Get-Content .docs/hip-testing/run-001/hip.trx
$trx.TestRun.ResultSummary.Counters
$trx.TestRun.Results.UnitTestResult | Where-Object outcome -ne 'Passed' |
    Select-Object testName, outcome
```

For the surrounding integration, run the bounded managed/application selection:

```text
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~Models.SafeTensors|FullyQualifiedName~Tokenizers.Hf.|FullyQualifiedName~BpeTokenizerTests|FullyQualifiedName~PreTokenizerPolicyTests|FullyQualifiedName~HfChatTemplateCliTests|FullyQualifiedName~DotLLM.Tests.Unit.Cli.Device|FullyQualifiedName~DotLLM.Tests.Unit.Server.Device|FullyQualifiedName~CliHelpTests|FullyQualifiedName~SafetensorsApplicationTests|FullyQualifiedName~SafetensorsRoutingTests|FullyQualifiedName~ServerStateResidencyTests|FullyQualifiedName~ModelResidencyManagerTests|FullyQualifiedName~ModelMetaHealthTests|FullyQualifiedName~TextGeneratorPrefillChunk|FullyQualifiedName~CudaQuantExpansionGateTests" --logger "trx;LogFileName=integration.trx" --results-directory .docs/hip-testing/run-001
```

With the F16 HF fixture, also set `DOTLLM_TEST_SMOLLM2_TOKENIZER_JSON` to its
`tokenizer.json` and `DOTLLM_TEST_HF_TOKENIZER_REQUIRED=1` to require that tokenizer
comparison. Other external tokenizer/residency fixtures may be absent; list them
individually. The historical combined selection was 734 passed / 27 skipped /
0 failed, not the entire repository suite and not a count to hardcode after changes.

## Long-context quality

HIP uses F16 KV by default. Selecting Q8 for **either** plane requires the exact
process environment value `DOTLLM_HIP_ALLOW_Q8_KV=1`, plus the existing cache
selection, for example `--cache-type-k q8_0 --cache-type-v q8_0`. Setting the
variable alone does not select Q8. It acknowledges the measured precision loss;
it does not improve the numerical results. This gate does not affect Q8 model
weights or other backends.

Test rejection with the variable unset, then acceptance with `1`, for Q8/F16,
F16/Q8 and Q8/Q8 through `run`, `chat`, `serve` and library cache creation. F16/F16
must continue to work without it. Use the automated gate tests for invalid values
and early rejection. An unacknowledged Q8 request must explain the precision loss
and how to opt in.

The tracked portable runner builds the **current checkout**, not a frozen private
DLL. Start with 1K, then use a fresh result directory for 8K. These examples use
PowerShell variables pointing to external fixtures:

```powershell
$model = 'E:\models\unsloth\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-F16.gguf'
$corpus = 'E:\corpora\alice-11.utf8.txt'
$env:DOTLLM_HIP_ALLOW_Q8_KV = '1'
dotnet run --project tests/scripts/hip-context-qualification/HipContextQualification.csproj -c Release --artifacts-path .docs/hip-testing/build -- --model $model --corpus $corpus --output .docs/hip-testing/context-1k --max-context 1024
$contextExit = $LASTEXITCODE
dotnet run --project tests/scripts/hip-context-qualification/HipContextQualification.csproj -c Release --artifacts-path .docs/hip-testing/build -- validate --results .docs/hip-testing/context-1k/results.jsonl
```

Repeat with `--max-context 8192 --output .docs/hip-testing/context-8k`. The supported
checkpoints are 1,024 / 2,048 / 4,096 / 8,192; the last 128 predictions at each
included checkpoint are scored. Independent CPU F32, CPU Q8, HIP F16 and HIP Q8
histories consume every input, with 32-token forwards outside scoring windows.
An 8K run scores 512 rows per comparison, including position 8,191 predicting
target 8,192. Merely reserving an 8K cache is insufficient.

The runner also supports `--cpu-only` with a separate output directory. This
explicitly skips HIP and is useful for corpus/oracle validation on a build VM.
It is not GPU qualification. The runner never enables the Q8 opt-in itself.

| Exit | Meaning |
| ---: | --- |
| `0` | Completed; all original numerical gates passed |
| `10` | Completed, but at least one original numerical gate failed; inspect which comparison failed |
| `2` | Runtime, input provenance, artifact or harness failure; not the expected Q8 result |

The original gates are **maximum relative L2 < .02 and KL < .002**, for each
scored comparison/window. Recorded HIP F16 results pass; CPU/HIP Q8 results fail.
An exit of 10 is expected when reproducing that Q8 result, but remains a quality
failure. Do not accept an F16 failure merely because the overall exit is also 10.
The format tests above use their separate short-row KL < .001 gate.

At 8K the historical HIP Q8 versus CPU F32 maximum L2 was .050690 and KL .0038931;
F16 was .006174 / .00005271. Q8 had 504/512 top-1 matches across all windows and
bounded combined PPL increased about .1553%. Independent CPU Q8 also failed the
gates; this establishes quantization loss without isolating a HIP-specific defect.
Small average PPL changes do not negate worst-row failures or establish general quality.

Inspect the emitted provenance, window metrics, negative-control and lifecycle
records. Require the vocabulary-rotation control to fail, and run `validate` to
check artifact geometry and demonstrate rejection of a shortened-context mutation.
Keep `results.jsonl` beside its companion `tokens.json` when copying artifacts
for standalone validation.
At 8K, each HIP cache should own 512 token pages: historical F16/Q8 totals were
188,989,440 / 100,515,840 bytes, including pointer tables. Empty/reset caches own
zero bytes; model disposal releases owned weights/compute allocations. Keep
payload, aligned allocations, compute scratch, caller-owned logits and driver-wide
memory deltas separate. Driver allocator retention is not an owned-memory counter.

The historical 8K run used `e2de144`; a later native-object replay checked 1K only
and matched aggregate metrics, not retained raw-logit bytes. A new 8K run of this
portable runner qualifies the current artifacts only when it actually completes.

## Linux: build checks first, then actual hardware qualification

A standard Hyper-V/WSL build environment is useful for compilation and CPU tests.
Nested virtualization does not supply an AMD GPU or an NVMe-to-GPU path. See
[HIP.md](HIP.md#hyper-v-build-vm) for the VM scope and vendor compatibility links.

For Linux x64 with .NET, a C++ compiler and ROCm headers/compiler:

```bash
export PATH=/opt/rocm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
HIP_ARCHS="gfx1030 gfx1100" bash native/hip/build.sh
dotnet build src/DotLLM.Hip/DotLLM.Hip.csproj -c Release -p:RequireHipKernels=true
ROCM_PATH=/opt/rocm DOTNET=dotnet bash tests/scripts/hip-linux-direct-io/run.sh
```

A bounded managed selection also runs without a GPU. Leave the hardware-required
flag unset for this stage and audit optional real-fixture skips:

```bash
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release \
  --filter 'FullyQualifiedName~Models.SafeTensors|FullyQualifiedName~HipRuntimeCandidateTests|FullyQualifiedName~HipDeviceResolutionTests|FullyQualifiedName~SafetensorsRoutingTests|FullyQualifiedName~CliHelpTests' \
  --logger 'trx;LogFileName=linux-managed.trx' --results-directory .docs/hip-testing/linux
```

The last command builds a private `libamdhip64.so` test double and runs the
production Linux loader with real libc `O_DIRECT` reads. Its 54 checks cover exact
offsets/tails, >32 MiB chunking, independent lifetime, ABI and injected failures.
Outputs are under `.docs/linux-hip-qualification`; use a fresh checkout/output
capture to avoid confusing runs. It sets the simulated runtime path only for
the probe child process. **Never add its `stub` directory to the environment for
real hardware tests.** Compilation and this probe are not AIS/GPU evidence.

For actual GPU qualification, use native Linux x64 or a separately validated
passthrough system. A successful `rocminfo` must enumerate the intended GPU;
confirm its `gfx` architecture is in the rebuilt objects. The loader additionally
needs `hipAmdFileRead`, AIS-capable amdgpu/KFD, kernel PCI P2PDMA and a compatible
model volume. It deliberately has no host-bounce fallback.

Have the system administrator run AMD's diagnostic, without changing drivers,
mounts or GPU assignment as part of the test:

```bash
rocminfo
sudo /opt/rocm/bin/ais-check --verbose
findmnt -T /path/on/ais-volume/model.gguf
```

AMD documents that `ais-check` needs root for accurate information. Its exit zero
means at least one volume qualifies; verify **the model volume's** `HIPFILE=yes`
row and all four stack checks, not just the process exit. The documented fastpath
uses a physical GPU function and direct local NVMe with XFS or ext4 ordered mode.
An ordinary VHDX/network/LVM/encrypted volume is not assumed suitable.
See [AMD's compatibility diagnostic](https://rocm.docs.amd.com/projects/hipFile/en/latest/how-to/checking-system-compatibility.html).

First execute the real direct-read test on that same volume. It creates and removes
a temporary 32 MiB byte fixture there, loads an unaligned range and checks the
bytes after loader disposal:

```bash
export DOTLLM_TEST_HIP_REQUIRED=1
export DOTLLM_TEST_HIP_LINUX_DIR=/path/on/ais-volume/test-scratch
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release \
  --filter FullyQualifiedName~LinuxDirectRead_ExactTailAndIndependentBufferLifetime \
  --logger 'trx;LogFileName=linux-direct.trx' --results-directory .docs/hip-testing/linux
```

Create that writable directory in advance. Require **one passed, zero skipped,
zero failed** in the TRX. This method can skip even with `DOTLLM_TEST_HIP_REQUIRED=1`
when the OS/GPU/directory prerequisite is missing. A skip blocks hardware qualification.
If it fails, retain the full HIP/storage status and stop model testing; do not
enable hipFile compatibility I/O to obtain a misleading pass.

Run the primitive filter from the Windows section on Linux next. It has no
Windows-only model-loader dependency. Inspect skips, including optional hipBLAS
checks; the dense model path itself does not require hipBLAS. Then run the portable
context runner with the real GGUF on the qualified volume:

```bash
export DOTLLM_HIP_ALLOW_Q8_KV=1
dotnet run --project tests/scripts/hip-context-qualification/HipContextQualification.csproj \
  -c Release --artifacts-path .docs/hip-testing/linux-build -- \
  --model /path/on/ais-volume/SmolLM2-135M-Instruct-F16.gguf \
  --corpus "$HOME/.dotllm/test-cache/corpora/gutenberg/alice-11.utf8.txt" \
  --output .docs/hip-testing/linux-context-1k --max-context 1024
context_exit=$?
dotnet run --project tests/scripts/hip-context-qualification/HipContextQualification.csproj \
  -c Release --artifacts-path .docs/hip-testing/linux-build -- \
  validate --results .docs/hip-testing/linux-context-1k/results.jsonl
```

If scripting with `set -e`, capture the runner status in an explicit conditional
so expected exit 10 does not discard validation. Repeat at 8K in a fresh directory
after checking the 1K report. Linux results need their own report and hashes.

**Existing test platform limits:** `HipTransformerModelTests` and
`HipSafetensorsFormatsTests` still have Windows guards. The cancellation methods
in `HipSafetensorsApplicationTests` also do. Running those classes on Linux with
the required flag will fail the Windows guard; without it they skip. Neither
qualifies Linux. The portable runner provides actual GGUF parity instead.
The genuine-HF F16 route test itself has no Windows guard and can be run separately,
with its fixture on the AIS volume:

```bash
DOTLLM_TEST_HIP_HF_MODEL=/path/on/ais-volume/hf-f16 dotnet test \
  tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release \
  --filter FullyQualifiedName~RealHfConfigRoute_MatchesCpu_AndWrongRotaryLayoutIsDetected \
  --logger 'trx;LogFileName=linux-hf-f16.trx' --results-directory .docs/hip-testing/linux
```

Require one executed pass. Linux full-model BF16/AWQ/GPTQ numerical qualification
and request cancellation still need platform-appropriate tests; primitive parity
or successful app text is insufficient to close those gaps.

## Published application and package checks

Publish on each platform using the code objects rebuilt there. Example Windows
commands (use `linux-x64` and the equivalent paths on Linux):

```powershell
dotnet publish src/DotLLM.Cli/DotLLM.Cli.csproj -c Release -r win-x64 --self-contained true -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true -p:DotLLMBundleDirectStorage=false -p:MinVerVersionOverride=0.0.0-hip.local -o .docs/hip-testing/publish
dotnet pack src/DotLLM.Cli/DotLLM.Cli.csproj -c Release -p:DotLLMBundleDirectStorage=false -p:MinVerVersionOverride=0.0.0-hip.local -o .docs/hip-testing/packages
```

Inspect the publish directory and NuGet ZIP entries: `dense_f16.co` and
`rmsnorm.co` must be present with the hashes of the intended compiler output;
the package must contain `DotLLM.Hip.dll`. Inspect the single-file manifest too
when verifying assembly inclusion. There must be no `dstorage.dll` or
`dstoragecore.dll` in either release artifact. Record package/executable hashes.
Run the new executable, not a globally installed `dotllm`:

```text
<published-dotllm> run <external-GGUF-or-HF-directory> --device hip:0 -p "The capital of France is" -n 8 --temp 0 --threads 1 --json
<published-dotllm> chat <external-GGUF-or-HF-directory> --device hip:0 -n 8 --temp 0 --no-prompt-cache
<published-dotllm> serve <external-GGUF-or-HF-directory> --device hip:0 --host 127.0.0.1 --port 18787 --no-warmup --no-prompt-cache --no-ui
```

Use a real terminal for interactive `chat`, then `/exit`. For `serve`, verify
`/props` reports ready and `resolved_device=hip:0`, `/v1/devices` marks the device
servable, and POST both `/v1/completions` and `/v1/chat/completions` using the model
ID returned by `/v1/models`. Check generated text, cancellation/disconnection,
and clean shutdown. Repeat with original BF16/AWQ/GPTQ fixtures; this is application
smoke, separate from numerical validation. Keep the server on loopback and stop
only the process you started. `bench`, `perplexity` and `embed` do not support HIP;
use the context runner for the numerical work.

For example, from a second PowerShell terminal while that server is running:

```powershell
$url = 'http://127.0.0.1:18787'
$props = Invoke-RestMethod "$url/props"
if (!$props.is_ready -or $props.resolved_device -ne 'hip:0') { throw 'HIP is not ready' }
Invoke-RestMethod "$url/v1/devices" | ConvertTo-Json -Depth 10
$modelId = (Invoke-RestMethod "$url/v1/models").data[0].id
$rawBody = @{model=$modelId; prompt='The capital of France is'; max_tokens=8; temperature=0; stream=$false} | ConvertTo-Json
Invoke-RestMethod "$url/v1/completions" -Method Post -ContentType 'application/json' -Body $rawBody
$chatBody = @{model=$modelId; messages=@(@{role='user'; content='What is the capital of France?'}); max_tokens=8; temperature=0; stream=$false} | ConvertTo-Json -Depth 5
Invoke-RestMethod "$url/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $chatBody
```

The same JSON requests work with Linux `curl`. Save responses in the report
directory and stop the foreground server with Ctrl+C, checking its exit status.

Also test explicit failures: missing native image, unavailable/invalid HIP device,
unsupported geometry or quantization, missing direct-loading prerequisite, and
Q8 without acknowledgement. They must report an error rather than run on CPU.
Use isolated artifact copies for missing-file tests; do not remove shared SDK files.

## Report template and acceptance

The 2026-11-14 opt-in/harness follow-up passed the bounded combined selection:
745 passed, 27 skipped, zero failures. Focused gate tests passed 36/36; deliberately
requiring both Q8 planes instead of either, and removing the library constructor
gate, each caused the intended tests to fail. Real CLI Q8 opt-in and default F16
generation succeeded. A current-source Windows 1K runner replay preserved the
recorded F16 pass and Q8 failures, with artifact validation and memory release.
Ubuntu/WSL built the runner with zero warnings/errors and completed CPU-only 1K
with exit 10 and validator exit 0; both negative controls were caught. This follow-up
did not repeat the full 8K run or exercise physical Linux GPU/AIS. The inline F16
fixture-generation recipe above was syntax-checked, not executed in this follow-up.

```text
Target repository, branch, HEAD, baseline and working diff:
OS/kernel; CPU/RAM; GPU/ordinal/gfx/VRAM; driver; HIP runtime/hipcc:
.NET SDK/runtime; loaded managed assembly and code-object hashes:
Storage volume/filesystem/topology; DirectStorage or AIS diagnostic:
Fixture paths/revisions/SHA-256; tokenizer/config hashes; oracle producer/pip freeze:
Exact commands and environment; start/end; exit codes; artifact directory:
Tests: selected / passed / failed / skipped; every skip and reason:
Context: actual consumed length, scored positions/count, each comparison L2/KL/top-1/PPL:
Memory: payload/aligned/compute/KV owned bytes, pages, driver deltas; reset/disposal:
Negative controls: deliberate change, expected failure, observed result, restored rerun:
Application/package: resolved backend, run/chat/HTTP, shutdown, payload/hash checks:
Verdict per scope: passed / precision failed / blocked / untested:
Remaining models, formats, GPUs, platforms, corpus and concurrency coverage:
```

Keep F16 acceptance, Q8 precision failure and unexecuted Linux checks distinct.
Do not relax a numerical gate to make the report green. For correctness changes,
demonstrate a relevant mutant or real-data substitution fails, then rebuild the
restored source and rerun. Retain the evidence without publishing model weights.
