# framework (Meteor Lake) — Intel Arc Vulkan validation instructions

Instructions for an agent (or James) working **on the `framework` box** — the Framework laptop:
Intel Core Ultra 7 155H (Meteor Lake), Intel Arc iGPU (driver 32.0.101.7026 at time of writing),
95 GB RAM, plus an *intermittently-attached* RTX 3060 eGPU. Written 2026-08-13 from the
Nemotron-3.5-Lightning research session (Strix Halo box); background evidence lives in that
session's report (`.docs/NEMOTRON_35_LIGHTNING_RESEARCH.md`, git-ignored, on the Strix Halo box)
and in issues **#374** (Vulkan device features) and **#375** (nemotron_h_moe umbrella).

## Standing constraints — read first

- **This machine is a highly-contended, crash-prone daily driver** (spontaneous Kernel-Power 41
  reboots every 2–6 days; the eGPU falls off the bus intermittently). Keep runs short, save
  results to disk *immediately*, and do not schedule anything long-running or unattended.
- **Coordinate GPU use through James before any GPU run** (same rule as every shared box).
  Use `scripts/gpu-lock.sh acquire <name> "<reason>"` / `release` around every GPU run —
  the lock is per-clone, which is fine here (one clone).
- Target the **Arc iGPU, not the RTX 3060**: set `DOTLLM_VULKAN_DEVICE_INDEX` to the Arc
  adapter's index. Enumerate first (any Vulkan test run logs the selected device, or use
  `vulkaninfo --summary`) and record which index is which — do NOT assume index 0; with the
  eGPU attached the ordering can change between boots.
- Model weights follow the global storage rules: HF hub cache via `hf download`, never ad-hoc
  folders. **Do not download any Nemotron-3.5-Lightning GGUF yet** — the smallest useful file
  is 17.55 GiB and nothing can run it until the #375 kernel work lands. The tasks below need
  no model downloads.

## Why this box matters

The Arc iGPU is dotLLM's first non-AMD Vulkan target. It differs from the Strix Halo dev box in
exactly the ways that expose latent bugs (see the #364/#367 memory: "validate GPU paths on
discrete/other-vendor hardware, not just UMA"):

| Property | Arc (framework) | gfx1151 (dev box) |
|---|---|---|
| `VK_KHR_cooperative_matrix` | **absent** | present |
| `maxStorageBufferRange` | **1 GiB** | 4 GiB |
| `maxComputeSharedMemorySize` | **32 KiB** | 64 KiB |
| `maxMemoryAllocationSize` | ~4 GiB | ~2 GiB |
| subgroup size | 32 (8–32) | wave32/64 |
| dp4a (`integerDotProduct4x8BitPackedSignedAccelerated`) | yes | yes |
| driver leniency re: unenabled SPIR-V features | **reportedly strict** | lenient |

## Task 1 — Vulkan SDK + validation layers (highest value, no GPU load)

1. Install the LunarG Vulkan SDK (brings `VK_LAYER_KHRONOS_validation`). The Strix Halo box
   does **not** have it, which left PR #370's validation-layer acceptance criterion unmet.
2. Verify the layer is visible: `vulkaninfo --summary` must list `VK_LAYER_KHRONOS_validation`.

## Task 2 — Arc device bring-up smoke (short GPU run)

From a clone of the repo at current `dev` (`git clone` + `dotnet build -c Debug`):

```powershell
$env:DOTLLM_VULKAN_DEVICE_INDEX = "<arc-index>"
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Debug --nologo `
  --filter "FullyQualifiedName~VulkanMatMulF32KernelTests|FullyQualifiedName~VulkanMatMulQ8_0KernelTests|FullyQualifiedName~VulkanPackedSsboRoundUp|FullyQualifiedName~QuantFormatTests"
```

Expected outcomes and what each means:

- **Device creation fails / pipelines fail to create** → almost certainly issue **#374**
  (dotLLM never chains the 16-bit-storage / float16 / int8 feature structs that 17 shaders
  `require`; the AMD driver tolerates this, Intel's likely does not). That failure is *the
  expected positive result* — capture the exact error text and post it on #374; it converts
  #374's [INFERRED] severity into a verified blocker.
- **Tests pass** → also valuable: post on #374 that the Intel Windows driver tolerates
  unenabled features too, downgrading the issue from blocker to spec-hygiene.
- Either way, record: selected device name/index, driver version, pass/fail counts, and any
  validation output.

## Task 3 — the #361 odd-block tests under the validation layer (retires PR #370's open criterion)

With the SDK from Task 1 installed:

```powershell
$env:VK_INSTANCE_LAYERS = "VK_LAYER_KHRONOS_validation"
$env:DOTLLM_VULKAN_DEVICE_INDEX = "<arc-index>"
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Debug --nologo `
  --filter "FullyQualifiedName~VulkanPackedSsboRoundUp|FullyQualifiedName~VulkanMatMulQ8_0KernelTests|FullyQualifiedName~VulkanMatMulQ3KGemvF32KernelTests|FullyQualifiedName~VulkanMatMulIq4NlGemvF32KernelTests" 2>&1 |
  Tee-Object validation-run.log
```

Success = **zero out-of-bounds / VUID messages in the log** for the odd-block cases. Report the
result on issue **#361** / PR **#370** (the criterion is quoted there). Expect *unrelated*
validation chatter — the codebase notes a known benign false-positive around
`VkPhysicalDeviceSubgroupSizeControl` naming; log everything, judge nothing silently.

## Task 4 — capability report for the record

Attach to #374: full `vulkaninfo` output for the Arc device (features2 blocks included), so the
feature-chaining fix can enable exactly what Arc reports. If the RTX 3060 is attached, capture
its block too (free extra data point; it *does* have coopmat).

## Reporting

Post findings as comments on #374 (Tasks 2/4) and #370/#361 (Task 3). If a finding changes the
support-gap picture for #375 (Nemotron-3.5-Lightning), note it there too. Keep raw logs; quote
exact error text, not paraphrases.
