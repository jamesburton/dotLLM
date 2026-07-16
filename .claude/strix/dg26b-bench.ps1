# DiffusionGemma-26B throughput battery (#41/#33) — sequential, single-GPU-owner.
$ErrorActionPreference = 'Continue'
Set-Location C:\Development\dotLLM
$log = 'C:\Development\dotLLM\.claude\strix\dg26b-bench.log'
"=== DG26B BENCH START $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8

function Log([string]$m) { $m | Out-File $log -Append -Encoding utf8 }

# 0) GPU idle sample
try {
  $s = Get-Counter "\GPU Engine(*)\Utilization Percentage" -SampleInterval 3 -MaxSamples 5
  $avg = ($s | ForEach-Object { ($_.CounterSamples | Measure-Object CookedValue -Sum).Sum } | Measure-Object -Average).Average
  Log ("[0] pre-bench GPU avg={0:N1}%" -f $avg)
} catch { Log "[0] GPU sample failed: $_" }

$env:DOTLLM_DIFFUSIONGEMMA_GGUF = 'C:\models\llada\diffusiongemma-26B-A4B-it-Q4_K_M.gguf'

# 1) dotLLM Vulkan diffusion, FULL canvas 256 (directly comparable to llama.cpp ms/step)
Log "[1] dotLLM Vulkan diffusion canvas=256 steps<=48 PKV=on"
$env:DOTLLM_DG_CANVAS = '256'; $env:DOTLLM_DG_STEPS = '48'; $env:DOTLLM_DG_PKV = '1'
dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter FullyQualifiedName~DiffusionGemmaVulkanRealGenerationTests --logger "console;verbosity=detailed" 2>&1 |
  Select-String -Pattern 'dg-26B|load |gen wall|step latency|effective|distinct|text |Passed|Failed' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }

# 2) PKV off A/B at canvas 32
Log "[2] dotLLM Vulkan diffusion canvas=32 steps<=16 PKV=off"
$env:DOTLLM_DG_CANVAS = '32'; $env:DOTLLM_DG_STEPS = '16'; $env:DOTLLM_DG_PKV = '0'
dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter FullyQualifiedName~DiffusionGemmaVulkanRealGenerationTests --logger "console;verbosity=detailed" 2>&1 |
  Select-String -Pattern 'gen wall|step latency|effective|Passed|Failed' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }
Remove-Item Env:DOTLLM_DG_CANVAS,Env:DOTLLM_DG_STEPS,Env:DOTLLM_DG_PKV -ErrorAction SilentlyContinue

# 3) llama.cpp Vulkan diffusion baseline, same file, GPU (fresh same-session numbers)
Log "[3] llama-diffusion-cli Vulkan -ngl 99 -n 128"
& C:\Development\DiffusionGemmaTests\benchmarks\backends\llamacpp-vulkan\llama-diffusion-cli.exe `
  -m $env:DOTLLM_DIFFUSIONGEMMA_GGUF -ngl 99 -n 128 -p "The Eiffel Tower is located in" 2>&1 |
  Select-String -Pattern 'total time|throughput|diffusion_eb|error' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }

# 4) dotLLM Vulkan AR decode on LLaDA-8B (plain Llama backbone) — cross-engine ratio leg A
Log "[4] dotLLM VulkanForwardPerfHarness LLaDA-8B Q4_K_M"
$env:DOTLLM_VULKAN_PERF = '1'
$env:DOTLLM_VULKAN_PERF_MODEL = 'C:\models\llada\LLaDA-8B-Instruct.Q4_K_M.gguf'
dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter FullyQualifiedName~VulkanForwardPerfHarness --logger "console;verbosity=detailed" 2>&1 |
  Select-String -Pattern 'decode_avg_ms|decode_min_ms|decode_tok_per_sec|Passed|Failed' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }
Remove-Item Env:DOTLLM_VULKAN_PERF,Env:DOTLLM_VULKAN_PERF_MODEL -ErrorAction SilentlyContinue

# 5) llama-bench same 8B file — cross-engine ratio leg B
Log "[5] llama-bench Vulkan LLaDA-8B Q4_K_M (-n 32, -r 2)"
& C:\Development\llamacpp-vulkan\llama-bench.exe -m C:\models\llada\LLaDA-8B-Instruct.Q4_K_M.gguf -p 0 -n 32 -r 2 2>&1 |
  ForEach-Object { Log ("  " + $_) }

Log "=== BENCH DONE $(Get-Date -Format s) ==="
