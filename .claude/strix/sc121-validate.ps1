# #121 validation battery: Vulkan parity + real-26B sparse-SC A/B + canvas-256 TDR gauntlet.
$ErrorActionPreference = 'Continue'
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]0x80000003) | Out-Null

Set-Location C:\Development\dotLLM
$log = 'C:\Development\dotLLM\.claude\strix\sc121-validate.log'
"=== SC121 VALIDATE START $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8
function Log([string]$m) { $m | Out-File $log -Append -Encoding utf8 }

# GPU-free gate: abort if the iGPU is busy with someone else's work.
try {
  $s = Get-Counter "\GPU Engine(*)\Utilization Percentage" -SampleInterval 5 -MaxSamples 6
  $avg = ($s | ForEach-Object { ($_.CounterSamples | Measure-Object CookedValue -Sum).Sum } | Measure-Object -Average).Average
  Log ("[gate] GPU avg={0:N1}%" -f $avg)
  if ($avg -gt 10) { Log "GPU BUSY — ABORTING (rerun when free)"; Log "=== SC121 VALIDATE DONE (aborted) ==="; exit 1 }
} catch { Log "[gate] GPU sample failed: $_" }

git fetch origin 2>&1 | Out-Null
git reset --hard origin/issue/121-sc-sparsify-topk 2>&1 | Select-Object -Last 1 | Add-Content $log
dotnet build -c Release --nologo -v q 2>&1 | Select-Object -Last 2 | Add-Content $log
Log "BUILD exit=$LASTEXITCODE"

# 1) Vulkan synthetic parity (new top-K tests + existing diffusion parity)
Log "[1] Vulkan synthetic parity"
dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter "FullyQualifiedName~Gemma4DiffusionScTopKVulkanTests|FullyQualifiedName~Gemma4DiffusionVulkanTests" 2>&1 |
  Select-String -Pattern 'Passed!|Failed|FAIL|error' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }

$env:DOTLLM_DIFFUSIONGEMMA_GGUF = 'C:\models\llada\diffusiongemma-26B-A4B-it-Q4_K_M.gguf'
function RunReal([string]$label, [hashtable]$envs) {
  Log "[$label]"
  foreach ($k in $envs.Keys) { Set-Item "Env:$k" $envs[$k] }
  dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter FullyQualifiedName~DiffusionGemmaVulkanRealGenerationTests --logger "console;verbosity=detailed" 2>&1 |
    Select-String -Pattern 'load |gen wall|step latency|effective|distinct|text |Passed|Failed|DEVICE_LOST|OUT_OF_DEVICE|Exception' | ForEach-Object { Log ("  " + $_.Line.TrimEnd()) }
  foreach ($k in $envs.Keys) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
}

# 2) canvas 32: sparse K=256 (new default) vs dense K=0 — quality + speed A/B
RunReal '2a c32 sparse K=256' @{ DOTLLM_DG_CANVAS='32'; DOTLLM_DG_STEPS='16'; DOTLLM_DG_SC_TOPK='256' }
RunReal '2b c32 dense K=0'    @{ DOTLLM_DG_CANVAS='32'; DOTLLM_DG_STEPS='16'; DOTLLM_DG_SC_TOPK='0' }

# 3) canvas 256 gauntlet (previously VK_ERROR_DEVICE_LOST): defaults K=256, chunk 32
RunReal '3 c256 K=256 chunk=32' @{ DOTLLM_DG_CANVAS='256'; DOTLLM_DG_STEPS='48' }

# 4) chunk-rows sweep at canvas 256 (only meaningful if [3] passed)
RunReal '4a c256 chunk=16' @{ DOTLLM_DG_CANVAS='256'; DOTLLM_DG_STEPS='48'; DOTLLM_DG_HEAD_CHUNK_ROWS='16' }
RunReal '4b c256 chunk=64' @{ DOTLLM_DG_CANVAS='256'; DOTLLM_DG_STEPS='48'; DOTLLM_DG_HEAD_CHUNK_ROWS='64' }

[Win32.Power]::SetThreadExecutionState([uint32]0x80000000) | Out-Null
"=== SC121 VALIDATE DONE $(Get-Date -Format s) ===" | Add-Content $log
