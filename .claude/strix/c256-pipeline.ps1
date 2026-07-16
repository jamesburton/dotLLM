# Canvas-256 attempt 3: keep-awake + sync + build + run, fully detached from SSH.
$ErrorActionPreference = 'Continue'
Add-Type -Name Power -Namespace Win32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]0x80000003) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

Set-Location C:\Development\dotLLM
$plog = 'C:\Development\dotLLM\.claude\strix\c256-pipeline.log'
"=== PIPELINE START $(Get-Date -Format s) ===" | Out-File $plog -Encoding utf8
git fetch origin 2>&1 | Out-Null
git reset --hard origin/issue/40-vulkan-real-26b 2>&1 | Select-Object -Last 1 | Add-Content $plog
dotnet build -c Release --nologo -v q 2>&1 | Select-Object -Last 2 | Add-Content $plog
"BUILD exit=$LASTEXITCODE" | Add-Content $plog

$env:DOTLLM_DIFFUSIONGEMMA_GGUF = 'C:\models\llada\diffusiongemma-26B-A4B-it-Q4_K_M.gguf'
$env:DOTLLM_DG_CANVAS = '256'
$env:DOTLLM_DG_STEPS = '48'
$env:DOTLLM_DG_PKV = '1'
dotnet test tests/DotLLM.Tests.Integration -c Release --no-build --nologo --filter FullyQualifiedName~DiffusionGemmaVulkanRealGenerationTests --logger "console;verbosity=detailed" > C:\Development\dotLLM\.claude\strix\dg26b-c256.log 2>&1
"TEST exit=$LASTEXITCODE" | Add-Content $plog

[Win32.Power]::SetThreadExecutionState([uint32]0x80000000) | Out-Null  # release keep-awake
"=== PIPELINE DONE $(Get-Date -Format s) ===" | Add-Content $plog
