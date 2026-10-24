# Issue #438 — host + device memory residency of a Vulkan model on UMA.
#
# Runs `dotllm bench` twice against the same GGUF with DOTLLM_MEM_PROBE=1:
#   arm A (baseline): the GGUF mmap is kept alive for the model's lifetime (today's behaviour)
#   arm B (release) : DOTLLM_BENCH_RELEASE_GGUF=1 unmaps the GGUF immediately after upload
#
# Each arm prints [mem-probe:<point>] lines on stderr at before-load / after-load /
# (after-gguf-dispose) / after-decode. Machine-wide free physical memory is captured from
# WMI around each arm, and the top processes by working set before and after, so a
# contaminating GPU consumer (a game, vmmemWSL) is visible in the record.
#
# Take scripts/gpu-lock.sh before running this: it loads a model onto the GPU.

[CmdletBinding()]
param(
    [string]$Model = "$HOME/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PQ2_0.gguf",
    [int]$PromptTokens = 32,
    [int]$GenTokens = 8,
    [int]$Reps = 1,
    [string]$OutDir = "$env:TEMP/438-residency",
    [switch]$SkipBaseline,
    [switch]$SkipRelease
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$cli  = Join-Path $repo 'src/DotLLM.Cli/bin/Release/net10.0/DotLLM.Cli.exe'

if (-not (Test-Path $cli))   { throw "CLI not built: $cli  (dotnet build src/DotLLM.Cli -c Release)" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Get-FreeMiB {
    [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024, 0)
}
function Get-TopProcesses {
    Get-Process | Sort-Object WS -Descending | Select-Object -First 8 |
        ForEach-Object { '{0} ws={1:N0} MiB started={2}' -f $_.ProcessName, ($_.WS / 1MB), $(try { $_.StartTime.ToString('HH:mm:ss') } catch { 'n/a' }) }
}

function Invoke-Arm {
    param([string]$Name, [bool]$ReleaseGguf)

    $log = Join-Path $OutDir "$Name.log"
    Write-Host "=== arm $Name ===" -ForegroundColor Cyan
    Write-Host "free before (WMI Win32_OperatingSystem.FreePhysicalMemory): $(Get-FreeMiB) MiB"
    Write-Host "top processes BEFORE:"; Get-TopProcesses | ForEach-Object { "  $_" }

    $env:DOTLLM_MEM_PROBE = '1'
    $env:DOTLLM_BENCH_DUMP_TOKENS = (Join-Path $OutDir "$Name.tokens")
    if ($ReleaseGguf) { $env:DOTLLM_BENCH_RELEASE_GGUF = '1' } else { Remove-Item Env:DOTLLM_BENCH_RELEASE_GGUF -ErrorAction SilentlyContinue }
    Remove-Item $env:DOTLLM_BENCH_DUMP_TOKENS -ErrorAction SilentlyContinue

    & $cli bench $Model --device vulkan -p $PromptTokens -n $GenTokens -r $Reps --json 2>&1 |
        Tee-Object -FilePath $log

    Write-Host "free after: $(Get-FreeMiB) MiB"
    Write-Host "top processes AFTER:"; Get-TopProcesses | ForEach-Object { "  $_" }
    Write-Host "log: $log"
}

if (-not $SkipBaseline) { Invoke-Arm -Name 'A-baseline' -ReleaseGguf $false }
if (-not $SkipRelease)  { Invoke-Arm -Name 'B-release'  -ReleaseGguf $true }

Write-Host ""
Write-Host "=== probe lines ===" -ForegroundColor Cyan
Get-ChildItem $OutDir -Filter '*.log' | ForEach-Object {
    Write-Host "--- $($_.Name)"
    Select-String -Path $_.FullName -Pattern '\[mem-probe:' | ForEach-Object { $_.Line }
}

$ta = Join-Path $OutDir 'A-baseline.tokens'; $tb = Join-Path $OutDir 'B-release.tokens'
if ((Test-Path $ta) -and (Test-Path $tb)) {
    $same = (Get-Content $ta -Raw) -eq (Get-Content $tb -Raw)
    Write-Host "generated token ids identical across arms: $same"
}
