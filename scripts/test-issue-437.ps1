<#
.SYNOPSIS
    Runs every test that covers issue #437 (lazy KV growth + opt-in F16 KV store).

.DESCRIPTION
    Rebuilds the TEST project — not just src/DotLLM.Vulkan — because `dotnet build
    src/DotLLM.Vulkan` does not refresh tests/DotLLM.Tests.Unit/bin, and a subsequent
    --no-build run would then silently test a stale binary.

    Takes the shared GPU lock for the duration. The lock directory is resolved through
    `git rev-parse --git-common-dir`, so it genuinely serialises every worktree on this box.

    This runs CORRECTNESS tests only. It deliberately runs no throughput benchmark: the
    numbers it prints are memory footprints, not rates.

.PARAMETER SkipLock
    Run without taking the GPU lock. Only for a machine you know is idle.

.EXAMPLE
    pwsh scripts/test-issue-437.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipLock,
    [int]$LockTimeoutSeconds = 3600
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# gpu-lock.sh is a POSIX script with CRLF line endings, which WSL's bash (the `bash` on PATH in a
# default Windows install) rejects with "syntax error near unexpected token". Git Bash handles it,
# so resolve that explicitly rather than whatever `bash` happens to mean here.
$gitBash = @(
    "$env:ProgramFiles\Git\bin\bash.exe",
    "${env:ProgramFiles(x86)}\Git\bin\bash.exe",
    "$env:LOCALAPPDATA\Programs\Git\bin\bash.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

$filter = 'FullyQualifiedName~VulkanNemotronHTransformerModelForwardTests' +
          '|FullyQualifiedName~VulkanKvCacheGrowthHandleReuseTests' +
          '|FullyQualifiedName~VulkanKvCacheFootprintTests'

Write-Host '== Processes by working set (look for a non-agent GPU user before trusting anything) =='
Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name, @{n = 'WS_MB'; e = { [int]($_.WS / 1MB) } } | Format-Table

Write-Host '== Building the TEST project (this is what --no-build will run) =='
dotnet build tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -v q --nologo
if ($LASTEXITCODE -ne 0) { throw 'build failed' }

$locked = $false
try {
    if (-not $SkipLock) {
        if (-not $gitBash) { throw 'Git Bash not found; re-run with -SkipLock on a machine you know is idle.' }
        Write-Host '== Acquiring the shared GPU lock =='
        & $gitBash scripts/gpu-lock.sh acquire issue437 'issue 437 KV growth + F16 correctness' $LockTimeoutSeconds
        if ($LASTEXITCODE -ne 0) { throw 'could not acquire the GPU lock' }
        $locked = $true
    }

    Write-Host '== Running the issue-437 tests (detailed, so the footprint table prints) =='
    dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --no-build --nologo `
        --filter $filter --logger 'console;verbosity=detailed'
    $testExit = $LASTEXITCODE
}
finally {
    if ($locked) { & $gitBash scripts/gpu-lock.sh release issue437 }
}

exit $testExit

<#
EXPECTED OUTPUT
---------------
    Passed!  - Failed:     0, Passed:    20, Skipped:     0, Total:    20

20 = 6 pre-existing NemotronH CPU-parity tests + 14 added for #437.

The footprint table is printed by VulkanKvCacheFootprintTests and should read, at Bonsai 2
geometry (16 full-attention layers, head_count_kv=4, key_length=value_length=256 => 128 KiB
per token). `ours=` is the allocator's own live-byte accounting; the driver's usage/budget
fields are per-process and a host working-set reading does not show device-local memory at all.

    === ctx 4096 ===
    EAGER   (pre-#437):    512.0 MiB   capacity=4096
      device-local fallbacks during eager alloc: 0
    LAZY    (default):      64.0 MiB   capacity=512
    LAZY    after 2k ctx:  256.0 MiB   capacity=2048
    EAGER F16 (opt-in):    256.0 MiB   capacity=4096
    LAZY  F16 after 2k:    128.0 MiB   capacity=2048

    === ctx 32768 ===
    EAGER   (pre-#437):   4096.0 MiB   capacity=32768
      device-local fallbacks during eager alloc: 0
    LAZY    (default):      64.0 MiB   capacity=512
    LAZY    after 2k ctx:  256.0 MiB   capacity=2048
    EAGER F16 (opt-in):   2048.0 MiB   capacity=32768
    LAZY  F16 after 2k:    128.0 MiB   capacity=2048

The ctx-32768 eager case allocates a real 4 GiB of device-local memory for a moment. On a
32 GB UMA box with WSL running, close other GPU consumers first.

KNOWN PRE-EXISTING FAILURES ELSEWHERE (not touched by this filter, do not chase):
  6x VulkanMatMulI2SGemvF32KernelTests.MultiRow_IsBitIdenticalToProduction
  1x VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward (flaky ~1 in 5, #433)

WIDER REGRESSION SWEEP (~9 min GPU) — the suites that share the changed code:
  dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj --no-build -v q --nologo --filter `
    "FullyQualifiedName~Qwen3HybridDense|FullyQualifiedName~Qwen3MoeHybrid|FullyQualifiedName~VulkanNemotronH|FullyQualifiedName~VulkanRopeKvWriteFusion|FullyQualifiedName~VulkanSplitKvAttention|FullyQualifiedName~VulkanAttention|FullyQualifiedName~VulkanKvCache|FullyQualifiedName~VulkanFlashAttention"
  Expected: Passed: 158, Failed: 0, Skipped: 55 (the skips need GGUF fixtures that are absent here).
#>
