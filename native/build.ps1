# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit) on PATH
# Output: native\ptx\*.ptx
#
# PTX is forward-compatible: compute_75 PTX runs on all GPUs from Turing onward.
# CUDA 13 dropped Pascal (SM 6.x) and Volta (SM 7.0); Turing (SM 7.5) is the floor.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outDir = Join-Path $scriptDir "ptx"
$kernelDir = Join-Path $scriptDir "kernels"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$arch = "compute_75"

# The PTX ISA version every committed artifact must declare (CUDA 12.8 emits 8.7).
# PTX whose .version exceeds the driver's is rejected outright with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION — CUDA 13.1 emits 9.1, unloadable on any
# pre-13.1 driver. This has regressed twice from a newer toolkit being picked up
# silently (#124, #318), so every generated file is checked below rather than
# discovered later by a user on an older driver. Override only when deliberately
# moving the whole tree to a new toolkit baseline.
$expectPtxVersion = if ($env:DOTLLM_PTX_EXPECT_VERSION) { $env:DOTLLM_PTX_EXPECT_VERSION } else { "8.7" }

function Assert-PtxVersion {
    param([string]$Path)

    $line = Select-String -Path $Path -Pattern '^\.version\s+(\S+)' | Select-Object -First 1
    $actual = if ($line) { $line.Matches[0].Groups[1].Value } else { "(none)" }
    if ($actual -ne $expectPtxVersion) {
        $nvccPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
        throw "$([System.IO.Path]::GetFileName($Path)) declares .version $actual, expected $expectPtxVersion. " +
              "nvcc in use: $nvccPath. A committed PTX file at the wrong ISA version fails to load with " +
              "CUDA_ERROR_UNSUPPORTED_PTX_VERSION on older drivers (see #124, #318). Point nvcc at the " +
              "CUDA 12.8 toolkit, or set DOTLLM_PTX_EXPECT_VERSION if you are deliberately re-baselining the tree."
    }
}

# Kernels requiring --fmad=false for bit-perfect parity with the CPU scalar
# reference. .NET RyuJIT does not emit FMA from a*b+c without an explicit
# MathF.FusedMultiplyAdd, so without this flag the GPU result drifts by ~1 ULP
# per accumulation. The Qwen3MoeHybrid GDN recurrence compounds those errors
# across time steps, so the two kernels backing it disable FMA fusion AND
# precise math (no --use_fast_math). The bit-perfect set is small; everything
# else stays on the legacy fast-math path that this build script ships with.
$bitPerfect = @('conv1d_causal', 'gated_delta_net_scan')

Write-Host "Compiling CUDA kernels -> PTX (target: $arch)..."

foreach ($cuFile in Get-ChildItem "$kernelDir\*.cu") {
    $base = $cuFile.BaseName

    if ($bitPerfect -contains $base) {
        & nvcc -ptx -arch=$arch `
            -fmad=false `
            -o "$outDir\$base.ptx" `
            $cuFile.FullName
    } else {
        & nvcc -ptx -arch=$arch `
            --use_fast_math `
            -o "$outDir\$base.ptx" `
            $cuFile.FullName
    }

    if ($LASTEXITCODE -ne 0) {
        throw "nvcc failed for $($cuFile.Name)"
    }

    Assert-PtxVersion "$outDir\$base.ptx"

    Write-Host "  $($cuFile.Name) -> $base.ptx"
}

Write-Host "Done. PTX files in $outDir\"
