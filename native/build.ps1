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

    Write-Host "  $($cuFile.Name) -> $base.ptx"
}

Write-Host "Done. PTX files in $outDir\"
