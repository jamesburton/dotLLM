# Compile all .hip kernels to code objects (.co) for dotLLM HIP backend.
# Requires: hipcc (ROCm on Windows)
# Output: native/hip/co/*.co
#
# hipcc --genco produces a fat code-object ELF containing bundled AMDGPU ISA
# for each --offload-arch target. The HIP runtime picks the matching arch at
# hipModuleLoadData time.

$ErrorActionPreference = 'Stop'

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir     = Join-Path $ScriptDir 'co'
$KernelDir  = Join-Path $ScriptDir 'kernels'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Default architectures: RDNA2 (gfx1030, RX 6000), RDNA3 (gfx1100, RX 7000).
# Override with $env:HIP_ARCHS, e.g. "gfx942 gfx90a" for MI300/MI250.
$Archs = if ($env:HIP_ARCHS) { $env:HIP_ARCHS -split '\s+' } else { @('gfx1030', 'gfx1100') }

$OffloadFlags = @()
foreach ($a in $Archs) { $OffloadFlags += "--offload-arch=$a" }

$hipcc = Get-Command hipcc -ErrorAction SilentlyContinue
if (-not $hipcc) {
    Write-Error 'hipcc not found in PATH. Install ROCm to build HIP kernels.'
    exit 1
}

Write-Host "Compiling HIP kernels to code objects (archs: $($Archs -join ' '))..."

Get-ChildItem -Path $KernelDir -Filter '*.hip' | ForEach-Object {
    $src  = $_.FullName
    $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $out  = Join-Path $OutDir "$name.co"
    Write-Host "  hipcc --genco $($OffloadFlags -join ' ') -o $out $src"
    & hipcc --genco @OffloadFlags -o $out $src
    if ($LASTEXITCODE -ne 0) { throw "hipcc failed for $src" }
}

Write-Host "Done. Code objects in $OutDir"
