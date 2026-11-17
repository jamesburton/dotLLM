# Compile all .hip kernels to code objects (.co) for dotLLM HIP backend.
# Requires: hipcc (ROCm on Windows)
# Output: native/hip/co/*.co
#
# hipcc --genco produces a fat code-object ELF containing bundled AMDGPU ISA
# for each --offload-arch target. The HIP runtime picks the matching arch at
# hipModuleLoadData time.

param([string[]]$Architectures, [string]$HipCompiler)

$ErrorActionPreference = 'Stop'

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutDir     = Join-Path $ScriptDir 'co'
$KernelDir  = Join-Path $ScriptDir 'kernels'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Default architectures include Strix Halo (gfx1151).
# Override with $env:HIP_ARCHS, e.g. "gfx942 gfx90a" for MI300/MI250.
$Archs = if ($Architectures) { $Architectures } elseif ($env:HIP_ARCHS) { $env:HIP_ARCHS -split '\s+' } else { @('gfx1030', 'gfx1100', 'gfx1151') }

$OffloadFlags = @()
foreach ($a in $Archs) { $OffloadFlags += "--offload-arch=$a" }

if (-not $HipCompiler) {
    $hipcc = Get-Command hipcc -ErrorAction SilentlyContinue
    if ($hipcc) { $HipCompiler = $hipcc.Source }
    elseif ($env:HIP_PATH -and (Test-Path -LiteralPath (Join-Path $env:HIP_PATH 'bin\hipcc.exe'))) {
        $HipCompiler = Join-Path $env:HIP_PATH 'bin\hipcc.exe'
    }
}
if (-not $HipCompiler) { throw 'hipcc not found. Set HIP_PATH or PATH to the HIP SDK and use a Visual Studio C++ developer shell on Windows.' }

$CompilerVersion = (& $HipCompiler --version | Out-String).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to run the HIP compiler.' }
Write-Host $CompilerVersion

Write-Host "Compiling HIP kernels to code objects (archs: $($Archs -join ' '))..."

$kernels = @(Get-ChildItem -LiteralPath $KernelDir -Filter '*.hip' -File)
if ($kernels.Count -eq 0) { throw "No HIP sources found in $KernelDir" }
$kernels | ForEach-Object {
    $src  = $_.FullName
    $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
    $out  = Join-Path $OutDir "$name.co"
    Write-Host "  hipcc --genco $($OffloadFlags -join ' ') -o $out $src"
    & $HipCompiler --genco @OffloadFlags -o $out $src
    if ($LASTEXITCODE -ne 0) { throw "hipcc failed for $src" }
}

$manifest = @(
    "Built UTC: $([DateTime]::UtcNow.ToString('O'))",
    "Platform: Windows x64",
    "Targets (each object): $($Archs -join ' ')",
    'Command: hipcc --genco <targets> -o co/<name>.co kernels/<name>.hip',
    "Compiler: $CompilerVersion",
    'SHA256  file'
)
foreach ($kernel in $kernels) {
    $object = Join-Path $OutDir ($kernel.BaseName + '.co')
    $manifest += "$((Get-FileHash -Algorithm SHA256 -LiteralPath $object).Hash.ToLowerInvariant())  $($kernel.BaseName).co"
    $manifest += "$((Get-FileHash -Algorithm SHA256 -LiteralPath $kernel.FullName).Hash.ToLowerInvariant())  ../kernels/$($kernel.Name)"
}
$manifest | Set-Content -LiteralPath (Join-Path $OutDir 'MANIFEST') -Encoding utf8
Write-Host "Done. Code objects and MANIFEST in $OutDir"
