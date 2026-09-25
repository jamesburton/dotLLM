# Compile all GLSL compute shaders to SPIR-V for the dotLLM Vulkan backend.
# Requires: glslc (ships with the Vulkan SDK) on PATH.
# Output: native\vulkan\spv\*.spv
#
# End users do NOT need the Vulkan SDK — .spv blobs ship alongside the .NET
# assembly and are loaded verbatim at runtime. Only shader *authors* need glslc.

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$outDir    = Join-Path $scriptDir "spv"
$shaderDir = Join-Path $scriptDir "shaders"

if (-not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir | Out-Null }

$targetEnv = "vulkan1.2"

# Discover all shader sources up-front so we can report what's missing even
# when glslc isn't installed. This keeps the parity rig honest: a developer
# without the Vulkan SDK still sees the list of shaders the build expects to
# produce — useful for the Vulkan-perf workflow on Strix Halo where new
# shaders are added on a host without the SDK and compiled later.
$shaderSources = Get-ChildItem "$shaderDir\*.comp"
$missingSpv = @()
foreach ($compFile in $shaderSources) {
    $expectedSpv = Join-Path $outDir "$($compFile.BaseName).spv"
    if (-not (Test-Path $expectedSpv)) { $missingSpv += $compFile.Name }
}

$glslc = Get-Command glslc -ErrorAction SilentlyContinue
if ($null -eq $glslc) {
    Write-Warning "glslc not found on PATH. Install the Vulkan SDK (https://vulkan.lunarg.com/) to compile shaders."
    Write-Host ""
    Write-Host "Discovered $($shaderSources.Count) .comp files in $shaderDir."
    if ($missingSpv.Count -eq 0) {
        Write-Host "All shaders already have .spv outputs in $outDir."
    } else {
        Write-Host "Missing .spv outputs (these will need compilation once glslc is available):" -ForegroundColor Yellow
        foreach ($name in $missingSpv) { Write-Host "  $name" }
    }
    exit 1
}

Write-Host "Compiling GLSL compute shaders -> SPIR-V (target: $targetEnv)..."

foreach ($compFile in $shaderSources) {
    $base = $compFile.BaseName

    & glslc --target-env=$targetEnv -o "$outDir\$base.spv" $compFile.FullName

    if ($LASTEXITCODE -ne 0) {
        throw "glslc failed for $($compFile.Name)"
    }

    Write-Host "  $($compFile.Name) -> $base.spv"
}

Write-Host "Done. SPIR-V files in $outDir\"
