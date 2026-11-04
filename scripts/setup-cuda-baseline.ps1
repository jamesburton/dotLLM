<#
.SYNOPSIS
    Installs the dotLLM baseline CUDA toolkit (12.8 -> PTX ISA .version 8.7) side by side with any
    newer toolkit, plus optionally a CUDA-compatible MSVC toolset. Dry run by default. Issue #1635.

.DESCRIPTION
    The build never needs this: DotLLM.Cuda compiles with whatever toolchain it finds and pins or
    warns (docs/CUDA.md "Building on newer CUDA SDKs"). Install the baseline when you want PTX that
    is byte-for-byte what the committed tree is built with, e.g. to commit a new kernel's .ptx.

    What -Apply does (needs an elevated shell):
      1. Downloads NVIDIA's signed CUDA <release> network installer (~14 MB) and verifies its
         Authenticode signature is valid and from NVIDIA Corporation.
      2. Silently installs ONLY the compiler components:  nvcc_<v> cudart_<v>  (cudart pulls in
         thrust_<v>, which carries the CCCL/libcu++ headers cooperative_groups needs). With
         -WithNvrtc also nvrtc_<v> nvrtc_dev_<v>. No display driver, no Nsight, no Visual Studio
         integration. Lands in "%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v<v>" next to any
         other version and sets CUDA_PATH_V<maj>_<min>.
      3. The installer also repoints the machine-wide CUDA_PATH at the new toolkit; the script
         restores the previous value (unless -KeepCudaPath) so other projects are unaffected.
         The dotLLM build finds the baseline through CUDA_PATH_V12_8 regardless.
      4. With -InstallMsvcToolset: if no MSVC toolset below 14.50 exists (CUDA 12.x / 13.0 / 13.1
         reject VS 2026's 14.5x), adds the "MSVC v14.44 (VS 2022 17.14)" component to the newest
         Visual Studio instance via the VS installer.

    Safe to re-run: anything already present is reported and skipped.

.EXAMPLE
    ./scripts/setup-cuda-baseline.ps1                       # report + plan only
.EXAMPLE
    ./scripts/setup-cuda-baseline.ps1 -Apply                # from an elevated shell
.EXAMPLE
    ./scripts/setup-cuda-baseline.ps1 -Apply -InstallMsvcToolset
#>
[CmdletBinding()]
param(
    [string] $CudaVersion = '12.8',
    [string] $CudaRelease = '12.8.1',
    [switch] $Apply,
    [switch] $WithNvrtc,
    [switch] $InstallMsvcToolset,
    [switch] $KeepCudaPath,
    [string] $MsvcComponent = 'Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64',
    [string] $DownloadDir = (Join-Path ([IO.Path]::GetTempPath()) 'dotllm-cuda-setup')
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $IsWindows -and $PSVersionTable.PSEdition -eq 'Core') {
    Write-Host "Linux/macOS: install the toolkit packages side by side instead, e.g. (Ubuntu, NVIDIA apt repo):"
    Write-Host "  sudo apt-get install cuda-nvcc-$($CudaVersion -replace '\.','-') cuda-cudart-dev-$($CudaVersion -replace '\.','-')"
    Write-Host "  -> /usr/local/cuda-$CudaVersion, which the dotLLM build discovers automatically."
    return
}

$cudaRoot = Join-Path $env:ProgramFiles 'NVIDIA GPU Computing Toolkit\CUDA'
$baselineDir = Join-Path $cudaRoot "v$CudaVersion"
$envName = 'CUDA_PATH_V' + ($CudaVersion -replace '\.', '_')
$installerUrl = "https://developer.download.nvidia.com/compute/cuda/$CudaRelease/network_installers/cuda_${CudaRelease}_windows_network.exe"
$components = @("nvcc_$CudaVersion", "cudart_$CudaVersion")
if ($WithNvrtc) { $components += @("nvrtc_$CudaVersion", "nvrtc_dev_$CudaVersion") }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$scriptPath = $MyInvocation.MyCommand.Path

function Get-NvccRelease([string] $root) {
    $nvcc = Join-Path $root 'bin\nvcc.exe'
    if (-not (Test-Path $nvcc)) { return $null }
    $v = & $nvcc --version 2>$null | Select-String 'release (\d+\.\d+)'
    if ($v) { return $v.Matches[0].Groups[1].Value } else { return '?' }
}

function Get-DirSizeMB([string] $path) {
    if (-not (Test-Path $path)) { return 0 }
    [math]::Round(((Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum) / 1MB, 0)
}

function Get-MsvcToolsets {
    $found = @()
    if (Test-Path $vswhere) {
        $found = & $vswhere -products * -find 'VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe' 2>$null
    }
    foreach ($cl in $found) {
        $ver = [regex]::Match($cl, '\\MSVC\\(\d+\.\d+\.\d+)').Groups[1].Value
        [pscustomobject]@{ Version = [version]$ver; BinDir = Split-Path $cl }
    }
}

# ── report ─────────────────────────────────────────────────────────────────────────────────────
Write-Host "== CUDA toolkits"
$toolkits = @()
$roots = @()
if (Test-Path $cudaRoot) { $roots += Get-ChildItem $cudaRoot -Directory -Filter 'v*' | ForEach-Object FullName }
$roots += Get-ChildItem env: | Where-Object Name -like 'CUDA_PATH_V*' | ForEach-Object Value
foreach ($r in ($roots | Sort-Object -Unique)) {
    $rel = Get-NvccRelease $r
    $toolkits += [pscustomobject]@{ Root = $r; Nvcc = if ($rel) { $rel } else { '(no nvcc)' } }
}
if ($toolkits) { $toolkits | Format-Table -AutoSize | Out-String | Write-Host } else { Write-Host "  none installed" }
Write-Host "  CUDA_PATH = $([Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine'))"
$haveBaseline = (Get-NvccRelease $baselineDir) -eq $CudaVersion

Write-Host "== MSVC host toolsets"
$msvc = @(Get-MsvcToolsets | Sort-Object Version -Descending)
if ($msvc) { $msvc | ForEach-Object { Write-Host ("  {0,-14} {1}{2}" -f $_.Version, $_.BinDir, $(if ($_.Version -lt [version]'14.50') { '   <- CUDA 12.x/13.0/13.1 compatible' } else { '   (VS 2026: CUDA 13.2+ only, or override flags)' })) } }
else { Write-Host "  none found (vswhere: $vswhere)" }
$haveCompatMsvc = [bool]($msvc | Where-Object { $_.Version -lt [version]'14.50' })

$drive = Get-PSDrive ($env:ProgramFiles.Substring(0, 1))
Write-Host ("== Free space on {0}: {1:N1} GB" -f $drive.Name, ($drive.Free / 1GB))

# ── plan ───────────────────────────────────────────────────────────────────────────────────────
$plan = @()
if ($haveBaseline) { Write-Host "OK  CUDA $CudaVersion already installed at $baselineDir" }
else { $plan += "install CUDA $CudaVersion ($CudaRelease) components [$($components -join ' ')] -> $baselineDir (~0.3-0.5 GB; no driver)" }
if ($haveCompatMsvc) { Write-Host "OK  a CUDA-compatible MSVC toolset (< 14.50) is installed" }
elseif ($InstallMsvcToolset) { $plan += "add VS component $MsvcComponent (MSVC v14.44) to the newest Visual Studio instance (~1.5-2 GB)" }
else { Write-Host "NOTE no MSVC toolset < 14.50: CUDA $CudaVersion needs one (VS 2022 / MSVC 14.4x). Re-run with -InstallMsvcToolset, or the build will fall back to override flags." }

if (-not $plan) {
    Write-Host "`nNothing to do."
}
else {
    Write-Host "`n== Plan"
    $plan | ForEach-Object { Write-Host "  - $_" }
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
$applyCmd = "pwsh -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -Apply" + $(if ($InstallMsvcToolset) { ' -InstallMsvcToolset' } else { '' }) + $(if ($WithNvrtc) { ' -WithNvrtc' } else { '' })

if ($plan -and -not $Apply) {
    Write-Host "`nDry run. To apply, from an ELEVATED (Administrator) shell run:"
    Write-Host "  $applyCmd"
}
if ($plan -and $Apply -and -not $isAdmin) {
    Write-Error "-Apply needs an elevated shell. From an Administrator terminal run:`n  $applyCmd"
}

# ── apply ──────────────────────────────────────────────────────────────────────────────────────
if ($plan -and $Apply) {
    if (-not $haveBaseline) {
        $freeBefore = $drive.Free
        New-Item -ItemType Directory -Force $DownloadDir | Out-Null
        $exe = Join-Path $DownloadDir "cuda_${CudaRelease}_windows_network.exe"
        if (-not (Test-Path $exe)) {
            Write-Host "Downloading $installerUrl"
            Invoke-WebRequest -UseBasicParsing -Uri $installerUrl -OutFile $exe
        }
        $sig = Get-AuthenticodeSignature $exe
        if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=NVIDIA Corporation') {
            Remove-Item $exe -Force
            throw "Installer signature check failed ($($sig.Status), $($sig.SignerCertificate.Subject)); deleted it."
        }
        $prevCudaPath = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
        Write-Host "Installing (silent): $($components -join ' ') ... this downloads the packages and takes a few minutes."
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $p = Start-Process -FilePath $exe -ArgumentList (@('-s') + $components + @('-n')) -Wait -PassThru
        Write-Host ("Installer exit code {0} after {1:N0} s" -f $p.ExitCode, $sw.Elapsed.TotalSeconds)
        $rel = Get-NvccRelease $baselineDir
        if ($rel -ne $CudaVersion) { throw "Install did not produce nvcc $CudaVersion at $baselineDir (found: $rel)." }
        if (-not $KeepCudaPath -and $prevCudaPath -and $prevCudaPath -ne [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')) {
            [Environment]::SetEnvironmentVariable('CUDA_PATH', $prevCudaPath, 'Machine')
            Write-Host "Restored machine CUDA_PATH -> $prevCudaPath"
        }
        $drive = Get-PSDrive $drive.Name
        Write-Host ("Installed: {0} MB in {1}; free space change {2:N2} GB" -f (Get-DirSizeMB $baselineDir), $baselineDir, (($freeBefore - $drive.Free) / 1GB))
    }
    if (-not $haveCompatMsvc -and $InstallMsvcToolset) {
        $vsPath = & $vswhere -latest -products * -property installationPath 2>$null | Select-Object -First 1
        $vsSetup = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\setup.exe'
        if (-not $vsPath -or -not (Test-Path $vsSetup)) { throw "No Visual Studio instance / installer found; install 'Build Tools for Visual Studio 2022' instead." }
        Write-Host "Adding $MsvcComponent to $vsPath"
        $p = Start-Process -FilePath $vsSetup -ArgumentList @('modify', '--installPath', "`"$vsPath`"", '--add', $MsvcComponent, '--quiet', '--norestart') -Wait -PassThru
        Write-Host "VS installer exit code $($p.ExitCode) (0 = ok, 3010 = ok, reboot pending)"
    }
}

# ── how to use ─────────────────────────────────────────────────────────────────────────────────
Write-Host @"

== Using it with dotLLM
  The build prefers $envName / $baselineDir automatically. Optional explicit pins:
    dotnet build -p:DotLLMCudaToolkitPath="$baselineDir"
    dotnet build -p:DotLLMMsvcBinDir="<VS>\VC\Tools\MSVC\14.44.xxxxx\bin\Hostx64\x64"
  (or the DOTLLM_CUDA_TOOLKIT_PATH / DOTLLM_MSVC_BIN_DIR environment variables).
  Regenerate one kernel's PTX strictly at the baseline:
    touch native/kernels/<name>.cu
    dotnet build src/DotLLM.Cuda -p:DotLLMCudaBuildMode=compile -p:DotLLMPtxVersionPolicy=strict
"@
