@echo off
REM Build all CUDA kernels to PTX.
REM Requires: %CUDA_PATH% set to a CUDA toolkit that supports the host MSVC
REM (CUDA 13.x supports VS 2022/2026 MSVC; CUDA 11.8 does not).
REM Usage: build_ptx.bat [arch]     (default: compute_75)
REM
REM compute_75 = Turing, the CUDA 13 floor. PTX is forward-compatible so this
REM runs on any Turing (SM 7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0),
REM or Blackwell (10.0/12.0) GPU. CUDA 13 dropped Pascal/Volta (SM 6.x/7.0).
setlocal EnableDelayedExpansion

set ARCH=%1
if "%ARCH%"=="" set ARCH=compute_75

if not defined CUDA_PATH (
    echo CUDA_PATH is not set. Install a CUDA toolkit and ensure CUDA_PATH points at it.
    exit /b 1
)
set "NVCC=%CUDA_PATH%\bin\nvcc.exe"
if not exist "%NVCC%" (
    echo nvcc.exe not found at %NVCC%
    exit /b 1
)

REM Locate latest MSVC toolchain under VS 2026
set VSROOT=E:\Program Files\Microsoft Visual Studio\18\Community
for /d %%D in ("%VSROOT%\VC\Tools\MSVC\*") do set MSVC_DIR=%%D
if not defined MSVC_DIR (
    echo Could not find VC Tools under %VSROOT%
    exit /b 1
)
set "MSVC_BIN=%MSVC_DIR%\bin\Hostx64\x64"
set "PATH=%MSVC_BIN%;%PATH%"
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo cl.exe not found at %MSVC_BIN%
    exit /b 1
)

set SCRIPT_DIR=%~dp0
set KERNEL_DIR=%SCRIPT_DIR%kernels
set OUT_DIR=%SCRIPT_DIR%ptx
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo Using nvcc: %NVCC%
echo Compiling CUDA kernels -^> PTX (target: %ARCH%)...

set FAIL=0
for %%F in ("%KERNEL_DIR%\*.cu") do (
    set BASE=%%~nF
    "%NVCC%" -ptx -arch=%ARCH% --use_fast_math -allow-unsupported-compiler -o "%OUT_DIR%\!BASE!.ptx" "%%F"
    if errorlevel 1 (
        echo FAILED: %%~nxF
        set FAIL=1
    ) else (
        echo   %%~nxF -^> !BASE!.ptx
    )
)

if "%FAIL%"=="1" exit /b 1
echo Done. PTX files in %OUT_DIR%
exit /b 0
