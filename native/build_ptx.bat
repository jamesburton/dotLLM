@echo off
REM Build all CUDA kernels to PTX via VS 2026 + CUDA 11.8.
REM Usage: build_ptx.bat [arch]     (default: compute_61)
setlocal EnableDelayedExpansion

set ARCH=%1
if "%ARCH%"=="" set ARCH=compute_61

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

echo Compiling CUDA kernels -^> PTX (target: %ARCH%)...

set FAIL=0
for %%F in ("%KERNEL_DIR%\*.cu") do (
    set BASE=%%~nF
    nvcc -ptx -arch=%ARCH% --use_fast_math -allow-unsupported-compiler -o "%OUT_DIR%\!BASE!.ptx" "%%F"
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
