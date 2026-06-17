@echo off
REM Build all CUDA kernels to PTX.
REM Requires: %CUDA_PATH% set to a CUDA toolkit that supports the host MSVC
REM (CUDA 13.x supports VS 2022/2026 MSVC; CUDA 11.8 does not).
REM Usage: build_ptx.bat [arch]     (default: compute_75)
REM
REM compute_75 = Turing, the CUDA 13 floor. PTX is forward-compatible so this
REM runs on any Turing (SM 7.5), Ampere (8.0/8.6), Ada (8.9), Hopper (9.0),
REM or Blackwell (10.0/12.0) GPU. CUDA 13 dropped Pascal/Volta (SM 6.x/7.0).
REM
REM ARCH POLICY: keep the default at compute_75. The driver JITs PTX to the
REM actual SM at module load, so compute_75 PTX already runs as native sm_86 on
REM (e.g.) an RTX 3060 with NO perf penalty. Raising the GLOBAL default does not
REM make anything faster; it only drops portability to Turing. The ONLY reason
REM to use compute_86+ is a kernel that emits Ampere-only PTX (mma.sync /
REM cp.async). For those: build that ONE kernel at the higher arch AND
REM dispatch-gate it to capable GPUs in C# (e.g. the G3 attention path is
REM GeForce-Ampere-gated, so its PTX never loads on Turing). Per-kernel arch
REM overrides live in the ARCH_86 list below; the committed native/ptx tree
REM stays uniform sm_75 except such gated kernels. See issue #70.
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

REM Locate a CUDA-compatible MSVC toolchain. Preference order:
REM   1. VS 2022 Build Tools at E:\VS2022\BuildTools (MSVC 14.3x-14.4x — CUDA 13.x compatible)
REM   2. VS 2026 Community at E:\Program Files\Microsoft Visual Studio\18\Community
REM      (MSVC 14.50 — _MSC_VER==1950, rejected by CUDA 13.1's host_config.h and by nvcc's OS-target check)
REM A user can override by pre-setting MSVC_DIR before invoking this script.
set "BT_ROOT=E:\VS2022\BuildTools"
set "VS2026_ROOT=E:\Program Files\Microsoft Visual Studio\18\Community"
if not defined MSVC_DIR if exist "%BT_ROOT%\VC\Tools\MSVC\" (
    for /d %%D in ("%BT_ROOT%\VC\Tools\MSVC\*") do set "MSVC_DIR=%%D"
)
if not defined MSVC_DIR if exist "%VS2026_ROOT%\VC\Tools\MSVC\" (
    for /d %%D in ("%VS2026_ROOT%\VC\Tools\MSVC\*") do set "MSVC_DIR=%%D"
)
if not defined MSVC_DIR (
    echo Could not locate any VC Tools install. Install VS 2022 Build Tools to E:\VS2022\BuildTools.
    exit /b 1
)
set "MSVC_BIN=%MSVC_DIR%\bin\Hostx64\x64"
REM Delayed expansion below: a pre-set MSVC_DIR may live under "Program Files
REM (x86)" — the literal ) in (x86) would prematurely close a %VAR% echo inside
REM a parenthesised if-block, so reference !MSVC_BIN! (delayed) instead.
if not exist "!MSVC_BIN!\cl.exe" (
    echo cl.exe not found at "!MSVC_BIN!"
    exit /b 1
)
set "PATH=%MSVC_BIN%;%PATH%"

set SCRIPT_DIR=%~dp0
set KERNEL_DIR=%SCRIPT_DIR%kernels
set OUT_DIR=%SCRIPT_DIR%ptx
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

REM Kernels safe under --use_fast_math (elementwise; no expf/rsqrtf/sin/cos/pow):
set "FAST_MATH=add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv"

REM Kernels requiring --fmad=false for bit-perfect parity with the CPU scalar
REM reference. .NET RyuJIT does NOT emit FMA from `a*b+c` patterns without an
REM explicit MathF.FusedMultiplyAdd call, so leaving nvcc's --fmad=true default
REM produces ~1 ULP precision drift per accumulation versus the CPU's separate
REM mul+add. The Qwen3MoeHybrid recurrence (GDN) compounds those tiny errors
REM over time steps, so the two kernels backing it must be compiled with FMA
REM fusion disabled. Costs minor perf; matches the CPU bit-for-bit.
set "NO_FMA=conv1d_causal gated_delta_net_scan elementwise_f32"

REM Kernels that emit Ampere-only PTX (mma.sync / cp.async) and so MUST be built
REM at compute_86 instead of the global default. These are dispatch-gated to
REM Ampere+ GPUs in C# (the PTX never loads on Turing), so overriding their arch
REM does not affect portability of the rest of the tree. See ARCH POLICY above.
set "ARCH_86=attention_flash_mma"

echo Using nvcc: %NVCC%
echo Compiling CUDA kernels -^> PTX (target: %ARCH%)...

set FAIL=0
for %%F in ("%KERNEL_DIR%\*.cu") do (
    set "BASE=%%~nF"
    set "FAST_FLAG="
    for %%M in (%FAST_MATH%) do (
        if /I "%%~nF"=="%%M" set "FAST_FLAG=--use_fast_math"
    )
    set "FMAD_FLAG="
    for %%M in (%NO_FMA%) do (
        if /I "%%~nF"=="%%M" set "FMAD_FLAG=-fmad=false"
    )
    REM Per-kernel arch override: Ampere-only kernels build at compute_86.
    set "KARCH=%ARCH%"
    for %%M in (%ARCH_86%) do (
        if /I "%%~nF"=="%%M" set "KARCH=compute_86"
    )
    "%NVCC%" -ptx -arch=!KARCH! !FAST_FLAG! !FMAD_FLAG! -allow-unsupported-compiler -o "%OUT_DIR%\!BASE!.ptx" "%%F"
    if errorlevel 1 (
        echo FAILED: %%~nxF
        set FAIL=1
    ) else (
        set "ARCH_NOTE="
        if /I not "!KARCH!"=="%ARCH%" set "ARCH_NOTE= [!KARCH!]"
        if defined FAST_FLAG (
            echo   %%~nxF -^> !BASE!.ptx ^(fast_math^)!ARCH_NOTE!
        ) else (
            if defined FMAD_FLAG (
                echo   %%~nxF -^> !BASE!.ptx ^(precise, no FMA — bit-perfect with CPU^)!ARCH_NOTE!
            ) else (
                echo   %%~nxF -^> !BASE!.ptx ^(precise^)!ARCH_NOTE!
            )
        )
    )
)

if "%FAIL%"=="1" exit /b 1
echo Done. PTX files in %OUT_DIR%
exit /b 0
