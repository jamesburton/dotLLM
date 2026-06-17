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
REM GeForce-Ampere-gated, so its PTX never loads on Turing). This script is
REM global-arch today; add per-kernel arch metadata when the first Ampere-only
REM kernel ships (issue #70 — fused mma.sync flash attention). Keep the
REM committed native/ptx tree uniform sm_75 except such gated kernels.
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

REM Locate a CUDA-compatible host MSVC toolchain (cl.exe under VC\Tools\MSVC\
REM <ver>\bin\Hostx64\x64). nvcc needs this on PATH to compile .cu -> .ptx.
REM All discovery is done in the :find_msvc subroutine at the end of this file
REM (subroutine lines parse independently, so paths containing "(x86)" and the
REM delayed-expansion blocks can't trip cmd's paren matching).
call :find_msvc
if errorlevel 1 exit /b 1
echo Using host MSVC: %MSVC_BIN%
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
    "%NVCC%" -ptx -arch=%ARCH% !FAST_FLAG! !FMAD_FLAG! -allow-unsupported-compiler -o "%OUT_DIR%\!BASE!.ptx" "%%F"
    if errorlevel 1 (
        echo FAILED: %%~nxF
        set FAIL=1
    ) else (
        if defined FAST_FLAG (
            echo   %%~nxF -^> !BASE!.ptx ^(fast_math^)
        ) else (
            if defined FMAD_FLAG (
                echo   %%~nxF -^> !BASE!.ptx ^(precise, no FMA — bit-perfect with CPU^)
            ) else (
                echo   %%~nxF -^> !BASE!.ptx ^(precise^)
            )
        )
    )
)

if "%FAIL%"=="1" exit /b 1
echo Done. PTX files in %OUT_DIR%
exit /b 0

REM ===================================================================
REM :find_msvc — locate a CUDA-compatible host MSVC toolchain.
REM Sets MSVC_BIN (the ...\bin\Hostx64\x64 directory containing cl.exe).
REM Returns 0 on success, 1 if no usable toolchain was found.
REM
REM Selection rationale (CUDA 13.x host compiler support):
REM   - MSVC 14.50+ (_MSC_VER >= 1950, shipped with VS 2026 / 18.x) is rejected
REM     by CUDA 13.1's host_config.h and nvcc's OS-target check, so we must NOT
REM     pick the numerically highest toolset when a 14.50+ is present.
REM   - We therefore prefer the highest toolset STRICTLY BELOW 14.50 (the 14.3x-
REM     14.4x VS 2022 range), and only fall back to a 14.50+ if nothing else
REM     exists (relying on -allow-unsupported-compiler).
REM
REM Discovery order:
REM   1. Honor a pre-set MSVC_BIN (full bin dir) or MSVC_DIR (toolset root).
REM   2. vswhere -find across ALL instances (no -latest: on this kind of box
REM      -latest can resolve to SSMS or a VS 2026 instance with no CUDA-usable
REM      VC tools). -products * is required because Build Tools' product id is
REM      excluded from vswhere's default query.
REM   3. Fallback: scan standard VS 2019/2022 install roots for VC\Tools\MSVC.
REM ===================================================================
:find_msvc
setlocal EnableDelayedExpansion

REM Capture the Program Files roots into plain vars. %ProgramFiles(x86)% has a
REM literal ')' in its name; referencing it inside ( ... ) blocks miscounts the
REM closing paren, so we only use these flat vars from here on.
set "PF_X86=%ProgramFiles(x86)%"
set "PF_64=%ProgramFiles%"

REM 1. Explicit override wins.
if not defined MSVC_BIN if defined MSVC_DIR set "MSVC_BIN=%MSVC_DIR%\bin\Hostx64\x64"
if defined MSVC_BIN goto :find_msvc_done

REM 2. vswhere: enumerate every cl.exe, pick the best CUDA-compatible version.
set "VSWHERE=!PF_X86!\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "!VSWHERE!" (
    for /f "usebackq delims=" %%I in (`""!VSWHERE!" -products * -find "VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe""`) do (
        call :consider_cl "%%I"
    )
)
if defined MSVC_BIN goto :find_msvc_done

REM 3. Fallback: scan the standard install roots directly (flat calls, no
REM parenthesized data block, so "(x86)" can't break parsing).
call :scan_root "!PF_64!\Microsoft Visual Studio\2022\BuildTools"
call :scan_root "!PF_64!\Microsoft Visual Studio\2022\Community"
call :scan_root "!PF_64!\Microsoft Visual Studio\2022\Professional"
call :scan_root "!PF_64!\Microsoft Visual Studio\2022\Enterprise"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2022\BuildTools"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2022\Community"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2022\Professional"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2022\Enterprise"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2019\BuildTools"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2019\Community"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2019\Professional"
call :scan_root "!PF_X86!\Microsoft Visual Studio\2019\Enterprise"

:find_msvc_done
if not defined MSVC_BIN (
    echo Could not locate a host MSVC toolchain via override, vswhere, or the
    echo standard VS install roots. Install VS 2022 Build Tools, or pre-set
    echo MSVC_BIN / MSVC_DIR before invoking this script.
    endlocal
    exit /b 1
)
if not exist "!MSVC_BIN!\cl.exe" (
    echo cl.exe not found at !MSVC_BIN!
    endlocal
    exit /b 1
)
REM Propagate MSVC_BIN out of the local scope to the caller. The FOR carries the
REM delayed-expanded value across the endlocal barrier on a single command line.
for /f "delims=" %%V in ("!MSVC_BIN!") do (endlocal & set "MSVC_BIN=%%V")
exit /b 0

REM :consider_cl <full path to cl.exe>
REM Records this toolset as the best so far if it is CUDA-compatible and higher
REM than any previously seen. "best" = highest version strictly below 14.50;
REM a 14.50+ is only kept if no sub-14.50 has been found yet.
:consider_cl
set "CL_PATH=%~1"
set "CL_BIN=%~dp1"
if "!CL_BIN:~-1!"=="\" set "CL_BIN=!CL_BIN:~0,-1!"
REM Extract the MSVC version dir: ...\VC\Tools\MSVC\<ver>\bin\Hostx64\x64\cl.exe
for %%P in ("!CL_BIN!\..\..\..") do set "CL_VER=%%~nxP"
REM Only consider 14.* toolsets.
if "!CL_VER:~0,3!" neq "14." exit /b 0
REM Build a sortable key. minor < 50 => CUDA-compatible (rank 1, preferred);
REM minor >= 50 => incompatible-but-usable-with-override (rank 0, last resort).
for /f "tokens=1,2 delims=." %%a in ("!CL_VER!") do set "CL_MINOR=%%b"
set "CL_RANK=1"
if !CL_MINOR! geq 50 set "CL_RANK=0"
if not defined BEST_RANK goto :consider_cl_take
REM Higher rank always wins; within a rank, higher version string wins.
if !CL_RANK! gtr !BEST_RANK! goto :consider_cl_take
if !CL_RANK! lss !BEST_RANK! exit /b 0
if "!CL_VER!" gtr "!BEST_VER!" goto :consider_cl_take
exit /b 0
:consider_cl_take
set "MSVC_BIN=!CL_BIN!"
set "BEST_RANK=!CL_RANK!"
set "BEST_VER=!CL_VER!"
exit /b 0

REM :scan_root <VS install root>
REM Adds every 14.* toolset found under <root>\VC\Tools\MSVC to consideration.
:scan_root
set "SR_ROOT=%~1"
if not exist "!SR_ROOT!\VC\Tools\MSVC\" exit /b 0
for /d %%D in ("!SR_ROOT!\VC\Tools\MSVC\14.*") do call :consider_cl "%%D\bin\Hostx64\x64\cl.exe"
exit /b 0
