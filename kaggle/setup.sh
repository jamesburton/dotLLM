#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# dotLLM — Kaggle dual-CUDA bring-up (issue #361)
#
# Provisions a Kaggle "GPU T4 ×2" notebook to build and test dotLLM's CUDA path,
# culminating in the cross-device KV-handoff validation (the StagedKvHandoffTransfer
# seam from #360 exercised across two physical GPUs).
#
# Why this works on Kaggle:
#   • The CUDA backend compiles .cu → PTX (compute_75) loaded via the driver API —
#     no CMake shared lib. The T4 IS sm_75, so the PTX runs natively (no JIT surprises).
#   • .NET 10 installs into $HOME with the official dotnet-install.sh (no root needed).
#   • Dual T4s have no P2P/NVLink, so device→host→device staging is the required path —
#     exactly what StagedKvHandoffTransfer implements.
#
# Usage (each notebook cell runs one step; see the .ipynb):
#   bash kaggle/setup.sh env        # versions + GPU enumeration sanity
#   bash kaggle/setup.sh dotnet     # install .NET 10 SDK into $HOME
#   bash kaggle/setup.sh ptx        # compile CUDA kernels → PTX
#   bash kaggle/setup.sh build      # restore + build the solution
#   bash kaggle/setup.sh test-cpu   # CPU parity tests (proves the seam + .NET 10)
#   bash kaggle/setup.sh test-cuda  # dual-device CUDA parity (needs 2 GPUs + #361 impl)
#   bash kaggle/setup.sh bench      # CUDA inference benchmark (prefill+decode tok/s); honours
#                                   # DOTLLM_CUDA_GEMM_16F / DOTLLM_CUDA_G3_ATTN env toggles
#   bash kaggle/setup.sh all        # env → dotnet → ptx → build → test-cpu
#
# NOTE: kernel launches need PTX whose ISA matches the installed driver. The repo's committed PTX is
# ISA 9.1 (CUDA 13.1+); on an older driver that JIT-fails with CUDA error 222. The ptx/build/bench
# steps rebuild PTX with the local toolkit and overwrite the copies in bin/ to avoid that.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Repo + branch. Override with DOTLLM_REPO / DOTLLM_BRANCH env vars.
# Default is the fork that carries the dev-track / #361 branch (upstream kkokosa/dotLLM does not).
REPO_URL="${DOTLLM_REPO:-https://github.com/jamesburton/dotLLM.git}"
# The cross-device seam (#360) lives on dev; the CUDA-side impl (#361) on its branch.
BRANCH="${DOTLLM_BRANCH:-issue/361-kaggle-dual-cuda-validation}"
WORK="${DOTLLM_WORK:-/kaggle/working}"
SRC="${DOTLLM_SRC:-$WORK/dotLLM}"
DOTNET_DIR="${DOTNET_ROOT:-$HOME/.dotnet}"
DOTNET_CHANNEL="${DOTNET_CHANNEL:-10.0}"

export DOTNET_ROOT="$DOTNET_DIR"
export PATH="$DOTNET_DIR:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
export DOTNET_NOLOGO=1

step="${1:-all}"

clone_or_update() {
  if [ ! -d "$SRC/.git" ]; then
    echo "→ cloning $REPO_URL ($BRANCH) into $SRC"
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SRC"
  else
    echo "→ updating $SRC ($BRANCH)"
    git -C "$SRC" fetch --depth 1 origin "$BRANCH"
    git -C "$SRC" checkout "$BRANCH"
    git -C "$SRC" reset --hard "origin/$BRANCH"
  fi
}

do_env() {
  echo "==== OS ===="; uname -a; (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || true
  echo "==== GPUs (expect 2× Tesla T4) ===="
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv || { echo "!! nvidia-smi failed — is the T4×2 accelerator enabled?"; exit 1; }
  GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  echo "GPU count: $GPU_COUNT"
  [ "$GPU_COUNT" -ge 2 ] || echo "!! WARNING: dual-device validation needs 2 GPUs; found $GPU_COUNT. Set accelerator to 'GPU T4 ×2'."
  echo "==== nvcc ===="; nvcc --version || { echo "!! nvcc not found — CUDA toolkit missing on this image."; exit 1; }
}

do_dotnet() {
  if [ -x "$DOTNET_DIR/dotnet" ]; then
    echo "→ .NET already present:"; "$DOTNET_DIR/dotnet" --version; return
  fi
  echo "→ installing .NET SDK channel $DOTNET_CHANNEL into $DOTNET_DIR (no root)"
  curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh
  bash /tmp/dotnet-install.sh --channel "$DOTNET_CHANNEL" --install-dir "$DOTNET_DIR"
  "$DOTNET_DIR/dotnet" --info | head -n 20
}

# Rebuild PTX with the LOCAL toolkit (matching THIS driver) and overwrite any committed/stale PTX
# already copied into build outputs. WHY: the repo commits prebuilt PTX (convenient when nvcc is
# absent), but those are ISA 9.1 (CUDA 13.1+ toolkit); a driver older than that JIT-fails kernel
# launches with CUDA error 222 (UNSUPPORTED_PTX_VERSION). The C# build copies the committed PTX into
# each project's bin/<cfg>/ptx, so the loader would run the stale ISA. Rebuilding locally + syncing to
# every bin/.../ptx makes the PTX ISA match the installed driver on any box.
sync_ptx() {
  bash "$SRC/native/build.sh" >/dev/null
  echo "  fresh PTX $(grep -m1 '^.version' "$SRC/native/ptx/rmsnorm.ptx" 2>/dev/null) (must be <= the driver's max)"
  find "$SRC" -type d -name ptx -path '*/bin/*' | while read -r d; do
    cp -f "$SRC"/native/ptx/*.ptx "$d/" 2>/dev/null || true
  done
}

do_ptx() {
  clone_or_update
  echo "→ compiling CUDA kernels → PTX (compute_75; T4 = sm_75) with the local toolkit"
  sync_ptx
  echo "PTX file count: $(ls -1 "$SRC/native/ptx"/*.ptx | wc -l)"
}

do_build() {
  clone_or_update
  echo "→ dotnet build (Release)"
  "$DOTNET_DIR/dotnet" build "$SRC/dotLLM.sln" -c Release --nologo -v m 2>&1 | tail -n 15 \
    || "$DOTNET_DIR/dotnet" build "$SRC/tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj" -c Release --nologo -v m 2>&1 | tail -n 15
  # Overwrite the committed PTX the build just copied into bin with locally-built, driver-matched PTX.
  sync_ptx
}

do_bench() {
  clone_or_update
  echo "→ CUDA inference benchmark — prefill + decode tok/s."
  echo "   Toggle the Turing-gated tensor-core prefill paths with env: DOTLLM_CUDA_GEMM_16F=1 (G1 FP16 GEMM),"
  echo "   DOTLLM_CUDA_G3_ATTN=1 (G3 cuBLAS TC attention). Model via DOTLLM_BENCH_MODEL_PATH or the filter."
  "$DOTNET_DIR/dotnet" build "$SRC/benchmarks/DotLLM.Benchmarks/DotLLM.Benchmarks.csproj" -c Release --nologo -v q 2>&1 | tail -2
  sync_ptx
  local filter="${DOTLLM_BENCH_FILTER:-*CudaInferenceBenchmarks*}"
  "$DOTNET_DIR/dotnet" run -c Release --no-build --project "$SRC/benchmarks/DotLLM.Benchmarks" -- --filter "$filter" 2>&1 \
    | grep -E "prefill=|decode=|CUDA error|error 222|Model override" | tail -n 6
}

do_test_cpu() {
  echo "→ CPU parity tests (DisaggregatedKvTransferTests — proves the staged seam + .NET 10 on Kaggle)"
  "$DOTNET_DIR/dotnet" test "$SRC/tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj" \
    -c Release --filter "FullyQualifiedName~DisaggregatedKvTransferTests" --nologo 2>&1 | tail -n 12
}

do_test_cuda() {
  echo "→ dual-device CUDA cross-device parity test"
  echo "   (requires the #361 CUDA IHostStagedKvCache impl + a 2-GPU parity test; gated to 2 CUDA devices)"
  "$DOTNET_DIR/dotnet" test "$SRC/tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj" \
    -c Release --filter "FullyQualifiedName~CudaCrossDeviceKvTransferTests" --nologo 2>&1 | tail -n 15
}

case "$step" in
  env)        do_env ;;
  dotnet)     do_dotnet ;;
  ptx)        do_ptx ;;
  build)      do_build ;;
  test-cpu)   do_test_cpu ;;
  test-cuda)  do_test_cuda ;;
  bench)      do_bench ;;
  all)        do_env; do_dotnet; do_ptx; do_build; do_test_cpu ;;
  *) echo "unknown step '$step' (env|dotnet|ptx|build|test-cpu|test-cuda|bench|all)"; exit 2 ;;
esac
echo "✓ step '$step' done"
