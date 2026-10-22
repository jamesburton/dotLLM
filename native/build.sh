#!/bin/bash
# Compile all .cu kernels to PTX for dotLLM CUDA backend.
# Requires: nvcc (CUDA Toolkit)
# Output: native/ptx/*.ptx
#
# PTX is forward-compatible: compute_75 PTX runs on all GPUs from Turing onward.
# The CUDA driver JIT-compiles PTX → SASS for the specific GPU at load time.
# CUDA 13 dropped Pascal (SM 6.x) and Volta (SM 7.0); Turing (SM 7.5) is the floor.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/ptx"
KERNEL_DIR="$SCRIPT_DIR/kernels"

mkdir -p "$OUT_DIR"

# Target the CUDA-13 floor (Turing / RTX 20xx). Driver JITs to the actual GPU's
# native ISA at load time (Ampere / Ada / Hopper / Blackwell all derive from this).
ARCH="compute_75"

# The PTX ISA version every committed artifact must declare (CUDA 12.8 emits 8.7).
# PTX whose .version exceeds the driver's is rejected outright with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION — CUDA 13.1 emits 9.1, unloadable on any
# pre-13.1 driver. This has regressed twice from a newer toolkit being picked up
# silently (#124, #318), so every generated file is checked here rather than
# discovered later by a user on an older driver. Override only when deliberately
# moving the whole tree to a new toolkit baseline.
EXPECT_PTX_VERSION="${DOTLLM_PTX_EXPECT_VERSION:-8.7}"

# assert_ptx_version <ptx file> — fail the build if the emitted ISA version is
# not the expected baseline, naming the toolkit that produced it.
assert_ptx_version() {
    local file="$1"
    local actual
    actual=$(grep -m1 '^\.version' "$file" | awk '{print $2}')
    if [ "$actual" != "$EXPECT_PTX_VERSION" ]; then
        echo "ERROR: $(basename "$file") declares .version $actual, expected $EXPECT_PTX_VERSION." >&2
        echo "       nvcc in use: $(command -v nvcc)" >&2
        nvcc --version 2>/dev/null | tail -2 >&2
        echo "       A committed PTX file at the wrong ISA version fails to load with" >&2
        echo "       CUDA_ERROR_UNSUPPORTED_PTX_VERSION on older drivers (see #124, #318)." >&2
        echo "       Point nvcc at the CUDA 12.8 toolkit, or set DOTLLM_PTX_EXPECT_VERSION" >&2
        echo "       if you are deliberately re-baselining the whole tree." >&2
        exit 1
    fi
}

# Kernels where --use_fast_math is safe (element-wise ops, no precision-sensitive math):
FAST_MATH_KERNELS="add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv"

# Kernels requiring precise math (expf, rsqrtf, sinf, cosf, powf):
# - softmax/attention: expf in softmax accumulates error
# - rmsnorm/fused_add_rmsnorm: rsqrtf precision matters
# - rope: sinf/cosf/powf precision matters for position encoding
# - quantized_gemv: feeds precision-sensitive downstream ops
PRECISE_KERNELS="softmax rmsnorm rmsnorm_f32 rmsnorm_f32in rope rope_f32 attention attention_f32 fused_add_rmsnorm per_head_rmsnorm per_head_rmsnorm_f32 quantized_gemv quantized_gemv_f32in"

# Kernels that emit Ampere-only PTX (mma.sync / cp.async) and so MUST be built at
# compute_86 instead of the global default (dispatch-gated to Ampere+ in C#).
# Mirrors build_ptx.bat's ARCH_86 list.
ARCH_86_KERNELS="attention_flash_mma attention_flash_mma_decode_gqa_split attention_flash_mma_decode_gqa_split_kvq attention_flash_mma_prefill_hd256 attention_flash_mma_prefill_hd256_kvq attention_flash_mma_prefill_hd256_xacc"

# Kernels that #include <cooperative_groups.h> (libcu++ needs C++17).
# Mirrors build_ptx.bat's CXX17 list.
CXX17_KERNELS="gated_delta_net_scan attention_f32 attention_flash_mma_decode_gqa_split attention_flash_mma_decode_gqa_split_kvq attention_flash_mma_prefill_hd256 attention_flash_mma_prefill_hd256_kvq attention_flash_mma_prefill_hd256_xacc"

# in_list <name> <space-separated list> — success when name is in the list.
in_list() {
    local name="$1"
    for item in $2; do
        [ "$item" = "$name" ] && return 0
    done
    return 1
}

is_fast_math_kernel() {
    in_list "$1" "$FAST_MATH_KERNELS"
}

echo "Compiling CUDA kernels → PTX (target: $ARCH)..."

for cu_file in "$KERNEL_DIR"/*.cu; do
    [ -f "$cu_file" ] || continue
    base=$(basename "$cu_file" .cu)

    karch="$ARCH"
    in_list "$base" "$ARCH_86_KERNELS" && karch="compute_86"
    cxx17=""
    in_list "$base" "$CXX17_KERNELS" && cxx17="-std=c++17"

    if is_fast_math_kernel "$base"; then
        nvcc -ptx -arch="$karch" $cxx17 \
             --use_fast_math \
             -o "$OUT_DIR/$base.ptx" \
             "$cu_file"
        assert_ptx_version "$OUT_DIR/$base.ptx"
        echo "  $base.cu → $base.ptx (fast_math)"
    else
        nvcc -ptx -arch="$karch" $cxx17 \
             -o "$OUT_DIR/$base.ptx" \
             "$cu_file"
        assert_ptx_version "$OUT_DIR/$base.ptx"
        echo "  $base.cu → $base.ptx (precise)"
    fi
done

echo "Done. PTX files in $OUT_DIR/"
