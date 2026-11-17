#!/bin/bash
# Compile all .hip kernels to code objects (.co) for dotLLM HIP backend.
# Requires: hipcc (ROCm)
# Output: native/hip/co/*.co
#
# hipcc --genco produces a fat code-object ELF containing bundled AMDGPU ISA for
# each --offload-arch target. The HIP runtime picks the matching arch at
# hipModuleLoadData time.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/co"
KERNEL_DIR="$SCRIPT_DIR/kernels"

mkdir -p "$OUT_DIR"

# Default architectures: RDNA2 (gfx1030, RX 6000), RDNA3 (gfx1100, RX 7000).
# Override with HIP_ARCHS env var, e.g. HIP_ARCHS="gfx942 gfx90a" for MI300/MI250.
ARCHS="${HIP_ARCHS:-gfx1030 gfx1100 gfx1151}"
HIP_COMPILER="${HIP_COMPILER:-hipcc}"

OFFLOAD_FLAGS=()
for a in $ARCHS; do
    OFFLOAD_FLAGS+=("--offload-arch=$a")
done

if ! command -v "$HIP_COMPILER" >/dev/null 2>&1; then
    echo "hipcc not found in PATH. Install ROCm to build HIP kernels." >&2
    exit 1
fi

echo "Compiling HIP kernels → code objects (archs: $ARCHS)..."
COMPILER_VERSION="$("$HIP_COMPILER" --version)"
echo "$COMPILER_VERSION"

shopt -s nullglob
sources=("$KERNEL_DIR"/*.hip)
if [ "${#sources[@]}" -eq 0 ]; then
    echo "No HIP sources found in $KERNEL_DIR" >&2
    exit 1
fi

for src in "${sources[@]}"; do
    name="$(basename "$src" .hip)"
    out="$OUT_DIR/$name.co"
    echo "  hipcc --genco ${OFFLOAD_FLAGS[*]} -o $out $src"
    "$HIP_COMPILER" --genco "${OFFLOAD_FLAGS[@]}" -o "$out" "$src"
done

{
    echo "Built UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Platform: $(uname -s) $(uname -m)"
    echo "Targets (each object): $ARCHS"
    echo 'Command: hipcc --genco <targets> -o co/<name>.co kernels/<name>.hip'
    echo "Compiler: $COMPILER_VERSION"
    echo 'SHA256  file'
    cd "$OUT_DIR"
    for src in "${sources[@]}"; do
        name="$(basename "$src" .hip)"
        sha256sum "$name.co" "../kernels/$name.hip"
    done
} > "$OUT_DIR/MANIFEST"
echo "Done. Code objects and MANIFEST in $OUT_DIR"
