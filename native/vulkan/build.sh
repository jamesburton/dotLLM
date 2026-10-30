#!/bin/bash
# Compile all GLSL compute shaders to SPIR-V for the dotLLM Vulkan backend.
# Requires: glslc (ships with the Vulkan SDK — https://vulkan.lunarg.com/).
# Output: native/vulkan/spv/*.spv
#
# SPIR-V is forward-compatible across the target Vulkan version.
# End users do NOT need the Vulkan SDK — .spv blobs ship alongside the .NET
# assembly and are loaded verbatim at runtime. Only shader *authors* need glslc.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/spv"
SHADER_DIR="$SCRIPT_DIR/shaders"

mkdir -p "$OUT_DIR"

# Target env — Vulkan 1.2 is a widely-supported baseline (AMDGPU, Intel, NVIDIA,
# MoltenVK). Bump to vulkan1.3 once VK_NV_cooperative_matrix2 kernels are added.
TARGET_ENV="vulkan1.2"

# Always optimise (-O): all SPIR-V is built with -O so a rebuild is byte-reproducible
# (#1254, #1310). A plain build is ~80% larger on big shaders and is NOT what ships.
GLSLC_FLAGS="-O"

PIN="$(head -n1 "$SCRIPT_DIR/glslc.version" 2>/dev/null | tr -d '')"
GOT="$(glslc --version | head -n1)"
case "$GOT" in "$PIN"*) ;; *) echo "WARNING: glslc '$GOT' != pinned '$PIN' (glslc.version); output bytes will differ from the committed .spv (#1364)" >&2 ;; esac

echo "Compiling GLSL compute shaders -> SPIR-V (target: $TARGET_ENV)..."

for comp_file in "$SHADER_DIR"/*.comp; do
    [ -f "$comp_file" ] || continue
    base=$(basename "$comp_file" .comp)
    glslc --target-env="$TARGET_ENV" $GLSLC_FLAGS -o "$OUT_DIR/$base.spv" "$comp_file"
    echo "  $base.comp -> $base.spv"
done

echo "Done. SPIR-V files in $OUT_DIR/"
