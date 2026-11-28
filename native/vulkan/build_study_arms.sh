#!/usr/bin/env bash
# Builds the activation-precision STUDY arms of the Inkling 2 Vulkan vision tower (epic #2025): the F16 / F32 activation twins of the vit_* shaders and the
# F32-RoPE-table twins of the study GEMM. Compiled with command-line defines (the shipped vit_*.comp are untouched; their BF16 spv stay the default).
set -euo pipefail
cd "$(dirname "$0")/shaders"
OUT=../spv
G="glslc --target-env=vulkan1.2 -O"
for arm in f16act f32act; do
  D=-DACT_${arm%act}; D=$(echo $D | tr a-z A-Z)
  for k in vit_cls vit_layernorm vit_patch_embd vit_attn; do $G $D -o $OUT/${k}_$arm.spv $k.comp; done
  $G $D -o $OUT/vit_gemm_q8_study_$arm.spv vit_gemm_q8_study.comp
  $G $D -DROPE_F32 -o $OUT/vit_gemm_q8_study_${arm}_rope32.spv vit_gemm_q8_study.comp
done
$G -o $OUT/vit_gemm_q8_study.spv vit_gemm_q8_study.comp
$G -DROPE_F32 -o $OUT/vit_gemm_q8_study_rope32.spv vit_gemm_q8_study.comp
echo done
