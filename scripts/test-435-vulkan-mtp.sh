#!/usr/bin/env bash
# Issue #435 — Vulkan MTP ("NextN") correctness validation.
#
# Runs the targeted MTP test set only: the CPU MTP suite (the oracle), the new Vulkan/CPU MTP
# parity suite, and the Vulkan hybrid-dense GGUF loader suite (which the all-row-logits change
# touches). It does NOT run the full 1352-test Vulkan sweep and takes NO throughput measurements.
#
# GPU access is serialised through scripts/gpu-lock.sh — several agents share this box.
#
# Usage:  bash scripts/test-435-vulkan-mtp.sh [acquire-timeout-sec]
set -u

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOCK_TIMEOUT="${1:-3600}"
LOCK_NAME="a435"

cd "$REPO" || exit 2

echo "== Building (library AND test project — a stale tests/bin silently tests old binaries) =="
dotnet build tests/DotLLM.Tests.Unit -v q --nologo || exit 1

echo
echo "== CPU MTP oracle suite (no GPU, no lock needed) =="
dotnet test tests/DotLLM.Tests.Unit --no-build \
    --filter "FullyQualifiedName~Models.Architectures.Qwen3HybridDenseMtpTests" \
    || echo "!! CPU MTP suite FAILED"

echo
echo "== Acquiring the GPU lock (timeout ${LOCK_TIMEOUT}s) =="
if ! bash scripts/gpu-lock.sh acquire "$LOCK_NAME" "issue 435 Vulkan MTP correctness" "$LOCK_TIMEOUT"; then
    echo "!! Could not acquire the GPU lock — skipping every GPU test."
    exit 1
fi

cleanup() { bash scripts/gpu-lock.sh release "$LOCK_NAME" >/dev/null 2>&1; }
trap cleanup EXIT

echo
echo "== Vulkan MTP parity vs the CPU oracle =="
dotnet test tests/DotLLM.Tests.Unit --no-build \
    --filter "FullyQualifiedName~Vulkan.VulkanQwen3HybridDenseMtpTests"
VK_MTP=$?

echo
echo "== Vulkan hybrid-dense GGUF loader (regression surface for the all-row logits change) =="
dotnet test tests/DotLLM.Tests.Unit --no-build \
    --filter "FullyQualifiedName~Vulkan.VulkanQwen3HybridDenseGgufLoaderTests"
VK_LOADER=$?

echo
echo "================ SUMMARY ================"
echo "Vulkan MTP parity suite   exit=$VK_MTP    (expect 0 / all passed)"
echo "Vulkan loader suite       exit=$VK_LOADER (expect 0 / all passed)"
echo
echo "Expected output:"
echo "  VulkanQwen3HybridDenseMtpTests  — 5 passed, 0 failed"
echo "     Model_WithMtpCheckpoint_ExposesMtpAndRecurrentCheckpoint"
echo "     Model_WithoutMtpCheckpoint_ReportsNoMtpSupport"
echo "     Forward_ShortBatch_ReturnsOneLogitRowPerPosition_MatchingCpu"
echo "     ForwardMtp_DraftTokens_MatchCpuOracle(mtpHasOwnHeadTensors: True)"
echo "     ForwardMtp_DraftTokens_MatchCpuOracle(mtpHasOwnHeadTensors: False)"
echo "  VulkanQwen3HybridDenseGgufLoaderTests — all passed"
echo
echo "Known pre-existing failures elsewhere in the Vulkan suite (do NOT chase, not touched here):"
echo "  6x VulkanMatMulI2SGemvF32KernelTests.MultiRow_IsBitIdenticalToProduction"
echo "  1x VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward (flaky, #433)"

exit $(( VK_MTP != 0 || VK_LOADER != 0 ))
