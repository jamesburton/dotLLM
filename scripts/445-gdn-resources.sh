#!/usr/bin/env bash
# #445 — driver-reported post-compile resources for every GDN-scan arm (VK_AMD_shader_info).
# Hardware data, not timing: tests whether occupancy, rather than the memory level, explains the
# A/B. Takes the GPU lock because it creates a device and compiles pipelines.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 GDN arm shader-resource probe" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter 'FullyQualifiedName~VulkanGdnScanVariantBench.ShaderResources_PerArm' \
  --logger 'console;verbosity=detailed' 2>&1 | grep -v -E '^\s*$' | tail -25
