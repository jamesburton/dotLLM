#!/usr/bin/env bash
# #445 — the accounting cover for the per-op profiler, under the GPU lock.
#
# This is the test that guards the sub-bucket split itself: CategoryNames_CoverEveryCategory
# pins the enum against the name table (six values were added), and the round-trip theory
# asserts that the graph reaches gdn_scan_core / attn_core and that Format() still rolls them
# back up to the #434 parent names.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 profiler accounting cover" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter 'FullyQualifiedName~VulkanHybridDenseOpProfilerTests' \
  --logger 'console;verbosity=detailed' 2>&1 | grep -v -E '^\s*$' | tail -35
