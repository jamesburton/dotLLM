#!/usr/bin/env bash
# #445 — interleaved, order-reversed A/B of the four GDN-scan factorial arms, under the GPU lock.
#
# All four arms run inside ONE process (see VulkanGdnScanVariantBench remarks): per-arm process
# launches cannot control for GPU clock ramp or UMA memory contention, both of which are
# process-scoped and have produced 2-3x phantom deltas on this box before.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 GDN scan 2x2 A/B" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
export DOTLLM_GDN_SCAN_AB=1
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter 'FullyQualifiedName~VulkanGdnScanVariantBench' \
  --logger 'console;verbosity=detailed' 2>&1 | grep -v -E '^\s*$' | tail -45
rc=${PIPESTATUS[0]}

echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

exit $rc
