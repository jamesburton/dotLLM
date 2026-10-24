#!/usr/bin/env bash
# #445 — parity for every GDN scan factorial arm, under the GPU lock.
#
# Asserts each arm is BIT-IDENTICAL to the shipping kernel (not merely within the 4 ULP
# the CPU-oracle theory allows), plus the existing CPU-oracle parity theory.
#
# The lock is taken in the FOREGROUND with a short wait and released from a trap. A
# backgrounded or timed-out acquire has twice kept waiting and silently taken the GPU
# long after the agent that asked for it had moved on.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 GDN scan arm parity" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter 'FullyQualifiedName~VulkanGdnScanMultiTokenF32KernelTests' 2>&1 | tail -30
rc=${PIPESTATUS[0]}

echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

exit $rc
