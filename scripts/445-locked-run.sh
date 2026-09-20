#!/usr/bin/env bash
# #445 — run a command while holding the GPU lock, releasing it from a trap on every path.
#
# The lock is taken in the FOREGROUND with a short wait. A backgrounded or timed-out
# `gpu-lock.sh acquire` has twice kept waiting and silently taken the GPU long after the
# agent that asked for it had moved on; failing fast is the only safe shape here.
#
# Usage: scripts/445-locked-run.sh "<reason>" <command...>

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"
REASON="${1:?usage: 445-locked-run.sh \"<reason>\" <command...>}"
shift

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "$REASON" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

"$@"
rc=$?

echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

exit $rc
