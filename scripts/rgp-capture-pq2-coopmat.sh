#!/usr/bin/env bash
# Issue #440 — capture an RGP SQTT profile of the shipping PQ2_0 coopmat32 GEMM on lm_head.
#
# Headless: RadeonDeveloperPanelCLI attaches to the test host and auto-captures one compute
# dispatch by index. The capture target (VulkanPQ2_0CoopmatCaptureBench) dispatches ONLY
# matmul_pq2_0_f32_gemm_coopmat32, one dispatch per submit, so the index identifies the kernel.
#
# Usage: scripts/rgp-capture-pq2-coopmat.sh [output.rgp]
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RDTS="/c/Development/tools/RadeonDeveloperToolSuite"
RDTS_WIN="C:\Development\tools\RadeonDeveloperToolSuite"
OUT="${1:-$REPO/.docs/440-lmhead-coopmat32.rgp}"
OUT_WIN="$(cygpath -w "$OUT" 2>/dev/null || echo "$OUT")"
PANEL_LOG="${OUT%.rgp}-panel.log"
LOCK_ID="a440"

# Panel knobs — SQTT volume is the failure mode, so these are dialled down per attempt.
#   RGP_EXTRA: extra PanelCLI args, PowerShell list syntax, e.g. "'--rgp-counter-collection',"
RGP_AUTO="${RGP_AUTO:-dispatch:8:1}"
RGP_BUFFER="${RGP_BUFFER:-maximum}"
RGP_EXTRA="${RGP_EXTRA:-'--rgp-instruction-tracing','--rgp-counter-collection',}"

mkdir -p "$(dirname "$OUT")"

cleanup() {
  powershell.exe -NoProfile -Command \
    "Get-Process RadeonDeveloperPanelCLI,RadeonDeveloperServiceCLI,RadeonDeveloperService -ErrorAction SilentlyContinue | Stop-Process -Force" \
    >/dev/null 2>&1
  bash "$REPO/scripts/gpu-lock.sh" release "$LOCK_ID" >/dev/null 2>&1
  echo "[cleanup] panel killed, gpu lock released"
}
trap cleanup EXIT INT TERM

echo "=== process census BEFORE ==="
powershell.exe -NoProfile -Command "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,Id,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize | Out-String -Width 120"

echo "=== acquiring gpu lock ==="
bash "$REPO/scripts/gpu-lock.sh" acquire "$LOCK_ID" "RGP SQTT capture of the PQ2_0 coopmat32 GEMM (#440)" 1800 || exit 1

echo "=== starting RadeonDeveloperPanelCLI (auto-capture dispatch:8:1) ==="
powershell.exe -NoProfile -Command \
  "Start-Process -FilePath '$RDTS_WIN\RadeonDeveloperPanelCLI.exe' -WorkingDirectory '$RDTS_WIN' -WindowStyle Hidden \
     -RedirectStandardOutput '$(cygpath -w "$PANEL_LOG")' -RedirectStandardError '$(cygpath -w "${PANEL_LOG%.log}-err.log")' \
     -ArgumentList '--mode','profiling','-p','testhost','--rgp-capture-mode','dispatch', \
                   '--rgp-auto-capture','$RGP_AUTO','--rgp-render-op-count','1', \
                   $RGP_EXTRA '--rgp-sqtt-buffer-size','$RGP_BUFFER','--verbose','-o','$OUT_WIN'"
sleep 4
powershell.exe -NoProfile -Command "Get-Process RadeonDeveloperPanelCLI -ErrorAction SilentlyContinue | Select-Object Id,ProcessName | Format-Table -AutoSize | Out-String"

echo "=== running the capture target ==="
export DOTLLM_PQ2_0_CAPTURE=1
export DOTLLM_PQ2_0_CAPTURE_TOKENS="${DOTLLM_PQ2_0_CAPTURE_TOKENS:-32}"
export DOTLLM_PQ2_0_CAPTURE_HOLD_MS="${DOTLLM_PQ2_0_CAPTURE_HOLD_MS:-2000}"
cd "$REPO"
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter "FullyQualifiedName~VulkanPQ2_0CoopmatCaptureBench" --logger "console;verbosity=detailed" 2>&1 \
  | grep -v -E '^\s*$' | tail -60

echo "=== waiting up to 60 s for the capture to be written ==="
for _ in $(seq 1 60); do
  [ -s "$OUT" ] && break
  sleep 1
done

echo "=== process census AFTER ==="
powershell.exe -NoProfile -Command "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,Id,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize | Out-String -Width 120"

echo "=== panel log ==="
sed -n 1,80p "$PANEL_LOG" 2>/dev/null
sed -n 1,20p "${PANEL_LOG%.log}-err.log" 2>/dev/null

echo "=== output ==="
ls -la "$(dirname "$OUT")" | grep -i rgp || echo "NO .rgp PRODUCED"
