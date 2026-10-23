#!/usr/bin/env bash
# #445 — RGP SQTT capture of ONE GDN-scan factorial arm.
#
# Derived from scripts/rgp-capture-pq2-coopmat.sh (#440), which established that RDP captures
# headlessly on gfx1151 with no swapchain. Two things carried over because they cost GPU time
# to rediscover:
#   - the output path must be ABSOLUTE: the panel's working directory is the RDTS folder, and a
#     relative -o reports "Failed to write trace file" AFTER a successful capture;
#   - do NOT start RadeonDeveloperServiceCLI separately — the Panel is its own router.
#
# The capture target runs a SINGLE arm (DOTLLM_GDN_SCAN_AB_ARM), one dispatch per submit-and-wait,
# so dispatch index == submit index and the index identifies the kernel unambiguously.
#
# seqLen defaults to 64 rather than 512: SQTT volume scales with the traced instruction count and
# the scan issues ~1000 instructions per token per wave. The instruction MIX is per-token
# identical, so the ratio transfers; the absolute hit counts do not.
#
# Usage: scripts/445-rgp-gdn.sh <arm> [output.rgp]
#   arm = Baseline | Fused | Lds | LdsFused

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RDTS_WIN="C:\\Development\\tools\\RadeonDeveloperToolSuite"
ARM="${1:?usage: 445-rgp-gdn.sh <Baseline|Fused|Lds|LdsFused> [out.rgp]}"
OUT="${2:-$REPO/.docs/445-gdn-$ARM.rgp}"
OUT_WIN="$(cygpath -w "$OUT" 2>/dev/null || echo "$OUT")"
PANEL_LOG="${OUT%.rgp}-panel.log"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"

# warmups=2 then rounds=1 => dispatches 0,1 are warmups and dispatch 2 is the measured one.
export DOTLLM_GDN_SCAN_AB=1
export DOTLLM_GDN_SCAN_AB_ARM="$ARM"
export DOTLLM_GDN_SCAN_AB_SEQ="${DOTLLM_GDN_SCAN_AB_SEQ:-64}"
export DOTLLM_GDN_SCAN_AB_WARMUPS="${DOTLLM_GDN_SCAN_AB_WARMUPS:-2}"
# A long dispatch stream, not a short one: at ~20 ms per dispatch, 200 rounds keeps the target
# issuing for several seconds so the panel is certainly armed well before the captured index.
export DOTLLM_GDN_SCAN_AB_ROUNDS="${DOTLLM_GDN_SCAN_AB_ROUNDS:-200}"
# ...and hold before the first dispatch, mirroring #440's DOTLLM_PQ2_0_CAPTURE_HOLD_MS. A target
# that finishes inside the panel's connect handshake fails with an error that reads like an SQTT
# overflow but is not one.
export DOTLLM_GDN_SCAN_AB_HOLD_MS="${DOTLLM_GDN_SCAN_AB_HOLD_MS:-4000}"
RGP_AUTO="${RGP_AUTO:-dispatch:100:1}"
RGP_EXTRA="${RGP_EXTRA:-'--rgp-instruction-tracing','--rgp-counter-collection',}"

mkdir -p "$(dirname "$OUT")"

cleanup() {
  powershell.exe -NoProfile -Command \
    "Get-Process RadeonDeveloperPanelCLI,RadeonDeveloperServiceCLI,RadeonDeveloperService -ErrorAction SilentlyContinue | Stop-Process -Force" \
    >/dev/null 2>&1
  bash "$REPO/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1
  echo "[cleanup] panel killed, gpu lock released"
}
trap cleanup EXIT INT TERM

echo "=== process census BEFORE ==="
powershell.exe -NoProfile -Command "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,Id,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize | Out-String -Width 120"

bash "$REPO/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 RGP capture of the GDN scan ($ARM)" "$WAIT" || exit 1

echo "=== starting RadeonDeveloperPanelCLI (auto-capture $RGP_AUTO) ==="
powershell.exe -NoProfile -Command \
  "Start-Process -FilePath '$RDTS_WIN\\RadeonDeveloperPanelCLI.exe' -WorkingDirectory '$RDTS_WIN' -WindowStyle Hidden \
     -RedirectStandardOutput '$(cygpath -w "$PANEL_LOG")' -RedirectStandardError '$(cygpath -w "${PANEL_LOG%.log}-err.log")' \
     -ArgumentList '--mode','profiling','-p','testhost','--rgp-capture-mode','dispatch', \
                   '--rgp-auto-capture','$RGP_AUTO','--rgp-render-op-count','1', \
                   $RGP_EXTRA '--rgp-sqtt-buffer-size','maximum','--verbose','-o','$OUT_WIN'"
sleep 4

echo "=== running the capture target (arm=$ARM seq=$DOTLLM_GDN_SCAN_AB_SEQ) ==="
cd "$REPO"
dotnet test tests/DotLLM.Tests.Unit -c Release --no-build \
  --filter 'FullyQualifiedName~VulkanGdnScanVariantBench' --logger 'console;verbosity=detailed' 2>&1 \
  | grep -v -E '^\s*$' | tail -30

echo "=== waiting up to 60 s for the capture ==="
for _ in $(seq 1 60); do [ -s "$OUT" ] && break; sleep 1; done

echo "=== process census AFTER ==="
powershell.exe -NoProfile -Command "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,Id,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize | Out-String -Width 120"

echo "=== panel log ==="
sed -n 1,60p "$PANEL_LOG" 2>/dev/null
sed -n 1,20p "${PANEL_LOG%.log}-err.log" 2>/dev/null

echo "=== output ==="
ls -la "$(dirname "$OUT")" | grep -i rgp || echo "NO .rgp PRODUCED"
