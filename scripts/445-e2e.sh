#!/usr/bin/env bash
# #445 — end-to-end pp512 confirmation of a GDN-scan arm on Bonsai 2.
#
# The kernel A/B (VulkanGdnScanVariantBench) is the trustworthy measurement because it runs
# every arm in one process. This script is the SEPARATE question of whether the kernel win shows
# up in the whole pass, and it necessarily costs one process launch per arm — so read the
# gdn_scan_core BUCKET ratio, which is what the change can move, and treat end-to-end tok/s as
# corroboration rather than the claim.
#
# Usage: scripts/445-e2e.sh <base|fused|lds|ldsfused|lds64|lds64fused> [out.txt]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:?usage: 445-e2e.sh <base|fused|lds|ldsfused|lds64|lds64fused> [out.txt]}"
OUT="${2:-$ROOT/.docs/445-prefill-$ARM.txt}"
AGENT="${GPU_LOCK_AGENT:-a445}"
WAIT="${GPU_LOCK_WAIT:-240}"
MODEL="${DOTLLM_BONSAI2_GGUF:-$HOME/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PQ2_0.gguf}"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 e2e pp512 ($ARM)" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
DOTLLM_VK_GDN_SCAN_VARIANT="$ARM" \
DOTLLM_VULKAN_HYBRID_PROFILE=1 \
DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ=2 \
DOTLLM_VULKAN_HYBRID_PROFILE_OUT="$OUT" \
  dotnet run --project "$ROOT/src/DotLLM.Cli" -c Release --no-build -- \
    bench "$MODEL" --device vulkan -p 512 -n 8 -r 2

echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"
