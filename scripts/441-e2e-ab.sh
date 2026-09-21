#!/usr/bin/env bash
# #441 - end-to-end pp512 A/B of the wide-head (headDim 256) flash-attention variant on Bonsai 2.
#
# The trustworthy measurement is the kernel A/B (VulkanFlashAttentionHd256VariantBench): it runs
# every arm inside ONE process, interleaved and order-reversed, which is the only defence against
# this box's two process-scoped confounds (UMA memory-bandwidth contention moves absolute
# throughput ~40 %, and a cold-vs-warm launch can fake a 2-3x delta from GPU clock ramp).
#
# This script answers the SEPARATE question of whether the kernel win survives into the whole
# pass, and it necessarily costs one process launch per arm. So: it runs the arms in BOTH orders
# (off, on, on, off), and you read the attn_core BUCKET ratio - the only thing this change can
# move - with end-to-end tok/s as corroboration, never as the claim.
#
# Usage: scripts/441-e2e-ab.sh [outdir]

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="${1:-$ROOT/.docs}"
AGENT="${GPU_LOCK_AGENT:-a441}"
WAIT="${GPU_LOCK_WAIT:-600}"
MODEL="${DOTLLM_BONSAI2_GGUF:-$HOME/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PQ2_0.gguf}"

mkdir -p "$OUTDIR"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 441 e2e pp512 A/B" "$WAIT" || exit 1
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" >/dev/null 2>&1 || true' EXIT INT TERM

cd "$ROOT"
# Order reversed across the four launches so neither arm owns the favourable slot.
i=0
for arm in off br4 br4 off; do
  out="$OUTDIR/441-e2e-$arm-$i.txt"
  echo "=== launch $i: DOTLLM_VK_FLASH_HD256=$arm -> $out ==="
  DOTLLM_VK_FLASH_HD256="$arm" \
  DOTLLM_VULKAN_HYBRID_PROFILE=1 \
  DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ=2 \
  DOTLLM_VULKAN_HYBRID_PROFILE_OUT="$out" \
    dotnet run --project "$ROOT/src/DotLLM.Cli" -c Release --no-build -- \
      bench "$MODEL" --device vulkan -p 512 -n 8 -r 2
  i=$((i + 1))
done

echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize"

echo
echo "== attn_core per launch =="
grep -H "attn_core" "$OUTDIR"/441-e2e-*.txt
