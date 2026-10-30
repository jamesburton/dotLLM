#!/usr/bin/env bash
# Per-op prefill attribution for the Vulkan hybrid-dense model (issue #434).
#
# Runs one short `bench` against a qwen35 GGUF with the profiler armed and prints the
# per-category breakdown of the pp512 prefill pass. The point is the SHARE each bucket holds,
# not the absolute milliseconds: absolute throughput on a UMA box swings ~40% with CPU
# memory-bandwidth contention and decayed 36% across five consecutive launches in the
# 2026-09-19 A/B session, whereas the shares are robust to that drift.
#
# Usage:
#   scripts/profile-hybrid-dense-prefill.sh <model.gguf> [prompt-tokens] [reps]
#
# Holds scripts/gpu-lock.sh for the duration and releases it on any exit path.

set -euo pipefail

MODEL="${1:?usage: profile-hybrid-dense-prefill.sh <model.gguf> [prompt-tokens] [reps]}"
PROMPT="${2:-512}"
REPS="${3:-2}"
AGENT="${GPU_LOCK_AGENT:-profile434}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${DOTLLM_PROFILE_OUT:-$ROOT/.docs/hybrid-prefill-profile.txt}"
mkdir -p "$(dirname "$OUT")"
: > "$OUT"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize" \
  || echo "(could not enumerate processes)"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 434 hybrid-dense prefill attribution" 3600
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" || true' EXIT

# MINSEQ=2 keeps decode steps out of the report; the prefill Forward is the only one profiled.
# bench discards its first repetition as a warm-up, so the first block below is the warm-up and
# is reported separately rather than averaged in.
DOTLLM_VULKAN_HYBRID_PROFILE=1 \
DOTLLM_VULKAN_HYBRID_PROFILE_MINSEQ=2 \
DOTLLM_VULKAN_HYBRID_PROFILE_OUT="$OUT" \
  dotnet run --project "$ROOT/src/DotLLM.Cli" -c Release --no-build -- \
    bench "$MODEL" --device vulkan -p "$PROMPT" -n 8 -r "$REPS"

echo
echo "== GPU consumers AFTER =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize" \
  || echo "(could not enumerate processes)"

echo
echo "== per-forward attribution ($OUT) =="
cat "$OUT"
