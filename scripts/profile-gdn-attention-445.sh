#!/usr/bin/env bash
# #445 — sub-bucket attribution for gdn_scan and attention at prefill.
#
# Same idea as scripts/profile-hybrid-dense-prefill.sh (#434) but:
#   - takes the lock with a SHORT wait timeout, in the foreground, so a blocked acquire
#     fails fast instead of silently taking the GPU much later (see the campaign's
#     "backgrounded gpu-lock acquire resurfaces" note);
#   - releases from a trap on every exit path;
#   - names the sub-buckets added in #445 (gdn_scan_core / gdn_postgate /
#     attn_rope / attn_kvupdate / attn_core / attn_gate) and prints the #434 roll-up too.
#
# Usage:
#   scripts/profile-gdn-attention-445.sh <model.gguf> [prompt-tokens] [reps] [out-file]

set -euo pipefail

MODEL="${1:?usage: profile-gdn-attention-445.sh <model.gguf> [prompt-tokens] [reps] [out]}"
PROMPT="${2:-512}"
REPS="${3:-2}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${4:-$ROOT/.docs/445-prefill-profile.txt}"
AGENT="${GPU_LOCK_AGENT:-a445}"
# Short wait: if #443/#446 hold the GPU we want to know now, not in an hour.
WAIT="${GPU_LOCK_WAIT:-240}"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

echo "== GPU consumers BEFORE =="
powershell.exe -NoProfile -Command \
  "Get-Process | Sort-Object WS -Descending | Select-Object -First 8 Name,@{n='WS_MB';e={[int](\$_.WS/1MB)}} | Format-Table -AutoSize" \
  || echo "(could not enumerate processes)"

bash "$ROOT/scripts/gpu-lock.sh" acquire "$AGENT" "issue 445 gdn_scan/attention sub-bucket attribution" "$WAIT"
trap 'bash "$ROOT/scripts/gpu-lock.sh" release "$AGENT" || true' EXIT

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
