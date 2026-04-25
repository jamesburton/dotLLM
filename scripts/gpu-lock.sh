#!/usr/bin/env bash
# GPU access mutex — single-holder file-based lock.
#
# Why: concurrent CUDA contexts on the same device trip CUDA_ERROR_ILLEGAL_ADDRESS
# (error 700) in cuCtxCreate_v2. dotLLM's GPU tests + benchmarks each create their
# own context. Only one such process at a time is safe. Non-GPU work (code editing,
# doc writes, design) is fully parallelisable and should NOT take this lock.
#
# Lock primitive: a directory at $LOCK_DIR. mkdir is atomic on every filesystem
# we care about (NTFS via MSYS2, ext4, APFS). Holder metadata lives in
# $LOCK_DIR/holder as a single line: PID|NAME|EPOCH|REASON.
#
# Staleness: timestamp-only. If the holder file's recorded epoch is older than
# stale-sec (default 1800 = 30 min), the lock is considered abandoned and any
# acquire will force-release it. Agents must call `release` promptly when their
# GPU operation finishes; PIDs are not used for liveness because a bash one-shot
# script's $$ doesn't outlive the operation it protects.
#
# Usage:
#   gpu-lock.sh acquire <name> <reason> [timeout-sec=900] [stale-sec=1800]
#   gpu-lock.sh refresh <name>           # bump timestamp during long operations
#   gpu-lock.sh release <name>           # idempotent; only releases if you own it
#   gpu-lock.sh status                   # prints holder or "FREE"
#   gpu-lock.sh force-clear              # admin override (overseer only)
#
# Exit codes:
#   0  success (acquire took the lock; release / status / force-clear ran cleanly)
#   1  acquire timed out, OR release was a no-op because we don't own the lock
#   2  bad arguments

set -u

LOCK_DIR="${DOTLLM_GPU_LOCK_DIR:-/e/Development/dotLLM/.gpu-lock}"
HOLDER_FILE="$LOCK_DIR/holder"

cmd="${1:-}"

case "$cmd" in
  acquire)
    name="${2:-}"
    reason="${3:-}"
    timeout_sec="${4:-900}"
    stale_sec="${5:-1800}"
    if [ -z "$name" ] || [ -z "$reason" ]; then
      echo "[gpu-lock] acquire requires <name> and <reason>" >&2
      exit 2
    fi

    start=$(date +%s)
    while true; do
      # Try to take the lock atomically.
      if mkdir "$LOCK_DIR" 2>/dev/null; then
        printf '%s|%s|%s|%s\n' "$$" "$name" "$(date +%s)" "$reason" > "$HOLDER_FILE"
        echo "[gpu-lock] acquired by '$name' (pid $$): $reason"
        exit 0
      fi

      # Lock held — inspect holder to see if it's stale by timestamp.
      if [ -f "$HOLDER_FILE" ]; then
        IFS='|' read -r holder_pid holder_name holder_epoch holder_reason < "$HOLDER_FILE" 2>/dev/null
        now=$(date +%s)
        age=$((now - ${holder_epoch:-0}))
        if [ "$age" -gt "$stale_sec" ]; then
          echo "[gpu-lock] forcing release of stale lock ('$holder_name', age ${age}s > ${stale_sec}s)" >&2
          rm -rf "$LOCK_DIR"
          continue
        fi
      else
        # Directory exists but no holder file — racy state. Wait briefly.
        :
      fi

      now=$(date +%s)
      elapsed=$((now - start))
      if [ "$elapsed" -ge "$timeout_sec" ]; then
        echo "[gpu-lock] timed out after ${elapsed}s waiting for lock (held by '${holder_name:-?}', $((now - ${holder_epoch:-now}))s old: ${holder_reason:-?})" >&2
        exit 1
      fi
      sleep 3
    done
    ;;

  release)
    name="${2:-}"
    if [ -z "$name" ]; then
      echo "[gpu-lock] release requires <name>" >&2
      exit 2
    fi
    if [ ! -d "$LOCK_DIR" ]; then
      echo "[gpu-lock] release: no lock held (no-op)"
      exit 0
    fi
    if [ -f "$HOLDER_FILE" ]; then
      IFS='|' read -r holder_pid holder_name holder_epoch holder_reason < "$HOLDER_FILE" 2>/dev/null
      if [ "$holder_name" != "$name" ]; then
        echo "[gpu-lock] release refused: lock held by '$holder_name' (pid $holder_pid), not '$name'" >&2
        exit 1
      fi
    fi
    rm -rf "$LOCK_DIR"
    echo "[gpu-lock] released by '$name'"
    exit 0
    ;;

  refresh)
    name="${2:-}"
    if [ -z "$name" ]; then
      echo "[gpu-lock] refresh requires <name>" >&2
      exit 2
    fi
    if [ ! -f "$HOLDER_FILE" ]; then
      echo "[gpu-lock] refresh: no lock currently held" >&2
      exit 1
    fi
    IFS='|' read -r holder_pid holder_name holder_epoch holder_reason < "$HOLDER_FILE" 2>/dev/null
    if [ "$holder_name" != "$name" ]; then
      echo "[gpu-lock] refresh refused: lock held by '$holder_name', not '$name'" >&2
      exit 1
    fi
    printf '%s|%s|%s|%s\n' "$holder_pid" "$name" "$(date +%s)" "$holder_reason" > "$HOLDER_FILE"
    echo "[gpu-lock] refreshed by '$name'"
    exit 0
    ;;

  status)
    if [ ! -d "$LOCK_DIR" ]; then
      echo "FREE"
      exit 0
    fi
    if [ -f "$HOLDER_FILE" ]; then
      IFS='|' read -r holder_pid holder_name holder_epoch holder_reason < "$HOLDER_FILE" 2>/dev/null
      now=$(date +%s)
      age=$((now - ${holder_epoch:-0}))
      echo "HELD name='$holder_name' pid=$holder_pid age=${age}s reason='$holder_reason'"
      exit 0
    fi
    echo "HELD (no metadata)"
    exit 0
    ;;

  force-clear)
    rm -rf "$LOCK_DIR"
    echo "[gpu-lock] force-cleared"
    exit 0
    ;;

  *)
    echo "Usage: $0 {acquire <name> <reason> [timeout-sec] [stale-sec] | refresh <name> | release <name> | status | force-clear}" >&2
    exit 2
    ;;
esac
