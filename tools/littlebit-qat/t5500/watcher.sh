#!/bin/bash
# keeps the gpu-lock fresh while the trainer's windows pid lives; releases when it is gone (finished/crashed/killed)
L=/e/Development/dotLLM/scripts/gpu-lock.sh
for i in $(seq 1 300); do [ -s /c/littlebit/pids/winpid ] && break; sleep 5; done
W=$(cat /c/littlebit/pids/winpid 2>/dev/null)
while [ -n "$W" ] && tasklist //FI "PID eq $W" 2>/dev/null | grep -q " $W "; do
  bash $L refresh littlebit-qat >/dev/null 2>&1; sleep 240
done
bash $L release littlebit-qat
echo "watcher released lock at $(date)" >> /c/littlebit/watcher.log
