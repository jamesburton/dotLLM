#!/bin/bash
# usage: run_train.sh OUTDIR LOGFILE python-args...   (lock held until watcher.sh sees python exit)
OUT=$1; LOGF=$2; ARGF=$3
L=/e/Development/dotLLM/scripts/gpu-lock.sh
export HF_HOME=C:/hf-home HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p /c/littlebit/pids
rm -f /c/littlebit/pids/winpid
bash $L acquire littlebit-qat "littlebit QAT run $OUT (watcher releases)" 7200 5400 || { echo "lock acquire failed" >> $LOGF; exit 1; }
cd /c/littlebit
/c/Python311/python.exe train.py --out $OUT --resume $(cat $ARGF) >> $LOGF 2>&1 &
PID=$!
cat /proc/$PID/winpid > /c/littlebit/pids/winpid
echo "started winpid $(cat /c/littlebit/pids/winpid) at $(date)" >> $LOGF
wait $PID
echo "python exited code $? at $(date)" >> $LOGF
