# Runs the standalone warm-start on Strix's ROCm iGPU, detached, with flushed logging.
$ErrorActionPreference = 'Continue'
$env:TORCHDYNAMO_DISABLE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$PY  = 'C:\Development\strix-pytorch\.venv-win-rocm721\Scripts\python.exe'
$WD  = 'C:\Development\dotLLM-warmstart\scripts\lora'
$OUT = 'C:\Development\dotLLM-warmstart\_ws'
New-Item -ItemType Directory -Force -Path $OUT | Out-Null
"### STRIX WARM-START launch $(Get-Date -Format s) (all-GPU / iGPU, no hessian-cpu) ###" |
  Out-File "$OUT\ws.log" -Encoding utf8
$p = Start-Process -FilePath $PY `
  -ArgumentList 'make_warmstart.py','--base','Qwen/Qwen3-1.7B','--device','cuda','--out',$OUT,'--calib-batches','16','--calib-batch-size','2' `
  -WorkingDirectory $WD `
  -RedirectStandardOutput "$OUT\ws.out" -RedirectStandardError "$OUT\ws.err" `
  -WindowStyle Hidden -PassThru
$p.Id | Out-File "$OUT\ws.pid" -Encoding ascii
"launched pid $($p.Id) at $(Get-Date -Format HH:mm:ss)"
