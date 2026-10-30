param([string]$Threads = '32')
$s='C:\Users\james\AppData\Local\Temp\claude\C--Development-dotLLM\f5833a7b-2a87-4b52-b882-1367556b2484\scratchpad\emb'
$h="$env:USERPROFILE\.cache\huggingface\hub"
function G($repo,$pat){(Get-ChildItem "$h\models--$repo\snapshots\*\*" | ?{$_.Name -match $pat}|Select -First 1).FullName}
$m=@(
 @('minilm-f16','second-state--All-MiniLM-L6-v2-Embedding-GGUF','f16','texts'),@('minilm-q8','second-state--All-MiniLM-L6-v2-Embedding-GGUF','Q8_0','texts'),
 @('bge-small-f16','ChristianAzinn--bge-small-en-v1.5-gguf','f(p)?16','texts'),@('bge-small-q8','ChristianAzinn--bge-small-en-v1.5-gguf','Q8_0','texts'),
 @('e5-small-f16','ChristianAzinn--e5-small-v2-gguf','f(p)?16','texts-q'),@('gte-small-f16','ChristianAzinn--gte-small-gguf','f(p)?16','texts'),
 @('bge-base-f16','ChristianAzinn--bge-base-en-v1.5-gguf','f(p)?16','texts'),@('bge-base-q8','ChristianAzinn--bge-base-en-v1.5-gguf','Q8_0','texts'),
 @('e5-base-f16','ChristianAzinn--e5-base-v2-gguf','f(p)?16','texts-q'),@('gte-base-f16','ChristianAzinn--gte-base-gguf','f(p)?16','texts'),
 @('nomic-f16','nomic-ai--nomic-embed-text-v1.5-GGUF','f16','texts-nd'),@('nomic-q8','nomic-ai--nomic-embed-text-v1.5-GGUF','Q8_0','texts-nd'),@('nomic-q4km','nomic-ai--nomic-embed-text-v1.5-GGUF','Q4_K_M','texts-nd'),
 @('mxbai-f16','ChristianAzinn--mxbai-embed-large-v1-gguf','f(p)?16','texts'),@('mxbai-q8','ChristianAzinn--mxbai-embed-large-v1-gguf','Q8_0','texts'),
 @('qwen3-f16','Qwen--Qwen3-Embedding-0.6B-GGUF','f16','texts'))
foreach($e in $m){
  if ($env:ONLY -and $e[0] -notmatch $env:ONLY) { continue }
  $g = G $e[1] $e[2]
  $p = Start-Process -FilePath C:\Development\llama.cpp-b9016\llama-server.exe -ArgumentList '-m',$g,'--embeddings','-dev','none','-c','2048','-b','2048','-ub','2048','-np','1','--port','18081','-t',$Threads -PassThru -WindowStyle Hidden -RedirectStandardError "$s\llama-$($e[0]).err"
  $ok=$false; for($i=0;$i -lt 60;$i++){ Start-Sleep 1; try { if ((Invoke-WebRequest http://127.0.0.1:18081/health -UseBasicParsing).StatusCode -eq 200) { $ok=$true; break } } catch {} }
  try {
    $r = python -I "$s\llamavec.py" http://127.0.0.1:18081 "$s\$($e[3]).json" "$s\ll-$($e[0]).json"
    $o = "{`"model`":`"$($e[0])`",`"threads`":$Threads,`"r`":$r}"
    $o; Add-Content "$s\results\llama-e2e.jsonl" $o
  } finally { Stop-Process -Id $p.Id -Force -ea 0 }
}
