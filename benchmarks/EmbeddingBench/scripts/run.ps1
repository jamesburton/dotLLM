param([string]$Mode = 'throughput', [string]$Extra = '')
$x = 'C:\Development\dotLLM\.claude\worktrees\agent-a8bb921b2a99cce5a\benchmarks\EmbeddingBench\bin\Release\net10.0\EmbeddingBench.exe'
$h = "$env:USERPROFILE\.cache\huggingface\hub"
function G($repo, $pat) { (Get-ChildItem "$h\models--$repo\snapshots\*\*" | ? { $_.Name -match $pat } | Select -First 1).FullName }
$models = [ordered]@{
  'minilm-l6-f16'  = G 'second-state--All-MiniLM-L6-v2-Embedding-GGUF' 'f(p)?16'
  'bge-small-f16'  = G 'ChristianAzinn--bge-small-en-v1.5-gguf' 'f(p)?16'
  'e5-small-f16'   = G 'ChristianAzinn--e5-small-v2-gguf' 'f(p)?16'
  'gte-small-f16'  = G 'ChristianAzinn--gte-small-gguf' 'f(p)?16'
  'bge-base-f16'   = G 'ChristianAzinn--bge-base-en-v1.5-gguf' 'f(p)?16'
  'e5-base-f16'    = G 'ChristianAzinn--e5-base-v2-gguf' 'f(p)?16'
  'gte-base-f16'   = G 'ChristianAzinn--gte-base-gguf' 'f(p)?16'
  'nomic-v1.5-f16' = G 'nomic-ai--nomic-embed-text-v1.5-GGUF' 'f(p)?16'
  'mxbai-large-f16'= G 'ChristianAzinn--mxbai-embed-large-v1-gguf' 'f(p)?16'
  'qwen3-emb-0.6b-f16' = G 'Qwen--Qwen3-Embedding-0.6B-GGUF' 'f(p)?16'
  'minilm-l6-q8'   = G 'second-state--All-MiniLM-L6-v2-Embedding-GGUF' 'Q8_0'
  'bge-base-q8'    = G 'ChristianAzinn--bge-base-en-v1.5-gguf' 'Q8_0'
  'nomic-v1.5-q8'  = G 'nomic-ai--nomic-embed-text-v1.5-GGUF' 'Q8_0'
  'nomic-v1.5-q4km'= G 'nomic-ai--nomic-embed-text-v1.5-GGUF' 'Q4_K_M'
  'mxbai-large-q8' = G 'ChristianAzinn--mxbai-embed-large-v1-gguf' 'Q8_0'
}
$only = $env:ONLY
$out = "C:\Users\james\AppData\Local\Temp\claude\C--Development-dotLLM\f5833a7b-2a87-4b52-b882-1367556b2484\scratchpad\emb\results"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($k in $models.Keys) {
  if ($only -and ($k -notmatch $only)) { continue }
  $p = $models[$k]
  if (-not $p) { "MISSING $k" ; continue }
  "== $k $Mode"
  $args2 = @($Mode, $p, '--label', $k) + ($Extra -split ' ' | ? { $_ })
  & $x @args2 2>&1 | Tee-Object -FilePath "$out\$Mode-$k.jsonl"
}

