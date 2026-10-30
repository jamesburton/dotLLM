$s='C:\Users\james\AppData\Local\Temp\claude\C--Development-dotLLM\f5833a7b-2a87-4b52-b882-1367556b2484\scratchpad\emb'
$x='C:\Development\dotLLM\.claude\worktrees\agent-a8bb921b2a99cce5a\benchmarks\EmbeddingBench\bin\Release\net10.0\EmbeddingBench.exe'
$h="$env:USERPROFILE\.cache\huggingface\hub"
function G($repo,$pat){(Get-ChildItem "$h\models--$repo\snapshots\*\*" | ?{$_.Name -match $pat}|Select -First 1).FullName}
$m=@(
 @('minilm','second-state--All-MiniLM-L6-v2-Embedding-GGUF','f16','texts'),@('minilm-q8','second-state--All-MiniLM-L6-v2-Embedding-GGUF','Q8_0','texts'),
 @('bge-small','ChristianAzinn--bge-small-en-v1.5-gguf','f(p)?16','texts'),@('bge-small-q8','ChristianAzinn--bge-small-en-v1.5-gguf','Q8_0','texts'),
 @('bge-base','ChristianAzinn--bge-base-en-v1.5-gguf','f(p)?16','texts'),@('bge-base-q8','ChristianAzinn--bge-base-en-v1.5-gguf','Q8_0','texts'),
 @('e5-small','ChristianAzinn--e5-small-v2-gguf','f(p)?16','texts-q'),@('e5-base','ChristianAzinn--e5-base-v2-gguf','f(p)?16','texts-q'),
 @('gte-small','ChristianAzinn--gte-small-gguf','f(p)?16','texts'),@('gte-base','ChristianAzinn--gte-base-gguf','f(p)?16','texts'),
 @('mxbai','ChristianAzinn--mxbai-embed-large-v1-gguf','f(p)?16','texts'),@('mxbai-q8','ChristianAzinn--mxbai-embed-large-v1-gguf','Q8_0','texts'),
 @('nomic','nomic-ai--nomic-embed-text-v1.5-GGUF','f16','texts-nd'),@('nomic-q8','nomic-ai--nomic-embed-text-v1.5-GGUF','Q8_0','texts-nd'),@('nomic-q4km','nomic-ai--nomic-embed-text-v1.5-GGUF','Q4_K_M','texts-nd'),
 @('qwen3','Qwen--Qwen3-Embedding-0.6B-GGUF','f16','texts'))
foreach($e in $m){ $p=G $e[1] $e[2]; & $x dump $p "$s\$($e[3]).json" "$s\dl-$($e[0]).json" --label $e[0] --threads 8 2>&1 | Select -Last 1 | Out-Null; "$($e[0]) done" }
