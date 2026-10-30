param([string[]]$Names)
$s = Split-Path -Parent $MyInvocation.MyCommand.Path
$T = "$HOME\.dotllm\test-cache\nvidia\nemotron-3.5-lightning\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf"
foreach ($n in $Names) {
  "== $n"
  & C:\Development\llama.cpp\llama-perplexity.exe -m $T -dev none -ngl 0 -f "$s\corp\$n.txt" -c 512 --chunks 2 --save-all-logits "$s\corp\$n.kld.bin" 2>&1 |
     Select-String -Pattern "tokens|Final|PPL|chunk|error|you need" | Select-Object -Last 8
}
