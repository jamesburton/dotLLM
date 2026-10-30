param([string[]]$Names, [string]$Device = "cpu", [string]$Model = "trunk52")
$s = Split-Path -Parent $MyInvocation.MyCommand.Path
$wt = "C:\Development\dotLLM\.claude\worktrees\agent-a08e22aa7248079ed"
$d = "$HOME\.dotllm\test-cache\nvidia\nemotron-3.5-lightning"
$T = if ($Model -eq "trunk52") { "$d\nemotron-3.5-lightning-Q4_K_M-trunk52.gguf" } else { "$d\nemotron-3.5-lightning-Q4_K_M.gguf" }
$exe = "$wt\benchmarks\NemotronProbe\bin\Release\net10.0\NemotronProbe.exe"
foreach ($n in $Names) {
  "== $n ($Device, $Model)"
  python -I "$wt\benchmarks\NemotronProbe\nemprobe_kl.py" ids "$s\corp\$n.kld.bin" "$s\corp\$n.ids.txt"
  & $exe logits $T "$s\corp\$n.ids.txt" "$s\corp\$n.$Device.$Model" 512 2 $Device
  python -I "$wt\benchmarks\NemotronProbe\nemprobe_kl.py" cmp "$s\corp\$n.kld.bin" "$s\corp\$n.$Device.$Model"
}
