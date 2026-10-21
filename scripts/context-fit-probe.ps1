# Starts `dotllm serve` once per KV configuration, captures the fitted context-window line, then stops only that process.
# Example: ./scripts/context-fit-probe.ps1 -Cli .\src\DotLLM.Cli\bin\Release\net10.0\DotLLM.Cli.exe -Model C:\path\model.gguf
param(
    [Parameter(Mandatory)][string]$Cli,
    [Parameter(Mandatory)][string]$Model,
    [string]$Tag = "fit",
    [int]$Port = 18080,
    [string]$LogDir = (Join-Path $env:TEMP "dotllm-fit-probe"),
    # Name -> extra `serve` arguments; the default compares KV quantization with and without MTP.
    [System.Collections.IDictionary]$Configs = ([ordered]@{
        'q8_mtp'   = @('--kv-quant-k', 'q8_0', '--kv-quant-v', 'q8_0')
        'q8_nomtp' = @('--kv-quant-k', 'q8_0', '--kv-quant-v', 'q8_0', '--no-mtp')
        'q4_mtp'   = @('--kv-quant-k', 'q4_0', '--kv-quant-v', 'q4_0')
        'q4_nomtp' = @('--kv-quant-k', 'q4_0', '--kv-quant-v', 'q4_0', '--no-mtp')
    })
)
New-Item -ItemType Directory -Force $LogDir | Out-Null
foreach ($name in $Configs.Keys) {
    # Wait for the previous server's VRAM to drain.
    for ($i = 0; $i -lt 60; $i++) { if ([int](@(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)[0]) -lt 300) { break }; Start-Sleep 2 }
    $out = "$LogDir\${Tag}_$name.out"; $err = "$LogDir\${Tag}_$name.err"
    $a = @('serve', $Model, '-d', 'cuda', '-p', $Port) + $Configs[$name]
    $p = Start-Process -FilePath $Cli -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
    $line = $null; $t0 = Get-Date
    while (-not $p.HasExited -and ((Get-Date) - $t0).TotalSeconds -lt 300) {
        $line = Select-String -Path $out, $err -Pattern 'Context window' -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($line) { break }
        Start-Sleep 2
    }
    $vram = @(nvidia-smi --query-gpu=memory.used --format=csv,noheader)[0]
    if (-not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    "[$name] vram=$vram load_s=$([int]((Get-Date) - $t0).TotalSeconds) $($line.Line)"
}
