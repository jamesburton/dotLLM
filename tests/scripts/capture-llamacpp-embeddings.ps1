<#
.SYNOPSIS
    Regenerates the llama.cpp external-reference embeddings used to anchor dotLLM's
    POST /v1/embeddings implementation (issue #451).

.DESCRIPTION
    Runs llama.cpp's llama-server on the SAME GGUF the integration test loads, once per
    pooling mode (last / mean / cls), queries its OpenAI-compatible POST /v1/embeddings
    endpoint, and writes the vectors plus full provenance to

        tests/DotLLM.Tests.Integration/Fixtures/Embeddings/llamacpp-smollm2-135m-instruct-q8_0.json

    -ngl 0 pins llama.cpp to its CPU backend, matching dotLLM's CPU-only embeddings path.
    llama.cpp's --embd-normalize default is 2 (Euclidean), so the captured vectors are unit-norm.
    SmolLM2-135M-Instruct is a generative causal decoder, so --pooling must be passed explicitly
    (llama.cpp's own default for such a model is LLAMA_POOLING_TYPE_NONE).

.EXAMPLE
    ./capture-llamacpp-embeddings.ps1 `
        -LlamaServer C:\path\to\llama-server.exe `
        -ModelPath   $HOME\.dotllm\test-cache\bartowski\SmolLM2-135M-Instruct-GGUF\SmolLM2-135M-Instruct-Q8_0.gguf
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $LlamaServer,
    [Parameter(Mandatory = $true)][string] $ModelPath,
    [int] $Port = 18451,
    [string] $OutFile
)

$ErrorActionPreference = 'Stop'

if (-not $OutFile) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..' '..')
    $OutFile = Join-Path $repoRoot 'tests/DotLLM.Tests.Integration/Fixtures/Embeddings/llamacpp-smollm2-135m-instruct-q8_0.json'
}

$texts = @(
    'The quick brown fox jumps over the lazy dog.',
    'dotLLM is a native .NET inference engine.'
)

$versionLine = (& $LlamaServer --version 2>&1 | Select-Object -First 1)
$builtLine   = (& $LlamaServer --version 2>&1 | Select-Object -Skip 1 -First 1)

function Start-EmbeddingServer([string] $pooling) {
    $args = @(
        '-m', $ModelPath, '--embeddings', '--pooling', $pooling,
        '-ngl', '0', '--port', "$Port", '--host', '127.0.0.1',
        '-c', '2048', '-ub', '2048', '-b', '2048', '--no-warmup'
    )
    $proc = Start-Process -FilePath $LlamaServer -ArgumentList $args -WindowStyle Hidden -PassThru
    for ($i = 0; $i -lt 120; $i++) {
        try {
            if ((Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 2).status -eq 'ok') { return $proc }
        } catch { }
        Start-Sleep -Milliseconds 500
    }
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    throw "llama-server did not become healthy for pooling '$pooling'."
}

$pooling = [ordered]@{}
$tokenIds = $null

foreach ($mode in @('last', 'mean', 'cls')) {
    $proc = Start-EmbeddingServer $mode
    try {
        $body = @{ model = 'reference'; input = $texts } | ConvertTo-Json -Depth 5
        $r = Invoke-RestMethod "http://127.0.0.1:$Port/v1/embeddings" -Method Post -ContentType 'application/json' -Body $body
        $pooling[$mode] = [ordered]@{
            embeddings          = @($r.data | Sort-Object index | ForEach-Object { , $_.embedding })
            usage_prompt_tokens = $r.usage.prompt_tokens
        }
        if ($null -eq $tokenIds) {
            $tokenIds = @($texts | ForEach-Object {
                $tb = @{ content = $_ } | ConvertTo-Json
                , (Invoke-RestMethod "http://127.0.0.1:$Port/tokenize" -Method Post -ContentType 'application/json' -Body $tb).tokens
            })
        }
    } finally {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 800
    }
}

$doc = [ordered]@{
    _comment  = 'External reference embeddings produced by llama.cpp, used to anchor dotLLM''s POST /v1/embeddings implementation (issue #451). Regenerate with tests/scripts/capture-llamacpp-embeddings.ps1.'
    reference = [ordered]@{
        tool       = 'llama-server (llama.cpp)'
        version    = ($versionLine -replace '^version:\s*', '')
        built_with = ($builtLine -replace '^built with\s*', '')
        endpoint   = 'POST /v1/embeddings'
        flags      = '-m <gguf> --embeddings --pooling <pooling> -ngl 0 -c 2048 -ub 2048 -b 2048 --no-warmup'
        captured   = (Get-Date -Format 'yyyy-MM-dd')
        notes      = @(
            "-ngl 0 forces the llama.cpp CPU backend, matching dotLLM's CPU-only embeddings path.",
            "llama.cpp's default --embd-normalize is 2 (Euclidean/L2), so these vectors are unit-norm.",
            'The model is a generative causal decoder, so llama.cpp requires --pooling to be passed explicitly.'
        )
    }
    model = [ordered]@{
        repo        = 'bartowski/SmolLM2-135M-Instruct-GGUF'
        file        = 'SmolLM2-135M-Instruct-Q8_0.gguf'
        hidden_size = $pooling['last'].embeddings[0].Count
    }
    inputs  = @(0..($texts.Count - 1) | ForEach-Object { [ordered]@{ text = $texts[$_]; tokens = $tokenIds[$_] } })
    pooling = $pooling
}

$doc | ConvertTo-Json -Depth 8 | Set-Content -Path $OutFile -Encoding utf8
Write-Host "Wrote $OutFile"
