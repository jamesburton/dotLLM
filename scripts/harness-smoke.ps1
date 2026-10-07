<#
.SYNOPSIS
  Serve a model ref (pulls on a miss) with default flags, and smoke-test the API surface an external
  client / eval harness uses (non-stream chat, streamed chat, tool call, structured output),
  recording decode and prefill tok/s.

.DESCRIPTION
  One model ref in, a PASS/FAIL/WARN table + JSON result out. The server is ALWAYS stopped
  in a finally block (graceful /v1/admin/shutdown, then process-tree kill).

  This script does NOT take scripts/gpu-lock.sh. On a shared GPU box the caller must:
      bash scripts/gpu-lock.sh acquire <name> "<reason>" && pwsh scripts/harness-smoke.ps1 ... ; \
      bash scripts/gpu-lock.sh release <name>
  (and check Get-Process for human GPU processes - the lock only sees agents).

.PARAMETER Model
  Model ref: a GGUF path, or owner/repo[:quant] (ModelResolver refs, e.g.
  unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M).

.PARAMETER Cli
  How to invoke dotllm. Default: `dnx dotllm --prerelease --yes --` (the user-facing path;
  falls back to `dotnet dnx ...` when the `dnx` shim is not on PATH). NOTE: a bare `dotllm` on PATH may be
  the frozen upstream `dotllm.cli` 0.1.0-preview (CPU default, no auto-pull) - do not rely on it. Pass a path to a
  DotLLM.Cli.exe (or "dotllm") to test a local build. Tokens are split on spaces.

.PARAMETER Version
  Pin the dnx package version (e.g. 0.3.0-dev.2470) -> `dnx dotllm@<Version> --yes --`. Always pin for
  reproducible runs: `--version` prints 0.1.0 for every build, so the result JSON records the DLL
  ProductVersion (which embeds the git SHA) of the process that actually served.

.PARAMETER PlainThinking
  `off` (default): the plain-chat checks (chat, stream, tool call, throughput) send `enable_thinking=false`
  so a thinking model answers in `content` within a small token budget (since #767/#778 a thinking model
  otherwise spends the budget on `reasoning_content` and returns an EMPTY `content`). Templates with no
  such variable ignore it. `default`: do not send it (the model's own default; budgets fall back to
  -ThinkingMaxTokens when the model reasons). The dedicated `reasoning` / `reasoning-stream` checks always
  turn thinking ON (`enable_thinking=true`) and assert `reasoning_content` is populated, `content` holds the
  answer, and no `<think>` text leaks into `content`. A non-reasoning model reports SKIP there.

.PARAMETER ThinkingMaxTokens
  max_tokens for the thinking-on checks (default 2048).

.PARAMETER AllowPull
  By default the script REFUSES to start `serve` when the model ref is not already in the HF hub cache
  (the CLI would otherwise silently download several GB; `owner/repo` defaults to Q4_K_M and an
  Unsloth `UD-Q4_K_M` file may not match). Pass -AllowPull to permit the download.

.PARAMETER Device
  Passed to `serve --device`. Default: do not pass it (the server default, `auto`). An explicit cuda/gpu on a host with no
  NVIDIA driver reports SKIP ("not measured here") and exits 0.

.PARAMETER ServeArgs
  Extra serve args, e.g. -ServeArgs '--mtp'. Default: none (the point is that defaults work).

.EXAMPLE
  pwsh scripts/harness-smoke.ps1 -Model bartowski/Llama-3.2-1B-Instruct-GGUF:Q8_0
.EXAMPLE
  pwsh scripts/harness-smoke.ps1 -Model C:\models\x.gguf -Cli C:\build\cli\DotLLM.Cli.exe -Device vulkan -Runs 5
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)][string]$Model,
    [string]$Cli = '',
    [string]$Version = '',
    [string]$Device = '',
    [string[]]$ServeArgs = @(),
    [int]$Port = 18181,
    [int]$Runs = 3,
    [int]$MaxTokens = 128,
    [int]$PrefillWords = 700,
    [int]$ReadyTimeoutSec = 1800,
    [int]$ThinkingMaxTokens = 2048,
    [ValidateSet('off', 'default')][string]$PlainThinking = 'off',
    [switch]$AllowPull,
    [string]$OutDir = ''
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------- CLI resolution
if (-not $Cli) {
    $pkg = if ($Version) { "dotllm@$Version" } else { 'dotllm --prerelease' }
    $Cli = if (Get-Command dnx -ErrorAction SilentlyContinue) { "dnx $pkg --yes --" } else { "dotnet dnx $pkg --yes --" }
}
$cliTokens = @($Cli -split '\s+' | Where-Object { $_ })
$cliExe = $cliTokens[0]
$cliPre = if ($cliTokens.Count -gt 1) { $cliTokens[1..($cliTokens.Count - 1)] } else { @() }

if (-not $OutDir) {
    $root = Split-Path -Parent $PSScriptRoot
    $OutDir = Join-Path $root '.docs/harness-smoke'
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$slug = ($Model -replace '[^A-Za-z0-9._-]', '_')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$serverOut = Join-Path $OutDir "$slug-$stamp.server.out.log"
$serverErr = Join-Path $OutDir "$slug-$stamp.server.err.log"
$resultPath = Join-Path $OutDir "$slug-$stamp.json"
$base = "http://127.0.0.1:$Port"

$results = [System.Collections.Generic.List[object]]::new()
function Add-Result([string]$Name, [string]$Status, [string]$Detail, $Data = $null) {
    $results.Add([pscustomobject]@{ check = $Name; status = $Status; detail = $Detail; data = $Data })
    $color = switch ($Status) { 'PASS' { 'Green' } 'WARN' { 'Yellow' } 'GAP' { 'Magenta' } 'SKIP' { 'DarkGray' } default { 'Red' } }
    Write-Host ('{0,-5} {1,-22} {2}' -f $Status, $Name, $Detail) -ForegroundColor $color
}

# ---------------------------------------------------------------- HTTP helpers
$http = [System.Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromMinutes(20)

function Invoke-Json([string]$Method, [string]$Path, $Body = $null) {
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::new($Method), "$base$Path")
    if ($null -ne $Body) {
        $req.Content = [System.Net.Http.StringContent]::new(($Body | ConvertTo-Json -Depth 20 -Compress), [Text.Encoding]::UTF8, 'application/json')
    }
    $resp = $http.SendAsync($req).GetAwaiter().GetResult()
    $text = $resp.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    [pscustomobject]@{ Code = [int]$resp.StatusCode; Text = $text; Json = $(try { $text | ConvertFrom-Json -Depth 20 } catch { $null }) }
}

# Streams /v1/chat/completions; returns content, tool calls, timings, TTFT, wall-clock decode rate.
function Invoke-Stream($Body) {
    $Body.stream = $true
    $Body.stream_options = @{ include_usage = $true }
    $req = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$base/v1/chat/completions")
    $req.Content = [System.Net.Http.StringContent]::new(($Body | ConvertTo-Json -Depth 20 -Compress), [Text.Encoding]::UTF8, 'application/json')
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $resp = $http.SendAsync($req, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
    if (-not $resp.IsSuccessStatusCode) {
        return [pscustomobject]@{ Ok = $false; Error = "HTTP $([int]$resp.StatusCode): $($resp.Content.ReadAsStringAsync().GetAwaiter().GetResult())" }
    }
    $reader = [IO.StreamReader]::new($resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
    $content = [Text.StringBuilder]::new(); $reasoning = [Text.StringBuilder]::new()
    $chunks = 0; $ttft = $null; $tLast = $null; $finish = $null; $usage = $null; $timings = $null; $doneSeen = $false
    $tools = @{}
    while ($null -ne ($line = $reader.ReadLine())) {
        if (-not $line.StartsWith('data:')) { continue }
        $payload = $line.Substring(5).Trim()
        if ($payload -eq '[DONE]') { $doneSeen = $true; break }
        $o = $payload | ConvertFrom-Json -Depth 20
        if ($o.PSObject.Properties['usage'] -and $o.usage) { $usage = $o.usage }
        if ($o.PSObject.Properties['timings'] -and $o.timings) { $timings = $o.timings }
        foreach ($c in @($o.choices)) {
            if ($c.finish_reason) { $finish = $c.finish_reason }
            $d = $c.delta
            if ($null -eq $d) { continue }
            $piece = $false
            if ($d.PSObject.Properties['content'] -and $d.content) { [void]$content.Append($d.content); $piece = $true }
            if ($d.PSObject.Properties['reasoning_content'] -and $d.reasoning_content) { [void]$reasoning.Append($d.reasoning_content); $piece = $true }
            if ($d.PSObject.Properties['tool_calls'] -and $d.tool_calls) {
                $piece = $true
                foreach ($tc in $d.tool_calls) {
                    $i = if ($tc.PSObject.Properties['index']) { [int]$tc.index } else { 0 }
                    if (-not $tools.ContainsKey($i)) { $tools[$i] = @{ name = ''; args = [Text.StringBuilder]::new() } }
                    if ($tc.PSObject.Properties['function'] -and $tc.function) {
                        if ($tc.function.PSObject.Properties['name'] -and $tc.function.name) { $tools[$i].name = $tc.function.name }
                        if ($tc.function.PSObject.Properties['arguments'] -and $tc.function.arguments) { [void]$tools[$i].args.Append($tc.function.arguments) }
                    }
                }
            }
            if ($piece) { $chunks++; if ($null -eq $ttft) { $ttft = $sw.Elapsed.TotalMilliseconds }; $tLast = $sw.Elapsed.TotalMilliseconds }
        }
    }
    $reader.Dispose(); $resp.Dispose()
    $genTok = if ($usage) { [int]$usage.completion_tokens } else { $chunks }
    $wallTps = if ($ttft -ne $null -and $tLast -gt $ttft -and $genTok -gt 1) { ($genTok - 1) / (($tLast - $ttft) / 1000.0) } else { $null }
    [pscustomobject]@{
        Ok = $true; Content = $content.ToString(); Reasoning = $reasoning.ToString(); Chunks = $chunks
        TtftMs = $ttft; Finish = $finish; Usage = $usage; Timings = $timings; DoneSeen = $doneSeen
        GenTokens = $genTok; WallDecodeTps = $wallTps
        ToolCalls = @($tools.Keys | Sort-Object | ForEach-Object { [pscustomobject]@{ name = $tools[$_].name; arguments = $tools[$_].args.ToString() } })
    }
}

# Plain-chat arm: thinking OFF so a thinking model answers in `content` inside a small budget (#767/#778).
function Plain($body) {
    if ($PlainThinking -eq 'off') { $body.enable_thinking = $false }
    $body
}

function Prop($o, [string]$n) { if ($null -ne $o -and $o.PSObject.Properties[$n]) { $o.$n } else { $null } }

function Get-Median([double[]]$xs) {
    if (-not $xs -or $xs.Count -eq 0) { return $null }
    $s = $xs | Sort-Object; $n = $s.Count
    if ($n % 2) { $s[[int](($n - 1) / 2)] } else { ($s[$n / 2 - 1] + $s[$n / 2]) / 2 }
}

function Stop-ServerTree($proc) {
    if ($null -eq $proc) { return }
    try { Invoke-RestMethod -Method Post -Uri "$base/v1/admin/shutdown" -TimeoutSec 5 | Out-Null } catch { }
    if (-not $proc.WaitForExit(15000)) {
        & taskkill /T /F /PID $proc.Id 2>&1 | Out-Null
    }
    # dnx launches a child; make sure nothing is left listening on our port.
    Start-Sleep -Milliseconds 500
    Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { & taskkill /T /F /PID $_.OwningProcess 2>&1 | Out-Null }
}

# ---------------------------------------------------------------- helpers (model-aware)
function Cut([string]$t, [int]$n) { if ($null -eq $t) { return '' }; $t = $t -replace '\s+', ' '; $t.Substring(0, [math]::Min($n, $t.Length)) }

# Reasoning models put <think>...</think> inline in `content` (the server has no reasoning_content
# and no think-off switch). Return what a harness would treat as the visible answer.
function Split-Think([string]$text) {
    if ($null -eq $text) { $text = '' }
    $hasThink = ($text -match '<think>') -or ($text -match '</think>')
    $visible = $text; $open = $false
    if ($text -match '</think>') { $visible = $text.Substring($text.LastIndexOf('</think>') + 8) }
    elseif ($text -match '<think>') { $visible = ''; $open = $true }
    [pscustomobject]@{ HasThink = $hasThink; Visible = $visible.Trim(); Open = $open }
}

# Raw tool-call markup in content means the MODEL produced a call that our PARSER did not lift
# into `tool_calls` (a parser gap), as opposed to a model that never tried.
function Find-ToolMarkup([string]$text) {
    foreach ($m in '<tool_call>', '<function=', '[TOOL_CALLS]', '<|python_tag|>', '<|channel|>', '<|tool_call', '<start_function_call>', '"tool_calls"') {
        if ($text -and $text.Contains($m)) { return $m }
    }
    return $null
}

function Test-ModelCached([string]$ref) {
    if (Test-Path -LiteralPath $ref) { return @{ Ok = $true; Detail = 'local path' } }
    $r = $ref -replace '^(hf\.co/|hf://)', ''
    if ($r -notmatch '^([^/:]+)/([^/:]+)(?::(.+))?$') { return @{ Ok = $true; Detail = 'not an owner/repo ref (guard skipped)' } }
    $owner = $Matches[1]; $repo = $Matches[2]; $quant = if ($Matches[3]) { $Matches[3] } else { 'Q4_K_M' }
    $hub = if ($env:HF_HUB_CACHE) { $env:HF_HUB_CACHE } elseif ($env:HF_HOME) { Join-Path $env:HF_HOME 'hub' } else { Join-Path $HOME '.cache/huggingface/hub' }
    $files = @()
    $d = Join-Path $hub "models--$owner--$repo"
    if (Test-Path $d) { $files += Get-ChildItem $d -Recurse -Filter *.gguf -ErrorAction SilentlyContinue }
    $flat = Join-Path $HOME ".dotllm/models/$owner/$repo"
    if (Test-Path $flat) { $files += Get-ChildItem $flat -Recurse -Filter *.gguf -ErrorAction SilentlyContinue }
    $files = @($files | Where-Object { $_.Name -notmatch 'mmproj' })
    $hit = @($files | Where-Object { $_.Name -match [regex]::Escape($quant) })
    if ($hit.Count -gt 0) { return @{ Ok = $true; Detail = $hit[0].Name } }
    @{ Ok = $false; Detail = ("no cached *{0}* GGUF for {1}/{2}; cached: {3}" -f $quant, $owner, $repo, $(if ($files) { ($files.Name | Select-Object -Unique) -join ', ' } else { '(none)' })) }
}

function Get-ServerBuild {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $c) { return $null }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)" -ErrorAction SilentlyContinue
    if (-not $cim) { return $null }
    $dll = $null
    if ($cim.CommandLine -match '"?([^"\s]*DotLLM\.Cli\.dll)') { $dll = $Matches[1] }
    elseif ($cim.ExecutablePath -match 'DotLLM\.Cli\.exe$') { $dll = $cim.ExecutablePath -replace '\.exe$', '.dll' }
    if ($dll -and (Test-Path -LiteralPath $dll)) { return @{ Dll = $dll; ProductVersion = (Get-Item -LiteralPath $dll).VersionInfo.ProductVersion } }
    $null
}

# ---------------------------------------------------------------- main
$proc = $null; $skipRun = $false
$meta = [ordered]@{ model = $Model; cli = $Cli; device = $(if ($Device) { $Device } else { '(default)' }); serveArgs = $ServeArgs; started = (Get-Date).ToString('o'); host = $env:COMPUTERNAME }
try {
    # Guard: an explicit CUDA request on a host with no NVIDIA driver is "not measured here", not a failure.
    if ($Device -match '^(cuda|gpu)') {
        $hasCuda = (Test-Path (Join-Path $env:windir 'System32\nvcuda.dll')) -or [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
        if (-not $hasCuda) {
            $meta.skipped = 'no CUDA on this host'
            Add-Result 'harness' 'SKIP' "--device $Device requested but no NVIDIA driver (nvcuda.dll / nvidia-smi) on $($env:COMPUTERNAME): not measured here"
            $skipRun = $true
        }
    }
    if ($skipRun) { return }

    # Guard: refuse to start if the port is taken (a stale server would answer our checks).
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port already in use; pick another with -Port."
    }

    # Guard: `dotllm serve <ref>` pulls on a miss by itself (ModelResolver; `owner/repo` defaults to
    # Q4_K_M, which may not match an Unsloth UD-Q4_K_M file). Do not start a multi-GB download unless asked.
    $cached = Test-ModelCached $Model
    $meta.cacheCheck = $cached.Detail
    if (-not $cached.Ok -and -not $AllowPull) {
        throw "refusing to serve: $($cached.Detail). Pass the exact cached quant tag (owner/repo:QUANT), a file path, or -AllowPull."
    }
    Write-Host ("cache: {0}" -f $cached.Detail)

    # --- serve (default flags unless overridden). Pull time, if any, is part of serve-ready.
    # (`dotllm model pull` needs an interactive prompt without --file, so it is NOT used.)
    $serveCli = @($cliPre) + @('serve', $Model, '--port', "$Port", '--no-browser')
    if ($Device) { $serveCli += @('--device', $Device) }
    $serveCli += $ServeArgs
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $proc = Start-Process -FilePath $cliExe -ArgumentList $serveCli -PassThru -NoNewWindow `
        -RedirectStandardOutput $serverOut -RedirectStandardError $serverErr
    $ready = $false
    while ($sw.Elapsed.TotalSeconds -lt $ReadyTimeoutSec) {
        if ($proc.HasExited) { break }
        try {
            $r = Invoke-WebRequest -Uri "$base/ready" -TimeoutSec 3 -SkipHttpErrorCheck
            if ($r.StatusCode -eq 200) { $ready = $true; break }
        } catch { }
        if (-not $AllowPull -and (Test-Path $serverOut) -and ((Get-Content $serverOut -Raw -ErrorAction SilentlyContinue) -match '(?i)downloading|pulling ')) {
            throw 'serve started a download although -AllowPull was not given; aborting'
        }
        Start-Sleep -Milliseconds 500
    }
    $loadSec = $sw.Elapsed.TotalSeconds
    if (-not $ready) {
        $tail = if (Test-Path $serverErr) { (Get-Content $serverErr -Tail 8) -join ' | ' } else { '' }
        Add-Result 'serve-ready' 'FAIL' ("not ready after {0:N0}s (exited={1}) {2}" -f $loadSec, $proc.HasExited, $tail)
        throw 'server did not become ready'
    }
    $build = Get-ServerBuild
    if ($build) { $meta.buildProductVersion = $build.ProductVersion; $meta.buildDll = $build.Dll }
    Add-Result 'serve-ready' 'PASS' ("{0:N1}s to ready; build {1}" -f $loadSec, $(if ($build) { $build.ProductVersion } else { 'unknown' }))
    $meta.loadSeconds = [math]::Round($loadSec, 1)

    $props = (Invoke-Json GET '/props').Json
    if ($props) {
        $pa = Prop $props 'architecture'; $pd = Prop $props 'resolved_device'; $pm = Prop $props 'mtp_active'; $pw = Prop $props 'device_fallback_warning'
        $meta.architecture = $pa; $meta.resolvedDevice = $pd; $meta.mtpActive = $pm; $meta.mtpStatus = Prop $props 'mtp_status'
        $meta.deviceFallbackWarning = $pw; $meta.serverVersion = Prop $props 'version'; $meta.modelId = Prop $props 'model_id'; $meta.maxSeq = Prop $props 'max_sequence_length'
        $warn = if ($pw) { " FALLBACK: $pw" } else { '' }
        Add-Result 'props' $(if ($pw) { 'WARN' } else { 'PASS' }) ("arch={0} device={1} mtp={2}{3}" -f $pa, $pd, $pm, $warn) $props
    } else { Add-Result 'props' 'FAIL' '/props unreadable' }

    $models = Invoke-Json GET '/v1/models'
    $modelName = $null
    if ($models.Json -and @($models.Json.data).Count -gt 0) { $modelName = @($models.Json.data)[0].id }
    Add-Result 'v1/models' $(if ($modelName) { 'PASS' } else { 'FAIL' }) "id=$modelName"
    if (-not $modelName) { $modelName = $Model }

    # --- non-stream chat (retried with a larger budget when the model reasons inline)
    $meta.reasoningInline = $false; $meta.reasoningSplit = $false
    $budget = $MaxTokens
    $chatMsgs = @(@{ role = 'user'; content = 'What is the capital of France? Answer in one short sentence.' })
    for ($attempt = 0; $attempt -lt 2; $attempt++) {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-Json POST '/v1/chat/completions' (Plain @{ model = $modelName; temperature = 0; max_tokens = $budget; messages = $chatMsgs })
        $ms = $sw.Elapsed.TotalMilliseconds
        if ($r.Code -ne 200 -or -not $r.Json) { break }
        $msg = $r.Json.choices[0].message
        $sp = Split-Think ([string]$msg.content)
        $rc = [string](Prop $msg 'reasoning_content')
        if ($sp.HasThink) { $meta.reasoningInline = $true }
        if ($rc) { $meta.reasoningSplit = $true }
        # empty `content` because the model spent the budget thinking (split or inline): retry bigger once
        $spent = (($sp.HasThink -and -not $sp.Visible) -or ($rc -and -not ([string]$msg.content).Trim()))
        if ($spent -and $attempt -eq 0) { $budget = $ThinkingMaxTokens; continue }
        break
    }
    if ($r.Code -eq 200 -and $r.Json) {
        $raw = [string]$msg.content
        $hit = $sp.Visible -match '(?i)paris'
        $status = if ($hit -and -not $sp.HasThink) { 'PASS' } elseif ($hit) { 'WARN' } else { 'FAIL' }
        $note = if ($sp.HasThink) { ' [<think> text INLINE in content: reasoning not split (#767)]' } else { '' }
        if ($rc) { $note += ' [model still thinking with enable_thinking=false: reasoning_content populated]' }
        if (-not $hit -and $sp.Open) { $note += ' [think block never closed: out of tokens]' }
        if (-not $hit -and -not ([string]$msg.content).Trim()) { $note += ' [EMPTY content: budget spent on reasoning?]' }
        Add-Result 'chat-nonstream' $status ("{0:N0} ms finish={1} tokens={2}{3} :: {4}" -f $ms, $r.Json.choices[0].finish_reason, $r.Json.usage.completion_tokens, $note, (Cut $(if ($sp.Visible) { $sp.Visible } else { $raw }) 80)) @{ raw = $raw }
    } else { Add-Result 'chat-nonstream' 'FAIL' "HTTP $($r.Code): $(Cut $r.Text 200)" }
    $thinkingSeen = [bool]($meta.reasoningInline -or $meta.reasoningSplit)
    $toolBudget = if ($thinkingSeen) { $ThinkingMaxTokens } else { 384 }

    # --- streamed chat
    $s = Invoke-Stream (Plain @{ model = $modelName; temperature = 0; max_tokens = $budget
        messages = @(@{ role = 'user'; content = 'Count from 1 to 10 in words, separated by commas.' }) })
    if ($s.Ok) {
        $okS = ($s.Chunks -gt 1) -and $s.DoneSeen -and $s.Content -and $s.Finish
        $usageNote = if ($s.Usage) { '' } else { ' (no usage chunk)' }
        Add-Result 'chat-stream' $(if ($okS -and $s.Usage) { 'PASS' } elseif ($okS) { 'WARN' } else { 'FAIL' }) `
            ("chunks={0} ttft={1:N0}ms finish={2}{3}" -f $s.Chunks, $s.TtftMs, $s.Finish, $usageNote)
    } else { Add-Result 'chat-stream' 'FAIL' $s.Error }

    # --- tool call (+ round trip). GAP = model emitted tool markup the server did not parse.
    $weatherTool = @{ type = 'function'; function = @{
        name = 'get_weather'; description = 'Get the current weather for a city.'
        parameters = @{ type = 'object'; properties = @{ city = @{ type = 'string'; description = 'City name' } }; required = @('city') } } }
    $toolMsgs = @(@{ role = 'user'; content = 'What is the weather in Paris right now? Use the tool.' })
    $t = Invoke-Json POST '/v1/chat/completions' (Plain @{ model = $modelName; temperature = 0; max_tokens = $toolBudget; tools = @($weatherTool); messages = $toolMsgs })
    if ($t.Code -eq 200 -and $t.Json) {
        $m = $t.Json.choices[0].message
        $rawTool = [string]$m.content
        $tcs = if ($m.PSObject.Properties['tool_calls'] -and $m.tool_calls) { @($m.tool_calls) } else { @() }
        if ($tcs.Count -gt 0 -and $tcs[0].function.name -eq 'get_weather') {
            $argsOk = $false
            try { $a = $tcs[0].function.arguments | ConvertFrom-Json; $argsOk = ([string]$a.city -match '(?i)paris') } catch { }
            if ($argsOk) {
                $m2 = @($toolMsgs) + @(@{ role = 'assistant'; content = $null; tool_calls = @($tcs) }) +
                      @(@{ role = 'tool'; tool_call_id = $tcs[0].id; content = '{"temp_c": 17, "conditions": "light rain"}' })
                $t2 = Invoke-Json POST '/v1/chat/completions' (Plain @{ model = $modelName; temperature = 0; max_tokens = $toolBudget; tools = @($weatherTool); messages = $m2 })
                $final = if ($t2.Code -eq 200 -and $t2.Json) { (Split-Think ([string]$t2.Json.choices[0].message.content)).Visible } else { '' }
                $toolOk = $final -match '17|rain'
                Add-Result 'tool-call' $(if ($toolOk) { 'PASS' } else { 'WARN' }) ("call ok ({0}); round-trip answer {1}" -f $tcs[0].function.arguments, $(if ($toolOk) { 'uses result' } else { "missing result: '$(Cut $final 80)'" })) @{ raw = $rawTool; final = $final }
            } else { Add-Result 'tool-call' 'FAIL' "bad arguments: $($tcs[0].function.arguments)" @{ raw = $rawTool } }
        } else {
            $mk = Find-ToolMarkup $rawTool
            if ($mk) { Add-Result 'tool-call' 'GAP' ("model emitted tool markup '{0}' but no tool_calls were parsed :: {1}" -f $mk, (Cut $rawTool 120)) @{ raw = $rawTool } }
            else { Add-Result 'tool-call' 'FAIL' ("no tool_call and no tool markup; content='{0}'" -f (Cut $rawTool 100)) @{ raw = $rawTool } }
        }
    } else { Add-Result 'tool-call' 'FAIL' "HTTP $($t.Code): $(Cut $t.Text 200)" }

    # --- structured output (json_schema): exact keys, exact types, no extras
    $schema = @{ type = 'object'; properties = @{ name = @{ type = 'string' }; age = @{ type = 'integer' }; tags = @{ type = 'array'; items = @{ type = 'string' } } }
                 required = @('name', 'age', 'tags'); additionalProperties = $false }
    $j = Invoke-Json POST '/v1/chat/completions' @{ model = $modelName; temperature = 0; max_tokens = 200
        response_format = @{ type = 'json_schema'; json_schema = @{ name = 'person'; strict = $true; schema = $schema } }
        messages = @(@{ role = 'user'; content = 'Invent a person with a name, an age and two tags. Reply as JSON.' }) }
    if ($j.Code -eq 200 -and $j.Json) {
        $txt = [string]$j.Json.choices[0].message.content
        try {
            $o = $txt | ConvertFrom-Json
            $keys = @($o.PSObject.Properties.Name | Sort-Object)
            $badTag = @(@($o.tags) | Where-Object { $_ -isnot [string] }).Count
            $valid = (($keys -join ',') -eq 'age,name,tags') -and ($o.name -is [string]) -and ($o.age -is [int] -or $o.age -is [long]) -and ($badTag -eq 0)
            Add-Result 'structured-output' $(if ($valid) { 'PASS' } else { 'FAIL' }) (Cut $txt 160) @{ raw = $txt }
        } catch { Add-Result 'structured-output' 'FAIL' "not valid JSON: $(Cut $txt 160)" @{ raw = $txt } }
    } else { Add-Result 'structured-output' 'FAIL' "HTTP $($j.Code): $(Cut $j.Text 200)" }

    # --- typed arguments + parallel calls (#782 parsers: Qwen XML, Gemma-4, hardened Llama/Hermes).
    # A strict, multi-typed schema: strings must stay strings, ints/bools/arrays must be typed (ToolArgumentCoercer).
    $alarmTool = @{ type = 'function'; function = @{
        name = 'set_alarm'; description = 'Set an alarm.'
        parameters = @{ type = 'object'; properties = @{
            hour = @{ type = 'integer'; description = 'Hour 0-23' }; label = @{ type = 'string' }
            repeat = @{ type = 'boolean' }; days = @{ type = 'array'; items = @{ type = 'string' } } }
          required = @('hour', 'label') } } }
    $ta = Invoke-Json POST '/v1/chat/completions' (Plain @{ model = $modelName; temperature = 0; max_tokens = $toolBudget; tools = @($alarmTool)
        messages = @(@{ role = 'user'; content = 'Set an alarm for 7 o''clock labelled "gym", repeating on monday and friday. Use the tool.' }) })
    if ($ta.Code -eq 200 -and $ta.Json) {
        $am = $ta.Json.choices[0].message; $araw = [string]$am.content
        $atc = if ($am.PSObject.Properties['tool_calls'] -and $am.tool_calls) { @($am.tool_calls) } else { @() }
        if ($atc.Count -gt 0 -and $atc[0].function.name -eq 'set_alarm') {
            try {
                $aa = $atc[0].function.arguments | ConvertFrom-Json
                $typed = ($aa.hour -is [int] -or $aa.hour -is [long]) -and ([int]$aa.hour -eq 7) -and ($aa.label -is [string])
                $extra = ''
                if ($aa.PSObject.Properties['repeat']) { if ($aa.repeat -isnot [bool]) { $typed = $false; $extra += ' repeat!=bool' } }
                if ($aa.PSObject.Properties['days']) { if (@($aa.days).Count -eq 0 -or (@($aa.days) | Where-Object { $_ -isnot [string] })) { $typed = $false; $extra += ' days!=string[]' } }
                Add-Result 'tool-call-typed' $(if ($typed) { 'PASS' } else { 'FAIL' }) ("args {0}{1}" -f (Cut $atc[0].function.arguments 120), $extra) @{ raw = $araw }
            } catch { Add-Result 'tool-call-typed' 'FAIL' "arguments not JSON: $(Cut $atc[0].function.arguments 120)" }
        } else {
            $mk = Find-ToolMarkup $araw
            if ($mk) { Add-Result 'tool-call-typed' 'GAP' ("markup '{0}' not parsed :: {1}" -f $mk, (Cut $araw 120)) @{ raw = $araw } }
            else { Add-Result 'tool-call-typed' 'WARN' ("model made no call; content='{0}'" -f (Cut $araw 100)) @{ raw = $araw } }
        }
    } else { Add-Result 'tool-call-typed' 'FAIL' "HTTP $($ta.Code): $(Cut $ta.Text 200)" }

    $tp = Invoke-Json POST '/v1/chat/completions' (Plain @{ model = $modelName; temperature = 0; max_tokens = $toolBudget; tools = @($weatherTool); parallel_tool_calls = $true
        messages = @(@{ role = 'user'; content = 'What is the weather in Paris and in Tokyo? Call the tool once for each city.' }) })
    if ($tp.Code -eq 200 -and $tp.Json) {
        $pm = $tp.Json.choices[0].message; $praw = [string]$pm.content
        $ptc = if ($pm.PSObject.Properties['tool_calls'] -and $pm.tool_calls) { @($pm.tool_calls) } else { @() }
        $cities = @($ptc | ForEach-Object { try { [string](($_.function.arguments | ConvertFrom-Json).city) } catch { '' } } | Where-Object { $_ })
        if ($ptc.Count -ge 2 -and ($cities -join ',') -match '(?i)paris' -and ($cities -join ',') -match '(?i)tokyo') {
            Add-Result 'tool-call-parallel' 'PASS' ("{0} calls: {1}" -f $ptc.Count, ($cities -join ', '))
        } elseif ($ptc.Count -eq 1) {
            # one call is a legal (if lazy) model answer, but a parser that drops the second block looks the same: say so.
            Add-Result 'tool-call-parallel' 'WARN' ("only 1 call parsed ({0}); model laziness OR parser dropped a block - raw: {1}" -f ($cities -join ', '), (Cut $praw 80)) @{ raw = $praw }
        } else {
            $mk = Find-ToolMarkup $praw
            if ($mk) { Add-Result 'tool-call-parallel' 'GAP' ("markup '{0}' not parsed :: {1}" -f $mk, (Cut $praw 120)) @{ raw = $praw } }
            else { Add-Result 'tool-call-parallel' 'WARN' ("no calls; content='{0}'" -f (Cut $praw 100)) @{ raw = $praw } }
        }
    } else { Add-Result 'tool-call-parallel' 'FAIL' "HTTP $($tp.Code): $(Cut $tp.Text 200)" }

    # streamed tool call: tool_calls deltas must carry the call and raw markup must NOT leak as delta.content (#782)
    $ts = Invoke-Stream (Plain @{ model = $modelName; temperature = 0; max_tokens = $toolBudget; tools = @($weatherTool); messages = $toolMsgs })
    if ($ts.Ok) {
        $leak = Find-ToolMarkup $ts.Content
        if ($ts.ToolCalls.Count -gt 0 -and $ts.ToolCalls[0].name -eq 'get_weather') {
            $argOk = $false; try { $argOk = ([string](($ts.ToolCalls[0].arguments | ConvertFrom-Json).city) -match '(?i)paris') } catch { }
            $st = if ($argOk -and -not $leak -and $ts.Finish -eq 'tool_calls') { 'PASS' } elseif ($argOk -and -not $leak) { 'WARN' } else { 'FAIL' }
            Add-Result 'tool-call-stream' $st ("finish={0} args={1}{2}" -f $ts.Finish, (Cut $ts.ToolCalls[0].arguments 60), $(if ($leak) { " LEAK '$leak' in delta.content" } else { '' }))
        } elseif ($leak) { Add-Result 'tool-call-stream' 'GAP' ("markup '{0}' leaked as delta.content, no tool_calls delta :: {1}" -f $leak, (Cut $ts.Content 100)) }
        else { Add-Result 'tool-call-stream' 'WARN' ("no tool call streamed; content='{0}'" -f (Cut $ts.Content 100)) }
    } else { Add-Result 'tool-call-stream' 'FAIL' $ts.Error }

    # --- reasoning (#767/#778): thinking ON with an adequate budget. A reasoning model must put its thinking in
    # `reasoning_content` and the answer in `content` with NO <think> text in `content`. A non-reasoning model -> SKIP.
    $rq = @(@{ role = 'user'; content = 'A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How many cents is the ball? Answer with just the number.' })
    $rr = Invoke-Json POST '/v1/chat/completions' @{ model = $modelName; temperature = 0; max_tokens = $ThinkingMaxTokens; enable_thinking = $true; messages = $rq }
    if ($rr.Code -eq 200 -and $rr.Json) {
        $rm = $rr.Json.choices[0].message; $rcontent = [string]$rm.content; $rreason = [string](Prop $rm 'reasoning_content')
        $inline = Split-Think $rcontent
        $rtoks = $null; $dets = Prop $rr.Json.usage 'completion_tokens_details'; if ($dets) { $rtoks = Prop $dets 'reasoning_tokens' }
        $fin = $rr.Json.choices[0].finish_reason
        $meta.reasoningTokens = $rtoks
        if ($rreason) {
            $has5 = $rcontent -match '(?<!\d)5(?!\d)'
            $st = if ($inline.HasThink) { 'FAIL' } elseif (-not $rcontent.Trim()) { 'WARN' } elseif ($has5) { 'PASS' } else { 'WARN' }
            $why = if ($inline.HasThink) { ' <think> leaked into content' } elseif (-not $rcontent.Trim()) { " EMPTY content (finish=${fin}: budget $ThinkingMaxTokens spent thinking)" } elseif (-not $has5) { ' answer not 5' } else { '' }
            Add-Result 'reasoning' $st ("reasoning_content {0} chars, reasoning_tokens={1}, content='{2}'{3}" -f $rreason.Length, $rtoks, (Cut $rcontent 40), $why) @{ reasoning = (Cut $rreason 300); content = $rcontent }
        } elseif ($inline.HasThink) {
            Add-Result 'reasoning' 'FAIL' ("<think> text INLINE in content and reasoning_content empty: the split did not engage :: {0}" -f (Cut $rcontent 80)) @{ content = $rcontent }
        } else {
            Add-Result 'reasoning' 'SKIP' ("no reasoning emitted with enable_thinking=true (non-reasoning template); content='{0}'" -f (Cut $rcontent 60))
        }
        $meta.reasoningCapable = [bool]($rreason -or $inline.HasThink)
    } else { Add-Result 'reasoning' 'FAIL' "HTTP $($rr.Code): $(Cut $rr.Text 200)" }

    if ($meta.reasoningCapable) {
        $rs = Invoke-Stream @{ model = $modelName; temperature = 0; max_tokens = $ThinkingMaxTokens; enable_thinking = $true; messages = $rq }
        if ($rs.Ok) {
            $inl = Split-Think $rs.Content
            $st = if ($inl.HasThink) { 'FAIL' } elseif (-not $rs.Reasoning) { 'FAIL' } elseif (-not $rs.Content.Trim()) { 'WARN' } else { 'PASS' }
            Add-Result 'reasoning-stream' $st ("reasoning_content deltas {0} chars; content='{1}' finish={2}{3}" -f $rs.Reasoning.Length, (Cut $rs.Content 40), $rs.Finish, $(if ($inl.HasThink) { ' <think> leaked into delta.content' } else { '' }))
        } else { Add-Result 'reasoning-stream' 'FAIL' $rs.Error }
    }

    # --- throughput (steady state): 1 warm-up discarded, then N measured, median + per-run values.
    # MTP models: the adaptive gate probes both arms over the first requests, so use >= 6 runs and
    # re-read /props mtp_status afterwards.
    $mtpInvolved = ($meta.mtpActive -eq $true) -or ($meta.mtpStatus)
    $effRuns = if ($mtpInvolved) { [math]::Max($Runs, 6) } else { $Runs }
    $decode = [System.Collections.Generic.List[double]]::new(); $wall = [System.Collections.Generic.List[double]]::new()
    $genPrompt = @(@{ role = 'user'; content = 'Write a long, detailed story about a lighthouse keeper. Do not stop early.' })
    for ($i = 0; $i -le $effRuns; $i++) {
        $s = Invoke-Stream (Plain @{ model = $modelName; temperature = 0; max_tokens = $MaxTokens; messages = $genPrompt })
        if (-not $s.Ok) { Add-Result 'decode-tps' 'FAIL' $s.Error; break }
        if ($i -eq 0) { continue }   # warm-up
        if ($s.Timings -and $s.Timings.decode_tokens_per_sec) { $decode.Add([double]$s.Timings.decode_tokens_per_sec) }
        if ($null -ne $s.WallDecodeTps) { $wall.Add([double]$s.WallDecodeTps) }
    }
    if ($decode.Count -gt 0 -or $wall.Count -gt 0) {
        $meta.decodeTpsMedian = Get-Median $decode; $meta.decodeTpsBest = ($decode | Measure-Object -Maximum).Maximum
        $meta.decodeTpsWallMedian = Get-Median $wall; $meta.decodeTpsRuns = @($decode | ForEach-Object { [math]::Round($_, 1) })
        Add-Result 'decode-tps' 'PASS' ("server median {0:N1} (best {1:N1}); client-wall median {2:N1}; n={3} max_tokens={4}; runs: {5}" -f $meta.decodeTpsMedian, $meta.decodeTpsBest, $meta.decodeTpsWallMedian, $decode.Count, $MaxTokens, ($meta.decodeTpsRuns -join ' '))
    }
    $props2 = (Invoke-Json GET '/props').Json
    if ($props2) { $meta.mtpActiveAfter = Prop $props2 'mtp_active'; $meta.mtpStatusAfter = Prop $props2 'mtp_status' }

    # prefill: a long prompt, 1 new token
    $para = 'The quick brown fox jumps over the lazy dog while the committee reviews quarterly infrastructure budgets in considerable detail. '
    $long = ($para * [math]::Ceiling($PrefillWords / 19.0))
    $pf = [System.Collections.Generic.List[double]]::new(); $pTok = 0
    for ($i = 0; $i -le [math]::Min($Runs, 2); $i++) {
        # vary the first words so the prompt cache cannot hide prefill
        $s = Invoke-Stream (Plain @{ model = $modelName; temperature = 0; max_tokens = 1
            messages = @(@{ role = 'user'; content = "[$i-$stamp] $long`nSummarise the above in five words." }) })
        if (-not $s.Ok) { Add-Result 'prefill-tps' 'FAIL' $s.Error; break }
        if ($i -eq 0) { continue }
        if ($s.Timings -and $s.Timings.prefill_tokens_per_sec) { $pf.Add([double]$s.Timings.prefill_tokens_per_sec); $pTok = $s.Timings.prompt_tokens }
    }
    if ($pf.Count -gt 0) {
        $meta.prefillTpsMedian = Get-Median $pf; $meta.prefillPromptTokens = $pTok
        Add-Result 'prefill-tps' 'PASS' ("median {0:N0} tok/s over a {1}-token prompt" -f $meta.prefillTpsMedian, $pTok)
    }
}
catch {
    Add-Result 'harness' 'FAIL' "$($_.Exception.Message)"
}
finally {
    Stop-ServerTree $proc
    $http.Dispose()
    $meta.finished = (Get-Date).ToString('o')
    $failed = @($results | Where-Object status -eq 'FAIL').Count
    $meta.failed = $failed
    @{ meta = $meta; checks = $results } | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 $resultPath
    Write-Host ''
    Write-Host ("result: {0}  (server logs: {1})" -f $resultPath, $serverErr)
}
if ($failed -gt 0) { exit 1 } else { exit 0 }
