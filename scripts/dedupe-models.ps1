#Requires -Version 7.0
<#
.SYNOPSIS
    Collapses duplicate model files across dotLLM's model stores into hardlinks (#1279).

.DESCRIPTION
    Scans the three places a model can live:
      - the dotLLM model store   $DOTLLM_MODELS_DIR or ~/.dotllm/models   ({org}/{repo}/...)
      - the legacy test cache    ~/.dotllm/test-cache                    ({org}/{repo}/...)
      - the Hugging Face hub     $HF_HUB_CACHE / $HF_HOME/hub / ~/.cache/huggingface/hub

    Phase 1 (migration): every model file at test-cache/{org}/{repo}/... moves to the same
    relative path under the model store. If the store already holds an identical file (same
    SHA-256) the test-cache name is removed instead; a different file there is a conflict and
    is left alone. Generated, non-model fixtures (corpora, token-oracles, mimo2-tiny*,
    nanoquant-oracle, littlebit-*) are never moved.

    Phase 2 (dedupe): files are grouped by size, candidates confirmed by streaming SHA-256
    (cached by path+size+mtime in ~/.dotllm/dedupe-hash-cache.json), and every duplicate set on
    one volume is reduced to one physical copy: the others are replaced by hardlinks to it.
    The kept copy is an HF blob when there is one (HF snapshot symlinks are resolved to their
    blob, and nothing under the hub cache is ever modified), else the model-store file.
    Cross-volume duplicates are reported, never linked. A replacement is made by creating the
    hardlink beside the duplicate, checking it is the keeper's file, then renaming it over the
    duplicate, so the bytes are never down to zero copies.

    Dry run by default: prints the planned actions and the GB they would reclaim.

.PARAMETER Apply
    Perform the actions. Without it nothing on disk changes (only the hash cache is written).

.PARAMETER SelfTest
    Builds throwaway stores in a temp directory, runs the script's logic against them with
    -Apply, and checks the outcome. Touches nothing else.

.PARAMETER MinBinSizeMB
    *.bin files smaller than this are ignored (small .bin files are rarely weights).

.EXAMPLE
    pwsh scripts/dedupe-models.ps1            # dry run
    pwsh scripts/dedupe-models.ps1 -Apply     # act
    pwsh scripts/dedupe-models.ps1 -SelfTest
#>
[CmdletBinding()]
param(
    [switch] $Apply,
    [switch] $SelfTest,
    [int] $MinBinSizeMB = 100,
    [string] $ModelsDir,
    [string] $TestCacheDir,
    [string] $HubDir,
    [string] $HashCachePath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$UserHome = [Environment]::GetFolderPath('UserProfile')
$GeneratedTestCacheDirs = @('corpora', 'token-oracles', 'mimo2-tiny', 'mimo2-tiny-k256', 'nanoquant-oracle', 'quant-ladder')
$GeneratedTestCachePrefixes = @('littlebit-')

function Get-DefaultHubDir {
    foreach ($v in 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE') {
        $raw = [Environment]::GetEnvironmentVariable($v)
        if (-not [string]::IsNullOrWhiteSpace($raw)) { return [IO.Path]::GetFullPath($raw) }
    }
    if (-not [string]::IsNullOrWhiteSpace($env:HF_HOME)) { return Join-Path ([IO.Path]::GetFullPath($env:HF_HOME)) 'hub' }
    return Join-Path $UserHome '.cache/huggingface/hub'
}

# ── Physical file identity (volume + file id + link count) ──

if ($IsWindows -and -not ('DotLlmDedupe.FileId' -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
namespace DotLlmDedupe {
    public static class FileId {
        // BY_HANDLE_FILE_INFORMATION: FILETIMEs are two DWORDs (4-byte aligned), hence Pack = 4.
        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        private struct Info {
            public uint Attributes; public long Created, Accessed, Written;
            public uint VolumeSerial, SizeHigh, SizeLow, Links, IndexHigh, IndexLow;
        }
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetFileInformationByHandle(SafeFileHandle h, out Info info);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        private static extern SafeFileHandle CreateFileW(string name, uint access, uint share, IntPtr sa, uint disposition, uint flags, IntPtr template);
        // Returns { volume, "volume:index", links }. Opens with no data access (query only) and
        // FILE_FLAG_BACKUP_SEMANTICS so directories and files in use both work.
        public static object[] Get(string path) {
            using (SafeFileHandle h = CreateFileW(path, 0, 7, IntPtr.Zero, 3, 0x02000000, IntPtr.Zero)) {
                if (h.IsInvalid) throw new IOException("cannot open " + path + " (Win32 error " + Marshal.GetLastWin32Error() + ")");
                Info i;
                if (!GetFileInformationByHandle(h, out i)) throw new IOException("GetFileInformationByHandle failed: " + path);
                ulong index = ((ulong)i.IndexHigh << 32) | i.IndexLow;
                return new object[] { i.VolumeSerial.ToString("X8"), i.VolumeSerial.ToString("X8") + ":" + index, (int)i.Links };
            }
        }
    }
}
'@
}

function Get-FileIdentity([string] $Path) {
    if ($IsWindows) {
        $r = [DotLlmDedupe.FileId]::Get($Path)
        return [pscustomobject]@{ Volume = $r[0]; Id = $r[1]; Links = $r[2] }
    }

    # GNU stat, then BSD/macOS stat.
    $out = & stat -L -c '%d %i %h' -- $Path 2>$null
    if ($LASTEXITCODE -ne 0) { $out = & stat -L -f '%d %i %l' -- $Path }
    $dev, $ino, $links = "$out".Trim() -split '\s+'
    return [pscustomobject]@{ Volume = $dev; Id = "${dev}:$ino"; Links = [int]$links }
}

# ── Inventory ──

function Test-ModelFile([IO.FileInfo] $Physical, [string] $Name) {
    $ext = [IO.Path]::GetExtension($Name).ToLowerInvariant()
    if ($ext -in '.gguf', '.safetensors') { return $true }
    return $ext -eq '.bin' -and $Physical.Length -ge ($MinBinSizeMB * 1MB)
}

function Get-StoreFiles([string] $Root, [string] $Store) {
    if (-not $Root -or -not (Test-Path -LiteralPath $Root -PathType Container)) { return }
    $opts = [IO.EnumerationOptions]::new()
    $opts.RecurseSubdirectories = $true
    $opts.IgnoreInaccessible = $true
    $opts.AttributesToSkip = [IO.FileAttributes]::None
    $searchRoot = $Root
    foreach ($path in [IO.Directory]::EnumerateFiles($searchRoot, '*', $opts)) {
        if ($path -match '\.(incomplete|dedupe-tmp)$') { continue }
        $rel = [IO.Path]::GetRelativePath($Root, $path)
        if ($Store -eq 'hf') {
            # Only snapshot entries name files; blobs/ is reached through them.
            $parts = $rel -split '[\\/]'
            if ($parts.Count -lt 4 -or $parts[1] -ne 'snapshots') { continue }
        }

        $fi = [IO.FileInfo]::new($path)
        $physical = $fi
        if ($fi.LinkTarget) {
            try { $physical = $fi.ResolveLinkTarget($true) } catch { Write-Warning "unresolvable link: $path"; continue }
            if (-not $physical -or -not $physical.Exists) { continue }
        }
        if (-not (Test-ModelFile $physical $path)) { continue }

        try { $ident = Get-FileIdentity $physical.FullName } catch { Write-Warning "skipping ${path}: $($_.Exception.Message)"; continue }
        [pscustomobject]@{
            Path     = $path
            Rel      = $rel
            Store    = $Store
            Physical = $physical.FullName
            IsLink   = [bool]$fi.LinkTarget
            Size     = $physical.Length
            Volume   = $ident.Volume
            Id       = $ident.Id
            Links    = $ident.Links
        }
    }
}

function Get-Inventory($Stores) {
    $list = [Collections.Generic.List[object]]::new()
    foreach ($s in 'models', 'test-cache', 'hf') {
        foreach ($r in Get-StoreFiles $Stores[$s] $s) { $list.Add($r) }
    }
    return , $list
}

# ── Hashing (streamed, cached) ──

function Read-HashCache([string] $Path) {
    if (Test-Path -LiteralPath $Path) {
        try { return (Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json -AsHashtable) ?? @{} } catch { Write-Warning "ignoring unreadable hash cache $Path" }
    }
    return @{}
}

function Get-Sha256([string] $Path, [hashtable] $Cache, [string] $CachePath) {
    $fi = [IO.FileInfo]::new($Path)
    $key = '{0}|{1}|{2}' -f $fi.FullName, $fi.Length, $fi.LastWriteTimeUtc.Ticks
    if ($Cache.ContainsKey($key)) { return $Cache[$key] }

    Write-Host ("  hashing {0:N1} GB  {1}" -f ($fi.Length / 1GB), $fi.FullName)
    $fs = [IO.FileStream]::new($fi.FullName, 'Open', 'Read', [IO.FileShare]'ReadWrite, Delete', 4MB, [IO.FileOptions]::SequentialScan)
    try { $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($fs)) } finally { $fs.Dispose() }
    $Cache[$key] = $hash
    $dir = Split-Path -Parent $CachePath
    if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
    $Cache | ConvertTo-Json -Depth 2 | Set-Content -LiteralPath $CachePath -Encoding utf8NoBOM
    return $hash
}

function Test-SameHash([string] $A, [string] $B, [hashtable] $Cache, [string] $CachePath) {
    try { return (Get-Sha256 $A $Cache $CachePath) -eq (Get-Sha256 $B $Cache $CachePath) }
    catch { Write-Warning "cannot hash ($A, $B): $($_.Exception.Message)"; return $false }
}

# ── Planning ──

function New-Action($Kind, $Path, $Target, [long] $Size, [long] $Reclaim, $Detail) {
    [pscustomobject]@{ Action = $Kind; Path = $Path; Target = $Target; Size = $Size; Reclaim = $Reclaim; Detail = $Detail }
}

function Test-GeneratedTestCacheEntry([string] $Rel) {
    $parts = $Rel -split '[\\/]'
    if ($parts.Count -lt 3) { return $true }   # not {org}/{repo}/{file}
    if ($parts[0] -in $GeneratedTestCacheDirs) { return $true }
    foreach ($p in $GeneratedTestCachePrefixes) { if ($parts[0].StartsWith($p, [StringComparison]::OrdinalIgnoreCase)) { return $true } }
    return $false
}

# Phase 1: test-cache/{org}/{repo}/... -> models/{org}/{repo}/... . Mutates $Inv to the
# post-migration state so phase 2 plans against it (dry run and apply alike).
function Get-MigrationPlan($Inv, $Stores, [hashtable] $Cache, [string] $CachePath) {
    $actions = [Collections.Generic.List[object]]::new()
    foreach ($t in @($Inv | Where-Object Store -eq 'test-cache')) {
        if (Test-GeneratedTestCacheEntry $t.Rel) { continue }
        $target = Join-Path $Stores['models'] $t.Rel
        $existing = $Inv | Where-Object { $_.Store -eq 'models' -and $_.Path -eq $target } | Select-Object -First 1
        if (-not $existing -and (Test-Path -LiteralPath $target)) {
            $actions.Add((New-Action 'conflict' $t.Path $target $t.Size 0 'target exists but is not a model file')); continue
        }

        if (-not $existing) {
            $tVol = (Get-FileIdentity (Split-Path -Parent $t.Physical)).Volume
            $mRoot = $Stores['models']
            if (-not (Test-Path -LiteralPath $mRoot)) { New-Item -ItemType Directory -Path $mRoot -Force | Out-Null }
            if ($tVol -ne (Get-FileIdentity $mRoot).Volume) {
                $actions.Add((New-Action 'cross-volume' $t.Path $target $t.Size 0 'move would copy bytes; left in place')); continue
            }
            $actions.Add((New-Action 'move' $t.Path $target $t.Size 0 'rename into model store'))
            # Physical stays the current on-disk path: phase 2 hashes it, and in a dry run the target does not exist.
            $t.Path = $target; $t.Store = 'models'
            continue
        }

        if ($existing.Id -eq $t.Id) {
            $actions.Add((New-Action 'remove-legacy-name' $t.Path $target $t.Size 0 'already the same file (hardlink)'))
        }
        elseif ($existing.Size -eq $t.Size -and (Test-SameHash $t.Physical $existing.Physical $Cache $CachePath)) {
            $sharers = @($Inv | Where-Object Id -eq $t.Id).Count
            $reclaim = if ($t.Links -le $sharers) { $t.Size } else { 0 }
            $actions.Add((New-Action 'remove-identical-copy' $t.Path $target $t.Size $reclaim 'identical bytes already in model store'))
        }
        else {
            $actions.Add((New-Action 'conflict' $t.Path $target $t.Size 0 'model store holds a different file at this path')); continue
        }

        foreach ($r in @($Inv | Where-Object Id -eq $t.Id)) { $r.Links-- }
        [void]$Inv.Remove($t)
    }
    return , $actions
}

# Phase 2: identical bytes across distinct physical files -> hardlinks to one keeper.
function Get-DedupePlan($Inv, [hashtable] $Cache, [string] $CachePath) {
    $actions = [Collections.Generic.List[object]]::new()
    $storeRank = @{ 'hf' = 0; 'models' = 1; 'test-cache' = 2 }

    foreach ($sizeGroup in $Inv | Group-Object Size | Where-Object { @($_.Group.Id | Sort-Object -Unique).Count -gt 1 }) {
        $byHash = @{}
        foreach ($idGroup in $sizeGroup.Group | Group-Object Id) {
            # Another process may remove a file mid-scan (e.g. a sweep harness cleaning its links).
            try { $h = Get-Sha256 $idGroup.Group[0].Physical $Cache $CachePath }
            catch { Write-Warning "skipping $($idGroup.Group[0].Physical): $($_.Exception.Message)"; continue }
            if (-not $byHash.ContainsKey($h)) { $byHash[$h] = [Collections.Generic.List[object]]::new() }
            $byHash[$h].Add($idGroup)
        }

        foreach ($h in $byHash.Keys) {
            $ids = $byHash[$h]
            if ($ids.Count -lt 2) { continue }

            # Keeper: an HF blob if any (it must stay untouched), else the model store, then path order.
            $ranked = $ids | Sort-Object { ($_.Group | ForEach-Object { $storeRank[$_.Store] } | Measure-Object -Minimum).Minimum }, { $_.Group[0].Physical }
            $keeper = $ranked[0].Group[0]
            foreach ($dup in $ranked | Select-Object -Skip 1) {
                $records = @($dup.Group)
                $size = $records[0].Size
                if ($records | Where-Object Store -eq 'hf') {
                    $actions.Add((New-Action 'skip-hf' $records[0].Physical $keeper.Physical $size 0 'duplicate inside the HF cache; HF content is never modified')); continue
                }
                if ($records[0].Volume -ne $keeper.Volume) {
                    foreach ($r in $records) { $actions.Add((New-Action 'cross-volume' $r.Path $keeper.Physical $size 0 'hardlinks cannot span volumes')) }
                    continue
                }
                $reclaim = if ($records[0].Links -le $records.Count) { $size } else { 0 }
                $first = $true
                foreach ($r in $records) {
                    $detail = if ($reclaim -eq 0 -and $first) { 'other hardlinks keep these bytes alive' } else { "sha256 $($h.Substring(0, 12))" }
                    $actions.Add((New-Action 'link' $r.Path $keeper.Physical $size ($(if ($first) { $reclaim } else { 0 })) $detail))
                    $first = $false
                }
            }
        }
    }
    return , $actions
}

# ── Applying ──

function Remove-EmptyParents([string] $Path, [string] $StopAt) {
    $dir = Split-Path -Parent $Path
    $stop = [IO.Path]::GetFullPath($StopAt).TrimEnd('\', '/')
    while ($dir -and $dir.TrimEnd('\', '/') -ne $stop -and (Test-Path -LiteralPath $dir) -and -not (Get-ChildItem -LiteralPath $dir -Force | Select-Object -First 1)) {
        Remove-Item -LiteralPath $dir
        $dir = Split-Path -Parent $dir
    }
}

function Invoke-Action($A, $Stores) {
    switch ($A.Action) {
        'move' {
            New-Item -ItemType Directory -Path (Split-Path -Parent $A.Target) -Force | Out-Null
            [IO.File]::Move($A.Path, $A.Target, $false)
            Remove-EmptyParents $A.Path $Stores['test-cache']
        }
        { $_ -in 'remove-legacy-name', 'remove-identical-copy' } {
            # The model-store copy is confirmed present (same file id, or same SHA-256) — this is not the last copy.
            if (-not (Test-Path -LiteralPath $A.Target)) { throw "refusing to remove $($A.Path): $($A.Target) is missing" }
            Remove-Item -LiteralPath $A.Path
            Remove-EmptyParents $A.Path $Stores['test-cache']
        }
        'link' {
            $tmp = "$($A.Path).dedupe-tmp"
            if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp }
            New-Item -ItemType HardLink -Path $tmp -Target $A.Target | Out-Null
            try {
                if ((Get-FileIdentity $tmp).Id -ne (Get-FileIdentity $A.Target).Id) { throw "new link $tmp is not the keeper's file" }
                [IO.File]::Move($tmp, $A.Path, $true)   # atomic replace on one volume
            }
            catch {
                Remove-Item -LiteralPath $tmp -ErrorAction SilentlyContinue
                throw
            }
        }
    }
}

# ── Driver ──

function Invoke-Dedupe($Stores, [string] $CachePath, [bool] $DoApply, [bool] $Quiet) {
    $cache = Read-HashCache $CachePath
    foreach ($s in $Stores.Keys) { if (-not $Quiet) { Write-Host ("{0,-10} {1}" -f $s, $Stores[$s]) } }

    $inv = Get-Inventory $Stores
    if (-not $Quiet) { Write-Host ("Scanned {0} model file entries." -f $inv.Count) }

    $migration = Get-MigrationPlan $inv $Stores $cache $CachePath
    if ($DoApply) {
        foreach ($a in $migration | Where-Object Action -in 'move', 'remove-legacy-name', 'remove-identical-copy') {
            try { Invoke-Action $a $Stores } catch { $a.Action = 'failed'; $a.Detail = $_.Exception.Message }
        }
        $inv = Get-Inventory $Stores   # plan phase 2 against what is really on disk now
    }

    $dedupe = Get-DedupePlan $inv $cache $CachePath
    if ($DoApply) {
        foreach ($a in $dedupe | Where-Object Action -eq 'link') {
            try { Invoke-Action $a $Stores } catch { $a.Action = 'failed'; $a.Reclaim = 0; $a.Detail = $_.Exception.Message }
        }
    }

    $all = @($migration) + @($dedupe)
    if (-not $Quiet) {
        $all | Sort-Object Action, Path | Format-Table -AutoSize -Wrap @{ n = 'Action'; e = { $_.Action } },
            @{ n = 'GB'; e = { '{0:N2}' -f ($_.Size / 1GB) }; a = 'right' },
            @{ n = 'Reclaim GB'; e = { '{0:N2}' -f ($_.Reclaim / 1GB) }; a = 'right' },
            @{ n = 'Path'; e = { $_.Path } }, @{ n = 'Target / keeper'; e = { $_.Target } }, Detail | Out-Host

        Write-Host 'Summary:'
        $all | Group-Object Action | ForEach-Object {
            Write-Host ("  {0,-22} {1,4}  {2,9:N2} GB reclaim" -f $_.Name, $_.Count, (($_.Group | Measure-Object Reclaim -Sum).Sum / 1GB))
        }
        $total = ($all | Where-Object Action -ne 'failed' | Measure-Object Reclaim -Sum).Sum
        $mode = if ($DoApply) { 'reclaimed' } else { 'reclaimable (dry run — pass -Apply to act)' }
        Write-Host ("Total {0}: {1:N2} GB" -f $mode, ($total / 1GB))
    }
    return , $all
}

function Invoke-SelfTest {
    $root = Join-Path ([IO.Path]::GetTempPath()) ("dotllm-dedupe-selftest-" + [guid]::NewGuid().ToString('N'))
    $stores = [ordered]@{
        'models'     = Join-Path $root 'models'
        'test-cache' = Join-Path $root 'test-cache'
        'hf'         = Join-Path $root 'hub'
    }
    $failures = [Collections.Generic.List[string]]::new()
    function Check([bool] $Ok, [string] $What) { if ($Ok) { Write-Host "  PASS $What" } else { Write-Host "  FAIL $What"; $failures.Add($What) } }
    function Put([string] $Path, [byte[]] $Bytes) { New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null; [IO.File]::WriteAllBytes($Path, $Bytes) }

    try {
        $a = [byte[]](1..64); $b = [byte[]](64..1)   # same size, different content
        $snap = Join-Path $stores['hf'] 'models--o--r/snapshots/abc'
        $blob = Join-Path $stores['hf'] 'models--o--r/blobs/deadbeef'
        Put $blob $a
        $hfEntry = Join-Path $snap 'm.gguf'
        New-Item -ItemType Directory -Path $snap -Force | Out-Null
        try { New-Item -ItemType SymbolicLink -Path $hfEntry -Target $blob | Out-Null } catch { Copy-Item -LiteralPath $blob $hfEntry }

        # Identity probe sanity: one name -> 1 link, and the volume matches the temp root's.
        $probe = Join-Path $root 'probe.bin'; Put $probe ([byte[]](1))
        $pid1 = Get-FileIdentity $probe
        New-Item -ItemType HardLink -Path (Join-Path $root 'probe-link.bin') -Target $probe | Out-Null
        $pid2 = Get-FileIdentity $probe
        Check ($pid1.Links -eq 1 -and $pid2.Links -eq 2 -and $pid1.Id -eq $pid2.Id -and $pid1.Volume -eq (Get-FileIdentity $root).Volume) 'file identity: link count, id and volume'
        Check ((Get-FileIdentity $blob).Id -ne $pid1.Id) 'file identity: distinct files differ'

        $dupOfHf   = Join-Path $stores['models'] 'o/r/m.gguf';        Put $dupOfHf $a       # identical -> linked to the blob
        $sameSize  = Join-Path $stores['models'] 'o/r/other.gguf';    Put $sameSize $b      # same size, different -> untouched
        $legacy    = Join-Path $stores['test-cache'] 'x/y/z.gguf';    Put $legacy ([byte[]](7, 7, 7))   # -> moved
        $legacyDup = Join-Path $stores['test-cache'] 'o/r/m.gguf';    Put $legacyDup $a     # identical copy already in models -> removed
        $generated = Join-Path $stores['test-cache'] 'mimo2-tiny/t.gguf'; Put $generated ([byte[]](9))   # generated -> untouched

        $dry = Invoke-Dedupe $stores (Join-Path $root 'hash-cache.json') $false $true
        Check ((Get-FileIdentity $dupOfHf).Id -ne (Get-FileIdentity $blob).Id) 'dry run changes nothing'
        Check (@($dry | Where-Object Action -eq 'link').Count -ge 1) 'dry run plans a link'

        $null = Invoke-Dedupe $stores (Join-Path $root 'hash-cache.json') $true $true
        Check ((Get-FileIdentity $dupOfHf).Id -eq (Get-FileIdentity $blob).Id) 'identical model-store file is now a hardlink to the HF blob'
        Check ([IO.File]::ReadAllBytes($blob).Length -eq 64 -and (Test-Path -LiteralPath $hfEntry)) 'HF blob and snapshot entry intact'
        Check ((Get-FileIdentity $sameSize).Id -ne (Get-FileIdentity $blob).Id) 'same size, different content: not linked'
        Check ([Linq.Enumerable]::SequenceEqual([IO.File]::ReadAllBytes($sameSize), $b)) 'same size, different content: bytes unchanged'
        Check ((Test-Path -LiteralPath (Join-Path $stores['models'] 'x/y/z.gguf')) -and -not (Test-Path -LiteralPath $legacy)) 'legacy test-cache model moved into the model store'
        Check (-not (Test-Path -LiteralPath $legacyDup)) 'legacy identical copy removed'
        Check (Test-Path -LiteralPath $generated) 'generated test-cache fixture untouched'

        $again = Invoke-Dedupe $stores (Join-Path $root 'hash-cache.json') $false $true
        Check (@($again | Where-Object Action -in 'link', 'move').Count -eq 0) 'second run is a no-op'
    }
    finally {
        Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
    }

    if ($failures.Count) { Write-Host "SELF-TEST FAILED ($($failures.Count))"; exit 1 }
    Write-Host 'SELF-TEST PASSED'
}

if ($SelfTest) { Invoke-SelfTest; return }

$stores = [ordered]@{
    'models'     = if ($ModelsDir) { [IO.Path]::GetFullPath($ModelsDir) } elseif ($env:DOTLLM_MODELS_DIR) { [IO.Path]::GetFullPath($env:DOTLLM_MODELS_DIR) } else { Join-Path $UserHome '.dotllm/models' }
    'test-cache' = if ($TestCacheDir) { [IO.Path]::GetFullPath($TestCacheDir) } else { Join-Path $UserHome '.dotllm/test-cache' }
    'hf'         = if ($HubDir) { [IO.Path]::GetFullPath($HubDir) } else { Get-DefaultHubDir }
}
$cachePath = if ($HashCachePath) { $HashCachePath } else { Join-Path $UserHome '.dotllm/dedupe-hash-cache.json' }
$null = Invoke-Dedupe $stores $cachePath $Apply.IsPresent $false
