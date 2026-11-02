# Chunk-cache client (`DotLLM.ChunkCache.Client`)

The client half of the chunk-cache wire convention v1: a harness marks file chunks in the requests it sends to a dotLLM
server started with `--chunk-cache`, and manages the server's chunk store. Source: `src/DotLLM.ChunkCache.Client/`; tests:
`tests/DotLLM.ChunkCache.Client.Tests/`.

## Licence and the boundary that keeps it honest

- **MIT**, unlike the rest of the repository (GPL-3.0-only), so harnesses under any licence can link it. The terms are in
  `src/DotLLM.ChunkCache.Client/LICENSE` (packed into the nupkg), every source file there carries
  `// SPDX-License-Identifier: MIT`, and the package sets `PackageLicenseExpression` MIT. The copyright line follows the
  repository's convention (`Directory.Build.props`); the holder for this package, the inherited `<Authors>` and the engine
  `icon.png` are open questions for the maintainer.
- **Pure HTTP plus DTOs.** No `ProjectReference` to any `DotLLM.*` project, and no engine assembly at run time. The wire
  DTOs are the client's own copies; they are never shared with the server. `Package_HasNoDotLlmEngineReference` fails the
  build's tests if either rule breaks.
- Targets `net8.0` and `net10.0`, with `IsAotCompatible` (trim and AOT analyzers on; warnings are errors under `src/`).
- Packed with the other libraries; not pushed to nuget.org. See [PACKAGING.md](PACKAGING.md): it does appear as an asset on
  every dev prerelease.

## What it provides

| type | role |
|---|---|
| `ChunkSpan`, `ChunkSourceInfo`, `IChunkSource`, `ChunkedText`, `LineRange`, `IInvalidationSignal`, `ChunkInvalidation`, `InvalidationKind`, `IRecomputePolicy`, `ChunkRequestContext`, `ChunkRequestOptions`, `IChunkCacheClient` | the harness-facing abstractions. Offsets in `ChunkSpan` are UTF-16 chars (.NET strings); the wire uses UTF-8 byte offsets |
| `ExactPolicy` | the default `IRecomputePolicy`: always `exact` / `prefill` / `summary`, whatever the context. Nothing lossy without an explicit opt-in, and opting in is a different policy |
| `ChunkHash.Of(text)` | `"sha256:"` + lowercase hex of the UTF-8 bytes of the **decoded** text. No normalisation: CRLF vs LF, a BOM and trailing whitespace all count, so a CRLF text misses an LF-indexed entry |
| `ChunkCacheHttpClient` | `IChunkCacheClient` over the server (below) |
| `ChunkCacheProbe.ProbeAsync` / `ChunkCacheProbeResult` | the fail-closed probe: never throws for a server or transport failure; `IsCapable` false means strip annotations |
| `IChunkIndexQueries`, `CliChunkIndexQueries`, `IProcessRunner`, `ProcessRunner` | `select` / `status`, which read a local index and have no server endpoint: runs `chunkcache select\|status --json ...` and parses stdout |
| `ChunkCacheCapabilities`, `SelectRequest/Result`, `PrewarmRequest/Result`, `StatusRequest/Result`, `EvictRequest/Result`, `ExplainResult` | control-plane DTOs, snake_case on the wire |
| `ChunkCacheJsonContext` | source-generated JSON metadata for those DTOs |

## Server calls

| call | request | notes |
|---|---|---|
| `GetCapabilitiesAsync` | `GET /v1/chunk-cache` | `null` on 404, and on a 200 body that is not JSON or whose `versions` lacks `1` (absent). 401 / 403 / 5xx throw `ChunkCacheRequestException` with the status, for diagnostics |
| `ProbeAsync` (extension) | same | **contract ruling 16: any exception or non-2xx from the probe (401, 403, 5xx, timeout, 404) means "strip annotations and send a plain request".** That is the caller's rule; `ProbeAsync` applies it (only the caller's own cancellation propagates). Install the annotating pipeline from `ProbeAsync`, never from a probe whose exception was ignored |
| `PrewarmAsync` | `POST /v1/chunk-cache/prewarm` `{"texts":[{id,text,source?:{path}}],"max_tokens"?}` | **text only**. A `select` input is resolved locally with `include_text`; its hashes are never sent (a hash-only API would be a membership oracle). A selected chunk without text throws before anything is sent. `source` is `{path}` only and **advisory** (ruling 14): the server applies its deny-list when present but cannot rely on it. Nothing to send: no request, all-zero result |
| `EvictAsync` | `POST /v1/chunk-cache/evict` `{mode,path?,namespace?,dry_run}` | `mode` and `dry_run` are always sent resolved: unset means `orphans` and `dry_run: true` |
| `ExplainAsync` | `POST /v1/chunk-cache/explain` `{"request": <full body>}` | shape pinned by contract ruling 15 (`chunkcache select --server` uses it too); the caller's `JsonObject` is copied, not consumed |

- The base URL may end in `/v1` or not; the client never doubles it.
- The API key, when given and non-empty, goes in `X-API-Key` on every call. User info in the base URL is dropped.
- A 2xx body that is not the expected JSON also throws `ChunkCacheRequestException`.
- Tool-level `store` / `server` inputs (spec 7.2) are for the CLI and MCP layers and are not sent.
- Errors use dotLLM's envelope; `ChunkCacheRequestException` carries the status, `error.type` and `error.code` (for example
  `503` / `api_error` / `chunk_store_unavailable`).

## `chunkcache` CLI mapping (select / status)

`store` → `--store`; `path` → `--solution` for `.sln` / `.slnx` / `.csproj`, else `--directory`; `chunker` → `--chunker`;
`paths` → one `--path` each; `exclude` → one `--exclude` each; `symbol` → `--symbol`; `hops` → `--hops`; `changed_since` →
`--since`; `token_budget` → `--budget`; `include_text` → `--include-text`; `order` → `--order`; `list_missing` →
`--list-missing`; `server` → `--server`. Every call adds `--json`. Unset inputs add nothing. A non-zero exit throws
`ChunkCacheCliException` with the exit code and stderr, even if stdout holds JSON.

- **Executable:** a bare name is resolved against the fully qualified `PATH` entries only, to an absolute path; relative
  entries (`.`) and the working directory are never searched (a harness's working directory is usually an untrusted
  repository). A relative path with a directory part is rejected. `PathVariable` overrides `PATH`.
- **Timeout:** each call is cancelled after `Timeout` (default `CliChunkIndexQueries.DefaultTimeout`, 5 minutes;
  `Timeout.InfiniteTimeSpan` disables it) and reported as `ChunkCacheCliException` with an inner `TimeoutException`. Only the
  process tree the call started is killed.
- **List values:** today's CLI splits list options on commas and trims entries, so a `paths` / `exclude` value with a comma
  (such as `src/**/*.{cs,razor}`) or leading / trailing whitespace is rejected with `ArgumentException` until the CLI gets a
  non-splitting repeatable option (ruling 18, Task 17).
- **What the CLI must provide (ruling 18, Task 17):** UTF-8 JSON on stdout whatever the console code page (the client reads
  UTF-8); the API key from `CHUNKCACHE_API_KEY`, never argv; `--server`; select / status without `--model`; a non-splitting
  list option.

## Hash details

`ChunkHash.Of` hashes the UTF-8 encoding of the decoded string with no normalisation. A lone surrogate encodes as U+FFFD
(`EF BF BD`), the default `Encoding.UTF8` replacement; contract ruling 17 requires the server to do the same. One code path
serves both target frameworks.

## JSON DTO rule

All DTOs are records with `{ get; init; }` properties and **no initializers** (CLAUDE.md, #462). Optional inputs are
nullable and resolved by `[JsonIgnore]` `Effective*` accessors (`EvictRequest.EffectiveDryRun` is `true` when unset).
`ChunkCacheJsonContextDefaultsTests` guards both contexts (`ChunkCacheJsonContext` and the internal
`ChunkCacheHttpJsonContext`).

## Contract fixtures

`tests/DotLLM.ChunkCache.Client.Tests/Fixtures/v1/` holds byte-exact copies of the wire-contract fixture set
(`fixtures.sha256` lists them; `.gitattributes` disables line-ending conversion). `ContractFixtureTests` checks every file
against the list and pins the list's own SHA-256, so a fixture changed together with its list entry still fails. To take a
new contract revision, re-copy the whole set and update the pinned hash in the same change.
