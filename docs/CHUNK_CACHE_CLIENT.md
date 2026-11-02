# Chunk-cache client (`DotLLM.ChunkCache.Client`)

The client half of the chunk-cache wire convention v1: a harness marks file chunks in the requests it sends to a dotLLM
server started with `--chunk-cache`, and manages the server's chunk store. Source: `src/DotLLM.ChunkCache.Client/`; tests:
`tests/DotLLM.ChunkCache.Client.Tests/`.

## Licence and the boundary that keeps it honest

- **MIT**, unlike the rest of the repository (GPL-3.0-only), via its own `PackageLicenseExpression`, so harnesses under any
  licence can link it.
- **Pure HTTP plus DTOs.** No `ProjectReference` to any `DotLLM.*` project, and no engine assembly at run time. The wire
  DTOs are the client's own copies; they are never shared with the server. `Package_HasNoDotLlmEngineReference` fails the
  build's tests if either rule breaks.
- Targets `net8.0` and `net10.0`. Packed by CI like the other libraries; not pushed to nuget.org.

## What it provides

| type | role |
|---|---|
| `ChunkSpan`, `ChunkSourceInfo`, `IChunkSource`, `ChunkedText`, `LineRange`, `IInvalidationSignal`, `ChunkInvalidation`, `InvalidationKind`, `IRecomputePolicy`, `ChunkRequestContext`, `ChunkRequestOptions`, `IChunkCacheClient` | the harness-facing abstractions. Offsets in `ChunkSpan` are UTF-16 chars (.NET strings); the wire uses UTF-8 byte offsets |
| `ExactPolicy` | the default `IRecomputePolicy`: always `exact` / `prefill` / `summary`, whatever the context. Nothing lossy without an explicit opt-in, and opting in is a different policy |
| `ChunkHash.Of(text)` | `"sha256:"` + lowercase hex of the UTF-8 bytes of the **decoded** text. No normalisation: CRLF vs LF, a BOM and trailing whitespace all count, so a CRLF text misses an LF-indexed entry |
| `ChunkCacheHttpClient` | `IChunkCacheClient` over the server (below) |
| `IChunkIndexQueries`, `CliChunkIndexQueries`, `IProcessRunner`, `ProcessRunner` | `select` / `status`, which read a local index and have no server endpoint: runs `chunkcache select\|status --json ...` and parses stdout |
| `ChunkCacheCapabilities`, `SelectRequest/Result`, `PrewarmRequest/Result`, `StatusRequest/Result`, `EvictRequest/Result`, `ExplainResult` | control-plane DTOs, snake_case on the wire |
| `ChunkCacheJsonContext` | source-generated JSON metadata for those DTOs |

## Server calls

| call | request | notes |
|---|---|---|
| `GetCapabilitiesAsync` | `GET /v1/chunk-cache` | `null` on 404, and on a 200 body that is not JSON or whose `versions` lacks `1`: the convention is absent, strip annotations. 401 / 403 / 5xx throw `ChunkCacheRequestException`: a server refusing the key is present, not absent |
| `PrewarmAsync` | `POST /v1/chunk-cache/prewarm` `{"texts":[{id,text,source?}],"max_tokens"?}` | **text only**. A `select` input is resolved locally with `include_text`; its hashes are never sent (a hash-only API would be a membership oracle). A selected chunk without text throws before anything is sent |
| `EvictAsync` | `POST /v1/chunk-cache/evict` `{mode,path?,namespace?,dry_run}` | `mode` and `dry_run` are always sent resolved: unset means `orphans` and `dry_run: true` |
| `ExplainAsync` | `POST /v1/chunk-cache/explain` `{"request": <full body>}` | the caller's `JsonObject` is copied, not consumed |

- The base URL may end in `/v1` or not; the client never doubles it.
- The API key, when given, goes in `X-API-Key` on every call.
- Tool-level `store` / `server` inputs (spec 7.2) are for the CLI and MCP layers and are not sent.
- Errors use dotLLM's envelope; `ChunkCacheRequestException` carries the status, `error.type` and `error.code` (for example
  `503` / `api_error` / `chunk_store_unavailable`).

## `chunkcache` CLI mapping (select / status)

`store` → `--store`; `path` → `--solution` for `.sln` / `.slnx` / `.csproj`, else `--directory`; `chunker` → `--chunker`;
`paths` → one `--path` each; `exclude` → one `--exclude` each; `symbol` → `--symbol`; `hops` → `--hops`; `changed_since` →
`--since`; `token_budget` → `--budget`; `include_text` → `--include-text`; `order` → `--order`; `list_missing` →
`--list-missing`; `server` → `--server`. Every call adds `--json`. Unset inputs add nothing. A non-zero exit throws
`ChunkCacheCliException` with the exit code and stderr, even if stdout holds JSON.

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
