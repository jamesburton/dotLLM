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
  DTOs are the client's own copies; they are never shared with the server. The one package dependency is
  `Microsoft.Extensions.AI.Abstractions` (MIT, pinned exactly in `Directory.Packages.props`; on `net8.0` it brings
  `System.Text.Json` 10.x), for the decorator's `DelegatingChatClient` base. `Package_HasNoDotLlmEngineReference`
  fails the tests if either the reference rule or the package list breaks.
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
| `IChunkSpanRegistry`, `ChunkSpanRegistry` | part text → spans (below) |
| `ChunkAnnotatingChatClient`, `ChunkAnnotatingOptions`, `ChunkSessionFlags`, `ChunkCacheCallEventArgs` | the MEAI decorator (below) |
| `ChunkCacheBodyHandler` | the HTTP body handler (below) |
| `ChunkManifestScope`, `ChunkPart`, `ChunkManifestOutcome`, `ChunkSkipReasons` | the decorator → handler hand-off and what the handler did |
| `IChunkCapabilitySource`, `ChunkCapabilityCache`, `FixedChunkCapabilities` | the cached, fail-closed capability probe |
| `ChunkCacheReport`, `SseChunkCacheScanner` | the response's `chunk_cache` report in MEAI types; the lazy SSE reader |

## Annotating pipeline (spec 8.3)

```
harness ─► Retrying ─► ChunkAnnotatingChatClient ─► provider IChatClient ─► HttpClient
                              │  ChunkManifestScope (AsyncLocal)                 │
                              └──────────────────────────────► ChunkCacheBodyHandler ─► (other handlers) ─► dotLLM
```

- **Install** the body handler only on providers that are really dotLLM (gate on the provider kind, not on the shared
  OpenAI-compatible builder), next to any other body-rewriting handler. Put the decorator **below** any retrying client,
  so each retry rebuilds the same manifest. Share one `ChunkCapabilityCache` between them.
- **Registry.** The file-read tool registers each result it returns: `registry.Register(text, spans)` with the
  `ChunkedText` from an `IChunkSource`. Keyed by part **text**, never by tool-call id (the decorator does not see which
  tool produced a result). Bounded LRU, 4,096 texts by default; registering a text again replaces its spans, with no
  spans removes it. Spans are UTF-16 offsets; ranges are checked at registration. A text found with a suffix appended
  (a hook's additional context) still matches its registered prefix (longest wins; texts of 32+ chars).
- **Decorator** (`ChunkAnnotatingChatClient(inner, registry, policy[, options])`): reads `TextContent.Text` and string
  `FunctionResultContent.Result`, looks each up in the registry, and with at least one hit (and capabilities, when
  `options.Capabilities` is set) opens a `ChunkManifestScope` with a snapshot of those parts and the policy's options.
  - **Nothing lossy by default.** `ExactPolicy` is the default; a policy other than `exact` / `shadow` is sent as `exact`
    unless `ChunkSessionFlags.UserOptedIntoApproximation` is set for the call (`options.SessionFlags`; never read from
    `ChatOptions.AdditionalProperties`, which some bridges put on the wire).
  - **Report.** Blocking: `ChatResponse.AdditionalProperties["chunk_cache"]` (a `JsonObject`). Streaming: on the first
    update produced after the report was read (the final chunk's update, usually), else on one extra content-free update;
    `ToChatResponse` moves it to the message, and `ChunkCacheReport.TryGet(ChatResponse, …)` looks there too.
  - `CallCompleted` fires per annotated call with the options, the handler's `ChunkManifestOutcome` (spans sent,
    not located, hash mismatches, invalid, overlapping, over the limit, fallback) and the report: log it per call.
  - Stacked decorators annotate once (the outermost scope wins). Streaming re-enters the scope on every step, so a
    request made after the first update still sees it.
- **Body handler** (`ChunkCacheBodyHandler(caps | source)`):
  - **Writes** only for a POST to `…/chat/completions` or `…/messages` with a current scope, a JSON body that has no
    `chunk_cache` and no inline `chunk` (left alone: idempotent, never a conflict), capabilities with `versions` ∋ 1
    (ruling 16), and at least one surviving span. In every other case the caller's content object goes out untouched:
    same bytes, same headers.
  - **Locates** by **decoded** text: `messages[i].content` as a string is part 0 (the only v1 form on the OpenAI surface,
    ruling 19); in an array, text parts keep their array index. A registered text is found as the whole part, as its
    prefix, or inside it (a bridge that joined texts).
  - **Verifies** every span by re-hashing its text in the serialised part: a mismatch drops it (a mutated part yields no
    annotation, never a wrong one). The client does not normalise: a CRLF text misses an LF-indexed hash, which is the
    chunk source's job to avoid (spec 5.3 item 1). Spans with a lone surrogate (including a boundary inside a surrogate
    pair) are dropped, and a part whose JSON holds a lone-surrogate escape is never decoded.
  - **Writes** placement B: UTF-8 byte offsets, `start` / `end` omitted for a whole part, a span registered "to the end"
    ending where the registered text ends (so an appended suffix is never inside it), document order, a span overlapping
    an earlier one of its part dropped, at most `max_chunks_per_request`, ids cut to 256 UTF-16 chars. Options at their
    defaults are omitted (`exact`, `prefill`, `summary`, not strict, server-default recompute). The member is spliced in
    before the closing brace: every other byte of the body is unchanged.
  - **Retries.** A resend of the same request message restarts from the caller's original content (kept in
    `HttpRequestMessage.Options`), so it carries the same manifest, or none if the capability was lost meanwhile.
    Verified with a retrying handler on either side of the body handler, a retrying chat client above the decorator,
    and the real OpenAI SDK pipeline: the AsyncLocal hand-off survives `HttpClient` and awaited retry delays (spec open
    question 1, for a plain `HttpClient` stack).
  - **Errors.** Strict: a `chunk_*` error reaches the caller unchanged. Non-strict: a 400 / 503 with a `chunk_*` code
    (which a conforming server never sends) invalidates the capability answer and resends once with the caller's
    original body (`FallBackOnChunkError`, default on). Other errors pass through.
  - **Report.** Blocking: read from the buffered JSON body, which stays readable. `text/event-stream`: the body is wrapped
    in `SseChunkCacheScanner`, which holds only the current line and parses only `data:` lines containing
    `"chunk_cache"` (the OpenAI final chunk, or the Anthropic `message_delta` event); nothing is buffered. Other media
    types pass through unread.
- **Capabilities** (`ChunkCapabilityCache(client, successTtl = 5 min, failureTtl = 30 s, probeTimeout = 5 s)`): any
  probe failure (404, 401, 403, 5xx, transport error, timeout) is "not capable" for the failure TTL, so a flapping server
  turns annotations off at the first failed probe and back on at the first good one. A change between two known model
  ids probes again; one probe serves concurrent callers; a caller's cancellation cancels only its own wait.
- **Overhead:** under 5 ms for 100 spans in a 200 KB body (median of 20, `Overhead_Under5ms`).

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
