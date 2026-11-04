# DotLLM.ChunkCache.Client

Client for dotLLM's **chunk-cache wire convention** (v1): a harness marks the file chunks it sends to a dotLLM server
started with `--chunk-cache`, and the server can recognise chunks whose K/V it already holds. **MIT-licensed** (see `LICENSE`), pure HTTP
plus DTOs, with no reference to any dotLLM engine assembly, so harnesses under any licence can use it.

What is in the package today:

| type | what it does |
|---|---|
| `ChunkHash.Of(text)` | the wire hash: `"sha256:"` + lowercase hex of the UTF-8 bytes of the decoded text (no normalisation; CRLF, BOM and trailing whitespace count) |
| `ChunkCacheProbe.ProbeAsync(client)` | the fail-closed probe: `IsCapable == false` (404, 401, 403, 5xx, timeout, transport error) means send requests unannotated |
| `ChunkCacheHttpClient` | `IChunkCacheClient` over `GET /v1/chunk-cache` (capabilities; `null` = convention absent, strip annotations), `POST /v1/chunk-cache/prewarm` (text only), `/evict` (dry run by default), `/explain` |
| `CliChunkIndexQueries` | `select` / `status` through the `chunkcache` CLI (`--json`): local index operations the server does not serve |
| `ExactPolicy` | the default `IRecomputePolicy`: always `exact`, `prefill`, `summary` |
| `ChunkCacheJsonContext` | source-generated, snake_case metadata for every wire DTO |
| `ChunkSpanRegistry` | part text → spans, registered by the file-read tool (bounded LRU, keyed by text, prefix lookup) |
| `ChunkAnnotatingChatClient` | the `Microsoft.Extensions.AI` decorator: finds registered parts in the outgoing messages and hands them to the handler; lifts the response's `chunk_cache` report into `AdditionalProperties` |
| `ChunkCacheBodyHandler` | the `DelegatingHandler` on the provider's `HttpClient` that writes the `chunk_cache` manifest into the request body and reads the report back (streaming without buffering) |
| `ChunkCapabilityCache` | the cached, fail-closed capability probe the decorator and handler share |

```csharp
var registry = new ChunkSpanRegistry();
var caps = new ChunkCapabilityCache(new ChunkCacheHttpClient(new HttpClient(), serverUri, apiKey));

// The provider's HttpClient (dotLLM providers only): the body handler next to any other body-rewriting handler.
var http = new HttpClient(new ChunkCacheBodyHandler(caps) { InnerHandler = new SocketsHttpHandler() });
IChatClient provider = /* an OpenAI-compatible IChatClient over `http` */;

// Below any retrying client, so a retry resends the same manifest.
IChatClient chat = new ChunkAnnotatingChatClient(provider, registry, ExactPolicy.Instance,
    new ChunkAnnotatingOptions { Capabilities = caps });

// The file-read tool registers what it returns (IChunkSource gives the text and spans):
ChunkedText read = await chunkSource.ReadAsync(path, window: null, ct);
registry.Register(read.Text, read.Spans);

ChatResponse response = await chat.GetResponseAsync(messages);
if (ChunkCacheReport.TryGet(response, out JsonObject? report)) { /* log policy_applied, counts, warnings */ }
```

Without capabilities (probe failed, 404, 401, timeout) nothing is annotated and every request is byte-identical to the
plain one. The full convention, contract fixtures and rulings are documented in `docs/CHUNK_CACHE_CLIENT.md` in the dotLLM
repository.
