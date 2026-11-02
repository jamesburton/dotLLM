# DotLLM.ChunkCache.Client

Client for dotLLM's **chunk-cache wire convention** (v1): a harness marks the file chunks it sends to a dotLLM server
started with `--chunk-cache`, and the server can recognise chunks whose K/V it already holds. **MIT-licensed**, pure HTTP
plus DTOs, with no reference to any dotLLM engine assembly, so harnesses under any licence can use it.

What is in the package today:

| type | what it does |
|---|---|
| `ChunkHash.Of(text)` | the wire hash: `"sha256:"` + lowercase hex of the UTF-8 bytes of the decoded text (no normalisation; CRLF, BOM and trailing whitespace count) |
| `ChunkCacheHttpClient` | `IChunkCacheClient` over `GET /v1/chunk-cache` (capabilities; `null` = convention absent, strip annotations), `POST /v1/chunk-cache/prewarm` (text only), `/evict` (dry run by default), `/explain` |
| `CliChunkIndexQueries` | `select` / `status` through the `chunkcache` CLI (`--json`): local index operations the server does not serve |
| `ExactPolicy` | the default `IRecomputePolicy`: always `exact`, `prefill`, `summary` |
| `ChunkCacheJsonContext` | source-generated, snake_case metadata for every wire DTO |

```csharp
using var http = new HttpClient();
var client = new ChunkCacheHttpClient(http, new Uri("http://localhost:8080"), apiKey: null);
ChunkCacheCapabilities? caps = await client.GetCapabilitiesAsync(ct);
if (caps is null) { /* not a chunk-cache server: send requests unannotated */ }
```

The MEAI decorator and the HTTP body handler that write the `chunk_cache` manifest into requests come next. The full
convention, contract fixtures and rulings are documented in `docs/CHUNK_CACHE_CLIENT.md` in the dotLLM repository.
