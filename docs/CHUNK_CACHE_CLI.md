# `chunkcache` CLI

`chunkcache` (project `tools/DotLLM.ChunkCache`, not packed) builds and inspects the chunk-level KV store behind the
`chunk_cache` request convention (see [CHUNK_CACHE.md](CHUNK_CACHE.md) for the wire convention and
[CHUNK_REUSE.md](CHUNK_REUSE.md) for the library). It scans a solution or directory, keeps one index per chunker and model,
and either queues the missing chunks through a server (`--server`) or encodes them itself (`--offline`).

`chunkcache <verb> --help` prints every option of a verb; this page adds the model of operation, the safety rules and examples.

## Who writes the store

| Mode | Who encodes | Writes | Use when |
|---|---|---|---|
| `--server URL` | the server (idle-only, budget, admission) | the server, from chunk TEXT the CLI posts to `POST /v1/chunk-cache/prewarm`; the CLI never writes entries or the server ledger | a server owns the store |
| `--offline` | this CPU | entries and indexes in `--store`, holding the store's **owner lock for the whole run** | no server owns the store |
| `--index-only` (build) | nobody | the index only | you want chunk lists and status, no encode |

`build` with none of the three is a usage error (exit 2). An offline run is refused (exit 1, `store_owned`) while a server or another
offline writer owns the store. The ownership is an open file handle, released by the OS when its process exits: there is no stale lock to
delete; if nothing is running, look for a leftover `chunkcache` or `dotllm` process.

## Environment and secrets

- `CHUNKCACHE_API_KEY`: the API key for `--server`. It is read from the environment **only** (there is no key option), sent as `X-API-Key`, and replaced
  by `***` in every message the CLI prints, including the `--json` error object. Automatic redirects are off, so a redirect cannot carry the key to another
  origin. A warning is printed when the key would travel over plain `http` to a non-loopback host.
- The secrets deny-list (`.env*`, `*.pem`, `*.key`, `*.pfx`, `id_*`, `secrets.json`, `appsettings.*.Development.json`, `.git/**`) applies to the scanned tree:
  denied files are never chunked, indexed, encoded or sent (`denied_files` counts them). `prewarm --texts` refuses a `source.path` that is denied, absolute,
  has a drive, NUL or `..`.

## Output and exit codes

- Text output goes to stdout (stderr under `--json`); progress and warnings go to stderr.
- `--json` prints one JSON object on stdout (snake_case). Under `--json` an error is `{"error": {"code", "message", "exit_code"}}` with a non-zero exit;
  codes include `store_owned`, `server_error`, `server_unreachable`. Schemas of each verb's output live under
  `tests/DotLLM.Tests.Unit/ChunkCache/Fixtures/v1/schemas/`.
- Exit codes: `0` ok; `1` error (I/O, busy store lock, model load, symbol not found, unreadable index, store owned); `2` usage (unknown verb or option, missing or
  bad value); `3` strict failure (`build --strict`: skipped projects, load failures, excluded files; `status --strict`: not up to date, or the ledger is unreadable).

## Verbs

### build

```
chunkcache build --model M.gguf --store DIR (--solution X.sln|.slnx|.csproj | --directory D) (--server URL | --offline | --index-only) [options]
```

Scans the tree, chunks it with a wire-eligible chunker (`symbol-512`; anything else needs `--lab`), updates the index and queues/encodes every chunk the store
lacks. A rerun re-chunks only changed files. `--max-chunks N` counts chunks, `--max-encode-tokens N` counts tokens. `--dry-run` lists what would be encoded and
writes nothing. The result reports `wire_ineligible_chunks` (off the allow-list, a token-id slice, or a failing round trip) and `denied_files`.

```
# Queue everything through the server that serves M.gguf
CHUNKCACHE_API_KEY=... chunkcache build --model M.gguf --store ./store --solution My.sln --server http://localhost:8080 --json

# No server: encode on this CPU (takes the owner lock for the whole run)
chunkcache build --model M.gguf --store ./store --directory ./src --offline --max-encode-tokens 200000

# Plan only
chunkcache build --model M.gguf --store ./store --directory ./src --index-only --dry-run
```

With `--server` the index is named by the server's `model.id` and the first 8 digits of its `model.producer`; the local GGUF must compute the same producer
or the build stops (exit 1).

### select

```
chunkcache select --model M.gguf (--store DIR | --solution X | --directory D) [criteria] [--budget T] [--server URL]
```

Chooses chunks for a context (the union of `--path`, `--symbol --hops`, `--since [--merge-base]`), ranked definition > body > path > changed > neighbour and trimmed to
`--budget`. With `--server`, each chunk's `stored` comes from one `explain` request per batch (text only).

```
chunkcache select --model M.gguf --store ./store --symbol Greeter --hops 1 --budget 4000 --json
```

### status

```
chunkcache status --model M.gguf --store DIR [--server URL] [--no-scan] [--strict] [--list-missing]
```

Lists chunks missing from the store and, after rescanning, stale/new/deleted files. `--strict` exits 3 when not up to date (CI). With `--server` it also reports the
server's own entries, bytes and `budget_bytes`.

### evict

```
chunkcache evict (--server URL | --offline --store DIR) [--mode orphans|path|namespace|all] [--apply] [options]
```

A dry run unless `--apply`. `orphans` (default): entries no index and no ledger reference. `path --path G` and `namespace [--namespace N]` need `--server` (their
provenance lives in the server's ledger). `--offline --mode all --model M.gguf` removes that model's entries and indexes.

Safety for an applying offline pass:

- `--store` is resolved through links and junctions first (the real root is printed on stderr) and every lock and deletion uses the real root.
- A directory that does not look like a chunk store (no store lock, owner lock, registry, server ledger, `index-*.json` or two-hex-digit entry directory) is
  refused with exit 2 before anything is read or deleted, so a parent directory or an unrelated folder is never swept.
- The pass holds the store's owner lock throughout; an unreadable index or ledger blocks it; index files named by a bad registry row are refused.
- `kept_referenced` after `--apply` is what the store holds afterwards.

```
chunkcache evict --offline --store ./store                 # dry run: list orphans
chunkcache evict --offline --store ./store --apply         # delete them
chunkcache evict --server http://localhost:8080 --mode orphans --apply
```

The old `--all-for-model` is `--mode all --model M.gguf --offline`.

### prewarm

```
chunkcache prewarm --server URL (--texts FILE | --model M.gguf (--solution X | --directory D) [selection]) [--max-tokens N]
```

Queues chunk TEXT on the server (`POST /v1/chunk-cache/prewarm`); the server decides what it stores. `--texts` is a JSON array of
`{"id": "...", "text": "...", "source": {"path": "rel/path.cs"}?}`.

### explain

```
chunkcache explain --server URL --request FILE.json
```

Posts `{"request": <FILE>}` to `POST /v1/chunk-cache/explain` and prints whether the policy would apply, the reason, per-chunk status
(`prefix | hit | miss | ignored`), predicted times and warnings.

## Not in this tool

`compare` and `recompute` (lab commands) stay on the research branch. Applied reuse in the server (`balanced`) is not part of this CLI.
