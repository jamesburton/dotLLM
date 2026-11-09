# `chunkcache` CLI

`chunkcache` (project `tools/DotLLM.ChunkCache`, not packed) builds and inspects the chunk-level KV store behind the
`chunk_cache` request convention (see [CHUNK_CACHE.md](CHUNK_CACHE.md) for the wire convention and
[CHUNK_REUSE.md](CHUNK_REUSE.md) for the library). It scans a solution or directory, keeps one index per chunker and model,
and either queues the missing chunks through a server (`--server`) or encodes them itself (`--offline`).

`chunkcache <verb> --help` prints every option of a verb: its purpose, value form, default, whether it repeats, the environment variable it
uses, the exit codes, the `--json` fields and examples. This page adds the model of operation, the safety rules, and examples. The option
tables below are generated from the same registry the parser and `--help` read, and a test fails when they drift or when a registered option has
no help text. The main `dotllm` CLI is documented in [CLI.md](CLI.md).

## Command index

| Verb | What it does | Writes |
|---|---|---|
| [`build`](#build) | scan a tree, update the index, encode or queue the chunks the store lacks | the index, the registry, and (with `--offline`) entries |
| [`select`](#select) | choose chunks for a context under a token budget | nothing |
| [`status`](#status) | which chunks are missing or stale, and the store's size | nothing |
| [`list`](#list) | read-only inventory of one or more stores | nothing |
| [`evict`](#evict) | evict orphans or a scope, through a server or offline | entries, with `--apply` |
| [`prewarm`](#prewarm) | queue chunk texts on a server for it to encode | nothing locally |
| [`explain`](#explain) | dry run of the server's `chunk_cache` plan for a request | nothing |

## Who writes the store

| Mode | Who encodes | Writes | Use when |
|---|---|---|---|
| `--server URL` | the server (idle-only, budget, admission) | the server, from chunk TEXT the CLI posts to `POST /v1/chunk-cache/prewarm`; the CLI never writes entries or the server ledger | a server owns the store |
| `--offline` | this CPU | entries and indexes in `--store`, holding the store's **owner lock for the whole run** | no server owns the store |
| `--index-only` (build) | nobody | the index only | you want chunk lists and status, no encode |

`build` with none of the three is a usage error (exit 2). An offline run is refused (exit 1, `store_owned`) while a server or another offline writer
owns the store. The ownership is an open file handle, released by the OS when its process exits: there is no stale lock to
delete; if nothing is running, look for a leftover `chunkcache` or `dotllm` process.

## Environment and secrets

| Variable | Used by | Meaning | Default |
|---|---|---|---|
| `CHUNKCACHE_API_KEY` | the `--server` option of `build`, `select`, `status`, `list`, `evict`, `prewarm` and `explain` | the API key for the server, sent as `X-API-Key` | unset (no header) |

- The key is read from the environment **only** (there is no key option, so it never shows in a process list or a shell history), and is replaced
  by `***` in every message the CLI prints, including the `--json` error object. Automatic redirects are off, so a redirect cannot carry the key to another
  origin. A warning is printed when the key would travel over plain `http` to a non-loopback host.
- The secrets deny-list (`.env*`, `*.pem`, `*.key`, `*.pfx`, `id_*`, `secrets.json`, `appsettings.*.Development.json`, `.git/**`) applies to the scanned tree:
  denied files are never chunked, indexed, encoded or sent (`denied_files` counts them). `prewarm --texts` refuses a `source.path` that is denied, absolute,
  has a drive, NUL or `..`.

## Option conventions

Every option's help follows one rule: `Text. Values: valid values or range with its unit. Repeatable: ... Default: X. Env: VAR.`

- Units are in the text: `--max-chunks` counts **chunks**, `--max-encode-tokens`, `--chunk-tokens`, `--budget` and `--max-tokens` count **tokens**, `--lock-timeout` is in
  **seconds**, `--show` is a number of **lines**.
- A list option (`--path`, `--exclude`, `--store` of `list`) repeats once per value (`--path a --path b`); a value with a comma is a usage error (no splitting, no
  trimming, so a glob is passed exactly as given).
- `--verbose` is allowed anywhere on the command line; it adds stack traces to error messages on stderr and changes nothing else.
- Some options are registered on a verb only so that the parser can say precisely why they do not apply there (`explain --model`, `prewarm --offline`,
  `evict --all-for-model`); their help says "usage error" and their default says `rejected`.

## Output and exit codes

- Text output goes to stdout (nothing but the JSON document under `--json`); progress and warnings go to stderr.
- `--json` prints one JSON object on stdout (snake_case). Under `--json` an error is `{"error": {"code", "message", "exit_code"}}` with a non-zero exit;
  codes include `usage`, `strict`, `store_busy`, `store_owned`, `ledger_unreadable`, `server_error`, `server_unreachable`, `symbol_not_found`, `error`, `internal`.
  Schemas of each verb's output (`build.json`, `select.json`, `status.json`, `list.json`, `evict.json`, `prewarm.json`, `explain.json`, `error.json`) live under
  `tests/DotLLM.Tests.Unit/ChunkCache/Fixtures/v1/schemas/`; `--help` lists every field.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | ok |
| `1` | error: I/O, a busy store lock, a store owned by a server or another writer, a model that cannot be loaded, a symbol not found, an unreadable index or ledger, a server error, an unreachable server |
| `2` | usage: unknown verb or option, a missing or bad value, options that contradict each other |
| `3` | strict failure: `build --strict` (skipped projects, load failures, excluded files); `status --strict` (not up to date, or the ledger is unreadable) |

## Verbs

### build

```
chunkcache build --model M.gguf --store DIR (--solution X.sln|.slnx|.csproj | --directory D) (--server URL | --offline | --index-only) [options]
```

Scans the tree, chunks it with a wire-eligible chunker (`symbol-512`; anything else needs `--lab`), updates the index and queues/encodes every chunk the store
lacks. A rerun re-chunks only changed files. `--max-chunks N` counts chunks, `--max-encode-tokens N` counts tokens. `--dry-run` lists what would be encoded and
writes nothing. The result reports `wire_ineligible_chunks` (off the allow-list, a token-id slice, or a failing round trip) and `denied_files`.

<!-- options:build -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | required | The GGUF the tokenizer and the chunk keys come from (the file the server serves). With --server it must compute the server's producer, or the build stops (exit 1). |
| `--strict-model-id` |  | off | Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built. |
| `--threads` | `N` | min(8, cores) | CPU threads the model is opened with. Only an --offline encode runs forward passes with them. Values: integer 1..1024. |
| `--store` | `DIR` | required | The store directory (entries, one index per model and chunker, the registry). Created when missing; --dry-run writes nothing. |
| `--solution` | `FILE` | none | Scan a .sln, .slnx or .csproj with Roslyn (authored C# files only). Give --solution or --directory, not both. One of the two is required. |
| `--directory` | `DIR` | none | Scan the C# files under this directory tree (secrets and build output are skipped). Give --solution or --directory, not both. |
| `--configuration` | `NAME` | Release | MSBuild configuration to evaluate --solution with. Only with --solution (an error with --directory). |
| `--chunker` | `symbol\|fixed` | symbol | How files are cut into chunks: symbol (one chunk per member) or fixed (token windows). Only symbol with 512 tokens is wire-eligible; anything else needs --lab. |
| `--chunk-tokens` | `N` | 512 | Maximum tokens per chunk. Part of the chunker name (symbol-512) and of the index file name. Values: integer 1..1048576 (tokens). |
| `--path` | `GLOB` | none | Encode only the chunks of files matching this glob (a leading '!' negates). Combine with --symbol and --since: the selection is their union. Repeatable: one value per flag, values may not contain commas. |
| `--exclude` | `GLOB` | none | Drop files matching this glob from the selection everywhere. Repeatable: one value per flag, values may not contain commas. |
| `--symbol` | `NAME` | none | Select this symbol's definition and its neighbourhood (qualified name or a unique suffix; an unknown symbol exits 1 with suggestions). |
| `--hops` | `N` | 1 | With --symbol: how many caller/callee hops around the definition to include. Values: integer 0..100. |
| `--since` | `REF` | none | Select files changed since this git ref, plus untracked files. |
| `--merge-base` |  | off | With --since: compare against the merge base of REF and HEAD instead of REF itself. |
| `--budget` | `T` | no limit | Trim the selection to at most this many tokens, strictly (lowest-ranked chunks go first). Values: integer >= 0 (tokens). |
| `--max-per-hop` | `N` | no limit | Keep only the N heaviest neighbours of each hop. Values: integer >= 0 (chunks). |
| `--server` | `URL` | none | The server owns the model identity and every K/V write: the index is named by its model.id and model.producer, and the chunks are queued as text through POST /v1/chunk-cache/prewarm. The CLI never writes entries or the ledger. Contradicts --offline. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | off | No server owns the store: encode on this CPU into --store, holding the store's owner lock for the whole run (refused with exit 1, store_owned, while a server or another offline writer owns it). Contradicts --server. |
| `--index-only` |  | off | Update the index only; nothing is encoded or queued. Contradicts --encode. |
| `--encode` |  | on with --server or --offline | Encode the missing chunks. This is what --server and --offline already do; accepted for clarity. Contradicts --index-only. |
| `--max-chunks` | `N` | no limit | Encode or queue at most this many missing chunks, in path order. Counts chunks (--max-encode-tokens counts tokens). Values: integer >= 0 (chunks). |
| `--max-encode-tokens` | `N` | no limit | Encode or queue at most this many tokens of missing chunks. Counts tokens (--max-chunks counts chunks). Values: integer >= 0 (tokens). |
| `--lab` |  | off | Allow a chunker that is not wire-eligible (only symbol-512 is), for a research build of a store that never leaves this machine. Every chunk then counts as wire-ineligible. |
| `--dry-run` |  | off | List what would be encoded and the estimated store size; write nothing (no index, registry or entry). |
| `--strict` |  | off | Exit 3, before anything is written, if a project was skipped, a load failure was not attributed to a project, or a file is on another volume than the tree root. |
| `--reroot` |  | off | Accept a different tree root than the index recorded and re-path every file (otherwise a changed root is refused). |
| `--lock-timeout` | `SECONDS` | 30 | How long to wait for the index lock and for each short store-lock scope before failing with store_busy (exit 1). Values: number >= 0 (seconds, fractions allowed). |
| `--show` | `N` | 50 | How many would-encode lines --dry-run prints in text mode. Values: integer >= 0 (lines). |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:build -->

```sh
# Queue everything through the server that serves M.gguf
CHUNKCACHE_API_KEY=... chunkcache build --model M.gguf --store ./store --solution My.sln --server http://localhost:8080 --json

# No server: encode on this CPU (takes the owner lock for the whole run), at most 200000 tokens, on 4 threads
chunkcache build --model M.gguf --store ./store --directory ./src --offline --max-encode-tokens 200000 --threads 4

# Plan only: index the tree, list what would be encoded, write nothing
chunkcache build --model M.gguf --store ./store --directory ./src --index-only --dry-run --show 20

# Encode only a symbol's neighbourhood (two hops), trimmed to 8000 tokens
chunkcache build --model M.gguf --store ./store --directory ./src --offline --symbol Greeter --hops 2 --budget 8000

# CI: fail (exit 3) before writing anything if a project was skipped
chunkcache build --model M.gguf --store ./store --solution My.sln --configuration Debug --server http://localhost:8080 --strict
```

With `--server` the index is named by the server's `model.id` and the first 8 digits of its `model.producer`; the local GGUF must compute the same producer
or the build stops (exit 1).

### select

```
chunkcache select --model M.gguf (--store DIR | --solution X | --directory D) [criteria] [--budget T] [--server URL]
```

Chooses chunks for a context (the union of `--path`, `--symbol --hops`, `--since [--merge-base]`), ranked definition > body > path > changed > neighbour and trimmed to
`--budget`. With `--server`, each chunk's `stored` comes from one `explain` request per batch (text only).

<!-- options:select -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | required | The GGUF the tokenizer and the chunk keys come from (the file the server serves). Only its tokenizer is used. |
| `--strict-model-id` |  | off | Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built. |
| `--threads` | `N` | min(8, cores) | CPU threads the model is opened with. select runs no forward passes. Values: integer 1..1024. |
| `--store` | `DIR` | none (then give --solution or --directory) | A store whose index records the source to scan and whose entries tell which selected chunks are stored. Never written or created. |
| `--solution` | `FILE` | none | Scan a .sln, .slnx or .csproj with Roslyn (authored C# files only). Give --solution or --directory, not both. Required unless --store names the source. |
| `--directory` | `DIR` | none | Scan the C# files under this directory tree (secrets and build output are skipped). Give --solution or --directory, not both. |
| `--configuration` | `NAME` | Release | MSBuild configuration to evaluate --solution with. Only with --solution (an error with --directory). |
| `--chunker` | `symbol\|fixed` | symbol | How files are cut into chunks: symbol (one chunk per member) or fixed (token windows). The chunker must match the index you compare with. |
| `--chunk-tokens` | `N` | 512 | Maximum tokens per chunk. Part of the chunker name (symbol-512) and of the index file name. Values: integer 1..1048576 (tokens). |
| `--path` | `GLOB` | none | Select the chunks of files matching this glob (a leading '!' negates). Combine with --symbol and --since: the selection is their union. Repeatable: one value per flag, values may not contain commas. |
| `--exclude` | `GLOB` | none | Drop files matching this glob from the selection everywhere. Repeatable: one value per flag, values may not contain commas. |
| `--symbol` | `NAME` | none | Select this symbol's definition and its neighbourhood (qualified name or a unique suffix; an unknown symbol exits 1 with suggestions). |
| `--hops` | `N` | 1 | With --symbol: how many caller/callee hops around the definition to include. Values: integer 0..100. |
| `--since` | `REF` | none | Select files changed since this git ref, plus untracked files. |
| `--merge-base` |  | off | With --since: compare against the merge base of REF and HEAD instead of REF itself. |
| `--budget` | `T` | no limit | Trim the selection to at most this many tokens, strictly (lowest-ranked chunks go first). Values: integer >= 0 (tokens). |
| `--max-per-hop` | `N` | no limit | Keep only the N heaviest neighbours of each hop. Values: integer >= 0 (chunks). |
| `--order` | `document\|rank` | document | Output order: document (file and line order) or rank (selection priority). |
| `--include-text` |  | off | Include each chunk's text in the result (large). |
| `--server` | `URL` | none | `stored` comes from the server: one POST /v1/chunk-cache/explain per batch of chunk texts (text only), instead of from --store. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | off | Accepted and ignored: select never writes, so there is nothing to take an owner lock for. |
| `--show` | `N` | 40 | How many chunk lines text mode prints. Values: integer >= 0 (lines). |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:select -->

```sh
chunkcache select --model M.gguf --store ./store --symbol Greeter --hops 1 --budget 4000 --json

# Files changed since the merge base with origin/main, as a 6000-token context, in rank order
chunkcache select --model M.gguf --directory ./src --since origin/main --merge-base --budget 6000 --order rank

# A glob with an exclusion; include each chunk's text
chunkcache select --model M.gguf --directory ./src --path src/Calc.cs --path "!src/Generated/**" --include-text
```

### status

```
chunkcache status --model M.gguf --store DIR [--server URL] [--no-scan] [--strict] [--list-missing]
```

Lists chunks missing from the store and, after rescanning, stale/new/deleted files. `--strict` exits 3 when not up to date (CI). With `--server` it also reports the
server's own entries, bytes and `budget_bytes`. Read-only: no lock, nothing written.

<!-- options:status -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | required | The GGUF the tokenizer and the chunk keys come from (the file the server serves). Names which index of the store to read. |
| `--strict-model-id` |  | off | Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built. |
| `--threads` | `N` | min(8, cores) | CPU threads the model is opened with. status runs no forward passes. Values: integer 1..1024. |
| `--store` | `DIR` | required | The store to check; it must exist (a missing one is exit 1). Never written or created. |
| `--solution` | `FILE` | none | Scan a .sln, .slnx or .csproj with Roslyn (authored C# files only). Give --solution or --directory, not both. Default: the source the index recorded. |
| `--directory` | `DIR` | none | Scan the C# files under this directory tree (secrets and build output are skipped). Give --solution or --directory, not both. |
| `--configuration` | `NAME` | Release | MSBuild configuration to evaluate --solution with. Only with --solution (an error with --directory). |
| `--chunker` | `symbol\|fixed` | symbol | How files are cut into chunks: symbol (one chunk per member) or fixed (token windows). Names which index of the store to read (symbol-512 by default). |
| `--chunk-tokens` | `N` | 512 | Maximum tokens per chunk. Part of the chunker name (symbol-512) and of the index file name. Values: integer 1..1048576 (tokens). |
| `--no-scan` |  | off | Check the store against the index only; edits made since the last build are not seen. Takes no source. |
| `--strict` |  | off | Exit 3 when the store is not up to date for the tree (for CI), or when the server ledger is unreadable. |
| `--list-missing` |  | off (the JSON omits `missing`) | Include every missing chunk in the JSON (text mode always lists up to --show). |
| `--server` | `URL` | none | Report the server's own figures (entries, bytes, budget_bytes) and name the index by its model identity. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | off | Accepted and ignored: status never writes. |
| `--show` | `N` | 40 | How many missing-chunk lines text mode prints. Values: integer >= 0 (lines). |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:status -->

```sh
chunkcache status --model M.gguf --store ./store

# CI gate: exit 3 if the store lacks chunks or a file changed; list every missing chunk in the JSON
chunkcache status --model M.gguf --store ./store --strict --list-missing --json

# The store against the index only (no rescan), for an index built with another chunker
chunkcache status --model M.gguf --store ./store --no-scan --chunker fixed --chunk-tokens 64
```

### list

```
chunkcache list --store DIR [--store DIR ...] [--server URL] [--json]
```

A read-only inventory. For each store it reports the entries and their bytes, leftover temp files, whether another process holds the store's **owner lock**, the
server ledger's state, and every index (one per chunker, model and producer) with its model, source, files, chunks and tokens, and how many of its distinct chunk keys the
store holds. It needs no model: indexes are read from their files, so the counts are the index's own (use `status` to rescan the tree). With `--server` it adds the
server's own figures (model id, producer, entries, bytes, budget) from `GET /v1/chunk-cache`.

`list` never takes the owner lock or the store lock, never writes the registry, the ledger or an index (a missing or corrupt registry is rebuilt in memory and reported as
`registry_note`, not saved), and never creates a directory: a `--store` that does not exist is exit 1. The owner-lock state is **advisory**: it is one probe (the lock file opened
exclusively and closed at once), it can be stale the moment it is printed, and a lock file that cannot be probed counts as held. A writer never relies on it; it holds the
lock itself for its whole run.

<!-- options:list -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--store` | `DIR` | required | A store directory to inventory (it must exist). Repeat for several; they are listed in the order given. Repeatable: one value per flag, values may not contain commas. |
| `--server` | `URL` | none | Also report the server's own figures (model id, producer, entries, bytes, budget_bytes for its namespace) from GET /v1/chunk-cache. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:list -->

```sh
chunkcache list --store ./store

# Several stores in one call, as JSON
chunkcache list --store ./store --store D:/shared/chunk-store --json

# Add what the server says about its own store
CHUNKCACHE_API_KEY=... chunkcache list --store ./store --server http://localhost:8080 --json
```

Text output (abridged; sizes and ids are illustrative):

```
store /data/store: 4312 entries, 1.82 GiB; owner lock: free (advisory: one probe, may change at once)
  server ledger: none
  symbol-512  model 9f2c4a1b7d03  producer 5be0d7c1a9e4  [index-symbol-512.9f2c4a1b7d03.5be0d7c1.default.json]
    214 files, 4312 chunks, 1180220 tokens; 4300 of 4310 keys stored, 10 missing
    source solution /src/My.sln; built 2026-11-09T14:02:11.0000000Z
```

`--json` prints (abridged):

```json
{
  "stores": [
    {
      "path": "/data/store",
      "entries": 4312, "entry_bytes": 1954000000, "temp_files": 0, "temp_bytes": 0,
      "owner_lock": "free",
      "ledger": { "state": "absent" },
      "indexes": [
        { "file": "index-symbol-512.9f2c4a1b7d03.5be0d7c1.default.json", "chunker": "symbol-512", "model_id": "9f2c...", "producer": "5be0...",
          "state": "ok", "files": 214, "chunks": 4312, "tokens": 1180220, "distinct_keys": 4310, "stored_keys": 4300, "missing_keys": 10 }
      ]
    }
  ]
}
```

### evict

```
chunkcache evict (--server URL | --offline --store DIR) [--mode orphans|path|namespace|all] [--apply] [options]
```

A dry run unless `--apply`. `orphans` (default): entries no index and no ledger reference. `path --path G` and `namespace [--namespace N]` need `--server` (their
provenance lives in the server's ledger). `--offline --mode all --model M.gguf` removes that model's entries and indexes.

Safety for an applying offline pass:

- `--store` is resolved through links and junctions first (the real root is printed on stderr) and every lock and deletion uses the real root.
- A directory that does not look like a chunk store is refused with exit 2 before anything is read or deleted, so a parent directory or an unrelated folder is never swept.
  A store has one strong mark (the store lock, owner lock, registry or server-ledger file) or both weak marks (an `index-*.json` and a two-hex-digit entry directory); one weak mark alone is not enough.
- The pass holds the store's owner lock throughout; an unreadable index or ledger blocks it; index files named by a bad registry row are refused.
- `kept_referenced` after `--apply` is what the store holds afterwards.

<!-- options:evict -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | none | With --offline --mode all: the model whose entries and indexes are removed (required there). Not allowed with --server or other modes (exit 2). |
| `--strict-model-id` |  | off | Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built. |
| `--threads` | `N` | min(8, cores) | Only used with --offline --mode all, which opens the model; not allowed otherwise (exit 2). Values: integer 1..1024. |
| `--server` | `URL` | none | The server evicts (POST /v1/chunk-cache/evict): it holds the write lock and the ledger. Give --server or --offline, not both. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | off | No server owns --store: list or delete orphans from it directly, holding the store's owner lock for the whole --apply pass (refused with exit 1, store_owned, while a server or another writer owns it). Contradicts --server. |
| `--store` | `DIR` | none | With --offline: the store to evict from (resolved through links first; a directory that does not look like a chunk store is refused with exit 2). Required there. |
| `--apply` |  | off (dry run) | Really evict. Without it nothing is deleted and the result says dry_run. |
| `--mode` | `orphans\|path\|namespace\|all` | orphans | What to evict: orphans, path (--path), namespace (--namespace) or all. |
| `--path` | `GLOB` | none | With --mode path (required there, an error otherwise): the glob of source paths to forget. |
| `--namespace` | `NAME` | the caller's namespace | With --mode namespace: the namespace to wipe. An admin key may name any; otherwise the caller's own. |
| `--all-for-model` |  | rejected | Removed. Always a usage error (exit 2) that points to --mode all --model M.gguf --offline. |
| `--lock-timeout` | `SECONDS` | 30 | How long to wait for each short store-lock scope of an applying offline pass before failing with store_busy (exit 1). Values: number >= 0 (seconds, fractions allowed). |
| `--show` | `N` | 20 | How many evicted keys text mode prints. Values: integer >= 0 (lines). |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:evict -->

```sh
chunkcache evict --offline --store ./store                 # dry run: list orphans
chunkcache evict --offline --store ./store --apply         # delete them
chunkcache evict --offline --store ./store --mode all --model M.gguf --apply   # remove one model's entries and indexes
chunkcache evict --server http://localhost:8080 --mode orphans --apply
chunkcache evict --server http://localhost:8080 --mode path --path "src/Legacy/**"   # dry run: forget a repo
```

The old `--all-for-model` is `--mode all --model M.gguf --offline`.

### prewarm

```
chunkcache prewarm --server URL (--texts FILE | --model M.gguf (--solution X | --directory D) [selection]) [--max-tokens N]
```

Queues chunk TEXT on the server (`POST /v1/chunk-cache/prewarm`); the server decides what it stores. `--texts` is a JSON array of
`{"id": "...", "text": "...", "source": {"path": "rel/path.cs"}?}`.

<!-- options:prewarm -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | required | The GGUF the tokenizer and the chunk keys come from (the file the server serves). Only needed to select chunks from a tree (not with --texts). |
| `--strict-model-id` |  | off | Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built. |
| `--threads` | `N` | min(8, cores) | CPU threads the model is opened with. A tree selection runs no forward passes. Values: integer 1..1024. |
| `--server` | `URL` | required | The server that encodes and stores. Required. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | rejected | Not used by prewarm: giving it is a usage error (exit 2), because prewarm always goes through --server. |
| `--texts` | `FILE` | none | A JSON file of chunk texts to queue (up to 64 MiB). Give --texts or a tree selection, not both. |
| `--max-tokens` | `N` | no limit | Ask the server to queue at most this many tokens in total. Values: integer >= 1 (tokens). |
| `--solution` | `FILE` | none | Scan a .sln, .slnx or .csproj with Roslyn (authored C# files only). Give --solution or --directory, not both. Required with a selection. |
| `--directory` | `DIR` | none | Scan the C# files under this directory tree (secrets and build output are skipped). Give --solution or --directory, not both. |
| `--configuration` | `NAME` | Release | MSBuild configuration to evaluate --solution with. Only with --solution (an error with --directory). |
| `--chunker` | `symbol\|fixed` | symbol | How files are cut into chunks: symbol (one chunk per member) or fixed (token windows). The server accepts symbol-512 only. |
| `--chunk-tokens` | `N` | 512 | Maximum tokens per chunk. Part of the chunker name (symbol-512) and of the index file name. Values: integer 1..1048576 (tokens). |
| `--path` | `GLOB` | none | Queue only the chunks of files matching this glob (a leading '!' negates). Combine with --symbol and --since: the selection is their union. Repeatable: one value per flag, values may not contain commas. |
| `--exclude` | `GLOB` | none | Drop files matching this glob from the selection everywhere. Repeatable: one value per flag, values may not contain commas. |
| `--symbol` | `NAME` | none | Select this symbol's definition and its neighbourhood (qualified name or a unique suffix; an unknown symbol exits 1 with suggestions). |
| `--hops` | `N` | 1 | With --symbol: how many caller/callee hops around the definition to include. Values: integer 0..100. |
| `--since` | `REF` | none | Select files changed since this git ref, plus untracked files. |
| `--merge-base` |  | off | With --since: compare against the merge base of REF and HEAD instead of REF itself. |
| `--budget` | `T` | no limit | Trim the selection to at most this many tokens, strictly (lowest-ranked chunks go first). Values: integer >= 0 (tokens). |
| `--max-per-hop` | `N` | no limit | Keep only the N heaviest neighbours of each hop. Values: integer >= 0 (chunks). |
| `--store` | `DIR` | none | Accepted and ignored: prewarm never reads or writes a store. |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:prewarm -->

```sh
CHUNKCACHE_API_KEY=... chunkcache prewarm --server http://localhost:8080 --texts texts.json --json

# Queue the chunks a selection picks, at most 100000 tokens
chunkcache prewarm --server http://localhost:8080 --model M.gguf --solution My.sln --since origin/main --max-tokens 100000
```

### explain

```
chunkcache explain --server URL --request FILE.json
```

Posts `{"request": <FILE>}` to `POST /v1/chunk-cache/explain` and prints whether the policy would apply, the reason, per-chunk status
(`prefix | hit | miss | ignored`), predicted times and warnings.

<!-- options:explain -->
| Option | Value | Default | Description |
|---|---|---|---|
| `--model` | `FILE.gguf` | rejected | Not used by explain: giving it is a usage error (exit 2), because explain only asks the server. |
| `--strict-model-id` |  | rejected | Not used by explain: giving it is a usage error (exit 2). |
| `--threads` | `N` | rejected | Not used by explain: giving it is a usage error (exit 2). |
| `--server` | `URL` | required | The server to ask. Required. Values: an absolute http:// or https:// URL; a trailing /v1 is ignored. Env: CHUNKCACHE_API_KEY. |
| `--offline` |  | rejected | Not used by explain: giving it is a usage error (exit 2). |
| `--request` | `FILE.json` | required | A JSON file holding a chat request body (up to 4 MiB; the endpoint refuses more). |
| `--show` | `N` | 40 | How many per-chunk lines text mode prints. Values: integer >= 0 (lines). |
| `--json` |  | off | Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr. |
| `--help` |  | off | Print this help and exit 0. Nothing else is read, opened or run. |
<!-- /options:explain -->

```sh
chunkcache explain --server http://localhost:8080 --request request.json
chunkcache explain --server http://localhost:8080 --request request.json --show 100 --json
```

## Not in this tool

`compare` and `recompute` (lab commands) stay on the research branch. Applied reuse in the server (`balanced`) is not part of this CLI.
