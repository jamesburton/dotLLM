namespace DotLLM.ChunkCache;

/// <summary>
/// The verbs of <c>chunkcache</c>: for each, its option registry (<see cref="VerbDef.Options"/>, from which the parser's
/// <see cref="VerbDef.Spec"/> is derived) and everything its <c>--help</c> prints. The option text rule (issue 1721) is
/// <c>Text. Values: v. Default: d. Env: E.</c>; a test fails on a registered option without text, a value form or a default, on a verb
/// without an example, and on a <c>--json</c> field the help does not mention.
/// </summary>
internal static class VerbDefs
{
    private const string Key = ServerConnection.ApiKeyVariable;
    private const OptionKind Flag = OptionKind.Flag;
    private const OptionKind Single = OptionKind.Single;
    private const OptionKind List = OptionKind.List;

    private static OptionSpec O(string name, OptionKind kind, string help, string? value = null, string @default = "", string? values = null, string? env = null) =>
        new(name, kind, help, value, @default, values, env);

    // ---------------------------------------------------------------------------------------------------- shared options

    private static OptionSpec Json() => O("json", Flag,
        "Print the result as one JSON document on stdout (snake_case names, listed below) instead of text. Progress and warnings stay on stderr.", @default: "off");

    private static OptionSpec HelpOption() => O("help", Flag, "Print this help and exit 0. Nothing else is read, opened or run.", @default: "off");

    private static OptionSpec Model(string when) => O("model", Single,
        $"The GGUF the tokenizer and the chunk keys come from (the file the server serves). {when}", "FILE.gguf", "required");

    private static OptionSpec StrictModelId() => O("strict-model-id", Flag,
        "Identify the model by a SHA-256 of every byte of the file instead of its header and tensor head/tail samples. Slower; it must match how the store was built.",
        @default: "off");

    private static OptionSpec Threads(string note) => O("threads", Single, $"CPU threads the model is opened with. {note}", "N", "min(8, cores)", "integer 1..1024");

    private static OptionSpec Server(string what) => O("server", Single, what, "URL", "none", "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key);

    private static OptionSpec Show(string what, int @default) => O("show", Single, what, "N", @default.ToString(System.Globalization.CultureInfo.InvariantCulture), "integer >= 0 (lines)");

    private static OptionSpec[] SourceOptions(string required) =>
    [
        O("solution", Single, $"Scan a .sln, .slnx or .csproj with Roslyn (authored C# files only). Give --solution or --directory, not both. {required}", "FILE", "none"),
        O("directory", Single, "Scan the C# files under this directory tree (secrets and build output are skipped). Give --solution or --directory, not both.", "DIR", "none"),
        O("configuration", Single, "MSBuild configuration to evaluate --solution with. Only with --solution (an error with --directory).", "NAME", "Release"),
    ];

    private static OptionSpec[] ChunkerOptions(string note) =>
    [
        O("chunker", Single, $"How files are cut into chunks: symbol (one chunk per member) or fixed (token windows). {note}", "symbol|fixed", "symbol"),
        O("chunk-tokens", Single, "Maximum tokens per chunk. Part of the chunker name (symbol-512) and of the index file name.", "N", "512", "integer 1..1048576 (tokens)"),
    ];

    private static OptionSpec[] SelectionOptions(string effect) =>
    [
        O("path", List, $"{effect} files matching this glob (a leading '!' negates). Combine with --symbol and --since: the selection is their union.", "GLOB", "none"),
        O("exclude", List, "Drop files matching this glob from the selection everywhere.", "GLOB", "none"),
        O("symbol", Single, "Select this symbol's definition and its neighbourhood (qualified name or a unique suffix; an unknown symbol exits 1 with suggestions).", "NAME", "none"),
        O("hops", Single, "With --symbol: how many caller/callee hops around the definition to include.", "N", "1", "integer 0..100"),
        O("since", Single, "Select files changed since this git ref, plus untracked files.", "REF", "none"),
        O("merge-base", Flag, "With --since: compare against the merge base of REF and HEAD instead of REF itself.", @default: "off"),
        O("budget", Single, "Trim the selection to at most this many tokens, strictly (lowest-ranked chunks go first).", "T", "no limit", "integer >= 0 (tokens)"),
        O("max-per-hop", Single, "Keep only the N heaviest neighbours of each hop.", "N", "no limit", "integer >= 0 (chunks)"),
    ];

    // ---------------------------------------------------------------------------------------------------- build

    private static readonly VerbDef BuildDef = new("build",
        "scan a solution or directory, update the store's index, and encode or queue the chunks the store lacks",
        [
            "chunkcache build --model M.gguf --store DIR (--solution X.sln|.slnx|.csproj | --directory D) --server URL [options]",
            "chunkcache build --model M.gguf --store DIR (--solution X | --directory D) --offline [options]",
            "chunkcache build --model M.gguf --store DIR (--solution X | --directory D) --index-only [options]",
        ],
        """
        Scans the tree (files on the secrets deny-list are never read past the scan: .env*, *.pem, *.key, *.pfx, id_*, secrets.json,
        appsettings.*.Development.json, .git/**), chunks it with a wire-eligible chunker (symbol-512; anything else needs --lab), updates
        this model's and chunker's index in --store and handles every indexed chunk the store lacks. A rerun re-chunks only changed files
        and handles only what is still missing, so an interrupted build resumes. You must say who encodes: --server, --offline or
        --index-only (--dry-run needs none).
        """,
        [
            Model("With --server it must compute the server's producer, or the build stops (exit 1)."),
            StrictModelId(),
            Threads("Only an --offline encode runs forward passes with them."),
            O("store", Single, "The store directory (entries, one index per model and chunker, the registry). Created when missing; --dry-run writes nothing.", "DIR", "required"),
            .. SourceOptions("One of the two is required."),
            .. ChunkerOptions("Only symbol with 512 tokens is wire-eligible; anything else needs --lab."),
            .. SelectionOptions("Encode only the chunks of"),
            O("server", Single,
                "The server owns the model identity and every K/V write: the index is named by its model.id and model.producer, and the chunks are queued as text through "
                + "POST /v1/chunk-cache/prewarm. The CLI never writes entries or the ledger. Contradicts --offline.", "URL", "none",
                "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key),
            O("offline", Flag,
                "No server owns the store: encode on this CPU into --store, holding the store's owner lock for the whole run (refused with exit 1, store_owned, while a server or another offline writer owns it). Contradicts --server.",
                @default: "off"),
            O("index-only", Flag, "Update the index only; nothing is encoded or queued. Contradicts --encode.", @default: "off"),
            O("encode", Flag, "Encode the missing chunks. This is what --server and --offline already do; accepted for clarity. Contradicts --index-only.", @default: "on with --server or --offline"),
            O("max-chunks", Single, "Encode or queue at most this many missing chunks, in path order. Counts chunks (--max-encode-tokens counts tokens).", "N", "no limit", "integer >= 0 (chunks)"),
            O("max-encode-tokens", Single, "Encode or queue at most this many tokens of missing chunks. Counts tokens (--max-chunks counts chunks).", "N", "no limit", "integer >= 0 (tokens)"),
            O("lab", Flag, "Allow a chunker that is not wire-eligible (only symbol-512 is), for a research build of a store that never leaves this machine. Every chunk then counts as wire-ineligible.", @default: "off"),
            O("dry-run", Flag, "List what would be encoded and the estimated store size; write nothing (no index, registry or entry).", @default: "off"),
            O("strict", Flag, "Exit 3, before anything is written, if a project was skipped, a load failure was not attributed to a project, or a file is on another volume than the tree root.", @default: "off"),
            O("reroot", Flag, "Accept a different tree root than the index recorded and re-path every file (otherwise a changed root is refused).", @default: "off"),
            O("lock-timeout", Single, "How long to wait for the index lock and for each short store-lock scope before failing with store_busy (exit 1).", "SECONDS", "30", "number >= 0 (seconds, fractions allowed)"),
            Show("How many would-encode lines --dry-run prints in text mode.", 50),
            Json(),
            HelpOption(),
        ],
        """
        files, chunks, tokens                       the tree: indexed files, chunks and tokens
        wire_ineligible_chunks                      chunks off the wire allow-list, a token-id slice, or failing the text round trip
        denied_files                                files the secrets deny-list kept out
        added, removed, unchanged                   chunks added, removed or unchanged since the previous index
        encode_queued_tokens                        tokens queued or encoded by this run
        skipped_projects[]                          project, reason, message, fails_strict (a skipped project of a solution scan)
        elapsed_ms, scan_ms, index_ms, encode_ms    timings
        dry_run, strict_failed, mode                mode is prewarm (server), local (offline), index_only or dry_run
        store, root, index_path, index_load_note    where the store, the tree root and this index are
        chunker, model_id, fingerprint, server      the chunker name, the model identity, the server of a --server run
        distinct_keys, distinct_tokens              distinct chunk keys and their tokens
        already_stored, selected_for_encoding       keys the store already had; keys a selection picked
        to_encode_chunks, encoded_chunks, encoded_tokens
        store_hits, store_failures, tokens_per_second
        missing_after, missing_tokens_after         what the store still lacks afterwards
        estimated_bytes, estimated_full_store_bytes, bytes_per_token_per_layer, layers
        store_entries, store_bytes                  the store after the run
        would_encode[]                              dry run: path, start_line, end_line, tokens, key
        prewarm                                     server run: the summed prewarm answers (requested, already_stored, queued, queued_tokens, refused, refused_reasons, requests, skipped_ineligible)
        """,
        [typeof(BuildResult)],
        [
            "CHUNKCACHE_API_KEY=... chunkcache build --model M.gguf --store ./store --solution My.sln --server http://localhost:8080 --json",
            "chunkcache build --model M.gguf --store ./store --directory ./src --offline --max-encode-tokens 200000 --threads 4",
            "chunkcache build --model M.gguf --store ./store --directory ./src --index-only --dry-run --show 20",
            "chunkcache build --model M.gguf --store ./store --directory ./src --offline --symbol Greeter --hops 2 --budget 8000 --lock-timeout 60",
            "chunkcache build --model M.gguf --store ./store --solution My.sln --configuration Debug --server http://localhost:8080 --strict",
        ],
        strictMeaning: "build --strict: a project was skipped, a load failure was not attributed, or a file is on another volume than the tree root (checked before anything is written).",
        extraExitNotes: "Exit 1 includes store_owned (a server or another offline writer owns --store) and store_busy (a lock was not free within --lock-timeout).");

    // ---------------------------------------------------------------------------------------------------- select

    private static readonly VerbDef SelectDef = new("select",
        "choose chunks for a context (paths, a symbol's neighbourhood, git changes) under a token budget",
        ["chunkcache select --model M.gguf (--store DIR | --solution X | --directory D) (--path G | --symbol S | --since REF) [options]"],
        """
        Chooses chunks for a context: the union of the criteria, ranked definition > body > path > changed > neighbour and trimmed strictly
        to --budget. It needs the tokenizer only (no weights run). With --store the source defaults to the one the store's index records
        and each chunk says whether the store holds it; with --server `stored` comes from the server instead.
        """,
        [
            Model("Only its tokenizer is used."),
            StrictModelId(),
            Threads("select runs no forward passes."),
            O("store", Single, "A store whose index records the source to scan and whose entries tell which selected chunks are stored. Never written or created.", "DIR", "none (then give --solution or --directory)"),
            .. SourceOptions("Required unless --store names the source."),
            .. ChunkerOptions("The chunker must match the index you compare with."),
            .. SelectionOptions("Select the chunks of"),
            O("order", Single, "Output order: document (file and line order) or rank (selection priority).", "document|rank", "document"),
            O("include-text", Flag, "Include each chunk's text in the result (large).", @default: "off"),
            O("server", Single,
                "`stored` comes from the server: one POST /v1/chunk-cache/explain per batch of chunk texts (text only), instead of from --store.", "URL", "none",
                "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key),
            O("offline", Flag, "Accepted and ignored: select never writes, so there is nothing to take an owner lock for.", @default: "off"),
            Show("How many chunk lines text mode prints.", 40),
            Json(),
            HelpOption(),
        ],
        """
        chunks[]                          id, path, start_line, end_line, tokens, hash, reason, detail, hop, stored, text, reasons, priority, key
        selected_tokens, candidate_tokens tokens selected and tokens before trimming
        trimmed                           chunks dropped to fit --budget
        unmatched_patterns, notes         globs that matched nothing; remarks
        unmatched_changed_files           changed files that have no chunk
        symbol_definitions[]              qualified_name, kind, path, line
        order, root, chunker              the order, the tree root and the chunker used
        stored_chunks                     how many selected chunks the store (or server) holds, when asked
        elapsed_ms, server                timing; the server that answered `stored`
        """,
        [typeof(SelectResult)],
        [
            "chunkcache select --model M.gguf --store ./store --symbol Greeter --hops 1 --budget 4000 --json",
            "chunkcache select --model M.gguf --directory ./src --path src/Calc.cs --path !src/Generated/** --order rank --include-text",
            "chunkcache select --model M.gguf --directory ./src --since origin/main --merge-base --budget 6000 --show 100",
            "CHUNKCACHE_API_KEY=... chunkcache select --model M.gguf --store ./store --symbol Greeter --server http://localhost:8080",
        ],
        extraExitNotes: "Exit 1 includes a symbol that is not found (with a 'did you mean' list on stderr).");

    // ---------------------------------------------------------------------------------------------------- status

    private static readonly VerbDef StatusDef = new("status",
        "which indexed chunks the store lacks, which files changed since the last build, and the store's size",
        ["chunkcache status --model M.gguf --store DIR [--server URL] [options]"],
        """
        Lists the chunks of the store's index (this model and chunker) that the store lacks, by file and line range, and, after rescanning
        the tree the index records (or --solution/--directory), the stale, new and deleted files. Read-only: takes no lock and writes nothing.
        """,
        [
            Model("Names which index of the store to read."),
            StrictModelId(),
            Threads("status runs no forward passes."),
            O("store", Single, "The store to check; it must exist (a missing one is exit 1). Never written or created.", "DIR", "required"),
            .. SourceOptions("Default: the source the index recorded."),
            .. ChunkerOptions("Names which index of the store to read (symbol-512 by default)."),
            O("no-scan", Flag, "Check the store against the index only; edits made since the last build are not seen. Takes no source.", @default: "off"),
            O("strict", Flag, "Exit 3 when the store is not up to date for the tree (for CI), or when the server ledger is unreadable.", @default: "off"),
            O("list-missing", Flag, "Include every missing chunk in the JSON (text mode always lists up to --show).", @default: "off (the JSON omits `missing`)"),
            O("server", Single, "Report the server's own figures (entries, bytes, budget_bytes) and name the index by its model identity.", "URL", "none",
                "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key),
            O("offline", Flag, "Accepted and ignored: status never writes.", @default: "off"),
            Show("How many missing-chunk lines text mode prints.", 40),
            Json(),
            HelpOption(),
        ],
        """
        total_chunks, missing_chunks, missing_tokens   the index's chunks, those the store lacks, and their tokens
        up_to_date                                     whether nothing is missing and (when scanned) no file changed
        stale_files[], new_files[], deleted_files[]    changed, added and removed files since the last build
        missing[]                                      path, start_line, end_line, tokens (text mode, or with --list-missing)
        store_bytes, entries, budget_bytes             the store (the server's figures with --server; budget only from a server)
        temp_files, temp_bytes                         leftover temp files of a crashed writer
        scanned, index_path, index_load_note, root, chunker, server
        ledger                                         the server ledger: state (absent|ok|unreadable), rows, rows_without_entry, rows_undetermined, error
        """,
        [typeof(StatusResult)],
        [
            "chunkcache status --model M.gguf --store ./store",
            "chunkcache status --model M.gguf --store ./store --strict --list-missing --json",
            "chunkcache status --model M.gguf --store ./store --no-scan --chunk-tokens 64 --chunker fixed",
            "chunkcache status --model M.gguf --store ./store --solution My.sln --configuration Debug --show 100",
        ],
        strictMeaning: "status --strict: the store is not up to date for the tree, or the server ledger is unreadable.",
        extraExitNotes: "Exit 1 includes a store directory that does not exist and a model with no index in the store (run build first).");

    // ---------------------------------------------------------------------------------------------------- evict

    private static readonly VerbDef EvictDef = new("evict",
        "evict orphan or scoped entries, through a server or offline from a store no server owns (a dry run unless --apply)",
        [
            "chunkcache evict --server URL [--mode orphans|path|namespace|all] [--path G] [--namespace N] [--apply]",
            "chunkcache evict --offline --store DIR [--mode orphans|all --model M.gguf] [--apply]",
        ],
        """
        Default: a dry run that lists what would go. --mode orphans: entries no index and not the server ledger reference. --mode path --path G:
        forget a repo (entries first seen under the glob, their index references and sightings). --mode namespace and --mode all wipe a
        namespace or the store (across namespaces only for an admin key). path and namespace need --server: their provenance lives in the
        server's ledger, which the CLI never touches. With --offline only orphans, and --mode all --model M.gguf (that model's entries and
        indexes), exist. An applying offline pass holds the store's owner lock for its whole run and refuses a directory that does not
        look like a chunk store; an unreadable index or ledger blocks it (exit 1). The old --all-for-model is --mode all --model M.gguf --offline.
        """,
        [
            O("model", Single, "With --offline --mode all: the model whose entries and indexes are removed (required there). Not allowed with --server or other modes (exit 2).", "FILE.gguf", "none"),
            StrictModelId(),
            O("threads", Single, "Only used with --offline --mode all, which opens the model; not allowed otherwise (exit 2).", "N", "min(8, cores)", "integer 1..1024"),
            Server("The server evicts (POST /v1/chunk-cache/evict): it holds the write lock and the ledger. Give --server or --offline, not both."),
            O("offline", Flag, "No server owns --store: list or delete orphans from it directly, holding the store's owner lock for the whole --apply pass (refused with exit 1, store_owned, while a server or another writer owns it). Contradicts --server.", @default: "off"),
            O("store", Single, "With --offline: the store to evict from (resolved through links first; a directory that does not look like a chunk store is refused with exit 2). Required there.", "DIR", "none"),
            O("apply", Flag, "Really evict. Without it nothing is deleted and the result says dry_run.", @default: "off (dry run)"),
            O("mode", Single, "What to evict: orphans, path (--path), namespace (--namespace) or all.", "orphans|path|namespace|all", "orphans"),
            O("path", Single, "With --mode path (required there, an error otherwise): the glob of source paths to forget.", "GLOB", "none"),
            O("namespace", Single, "With --mode namespace: the namespace to wipe. An admin key may name any; otherwise the caller's own.", "NAME", "the caller's namespace"),
            O("all-for-model", Flag, "Removed. Always a usage error (exit 2) that points to --mode all --model M.gguf --offline.", @default: "rejected"),
            O("lock-timeout", Single, "How long to wait for each short store-lock scope of an applying offline pass before failing with store_busy (exit 1).", "SECONDS", "30", "number >= 0 (seconds, fractions allowed)"),
            Show("How many evicted keys text mode prints.", 20),
            Json(),
            HelpOption(),
        ],
        """
        evicted, freed_bytes, kept_referenced   entries evicted (or that would be), their bytes, entries kept because referenced
        dry_run, mode                           whether nothing was deleted; the mode used
        server                                  the server that evicted (server runs)
        unreadable_entries                      offline --mode all: entries whose header could not be read (kept)
        keys[], dropped_indexes[]               offline: the entry keys and the index files evicted or dropped
        temp_files_removed, store_entries_after offline --apply: leftover temp files removed; entries left
        """,
        [typeof(EvictResult)],
        [
            "chunkcache evict --offline --store ./store",
            "chunkcache evict --offline --store ./store --apply --lock-timeout 60",
            "chunkcache evict --offline --store ./store --mode all --model M.gguf --apply",
            "CHUNKCACHE_API_KEY=... chunkcache evict --server http://localhost:8080 --mode orphans --apply --json",
            "chunkcache evict --server http://localhost:8080 --mode path --path src/Legacy/** --show 50",
        ],
        extraExitNotes: "Exit 1 includes store_owned and an unreadable index or ledger (nothing is deleted).");

    // ---------------------------------------------------------------------------------------------------- prewarm

    private static readonly VerbDef PrewarmDef = new("prewarm",
        "queue chunk texts on a server for it to encode (text only)",
        ["chunkcache prewarm --server URL (--texts FILE | --model M.gguf (--solution X | --directory D) (--path G | --symbol S | --since REF)) [--max-tokens N]"],
        """
        Queues chunk TEXT on the server for an isolated encode (POST /v1/chunk-cache/prewarm). The server computes every key and writes every
        entry and decides what to store (idle-only, budget, admission); the CLI never writes entries. Input is either --texts FILE, or a tree
        and a selection (the chunks `select` would pick, wire-eligible only). --texts is a JSON array of
        {"id": "...", "text": "...", "source": {"path": "rel/path.cs"}?}; a source.path on the secrets deny-list, an absolute path or one with '..' is
        refused here before anything is sent.
        """,
        [
            Model("Only needed to select chunks from a tree (not with --texts)."),
            StrictModelId(),
            Threads("A tree selection runs no forward passes."),
            O("server", Single, "The server that encodes and stores. Required.", "URL", "required", "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key),
            O("offline", Flag, "Not used by prewarm: giving it is a usage error (exit 2), because prewarm always goes through --server.", @default: "rejected"),
            O("texts", Single, "A JSON file of chunk texts to queue (up to 64 MiB). Give --texts or a tree selection, not both.", "FILE", "none"),
            O("max-tokens", Single, "Ask the server to queue at most this many tokens in total.", "N", "no limit", "integer >= 1 (tokens)"),
            .. SourceOptions("Required with a selection."),
            .. ChunkerOptions("The server accepts symbol-512 only."),
            .. SelectionOptions("Queue only the chunks of"),
            O("store", Single, "Accepted and ignored: prewarm never reads or writes a store.", "DIR", "none"),
            Json(),
            HelpOption(),
        ],
        """
        requested, already_stored, queued   chunk texts sent, already held by the server, and queued for encoding
        queued_tokens                       tokens queued
        refused, refused_reasons[]          texts refused (here or by the server) and why (denied_path, path_outside_corpus_root, ...)
        requests, skipped_ineligible        prewarm requests made; selected chunks left out because they are not wire-eligible
        """,
        [typeof(PrewarmJson)],
        [
            "CHUNKCACHE_API_KEY=... chunkcache prewarm --server http://localhost:8080 --texts texts.json --json",
            "chunkcache prewarm --server http://localhost:8080 --model M.gguf --directory ./src --symbol Greeter --hops 1 --budget 4000",
            "chunkcache prewarm --server http://localhost:8080 --model M.gguf --solution My.sln --since origin/main --max-tokens 100000",
        ]);

    // ---------------------------------------------------------------------------------------------------- explain

    private static readonly VerbDef ExplainDef = new("explain",
        "a dry run of the server's chunk_cache plan for a chat request file",
        ["chunkcache explain --server URL --request FILE.json"],
        """
        Asks what the server would do with a chat request's chunk_cache annotations: posts {"request": <FILE>} to POST /v1/chunk-cache/explain
        and prints whether the policy would apply, the reason, the per-chunk status (prefix | hit | miss | ignored, with the reason for an
        ignored one: only for an authenticated caller's own namespace), predicted times and warnings. The file is sent as it is and nothing is
        written anywhere. It only asks the server: --model, --strict-model-id, --threads and --offline are not used and are a usage error.
        """,
        [
            O("model", Single, "Not used by explain: giving it is a usage error (exit 2), because explain only asks the server.", "FILE.gguf", "rejected"),
            O("strict-model-id", Flag, "Not used by explain: giving it is a usage error (exit 2).", @default: "rejected"),
            O("threads", Single, "Not used by explain: giving it is a usage error (exit 2).", "N", "rejected"),
            O("server", Single, "The server to ask. Required.", "URL", "required", "an absolute http:// or https:// URL; a trailing /v1 is ignored", Key),
            O("offline", Flag, "Not used by explain: giving it is a usage error (exit 2).", @default: "rejected"),
            O("request", Single, "A JSON file holding a chat request body (up to 4 MiB; the endpoint refuses more).", "FILE.json", "required"),
            Show("How many per-chunk lines text mode prints.", 40),
            Json(),
            HelpOption(),
        ],
        """
        policy_would_apply        the policy the server would apply (exact, ...), or absent
        reason                    why not, when it would not
        chunks[]                  id, status (prefix|hit|miss|ignored), why
        predicted_ms              reuse and full_prefill: predicted times in milliseconds
        warnings[]                the server's warnings
        server                    the server asked
        """,
        [typeof(ExplainJson)],
        [
            "chunkcache explain --server http://localhost:8080 --request request.json",
            "CHUNKCACHE_API_KEY=... chunkcache explain --server http://localhost:8080 --request request.json --show 100 --json",
        ]);

    // ---------------------------------------------------------------------------------------------------- list

    private static readonly VerbDef ListDef = new("list",
        "inventory of one or more stores: entries and sizes, owner-lock state, and each index's model and chunk counts",
        ["chunkcache list --store DIR [--store DIR ...] [--server URL] [--json]"],
        """
        A read-only inventory. For each store: its entries and bytes, leftover temp files, whether another process holds the store's owner lock,
        the server ledger's state, and every index (one per chunker, model and producer) with its model, source, files, chunks and tokens, and
        how many of its distinct chunk keys the store holds. It takes no lock, writes no registry, ledger or index, and never creates a
        directory: a store that does not exist is exit 1. The owner-lock state is ADVISORY: it is one probe (the lock file opened
        exclusively and closed at once) and can be stale the moment it is printed; an unreadable lock file counts as held. Indexes are read
        from their files, so a chunk count is the index's own, not a rescan of the tree (use status for that).
        """,
        [
            O("store", List, "A store directory to inventory (it must exist). Repeat for several; they are listed in the order given.", "DIR", "required"),
            Server("Also report the server's own figures (model id, producer, entries, bytes, budget_bytes for its namespace) from GET /v1/chunk-cache."),
            Json(),
            HelpOption(),
        ],
        """
        stores[]                            one per --store, in order
          path                              the store directory
          entries, entry_bytes              entry files (.ckv) and their bytes
          temp_files, temp_bytes            leftover temp files of a crashed writer
          owner_lock                        held (or not probeable) | free: advisory
          ledger                            state (absent|ok|unreadable), rows, rows_without_entry, rows_undetermined, error
          registry_note                     why the registry was rebuilt from the index files, when it was (never written back)
          invalid_registry_rows[]           registry rows refused (not a plain index file name)
          indexes[]                         file, chunker, model_id, producer, model_path, source_kind, source_path, root, updated_utc,
                                            state (ok|missing|unreadable), error, files, chunks, tokens, distinct_keys, stored_keys, missing_keys
        server                              url, model_id, producer, tokenizer_id, entries, bytes, budget_bytes (with --server)
        """,
        [typeof(ListResult)],
        [
            "chunkcache list --store ./store",
            "chunkcache list --store ./store --store D:/shared/chunk-store --json",
            "CHUNKCACHE_API_KEY=... chunkcache list --store ./store --server http://localhost:8080 --json",
        ]);

    /// <summary>Every verb, by name.</summary>
    public static readonly IReadOnlyDictionary<string, VerbDef> All = new Dictionary<string, VerbDef>(StringComparer.Ordinal)
    {
        ["build"] = BuildDef,
        ["select"] = SelectDef,
        ["status"] = StatusDef,
        ["list"] = ListDef,
        ["evict"] = EvictDef,
        ["prewarm"] = PrewarmDef,
        ["explain"] = ExplainDef,
    };
}
