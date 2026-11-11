using System.Runtime.CompilerServices;
using DotLLM.Cli.Commands;
#if DEBUG
using DotLLM.Cli.Commands.Debug;
#endif
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Cli;

/// <summary>
/// The dotllm command tree. Lives outside <c>Program.cs</c> so tests can build the same tree and walk it
/// (every option must have a description, every command must be documented in <c>docs/CLI.md</c>).
/// </summary>
internal static class CliConfiguration
{
    /// <summary>Exit code of a usage error (unknown command or option, missing argument). Shared exit-code table: 0 ok, 1 runtime error, 2 usage, 3 strict/check failure.</summary>
    internal const int UsageExitCode = 2;

    /// <summary>Set by <see cref="ExecutionStartedInterceptor"/> once parsing succeeded and the command is about to run; tells a usage error from a runtime failure.</summary>
    private sealed class ExecutionStartedInterceptor : ICommandInterceptor
    {
        public bool Started { get; private set; }
        public void Intercept(CommandContext context, CommandSettings settings) => Started = true;
    }

    /// <summary>
    /// Prints the same text as Spectre's default handler but returns the documented exit codes: 2 for a command-line usage error,
    /// 1 for any other failure (Spectre's own default is a code that cannot be told apart). A <see cref="CommandRuntimeException"/> is a
    /// usage error only while the command line is still being parsed and bound (unknown command or option, missing argument); once the
    /// command has started running, the same exception type thrown by a command is a runtime failure (exit 1).
    /// </summary>
    private static int HandleException(Exception ex, ExecutionStartedInterceptor interceptor)
    {
        if (ex is CommandRuntimeException { Pretty: { } pretty })
            AnsiConsole.Write(pretty);
        else
            AnsiConsole.WriteException(ex, ExceptionFormats.ShortenEverything);
        return ex is CommandRuntimeException && !interceptor.Started ? UsageExitCode : 1;
    }

    /// <summary>Registers every command, branch, description and example.</summary>
    public static void Configure(IConfigurator config)
    {
        config.SetApplicationName("dotllm");

        // Spectre's default handler prints only the exception MESSAGE. For a fault like
        // "Index was outside the bounds of the array" that is close to useless — the message names no
        // type, file or frame, so diagnosing it means guessing at call sites. Opt in to the full
        // exception (type, message, stack) with DOTLLM_DEBUG_ERRORS=1.
        if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_DEBUG_ERRORS"), "1", StringComparison.Ordinal))
            config.PropagateExceptions();
        else
        {
            var started = new ExecutionStartedInterceptor();
            config.SetInterceptor(started);
            config.SetExceptionHandler((ex, _) => HandleException(ex, started));
        }
        // The assembly informational version (0.3.0-dev.N+sha), not a literal: a harness must be able to
        // tell which build it is talking to (#774).
        config.SetApplicationVersion(DotLLM.Core.BuildInfo.Version);
        // Spectre.Console.Cli's example validation uses reflection that doesn't work under Native AOT
        if (RuntimeFeature.IsDynamicCodeSupported)
            config.ValidateExamples();
        config.Settings.StrictParsing = true;

        config.AddBranch("model", model =>
        {
            model.SetDescription("Manage models: search/pull from Hugging Face, add local files, list, delete, and create named profiles. Full reference: docs/CLI.md.");

            model.AddCommand<ModelSearchCommand>("search")
                .WithDescription("Search Hugging Face for GGUF model repositories (sorted by downloads by default).")
                .WithExample("model", "search", "llama", "--limit", "5")
                .WithExample("model", "search", "qwen 7b", "--sort", "likes");

            model.AddCommand<ModelPullCommand>("pull")
                .WithDescription("Download a GGUF model from Hugging Face (resumable, with progress). Without --file, lists the repo's GGUF files and asks which one.")
                .WithExample("model", "pull", "TheBloke/Llama-2-7B-GGUF", "--file", "llama-2-7b.Q4_K_M.gguf")
                .WithExample("model", "pull", "QuantFactory/SmolLM-135M-GGUF");

            model.AddCommand<ModelListCommand>("list")
                .WithDescription("List locally available models (dotllm store, Hugging Face hub cache, ollama store) and model profiles. Same as top-level 'dotllm list'. Models loaded in a running server: 'dotllm ps'.")
                .WithExample("model", "list")
                .WithExample("model", "list", "--paths")
                .WithExample("model", "list", "--json");

            model.AddCommand<ModelInfoCommand>("info")
                .WithDescription("Show details (files, sizes, quantizations) for a Hugging Face model repository, and which are already local.")
                .WithExample("model", "info", "TheBloke/Llama-2-7B-GGUF")
                .WithExample("model", "info", "TheBloke/Llama-2-7B-GGUF", "--json");

            model.AddCommand<ModelDeleteCommand>("delete")
                .WithAlias("rm")
                .WithDescription("Delete a locally downloaded model (asks first unless --yes), or a model profile. Alias: rm.")
                .WithExample("model", "delete", "Qwen/Qwen2.5-0.5B-Instruct-GGUF")
                .WithExample("model", "rm", "QuantFactory/SmolLM-135M-GGUF", "--quant", "Q8_0", "--yes");

            model.AddCommand<ModelAddCommand>("add")
                .WithDescription("Register a local .gguf file (linked, not copied) or download one from an http(s) URL. It is then listed as local/<name>.")
                .WithExample("model", "add", "C:\\models\\my-finetune.gguf", "--name", "my-finetune");

            model.AddCommand<ModelCreateCommand>("create")
                .WithDescription("Create (or overwrite) a named model profile: a base model plus system prompt, sampling defaults, device and keep-alive. The server applies it when a request names the profile.")
                .WithExample("model", "create", "terse", "--from", "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M", "--system", "Answer in one sentence.", "--temperature", "0.2")
                .WithExample("model", "create", "fast", "--from", "terse", "--device", "vulkan", "--keep-alive", "600");

            model.AddCommand<ModelImportOllamaCommand>("import-ollama")
                .WithDescription("Import models from an existing ollama installation as profiles (blobs are used in place, nothing is copied). Store: OLLAMA_MODELS or ~/.ollama/models.")
                .WithExample("model", "import-ollama")
                .WithExample("model", "import-ollama", "llama3.2:3b", "--force");

            model.AddCommand<ModelShowCommand>("show")
                .WithDescription("Show a profile's merged settings (through its base chain) and where its model resolves.")
                .WithExample("model", "show", "terse");

            model.AddCommand<ModelCopyCommand>("cp")
                .WithDescription("Copy a profile, or alias a model reference under a new profile name.")
                .WithExample("model", "cp", "terse", "terse-hot");
        });

        config.AddCommand<ModelListCommand>("list")
            .WithDescription("List locally available models (dotllm store, Hugging Face hub cache, ollama store) and model profiles. Alias of 'model list'. Models loaded in a running server: 'dotllm ps'.")
            .WithExample("list")
            .WithExample("list", "--json");

        config.AddCommand<RunCommand>("run")
            .WithDescription("One-shot text generation on a GGUF model (greedy unless --temp > 0), with a performance summary. <model> is a path, owner/repo or owner/repo:quant, a local name, a profile or an ollama tag; missing models are downloaded.")
            .WithExample("run", "QuantFactory/SmolLM-135M-GGUF", "--prompt", "The capital of France is", "--max-tokens", "64")
            .WithExample("run", "QuantFactory/SmolLM-135M-GGUF", "-p", "Hello", "--device", "vulkan:0", "--json")
            .WithExample("run", "QuantFactory/SmolLM-135M-GGUF", "-p", "Hello", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0");

        config.AddCommand<BenchCommand>("bench")
            .WithDescription("Benchmark prefill and decode throughput on synthetic prompts (llama-bench equivalent). Prints t/s per phase; --json for machine output.")
            .WithExample("bench", "QuantFactory/SmolLM-135M-GGUF", "--device", "cpu", "-p", "512", "-n", "128", "-r", "5")
            .WithExample("bench", "QuantFactory/SmolLM-135M-GGUF", "--device", "vulkan", "--depth-list", "512,2048,8192", "--json");

        config.AddCommand<ChatCommand>("chat")
            .WithDescription("Interactive multi-turn chat in the terminal, using the model's chat template and prompt caching. Slash commands inside the REPL: see docs/CLI.md.")
            .WithExample("chat", "QuantFactory/SmolLM-135M-GGUF", "--system", "You are a helpful assistant.")
            .WithExample("chat", "QuantFactory/SmolLM-135M-GGUF", "--device", "auto", "--paged", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0");

        config.AddCommand<EmbedCommand>("embed")
            .WithDescription("Print embeddings (JSON Lines, one object per input) for text using a GGUF embedding model. Reads stdin (one input per line) when no text is given.")
            .WithExample("embed", "nomic-ai/nomic-embed-text-v1.5-GGUF", "hello world", "--dimensions", "256")
            .WithExample("embed", "Qwen/Qwen3-Embedding-0.6B-GGUF", "hello", "--device", "vulkan");

        config.AddCommand<PerplexityCommand>("perplexity")
            .WithDescription("Compute perplexity over a text corpus (llama.cpp-comparable sliding window), or per-document NLL/KL with --docs-file. See docs/PERPLEXITY.md.")
            .WithExample("perplexity", "QuantFactory/SmolLM-135M-GGUF", "--corpus", "wiki.test.raw", "--context", "512", "--stride", "256")
            .WithExample("perplexity", "QuantFactory/SmolLM-135M-GGUF", "--corpus", "wiki.test.raw", "--kv-cache", "q8_0", "--max-tokens", "8192")
            .WithExample("perplexity", "QuantFactory/SmolLM-135M-GGUF", "--docs-file", "docs.jsonl", "--docs-out", "nll.jsonl");

        config.AddCommand<PsCommand>("ps")
            .WithDescription("List the models a running dotllm server has loaded: active flag, device, size, idle time and auto-unload countdown. Server URL: --url, else $DOTLLM_URL, else http://localhost:8080. For models on disk use 'dotllm list'.")
            .WithExample("ps")
            .WithExample("ps", "--url", "http://localhost:8080")
            .WithExample("ps", "--json");

        config.AddCommand<StopCommand>("stop")
            .WithDescription("Unload one model, all models (--all) or the whole server (--server) of a running server. Needs the server to be started with --allow-model-admin.")
            .WithExample("stop", "--all")
            .WithExample("stop", "llama3.2:3b")
            .WithExample("stop", "--server");

        config.AddCommand<ServeCommand>("serve")
            .WithDescription("Launch the HTTP server: OpenAI-, Anthropic- and ollama-compatible APIs plus a built-in web chat UI. Without <model> it starts empty and loads models on request. Reference: docs/SERVER.md.")
            .WithExample("serve", "Qwen/Qwen3-0.6B-GGUF", "--port", "8080")
            .WithExample("serve", "Qwen/Qwen3-0.6B-GGUF", "--device", "vulkan", "--ctx-size", "16384", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0")
            .WithExample("serve", "--max-resident-models", "2", "--keep-alive", "600", "--allow-model-admin", "--no-browser");

#if DEBUG
        config.AddBranch("debug", debug =>
        {
            debug.SetDescription("Debug/diagnostic commands (Debug build only).");

            debug.AddCommand<DebugGgufHeaderCommand>("gguf-header")
                .WithDescription("Parse and display the GGUF file header.")
                .WithExample("debug", "gguf-header", "model.gguf");

            debug.AddCommand<DebugGgufMetadataCommand>("gguf-metadata")
                .WithDescription("List all GGUF metadata key-value pairs.")
                .WithExample("debug", "gguf-metadata", "model.gguf");

            debug.AddCommand<DebugGgufTensorsCommand>("gguf-tensors")
                .WithDescription("List all tensor descriptors from a GGUF file.")
                .WithExample("debug", "gguf-tensors", "model.gguf");

            debug.AddCommand<DebugGgufConfigCommand>("gguf-config")
                .WithDescription("Extract and display ModelConfig from GGUF metadata.")
                .WithExample("debug", "gguf-config", "model.gguf");

            debug.AddCommand<DebugForwardPassCommand>("forward-pass")
                .WithDescription("Run a single forward pass and display logit diagnostics.")
                .WithExample("debug", "forward-pass", "model.gguf", "--prompt", "Hello");

            debug.AddCommand<DebugTokenizeCommand>("tokenize")
                .WithDescription("Tokenize text and verify round-trip decode.")
                .WithExample("debug", "tokenize", "model.gguf", "--text", "Hello world");

            debug.AddCommand<DebugEmbedLookupCommand>("embed-lookup")
                .WithDescription("Inspect the embedding vector for a single token.")
                .WithExample("debug", "embed-lookup", "model.gguf", "--token-id", "1");
        });
    #endif
    }
}
