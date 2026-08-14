using System.IO;
using System.Text.RegularExpressions;

namespace DotLLM.Tokenizers.Bpe;

/// <summary>
/// Maps GGUF <c>tokenizer.ggml.pre</c> type names to compiled regex pipelines
/// used for pre-tokenization in tiktoken-style BPE encodings.
/// Pre-tokenization splits input text at word/punctuation boundaries before BPE
/// merges are applied, ensuring merges do not cross segment boundaries.
/// </summary>
/// <remarks>
/// <para>Patterns are sourced from llama.cpp's <c>llama_vocab</c> (authoritative reference).
/// Each pattern is compiled once and reused across all tokenizer instances.</para>
/// <para><b>A pre-type maps to an ordered pipeline, not a single expression.</b> llama.cpp's
/// <c>regex_exprs</c> is a list applied in sequence: each expression further splits the segments
/// produced by the previous one. Several pre-types genuinely need more than one stage — the
/// StarCoder/SmolLM family isolates every digit with <c>\p{N}</c> before applying its main
/// pattern — so collapsing a pipeline to its last expression silently mis-tokenizes.</para>
/// </remarks>
internal static class TiktokenPreTokenizer
{
    // ── GPT-2 / default ─────────────────────────────────────────────
    // Contractions, letter runs, digit runs, punctuation, trailing whitespace.
    private static readonly Regex[] Gpt2Pipeline =
    [
        new(@"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── Llama 3 / llama-bpe ─────────────────────────────────────────
    // Case-insensitive contractions, optional-punct + letters, 1-3 digit groups,
    // punctuation with trailing newlines, standalone newlines, trailing whitespace.
    private static readonly Regex[] Llama3Pipeline =
    [
        new(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── StarCoder / SmolLM family ───────────────────────────────────
    // Two stages, in order: isolate every digit, then the GPT-2 pattern WITHOUT its
    // trailing `|\s+` alternative. Shared by StarCoder, Refact, Command-R, SmolLM,
    // CodeShell, EXAONE, Minerva and Mellum2 — llama.cpp falls all eight through one
    // case block.
    private static readonly Regex[] StarCoderPipeline =
    [
        new(@"\p{N}", RegexOptions.Compiled),
        new(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)",
            RegexOptions.Compiled),
    ];

    // ── DeepSeek LLM ────────────────────────────────────────────────
    private static readonly Regex[] DeepSeekLlmPipeline =
    [
        new(@"(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── DeepSeek Coder ──────────────────────────────────────────────
    // Identifiers, multi-digit numbers, non-whitespace runs, whitespace groups.
    private static readonly Regex[] DeepSeekCoderPipeline =
    [
        new(@"[a-zA-Z_][a-zA-Z0-9_]*|\p{N}+| ?[^\s\w]+|\s+(?!\S)|\s+", RegexOptions.Compiled),
    ];

    // GPT-4o / Llama 4
    private static readonly Regex[] Gpt4oPipeline =
    [
        new(
        @"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        RegexOptions.Compiled),
    ];

    // ── Qwen 2 / Qwen 3 (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN2) ──────
    // Identical to the Llama-3 expression EXCEPT the digit alternative is a bare
    // `\p{N}` (one digit per segment) instead of `\p{N}{1,3}`, so BPE never merges
    // across digits. This is the ORIGINAL tokenizer.json pattern that llama.cpp
    // quotes verbatim in the comment above its own copy (llama-vocab.cpp, the
    // QWEN2 case); llama.cpp spells the contractions out as `'[sS]|'[tT]|…` only
    // because std::regex has no `(?i:…)` group — the two are equivalent and .NET
    // supports the original form directly.
    private static readonly Regex[] Qwen2Pipeline =
    [
        new(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── Qwen 3.5 (llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35) ────────────
    // Qwen2's expression with combining marks folded into the letter run:
    // `[\p{L}\p{M}]+` instead of `\p{L}+`, and `\p{M}` excluded from the
    // punctuation class (`[^\s\p{L}\p{M}\p{N}]+`). A decomposed "e" + U+0301
    // therefore stays ONE segment here but splits into letter + mark under
    // qwen2/llama3/gpt2 — the property the discriminating test exercises.
    // Original tokenizer.json pattern, quoted verbatim by llama.cpp above its
    // QWEN35 case.
    private static readonly Regex[] Qwen35Pipeline =
    [
        new(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    // ── Tekken (Mistral NeMo / Pixtral-12B tokenizer family; also NVIDIA
    // Nemotron-Nano-9B-v2). llama.cpp LLAMA_VOCAB_PRE_TYPE_TEKKEN. This is the
    // ORIGINAL tokenizer.json pattern (quoted verbatim in llama-vocab.cpp:408);
    // llama.cpp itself ships a lookahead-based rewrite only because std::regex
    // lacks \p{Lu}-style classes — .NET supports them natively, so the faithful
    // original is used here.
    private static readonly Regex[] TekkenPipeline =
    [
        new(@"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            RegexOptions.Compiled),
    ];

    /// <summary>
    /// Returns the ordered pre-tokenization regex pipeline for the given GGUF
    /// <c>tokenizer.ggml.pre</c> type. Mirrors llama.cpp's <c>llama_vocab</c> policy
    /// (issue #373): an <b>absent</b> type falls back to the GPT-2 default pipeline
    /// (llama.cpp logs "missing pre-tokenizer type, using: 'default'"), and an
    /// <b>unknown</b> type throws — llama.cpp's exact behavior
    /// (<c>throw std::runtime_error("unknown pre-tokenizer type: ...")</c>).
    /// </summary>
    /// <remarks>
    /// <para>Before #373 an unknown type returned <c>null</c>, which the caller treated as
    /// "no pre-tokenization at all": BPE merges then cross boundaries the model was trained
    /// to respect, producing a token stream that mostly matches the reference and silently
    /// diverges at a small number of sites — the failure mode that motivated this table
    /// (see issue #237). Loud failure is deliberate; do not soften it back to a fallback.</para>
    /// <para>Escape hatch: set <c>DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER=1</c> to accept an
    /// unknown type and use the GPT-2 default pipeline (NOT the old "none") — output quality
    /// is then explicitly at the user's own risk.</para>
    /// </remarks>
    /// <exception cref="InvalidDataException">
    /// The type is non-empty and not in the table, and the escape hatch is unset.
    /// </exception>
    internal static Regex[] GetRegexes(string? preType) => preType switch
    {
        null or "" or "default" or "gpt2" => Gpt2Pipeline,
        // llama.cpp routes all of these through LLAMA_VOCAB_PRE_TYPE_LLAMA3
        // (llama-vocab.cpp, the "llama3" case block).
        "llama3" or "llama-v3" or "llama-bpe" or "falcon3" or "falcon-h1"
            or "pixtral" or "midm-2.0" or "lfm2" or "jina-v5-nano" => Llama3Pipeline,
        // "minerva-7b" is llama.cpp's actual spelling; "minerva" is kept because
        // earlier dotLLM releases accepted it and no GGUF is known to carry it.
        "starcoder" or "refact" or "command-r" or "smollm"
            or "codeshell" or "exaone" or "minerva" or "minerva-7b"
            or "mellum2" => StarCoderPipeline,
        "deepseek-llm" => DeepSeekLlmPipeline,
        "deepseek-coder" => DeepSeekCoderPipeline,
        // llama.cpp routes all of these to LLAMA_VOCAB_PRE_TYPE_QWEN2, and gives
        // STABLELM2 / HUNYUAN / SOLAR_OPEN the same regex block by case fall-through.
        "qwen2" or "deepseek-r1-qwen" or "kormo" or "f2llmv2" or "megrez"
            or "stablelm2" or "hunyuan" or "solar-open" => Qwen2Pipeline,
        "qwen35" => Qwen35Pipeline,
        "gpt-4o" or "llama4" => Gpt4oPipeline,
        "tekken" => TekkenPipeline,
        _ => Environment.GetEnvironmentVariable("DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER") == "1"
            ? Gpt2Pipeline
            : throw new InvalidDataException(
                $"Unknown tokenizer.ggml.pre type: '{preType}'. dotLLM has no pre-tokenization " +
                "regex pipeline for it, and running without one silently mis-tokenizes (issue #373; " +
                "llama.cpp throws on this too). Either add the pipeline to TiktokenPreTokenizer " +
                "(source it from llama.cpp llama-vocab.cpp), or set " +
                "DOTLLM_ALLOW_UNKNOWN_PRETOKENIZER=1 to proceed with the GPT-2 default pipeline " +
                "at your own risk."),
    };
}
