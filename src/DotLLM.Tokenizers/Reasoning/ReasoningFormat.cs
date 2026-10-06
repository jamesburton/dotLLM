namespace DotLLM.Tokenizers.Reasoning;

/// <summary>
/// How the server separates a model's reasoning ("thinking") from its answer. Mirrors llama.cpp's
/// <c>--reasoning-format</c> (the subset that applies to a <c>&lt;think&gt;…&lt;/think&gt;</c> convention).
/// </summary>
public enum ReasoningFormat
{
    /// <summary>
    /// Never split. <c>message.content</c> carries the raw model output, <c>&lt;/think&gt;</c> included —
    /// the pre-#767 behaviour, kept as an escape hatch.
    /// </summary>
    None,

    /// <summary>
    /// <b>Default.</b> Split when the chat template opened a think block (the rendered prompt ends in an
    /// unclosed <c>&lt;think&gt;</c>) or the model itself starts its output with <c>&lt;think&gt;</c>.
    /// A literal <c>&lt;think&gt;</c> later in the answer is left alone.
    /// </summary>
    Auto,

    /// <summary>
    /// As <see cref="Auto"/>, but <c>&lt;think&gt;…&lt;/think&gt;</c> blocks are also recognised
    /// <i>anywhere</i> in the output (DeepSeek-style interleaved reasoning), not only at the start.
    /// </summary>
    Deepseek,
}

/// <summary>Parsing and tag helpers for <see cref="ReasoningFormat"/>.</summary>
public static class ReasoningFormats
{
    /// <summary>The opening reasoning tag.</summary>
    public const string OpenTag = "<think>";

    /// <summary>The closing reasoning tag.</summary>
    public const string CloseTag = "</think>";

    /// <summary>Parses <c>none</c>, <c>auto</c> or <c>deepseek</c> (case-insensitive).</summary>
    /// <exception cref="ArgumentException">Unknown value.</exception>
    public static ReasoningFormat Parse(string value) => TryParse(value, out var f)
        ? f
        : throw new ArgumentException(
            $"Unknown reasoning format '{value}'. Expected: none, auto, deepseek.", nameof(value));

    /// <summary>Non-throwing <see cref="Parse"/>.</summary>
    public static bool TryParse(string? value, out ReasoningFormat format)
    {
        switch (value?.Trim().ToLowerInvariant())
        {
            case "none": format = ReasoningFormat.None; return true;
            case "auto": format = ReasoningFormat.Auto; return true;
            case "deepseek": format = ReasoningFormat.Deepseek; return true;
            default: format = ReasoningFormat.Auto; return false;
        }
    }

    /// <summary>The wire/CLI spelling of <paramref name="format"/>.</summary>
    public static string ToWireString(this ReasoningFormat format) => format switch
    {
        ReasoningFormat.None => "none",
        ReasoningFormat.Deepseek => "deepseek",
        _ => "auto",
    };

    /// <summary>
    /// True when the rendered prompt ends inside an open think block: the last <c>&lt;think&gt;</c>
    /// comes after the last <c>&lt;/think&gt;</c> and only whitespace follows it. That is exactly the
    /// generation prompt of Qwen3.x-style templates (<c>&lt;think&gt;\n</c>); the whitespace-only tail
    /// keeps a literal <c>&lt;think&gt;</c> in an earlier user message from being mistaken for it.
    /// </summary>
    public static bool PromptOpensThinking(string prompt)
    {
        int open = prompt.LastIndexOf(OpenTag, StringComparison.Ordinal);
        if (open < 0)
            return false;
        int close = prompt.LastIndexOf(CloseTag, StringComparison.Ordinal);
        if (close > open)
            return false;
        return prompt.AsSpan(open + OpenTag.Length).IsWhiteSpace();
    }

    /// <summary>
    /// Creates the splitter for one generation, or <see langword="null"/> when no splitting applies
    /// (<see cref="ReasoningFormat.None"/>).
    /// </summary>
    /// <param name="format">Server reasoning format.</param>
    /// <param name="prompt">The rendered prompt the model is about to continue.</param>
    public static ReasoningSplitter? CreateSplitter(ReasoningFormat format, string prompt)
        => format == ReasoningFormat.None
            ? null
            : new ReasoningSplitter(PromptOpensThinking(prompt), detectAnywhere: format == ReasoningFormat.Deepseek);
}
