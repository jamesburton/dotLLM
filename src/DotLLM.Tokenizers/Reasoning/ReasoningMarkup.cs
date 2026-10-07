namespace DotLLM.Tokenizers.Reasoning;

/// <summary>
/// The wire markup a model family uses to delimit its reasoning ("thinking") from its answer. This is a
/// property of the <i>model</i> (detected from its chat template), independent of the user-facing
/// <see cref="ReasoningFormat"/> policy (none / auto / deepseek).
/// </summary>
public enum ReasoningMarkup
{
    /// <summary><c>&lt;think&gt;…&lt;/think&gt;</c> (Qwen3.x, DeepSeek-R1, GLM, Nemotron, SmolLM3, ...). The default.</summary>
    Think,

    /// <summary>
    /// Gemma-4: <c>&lt;|channel&gt;thought\n…&lt;channel|&gt;</c>, mirroring llama.cpp's
    /// <c>common_chat_params_init_gemma4</c> (<c>thinking_start_tag</c> / <c>thinking_end_tag</c>).
    /// </summary>
    Gemma4Channel,

    /// <summary>
    /// OpenAI Harmony (gpt-oss): a sequence of channel messages
    /// <c>&lt;|channel|&gt;analysis&lt;|message|&gt;…&lt;|end|&gt;</c>, <c>&lt;|channel|&gt;final&lt;|message|&gt;…</c>,
    /// <c>&lt;|channel|&gt;commentary to=functions.NAME &lt;|constrain|&gt;json&lt;|message|&gt;{…}&lt;|call|&gt;</c>.
    /// </summary>
    Harmony,
}

/// <summary>Detection and tag helpers for <see cref="ReasoningMarkup"/>.</summary>
public static class ReasoningMarkups
{
    /// <summary>Gemma-4 opening tag (the token <c>&lt;|channel&gt;</c> plus the channel name).</summary>
    public const string Gemma4Open = "<|channel>thought";

    /// <summary>Gemma-4 closing tag.</summary>
    public const string Gemma4Close = "<channel|>";

    /// <summary>The opening tag of a single-block markup (<see cref="ReasoningMarkup.Think"/>, <see cref="ReasoningMarkup.Gemma4Channel"/>).</summary>
    public static string OpenTag(this ReasoningMarkup markup) => markup switch
    {
        ReasoningMarkup.Gemma4Channel => Gemma4Open,
        ReasoningMarkup.Harmony => HarmonySplitter.AnalysisOpen,
        _ => ReasoningFormats.OpenTag,
    };

    /// <summary>The closing tag of a single-block markup.</summary>
    public static string CloseTag(this ReasoningMarkup markup) => markup switch
    {
        ReasoningMarkup.Gemma4Channel => Gemma4Close,
        ReasoningMarkup.Harmony => HarmonySplitter.End,
        _ => ReasoningFormats.CloseTag,
    };

    /// <summary>
    /// Detects the markup from a chat template's source. Checked from the most specific token pair:
    /// Gemma-4's <c>&lt;|channel&gt;thought</c> (no trailing pipe) before Harmony's <c>&lt;|channel|&gt;</c> +
    /// <c>&lt;|message|&gt;</c>. Anything else is <see cref="ReasoningMarkup.Think"/>.
    /// </summary>
    public static ReasoningMarkup Detect(string? templateSource)
    {
        if (string.IsNullOrEmpty(templateSource))
            return ReasoningMarkup.Think;
        if (templateSource.Contains(Gemma4Open, StringComparison.Ordinal))
            return ReasoningMarkup.Gemma4Channel;
        if (templateSource.Contains("<|channel|>", StringComparison.Ordinal)
            && templateSource.Contains("<|message|>", StringComparison.Ordinal))
            return ReasoningMarkup.Harmony;
        return ReasoningMarkup.Think;
    }

    /// <summary>
    /// Stop strings that must NOT be used for a model of this markup. Harmony's <c>&lt;|end|&gt;</c> closes
    /// the <i>analysis</i> message and is followed by the final / commentary channel, so treating it as a
    /// stop string (as every other family's chat template wants) ends the turn after the thinking and
    /// before any answer or tool call (#798).
    /// </summary>
    public static bool IsStopStringForbidden(this ReasoningMarkup markup, string stop)
        => markup == ReasoningMarkup.Harmony && stop == HarmonySplitter.End;

    /// <summary>Removes the stop strings <see cref="IsStopStringForbidden"/> forbids.</summary>
    public static IReadOnlyList<string> FilterStops(this ReasoningMarkup markup, IReadOnlyList<string> stops)
    {
        if (markup != ReasoningMarkup.Harmony)
            return stops;
        var kept = new List<string>(stops.Count);
        foreach (var s in stops)
            if (!markup.IsStopStringForbidden(s))
                kept.Add(s);
        return kept.Count == stops.Count ? stops : kept;
    }
}
