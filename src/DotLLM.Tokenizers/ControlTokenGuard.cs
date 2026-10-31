namespace DotLLM.Tokenizers;

/// <summary>
/// Trust model for chat prompts (#1339). A chat template renders caller-supplied text (message content, tool
/// results, tool arguments) next to its own literals into ONE string, and the tokenizer then recognises
/// special-token spellings anywhere in that string. Left alone, a user message or a tool result containing
/// <c>&lt;|im_end|&gt;</c> becomes a real control token and can close a turn, open a new one or forge a
/// tool-response frame.
/// <para>
/// The guard is in-band: <see cref="ITokenizer.MarkUntrustedSpecials"/> inserts <see cref="Marker"/> after the
/// first character of every special-token spelling found in untrusted text, so the spelling no longer matches;
/// the tokenizer's encode path removes every marker and BPE-encodes the surrounding text as plain text. Template
/// literals are never marked, so the control tokens the template itself emits still tokenize to their ids.
/// </para>
/// <para>
/// The marker is U+FDD0, a Unicode noncharacter (never legitimate in text). It is stripped on every encode, so a
/// literal U+FDD0 in any prompt (including raw <c>/v1/completions</c>) is dropped rather than byte-encoded.
/// Raw completions and the CLI <c>run</c> raw path are otherwise unchanged: they are written by the caller and
/// keep scan-everything semantics.
/// </para>
/// </summary>
public static class ControlTokenGuard
{
    /// <summary>The in-band untrusted-span marker (U+FDD0, a noncharacter).</summary>
    public const char Marker = '﷐';

    /// <summary>Whether <paramref name="text"/> carries at least one marker.</summary>
    public static bool IsMarked(string text) => text.Contains(Marker);

    /// <summary>Removes every marker from <paramref name="text"/>.</summary>
    public static string Strip(string text)
        => text.Contains(Marker) ? text.Replace(Marker.ToString(), string.Empty, StringComparison.Ordinal) : text;
}
