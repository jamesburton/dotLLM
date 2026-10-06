namespace DotLLM.Core.Configuration;

/// <summary>
/// Suspends stop-string matching while the model is inside a reasoning block (#767), so a stop string that
/// occurs in the model's <i>thinking</i> cannot end the turn before it has answered.
/// </summary>
/// <remarks>
/// Tracked per generated sequence from the decoded text: stop strings are ignored while "inside" a
/// <c>OpenTag</c>…<c>CloseTag</c> block, and once the block closes they only match text that follows the
/// closing tag and the whitespace after it (the template's <c>\n\n</c> separator is not answer text).
/// EOS and max-tokens are never gated.
/// </remarks>
/// <param name="OpenTag">The tag that opens a reasoning block, e.g. <c>&lt;think&gt;</c>.</param>
/// <param name="CloseTag">The tag that closes it, e.g. <c>&lt;/think&gt;</c>.</param>
/// <param name="StartsInside">The prompt already opened a block, so generation starts inside it.</param>
/// <param name="OpenOnlyAtStart">
/// The model's own <c>OpenTag</c> counts only as the very first thing it generates (reasoning-format
/// <c>auto</c>); false recognises it anywhere (<c>deepseek</c>).
/// </param>
public sealed record StopGate(string OpenTag, string CloseTag, bool StartsInside, bool OpenOnlyAtStart);
