namespace DotLLM.Core.Configuration;

/// <summary>
/// A token budget for the model's think block (#1292), llama.cpp <c>--reasoning-budget</c> semantics: after
/// <see cref="Budget"/> thinking tokens the engine forces <see cref="Message"/> followed by <see cref="CloseTag"/>, and the
/// answer continues. 0 ends thinking immediately. A block that closes on its own first is not touched.
/// </summary>
/// <param name="Budget">Thinking tokens allowed (0 = close at once).</param>
/// <param name="OpenTag">The tag that opens a think block, e.g. <c>&lt;think&gt;</c>.</param>
/// <param name="CloseTag">The tag that closes it, e.g. <c>&lt;/think&gt;</c>.</param>
/// <param name="StartsInside">The prompt already opened a block, so counting starts with the first generated token.</param>
/// <param name="Message">Text forced before <see cref="CloseTag"/> when the budget runs out (llama.cpp
/// <c>--reasoning-budget-message</c>); null or empty forces the close tag alone.</param>
public sealed record ThinkingBudget(int Budget, string OpenTag, string CloseTag, bool StartsInside, string? Message);
