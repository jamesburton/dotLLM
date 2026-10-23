namespace DotLLM.HuggingFace;

/// <summary>
/// A model reference that names more than one local model - typically a bare quant filename such as <c>q4_k_m</c> shipped by
/// several repos. Thrown instead of guessing (#1021): the message lists each candidate's unambiguous id so the caller can retry.
/// </summary>
public sealed class AmbiguousModelReferenceException : Exception
{
    /// <summary>Creates the exception for <paramref name="reference"/> and the ids it matched.</summary>
    /// <param name="reference">The reference as the caller wrote it.</param>
    /// <param name="candidates">The ids (<see cref="ModelResolver.IdFor"/>) of every local model it matched; at least two.</param>
    public AmbiguousModelReferenceException(string reference, IReadOnlyList<string> candidates)
        : base(BuildMessage(reference, candidates))
    {
        Reference = reference;
        Candidates = candidates;
    }

    /// <summary>The reference as the caller wrote it.</summary>
    public string Reference { get; }

    /// <summary>The ids of every local model the reference matched (each one resolves uniquely).</summary>
    public IReadOnlyList<string> Candidates { get; }

    /// <summary>The error <c>code</c> an HTTP surface reports for this (status 400, type <c>invalid_request_error</c>).</summary>
    public const string ErrorCode = "ambiguous_model";

    /// <summary>
    /// True when <paramref name="message"/> is this exception's message - for callers that only see the activation error string
    /// (<c>ServerState.EnsureActiveAsync</c>) and must still pick <see cref="ErrorCode"/> over <c>model_not_found</c>.
    /// </summary>
    /// <param name="message">An error message.</param>
    /// <returns>True for an ambiguity message.</returns>
    public static bool IsMessage(string? message) =>
        message is not null && message.StartsWith("Model name '", StringComparison.Ordinal)
        && message.Contains("' is ambiguous: it matches ", StringComparison.Ordinal);

    /// <summary>The error code for an activation error message: <see cref="ErrorCode"/> or <c>model_not_found</c>.</summary>
    /// <param name="message">The activation error.</param>
    /// <returns>The code.</returns>
    public static string CodeFor(string message) => IsMessage(message) ? ErrorCode : "model_not_found";

    private static string BuildMessage(string reference, IReadOnlyList<string> candidates) =>
        $"Model name '{reference}' is ambiguous: it matches {candidates.Count} local models. Use the full id of one of: "
        + string.Join(", ", candidates);
}
