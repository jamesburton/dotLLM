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

    private static string BuildMessage(string reference, IReadOnlyList<string> candidates) =>
        $"Model name '{reference}' is ambiguous: it matches {candidates.Count} local models. Use the full id of one of: "
        + string.Join(", ", candidates);
}
