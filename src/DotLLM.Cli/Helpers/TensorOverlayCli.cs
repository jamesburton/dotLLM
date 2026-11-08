using DotLLM.Models.Gguf;

namespace DotLLM.Cli.Helpers;

/// <summary>
/// Shared <c>--tensor-overlay</c> / <c>--overlay-skip-base-hash</c> glue (issue #1706) for <c>run</c>, <c>perplexity</c> and <c>serve</c>.
/// Format and rules: <c>docs/QUANTIZATION.md</c> ("Tensor overlays"); the reader/validator is <see cref="TensorOverlay"/>.
/// </summary>
internal static class TensorOverlayCli
{
    /// <summary>Option name of the overlay path.</summary>
    public const string Option = "--tensor-overlay";

    /// <summary>Option name of the base-hash opt-out.</summary>
    public const string SkipHashOption = "--overlay-skip-base-hash";

    /// <summary>Help text of <see cref="Option"/>.</summary>
    public const string Help =
        "Overlay GGUF replacing named FFN tensors (blk.L.ffn_{gate,up,down}[.E].weight), e.g. with TERN_B64_F16 ternary experts. "
        + "Unknown names, shape mismatches and a base-hash mismatch are errors. CPU backend, plain transformer / MoE-GGUF models only.";

    /// <summary>Help text of <see cref="SkipHashOption"/>.</summary>
    public const string SkipHashHelp =
        "Do not verify the overlay's dotllm.overlay.base_sha256 against the base model file (hashing a multi-GB base takes seconds; the digest is cached).";

    /// <summary>
    /// Opens, validates and attaches the overlay to <paramref name="gguf"/>. Returns an error message (and disposes nothing) on failure, else null.
    /// </summary>
    public static string? TryAttach(GgufFile gguf, string baseModelPath, string? overlayPath, bool skipBaseHash)
    {
        if (overlayPath is null) return null;
        try
        {
            TensorOverlay.Attach(gguf, baseModelPath, overlayPath, verifyBaseHash: !skipBaseHash);
            return null;
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or NotSupportedException)
        {
            return $"{Option}: {ex.Message}";
        }
    }

    /// <summary>After the model is loaded: an overlay the loader did not consume is an error message, else null.</summary>
    public static string? TryVerifyApplied(GgufFile? gguf)
    {
        try { gguf?.Overlay?.VerifyAllApplied(); return null; }
        catch (NotSupportedException ex) { return ex.Message; }
    }

    /// <summary>One-line summary for the console.</summary>
    public static string Summary(GgufFile gguf, bool skippedHash)
        => $"tensor overlay: {gguf.Overlay!.Count} tensor(s) from {gguf.Overlay.Path}" + (skippedHash ? " (base hash NOT verified)" : "");
}
