namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Wraps a checkpoint-loading call (<c>GgufFile.Open</c>, <c>SafetensorsFile.Open</c>,
/// <c>ModelLoader.LoadFromSafetensors</c>/<c>LoadFromGguf</c>, a CUDA/Vulkan
/// architecture's <c>LoadFromGguf</c>/<c>LoadFromSafetensors</c>, etc.) so a checkpoint
/// that resolved as "present" via a <see cref="File.Exists(string)"/>-only probe (see
/// <see cref="TestFixtureResolver"/> and the many per-class resolvers that predate it)
/// but is actually a partial in-flight download or locked by another process fails the
/// test as a clean <b>skip</b> instead of a hard <see cref="IOException"/> failure
/// (issue #384).
/// </summary>
/// <remarks>
/// <para>
/// <b>What is treated as "present but unreadable" (skip):</b>
/// </para>
/// <list type="bullet">
///   <item><description>Any <see cref="IOException"/> — sharing violations (file locked
///     by a concurrent downloader), <see cref="EndOfStreamException"/> from a
///     <c>BinaryReader</c> hitting EOF mid-header, or the file vanishing between the
///     probe and the open.</description></item>
///   <item><description>An <see cref="InvalidDataException"/> whose message matches one
///     of <see cref="TruncationSignatures"/> — the small, closed set of messages
///     <c>GgufFile.Open</c> / <c>SafetensorsFile.Open</c> throw <b>only</b> when a
///     declared offset (tensor data, header length) runs past the file's actual length.
///     That is a deterministic truncation signature: a genuinely truncated file's header
///     still claims the size it was supposed to be, so the length checks in those loaders
///     fail in exactly this shape. A COMPLETE-but-corrupt file (bad JSON, wrong dtype,
///     mismatched shape) does not produce these specific messages and is intentionally
///     left uncaught below, per issue #384 task item 3: a parse error on a complete file
///     must still be a real failure, not a silent skip.</description></item>
/// </list>
/// <para>
/// <b>Documented compromise (issue #384 task item 3).</b> The issue also floats a
/// "file size changed during the test run" signal as an alternative truncation
/// detector. This helper does not implement it: a snapshot-then-recheck race adds
/// complexity without covering any failure this exception-signature approach misses —
/// any truncation that would move the file's size mid-load necessarily also violates
/// one of the bounds checks the signature list matches (the checked-in loaders validate
/// every tensor/header offset against the file length before returning). The remaining
/// gap is a genuinely <i>stalled</i> partial download (stable size, syntactically valid
/// but incomplete header/tensor table) that happens to parse as self-consistent — that
/// case is indistinguishable from real corruption without a known-good-size table per
/// fixture, and is intentionally left to fail loudly rather than risk masking a real
/// loader bug.
/// </para>
/// </remarks>
internal static class CheckpointGuard
{
    /// <summary>
    /// Message fragments that <c>DotLLM.Models.Gguf.GgufFile.Open</c> and
    /// <c>DotLLM.Models.SafeTensors.SafetensorsFile.Open</c> throw <see cref="InvalidDataException"/>
    /// with <b>only</b> when a declared data/header range extends past the file's actual
    /// length — i.e. truncation, never corruption of an otherwise-complete file. Keep in
    /// sync with those two loaders if their messages change.
    /// </summary>
    private static readonly string[] TruncationSignatures =
    [
        "extends beyond file boundary",               // GgufFile: tensor data range past EOF
        "exceed data section length",                  // SafetensorsFile: tensor data_offsets past EOF
        "unexpected EOF",                               // SafetensorsFile: header bytes truncated mid-read
        "would read past EOF",                          // SafetensorsFile: declared header length > file length
        "to contain an 8-byte header length prefix",    // SafetensorsFile: file shorter than the length prefix itself
    ];

    /// <summary>
    /// Runs <paramref name="load"/> (the checkpoint-opening/construction call — not the
    /// whole test) and converts a "present but unreadable" failure into a
    /// <see cref="Xunit.SkipException"/> instead of letting it fail the test.
    /// </summary>
    /// <param name="path">Path passed to the load call, for the skip message.</param>
    /// <param name="description">Short human-readable fixture name, for the skip message.</param>
    /// <param name="load">The load call to guard.</param>
    /// <exception cref="Xunit.SkipException">
    /// The checkpoint is present but unreadable (partial download or locked).
    /// </exception>
    public static T LoadOrSkip<T>(string path, string description, Func<T> load)
    {
        try
        {
            return load();
        }
        catch (IOException ex)
        {
            throw new Xunit.SkipException(SkipMessage(path, description, ex));
        }
        catch (InvalidDataException ex) when (IsTruncationSignature(ex.Message))
        {
            throw new Xunit.SkipException(SkipMessage(path, description, ex));
        }
    }

    private static bool IsTruncationSignature(string message)
    {
        foreach (string signature in TruncationSignatures)
        {
            if (message.Contains(signature, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    private static string SkipMessage(string path, string description, Exception ex)
        => $"checkpoint present but unreadable (partial download or locked): {path} — {ex.Message} ({description})";
}
