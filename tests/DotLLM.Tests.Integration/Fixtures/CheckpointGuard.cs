using System.Threading;

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
///     probe and the open. Unconditional: always a skip.</description></item>
///   <item><description>An <see cref="InvalidDataException"/> whose message matches one
///     of <see cref="TruncationSignatures"/> (declared-offset-vs-file-length overshoot)
///     <b>AND</b> <see cref="IsPlausiblyInFlight"/> finds independent evidence the file is
///     still being written (locked, or its size changes across a short recheck window).
///     Only then is it a skip — see the correctness note below for why the signature
///     alone is not sufficient.</description></item>
/// </list>
/// <para>
/// <b>Correctness note — the bounds-overshoot signature is NOT proof of truncation.</b>
/// An earlier version of this guard treated <see cref="TruncationSignatures"/> matches as
/// an unconditional skip, on the theory that "a declared offset running past the file's
/// actual length" only happens to genuinely truncated downloads. That is false: a
/// COMPLETE file whose header is self-consistent (valid JSON, correct length prefix) but
/// whose <c>data_offsets</c>/tensor-size field was corrupted independently (bit flip,
/// disk corruption, a loader bug writing a bad offset) produces the exact same message —
/// <c>GgufFile.Open</c>/<c>SafetensorsFile.Open</c> only compare declared offsets against
/// the file's actual length; they cannot tell "download died partway" apart from "offsets
/// corrupted in a complete file" from the exception alone. Masking the latter as a skip
/// would hide a real loader/data bug, which is worse than the noisy failure issue #384 set
/// out to fix. <see cref="IsPlausiblyInFlight"/> adds the missing signal: a checkpoint
/// still being downloaded is either still locked by the writer, or its size is still
/// changing a few hundred milliseconds later; a complete-but-corrupt file is neither.
/// </para>
/// <para>
/// <b>Why the recheck is safe to run after the failure.</b> Both loaders close their file
/// handle (and, for the bounds-overshoot case specifically, never open a memory-mapped
/// view at all — the check runs before that step) before the offset comparison that
/// throws, so by the time <see cref="LoadOrSkip{T}"/> catches the exception, nothing this
/// guard opens can conflict with a concurrent writer.
/// </para>
/// </remarks>
internal static class CheckpointGuard
{
    /// <summary>How long to wait between the two size samples in <see cref="IsPlausiblyInFlight"/>.</summary>
    private static readonly TimeSpan InFlightRecheckDelay = TimeSpan.FromMilliseconds(250);

    /// <summary>
    /// Message fragments that <c>DotLLM.Models.Gguf.GgufFile.Open</c> and
    /// <c>DotLLM.Models.SafeTensors.SafetensorsFile.Open</c> throw <see cref="InvalidDataException"/>
    /// with when a declared data/header range extends past the file's actual length. This is
    /// necessary but NOT sufficient evidence of truncation — see the correctness note on the
    /// type doc comment. Keep in sync with those two loaders if their messages change.
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
    /// <param name="path">Path passed to the load call, for the skip message and the
    /// in-flight recheck.</param>
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
        catch (InvalidDataException ex) when (IsTruncationSignature(ex.Message) && IsPlausiblyInFlight(path))
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

    /// <summary>
    /// Independent evidence that <paramref name="path"/> is still being written by another
    /// process, rather than sitting complete-but-corrupt: either something else currently
    /// holds it open (denies an exclusive read), or its length changes across a short
    /// recheck window. A stable, unlocked file is treated as NOT in-flight — a genuinely
    /// dead/stalled partial download will fail loudly here, which is the honest outcome:
    /// this guard cannot distinguish a dead partial download from real corruption without a
    /// known-good-size table, and failing loud is safer than silently skipping either one.
    /// </summary>
    private static bool IsPlausiblyInFlight(string path)
    {
        if (IsLocked(path))
            return true;

        long? sizeBefore = TryGetLength(path);
        if (sizeBefore is null)
            return true; // vanished between the failed load and this check — in-flight-ish.

        Thread.Sleep(InFlightRecheckDelay);

        long? sizeAfter = TryGetLength(path);
        if (sizeAfter is null)
            return true;

        return sizeBefore != sizeAfter;
    }

    private static bool IsLocked(string path)
    {
        try
        {
            using var probe = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
            return false;
        }
        catch (IOException)
        {
            // Covers both a genuine sharing violation and FileNotFoundException (a subtype of
            // IOException) if the file vanished between the failed load and this probe —
            // either way, treat it as in-flight.
            return true;
        }
    }

    private static long? TryGetLength(string path)
    {
        try
        {
            var info = new FileInfo(path);
            return info.Exists ? info.Length : null;
        }
        catch (IOException)
        {
            return null;
        }
    }

    private static string SkipMessage(string path, string description, Exception ex)
        => $"checkpoint present but unreadable (partial download or locked): {path} — {ex.Message} ({description})";
}
