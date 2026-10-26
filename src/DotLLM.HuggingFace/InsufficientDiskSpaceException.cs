namespace DotLLM.HuggingFace;

/// <summary>
/// A download was refused, or stopped, because the target volume lacks the space for it (#1296).
/// </summary>
public sealed class InsufficientDiskSpaceException : IOException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="path">The file that was being written.</param>
    /// <param name="requiredBytes">Bytes still needed, including the safety margin.</param>
    /// <param name="freeBytes">Bytes free on the target volume, when known.</param>
    /// <param name="inner">The I/O error that hit a full disk mid-write, if that is how it was detected.</param>
    public InsufficientDiskSpaceException(string path, long requiredBytes, long? freeBytes, Exception? inner = null)
        : base(Describe(path, requiredBytes, freeBytes, inner is not null), inner)
    {
        Path = path;
        RequiredBytes = requiredBytes;
        FreeBytes = freeBytes;
    }

    /// <summary>The file that was being written.</summary>
    public string Path { get; }

    /// <summary>Bytes still needed, including the safety margin.</summary>
    public long RequiredBytes { get; }

    /// <summary>Bytes free on the target volume, when known.</summary>
    public long? FreeBytes { get; }

    private static string Describe(string path, long required, long? free, bool midWrite)
    {
        string dir = System.IO.Path.GetDirectoryName(path) ?? path;
        string freeText = free is long f ? Gib(f) : "unknown";
        return midWrite
            ? $"The disk filled while downloading to {dir}: {Gib(required)} was needed, {freeText} is free. The partial file was removed."
            : $"Not enough disk space to download to {dir}: needs {Gib(required)} (including a safety margin), {freeText} is free.";
    }

    private static string Gib(long bytes) => FormattableString.Invariant($"{bytes / (1024.0 * 1024 * 1024):F2} GiB");
}
