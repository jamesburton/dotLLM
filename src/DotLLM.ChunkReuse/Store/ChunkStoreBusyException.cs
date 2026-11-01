namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// <see cref="FileChunkStore.Put"/> could not replace an invalid file at the entry's path because another handle holds it (on
/// Windows a mapped file cannot be replaced). Nothing was stored; the key stays a miss and a later Put can repair it.
/// </summary>
public sealed class ChunkStoreBusyException : IOException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="path">The entry path that could not be replaced.</param>
    /// <param name="inner">The rename failure.</param>
    public ChunkStoreBusyException(string path, Exception inner)
        : base($"Chunk store entry '{path}' is invalid and held open by another reader; it was not replaced.", inner)
        => Path = path;

    /// <summary>The entry path that could not be replaced.</summary>
    public string Path { get; }
}
