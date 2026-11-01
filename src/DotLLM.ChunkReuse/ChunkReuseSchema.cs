namespace DotLLM.ChunkReuse;

/// <summary>Versions of what chunk reuse persists.</summary>
public static class ChunkReuseSchema
{
    /// <summary>
    /// Version of the encoded K/V bits. It goes into <see cref="ChunkFingerprint.Compute"/> and <see cref="Store.ChunkKey.Create"/>
    /// in place of the build version, so stored chunks survive rebuilds that do not change K/V. Bump it by hand whenever a change
    /// (kernel, RoPE table, encoder, model loader) moves the K/V a chunk encodes to; the golden test
    /// <c>ChunkReuseSchemaTests.Encode_TinyModel_MatchesGoldenForCurrentKvOutputVersion</c> fails until a golden row for the new
    /// version is added.
    /// </summary>
    public const int KvOutputVersion = 1;
}
