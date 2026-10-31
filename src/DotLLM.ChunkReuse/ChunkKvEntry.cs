namespace DotLLM.ChunkReuse;

/// <summary>
/// The per-layer K/V of one token chunk, encoded in isolation at positions <c>0..n-1</c> (position-independent form: keys are
/// post-RoPE at those positions and are moved with <see cref="RopeShift"/> on assembly). Host F32 arrays by design: this is an
/// off-hot-path research cache entry, not inference storage.
/// </summary>
/// <param name="ModelFingerprint">Identity of the model/engine that produced the entry (<see cref="ChunkFingerprint.Compute"/>).</param>
/// <param name="TokenIds">The chunk's token ids; chunks are never re-tokenised from text.</param>
/// <param name="KeysPerLayer">Per KV layer: <c>[n, kvStride(layer)]</c> keys, row-major, heads contiguous within a row.</param>
/// <param name="ValuesPerLayer">Per KV layer: <c>[n, kvStride(layer)]</c> values, same layout.</param>
public sealed record ChunkKvEntry(string ModelFingerprint, int[] TokenIds, float[][] KeysPerLayer, float[][] ValuesPerLayer);
