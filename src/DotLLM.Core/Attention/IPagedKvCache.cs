using DotLLM.Core.Tensors;

namespace DotLLM.Core.Attention;

/// <summary>
/// Host KV cache stored in fixed-size physical blocks of a shared pool, which attention kernels read in place:
/// no gather of the history into a contiguous copy (#1268).
/// <para>
/// <b>Element type is generic.</b> The block views carry it in <see cref="TensorRef.DType"/>
/// (<see cref="DType.Float16"/>, <see cref="DType.Q8_0"/>, or <see cref="DType.Float32"/> for reference pools);
/// a row is <c>DType.ComputeByteCount(KvStride)</c> bytes and Q8_0 rows keep each head on whole 32-element blocks.
/// </para>
/// <para>
/// <b>Uniform stride only.</b> Every layer has the same <see cref="KvStride"/>. Models with per-layer KV widths
/// (Gemma-4 dual head dim, MiMo-V2 sliding/global KV heads) are not supported and are routed to the simple cache.
/// </para>
/// <para>Views are borrowed until the next cache mutation or disposal.</para>
/// </summary>
public interface IPagedKvCache : IKvCache
{
    /// <summary>Tokens per physical block.</summary>
    int BlockSize { get; }

    /// <summary>Elements per token row (KV heads × head dimension), identical for every layer.</summary>
    int KvStride { get; }

    /// <summary>Physical block IDs in logical order; entry <c>i</c> holds tokens <c>[i * BlockSize, (i + 1) * BlockSize)</c>.</summary>
    ReadOnlySpan<int> BlockIds { get; }

    /// <summary>
    /// Base of all physical key blocks of one layer: <c>[totalBlocks * BlockSize, KvStride]</c> rows in the pool's element type.
    /// </summary>
    /// <param name="layerIndex">Layer index.</param>
    /// <returns>Borrowed view of the layer's key block storage.</returns>
    TensorRef GetKeyBlocksRef(int layerIndex);

    /// <summary>
    /// Base of all physical value blocks of one layer: <c>[totalBlocks * BlockSize, KvStride]</c> rows in the pool's element type.
    /// </summary>
    /// <param name="layerIndex">Layer index.</param>
    /// <returns>Borrowed view of the layer's value block storage.</returns>
    TensorRef GetValueBlocksRef(int layerIndex);
}
