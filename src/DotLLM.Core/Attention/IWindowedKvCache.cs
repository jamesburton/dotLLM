namespace DotLLM.Core.Attention;

/// <summary>
/// Optional capability of a KV-cache whose sliding-window layers keep only their recent rows. Row 0 of such a layer's
/// buffers holds position <see cref="RowBaseOf"/>, so <see cref="IKvCache.GetKeysRef"/> / <see cref="IKvCache.GetValuesRef"/>
/// cover positions <c>[RowBaseOf(layer), CurrentLength)</c>. Attention over that layer passes
/// <c>positionOffset - base</c> and <c>seqKv - base</c>: causal and window masks are relative and K is stored after RoPE,
/// so shifting both by the same base changes nothing else.
/// </summary>
public interface IWindowedKvCache
{
    /// <summary>The position held in row 0 of <paramref name="layerIndex"/>'s buffers; 0 for a full-attention layer.</summary>
    /// <param name="layerIndex">Cache slot (layer) index.</param>
    /// <returns>The first position the layer still holds.</returns>
    int RowBaseOf(int layerIndex);
}
