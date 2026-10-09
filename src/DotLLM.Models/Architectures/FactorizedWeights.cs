using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels.Experimental;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Registry behind <see cref="DotLLM.Core.Configuration.QuantizationType.LittleBit"/> weights (issue #864). The
/// "weight pointer" stored in <c>TransformerLayerWeights</c> is a 64-byte native token block (so the existing
/// owned-allocation bookkeeping frees it); this registry maps the token to the managed
/// <see cref="LittleBitLayer"/> and disposes the layer when the token is released with the weights.
/// </summary>
internal static class FactorizedWeights
{
    private static readonly ConcurrentDictionary<nint, LittleBitLayer> Layers = new();

    /// <summary>Registers a layer; returns the token to store as the weight pointer. Add it to the owned list.</summary>
    public static unsafe nint Register(LittleBitLayer layer)
    {
        nint token = (nint)NativeMemory.AlignedAlloc(64, 64);
        Layers[token] = layer;
        return token;
    }

    /// <summary>Resolves a token created by <see cref="Register"/>.</summary>
    public static LittleBitLayer Get(nint token) => Layers[token];

    /// <summary>Disposes the layer behind a token if it is one (called when the owning weights are disposed).</summary>
    public static void Release(nint token)
    {
        if (Layers.TryRemove(token, out var l)) l.Dispose();
    }

    /// <summary>Total bytes the kernels stream for all live layers (packed bits + F32 scales).</summary>
    public static long LiveBytes => Layers.Values.Sum(l => l.WeightBytes);

    /// <summary>True when the checkpoint stores any linear as a LittleBit composite.</summary>
    public static bool IsFactorized(DotLLM.Models.SafeTensors.ISafetensorsTensorSource file)
        => file.TensorsByName.ContainsKey("model.layers.0.self_attn.q_proj.U_packed");
}
