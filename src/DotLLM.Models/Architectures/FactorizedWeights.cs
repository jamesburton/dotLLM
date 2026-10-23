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
    private static readonly ConcurrentDictionary<nint, NanoQuantSalient> Salients = new();

    /// <summary>Registers a layer; returns the token to store as the weight pointer. Add it to the owned list.</summary>
    public static unsafe nint Register(LittleBitLayer layer)
    {
        nint token = (nint)NativeMemory.AlignedAlloc(64, 64);
        Layers[token] = layer;
        return token;
    }

    /// <summary>
    /// Registers a NanoQuant layer (issue #869): the single-path base layer plus its optional salient-column side path,
    /// which <c>TransformerModel.GemmLittleBit</c> applies after the base GEMM.
    /// </summary>
    public static nint Register(NanoQuantLayer layer)
    {
        nint token = Register(layer.Base);
        if (layer.Salient is { } s) Salients[token] = s;
        return token;
    }

    /// <summary>The salient side path behind a token, or null.</summary>
    public static NanoQuantSalient? GetSalient(nint token) => Salients.TryGetValue(token, out var s) ? s : null;

    /// <summary>Resolves a token created by <see cref="Register(LittleBitLayer)"/>.</summary>
    public static LittleBitLayer Get(nint token) => Layers[token];

    /// <summary>Disposes the layer behind a token if it is one (called when the owning weights are disposed).</summary>
    public static void Release(nint token)
    {
        Salients.TryRemove(token, out _);
        if (Layers.TryRemove(token, out var l)) l.Dispose();
    }

    /// <summary>
    /// Loads a NanoQuant composite (issue #869) from a GGUF and registers it as a LittleBit-kind weight. A non-empty
    /// <paramref name="rowSplits"/> splits the stacked output rows (attn_qkv) into one weight per entry. With
    /// <c>DOTLLM_LITTLEBIT_DENSE_CONTROL=1</c> every layer is decoded to dense F32 instead (the F32-decoded control).
    /// </summary>
    public static unsafe (nint ptr, DotLLM.Core.Configuration.QuantizationType qt, int m, int k)[] ResolveNanoQuant(
        DotLLM.Models.Gguf.GgufFile file, string baseName, int[] rowSplits, List<nint> owned)
    {
        var layers = DotLLM.Models.Quantization.NanoQuantLoader.LoadSplit(file, baseName, rowSplits);
        bool dense = Environment.GetEnvironmentVariable("DOTLLM_LITTLEBIT_DENSE_CONTROL") == "1";
        var result = new (nint, DotLLM.Core.Configuration.QuantizationType, int, int)[layers.Length];
        for (int i = 0; i < layers.Length; i++)
        {
            var l = layers[i];
            int dOut = l.DOut, dIn = l.DIn;
            if (dense)
            {
                nint w = (nint)l.DecodeDense();
                result[i] = (w, DotLLM.Core.Configuration.QuantizationType.F32, dOut, dIn);
                owned.Add(w);
                l.Dispose();
            }
            else
            {
                nint token = Register(l);
                owned.Add(token);
                result[i] = (token, DotLLM.Core.Configuration.QuantizationType.LittleBit, dOut, dIn);
            }
        }
        return result;
    }

    /// <summary>Total bytes the kernels stream for all live layers (packed bits + F32 scales).</summary>
    public static long LiveBytes => Layers.Values.Sum(l => l.WeightBytes) + Salients.Values.Sum(s => s.Bytes);

    /// <summary>True when the checkpoint stores any linear as a LittleBit composite.</summary>
    public static bool IsFactorized(DotLLM.Models.SafeTensors.ISafetensorsTensorSource file)
        => file.TensorsByName.ContainsKey("model.layers.0.self_attn.q_proj.U_packed");
}
