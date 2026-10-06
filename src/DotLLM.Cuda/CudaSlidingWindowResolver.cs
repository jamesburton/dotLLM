namespace DotLLM.Cuda;

/// <summary>
/// Per-layer sliding-window resolution for CUDA attention dispatch. Mirrors the CPU
/// reference <c>TransformerModel.GetLayerSlidingWindow</c> (TransformerModel.cs:706-715)
/// exactly, but returns the CUDA kernel convention: 0 = dense/full attention, positive =
/// window length (attention_f32.cu masks <c>pos_q - tkv &gt;= sliding_window</c> only when
/// the value is &gt; 0). Pattern N&gt;0 windows layers where <c>layer % N &lt; N - 1</c>
/// (llama.cpp <c>set_swa_pattern(N, dense_first=false)</c>; gpt-oss N=2 → even layers
/// windowed, odd dense; Gemma-3 uses N=6).
/// </summary>
internal static class CudaSlidingWindowResolver
{
    internal static int Resolve(int? slidingWindowSize, int pattern,
        System.Collections.Generic.IReadOnlyList<int?>? perLayer, int layer)
    {
        if (perLayer is not null && (uint)layer < (uint)perLayer.Count)
            return perLayer[layer] ?? 0;
        if (slidingWindowSize is null) return 0;
        if (pattern <= 0) return slidingWindowSize.Value;
        return (layer % pattern) < pattern - 1 ? slidingWindowSize.Value : 0;
    }
}
