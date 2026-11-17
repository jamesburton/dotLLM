namespace DotLLM.Core.Configuration;

/// <summary>An iGPU that could take the layers the dGPU cannot hold (issue #1851).</summary>
/// <param name="Device">Device string for messages, e.g. <c>vulkan:1</c>.</param>
/// <param name="Name">Device name for messages, e.g. "Intel(R) Arc(TM) Graphics".</param>
/// <param name="FreeBytes">Device memory the tail may use (UMA: the device-local heap).</param>
/// <param name="ReserveBytes">Bytes kept free on the iGPU.</param>
/// <param name="Supported">False while no dGPU + iGPU host exists: a fitting iGPU is then reported, and the tail goes to the CPU.</param>
public sealed record IgpuTier(string Device, string Name, long FreeBytes, long ReserveBytes, bool Supported);

/// <summary>
/// Memory figures for <see cref="LayerSplitPlanner.Plan"/> (issue #1851). The dGPU runs the layer PREFIX <c>[0, N)</c> plus the
/// token embedding; the tail (iGPU or CPU) runs the rest, the final norm and the LM head, as <c>HybridTransformerModel</c> does.
/// </summary>
/// <param name="DgpuFreeBytes">Free dGPU memory, measured with the CUDA context, stream, cuBLAS and kernels already created.</param>
/// <param name="ReserveBytes">dGPU bytes kept free (the admission reserve, <c>DOTLLM_CUDA_VRAM_RESERVE_MB</c>).</param>
/// <param name="DgpuFixedBytes">dGPU bytes paid once whatever N is: the token embedding as uploaded and the fixed forward buffers.</param>
/// <param name="LayerWeightBytes">Device weight bytes of each layer, as the dGPU upload stores them.</param>
/// <param name="KvBytesPerTokenPerLayer">dGPU KV bytes per context token per layer (F16 K and V).</param>
/// <param name="ContextTokens">The context the split must serve.</param>
/// <param name="ActivationBytesPerToken">dGPU forward scratch per token of the largest forward (paid once, not per layer).</param>
/// <param name="ActivationTokens">Tokens of the largest forward the scratch must hold (the prefill, or its chunk).</param>
/// <param name="HeadBytes">The final norm and LM head: on the dGPU only when every layer is; otherwise on the tail.</param>
/// <param name="Igpu">The iGPU tier, or <see langword="null"/> when there is none.</param>
public sealed record LayerSplitInput(
    long DgpuFreeBytes, long ReserveBytes, long DgpuFixedBytes, IReadOnlyList<long> LayerWeightBytes,
    long KvBytesPerTokenPerLayer, int ContextTokens, long ActivationBytesPerToken, int ActivationTokens,
    long HeadBytes, IgpuTier? Igpu);

/// <summary>The placement <see cref="LayerSplitPlanner.Plan"/> chose.</summary>
/// <param name="DgpuLayers">Layers <c>[0, DgpuLayers)</c> on the dGPU (CUDA).</param>
/// <param name="IgpuLayers">Layers after those on the iGPU (Vulkan).</param>
/// <param name="CpuLayers">Layers after those on the CPU.</param>
/// <param name="IgpuDevice">The iGPU device when <see cref="IgpuLayers"/> is non-zero.</param>
/// <param name="Reason">Why, with the numbers, for the log.</param>
public sealed record LayerSplitPlan(int DgpuLayers, int IgpuLayers, int CpuLayers, string? IgpuDevice, string Reason)
{
    /// <summary>Total layers.</summary>
    public int TotalLayers => DgpuLayers + IgpuLayers + CpuLayers;

    /// <summary>True when every layer is on the dGPU.</summary>
    public bool AllOnDgpu => IgpuLayers == 0 && CpuLayers == 0;

    /// <summary>The placement line, e.g. "36 cuda + 4 cpu layers".</summary>
    public string Describe()
    {
        var parts = new List<string> { $"{DgpuLayers} cuda" };
        if (IgpuLayers > 0) parts.Add($"{IgpuLayers} vulkan(iGPU {IgpuDevice})");
        if (CpuLayers > 0) parts.Add($"{CpuLayers} cpu");
        return string.Join(" + ", parts) + " layers";
    }
}

/// <summary>
/// Chooses <c>--gpu-layers</c> automatically (issue #1851): the largest dGPU layer prefix whose weights, KV for the served
/// context, forward scratch and reserve fit in free dGPU memory, then the remainder on the iGPU when one is present, supported
/// and holds all of it, otherwise on the CPU. Pure, so every term is unit-testable without a GPU. Modelled on
/// <c>CudaTransformerModel.PlanCpuMoeLayers</c>.
/// </summary>
public static class LayerSplitPlanner
{
    /// <summary>Follow-up issue for the dGPU + iGPU host (the planner already places layers there).</summary>
    public const string IgpuTierIssueUrl = "https://github.com/jamesburton/dotLLM/issues/1863";

    /// <summary>Plans the split.</summary>
    /// <param name="input">Memory figures.</param>
    /// <returns>The placement. <see cref="LayerSplitPlan.DgpuLayers"/> is 0 when not even one layer fits.</returns>
    public static LayerSplitPlan Plan(LayerSplitInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        int total = input.LayerWeightBytes.Count;
        if (total == 0) return new(0, 0, 0, null, "the model has no layers");

        long ctx = Math.Max(0, input.ContextTokens);
        long kvPerLayer = input.KvBytesPerTokenPerLayer * ctx;
        long budget = input.DgpuFreeBytes - input.ReserveBytes;
        long once = input.DgpuFixedBytes + input.ActivationBytesPerToken * Math.Max(0, input.ActivationTokens);

        long all = once + input.HeadBytes;
        for (int l = 0; l < total; l++) all += input.LayerWeightBytes[l] + kvPerLayer;

        string figures = $"{Mib(input.DgpuFreeBytes)} MiB free, {Mib(input.ReserveBytes)} MiB reserve, {Mib(once)} MiB embedding + scratch, " +
                         $"{Mib(kvPerLayer)} MiB f16 KV per layer for {ctx} tokens";
        if (all <= budget)
            return new(total, 0, 0, null, $"every layer fits on the dGPU ({Mib(all)} MiB needed; {figures})");

        // Largest prefix that fits. The tail always keeps the head, so at most total - 1 layers go to the dGPU.
        long used = once;
        int n = 0;
        while (n < total - 1 && used + input.LayerWeightBytes[n] + kvPerLayer <= budget)
        {
            used += input.LayerWeightBytes[n] + kvPerLayer;
            n++;
        }

        int rest = total - n;
        string why = n == 0
            ? $"not even one layer fits on the dGPU ({Mib(all)} MiB for the whole model; {figures})"
            : $"{n} of {total} layers fit on the dGPU ({Mib(used)} MiB of {Mib(Math.Max(0, budget))} MiB usable; the whole model needs {Mib(all)} MiB; {figures})";

        if (input.Igpu is { } igpu && n > 0)
        {
            long tail = input.HeadBytes;
            for (int l = n; l < total; l++) tail += input.LayerWeightBytes[l] + kvPerLayer;
            bool fits = tail <= igpu.FreeBytes - igpu.ReserveBytes;
            if (fits && igpu.Supported)
                return new(n, rest, 0, igpu.Device,
                    $"{why}; the remaining {rest} layers and the LM head ({Mib(tail)} MiB) fit on the iGPU {igpu.Device} ({igpu.Name})");
            why += fits
                ? $"; the iGPU {igpu.Device} ({igpu.Name}) would hold the remaining {rest} layers ({Mib(tail)} MiB), but the dGPU + iGPU " +
                  $"split is not available yet ({IgpuTierIssueUrl}), so they run on the CPU"
                : $"; the iGPU {igpu.Device} ({igpu.Name}) cannot hold the remaining {rest} layers ({Mib(tail)} MiB needed, " +
                  $"{Mib(Math.Max(0, igpu.FreeBytes - igpu.ReserveBytes))} MiB usable), so they run on the CPU";
        }
        else if (n > 0)
        {
            why += $"; no iGPU, so the remaining {rest} layers run on the CPU";
        }
        return new(n, 0, rest, null, why);
    }

    private static long Mib(long bytes) => (bytes + (1L << 20) - 1) >> 20;
}
