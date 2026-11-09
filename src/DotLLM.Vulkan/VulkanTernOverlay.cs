using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-side storage of the per-expert <see cref="QuantizationType.TERN_B64_F16"/> overlay tensors of ONE MoE layer (#1706 S3): per projection
/// (gate, up, down) one bank holding the overlaid experts' tensors back to back, PACKED (18 B per 64 weights, never widened), and an int32
/// slot table <c>[numExperts]</c> mapping an expert id to its slot in the bank or -1 (the expert keeps its original format and goes through the
/// existing indexed kernels). The original bank still holds the replaced experts' original weights (they are simply never read for the
/// cells the ternary pass overwrites).
/// </summary>
internal sealed class VulkanTernMoeLayer : IDisposable
{
    /// <summary>Packed ternary banks per projection (0 gate, 1 up, 2 down); null when no expert of the layer is ternary for that projection.</summary>
    public VulkanDevice.Buffer?[] Bank { get; } = new VulkanDevice.Buffer?[3];

    /// <summary>int32 slot tables per projection, <c>[numExperts]</c>; null alongside <see cref="Bank"/>.</summary>
    public VulkanDevice.Buffer?[] Slot { get; } = new VulkanDevice.Buffer?[3];

    /// <summary>Host copy of the slot tables (diagnostics / tests).</summary>
    public int[]?[] HostSlot { get; } = new int[]?[3];

    /// <summary>Device bytes held (banks + tables).</summary>
    public long DeviceBytes { get; internal set; }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = 0; i < 3; i++) { Bank[i]?.Dispose(); Slot[i]?.Dispose(); }
    }
}

/// <summary>
/// The Vulkan side of <c>--tensor-overlay</c> for routed experts (#1706 S3): validates the overlay's expert entries against the loaded MoE layers
/// and uploads them packed. Dense-FFN entries never reach here: they are swapped into the CPU layer struct before the normal upload, which keeps
/// them packed through <c>VulkanWeights</c> (<c>DeviceQuantType = TERN_B64_F16</c>) and dispatches them from <c>RecordMatmul</c>.
/// </summary>
internal sealed class VulkanTernOverlay : IDisposable
{
    private readonly Dictionary<int, VulkanTernMoeLayer> _layers = new();

    /// <summary>Number of expert tensors uploaded.</summary>
    public int ExpertTensorCount { get; private set; }

    /// <summary>Device bytes held by all banks and tables.</summary>
    public long DeviceBytes { get; private set; }

    /// <summary>True when layer <paramref name="layer"/> has at least one ternary expert tensor.</summary>
    public bool TryGetLayer(int layer, out VulkanTernMoeLayer moe) => _layers.TryGetValue(layer, out moe!);

    /// <summary>
    /// Architectures whose Vulkan FFN is plain SwiGLU (<c>silu(gate) * up</c> -> down, top-k softmax routing), the only ones the overlay supports.
    /// Anything else must fail loudly rather than run the unmodified base.
    /// </summary>
    public static void ValidateArchitecture(ModelConfig config)
    {
        bool ok = config.Architecture is Architecture.Llama or Architecture.Mistral or Architecture.Qwen or Architecture.Mixtral or Architecture.QwenMoe
                  && config.ActivationFunction == ActivationFunction.SiLU
                  && config.MlaConfig is null && !config.IsGemma4DensePle && config.Moe is not { SigmoidGating: true };
        if (!ok)
            throw new NotSupportedException(
                $"--tensor-overlay on the Vulkan backend supports plain-SwiGLU dense and routed-MoE models (Llama, Mistral, Qwen, Mixtral, OLMoE/Qwen-MoE); "
                + $"architecture {config.Architecture} is not supported yet. Use --device cpu.");
    }

    /// <summary>
    /// Validates and uploads the expert entries of <paramref name="overlay"/>. Returns null when the overlay has none.
    /// </summary>
    /// <exception cref="NotSupportedException">An expert entry is not TERN_B64_F16, or the model window is a pipeline stage.</exception>
    /// <exception cref="InvalidDataException">Unknown layer/expert, dense layer named as MoE, shape mismatch.</exception>
    public static unsafe VulkanTernOverlay? Create(VulkanDevice device, TensorOverlay overlay, TransformerWeights cpuWeights)
    {
        var groups = new Dictionary<(int Layer, int Proj), List<TensorOverlay.GpuEntry>>();
        foreach (var e in overlay.GpuEntries())
        {
            if (e.Expert < 0) continue;
            string what = $"Tensor overlay '{overlay.Path}': '{e.Name}'";
            if (e.Qt != QuantizationType.TERN_B64_F16)
                throw new NotSupportedException($"{what} is {e.Qt}; the Vulkan overlay path supports TERN_B64_F16 tensors only (use --device cpu for other formats).");
            if (e.Layer >= cpuWeights.Layers.Length)
                throw new InvalidDataException($"{what} names layer {e.Layer}, but the model has {cpuWeights.Layers.Length} layers.");
            var moe = cpuWeights.Layers[e.Layer].Moe
                ?? throw new InvalidDataException($"{what} names an expert, but layer {e.Layer} of the base model is a dense FFN layer.");
            if (e.Expert >= moe.NumExperts)
                throw new InvalidDataException($"{what} names expert {e.Expert}, but the layer has {moe.NumExperts} experts.");
            int expIn = e.Proj == 2 ? moe.IntermediateSize : moe.HiddenSize;
            int expOut = e.Proj == 2 ? moe.HiddenSize : moe.IntermediateSize;
            if (e.In != expIn || e.Out != expOut)
                throw new InvalidDataException($"{what} has shape [in={e.In}, out={e.Out}], but the base projection is [in={expIn}, out={expOut}].");
            if (!groups.TryGetValue((e.Layer, e.Proj), out var list)) groups[(e.Layer, e.Proj)] = list = [];
            list.Add(e);
        }
        if (groups.Count == 0) return null;

        var result = new VulkanTernOverlay();
        VulkanStagingBuffer? staging = null;
        try
        {
            long maxTensor = groups.Values.SelectMany(l => l).Max(t => (long)t.Out * (t.In / QuantFormat.TernB64GroupSize) * QuantFormat.TernB64F16BlockBytes);
            staging = VulkanStagingBuffer.Create(device, maxTensor);
            foreach (var ((layer, proj), list) in groups.OrderBy(g => g.Key.Layer).ThenBy(g => g.Key.Proj))
            {
                int numE = cpuWeights.Layers[layer].Moe!.NumExperts;
                if (!result._layers.TryGetValue(layer, out var vl)) result._layers[layer] = vl = new VulkanTernMoeLayer();
                list.Sort((a, b) => a.Expert.CompareTo(b.Expert));
                long tensorBytes = (long)list[0].Out * (list[0].In / QuantFormat.TernB64GroupSize) * QuantFormat.TernB64F16BlockBytes;
                long total = tensorBytes * list.Count;
                long padded = (total + 3) & ~3L;
                var bank = device.AllocateDeviceLocal(padded);
                vl.Bank[proj] = bank;
                var host = new int[numE];
                Array.Fill(host, -1);
                for (int s = 0; s < list.Count; s++)
                {
                    if (host[list[s].Expert] != -1)
                        throw new InvalidDataException($"Tensor overlay '{overlay.Path}': duplicate expert {list[s].Expert} for layer {layer}.");
                    host[list[s].Expert] = s;
                    staging.UploadBytes(list[s].Ptr, tensorBytes, bank, dstOffset: s * tensorBytes,
                        zeroTailBytes: s == list.Count - 1 ? padded - total : 0);
                }
                var slot = device.AllocateDeviceLocal((long)numE * sizeof(int));
                fixed (int* hp = host) staging.UploadBytes((nint)hp, (long)numE * sizeof(int), slot);
                vl.Slot[proj] = slot;
                vl.HostSlot[proj] = host;
                vl.DeviceBytes += padded + (long)numE * sizeof(int);
                result.DeviceBytes += padded + (long)numE * sizeof(int);
                result.ExpertTensorCount += list.Count;
                foreach (var t in list) overlay.MarkApplied(t.Name);
            }
            staging.WaitAll();
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            staging?.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var l in _layers.Values) l.Dispose();
        _layers.Clear();
    }
}
