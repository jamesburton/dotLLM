using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Device side of the qwen4exp n-gram (PLE) branch for short forwards (#885): the two projections of the gathered table rows
/// (<c>key = Wk emb</c>, <c>value = Wv emb</c>) depend only on the token ids, so they run on the GPU at the START of the forward
/// (ahead of layer 0, no host sync) instead of as a 131 MB F32 CPU GEMM between layer 0 and layer 1 (~4 ms of the ~57 ms decode step).
/// The residual-dependent remainder (query norm, gate, dilated conv with its carried history) stays on the host
/// (<see cref="DotLLM.Cpu.Kernels.Qwen4ExpPleBranch.ApplyProjected"/>) - it is a few hundred KFLOP.
/// </summary>
/// <remarks>
/// Weights are held as exact F32 (dequantised Q8_0, 105 + 26 MB), activations stay F32: bit-for-bit the same operands as the CPU
/// branch, only the summation order differs. All per-forward buffers are tiny and live for the model's lifetime.
/// </remarks>
internal sealed class VulkanQwen4ExpPleGpu : IDisposable
{
    /// <summary>Largest row count handled (decode 1, MTP verify up to 8); longer chunks use the all-host branch.</summary>
    public const int MaxRows = 8;

    public VulkanDevice.Buffer KeyW { get; }
    public VulkanDevice.Buffer ValueW { get; }
    /// <summary>Gathered table rows, <c>[MaxRows, EmbDim]</c> F32, host-written before the submit.</summary>
    public VulkanDevice.Buffer Emb { get; }
    /// <summary>Raw key projection <c>[MaxRows, KeyDim]</c>, host-readable (cached).</summary>
    public VulkanDevice.Buffer KeyOut { get; }
    /// <summary>Value projection <c>[MaxRows, ValueDim]</c>, host-readable (cached).</summary>
    public VulkanDevice.Buffer ValueOut { get; }
    /// <summary>Copy of the residual after the layer before the PLE layer, host-readable (cached).</summary>
    public VulkanDevice.Buffer ResidualOut { get; }
    /// <summary>The corrected residual, host-written and copied back into the device residual.</summary>
    public VulkanDevice.Buffer ResidualIn { get; }
    public int EmbDim { get; }
    public int KeyDim { get; }
    public int ValueDim { get; }
    public long Bytes { get; }

    private VulkanQwen4ExpPleGpu(VulkanDevice.Buffer keyW, VulkanDevice.Buffer valueW, int embDim, int keyDim, int valueDim, int residualRowFloats, long bytes,
        VulkanDevice device)
    {
        KeyW = keyW; ValueW = valueW; EmbDim = embDim; KeyDim = keyDim; ValueDim = valueDim; Bytes = bytes;
        Emb = device.Allocate((long)MaxRows * embDim * 4);
        KeyOut = device.AllocateHostReadback((long)MaxRows * keyDim * 4);
        ValueOut = device.AllocateHostReadback((long)MaxRows * valueDim * 4);
        ResidualOut = device.AllocateHostReadback((long)MaxRows * residualRowFloats * 4);
        ResidualIn = device.Allocate((long)MaxRows * residualRowFloats * 4);
    }

    /// <summary>Uploads the two projections (exact F32) and allocates the scratch.</summary>
    public static VulkanQwen4ExpPleGpu Upload(VulkanDevice device, VulkanStagingBuffer staging, GgufFile gguf, string keyName, string valueName, int residualRowFloats)
    {
        var t = gguf.TensorsByName;
        long total = 0;
        VulkanDevice.Buffer Proj(string name, out int inDim, out int outDim)
        {
            var d = t[name];
            inDim = d.Shape[0]; outDim = d.Shape[1];
            var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging, gguf.TensorDataPointer(d), d.QuantizationType,
                outputDim: outDim, inputDim: inDim, forceF32: true, out _, out long bytes);
            total += bytes;
            return buf;
        }
        var k = Proj(keyName, out int kIn, out int kOut);
        var v = Proj(valueName, out int vIn, out int vOut);
        if (kIn != vIn) throw new InvalidDataException("PLE key/value projections disagree on their input width.");
        return new VulkanQwen4ExpPleGpu(k, v, kIn, kOut, vOut, residualRowFloats, total, device);
    }

    /// <summary>Records <c>key = Wk emb</c> and <c>value = Wv emb</c> for <paramref name="rows"/> rows (the host already wrote <see cref="Emb"/>).</summary>
    public void Record(VulkanQwen3MoeHybridTransformerModel core, nint cmd, int rows)
    {
        core.Q4RecordMatmul(cmd, KeyW, QuantizationType.F32, Emb, KeyOut, outputDim: KeyDim, inputDim: EmbDim, seqLen: rows);
        core.Q4RecordMatmul(cmd, ValueW, QuantizationType.F32, Emb, ValueOut, outputDim: ValueDim, inputDim: EmbDim, seqLen: rows);
    }

    public void Dispose()
    {
        KeyW.Dispose(); ValueW.Dispose(); Emb.Dispose(); KeyOut.Dispose(); ValueOut.Dispose(); ResidualOut.Dispose(); ResidualIn.Dispose();
    }
}
