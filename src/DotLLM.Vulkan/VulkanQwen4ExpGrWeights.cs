using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;

namespace DotLLM.Vulkan;

/// <summary>
/// Device-resident weights of one Qwen4-Exp gated-residual ("hyper-connection") module: the per-stream norm gamma (F32,
/// <c>1 + w</c> already folded by the converter), the low-rank down / up projections and - for the two per-block modules, not the
/// final head mixer - the write-gain inject projection. Projections keep their GGUF quant on the device when the contraction axis
/// is aligned (<see cref="VulkanQwen3MoeHybridWeights.UploadProjectionMatrix"/>), otherwise they widen to F32.
/// </summary>
internal sealed class VulkanQwen4ExpGrWeights : IDisposable
{
    /// <summary>Gamma <c>[S * H]</c>, F32.</summary>
    public VulkanDevice.Buffer Norm { get; }
    /// <summary>Down projection (<c>S*H -&gt; lowRank</c>).</summary>
    public VulkanDevice.Buffer Down { get; }
    /// <summary>Device storage type of <see cref="Down"/>.</summary>
    public QuantizationType DownQt { get; }
    /// <summary>Up projection (<c>lowRank -&gt; S*H</c>).</summary>
    public VulkanDevice.Buffer Up { get; }
    /// <summary>Device storage type of <see cref="Up"/>.</summary>
    public QuantizationType UpQt { get; }
    /// <summary>Inject projection (<c>S*H -&gt; S</c>); null for the head mixer.</summary>
    public VulkanDevice.Buffer? Inject { get; }
    /// <summary>Device storage type of <see cref="Inject"/>.</summary>
    public QuantizationType InjectQt { get; }
    /// <summary>Total uploaded bytes.</summary>
    public long Bytes { get; }

    private VulkanQwen4ExpGrWeights(VulkanDevice.Buffer norm, VulkanDevice.Buffer down, QuantizationType downQt,
        VulkanDevice.Buffer up, QuantizationType upQt, VulkanDevice.Buffer? inject, QuantizationType injectQt, long bytes)
    {
        Norm = norm; Down = down; DownQt = downQt; Up = up; UpQt = upQt; Inject = inject; InjectQt = injectQt; Bytes = bytes;
    }

    /// <summary>
    /// Uploads one module from the GGUF.
    /// </summary>
    /// <param name="device">Target device.</param>
    /// <param name="staging">Shared staging buffer.</param>
    /// <param name="gguf">Source file.</param>
    /// <param name="norm">Gamma tensor name.</param>
    /// <param name="down">Down projection tensor name.</param>
    /// <param name="up">Up projection tensor name.</param>
    /// <param name="inject">Inject projection tensor name, or null (head mixer).</param>
    public static unsafe VulkanQwen4ExpGrWeights Upload(VulkanDevice device, VulkanStagingBuffer staging, GgufFile gguf,
        string norm, string down, string up, string? inject)
    {
        var t = gguf.TensorsByName;
        long total = 0;

        var nd = t[norm];
        long normCount = 1;
        for (int i = 0; i < nd.Shape.Rank; i++) normCount *= nd.Shape[i];
        var gamma = new float[normCount];
        DotLLM.Cpu.Kernels.Dequantize.ToFloat32(gguf.TensorDataPointer(nd), normCount, nd.QuantizationType, gamma);
        var normBuf = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, gamma);
        total += normCount * 4;

        VulkanDevice.Buffer Proj(string name, out QuantizationType qt)
        {
            var d = t[name];
            var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging, gguf.TensorDataPointer(d), d.QuantizationType,
                outputDim: d.Shape[1], inputDim: d.Shape[0], forceF32: false, out qt, out long bytes);
            total += bytes;
            return buf;
        }
        var downBuf = Proj(down, out var downQt);
        var upBuf = Proj(up, out var upQt);
        VulkanDevice.Buffer? injBuf = null;
        QuantizationType injQt = QuantizationType.F32;
        if (inject is not null) injBuf = Proj(inject, out injQt);
        return new VulkanQwen4ExpGrWeights(normBuf, downBuf, downQt, upBuf, upQt, injBuf, injQt, total);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Norm.Dispose(); Down.Dispose(); Up.Dispose(); Inject?.Dispose();
    }
}
