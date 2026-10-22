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
            if (Bf16AsF16Enabled && d.QuantizationType == QuantizationType.BF16 && d.Shape.Rank == 2
                && TryUploadBf16AsF16(device, staging, gguf.TensorDataPointer(d), d.Shape[0], d.Shape[1], out var f16Buf, out long f16Bytes))
            {
                qt = QuantizationType.F16;
                total += f16Bytes;
                return f16Buf!;
            }
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

    /// <summary>
    /// #823: the ISTA GSQ-RCO files store the gated-residual low-rank projections as BF16, whose prefill GEMM is the plain tiled F32 kernel
    /// (about 3.4 s of a 7 s 1K-token prefill: 4 projections x 48 layers). F16 has the cooperative-matrix GEMM, and a BF16 weight is exactly
    /// representable in F16 unless it is tiny (below 2^-14 it loses mantissa bits, below 2^-24 it flushes), which for these weights moves no
    /// output measurably; the conversion is skipped (BF16 kept) when it would overflow F16 or be lossy for more than 0.1% of the elements.
    /// <c>DOTLLM_VK_Q4E_BF16_AS_F16=0</c> keeps BF16.
    /// </summary>
    internal static bool Bf16AsF16Enabled => Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_BF16_AS_F16") != "0";

    private static unsafe bool TryUploadBf16AsF16(VulkanDevice device, VulkanStagingBuffer staging, nint src, int inputDim, int outputDim,
        out VulkanDevice.Buffer? buffer, out long bytes)
    {
        buffer = null;
        bytes = 0;
        if (inputDim % 32 != 0 || outputDim < 16) return false;   // the F16 coopmat GEMM wants K % 32; tiny outputs are not worth a second format
        long elems = (long)inputDim * outputDim;
        var half = new Half[elems];
        ushort* p = (ushort*)src;
        long lossy = 0;
        for (long i = 0; i < elems; i++)
        {
            float f = BitConverter.UInt32BitsToSingle((uint)p[i] << 16);
            if (!float.IsFinite(f) || MathF.Abs(f) > 65000f) return false;
            Half h = (Half)f;
            if (f != 0f && (float)h != f && ++lossy > elems / 1000) return false;
            half[i] = h;
        }
        bytes = elems * 2;
        var buf = device.AllocateDeviceLocal(bytes);
        fixed (Half* hp = half) staging.UploadBytes((nint)hp, bytes, buf);
        buffer = buf;
        return true;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Norm.Dispose(); Down.Dispose(); Up.Dispose(); Inject?.Dispose();
    }
}
