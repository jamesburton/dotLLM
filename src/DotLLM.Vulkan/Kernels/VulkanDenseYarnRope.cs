using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Dense-YaRN RoPE for a Vulkan host (issue #1101): the <c>rope_yarn_f32</c> kernel, the device copy of the ramped
/// inverse frequencies from <see cref="DotLLM.Cpu.Kernels.RoPE.ResolveDenseYarn"/> (the helper the CPU table and the
/// CUDA upload use too) and the cos/sin multiplier. Created only when the model's RoPE has YaRN active; a host
/// without one keeps its theta-derived <see cref="RopeF32Kernel"/>.
/// </summary>
internal sealed class VulkanDenseYarnRope : IDisposable
{
    private const int WorkgroupSize = 256;

    private readonly SimpleComputeKernel _kernel;
    private readonly VulkanDevice.Buffer _invFreq;
    private bool _disposed;

    private VulkanDenseYarnRope(SimpleComputeKernel kernel, VulkanDevice.Buffer invFreq, int ropeDim, float mscale)
    {
        _kernel = kernel;
        _invFreq = invFreq;
        RopeDim = ropeDim;
        Mscale = mscale;
    }

    /// <summary>Rotated dims per head.</summary>
    public int RopeDim { get; }

    /// <summary>Cos/sin multiplier (<see cref="DotLLM.Core.PositionEncoding.RoPEConfig.ComputeYarnMscaleMultiplier"/>).</summary>
    public float Mscale { get; }

    /// <summary>Dispatches recorded (test evidence that the YaRN path ran).</summary>
    public long Dispatches { get; private set; }

    /// <summary>
    /// Creates the YaRN rotation for <paramref name="config"/>, or returns null when its RoPE has no active dense YaRN.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="config">Model configuration (RoPE and architecture).</param>
    /// <param name="ropeDim">Rotated dims per head; the YaRN ramp runs over these, not the head width.</param>
    /// <returns>The rotation, or null.</returns>
    public static VulkanDenseYarnRope? TryCreate(VulkanDevice device, string spvDir, ModelConfig config, int ropeDim)
    {
        var (invFreq, mscale) = DotLLM.Cpu.Kernels.RoPE.ResolveDenseYarn(config.RoPEConfig, config.Architecture, ropeDim);
        if (invFreq is null)
            return null;

        var kernel = SimpleComputeKernel.Create(device, spvDir, "rope_yarn_f32", buffers: 4, pushWords: 8);
        VulkanDevice.Buffer? buf = null;
        try
        {
            buf = device.Allocate((long)invFreq.Length * sizeof(float));
            device.Upload(invFreq.AsSpan(), buf);
            return new VulkanDenseYarnRope(kernel, buf, ropeDim, mscale);
        }
        catch
        {
            buf?.Dispose();
            kernel.Dispose();
            throw;
        }
    }

    /// <summary>Rotates Q and K in place (NeoX pairing) at the positions in <paramref name="positions"/>.</summary>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="q">Q rows, <c>[seqLen, numHeads, headDim]</c>.</param>
    /// <param name="k">K rows, <c>[seqLen, numKvHeads, headDim]</c>.</param>
    /// <param name="positions">Per-row positions (int32).</param>
    /// <param name="seqLen">Rows.</param>
    /// <param name="numHeads">Q heads to rotate.</param>
    /// <param name="numKvHeads">K heads to rotate.</param>
    /// <param name="headDim">Head width.</param>
    public void Record(nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer positions,
        int seqLen, int numHeads, int numKvHeads, int headDim)
    {
        Record(_kernel, cmdBuf, q, k, positions, _invFreq, seqLen, numHeads, numKvHeads, headDim, RopeDim, neox: true, Mscale);
        Dispatches++;
    }

    /// <summary>Records one <c>rope_yarn_f32</c> dispatch on <paramref name="kernel"/> (shared with gpt-oss).</summary>
    /// <param name="kernel">A <c>rope_yarn_f32</c> kernel (4 buffers, 8 push words).</param>
    /// <param name="cmdBuf">Command buffer being recorded.</param>
    /// <param name="q">Q rows.</param>
    /// <param name="k">K rows.</param>
    /// <param name="positions">Per-row positions.</param>
    /// <param name="invFreq">Inverse frequencies, <c>ropeDim / 2</c> floats.</param>
    /// <param name="seqLen">Rows.</param>
    /// <param name="numHeads">Q heads.</param>
    /// <param name="numKvHeads">K heads.</param>
    /// <param name="headDim">Head width.</param>
    /// <param name="ropeDim">Rotated dims per head.</param>
    /// <param name="neox">NeoX (<c>i, i + ropeDim/2</c>) pairing; otherwise adjacent pairs.</param>
    /// <param name="mscale">Cos/sin multiplier.</param>
    internal static void Record(SimpleComputeKernel kernel, nint cmdBuf, VulkanDevice.Buffer q, VulkanDevice.Buffer k,
        VulkanDevice.Buffer positions, VulkanDevice.Buffer invFreq,
        int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, bool neox, float mscale)
    {
        if (ropeDim <= 0 || (ropeDim & 1) != 0 || ropeDim > headDim)
            throw new ArgumentException($"ropeDim {ropeDim} invalid for headDim {headDim}.", nameof(ropeDim));
        ReadOnlySpan<nint> bufs = [q.Handle, k.Handle, positions.Handle, invFreq.Handle];
        ReadOnlySpan<uint> pc =
        [
            (uint)seqLen, (uint)numHeads, (uint)numKvHeads, (uint)headDim, (uint)ropeDim,
            neox ? 1u : 0u, BitConverter.SingleToUInt32Bits(mscale), (uint)(ropeDim / 2),
        ];
        long maxPairs = (long)seqLen * Math.Max(numHeads, numKvHeads) * (ropeDim / 2);
        kernel.Record(cmdBuf, bufs, pc, (uint)((maxPairs + WorkgroupSize - 1) / WorkgroupSize));
    }

    /// <summary>Drops the handle-keyed descriptor cache (after any bound buffer was re-created).</summary>
    public void InvalidateDescriptorCache() => _kernel.InvalidateDescriptorCache();

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _invFreq.Dispose();
        _kernel.Dispose();
    }
}
