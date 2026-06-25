using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Owns the GPU-resident IQ1_S codebook buffer — the 2048-entry <c>iq1s_grid</c>
/// (each entry is a uint64 packing 8 signed-int8 ternary lanes 0xff/0x00/0x01 =
/// -1/0/+1), uploaded once as <c>uint[]</c> (4096 words) and bound as a readonly
/// SSBO alongside the per-row weight bytes.
/// </summary>
/// <remarks>
/// Bytes come from <c>DotLLM.Cpu.Kernels.Dequantize.Iq1SGrid</c> — the same array
/// the CPU oracle uses, so the GPU and CPU paths can never diverge on the table
/// itself (only on the per-element decode arithmetic). Unlike the F32 GEMV shader,
/// which embeds the grid as a compile-time const, the MMVQ shader reads it from
/// this SSBO so the ternary lanes feed dp4a directly.
/// </remarks>
internal sealed class Iq1Codebooks : IDisposable
{
    public VulkanDevice.Buffer Iq1SGrid { get; }

    private bool _disposed;

    private Iq1Codebooks(VulkanDevice.Buffer grid)
    {
        Iq1SGrid = grid;
    }

    public static Iq1Codebooks Create(VulkanDevice device)
    {
        ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(Dequantize.Iq1SGrid);
        long padded = (bytes.Length + 3) & ~3L;
        var buf = device.Allocate(padded);
        if (padded == bytes.Length)
        {
            device.Upload(bytes, buf);
        }
        else
        {
            Span<byte> tmp = stackalloc byte[(int)padded];
            bytes.CopyTo(tmp);
            tmp[bytes.Length..].Clear();
            device.Upload(tmp, buf);
        }
        return new Iq1Codebooks(buf);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Iq1SGrid.Dispose();
    }
}
