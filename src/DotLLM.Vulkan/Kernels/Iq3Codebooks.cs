using DotLLM.Cpu.Kernels;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Owns the GPU-resident codebook buffers shared by every IQ3 kernel —
/// <c>iq3xxs_grid</c> (256x4 bytes = 1024 B), <c>iq3s_grid</c>
/// (512x4 bytes = 2048 B) and the IQ2-family-shared <c>ksigns_iq2xs</c>
/// (128 B). Uploaded once and bound as readonly SSBOs alongside the per-row
/// weight bytes.
/// </summary>
/// <remarks>
/// <para>
/// Codebook bytes come from <c>DotLLM.Cpu.Kernels.Dequantize.Iq3*Grid</c> /
/// <c>KsignsIq2Xs</c> — the same arrays the CPU oracle uses, so a pixel
/// drift between GPU and CPU paths can never originate from a divergent
/// table (only from the per-element decode arithmetic). Each table is
/// rounded up to a multiple of 4 bytes and uploaded as <c>uint[]</c>; the
/// shader extracts individual bytes via
/// <c>(u &gt;&gt; ((idx &amp; 3) * 8)) &amp; 0xFFu</c>.
/// </para>
/// <para>
/// Two upload strategies are supported:
/// <list type="bullet">
/// <item><see cref="Create"/> — fresh <c>Iq3Codebooks</c> per kernel (used by
/// the dequant / matmul kernel unit tests that don't already have one).</item>
/// <item>The host-wired path passes a single <c>Iq3Codebooks</c> to
/// <c>CreateWithCodebooks</c> on each of the six IQ3 kernels — that's the
/// production path. Mirrors <see cref="Iq2Codebooks"/>.</item>
/// </list>
/// </para>
/// </remarks>
internal sealed class Iq3Codebooks : IDisposable
{
    public VulkanDevice.Buffer Iq3XxsGrid { get; }
    public VulkanDevice.Buffer Iq3SGrid { get; }
    public VulkanDevice.Buffer Ksigns { get; }

    private bool _disposed;

    private Iq3Codebooks(VulkanDevice.Buffer xxs, VulkanDevice.Buffer s, VulkanDevice.Buffer ksigns)
    {
        Iq3XxsGrid = xxs;
        Iq3SGrid = s;
        Ksigns = ksigns;
    }

    public static Iq3Codebooks Create(VulkanDevice device)
    {
        var xxs = UploadTable(device, Dequantize.Iq3XxsGrid);
        var s   = UploadTable(device, Dequantize.Iq3SGrid);
        var k   = UploadTable(device, Dequantize.KsignsIq2Xs);
        return new Iq3Codebooks(xxs, s, k);
    }

    private static VulkanDevice.Buffer UploadTable(VulkanDevice device, ReadOnlySpan<byte> bytes)
    {
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
        return buf;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Ksigns.Dispose();
        Iq3SGrid.Dispose();
        Iq3XxsGrid.Dispose();
    }
}
