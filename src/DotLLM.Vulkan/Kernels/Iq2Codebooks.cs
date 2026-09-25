using DotLLM.Cpu.Kernels;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Owns the GPU-resident codebook buffers shared by every IQ2 kernel —
/// <c>iq2xxs_grid</c> (256x8 bytes), <c>iq2xs_grid</c> (512x8), <c>iq2s_grid</c>
/// (1024x8) and <c>ksigns_iq2xs</c> (128 bytes). Uploaded once and bound as
/// readonly SSBOs alongside the per-row weight bytes.
/// </summary>
/// <remarks>
/// Codebook bytes are sourced from <c>DotLLM.Cpu.Kernels.Dequantize.Iq2*Grid</c>
/// / <c>KsignsIq2Xs</c> — the same byte arrays the CPU oracle uses, so a pixel
/// drift between GPU and CPU paths can never originate from a divergent table
/// (only from the per-element decode arithmetic). Each table is rounded up to
/// a multiple of 4 bytes and uploaded as <c>uint[]</c>; the shader extracts
/// individual bytes via <c>(u >> ((idx &amp; 3) * 8)) &amp; 0xFFu</c>.
/// </remarks>
internal sealed class Iq2Codebooks : IDisposable
{
    public VulkanDevice.Buffer Iq2XxsGrid { get; }
    public VulkanDevice.Buffer Iq2XsGrid { get; }
    public VulkanDevice.Buffer Iq2SGrid { get; }
    public VulkanDevice.Buffer Ksigns { get; }

    private bool _disposed;

    private Iq2Codebooks(VulkanDevice.Buffer xxs, VulkanDevice.Buffer xs, VulkanDevice.Buffer s, VulkanDevice.Buffer ksigns)
    {
        Iq2XxsGrid = xxs;
        Iq2XsGrid = xs;
        Iq2SGrid = s;
        Ksigns = ksigns;
    }

    public static Iq2Codebooks Create(VulkanDevice device)
    {
        var xxs = UploadTable(device, Dequantize.Iq2XxsGrid);
        var xs  = UploadTable(device, Dequantize.Iq2XsGrid);
        var s   = UploadTable(device, Dequantize.Iq2SGrid);
        var k   = UploadTable(device, Dequantize.KsignsIq2Xs);
        return new Iq2Codebooks(xxs, xs, s, k);
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
        Iq2SGrid.Dispose();
        Iq2XsGrid.Dispose();
        Iq2XxsGrid.Dispose();
    }
}
