using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Models.Edge0;

namespace DotLLM.Vulkan;

/// <summary>The Edge0 MTP head's routed experts on the device (G6, #1990): <c>mtp/experts.bin</c>, 512 planar Q2_0 blobs, imported from the mapped file (or copied once when the import is refused).</summary>
internal sealed class Edge0MtpExperts : IDisposable
{
    /// <summary>The blob array (<c>512 * Edge0MtpPack.ExpertBlobBytes</c> bytes).</summary>
    public required VulkanDevice.Buffer Blobs { get; init; }

    /// <summary>True when <see cref="Blobs"/> aliases the mapped pages (no copy).</summary>
    public required bool Imported { get; init; }

    /// <summary>Imports (UMA) or copies the head's expert blobs.</summary>
    public static Edge0MtpExperts Upload(VulkanDevice device, VulkanStagingBuffer staging, DotLLM.Models.Edge0.Edge0MtpPack pack)
    {
        long bytes = DotLLM.Models.Edge0.Edge0MtpPack.Experts * DotLLM.Models.Edge0.Edge0MtpPack.ExpertBlobBytes;
        bool imported = VulkanWeightImportPolicy.TryImport(device, pack.ExpertsPointer, bytes, out var buf);
        if (!imported)
        {
            buf = device.AllocateDeviceLocal(bytes);
            try { staging.UploadBytes(pack.ExpertsPointer, bytes, buf); staging.WaitAll(); }
            catch { buf.Dispose(); throw; }
            VulkanWeightImportPolicy.NoteStaged(pack.ExpertsPointer, bytes);
        }
        return new Edge0MtpExperts { Blobs = buf!, Imported = imported };
    }

    /// <inheritdoc/>
    public void Dispose() => Blobs.Dispose();
}

/// <summary>
/// Device buffers of one Edge0 latent-MoE layer (G3, #1981). The routed experts are split from the pack's fixed 1,630,720 B records into three
/// contiguous per-layer banks (gate Q3_K, up Q3_K, down IQ4_NL) so the ordinary resident-bank kernels read them in place; D (all groups, one
/// matrix) and U (one matrix per group) stay in their on-disk Q8_0 / Q4_0 blocks. Nothing is widened.
/// </summary>
internal sealed class Edge0LayerBuffers : IDisposable
{
    /// <summary>BF16 router <c>[experts, hidden]</c> (owned by the bundle's <c>Gate</c>).</summary>
    public required VulkanDevice.Buffer Router { get; init; }
    /// <summary>The layer's expert records exactly as the pack stores them (<c>experts * RecordBytes</c> bytes of gate | up | down), imported from the mapped file or, when the import is
    /// refused, copied contiguously. Owned by the bundle (its <c>W1Bank</c>, <c>W2Bank</c> and <c>W3Bank</c> are all this one buffer).</summary>
    public required VulkanDevice.Buffer Records { get; init; }
    /// <summary>True when <see cref="Records"/> aliases the mapped pack pages (no copy at all).</summary>
    public required bool RecordsImported { get; init; }
    /// <summary>Bytes of one expert record (the byte stride of every projection bank).</summary>
    public required long RecordBytes { get; init; }
    /// <summary>Bytes of one gate or up projection; up starts at this offset in a record, down at twice it.</summary>
    public required long GateUpBytes { get; init; }
    /// <summary>D of all groups as one matrix <c>[groups * latent, hidden]</c> (group g = rows g*latent ..).</summary>
    public required VulkanDevice.Buffer D { get; init; }
    /// <summary>U per group <c>[hidden, latent]</c>.</summary>
    public required VulkanDevice.Buffer[] U { get; init; }
    /// <summary>int32 <c>[experts]</c>: the D/U group of each expert.</summary>
    public required VulkanDevice.Buffer Labels { get; init; }
    /// <summary>F32 <c>[latent]</c> inverse column scale (ones when the pack has none).</summary>
    public required VulkanDevice.Buffer InverseScale { get; init; }
    public required QuantizationType GateUpType { get; init; }
    public required QuantizationType DownType { get; init; }
    public required QuantizationType LatentDenseType { get; init; }
    public required int Groups { get; init; }
    public required int Latent { get; init; }
    public required int Hidden { get; init; }
    public required int Experts { get; init; }
    /// <summary>Device bytes of everything this layer uploaded (banks, router, D, U, small vectors).</summary>
    public required long Bytes { get; init; }

    /// <summary>Uploads layer <paramref name="layer"/> of <paramref name="pack"/>. <paramref name="routerBf16"/> is the skeleton's BF16 <c>ffn_gate_inp</c> view.</summary>
    public static unsafe Edge0LayerBuffers Upload(VulkanDevice device, VulkanStagingBuffer staging, Edge0Pack pack, int layer, nint routerBf16)
    {
        int E = pack.Experts, H = pack.Hidden, L = pack.Latent, F = pack.ExpertFf, G = pack.Groups;
        if (pack.GateUpType != QuantizationType.Q3_K || pack.DownType != QuantizationType.IQ4_NL)
            throw new NotSupportedException($"Edge0 Vulkan banks support Q3_K gate/up and IQ4_NL down, not {pack.GateUpType}/{pack.DownType}.");
        if (G != 8) throw new NotSupportedException($"The Edge0 Vulkan latent step sums eight groups, the pack has {G}.");
        var created = new List<VulkanDevice.Buffer>();
        VulkanDevice.Buffer Alloc(long bytes) { var b = device.AllocateDeviceLocal(bytes); created.Add(b); return b; }
        try
        {
            long gateBytes = pack.GateUpBytes, downBytes = pack.DownBytes, record = pack.ExpertRecordBytes;
            var router = Alloc((long)E * H * 2);
            staging.UploadBytes(routerBf16, (long)E * H * 2, router);
            // The records: imported in place from the mapped pack (UMA, shared writable view), else ONE contiguous copy (never split: the strided kernels read records).
            long recBytes = (long)E * record;
            nint rec0 = pack.GateBank(layer);
            bool imported = VulkanWeightImportPolicy.TryImport(device, rec0, recBytes, out var recBuf);
            if (!imported)
            {
                recBuf = device.AllocateDeviceLocal(recBytes);
                staging.UploadBytes(rec0, recBytes, recBuf);
                VulkanWeightImportPolicy.NoteStaged(rec0, recBytes);
            }
            created.Add(recBuf!);

            var dBuf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging, pack.D(layer, 0), pack.LatentDenseType,
                outputDim: G * L, inputDim: H, forceF32: false, out var dQt, out long dBytes);
            created.Add(dBuf);
            var uBufs = new VulkanDevice.Buffer[G];
            long uBytes = 0;
            for (int g = 0; g < G; g++)
            {
                uBufs[g] = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(device, staging, pack.U(layer, g), pack.LatentDenseType,
                    outputDim: H, inputDim: L, forceF32: false, out var uQt, out long b);
                created.Add(uBufs[g]);
                uBytes += b;
                if (uQt != pack.LatentDenseType) throw new NotSupportedException($"Edge0 U was widened to {uQt} on upload; the packed {pack.LatentDenseType} kernel is unavailable.");
            }
            if (dQt != pack.LatentDenseType) throw new NotSupportedException($"Edge0 D was widened to {dQt} on upload; the packed {pack.LatentDenseType} kernel is unavailable.");

            int[] labels = pack.Labels(layer).ToArray();
            var labelBuf = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, MemoryMarshal.Cast<int, float>(labels.AsSpan()).ToArray());
            created.Add(labelBuf);
            float[] inv = new float[L];
            var invSrc = pack.InverseColumnScale(layer);
            if (invSrc.IsEmpty) Array.Fill(inv, 1f); else invSrc.CopyTo(inv);
            var invBuf = VulkanQwen3MoeHybridWeights.UploadFloatArray(device, staging, inv);
            created.Add(invBuf);
            staging.WaitAll();
            return new Edge0LayerBuffers
            {
                Router = router, Records = recBuf!, RecordsImported = imported, RecordBytes = record, GateUpBytes = gateBytes, D = dBuf, U = uBufs, Labels = labelBuf, InverseScale = invBuf,
                GateUpType = pack.GateUpType, DownType = pack.DownType, LatentDenseType = pack.LatentDenseType, Groups = G, Latent = L, Hidden = H, Experts = E,
                Bytes = (long)E * H * 2 + (imported ? 0 : recBytes) + dBytes + uBytes + (long)E * 4 + (long)L * 4,
            };
        }
        catch { foreach (var b in created) b.Dispose(); throw; }
    }

    /// <summary>Releases D, U and the small vectors; the router and the three banks belong to the owning <c>LayerBundle</c>.</summary>
    public void Dispose()
    {
        D.Dispose();
        foreach (var u in U) u.Dispose();
        Labels.Dispose();
        InverseScale.Dispose();
    }
}
