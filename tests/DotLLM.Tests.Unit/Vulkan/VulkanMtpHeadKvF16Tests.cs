using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1141: the Vulkan MTP head stores its K/V in F16 (the trunk's own default), written by the trunk's F32 to F16 convert kernel
/// and read in place by the <c>_kvf16</c> attention twins. F32 stays only as the reference arm. The absorbed rows are a pure
/// function of the trunk hidden rows, so an F16 head's stored rows must equal the F32 head's rounded to F16 bit for bit; a draft
/// step reads the stored rows, so its logits move only by the F16 rounding of the history.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMtpHeadKvF16Tests : IDisposable
{
    private static readonly int KvStride =
        SyntheticQwen35HybridDenseMtpGguf.NumKvHeads * SyntheticQwen35HybridDenseMtpGguf.HeadDim;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-mtp-f16-{Guid.NewGuid():N}");

    public VulkanMtpHeadKvF16Tests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private sealed record Run(float[] Keys, float[] Values, float[] DraftLogits, long F16Writes, long AllocatedBytes, int Positions);

    private unsafe Run Drive(string path, string spvDir, VulkanKvStorage storage)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);
        using var state = new VulkanMtpState(device, config.HiddenSize, KvStride / config.HeadDim, config.HeadDim, 4096, storage);
        Assert.Equal(storage, state.Storage);

        int p = 0;
        foreach (int[] batch in new[] { new[] { 1, 2, 3 }, new[] { 4, 5, 6, 7 } })
        {
            int[] pos = Enumerable.Range(p, batch.Length).ToArray();
            using (ITensor _ = model.Forward(batch, pos, deviceId: -1, kv, adapter: null, state)) { }
            p += batch.Length;
        }
        // Draft three speculative slots, then verify over the same positions: the absorb overwrites them.
        for (int i = 0; i < 3; i++)
            using (ITensor _ = model.ForwardMtp(state, 9 - i, p + i)) { }
        int[] verify = [8, 10, 11];
        using (ITensor _ = model.Forward(verify, Enumerable.Range(p, verify.Length).ToArray(), deviceId: -1, kv, adapter: null, state)) { }
        p += verify.Length;
        Assert.Equal(p, state.CurrentLength);

        // One more draft step reads the whole committed history from the head cache.
        float[] logits;
        using (ITensor draft = model.ForwardMtp(state, 9, p))
            logits = new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize).ToArray();

        var keys = Download(device, state.KeyCache, storage, p);
        var values = Download(device, state.ValueCache, storage, p);
        return new Run(keys, values, logits, model.F16KvRowWritesForTest, state.AllocatedBytes, p);
    }

    private static unsafe float[] Download(VulkanDevice device, VulkanDevice.Buffer buffer, VulkanKvStorage storage, int rows)
    {
        int n = rows * KvStride;
        if (storage == VulkanKvStorage.F32)
        {
            var all = new float[buffer.Size / sizeof(float)];
            device.Download(buffer, all);
            return all.AsSpan(0, n).ToArray();
        }
        var raw = new float[buffer.Size / sizeof(float)];
        device.Download(buffer, raw);
        var halves = MemoryMarshal.Cast<float, Half>(raw.AsSpan()).ToArray();
        return halves.AsSpan(0, n).ToArray().Select(h => (float)h).ToArray();
    }

    [SkippableFact]
    public void F16Head_StoresTheF32RowsRoundedToF16_ThroughTheConvertKernel_InHalfTheBytes()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "mtp.gguf"), withMtp: true, mtpHasOwnHeadTensors: true);

        Run f32 = Drive(path, spvDir, VulkanKvStorage.F32);
        Run f16 = Drive(path, spvDir, VulkanKvStorage.F16);

        // The trunk's own F16 cache writes through the same convert kernel in both runs; the F16 head adds K and V dispatches per store.
        Assert.True(f16.F16Writes > f32.F16Writes,
            $"the F16 head must have been written through the F32->F16 convert kernel ({f16.F16Writes} vs {f32.F16Writes} dispatches)");

        // Absorbed rows are a function of the trunk hidden rows alone: F16 storage == Half(F32 storage), bit for bit.
        for (int i = 0; i < f32.Keys.Length; i++)
        {
            Assert.Equal((float)(Half)f32.Keys[i], f16.Keys[i]);
            Assert.Equal((float)(Half)f32.Values[i], f16.Values[i]);
        }

        // Bytes: the K/V half of the state is exactly halved.
        long hidden = SyntheticQwen35HybridDenseMtpGguf.HiddenSize * (long)sizeof(float);
        Assert.Equal((f32.AllocatedBytes - hidden) / 2 + hidden, f16.AllocatedBytes);

        // A draft step reading the F16 history: same argmax, logits within the F16 rounding of the history.
        int argmax32 = Array.IndexOf(f32.DraftLogits, f32.DraftLogits.Max());
        int argmax16 = Array.IndexOf(f16.DraftLogits, f16.DraftLogits.Max());
        Assert.Equal(argmax32, argmax16);
        float maxDiff = 0;
        for (int i = 0; i < f32.DraftLogits.Length; i++)
        {
            Assert.True(float.IsFinite(f16.DraftLogits[i]));
            maxDiff = Math.Max(maxDiff, MathF.Abs(f32.DraftLogits[i] - f16.DraftLogits[i]));
        }
        Assert.True(maxDiff <= 5e-2f, $"draft logits drift {maxDiff:E3} exceeds the F16-rounding bound");
    }
}
