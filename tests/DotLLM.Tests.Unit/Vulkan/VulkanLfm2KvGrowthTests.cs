using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Lfm2Moe;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1100 on the LFM2 Vulkan host: its KV is lazy and growable like the Bonsai one, so a growing session keeps its
/// recurrent-prefix hit. The forward grows the buffers (copying the kept rows) before recording and drops the descriptor caches.
/// </summary>
/// <remarks>
/// <para>Uses the synthetic <see cref="Lfm2TestFixture"/> (#1028: short-conv, attention, short-conv; 16-position window) and
/// shrinks <see cref="VulkanNemotronHKvCache.LazyGrowthQuantum"/> to 4 so a 16-token schedule crosses three growths.</para>
/// <para>Mutants: a <c>TryGrowTo</c> that refuses fails both tests; a growth that keeps the descriptor caches fails the
/// invalidation check (the Arc driver does not recycle a freed handle soon enough for a stale set to change the logits).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanLfm2KvGrowthTests : IDisposable
{
    private const int Window = 16;
    private readonly int _savedQuantum = VulkanNemotronHKvCache.LazyGrowthQuantum;

    /// <inheritdoc/>
    public void Dispose() => VulkanNemotronHKvCache.LazyGrowthQuantum = _savedQuantum;

    /// <summary>The model's KV starts empty and <c>TryGrowTo</c> only raises its limit, capped at the window.</summary>
    [SkippableFact]
    public void CreateKvCache_IsLazy_AndTryGrowToRaisesTheLimitUpToTheWindow()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = new Lfm2TestFixture();
        using var gguf = GgufFile.Open(fixture.Path);
        using var device = VulkanDevice.Create();
        using var model = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);
        Assert.Equal(Window, fixture.Config.MaxSequenceLength);

        using var kv = model.CreateKvCache(4);
        Assert.Equal(4, kv.MaxLength);
        Assert.Equal(0, kv.AllocatedBytes);
        Assert.True(kv.TryGrowTo(12));
        Assert.Equal(12, kv.MaxLength);
        Assert.Equal(0, kv.AllocatedBytes);     // nothing is allocated until a forward needs the rows
        Assert.False(kv.TryGrowTo(Window + 1));
        Assert.Equal(12, kv.MaxLength);

        using var capped = model.CreateKvCache(1000);
        Assert.Equal(Window, capped.MaxLength);  // the initial size is capped at the window
    }

    /// <summary>
    /// The same schedule through a KV that grows three times and one allocated in a single step gives bit-identical logits at
    /// every step: the rows survive each growth, and every growth dropped the descriptor caches.
    /// </summary>
    [SkippableFact]
    public void GrowingKv_AcrossForwards_MatchesAKvAllocatedInOneStep()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = new Lfm2TestFixture();
        using var gguf = GgufFile.Open(fixture.Path);
        using var device = VulkanDevice.Create();
        using var model = VulkanLfm2MoeTransformerModel.BuildFromGguf(device, gguf, fixture.Config, spvDir);

        VulkanNemotronHKvCache.LazyGrowthQuantum = Window;
        var oneStep = RunSchedule(model, initialLength: Window, expectGrowths: false);
        VulkanNemotronHKvCache.LazyGrowthQuantum = 4;
        var grown = RunSchedule(model, initialLength: 3, expectGrowths: true);

        Assert.Equal(oneStep.Count, grown.Count);
        for (int s = 0; s < oneStep.Count; s++)
        {
            Assert.Equal(oneStep[s].Length, grown[s].Length);
            for (int i = 0; i < oneStep[s].Length; i++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(oneStep[s][i]) == BitConverter.SingleToInt32Bits(grown[s][i]),
                    $"step {s} logits[{i}]: one-step {oneStep[s][i]:R} vs grown {grown[s][i]:R}");
            }
        }

        Assert.Contains(grown[^1], v => v != 0f);
    }

    /// <summary>
    /// A prefill of 3 then 13 decode steps up to the window, each grown first the way the engine does (<c>TryGrowTo</c>). A cache
    /// created at the full window allocates once; one created at 3 grows on decode steps. A decode step that re-created the KV
    /// buffers must have dropped the descriptor caches.
    /// </summary>
    private static List<float[]> RunSchedule(VulkanLfm2MoeTransformerModel model, int initialLength, bool expectGrowths)
    {
        IModel api = model;
        using var state = api.CreateSequenceState();
        using var kv = model.CreateKvCache(initialLength);
        var rows = new List<float[]>();
        int decodeGrowths = 0;

        Assert.True(kv.TryGrowTo(3));
        using (var prefill = model.Forward([1, 2, 3], [0, 1, 2], 0, kv, (ISsmState)state!))
            rows.Add(LastRow(prefill));

        for (int pos = 3; pos < Window; pos++)
        {
            Assert.True(kv.TryGrowTo(pos + 1));
            long bytesBefore = kv.AllocatedBytes;
            int invalidationsBefore = model.KvBufferInvalidationsForTest;
            using (var logits = model.Forward([1 + ((pos * 5) % 6)], [pos], 0, kv, (ISsmState)state!))
                rows.Add(LastRow(logits));
            if (kv.AllocatedBytes > bytesBefore)
            {
                decodeGrowths++;
                Assert.True(model.KvBufferInvalidationsForTest > invalidationsBefore,
                    $"the decode step at position {pos} grew the KV but kept the descriptor caches");
            }
        }

        if (expectGrowths)
            Assert.True(decodeGrowths >= 2, $"the decode steps must cross several growths (crossed {decodeGrowths})");
        else
            Assert.Equal(0, decodeGrowths);
        return rows;
    }

    private static float[] LastRow(ITensor logits)
    {
        int vocab = logits.Shape[^1];
        long rows = logits.ElementCount / vocab;
        return new ReadOnlySpan<float>((float*)logits.DataPointer + ((rows - 1) * vocab), vocab).ToArray();
    }
}
