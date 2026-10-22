using System.Globalization;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1100: the Vulkan dense hybrid (Bonsai 2) KV must grow, or a growing chat session loses its recurrent-prefix hit
/// on the second turn. The cache is lazy: <see cref="IKvCache.TryGrowTo"/> raises the logical limit and the forward grows
/// the device buffers (copying the kept rows) before it records.
/// </summary>
/// <remarks>
/// <para>Uses the tiny F32 <see cref="SyntheticQwen35HybridDenseMtpGguf"/> fixture (layer 0 GDN, layer 1 attention) with a
/// 4096-token window, and shrinks <see cref="VulkanNemotronHKvCache.LazyGrowthQuantum"/> so a few dozen tokens cross several
/// growths.</para>
/// <para>Mutants: a <c>TryGrowTo</c> that refuses (the old default) fails every test; a growth that does not drop the model's
/// descriptor caches fails the invalidation check on the decode steps that grow the KV. That mutant leaves the logits intact on
/// the Arc driver, which does not recycle a freed handle soon enough for a stale set to bind it, hence the explicit check.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseKvGrowthTests : IDisposable
{
    private const int Window = 4096;
    private const int Vocab = SyntheticQwen35HybridDenseMtpGguf.VocabSize;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-kv-grow-{Guid.NewGuid():N}");
    private readonly int _savedQuantum = VulkanNemotronHKvCache.LazyGrowthQuantum;

    /// <summary>Creates the scratch directory for the fixture.</summary>
    public VulkanQwen3HybridDenseKvGrowthTests() => Directory.CreateDirectory(_scratch);

    /// <inheritdoc/>
    public void Dispose()
    {
        VulkanNemotronHKvCache.LazyGrowthQuantum = _savedQuantum;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>The model's KV starts empty and <c>TryGrowTo</c> only raises its limit, capped at the window.</summary>
    [SkippableFact]
    public void CreateKvCache_IsLazy_AndTryGrowToRaisesTheLimitUpToTheWindow()
    {
        using var f = Open();
        using var kv = f.Model.CreateKvCache(12);
        Assert.Equal(12, kv.MaxLength);
        Assert.Equal(0, kv.AllocatedBytes);

        Assert.True(kv.TryGrowTo(3000));
        Assert.Equal(3000, kv.MaxLength);
        Assert.Equal(0, kv.AllocatedBytes);     // nothing is allocated until a forward needs the rows

        Assert.True(kv.TryGrowTo(100));          // never shrinks
        Assert.Equal(3000, kv.MaxLength);
        Assert.False(kv.TryGrowTo(Window + 1));
        Assert.Equal(3000, kv.MaxLength);
    }

    /// <summary>
    /// The same token schedule through a KV that grows four times and through one allocated in a single step gives
    /// bit-identical logits at every step: the rows survive each growth and no stale descriptor set is bound.
    /// </summary>
    [SkippableFact]
    public void GrowingKv_AcrossForwards_MatchesAKvAllocatedInOneStep()
    {
        using var f = Open();
        VulkanNemotronHKvCache.LazyGrowthQuantum = Window;
        var oneStep = RunSchedule(f, initialLength: 12);
        VulkanNemotronHKvCache.LazyGrowthQuantum = 16;
        var grown = RunSchedule(f, initialLength: 12, expectGrowths: true);

        Assert.Equal(oneStep.Count, grown.Count);
        for (int s = 0; s < oneStep.Count; s++)
        {
            for (int i = 0; i < Vocab; i++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(oneStep[s][i]) == BitConverter.SingleToInt32Bits(grown[s][i]),
                    $"step {s} logits[{i}]: one-step {oneStep[s][i]:R} vs grown {grown[s][i]:R}");
            }
        }

        Assert.Contains(grown[^1], v => v != 0f);
    }

    /// <summary>
    /// A prefix snapshot taken before a growth restores exactly afterwards: restoring it and forwarding the same suffix
    /// reproduces the logits of the run that never left the prefix.
    /// </summary>
    [SkippableFact]
    public void SnapshotTakenBeforeAGrowth_RestoresExactlyAfterIt()
    {
        using var f = Open();
        VulkanNemotronHKvCache.LazyGrowthQuantum = 16;
        int[] prefix = Tokens(20, salt: 1);
        int[] longTail = Tokens(60, salt: 2);
        int[] suffix = Tokens(7, salt: 3);

        using var gdn = f.Model.CreateGdnStateCache();
        using var kv = f.Model.CreateKvCache(prefix.Length);
        Forward(f, prefix, 0, kv, gdn);
        using var snapshot = f.Model.SnapshotSequencePrefix(kv, gdn, prefix.Length);
        Assert.NotNull(snapshot);
        float[] want = Forward(f, suffix, prefix.Length, kv, gdn, grow: true);

        // Move on well past the snapshot (several growths), then go back to it.
        kv.Rollback(prefix.Length);
        Forward(f, longTail, prefix.Length, kv, gdn, grow: true);
        f.Model.RestoreSequencePrefix(snapshot!, kv, gdn);
        float[] got = Forward(f, suffix, prefix.Length, kv, gdn, grow: true);

        for (int i = 0; i < Vocab; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                $"logits[{i}]: before growth {want[i]:R} vs restored after growth {got[i]:R}");
        }
    }

    /// <summary>
    /// A growing three-turn session through <see cref="TextGenerator"/> on the Vulkan model hits the recurrent prefix on
    /// every turn after the first, and generates what an uncached generator does.
    /// </summary>
    [SkippableFact]
    public void GrowingSession_ThroughTextGenerator_HitsEveryTurn()
    {
        using var f = Open();
        VulkanNemotronHKvCache.LazyGrowthQuantum = 16;
        var tok = new CsvTokenizer();
        Func<ModelConfig, int, IKvCache> kv = (_, size) => f.Model.CreateKvCache(size);
        var cached = new TextGenerator(f.Model, tok, kv, mtpEnabled: false, recurrentPrefixCache: true);
        var plain = new TextGenerator(f.Model, tok, kv, mtpEnabled: false);
        var opts = new InferenceOptions { Temperature = 0f, MaxTokens = 4 };

        int[] history = [];
        int previousLength = 0;
        for (int turn = 0; turn < 3; turn++)
        {
            // Each turn appends far more than the KV headroom a miss allocates (needed + 256).
            history = [.. history, .. Tokens(turn == 0 ? 40 : 300, salt: 10 + turn)];
            string prompt = string.Join(",", history);
            var want = plain.Generate(prompt, opts);
            var got = cached.Generate(prompt, opts);
            Assert.Equal(want.GeneratedTokenIds, got.GeneratedTokenIds);
            if (turn > 0)
                Assert.Equal(previousLength - 1, got.Timings.CachedTokenCount);
            previousLength = history.Length;
        }
    }

    private sealed class Fixture(GgufFile gguf, VulkanDevice device, VulkanQwen3HybridDenseTransformerModel model) : IDisposable
    {
        public VulkanQwen3HybridDenseTransformerModel Model { get; } = model;

        public void Dispose()
        {
            Model.Dispose();
            device.Dispose();
            gguf.Dispose();
        }
    }

    private Fixture Open()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "m.gguf"), withMtp: false, contextLength: Window, blockCount: 6);
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var device = VulkanDevice.Create();
        var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        return new Fixture(gguf, device, model);
    }

    /// <summary>
    /// A prefill of 10, one of 30, then 30 decode steps, each grown first the way the engine does. With a 16-position quantum the
    /// decode steps cross growths on forwards that resize nothing else, so only the KV epoch can refresh the descriptor caches.
    /// </summary>
    private static List<float[]> RunSchedule(Fixture f, int initialLength, bool expectGrowths = false)
    {
        f.Model.ResetSequenceState();
        using var gdn = f.Model.CreateGdnStateCache();
        using var kv = f.Model.CreateKvCache(initialLength);
        var rows = new List<float[]>();
        int pos = 0;
        long firstAllocation = -1;
        int decodeGrowths = 0;
        foreach (int[] chunk in new[] { Tokens(10, salt: 4), Tokens(30, salt: 5) }.Concat(Enumerable.Range(0, 30).Select(i => Tokens(1, salt: 6 + i))))
        {
            long bytesBefore = kv.AllocatedBytes;
            int invalidationsBefore = f.Model.KvBufferInvalidationsForTest;
            rows.Add(Forward(f, chunk, pos, kv, gdn, grow: true));
            pos += chunk.Length;
            if (firstAllocation < 0) firstAllocation = kv.AllocatedBytes;

            // A decode step that re-created the KV buffers must have dropped the descriptor caches: nothing else resized.
            if (chunk.Length == 1 && bytesBefore > 0 && kv.AllocatedBytes > bytesBefore)
            {
                decodeGrowths++;
                Assert.True(f.Model.KvBufferInvalidationsForTest > invalidationsBefore,
                    $"the decode step at position {pos - 1} grew the KV but kept the descriptor caches");
            }
        }

        if (expectGrowths)
            Assert.True(decodeGrowths > 0, "the decode steps must cross at least one growth");

        if (expectGrowths)
            Assert.True(kv.AllocatedBytes > firstAllocation, "the schedule must cross at least one growth");
        return rows;
    }

    private static unsafe float[] Forward(Fixture f, int[] tokens, int start, VulkanNemotronHKvCache kv, VulkanGdnStateCache gdn, bool grow = false)
    {
        if (grow) Assert.True(kv.TryGrowTo(start + tokens.Length));
        int[] positions = Enumerable.Range(start, tokens.Length).ToArray();
        using ITensor logits = f.Model.Forward(tokens, positions, deviceId: -1, kv, gdn, mtpState: null);
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * Vocab, Vocab).ToArray();
    }

    /// <summary>Deterministic token ids in [1, Vocab - 2] (never EOS).</summary>
    private static int[] Tokens(int n, int salt)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 1 + ((i * 7) + (salt * 5) + (i / 3)) % (Vocab - 2);
        return t;
    }

    /// <summary>Comma-separated token ids; EOS is the last id of the fixture's vocabulary.</summary>
    private sealed class CsvTokenizer : ITokenizer
    {
        public int VocabSize => Vocab;
        public int BosTokenId => 0;
        public int EosTokenId => Vocab - 1;
        public int[] Encode(string text)
            => text.Split(',', StringSplitOptions.RemoveEmptyEntries).Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => Encode(text).Length;
    }
}
