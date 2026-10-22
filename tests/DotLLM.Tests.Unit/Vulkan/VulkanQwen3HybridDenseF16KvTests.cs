using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1102: the Vulkan dense hybrid (Bonsai 2) can keep its KV as F16, half the bytes of the F32 default, and every attention
/// path the forward reaches reads the halves in place. These tests drive the whole model on the synthetic fixture: the kernel-level
/// bit parity lives in <see cref="VulkanF16KvAttentionKernelTests"/>.
/// </summary>
/// <remarks>
/// <para>The fixture is widened to a non-degenerate GQA shape (8 query heads over 4 KV heads; the default fixture's single KV head
/// would hide a wrong head broadcast) and run at two widths: head_dim 16 (F32 projections) and head_dim 256 with PQ2_0
/// projections (Bonsai 2's head width and weight type).</para>
/// <para>The schedule reaches every attention kernel the model dispatches: a short prefill, decode steps below the split-KV
/// threshold (the per-token kernel), a prefill tile at a position offset whose length is not a multiple of any tile (the flash
/// kernel, coopmat where the device has it), and decode steps past the threshold (split-KV).</para>
/// <para>F16 logits are compared with the F32 cache's at the tolerance the Vulkan hybrid-dense forward is held to against the CPU
/// reference (abs 5e-3 + rel 1e-3); the F16 rounding of K / V must stay inside the error the backend already accepts.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3HybridDenseF16KvTests : IDisposable
{
    private const int Window = 4096;
    private const int Vocab = SyntheticQwen35HybridDenseMtpGguf.VocabSize;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-kv-f16-{Guid.NewGuid():N}");
    private readonly int _savedQuantum = VulkanNemotronHKvCache.LazyGrowthQuantum;

    /// <summary>Creates the scratch directory for the fixture.</summary>
    public VulkanQwen3HybridDenseF16KvTests() => Directory.CreateDirectory(_scratch);

    /// <inheritdoc/>
    public void Dispose()
    {
        VulkanNemotronHKvCache.LazyGrowthQuantum = _savedQuantum;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>The configured storage reaches the caches, at half the bytes per position.</summary>
    [SkippableFact]
    public void ConfigureKvStorage_F16_HalvesTheBytesPerPosition()
    {
        using var f = Open(pq2: false);
        using var f32 = f.Model.CreateKvCache(64);
        long f32Bytes = f.Model.KvBytesPerPosition;
        f.Model.ConfigureKvStorage(VulkanKvStorage.F16);
        using var f16 = f.Model.CreateKvCache(64);

        Assert.Equal(VulkanKvStorage.F32, f32.Storage);
        Assert.Equal(VulkanKvStorage.F16, f16.Storage);
        Assert.Equal(f32.BytesPerPosition, 2 * f16.BytesPerPosition);
        Assert.Equal(f32Bytes, 2 * f.Model.KvBytesPerPosition);
        Assert.Equal(f16.BytesPerPosition, f.Model.KvBytesPerPosition);
    }

    /// <summary>
    /// The same schedule through an F16 and an F32 cache: every step's logits agree within the backend's tolerance, they are not
    /// identical (the F16 rows really are narrower), and the F16 rows were written by the convert kernel.
    /// </summary>
    /// <param name="pq2">Bonsai 2's shape (head_dim 256, PQ2_0 projections) instead of the small F32 fixture.</param>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void F16Cache_PrefillAndDecode_TrackTheF32Cache(bool pq2)
    {
        using var f = Open(pq2);
        var want = RunSchedule(f, VulkanKvStorage.F32);
        var got = RunSchedule(f, VulkanKvStorage.F16);

        Assert.Equal(want.Count, got.Count);
        float worst = 0f;
        for (int s = 0; s < want.Count; s++)
        {
            for (int i = 0; i < Vocab; i++)
            {
                float diff = MathF.Abs(want[s][i] - got[s][i]);
                worst = MathF.Max(worst, diff);
                Assert.True(diff <= AbsTol + RelTol * MathF.Abs(want[s][i]),
                    $"step {s} logits[{i}]: F32 cache {want[s][i]:R} vs F16 cache {got[s][i]:R}");
            }
        }

        Assert.True(worst > 0f, "the F16 cache produced the F32 cache's logits bit for bit: it is not storing F16");
        Assert.True(f.Model.F16KvRowWritesForTest > 0, "the F16 rows must be written by the convert kernel");
    }

    /// <summary>A prefix snapshot of an F16 cache, taken before a growth, restores exactly afterwards.</summary>
    [SkippableFact]
    public void F16Snapshot_TakenBeforeAGrowth_RestoresExactly()
    {
        using var f = Open(pq2: false);
        f.Model.ConfigureKvStorage(VulkanKvStorage.F16);
        VulkanNemotronHKvCache.LazyGrowthQuantum = 16;
        int[] prefix = Tokens(20, salt: 1);
        int[] longTail = Tokens(60, salt: 2);
        int[] suffix = Tokens(7, salt: 3);

        f.Model.ResetSequenceState();
        using var gdn = f.Model.CreateGdnStateCache();
        using var kv = f.Model.CreateKvCache(prefix.Length);
        Assert.Equal(VulkanKvStorage.F16, kv.Storage);
        Forward(f, prefix, 0, kv, gdn);
        using var snapshot = f.Model.SnapshotSequencePrefix(kv, gdn, prefix.Length);
        Assert.NotNull(snapshot);
        float[] want = Forward(f, suffix, prefix.Length, kv, gdn);

        kv.Rollback(prefix.Length);
        Forward(f, longTail, prefix.Length, kv, gdn);
        f.Model.RestoreSequencePrefix(snapshot!, kv, gdn);
        float[] got = Forward(f, suffix, prefix.Length, kv, gdn);

        for (int i = 0; i < Vocab; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i]),
                $"logits[{i}]: before growth {want[i]:R} vs restored after growth {got[i]:R}");
        }
    }

    /// <summary>An F16 cache that grows across forwards gives the logits of one allocated in a single step, bit for bit.</summary>
    [SkippableFact]
    public void F16Cache_GrowingAcrossForwards_MatchesOneStep()
    {
        using var f = Open(pq2: false);
        VulkanNemotronHKvCache.LazyGrowthQuantum = Window;
        var oneStep = RunSchedule(f, VulkanKvStorage.F16);
        VulkanNemotronHKvCache.LazyGrowthQuantum = 16;
        var grown = RunSchedule(f, VulkanKvStorage.F16);

        for (int s = 0; s < oneStep.Count; s++)
        {
            for (int i = 0; i < Vocab; i++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(oneStep[s][i]) == BitConverter.SingleToInt32Bits(grown[s][i]),
                    $"step {s} logits[{i}]: one-step {oneStep[s][i]:R} vs grown {grown[s][i]:R}");
            }
        }
    }

    /// <summary>
    /// The scheduler's <c>ForwardBatch</c> over two F16 sequences gives each the logits its own serial <c>Forward</c> run does, bit for
    /// bit: the batched path writes and reads each sequence's F16 cache with the same kernels.
    /// </summary>
    [SkippableFact]
    public void ForwardBatch_TwoF16Sequences_MatchSerialForward()
    {
        using var f = Open(pq2: true);
        f.Model.ConfigureKvStorage(VulkanKvStorage.F16);
        int[][] prompts = [Tokens(23, salt: 7), Tokens(9, salt: 8)];
        const int steps = 20;   // past the split-KV threshold for both

        var serial = new List<float[]>[prompts.Length];
        for (int s = 0; s < prompts.Length; s++)
        {
            using var gdn = f.Model.CreateGdnStateCache();
            using var kv = f.Model.CreateKvCache(prompts[s].Length + steps);
            serial[s] = [Forward(f, prompts[s], 0, kv, gdn)];
            for (int t = 0; t < steps; t++)
                serial[s].Add(Forward(f, [Argmax(serial[s][^1])], prompts[s].Length + t, kv, gdn));
        }

        var gdns = prompts.Select(_ => f.Model.CreateGdnStateCache()).ToArray();
        var kvs = prompts.Select(p => f.Model.CreateKvCache(p.Length + steps)).ToArray();
        try
        {
            var batched = prompts.Select(_ => new List<float[]>()).ToArray();
            for (int t = 0; t <= steps; t++)
            {
                var requests = new Core.Models.SequenceForwardRequest[prompts.Length];
                for (int s = 0; s < prompts.Length; s++)
                {
                    int[] ids = t == 0 ? prompts[s] : [Argmax(batched[s][^1])];
                    int start = t == 0 ? 0 : prompts[s].Length + t - 1;
                    requests[s] = new Core.Models.SequenceForwardRequest
                    {
                        TokenIds = ids.AsMemory(),
                        Positions = Enumerable.Range(start, ids.Length).ToArray().AsMemory(),
                        KvCache = kvs[s],
                        GdnState = gdns[s],
                    };
                }

                var results = f.Model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    for (int s = 0; s < prompts.Length; s++) batched[s].Add(LastRow(results[s]));
                }
                finally { foreach (var r in results) r.Dispose(); }
            }

            for (int s = 0; s < prompts.Length; s++)
            {
                for (int t = 0; t <= steps; t++)
                {
                    for (int i = 0; i < Vocab; i++)
                    {
                        Assert.True(BitConverter.SingleToInt32Bits(serial[s][t][i]) == BitConverter.SingleToInt32Bits(batched[s][t][i]),
                            $"sequence {s} step {t} logits[{i}]: serial {serial[s][t][i]:R} vs batched {batched[s][t][i]:R}");
                    }
                }
            }
        }
        finally
        {
            foreach (var kv in kvs) kv.Dispose();
            foreach (var g in gdns) g.Dispose();
        }
    }

    /// <summary>q8_0 and mixed K / V requests are refused with a message, never silently stored as something else.</summary>
    [Fact]
    public void ParseOptions_RefusesWhatTheCacheCannotStore()
    {
        Assert.Equal(VulkanKvStorage.F32, VulkanKvStorageExtensions.ParseOptions(null, null));
        Assert.Equal(VulkanKvStorage.F16, VulkanKvStorageExtensions.ParseOptions("f16", null));
        Assert.Equal(VulkanKvStorage.F16, VulkanKvStorageExtensions.ParseOptions(null, "FP16"));
        Assert.Equal(VulkanKvStorage.F16, VulkanKvStorageExtensions.ParseOptions("f16", "f16"));
        Assert.Equal(VulkanKvStorage.F32, VulkanKvStorageExtensions.ParseOptions("f32", ""));
        Assert.Throws<NotSupportedException>(() => VulkanKvStorageExtensions.ParseOptions("f16", "f32"));
        Assert.Throws<NotSupportedException>(() => VulkanKvStorageExtensions.ParseOptions("q4_0", null));
        Assert.Throws<NotSupportedException>(() => VulkanKvStorageExtensions.ParseOptions("tq4", "tq4"));
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

    private Fixture Open(bool pq2)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, $"m{(pq2 ? "-pq2" : "")}.gguf"), withMtp: false,
            contextLength: Window, blockCount: 6, pq2_0Projections: pq2, attentionHeads: 8, kvHeads: 4);
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(8, config.NumAttentionHeads);
        Assert.Equal(4, config.NumKvHeads);
        var device = VulkanDevice.Create();
        var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        return new Fixture(gguf, device, model);
    }

    /// <summary>
    /// Prefill 5, decode 8 steps (KV 6..13: below the split-KV threshold, the per-token kernel), prefill 37 at position 13 (a flash
    /// tile at an offset, no tile multiple), then decode 25 steps (KV 51..75: split-KV). Returns each forward's last-row logits.
    /// </summary>
    private static List<float[]> RunSchedule(Fixture f, VulkanKvStorage storage)
    {
        f.Model.ConfigureKvStorage(storage);
        f.Model.ResetSequenceState();
        using var gdn = f.Model.CreateGdnStateCache();
        using var kv = f.Model.CreateKvCache(8);
        Assert.Equal(storage, kv.Storage);
        var chunks = new List<int[]> { Tokens(5, salt: 4) };
        chunks.AddRange(Enumerable.Range(0, 8).Select(i => Tokens(1, salt: 20 + i)));
        chunks.Add(Tokens(37, salt: 5));
        chunks.AddRange(Enumerable.Range(0, 25).Select(i => Tokens(1, salt: 40 + i)));

        var rows = new List<float[]>();
        int pos = 0;
        foreach (int[] chunk in chunks)
        {
            rows.Add(Forward(f, chunk, pos, kv, gdn));
            pos += chunk.Length;
        }

        f.Model.ConfigureKvStorage(VulkanKvStorage.F32);
        return rows;
    }

    private static unsafe float[] Forward(Fixture f, int[] tokens, int start, VulkanNemotronHKvCache kv, VulkanGdnStateCache gdn)
    {
        Assert.True(kv.TryGrowTo(start + tokens.Length));
        int[] positions = Enumerable.Range(start, tokens.Length).ToArray();
        using ITensor logits = f.Model.Forward(tokens, positions, deviceId: -1, kv, gdn, mtpState: null);
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * Vocab, Vocab).ToArray();
    }

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (long)(rows - 1) * Vocab, Vocab).ToArray();
    }

    private static int Argmax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>Deterministic token ids in [1, Vocab - 2] (never EOS).</summary>
    private static int[] Tokens(int n, int salt)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 1 + ((i * 7) + (salt * 5) + (i / 3)) % (Vocab - 2);
        return t;
    }
}
