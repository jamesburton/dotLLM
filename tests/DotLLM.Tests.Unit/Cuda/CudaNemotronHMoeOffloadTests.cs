using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1070: <c>nemotron_h_moe</c> with its routed experts offloaded to a host store behind a small VRAM expert cache, held
/// to the same model with every expert resident on the synthetic Bolt / Lightning fixture of
/// <see cref="CudaNemotronHMoeParityTests"/>. Residency must not change the result: the same kernels run with different
/// weight pointers and group order, so the logits are compared bit for bit, prefill and decode.
/// </summary>
/// <remarks>
/// The cache is 1 MiB against about 3 MiB (Bolt shape) / 6 MiB (Lightning shape) of experts, so decode misses and evicts,
/// and a prefill block touches more experts than a pool holds, which streams the rest through the staging arena in waves
/// (one expert per wave when <c>waveExperts</c> is 1).
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHMoeOffloadTests : IDisposable
{
    private const int Vocab = 16;
    private static readonly int[] Prompt = [3, 9, 1, 14, 6, 0, 11];
    private static readonly int[] DecodeTokens = [5, 2, 13, 8, 0, 15, 7, 4];

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaNemotronHMoeOffloadTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-moe-offload-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <param name="latent">true = Bolt shape (LatentMoE); false = Lightning shape.</param>
    /// <param name="blockTokens">MoE token block; 3 splits the prefill into three lookups.</param>
    /// <param name="waveExperts">Experts per staging wave (null = the largest top-k); 1 forces one wave per streamed expert.</param>
    /// <param name="pinned">true = pinned host store; false = memory-mapped store through the pinned staging ring.</param>
    /// <param name="policy">Cache policy.</param>
    [SkippableTheory]
    [InlineData(true, 512, null, true, ExpertCachePolicy.Slru)]
    [InlineData(true, 3, null, true, ExpertCachePolicy.Slru)]
    [InlineData(true, 512, 1, true, ExpertCachePolicy.Lru)]
    [InlineData(true, 512, 1, false, ExpertCachePolicy.Slru)]
    [InlineData(true, 3, 1, false, ExpertCachePolicy.Lfu)]
    [InlineData(false, 512, 1, true, ExpertCachePolicy.Slru)]
    public void Offload_LogitsBitIdenticalToResident_AndCountersMatchOfflineReplay(
        bool latent, int blockTokens, int? waveExperts, bool pinned, ExpertCachePolicy policy)
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: false, f32Banks: false);

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        int? savedWave = CudaNemotronHTransformerModel.MoeArenaWaveExpertsOverride;
        var trace = new List<(int Layer, int[] Ids)>();
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        CudaNemotronHTransformerModel.MoeArenaWaveExpertsOverride = waveExperts;
        // Hold the copy stream back (~10 ms per copy batch at PCIe 2.0) so a missing compute-stream wait on the copy event
        // makes the kernels read stale slots; the small synthetic experts otherwise always land first.
        CudaNemotronHTransformerModel.MoeCopyStreamDelayBytesForTests = 64L << 20;
        try
        {
            CudaNemotronHTransformerModel.MoePathCounters.Reset();
            float[] resident = Run(path, ptxDir!, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, out _, out _);
            long residentLayerCalls = Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.LayerCalls);

            CudaNemotronHTransformerModel.MoePathCounters.Reset();
            CudaNemotronHTransformerModel.MoeLookupTraceHook = (layer, ids) => trace.Add((layer, ids));
            var options = new MoeOffloadOptions
            {
                Mode = MoeOffloadMode.On,
                CacheMib = 1,
                Policy = policy,
                PinBudgetFraction = pinned ? MoeOffloadOptions.DefaultPinBudgetFraction : 0,
                Log = _output.WriteLine,
            };
            float[] offload = Run(path, ptxDir!, options, out var info, out var counters);
            long groups = Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.Groups) / 2;   // up + down per group

            Assert.NotNull(info);
            _output.WriteLine($"latent={latent} block={blockTokens} wave={(waveExperts is { } we ? we.ToString(System.Globalization.CultureInfo.InvariantCulture) : "auto")} pinned={pinned} {policy}: " +
                $"{info!.CacheSlots} slots / {info.Pools} pools ({info.CacheBytes} B), hits {counters.Hits}, misses {counters.Misses}, " +
                $"evictions {counters.Evictions}, bypasses {counters.Bypasses}, lookups {trace.Count}");

            // The offloaded run took the offload path, with the store this case asked for, inside the requested size.
            Assert.Equal(pinned, info.PinnedStore);
            Assert.InRange(info.CacheBytes, 1, 1L << 20);
            Assert.Equal(residentLayerCalls, Interlocked.Read(ref CudaNemotronHTransformerModel.MoePathCounters.LayerCalls));

            // Bit-identical logits, prefill and every decode step.
            Assert.Equal(resident.Length, offload.Length);
            for (int i = 0; i < resident.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(resident[i]) == BitConverter.SingleToInt32Bits(offload[i]),
                    $"logit {i} (row {i / Vocab}): resident {resident[i]:R} != offload {offload[i]:R}");

            // Counters: one lookup per routed expert per layer call; the small cache must miss, evict and (prefill) bypass.
            Assert.Equal(groups, counters.Hits + counters.Misses);
            Assert.Equal(counters.Misses, counters.Insertions + counters.Bypasses);
            Assert.True(counters.Misses > 0, "no misses: the cache is not smaller than the experts");
            Assert.True(counters.Evictions > 0, "no evictions: the cache never filled");
            if (blockTokens >= Prompt.Length)
                Assert.True(counters.Bypasses > 0, "the prefill block never streamed an expert through the staging arena");

            // The counters agree with an offline replay of the same routing trace through the same cache shape.
            var replay = new ExpertCache(counters.ExpertsPerLayer, counters.PoolOfLayer, counters.SlotsPerPool, policy);
            foreach (var (layer, ids) in trace)
                replay.Lookup(layer, ids, new int[ids.Length], new ExpertLookupResult[ids.Length]);
            Assert.Equal(counters.Hits, replay.Hits);
            Assert.Equal(counters.Misses, replay.Misses);
            Assert.Equal(counters.Evictions, replay.Evictions);
            Assert.Equal(counters.Bypasses, replay.Bypasses);
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
            CudaNemotronHTransformerModel.MoeArenaWaveExpertsOverride = savedWave;
            CudaNemotronHTransformerModel.MoeCopyStreamDelayBytesForTests = 0;
            CudaNemotronHTransformerModel.MoeLookupTraceHook = null;
        }
    }

    [SkippableFact]
    public void Budget_CacheWithinTheRequest_FreeVramAboveTheReserve_AutoAndOffStayResidentWhenTheModelFits()
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent: true, zeroBias: false, f32Banks: false);

        using (var gguf = GgufFile.Open(path))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

            // The fixture fits any GPU: auto and off keep every expert resident.
            foreach (var mode in new[] { MoeOffloadMode.Auto, MoeOffloadMode.Off })
            {
                var (m, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir, new MoeOffloadOptions { Mode = mode });
                using (m) Assert.Null(Assert.IsType<CudaNemotronHTransformerModel>(m).MoeOffload);
            }

            // On, with a 1 MiB request: the cache stays within it and the device keeps the reserve free after the load.
            var (model, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir,
                new MoeOffloadOptions { Mode = MoeOffloadMode.On, CacheMib = 1, Log = _output.WriteLine });
            using (model)
            {
                var info = Assert.IsType<CudaNemotronHTransformerModel>(model).MoeOffload!;
                Assert.InRange(info.CacheBytes, 1, 1L << 20);
                Assert.False(info.RequestClamped);
                Assert.True(info.FreeBytesAfterLoad >= info.ReserveBytes, $"free after load {info.FreeBytesAfterLoad} < reserve {info.ReserveBytes}");
                CudaDriverApi.cuMemGetInfo_v2(out nuint free, out _).ThrowOnError();
                Assert.True((long)free >= info.ReserveBytes, $"free now {(long)free} < reserve {info.ReserveBytes}");
                Assert.Equal(0, info.SpillRetries);
            }

            // A request far above the measured budget is clamped (with a log line), never honoured.
            var log = new List<string>();
            var (big, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir,
                new MoeOffloadOptions { Mode = MoeOffloadMode.On, CacheMib = 1L << 30, Log = log.Add });
            using (big)
            {
                var info = Assert.IsType<CudaNemotronHTransformerModel>(big).MoeOffload!;
                Assert.True(info.RequestClamped);
                Assert.True(info.CacheBytes <= info.BudgetBytes);
                Assert.Contains(log, l => l.Contains("exceeds the measured VRAM budget", StringComparison.Ordinal));
                foreach (var l in log) _output.WriteLine(l);
            }
        }
    }

    /// <summary>The cache counters and shape after a run (read before the model is disposed).</summary>
    private sealed record CacheCounters(long Hits, long Misses, long Evictions, long Insertions, long Bypasses,
        int[] ExpertsPerLayer, int[] PoolOfLayer, int[] SlotsPerPool);

    /// <summary>Prompt prefill then <see cref="DecodeTokens"/> one at a time; rows = prompt rows + one per decode step.</summary>
    private static unsafe float[] Run(string path, string ptxDir, MoeOffloadOptions options,
        out CudaMoeOffloadInfo? info, out CacheCounters counters)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir, options);
        using (model)
        {
            var cuda = Assert.IsType<CudaNemotronHTransformerModel>(model);
            var rows = new List<float>();
            using (var logits = model.Forward(Prompt, [.. Enumerable.Range(0, Prompt.Length)], deviceId: 0))
                rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * Vocab).ToArray());
            for (int s = 0; s < DecodeTokens.Length; s++)
            {
                using var logits = model.Forward([DecodeTokens[s]], [Prompt.Length + s], deviceId: 0);
                rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Vocab).ToArray());
            }

            info = cuda.MoeOffload;
            counters = Snapshot(cuda.MoeExpertCache, config.NumLayers);
            return [.. rows];
        }
    }

    private static CacheCounters Snapshot(ExpertCache? c, int layers)
    {
        if (c is null) return new CacheCounters(0, 0, 0, 0, 0, [], [], []);
        int[] experts = [.. Enumerable.Range(0, layers).Select(c.ExpertsIn)];
        int[] poolOf = [.. Enumerable.Range(0, layers).Select(l => Math.Max(c.PoolOf(l), 0))];
        int[] slots = [.. Enumerable.Range(0, c.Pools).Select(c.SlotsIn)];
        return new CacheCounters(c.Hits, c.Misses, c.Evictions, c.Insertions, c.Bypasses, experts, poolOf, slots);
    }
}
