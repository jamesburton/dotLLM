using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.ChunkReuse;
using DotLLM.Tests.Unit.Diagnostics;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA model of one synthetic family for the #1920 tap suite. The decode graphs are switched off for the lifetime of the fixture so that the
/// untapped control of every comparison runs the same eager body as the tapped model (the graph kernels are not bit-identical to it on the dense host);
/// the graph bypass has its own tests below, which switch them back on.
/// </summary>
internal sealed class CudaHiddenStateTapFixture : HiddenStateTapFixture
{
    private readonly string _dir;
    private readonly GgufFile _gguf;
    private readonly ModelConfig _config;
    private readonly bool? _savedGraphOverride;

    public CudaHiddenStateTapFixture(string family, string ptxDir, bool graphs = false, bool withMtp = false)
    {
        Family = family;
        _dir = Path.Combine(Path.GetTempPath(), $"dotllm-taps-cuda-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, "m.gguf");
        switch (family)
        {
            case HiddenStateTapBackendSuite.Llama:
                File.WriteAllBytes(path, SyntheticDenseGguf.Build("llama"));
                Prompt = [3, 9, 14, 5, 21, 8, 30, 2];
                break;
            case HiddenStateTapBackendSuite.HybridDense:
                SyntheticQwen35HybridDenseMtpGguf.Write(path, withMtp: withMtp, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            case HiddenStateTapBackendSuite.MoeHybrid:
                SyntheticQwen35MoeGguf.Write(path, contextLength: 256);
                Prompt = [3, 4, 5, 6, 3, 7, 5, 4];
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(family));
        }
        _savedGraphOverride = CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride;
        CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride = graphs;
        _gguf = GgufFile.Open(path);
        _config = GgufModelConfigExtractor.Extract(_gguf.Metadata);
        Model = family switch
        {
            HiddenStateTapBackendSuite.Llama => LoadDense(path, ptxDir, graphs),
            HiddenStateTapBackendSuite.HybridDense => CudaQwen3HybridDenseTransformerModel.LoadFromGguf(_gguf, _config, deviceId: 0, ptxDir),
            _ => CudaQwen3MoeHybridTransformerModel.LoadFromGguf(_gguf, _config, deviceId: 0, ptxDir),
        };
    }

    private static CudaTransformerModel LoadDense(string path, string ptxDir, bool graphs)
    {
        var (model, gguf, _) = CudaModelLoader.LoadFromGguf(path, deviceId: 0, ptxDir);
        gguf.Dispose();   // the loader uploads every weight; the mapping is not needed afterwards
        model.UseGraphCapture = graphs && model.UseGraphCapture;
        return model;
    }

    public override float[] OutputNormWeight => ReadF32Row(_gguf, "output_norm.weight", 0, _config.HiddenSize);
    public override float NormEps => _config.NormEpsilon;
    public override float[] EmbeddingRow(int token) => ReadF32Row(_gguf, "token_embd.weight", token, _config.HiddenSize);

    /// <summary>The dense CUDA host stores the final-norm output and the residual in F16, so its device rows carry a half rounding; the hybrids are F32.</summary>
    public override float NormRelTol => Family == HiddenStateTapBackendSuite.Llama ? 3e-3f : 2e-5f;
    public override float EmbedRelTol => Family == HiddenStateTapBackendSuite.Llama ? 1e-3f : 1e-6f;

    public string Family { get; }
    public override IModel Model { get; }
    public override int NumLayers => _config.NumLayers;
    public override int Hidden => _config.HiddenSize;
    public override int Vocab => _config.VocabSize;
    public override int[] Prompt { get; }

    /// <summary>
    /// CUDA has no packed-matmul batch. The dense host's tapped ForwardBatch leaves the deferred pinned-logits batch for a per-request loop
    /// (<c>CudaTransformerModel.ForwardBatchTapped</c>) and bumps the counter once (first CUDA run, 2026-11-23); the two hybrid hosts only ever
    /// had the per-request interface-default loop (no override of a packed path), so for them the counter never moves. A ForwardBatchTapped that
    /// stopped bumping fails the Llama arm (expected 1, actual 0).
    /// </summary>
    public override bool ExpectsPackedBatchFallback => Family == HiddenStateTapBackendSuite.Llama;

    /// <summary>The GDN-hybrid MoE host takes an abortable tap only once it can checkpoint its recurrent state (#1919).</summary>
    public override bool SupportsAbortableTaps
        => Family != HiddenStateTapBackendSuite.MoeHybrid || Model.SupportsRecurrentStateCheckpoint;

    /// <summary>An untapped CUDA decode enqueues no tap copy at all (the helper is never entered): baseline zero.</summary>
    public override long[]? DeviceCounters() => [CudaHiddenStateTaps.EnqueuedCopies];
    public override long[]? BaselineDeviceCountersPer16Decodes => [0];

    public override IKvCache NewKv(int maxLen = 96) => Model switch
    {
        CudaTransformerModel m => m.CreateKvCache(maxLen),
        CudaQwen3HybridDenseTransformerModel m => m.CreateKvCache(maxLen),
        CudaQwen3MoeHybridTransformerModel m => m.CreateKvCache(maxLen),
        _ => throw new InvalidOperationException(Model.GetType().Name),
    };

    public override float[] KvBytes(IKvCache kv)
    {
        int rows = kv.CurrentLength;
        switch (Model)
        {
            case CudaTransformerModel:
                var all = new List<float>();
                var cache = (CudaKvCache)kv;
                int elems = rows * _config.NumKvHeads * _config.HeadDim;
                for (int l = 0; l < _config.NumLayers; l++)
                {
                    var k = new float[elems]; var v = new float[elems];
                    cache.DownloadLayer(l, k, v);
                    all.AddRange(k); all.AddRange(v);
                }
                return [.. all];
            case CudaQwen3HybridDenseTransformerModel m:
                return m.DebugReadKvRowsForTest(rows);
            case CudaQwen3MoeHybridTransformerModel m:
                return m.DebugReadKvRowsForTest(rows);
            default:
                throw new InvalidOperationException(Model.GetType().Name);
        }
    }

    public override void Dispose()
    {
        Model.Dispose();
        _gguf.Dispose();
        CudaQwen3HybridDenseTransformerModel.DecodeGraphOverride = _savedGraphOverride;
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }
}

/// <summary>The #1920 tap suite on the CUDA backend (all three families), plus the CUDA-specific paths: graph bypass, tiling, CPU parity.</summary>
/// <remarks>
/// Written without a GPU to run it on: every test skips when there is no CUDA device or no PTX. The mutants it is meant to kill
/// (and which test kills each) are listed in <c>.docs/progress-expert-integ.md</c>, marked NOT RUN.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class HiddenStateTapCudaTests : HiddenStateTapBackendSuite
{
    private static readonly HookPoint[] All4 =
        [HookPoint.PostEmbedding, HookPoint.PostLayer, HookPoint.PreFinalNorm, HookPoint.PreLmHead];

    private static string RequireCuda()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        return ptxDir!;
    }

    protected override HiddenStateTapFixture Open(string family) => new CudaHiddenStateTapFixture(family, RequireCuda());

    protected override HiddenStateTapFixture? OpenHybridDenseWithMtp()
        => new CudaHiddenStateTapFixture(HybridDense, RequireCuda(), withMtp: true);

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    private static List<(HookPoint P, int L, int R, int Pos, float[] D)> Capture(HiddenStateTapFixture fx, TapRows rows, bool decodeToo)
    {
        var seen = new List<(HookPoint, int, int, int, float[])>();
        using var h = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, All4, rows),
            s => seen.Add((s.Point, s.LayerIndex, s.RowIndex, s.TokenPosition, s.Data.ToArray())));
        using var kv = fx.NewKv();
        using (fx.Model.Forward(fx.Prompt, Enumerable.Range(0, fx.Prompt.Length).ToArray(), -1, kv)) { }
        if (decodeToo)
            using (fx.Model.Forward([fx.Prompt[0]], [fx.Prompt.Length], -1, kv)) { }
        return seen;
    }

    /// <summary>
    /// Every sample of every point and layer, all rows, agrees with the CPU model of the same weights to the F16-residual tolerance. This is
    /// the test that kills a wrong source buffer (for example reading <c>Residual</c> instead of <c>HiddenState</c> on the last layer): a
    /// perturbation test cannot see it, and PreFinalNorm equals PostLayer(last) in that mutant too.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void TappedRows_MatchTheCpuModel_AtEveryPointAndLayer(string family)
    {
        using var cuda = Open(family);
        using var cpu = new CpuHiddenStateTapFixture(family);
        var got = Capture(cuda, TapRows.All, decodeToo: true);
        var want = Capture(cpu, TapRows.All, decodeToo: true);
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            Assert.Equal((want[i].P, want[i].L, want[i].R, want[i].Pos), (got[i].P, got[i].L, got[i].R, got[i].Pos));
            float scale = want[i].D.Max(MathF.Abs);
            float tol = 0.03f * scale + 1e-3f;
            if (want[i].P == HookPoint.PostLayer && want[i].L > 0)
            {
                // I-9 control arm: the gate must be able to tell PostLayer(L) from PostLayer(L-1). Adjacent residual-stream layers are
                // highly correlated, so measure the CPU gap between them (same row) and tighten the bound to a quarter of it. The bound
                // must still clear the F16 storage floor, otherwise this fixture cannot discriminate and the test says so.
                int prev = Enumerable.Range(0, i).Last(j => want[j].P == HookPoint.PostLayer && want[j].L == want[i].L - 1 && want[j].R == want[i].R);
                float gap = 0;
                for (int j = 0; j < want[i].D.Length; j++) gap = MathF.Max(gap, MathF.Abs(want[i].D[j] - want[prev].D[j]));
                float floor = 2e-3f * scale + 1e-4f;
                tol = MathF.Min(tol, gap / 4);
                Assert.True(tol > floor, $"sample {i} (layer {want[i].L} row {want[i].R}): the CPU gap to the previous layer ({gap}) is too small for the F16 floor ({floor}); the gate cannot tell a wrong layer");
            }
            for (int j = 0; j < want[i].D.Length; j++)
                Assert.True(MathF.Abs(want[i].D[j] - got[i].D[j]) <= tol,
                    $"sample {i} ({want[i].P} layer {want[i].L} row {want[i].R}) element {j}: cpu {want[i].D[j]} cuda {got[i].D[j]} tol {tol}");
        }
    }

    /// <summary>The forced site-by-site delivery (what an abortable tap or an oversized plan uses) is identical to the deferred plan.</summary>
    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void DeferredAndImmediatePlans_DeliverIdenticalSamples(string family)
    {
        using var a = Open(family);
        using var b = Open(family);
        var deferred = Capture(a, TapRows.All, decodeToo: true);
        CudaHiddenStateTaps.ForceImmediateForTest = true;
        List<(HookPoint P, int L, int R, int Pos, float[] D)> immediate;
        try { immediate = Capture(b, TapRows.All, decodeToo: true); }
        finally { CudaHiddenStateTaps.ForceImmediateForTest = false; }
        Assert.Equal(deferred.Count, immediate.Count);
        for (int i = 0; i < deferred.Count; i++)
        {
            Assert.Equal((deferred[i].P, deferred[i].L, deferred[i].R), (immediate[i].P, immediate[i].L, immediate[i].R));
            Assert.True(deferred[i].D.AsSpan().SequenceEqual(immediate[i].D), $"sample {i} differs");
        }
    }

    /// <summary>
    /// With graphs on, an untapped decode step takes the graph and a tapped one does not: the counter moves for the tapped step only, the
    /// samples still arrive, and the logits agree with the untapped graph run to the usual tolerance.
    /// </summary>
    [SkippableTheory]
    [InlineData(HiddenStateTapBackendSuite.Llama)]
    [InlineData(HiddenStateTapBackendSuite.HybridDense)]
    public void DecodeGraph_IsBypassed_WhenATapIsAttached(string family)
    {
        string ptx = RequireCuda();
        // Watchdog: a bypass defect (a graph launched with a tap attached) wedged the test host in a mutant run. Run on a background thread
        // (fixture creation and disposal included, so a stuck stream cannot hang the test thread) and turn a hang into a test failure.
        var body = Task.Factory.StartNew(() => DecodeGraphBypassBody(family, ptx), TaskCreationOptions.LongRunning);
        if (!body.Wait(TimeSpan.FromSeconds(60)))
            Assert.Fail($"{family}: the tapped decode step did not finish within 60 s (decode graph not bypassed with a tap attached?)");
        body.GetAwaiter().GetResult();   // rethrows the body's failure or skip unwrapped
    }

    private static void DecodeGraphBypassBody(string family, string ptx)
    {
        using var fx = new CudaHiddenStateTapFixture(family, ptx, graphs: true);
        using var kv = fx.NewKv();
        int[] pp = Enumerable.Range(0, fx.Prompt.Length).ToArray();
        using (fx.Model.Forward(fx.Prompt, pp, -1, kv)) { }

        // Untapped control: warm one decode step, then confirm the graph is in play at all (else there is nothing to bypass).
        HiddenStateTapRegistry.PathCounters.Reset();
        for (int i = 0; i < 2; i++)
            using (fx.Model.Forward([fx.Prompt[i]], [fx.Prompt.Length + i], -1, kv)) { }
        Assert.Equal(0, HiddenStateTapRegistry.PathCounters.CudaGraphBypassed);
        bool graphInPlay = fx.Model switch
        {
            CudaQwen3HybridDenseTransformerModel m => m.DecodeGraphCounts.Launches > 0,
            CudaTransformerModel m => m.GraphReplayCount > 0,
            _ => false,
        };
        Skip.IfNot(graphInPlay, "the synthetic model does not take the decode graph on this device");

        int rows = 0;
        using var h = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, All4), _ => rows++);
        HiddenStateTapRegistry.PathCounters.Reset();
        using (fx.Model.Forward([fx.Prompt[2]], [fx.Prompt.Length + 2], -1, kv)) { }
        Assert.Equal(1, HiddenStateTapRegistry.PathCounters.CudaGraphBypassed);
        Assert.Equal(3 + fx.NumLayers, rows);
    }

    /// <summary>
    /// A hybrid-dense prefill tiled at 3 tokens: a last-row tap delivers one sample per point (final tile), an every-row tap runs it
    /// untiled and delivers rows 0..P-1, and an abort in the final tile rolls the KV back past the earlier tiles too.
    /// </summary>
    [SkippableFact]
    public void HybridDense_TiledPrefill_DeliversOncePerPoint_AndAbortUndoesEveryTile()
    {
        string ptx = RequireCuda();
        using var fx = new CudaHiddenStateTapFixture(HiddenStateTapBackendSuite.HybridDense, ptx);
        var model = (CudaQwen3HybridDenseTransformerModel)fx.Model;
        int P = fx.Prompt.Length;
        int[] pp = Enumerable.Range(0, P).ToArray();
        int? saved = CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride;
        CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = 3;
        try
        {
            var last = new List<(HookPoint P, int Row, int Pos)>();
            using (var h = model.RegisterTap(new HiddenStateTapSubscription([fx.NumLayers - 1], All4),
                s => last.Add((s.Point, s.RowIndex, s.TokenPosition))))
            {
                using var kv = model.CreateKvCache(96);
                using (model.Forward(fx.Prompt, pp, -1, kv, null, null, lastTokenLogitsOnly: true)) { }
            }
            Assert.Equal(4, last.Count);
            Assert.All(last, s => { Assert.Equal(P - 1, s.Row); Assert.Equal(P - 1, s.Pos); });

            var all = new List<(HookPoint P, int Row)>();
            using (var h = model.RegisterTap(new HiddenStateTapSubscription([fx.NumLayers - 1], All4, TapRows.All),
                s => all.Add((s.Point, s.RowIndex))))
            {
                using var kv = model.CreateKvCache(96);
                model.ResetSequenceState();
                using (model.Forward(fx.Prompt, pp, -1, kv, null, null, lastTokenLogitsOnly: true)) { }
            }
            foreach (var point in All4)
                Assert.Equal(Enumerable.Range(0, P), all.Where(s => s.P == point).Select(s => s.Row));

            model.ResetSequenceState();
            var aborter = new AbortAt(HookPoint.PostLayer, fx.NumLayers - 1);
            using (var h = model.RegisterAbortableTap(new HiddenStateTapSubscription([fx.NumLayers - 1], All4), aborter.OnSample))
            {
                using var kv = model.CreateKvCache(96);
                Assert.False(model.TryForward(fx.Prompt, pp, -1, kv, out var logits, out var abort));
                Assert.Null(logits);
                Assert.Equal(new ForwardAbortInfo(fx.NumLayers - 1, HookPoint.PostLayer), abort);
                Assert.Equal(0, kv.CurrentLength);
            }
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = saved;
        }
    }

    private sealed class AbortAt(HookPoint point, int layer)
    {
        public HookResult OnSample(HiddenStateSample s)
            => s.Point == point && s.LayerIndex == layer ? HookResult.AbortForward : HookResult.Continue;
    }

    /// <summary>
    /// The GDN-hybrid MoE host takes an abortable tap exactly when it can checkpoint recurrent state: it refuses (naming #1919) while
    /// <see cref="IModel.SupportsRecurrentStateCheckpoint"/> is false and accepts once it is true, so this passes whichever branch lands first.
    /// Read-only taps work either way (the rest of the suite).
    /// </summary>
    [SkippableFact]
    public void MoeHybrid_AbortableTap_RefusedWithoutRecurrentCheckpoint_AcceptedWith()
    {
        using var fx = Open(MoeHybrid);
        if (fx.Model.SupportsRecurrentStateCheckpoint)
        {
            using var h = fx.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => HookResult.Continue);
        }
        else
        {
            var ex = Assert.Throws<NotSupportedException>(() =>
                fx.Host.RegisterAbortableTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => HookResult.Continue));
            Assert.Contains("#1919", ex.Message);
            using var ro = fx.Host.RegisterTap(new HiddenStateTapSubscription(null, [HookPoint.PostEmbedding]), _ => { });
        }
    }
}
