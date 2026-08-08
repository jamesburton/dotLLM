using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Regression coverage for issue #291: <c>HybridTransformerModel</c> (the generic, Llama-style
/// CPU/GPU partial-offload splitter) threw <c>KeyNotFoundException: 'blk.0.attn_output.weight'</c>
/// loading any <c>Qwen3HybridDense</c> (<c>qwen35</c>) GGUF, because layer 0 is a Gated-DeltaNet
/// (GDN) layer with no attention-output projection at all — the generic splitter always assumed a
/// uniform Llama-style tensor-name set for every layer. <see cref="HybridQwen3HybridDenseTransformerModel"/>
/// is the architecture-aware fix: it composes a GPU head
/// (<see cref="CudaQwen3HybridDenseTransformerModel.LoadHeadFromGguf"/>) with a CPU tail
/// (<see cref="Qwen3HybridDenseTransformerModel.LoadTailFromGguf"/>), each reusing this
/// architecture's already-correct per-layer GDN-vs-attention tensor-name resolution.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a 4-layer fixture, not the default 2-layer one.</b> Per this project's own
/// "degenerate shapes miss real bugs" lesson (CLAUDE.md's cross-backend critical-bug section), a
/// split boundary that only ever lands between two single-layer runs (one GDN layer, one attention
/// layer, one on each side) cannot catch a local-vs-global layer-index / GDN-ordinal / KV-slot
/// mapping bug — every "local index" would trivially equal 0 either way. <see cref="FullAttnInterval"/>=2
/// over a 4-layer trunk gives layers <c>[GDN, Attn, GDN, Attn]</c>; splitting at
/// <see cref="NumGpuLayers"/>=2 puts a GDN+Attn PAIR on the GPU side and another GDN+Attn pair on
/// the CPU side — the boundary genuinely crosses both layer kinds on each half, exercising the
/// tail's local ordinal/KV-slot renumbering (see <c>Qwen3HybridDenseTransformerModel.LoadTailFromGguf</c>)
/// instead of accidentally being correct via trivial 0/1 counts.
/// </para>
/// <para>
/// <b>Oracle.</b> The pure CPU-only <see cref="Qwen3HybridDenseTransformerModel"/> path (already
/// proven correct — see <c>Qwen3HybridDenseRealGgufConstructionTests</c>) is the parity reference;
/// this suite pins the hybrid split's full-sequence logits against it.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class HybridQwen3HybridDenseTransformerModelSplitParityTests : IDisposable
{
    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    // 4-layer trunk, fullAttnInterval=2 → layer kinds [GDN, Attn, GDN, Attn] (0-indexed;
    // (i+1) % 2 == 0 is full-attention). Splitting at gpuLayers=2 puts a GDN+Attn pair on
    // EACH side of the boundary.
    private const int BlockCount = 4;
    private const int FullAttnInterval = 2;
    private const int NumGpuLayers = 2;
    private const float AbsTol = 1.5e-3f;
    private const float RelTol = 5e-3f;

    public HybridQwen3HybridDenseTransformerModelSplitParityTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-hybrid-qwen35-split-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

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

    /// <summary>
    /// Direct repro of issue #291's reported failure: a GDN-first layer (blk.0, no
    /// <c>attn_output.weight</c>) loaded through the CPU/GPU partial-offload path. Before the fix,
    /// this threw <c>KeyNotFoundException: 'blk.0.attn_output.weight'</c> — before the fix existed
    /// there was no architecture-aware hybrid loader at all, so this test also documents the fixed
    /// entry point.
    /// </summary>
    [SkippableFact]
    public void LoadFromGguf_GdnFirstLayer_PartialOffload_DoesNotThrow()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-gdn-first.gguf"), withMtp: false, blockCount: BlockCount,
            fullAttnInterval: FullAttnInterval);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        Assert.Equal(BlockCount, config.NumLayers);
        Assert.NotNull(config.HybridLayout);
        Assert.Equal(DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet, config.HybridLayout!.LayerKind[0]);
        Assert.Equal(DotLLM.Core.Models.HybridLayerKind.Attention, config.HybridLayout!.LayerKind[1]);

        using var model = HybridQwen3HybridDenseTransformerModel.LoadFromGguf(
            gguf, config, numGpuLayers: 1, deviceId: 0, ThreadingConfig.SingleThreaded);

        Assert.Equal(1, model.NumGpuLayers);
    }

    /// <summary>
    /// Full-sequence logits parity: the CPU/GPU split model's output must match the pure CPU-only
    /// oracle within the established hybrid-path tolerance band, across a split boundary that
    /// crosses both layer kinds on each side (see class remarks).
    /// </summary>
    [SkippableFact]
    public void HybridForward_SplitBoundaryCrossesBothLayerKinds_MatchesCpuOracle()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-split-parity.gguf"), withMtp: false, blockCount: BlockCount,
            fullAttnInterval: FullAttnInterval);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(BlockCount, config.NumLayers);
        var kinds = config.HybridLayout!.LayerKind;
        _out.WriteLine($"Layer kinds: [{string.Join(", ", kinds)}]");
        // Confirm the fixture actually has both kinds on both sides of the NumGpuLayers boundary —
        // if this assertion ever fails, the "crosses both kinds" claim above is no longer true and
        // the test stops being the regression guard it's meant to be.
        Assert.Contains(DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet, kinds[..NumGpuLayers]);
        Assert.Contains(DotLLM.Core.Models.HybridLayerKind.Attention, kinds[..NumGpuLayers]);
        Assert.Contains(DotLLM.Core.Models.HybridLayerKind.GatedDeltaNet, kinds[NumGpuLayers..]);
        Assert.Contains(DotLLM.Core.Models.HybridLayerKind.Attention, kinds[NumGpuLayers..]);

        int[] tokenIds = [3, 1, 4];
        int[] positions = [0, 1, 2];

        float[] cpuLogits = RunCpuOracle(gguf, config, tokenIds, positions);
        float[] hybridLogits = RunHybridSplit(gguf, config, tokenIds, positions, ptxDir!);

        AssertLogitsMatch(cpuLogits, hybridLogits);
    }

    private static float[] RunCpuOracle(GgufFile gguf, DotLLM.Core.Models.ModelConfig config,
        int[] tokenIds, int[] positions)
    {
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        int vocab = config.VocabSize;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * vocab);
        // Last row only — matches the hybrid path's own convention comparison below.
        return span.Slice((tokenIds.Length - 1) * vocab, vocab).ToArray();
    }

    private static float[] RunHybridSplit(GgufFile gguf, DotLLM.Core.Models.ModelConfig config,
        int[] tokenIds, int[] positions, string ptxDir)
    {
        using var model = HybridQwen3HybridDenseTransformerModel.LoadFromGguf(
            gguf, config, numGpuLayers: NumGpuLayers, deviceId: 0, ThreadingConfig.SingleThreaded);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        int vocab = config.VocabSize;
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * vocab);
        return span.Slice((tokenIds.Length - 1) * vocab, vocab).ToArray();
    }

    private void AssertLogitsMatch(float[] cpu, float[] hybrid)
    {
        Assert.Equal(cpu.Length, hybrid.Length);

        _out.WriteLine("col | cpu        | hybrid     | |diff|");
        _out.WriteLine("----+------------+------------+----------");
        float maxAbs = 0f;
        for (int c = 0; c < cpu.Length; c++)
        {
            float d = MathF.Abs(cpu[c] - hybrid[c]);
            if (d > maxAbs) maxAbs = d;
            _out.WriteLine($"{c,3} | {cpu[c],10:F6} | {hybrid[c],10:F6} | {d:E3}");
        }
        _out.WriteLine($"max|diff|={maxAbs:E3}  AbsTol={AbsTol:E3}");

        for (int c = 0; c < cpu.Length; c++)
        {
            float pref = cpu[c];
            float hyb = hybrid[c];
            Assert.True(float.IsFinite(pref), $"col={c}: cpu logit non-finite: {pref}");
            Assert.True(float.IsFinite(hyb), $"col={c}: hybrid logit non-finite: {hyb}");
            float diff = MathF.Abs(pref - hyb);
            float bar = AbsTol + RelTol * MathF.Abs(pref);
            Assert.True(diff <= bar,
                $"col={c}: cpu={pref:F6} vs hybrid={hyb:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }
}
