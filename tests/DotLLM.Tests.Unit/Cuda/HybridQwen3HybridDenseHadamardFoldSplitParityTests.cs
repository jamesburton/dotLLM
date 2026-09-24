using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #481, device side: <see cref="HybridQwen3HybridDenseTransformerModel"/> (GPU head + CPU
/// tail) must load a Hadamard-folded <c>qwen35</c> checkpoint and match the CPU-only model with the
/// same fold. Before #481 the CPU tail refused every folded declaration at load time.
/// </summary>
/// <remarks>
/// <para>
/// The fixture is the 4-block <c>[GDN, Attn, GDN, Attn]</c> fold trunk with 2 GDN key heads (so the
/// <c>gdn_v_grouped</c> permute is a real reordering) split at 2 — both layer kinds on each side —
/// and at 1 (an odd split, where local and global block parity differ). The CPU-only oracle rotates
/// every block, the embedding and the lm_head itself; the split only matches it if the GPU head
/// applies the embedding inverse + its blocks' rotations and the CPU tail applies its blocks' +
/// the lm_head's. A second assertion pins that the fold moves the split's own output, so a split
/// that silently dropped the fold on both halves could not pass by matching an unfolded oracle.
/// </para>
/// <para>
/// Host-only validation coverage lives in <see cref="Qwen3HybridDenseHadamardFoldSplitTests"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class HybridQwen3HybridDenseHadamardFoldSplitParityTests : IDisposable
{
    private const int Blocks = 4;
    // Same band as HybridQwen3HybridDenseTransformerModelSplitParityTests / the #479 fold parity test.
    private const float AbsTol = 1.5e-3f;
    private const float RelTol = 5e-3f;
    private const float MinDiscriminatingDiff = 2e-2f;

    private static readonly int[] TokenIds = [3, 1, 4, 1, 5];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public HybridQwen3HybridDenseHadamardFoldSplitParityTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-hybrid-qwen35-fold-split-{Guid.NewGuid():N}");
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

    private static bool PtxPresent()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "ptx");
        return Directory.Exists(dir) && File.Exists(Path.Combine(dir, "hadamard_fwht.ptx"));
    }

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(1)]
    public void FoldedSplit_MatchesFoldedCpuOracle(int numGpuLayers)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        Skip.IfNot(PtxPresent(), "hadamard_fwht.ptx not found next to the test binaries");

        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold-split.gguf"),
            withMtp: false, blockCount: Blocks);
        using var gguf = GgufFile.Open(path);
        var baseConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Blocks, baseConfig.NumLayers);
        var folded = baseConfig with { HadamardFold = SyntheticHadamardFold.For(baseConfig) };

        float[] cpu = RunCpu(gguf, folded);
        float[] split = RunSplit(gguf, folded, numGpuLayers);
        float[] splitUnfolded = RunSplit(gguf, baseConfig, numGpuLayers);

        AssertMatch(cpu, split);

        float moved = 0;
        for (int i = 0; i < split.Length; i++) moved = MathF.Max(moved, MathF.Abs(split[i] - splitUnfolded[i]));
        _out.WriteLine($"split@{numGpuLayers}: fold vs no fold max |diff| = {moved:E3}");
        Assert.True(moved > MinDiscriminatingDiff, "the fold does not move the split's output — the test would not discriminate");
    }

    private static float[] RunCpu(GgufFile gguf, ModelConfig config)
    {
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        return AllRows(model.Forward(TokenIds, Positions(), deviceId: -1), config.VocabSize);
    }

    private static float[] RunSplit(GgufFile gguf, ModelConfig config, int numGpuLayers)
    {
        using var model = HybridQwen3HybridDenseTransformerModel.LoadFromGguf(
            gguf, config, numGpuLayers, deviceId: 0, ThreadingConfig.SingleThreaded);
        return AllRows(model.Forward(TokenIds, Positions(), deviceId: -1), config.VocabSize);
    }

    private static int[] Positions() => Enumerable.Range(0, TokenIds.Length).ToArray();

    private static float[] AllRows(ITensor logits, int vocab)
    {
        using (logits)
        {
            Assert.Equal(TokenIds.Length, logits.Shape[0]);
            return new ReadOnlySpan<float>((void*)logits.DataPointer, TokenIds.Length * vocab).ToArray();
        }
    }

    private void AssertMatch(float[] cpu, float[] split)
    {
        Assert.Equal(cpu.Length, split.Length);
        float maxAbs = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(float.IsFinite(split[i]), $"[{i}] split logit non-finite: {split[i]}");
            float d = MathF.Abs(cpu[i] - split[i]);
            maxAbs = MathF.Max(maxAbs, d);
            float bar = AbsTol + RelTol * MathF.Abs(cpu[i]);
            Assert.True(d <= bar, $"[{i}] cpu={cpu[i]:F6} split={split[i]:F6} |diff|={d:E3} > {bar:E3}");
        }
        _out.WriteLine($"max |cpu - split| = {maxAbs:E3}");
    }
}
