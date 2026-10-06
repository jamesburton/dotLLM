using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #494: <see cref="CudaQwen3HybridDenseTransformerModel"/> tiles a long prefill over tokens
/// so peak VRAM is bounded by the tile, not by the prompt. A 4096-token single call pinned a 12 GB
/// RTX 3060 at 12,035 / 12,288 MiB on Bonsai 2 27B and thrashed; p=1024 was fine.
/// </summary>
/// <remarks>
/// <para>Two properties are under test. First, <b>the tiling is invisible in the output</b>: the
/// last row's logits from a 5-tile run must match a single-call run over the same tokens. Second,
/// <b>the gates hold</b>: with <c>lastTokenLogitsOnly: false</c> the caller wants a row per
/// position, so tiling must stay off and the shape must remain <c>[seqLen, vocab]</c> however small
/// the tile size is set.</para>
/// <para><b>Not bit-identical, by design.</b> Each layer's GEMMs run with a different M per tile
/// and cuBLAS selects algorithms per M, so the tolerance is this codebase's standard CUDA
/// fixture-parity bar (AbsTol 1e-4 + RelTol 1e-3) — the same one
/// <see cref="CudaQwen3HybridDenseLastTokenLogitsOnlyTest"/> uses for the closely related
/// kernel-routing drift, and for the same reason: two numerically-different routes to one logical
/// projection, not an imprecise computation.</para>
/// <para>Fresh model instances per run, sequentially: this model's GatedDeltaNet state is owned by
/// the model, not by the <c>IKvCache</c>, so reusing one instance would carry state from the first
/// run into the second and make the two incomparable. Bonsai's ~7.2 GB weight set also does not fit
/// twice on a 12 GB card.</para>
/// <para>Mutants: dropping the <c>!lastTokenLogitsOnly</c> gate in <c>ForwardCore</c> fails
/// <see cref="AllRowLogits_AreNeverTiled"/> (it would return one row instead of <c>seqLen</c>);
/// dropping the per-tile <c>positions</c> slice, or letting a non-final tile skip its layer stack,
/// moves the last row far outside the tolerance in
/// <see cref="TiledPrefill_MatchesSingleCall_LastRowLogits"/>.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaQwen3HybridDensePrefillChunkingTest
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";
    private const int SliceLen = 64;
    private const int SeqLen = 40;
    private const int TileTokens = 8;   // 40 tokens => 5 tiles

    private readonly ITestOutputHelper _out;
    public CudaQwen3HybridDensePrefillChunkingTest(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void TiledPrefill_MatchesSingleCall_LastRowLogits()
    {
        (string path, string ptxDir, DotLLM.Core.Models.ModelConfig config) = Setup();

        int[] tokens = SyntheticTokens();
        int[] positions = [.. Enumerable.Range(0, SeqLen)];

        _out.WriteLine("Single call (tiling disabled)...");
        float[] single = RunLastRow(path, config, ptxDir, tokens, positions, tile: 0, expectRows: 1);

        _out.WriteLine($"Tiled call ({SeqLen} tokens in tiles of {TileTokens})...");
        float[] tiled = RunLastRow(path, config, ptxDir, tokens, positions, tile: TileTokens, expectRows: 1);

        const float AbsTol = 1e-4f;
        const float RelTol = 1e-3f;
        for (int i = 0; i < SliceLen; i++)
        {
            float a = single[i];
            float diff = MathF.Abs(a - tiled[i]);
            float tol = AbsTol + RelTol * MathF.Abs(a);
            Assert.True(diff <= tol,
                $"logits[{i}]: single call={a}, {SeqLen}-in-{TileTokens} tiled={tiled[i]}, diff={diff} "
                + $"exceeds tolerance {tol} — tiling the prefill changed the result by more than the "
                + "expected per-M cuBLAS algorithm-selection drift.");
        }

        _out.WriteLine($"{SliceLen} logits compared — tiled vs single-call last row: within tolerance.");
    }

    [SkippableFact]
    public void AllRowLogits_AreNeverTiled()
    {
        (string path, string ptxDir, DotLLM.Core.Models.ModelConfig config) = Setup();

        int[] tokens = SyntheticTokens();
        int[] positions = [.. Enumerable.Range(0, SeqLen)];

        // A caller that wants every position's logits must get every position's logits, whatever
        // the tile size says — the tile loop only ever returns the final tile's rows.
        CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = TileTokens;
        try
        {
            using var gguf = GgufFile.Open(path);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            using var cache = model.CreateKvCache(maxSeqLen: 256);
            using ITensor logits = model.Forward(tokens, positions, deviceId: -1, cache,
                                                 lastTokenLogitsOnly: false);
            Assert.Equal(SeqLen, logits.Shape[0]);
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = null;
        }
    }

    // ── helpers ──

    private static int[] SyntheticTokens()
    {
        int[] tokens = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++) tokens[i] = (11 + i * 37) % 40000;
        return tokens;
    }

    private static float[] RunLastRow(string path, DotLLM.Core.Models.ModelConfig config,
                                      string ptxDir, int[] tokens, int[] positions,
                                      int tile, int expectRows)
    {
        CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = tile;
        // #500: tiling changes M, and M decides which PQ2_0 path a projection takes (dp4a int8 for
        // small S, dequant+cuBLAS F16 above). Comparing across that boundary measures the dp4a
        // numeric trade, not the tiling invariance this test is about, so pin one path for both arms.
        bool? dp4aWas = CudaSmallSGemvDispatch.Dp4aOverride;
        CudaSmallSGemvDispatch.Dp4aOverride = false;
        try
        {
            using var gguf = GgufFile.Open(path);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            using var cache = model.CreateKvCache(maxSeqLen: 256);
            using ITensor logits = model.Forward(tokens, positions, deviceId: -1, cache,
                                                 lastTokenLogitsOnly: true);
            Assert.Equal(expectRows, logits.Shape[0]);
            return ExtractRow(logits, logits.Shape[0] - 1, SliceLen);
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = dp4aWas;
        }
    }

    private (string Path, string PtxDir, DotLLM.Core.Models.ModelConfig Config) Setup()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        return (path!, ptxDir!, config);
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static unsafe float[] ExtractRow(ITensor logits, int row, int sliceLen)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        float* basePtr = (float*)logits.DataPointer + (long)row * vocab;
        var slice = new float[sliceLen];
        for (int i = 0; i < sliceLen; i++) slice[i] = basePtr[i];
        return slice;
    }

    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
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
}
