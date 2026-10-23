using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1094: with both K and V quantized, every multi-row forward that is not served by the per-row verify tier (a
/// prefill tile, a 9-row MTP verify) runs the flash prefill kernel that reads the Q8_0 / Q4_0 blocks directly
/// (<c>attention_flash_mma_prefill_hd256_kvq_*</c>), so the full-history F16 dequant scratch is never built and
/// <see cref="CudaQwen3HybridDenseTransformerModel.ContextBytesPerToken"/> stops counting it.
/// </summary>
/// <remarks>
/// <para>Gated on the real Bonsai 2 checkpoint (<c>DOTLLM_BONSAI_PQ2_0_GGUF</c>): the synthetic model's KV row is 16
/// elements, below the 32-element quantization block.</para>
/// <para>The kvq kernel is bit-identical to scratch-dequant + the F16 kernel (<c>CudaAttentionFlashPrefillHd256KvqTests</c>),
/// and nothing else in the forward changes, so the logits are compared bit for bit. The prefill chunks compare the last row's
/// logits: through the later layers' K/V it depends on every row's attention output. Asking for all rows' logits over more
/// than <c>CudaSmallSGemvDispatch.MaxColumns</c> rows would dequantize the whole LM head to F16 (about 2.5 GiB on Bonsai 2),
/// which leaves the scratch reference no room for its own allocation on a 12 GB card.</para>
/// <para>Mutants: leaving
/// <c>ResolveKvF16</c> on the kvq tier fails the counter assertions; routing the tier to the wrong K/V dtype pair or
/// dropping <c>positionOffset</c> breaks bit identity; keeping the scratch term fails the bytes-per-token check.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseKvqPrefillTests
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";

    /// <summary>Rows of the short prefill chunks: two 16-row query tiles, the second one partial.</summary>
    private const int TileRows = 20;

    /// <summary>Position of the second short chunk (deep enough that the history spans many KV tiles).</summary>
    private const int DeepOffset = 3000;

    /// <summary>History before the verify: past the fused decode tiers' minimum depth, like a real MTP round.</summary>
    private const int VerifyPrefix = 600;

    /// <summary>An MTP verify at the largest draft length: <c>MtpCostModel.MaxK</c> (8) + 1.</summary>
    private const int VerifyRows = 9;

    private readonly ITestOutputHelper _out;

    /// <summary>Creates the test class.</summary>
    /// <param name="output">xUnit output sink.</param>
    public CudaQwen3HybridDenseKvqPrefillTests(ITestOutputHelper output) => _out = output;

    /// <summary>The K/V storage pairs exercised: both symmetric formats and a mixed one.</summary>
    public static IEnumerable<object[]> Pairs() =>
    [
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0],
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0],
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0],
    ];

    /// <summary>A two-tile prefill at position 0 and at <see cref="DeepOffset"/> dequantizes no history.</summary>
    /// <param name="keys">Key storage.</param>
    /// <param name="values">Value storage.</param>
    [SkippableTheory]
    [MemberData(nameof(Pairs))]
    public void Prefill_OverQuantizedKv_AllocatesNoDequantScratch(KvCacheDType keys, KvCacheDType values)
    {
        using var loaded = LoadModel(keys, values);
        var model = loaded.Model;
        Assert.True(model.FlashPrefillHd256KvqForTest, "the kvq prefill tier must be on by default");
        var run = RunPrefill(model);
        _out.WriteLine($"{keys}/{values}: dequant calls {run.Dequant}, kvq prefill calls {run.Kvq} (layers {model.AttentionLayerCount})");
        Assert.Equal(0, run.Dequant);
        // Three forwards (20 @ 0, the fill to DeepOffset, 20 @ DeepOffset), each at least one call per attention layer.
        Assert.True(run.Kvq >= 3 * model.AttentionLayerCount, $"kvq prefill calls {run.Kvq}");
        Assert.Equal(0, run.Kvq % model.AttentionLayerCount);
    }

    /// <summary>The kvq tier's logits equal the scratch path's bit for bit, at position 0 and deep in the history.</summary>
    /// <param name="keys">Key storage.</param>
    /// <param name="values">Value storage.</param>
    [SkippableTheory]
    [MemberData(nameof(Pairs))]
    public void Prefill_OverQuantizedKv_MatchesTheScratchPath(KvCacheDType keys, KvCacheDType values)
    {
        using var loaded = LoadModel(keys, values);
        var model = loaded.Model;

        model.FlashPrefillHd256KvqForTest = false;
        var scratch = RunPrefill(model);
        Assert.True(scratch.Dequant > 0, "the reference must take the scratch path");
        model.FlashPrefillHd256KvqForTest = true;
        var kvq = RunPrefill(model);
        Assert.Equal(0, kvq.Dequant);

        AssertBitIdentical($"{keys}/{values} chunk @0", scratch.Head, kvq.Head);
        AssertBitIdentical($"{keys}/{values} fill last token", scratch.Fill, kvq.Fill);
        AssertBitIdentical($"{keys}/{values} chunk @{DeepOffset}", scratch.Deep, kvq.Deep);
    }

    /// <summary>A 9-row MTP verify (past the per-row tier's 8-row limit) dequantizes no history and matches the scratch path.</summary>
    /// <param name="keys">Key storage.</param>
    /// <param name="values">Value storage.</param>
    [SkippableTheory]
    [MemberData(nameof(Pairs))]
    public void NineRowVerify_OverQuantizedKv_AllocatesNoDequantScratch(KvCacheDType keys, KvCacheDType values)
    {
        Assert.True(VerifyRows > CudaQwen3HybridDenseTransformerModel.PerRowKvqMaxRows,
            "the verify must be wider than the per-row tier, or this test exercises the wrong path");
        using var loaded = LoadModel(keys, values);
        var model = loaded.Model;
        int vocab = model.Config.VocabSize;
        Skip.IfNot(model.SupportsRecurrentRowSnapshots, "row snapshots unsupported on this checkpoint");
        using var kv = model.CreateKvCache(VerifyPrefix + VerifyRows + 8);

        int[] prefix = Tokens(VerifyPrefix, salt: 0);
        using (model.Forward(prefix, Enumerable.Range(0, VerifyPrefix).ToArray(), deviceId: -1, kv, lastTokenLogitsOnly: true)) { }
        object? afterPrefix = model.CheckpointRecurrentState();

        int[] rows = Tokens(VerifyRows, salt: 17);
        int[] rowPositions = Enumerable.Range(VerifyPrefix, VerifyRows).ToArray();

        // Reference first: the scratch path admits its F16 dequant scratch before the verify's all-rows LM head (wider than
        // the multi-row GEMV) dequantizes the whole head to F16 and leaves a 12 GB card without room for it.
        model.FlashPrefillHd256KvqForTest = false;
        int dequant0 = model.DebugKvDequantCallsForTest;
        float[] scratch;
        using (ITensor t = model.ForwardWithRecurrentSnapshots(rows, rowPositions, deviceId: -1, kv, mtpState: null))
            scratch = Copy(t, (long)VerifyRows * vocab);
        Assert.Equal(2 * model.AttentionLayerCount, model.DebugKvDequantCallsForTest - dequant0);

        // The same verify through the kvq tier, from the same recurrent state and KV length.
        kv.Rollback(VerifyPrefix);
        model.RestoreRecurrentState(afterPrefix);
        model.FlashPrefillHd256KvqForTest = true;
        dequant0 = model.DebugKvDequantCallsForTest;
        int kvq0 = model.DebugFlashPrefillHd256KvqCallsForTest;
        int perRow0 = model.DebugPerRowKvqAttentionCallsForTest;
        float[] fused;
        using (ITensor t = model.ForwardWithRecurrentSnapshots(rows, rowPositions, deviceId: -1, kv, mtpState: null))
            fused = Copy(t, (long)VerifyRows * vocab);
        (afterPrefix as IDisposable)?.Dispose();
        int dequant = model.DebugKvDequantCallsForTest - dequant0;
        int kvq = model.DebugFlashPrefillHd256KvqCallsForTest - kvq0;
        _out.WriteLine($"{keys}/{values} 9-row verify: dequant calls {dequant}, kvq prefill calls {kvq} " +
                       $"(layers {model.AttentionLayerCount})");
        Assert.Equal(0, dequant);
        Assert.Equal(model.AttentionLayerCount, kvq);
        Assert.Equal(perRow0, model.DebugPerRowKvqAttentionCallsForTest);

        AssertBitIdentical($"{keys}/{values} 9-row verify", scratch, fused);
    }

    /// <summary>With the kvq prefill eligible, a token costs only its K/V rows: q4_0/q4_0 without MTP is 18 KiB.</summary>
    [SkippableFact]
    public void ContextBytesPerToken_DropsTheScratchTerm()
    {
        using var loaded = LoadModel(KvCacheDType.Q4_0, KvCacheDType.Q4_0);
        var model = loaded.Model;
        long perToken = model.ContextBytesPerToken(mtp: false);
        _out.WriteLine($"q4_0/q4_0 no MTP: {perToken} B/token (KV rows {model.ComputeKvBytes(1)} B)");
        Assert.Equal(model.ComputeKvBytes(1), perToken);
        Assert.Equal(18 * 1024, perToken);

        // With the tier off the scratch is back: 2 sides x kvElems x sizeof(half).
        model.FlashPrefillHd256KvqForTest = false;
        int kvElems = model.Config.NumKvHeads * model.Config.HeadDim;
        Assert.Equal(18 * 1024 + 2L * kvElems * sizeof(ushort), model.ContextBytesPerToken(mtp: false));
    }

    private sealed record PrefillRun(float[] Head, float[] Fill, float[] Deep, int Dequant, int Kvq);

    /// <summary>
    /// One independent sequence: <see cref="TileRows"/> rows at 0, the fill up to <see cref="DeepOffset"/>, then
    /// <see cref="TileRows"/> rows at <see cref="DeepOffset"/>; each returns its last row's logits (see the class remarks).
    /// </summary>
    private static PrefillRun RunPrefill(CudaQwen3HybridDenseTransformerModel model)
    {
        int vocab = model.Config.VocabSize;
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(DeepOffset + TileRows + 8);
        int dequant0 = model.DebugKvDequantCallsForTest;
        int kvq0 = model.DebugFlashPrefillHd256KvqCallsForTest;

        float[] head, fill, deep;
        using (ITensor t = model.Forward(Tokens(TileRows, salt: 0), Enumerable.Range(0, TileRows).ToArray(),
                                         deviceId: -1, kv, lastTokenLogitsOnly: true))
            head = Copy(t, vocab);
        using (ITensor t = model.Forward(Tokens(DeepOffset - TileRows, salt: 5), Enumerable.Range(TileRows, DeepOffset - TileRows).ToArray(),
                                         deviceId: -1, kv, lastTokenLogitsOnly: true))
            fill = Copy(t, vocab);
        using (ITensor t = model.Forward(Tokens(TileRows, salt: 11), Enumerable.Range(DeepOffset, TileRows).ToArray(),
                                         deviceId: -1, kv, lastTokenLogitsOnly: true))
            deep = Copy(t, vocab);

        return new PrefillRun(head, fill, deep,
            model.DebugKvDequantCallsForTest - dequant0, model.DebugFlashPrefillHd256KvqCallsForTest - kvq0);
    }

    private void AssertBitIdentical(string label, float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        int diff = 0;
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i])) diff++;
        _out.WriteLine($"{label}: bit-diff logits = {diff} / {want.Length}");
        Assert.Equal(0, diff);
    }

    private static int[] Tokens(int n, int salt)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 1 + ((i + salt * 131) * 7919 + i / 13) % 30000;
        return t;
    }

    private static unsafe float[] Copy(ITensor t, long count)
        => new ReadOnlySpan<float>((void*)t.DataPointer, checked((int)count)).ToArray();

    /// <summary>The model and the GGUF it maps: the weights (and the CPU-side embedding table) stay mapped until disposal.</summary>
    private sealed class Loaded(GgufFile gguf, CudaQwen3HybridDenseTransformerModel model) : IDisposable
    {
        public CudaQwen3HybridDenseTransformerModel Model { get; } = model;

        public void Dispose()
        {
            Model.Dispose();
            gguf.Dispose();
        }
    }

    private static Loaded LoadModel(KvCacheDType keys, KvCacheDType values)
    {
        string? ptxDir = SkipUnlessCuda();
        string? path = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        Skip.If(string.IsNullOrWhiteSpace(path) || !File.Exists(path), $"Set {ModelPathEnvVar} to a Bonsai-2 PQ2_0 GGUF.");

        var gguf = GgufFile.Open(path!);
        CudaQwen3HybridDenseTransformerModel? model = null;
        try
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
            model.SetKvStorageForTest(keys, values);
            Skip.IfNot(model.FlashPrefillHd256ForTest, "flash prefill hd256 disabled by environment");
            Skip.IfNot(model.HasFlashPrefillHd256KvqKernelForTest, "flash prefill hd256 kvq kernel not loaded on this device");
            return new Loaded(gguf, model);
        }
        catch
        {
            model?.Dispose();
            gguf.Dispose();
            throw;
        }
    }

    private static string? SkipUnlessCuda()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        Skip.If(true, "PTX files not found");
        return null;
    }
}
