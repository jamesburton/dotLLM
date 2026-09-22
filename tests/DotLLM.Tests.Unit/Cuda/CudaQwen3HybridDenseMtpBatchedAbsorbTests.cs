using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #472 on CUDA (ported in #478): the batched, KV-only MTP absorb must leave the head's
/// device KV-cache, its pending hidden row and the next draft's logits BIT-IDENTICAL to the #469
/// per-token absorb loop.
/// </summary>
/// <remarks>
/// <para>
/// Bit-identity (not a tolerance) because the CUDA batched absorb keeps the three projections as
/// single-row GEMVs — the same kernels the per-token path runs — and batches only per-row kernels
/// (RMSNorm, RoPE). See <c>CudaQwen3HybridDenseTransformerModel.AbsorbMtpBatched</c>.
/// </para>
/// <para>
/// Two trunk forwards, then a verify-shaped third after three speculative draft steps, so the
/// carried row (row 0 of a later batch), the captured-row pairing (rows 1..) and the rollback over
/// drafted slots are all exercised. The fixture's random weights make every hidden row distinct, so
/// a pairing or offset slip moves the K/V rows by O(1).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpBatchedAbsorbTests : IDisposable
{
    private static readonly int[][] Batches = [[1, 2, 3], [4, 5, 6, 7]];
    private static readonly int[] Verify = [8, 10, 11];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseMtpBatchedAbsorbTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mtp-absorb-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(true, false)]
    [InlineData(false, false)]
    [InlineData(true, true)]    // issue #486: Q8_0 head -> multi-column absorb vs single-row per-token GEMVs
    [InlineData(false, true)]
    public void BatchedAbsorb_IsBitIdenticalToPerTokenAbsorb(bool mtpHasOwnHeadTensors, bool q8_0MtpHead)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, $"mtp-own{mtpHasOwnHeadTensors}-q8{q8_0MtpHead}.gguf"),
            withMtp: true, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors, q8_0MtpHead: q8_0MtpHead);

        var perToken = Run(path, ptxDir!, perToken: true);
        var batched = Run(path, ptxDir!, perToken: false);

        Assert.Equal(perToken.Length, batched.Length);
        AssertBitEqual(perToken.Keys, batched.Keys, "K");
        AssertBitEqual(perToken.Values, batched.Values, "V");
        AssertBitEqual(perToken.Pending, batched.Pending, "pending hidden");
        AssertBitEqual(perToken.DraftLogits, batched.DraftLogits, "next draft step logits");

        // Guard against a vacuous pass: the absorbed rows must be real, distinct data.
        Assert.Contains(batched.Keys, v => v != 0f);
        _out.WriteLine($"absorbed {batched.Length} positions; K/V/pending/logits bit-identical " +
                       $"(Q8_0 head: {q8_0MtpHead}; register-blocked Q8_0 GEMV: {batched.UsesRbQ8})");
    }

    private sealed record Snapshot(int Length, float[] Keys, float[] Values, float[] Pending, float[] DraftLogits,
                                   bool UsesRbQ8);

    private static unsafe Snapshot Run(string path, string ptxDir, bool perToken)
    {
        MtpAbsorbDispatch.PerTokenOverride = perToken;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using var state = (CudaMtpState)model.CreateMtpState()!;

            int p = Drive(model, kv, state);
            Assert.Equal(p, state.CurrentLength);

            int n = p * state.KvStride;
            float[] keys = Download(state.KeyCacheDevicePtr, n);
            float[] values = Download(state.ValueCacheDevicePtr, n);
            float[] pending = Download(state.PendingHiddenDevicePtr, config.HiddenSize);

            using ITensor draft = model.ForwardMtp(state, 9, p);
            float[] logits = new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize).ToArray();
            return new Snapshot(p, keys, values, pending, logits, model.MtpUsesRbQ8Gemv);
        }
        finally
        {
            MtpAbsorbDispatch.PerTokenOverride = null;
        }
    }

    private static int Drive(IModel model, IKvCache kv, IMtpState state)
    {
        int p = 0;
        foreach (int[] batch in Batches)
        {
            int[] pos = Enumerable.Range(p, batch.Length).ToArray();
            using (ITensor _ = model.Forward(batch, pos, deviceId: -1, kv, adapter: null, state)) { }
            p += batch.Length;
        }
        // Three speculative draft slots, then a verify over the same positions: the absorb must
        // overwrite them rather than append.
        for (int i = 0; i < 3; i++)
            using (ITensor _ = model.ForwardMtp(state, 9 - i, p + i)) { }
        Assert.Equal(p + 3, state.CurrentLength);
        int[] vpos = Enumerable.Range(p, Verify.Length).ToArray();
        using (ITensor _ = model.Forward(Verify, vpos, deviceId: -1, kv, adapter: null, state)) { }
        p += Verify.Length;
        return p;
    }

    private static unsafe float[] Download(nint devicePtr, int count)
    {
        var host = new float[count];
        fixed (float* h = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)h, devicePtr, (nuint)((long)count * sizeof(float))).ThrowOnError();
        return host;
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}[{i}] is not finite");
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: per-token {expected[i]:R} vs batched {actual[i]:R}");
        }
    }

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
}
