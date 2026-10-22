using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1025: an MTP prefill must be tiled like a plain one. <c>ForwardCore</c> used to skip the
/// #494 tile loop whenever an MTP capture was active, so an MTP request's prefill sized every
/// per-token activation buffer for the WHOLE prompt (~600 KB/token on Bonsai 2 27B). An agent's
/// second request, the first to explore the MTP arm of the adaptive gate, oversubscribed a 12 GB
/// card at 5.8k tokens, spilled to shared memory and prefilled for tens of minutes.
/// </summary>
/// <remarks>
/// <para>Two properties. <b>The tile bound holds</b>: after a 12-token MTP prefill in tiles of 4
/// the forward state's per-token capacity is 4 rows, not 16 (capacity rounds to a power of two at
/// this scale, so the two are distinguishable). <b>Tiling is invisible to MTP</b>: the trunk's last-row
/// logits, the MTP head's KV-cache, its pending hidden row and the next draft's logits all match the
/// single call. Each tile captures and absorbs its own rows and seeds the carry from its last row,
/// which is exactly the sequence of calls a decode/verify loop already makes.</para>
/// <para>The tolerance is the same per-M cuBLAS bar as <see cref="CudaQwen3HybridDensePrefillChunkingTest"/>,
/// and the dp4a path is pinned off in both arms for the same reason (#500).</para>
/// <para>Mutant: restoring the <c>mtpCapture is not null</c> gate fails
/// <see cref="MtpPrefill_IsTiled_ForwardStateCapacityBoundedByTile"/>. Passing <c>mtpCapture: null</c>
/// to the tiles, or skipping the capture on non-final tiles, leaves the head's KV rows for those
/// tiles at zero or misaligns the pairing, and that fails the parity test.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpPrefillTilingTests : IDisposable
{
    private const int SeqLen = 12;
    private const int TileTokens = 4;   // 12 tokens => 3 tiles

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseMtpPrefillTilingTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mtp-tiling-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void MtpPrefill_IsTiled_ForwardStateCapacityBoundedByTile()
    {
        string ptxDir = Setup();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "cap.gguf"), withMtp: true);

        var run = Run(path, ptxDir, tile: TileTokens);

        Assert.Equal(TileTokens, run.ForwardStateRows);
        _out.WriteLine($"MTP prefill of {SeqLen} tokens in tiles of {TileTokens}: forward state holds {run.ForwardStateRows} rows.");
    }

    [SkippableFact]
    public void TiledMtpPrefill_MatchesSingleCall()
    {
        string ptxDir = Setup();
        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "parity.gguf"), withMtp: true);

        var single = Run(path, ptxDir, tile: 0);
        var tiled = Run(path, ptxDir, tile: TileTokens);

        // The single call really was single, and the tiled one really was tiled: otherwise this
        // compares a path with itself.
        Assert.Equal(16, single.ForwardStateRows);
        Assert.Equal(TileTokens, tiled.ForwardStateRows);

        Assert.Equal(SeqLen, single.MtpLength);
        Assert.Equal(SeqLen, tiled.MtpLength);
        AssertClose(single.LastRowLogits, tiled.LastRowLogits, "trunk last-row logits");
        AssertClose(single.Keys, tiled.Keys, "MTP K");
        AssertClose(single.Values, tiled.Values, "MTP V");
        AssertClose(single.Pending, tiled.Pending, "MTP pending hidden");
        AssertClose(single.DraftLogits, tiled.DraftLogits, "next draft logits");

        // Guard against a vacuous pass: every absorbed position wrote real K rows.
        int stride = single.Keys.Length / SeqLen;
        for (int p = 0; p < SeqLen; p++)
            Assert.Contains(tiled.Keys.AsSpan(p * stride, stride).ToArray(), v => v != 0f);
        _out.WriteLine($"{SeqLen}-token MTP prefill, single vs {TileTokens}-token tiles: logits, head KV, pending and draft logits within tolerance.");
    }

    private sealed record Result(int ForwardStateRows, int MtpLength, float[] LastRowLogits,
                                 float[] Keys, float[] Values, float[] Pending, float[] DraftLogits);

    private static unsafe Result Run(string path, string ptxDir, int tile)
    {
        CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = tile;
        bool? dp4aWas = CudaSmallSGemvDispatch.Dp4aOverride;
        CudaSmallSGemvDispatch.Dp4aOverride = false;
        try
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using var state = (CudaMtpState)model.CreateMtpState()!;

            int[] tokens = new int[SeqLen];
            for (int i = 0; i < SeqLen; i++) tokens[i] = 1 + (i * 5) % (config.VocabSize - 1);
            int[] positions = [.. Enumerable.Range(0, SeqLen)];

            float[] last;
            using (ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv, adapter: null, state,
                                                  lastTokenLogitsOnly: true))
            {
                Assert.Equal(1, logits.Shape[0]);
                last = new ReadOnlySpan<float>((void*)logits.DataPointer, config.VocabSize).ToArray();
            }

            int n = state.CurrentLength * state.KvStride;
            // Issue #1095: the head stores its KV in the trunk's type (F16 on this synthetic F16 trunk); widen on the host to compare.
            float[] keys = DownloadRows(state.KeyCacheDevicePtr, n, state.KeyFormat);
            float[] values = DownloadRows(state.ValueCacheDevicePtr, n, state.ValueFormat);
            float[] pending = Download(state.PendingHiddenDevicePtr, config.HiddenSize);
            using ITensor draft = model.ForwardMtp(state, 3, SeqLen);
            float[] draftLogits = new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize).ToArray();
            return new Result(model.ForwardStateSequenceCapacity, n / state.KvStride, last, keys, values, pending, draftLogits);
        }
        finally
        {
            CudaQwen3HybridDenseTransformerModel.PrefillChunkOverride = null;
            CudaSmallSGemvDispatch.Dp4aOverride = dp4aWas;
        }
    }

    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        const float AbsTol = 1e-4f;
        const float RelTol = 1e-3f;
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what}[{i}] is not finite");
            float tol = AbsTol + RelTol * MathF.Abs(expected[i]);
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"{what}[{i}]: single call {expected[i]:R} vs tiled {actual[i]:R} (tolerance {tol})");
        }
    }

    /// <summary>Downloads <paramref name="count"/> head-cache elements stored as F32 or F16 and returns them as floats.</summary>
    private static unsafe float[] DownloadRows(nint devicePtr, int count, CudaMtpKvFormat format)
    {
        if (format == CudaMtpKvFormat.F32)
            return Download(devicePtr, count);
        Assert.Equal(CudaMtpKvFormat.F16, format);
        var halves = new Half[count];
        fixed (Half* h = halves)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)h, devicePtr, (nuint)((long)count * sizeof(ushort))).ThrowOnError();
        return Array.ConvertAll(halves, x => (float)x);
    }

    private static unsafe float[] Download(nint devicePtr, int count)
    {
        var host = new float[count];
        fixed (float* h = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)h, devicePtr, (nuint)((long)count * sizeof(float))).ThrowOnError();
        return host;
    }

    private static string Setup()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        return ptxDir!;
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
