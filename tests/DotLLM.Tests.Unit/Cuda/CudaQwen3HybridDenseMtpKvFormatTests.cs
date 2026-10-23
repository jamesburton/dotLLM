using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1095: the CUDA MTP head stores its KV in the trunk's KV type (F16 / Q8_0 / Q4_0 per side) and reads it in place with
/// the trunk's decode kernels, instead of keeping F32 rows (8 KiB/token on Bonsai 2) that a growth copied.
/// </summary>
/// <remarks>
/// <para>Gated on the real Bonsai 2 checkpoint (<c>DOTLLM_BONSAI_PQ2_0_GGUF</c>): the synthetic model's KV row is 16 elements,
/// below the 32-element quantization block and the tensor-core kernels' head_dim of 256, so it keeps the F32 fallback.</para>
/// <para>Every arm runs the same trunk prefill (fresh KV cache, reset recurrent state), so the head's pre-storage F32 K/V rows
/// are the same bits in each arm and only its storage and read differ.</para>
/// <para>Three oracles. (1) Bit identity: the fused quantized read gives the same draft logits as dequantizing the head's own
/// history into the F16 scratch and running the F16 tensor-core kernel, the same values read two ways (as #1094 did for the
/// trunk). (2) Storage: the head's stored rows are byte for byte the F32 arm's rows stored through the trunk's KV write, and
/// for F16 also the CPU's own half conversion of them. (3) Drift against the F32 head is reported, not asserted: the trunk
/// verifies every draft, so it changes acceptance only. The F32 head exists here only as that reference arm (#1107).</para>
/// <para>Mutants: passing the K dtype for both sides to the fused kernel breaks bit identity on the mixed pair; skipping the K
/// store or storing at the wrong position, or skipping K-norm/RoPE before the store, fails the storage oracle; dropping the
/// head from <c>ContextBytesPerToken</c> or counting it twice in place fails the bytes test.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpKvFormatTests
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";

    /// <summary>History before the drafts: past the tensor-core decode kernels' minimum depth (256), like a real round.</summary>
    private const int Prefix = 600;

    /// <summary>Chained draft steps per arm (a K=3 round).</summary>
    private const int Steps = 3;

    private readonly ITestOutputHelper _out;

    /// <summary>Creates the test class.</summary>
    /// <param name="output">xUnit output sink.</param>
    public CudaQwen3HybridDenseMtpKvFormatTests(ITestOutputHelper output) => _out = output;

    /// <summary>The quantized K/V pairs: both symmetric formats and a mixed one (which catches a K/V dtype swap).</summary>
    public static IEnumerable<object[]> QuantPairs() =>
    [
        [KvCacheDType.Q4_0, KvCacheDType.Q4_0],
        [KvCacheDType.Q8_0, KvCacheDType.Q8_0],
        [KvCacheDType.Q8_0, KvCacheDType.Q4_0],
    ];

    /// <summary>
    /// A quantized head: formats follow the trunk, the fused read matches the scratch reference bit for bit, the stored rows
    /// are the F32 rows in that format, and the drift against the F32 head is reported.
    /// </summary>
    /// <param name="keys">Trunk (and head) key storage.</param>
    /// <param name="values">Trunk (and head) value storage.</param>
    [SkippableTheory]
    [MemberData(nameof(QuantPairs))]
    public void QuantizedHead_FusedReadIsBitIdentical_RowsAreTheF32RowsQuantized(KvCacheDType keys, KvCacheDType values)
    {
        using var loaded = LoadModel(keys, values);
        var model = loaded.Model;
        Skip.IfNot(model.SupportsMtp, "checkpoint has no MTP head");

        // F32 head: the pre-#1095 storage, the drift baseline and the storage oracle's source rows.
        model.MtpKvF32ForTest = true;
        var f32 = RunArm(model, readRows: true);
        Assert.Equal(CudaMtpKvFormat.F32, f32.KeyFormat);

        model.MtpKvF32ForTest = false;
        Assert.Equal(ToFormat(keys), model.ResolveMtpKvFormats().Key);
        Assert.Equal(ToFormat(values), model.ResolveMtpKvFormats().Value);

        model.MtpFusedKvReadForTest = false;
        var scratch = RunArm(model, readRows: false);
        Assert.Equal(0, scratch.Fused);
        Assert.Equal(2 * Steps, scratch.Dequant);

        model.MtpFusedKvReadForTest = true;
        var fused = RunArm(model, readRows: true);
        Assert.Equal(ToFormat(keys), fused.KeyFormat);
        Assert.Equal(ToFormat(values), fused.ValueFormat);
        Assert.Equal(Steps, fused.Fused);
        Assert.Equal(0, fused.Dequant);

        for (int i = 0; i < Steps; i++)
            AssertBitIdentical($"{keys}/{values} draft step {i}", scratch.Logits[i], fused.Logits[i]);

        AssertRowsAreStoredLikeTheTrunk("K", model, f32.KeyRows!, fused.KeyRows!, ToFormat(keys));
        AssertRowsAreStoredLikeTheTrunk("V", model, f32.ValueRows!, fused.ValueRows!, ToFormat(values));

        ReportDrift($"{keys}/{values}", f32.Logits, fused.Logits);
    }

    /// <summary>An F16 trunk gives an F16 head, read directly by the tensor-core kernel; its rows are the F32 rows as halves.</summary>
    [SkippableFact]
    public void F16Head_RowsAreTheF32RowsAsHalves()
    {
        using var loaded = LoadModel(KvCacheDType.F32, KvCacheDType.F32);
        var model = loaded.Model;
        Skip.IfNot(model.SupportsMtp, "checkpoint has no MTP head");

        model.MtpKvF32ForTest = true;
        var f32 = RunArm(model, readRows: true);
        model.MtpKvF32ForTest = false;
        var f16 = RunArm(model, readRows: true);
        Assert.Equal(CudaMtpKvFormat.F16, f16.KeyFormat);
        Assert.Equal(CudaMtpKvFormat.F16, f16.ValueFormat);
        Assert.Equal(0, f16.Fused);
        Assert.Equal(0, f16.Dequant);

        AssertHalfRows("K", f32.KeyRows!, f16.KeyRows!);
        AssertHalfRows("V", f32.ValueRows!, f16.ValueRows!);
        AssertRowsAreStoredLikeTheTrunk("K", model, f32.KeyRows!, f16.KeyRows!, CudaMtpKvFormat.F16);
        AssertRowsAreStoredLikeTheTrunk("V", model, f32.ValueRows!, f16.ValueRows!, CudaMtpKvFormat.F16);
        ReportDrift("f16", f32.Logits, f16.Logits);
    }

    /// <summary>
    /// <see cref="CudaQwen3HybridDenseTransformerModel.ContextBytesPerToken"/> counts the head once, in its own format, when it
    /// grows in place: q4_0 is 2 x 576 B of rows + a 4 B iota entry, against 2 x (2 x 4 KiB + 4 B) for the old F32 count.
    /// </summary>
    [SkippableFact]
    public void ContextBytesPerToken_CountsTheHeadOnceInItsFormat()
    {
        using var loaded = LoadModel(KvCacheDType.Q4_0, KvCacheDType.Q4_0);
        var model = loaded.Model;
        Skip.IfNot(model.SupportsMtp, "checkpoint has no MTP head");
        Skip.IfNot(model.KvGrowsInPlace, "CUDA virtual memory unavailable on this device");

        long trunk = model.ContextBytesPerToken(mtp: false);
        long head = model.ContextBytesPerToken(mtp: true) - trunk;
        _out.WriteLine($"q4_0 trunk {trunk} B/token, head {head} B/token (total {trunk + head})");
        Assert.Equal(18 * 1024, trunk);
        Assert.Equal(2 * 576 + sizeof(int), head);

        model.MtpKvF32ForTest = true;   // the fallback, still grown in place: once, at F32 width
        Assert.Equal(2 * 4096 + sizeof(int), model.ContextBytesPerToken(mtp: true) - trunk);
    }

    private sealed record Arm(float[][] Logits, byte[]? KeyRows, byte[]? ValueRows,
                              CudaMtpKvFormat KeyFormat, CudaMtpKvFormat ValueFormat, int Fused, int Dequant);

    /// <summary>
    /// One independent sequence: an MTP prefill of <see cref="Prefix"/> tokens (last-row logits only), then
    /// <see cref="Steps"/> chained greedy draft steps. Optionally reads back the head's committed rows <c>[0, Prefix)</c>.
    /// </summary>
    private static Arm RunArm(CudaQwen3HybridDenseTransformerModel model, bool readRows)
    {
        int vocab = model.Config.VocabSize;
        model.ResetSequenceState();
        using var kv = model.CreateKvCache(Prefix + Steps + 8);
        using var mtp = (CudaMtpState)model.CreateMtpState(kv.MaxLength)!;
        int[] prompt = Tokens(Prefix);
        int token;
        using (ITensor t = model.Forward(prompt, Enumerable.Range(0, Prefix).ToArray(), deviceId: -1, kv,
                                         adapter: null, mtp, lastTokenLogitsOnly: true))
            token = ArgMax(Copy(t, vocab));

        byte[]? keyRows = readRows ? ReadDevice(mtp.KeyCacheDevicePtr, Prefix * mtp.KeyRowBytes) : null;
        byte[]? valueRows = readRows ? ReadDevice(mtp.ValueCacheDevicePtr, Prefix * mtp.ValueRowBytes) : null;

        int fused0 = model.DebugMtpFusedKvqCallsForTest;
        int dequant0 = model.DebugMtpKvDequantCallsForTest;
        var logits = new float[Steps][];
        for (int i = 0; i < Steps; i++)
        {
            using ITensor t = model.ForwardMtp(mtp, token, Prefix + i);
            logits[i] = Copy(t, vocab);
            token = ArgMax(logits[i]);
        }
        return new Arm(logits, keyRows, valueRows, mtp.KeyFormat, mtp.ValueFormat,
            model.DebugMtpFusedKvqCallsForTest - fused0, model.DebugMtpKvDequantCallsForTest - dequant0);
    }

    /// <summary>
    /// The head's stored rows are, byte for byte, the F32 arm's rows stored through the trunk's own KV write (F32 -> F16 ->
    /// quantize-on-write). The quantizer is built with fast math, so a CPU replica cannot be exact; the GPU write path can.
    /// </summary>
    private void AssertRowsAreStoredLikeTheTrunk(string label, CudaQwen3HybridDenseTransformerModel model,
                                                 byte[] f32Rows, byte[] stored, CudaMtpKvFormat format)
    {
        byte[] want = model.StoreKvRowsForTest(MemoryMarshal.Cast<byte, float>(f32Rows), format);
        Assert.Equal(want.Length, stored.Length);
        int diff = 0;
        for (int i = 0; i < want.Length; i++)
            if (want[i] != stored[i]) diff++;
        _out.WriteLine($"{label} ({format}): {stored.Length} bytes, differing = {diff}");
        Assert.Equal(0, diff);
    }

    /// <summary>F16 rows are the F32 rows converted to half (round to nearest even), bit for bit.</summary>
    private void AssertHalfRows(string label, byte[] f32Rows, byte[] f16Rows)
    {
        var want = MemoryMarshal.Cast<byte, float>(f32Rows);
        var got = MemoryMarshal.Cast<byte, Half>(f16Rows);
        Assert.Equal(want.Length, got.Length);
        int diff = 0;
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.HalfToInt16Bits((Half)want[i]) != BitConverter.HalfToInt16Bits(got[i])) diff++;
        _out.WriteLine($"F16 {label}: {want.Length} elements, bit-diff = {diff}");
        Assert.Equal(0, diff);
    }

    private void ReportDrift(string label, float[][] f32, float[][] other)
    {
        for (int i = 0; i < Steps; i++)
        {
            double maxAbs = 0, maxRef = 0;
            for (int j = 0; j < f32[i].Length; j++)
            {
                Assert.True(float.IsFinite(other[i][j]), $"{label} step {i}: non-finite logit at {j}");
                maxAbs = Math.Max(maxAbs, Math.Abs(other[i][j] - f32[i][j]));
                maxRef = Math.Max(maxRef, Math.Abs(f32[i][j]));
            }
            _out.WriteLine($"{label} vs F32 head, step {i}: max |dlogit| = {maxAbs:G4} (max |logit| {maxRef:G4}), " +
                           $"argmax {ArgMax(f32[i])} vs {ArgMax(other[i])}");
        }
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

    private static CudaMtpKvFormat ToFormat(KvCacheDType d) => d switch
    {
        KvCacheDType.Q8_0 => CudaMtpKvFormat.Q8_0,
        KvCacheDType.Q4_0 => CudaMtpKvFormat.Q4_0,
        _ => CudaMtpKvFormat.F16,
    };

    private static int[] Tokens(int n)
    {
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = 1 + ((i * 7919) + i / 13) % 30000;
        return t;
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    private static unsafe float[] Copy(ITensor t, int count)
    {
        int rows = t.Shape[0];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * count, count).ToArray();
    }

    private static unsafe byte[] ReadDevice(nint device, long bytes)
    {
        var host = new byte[bytes];
        fixed (byte* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, device, (nuint)bytes).ThrowOnError();
        return host;
    }

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
