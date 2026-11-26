using System.IO.MemoryMappedFiles;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// FP8 (E4M3FN) rows of the n-gram table (Edge0 <c>ple-fp8.gguf</c>, #1974): the row decoder against a scalar reference written from the
/// OCP FP8 definition, at all three gather sites (plain, cache read-through, cache hit), with the table never widened.
/// </summary>
public sealed unsafe class Qwen4ExpPleFp8Tests(ITestOutputHelper output) : IDisposable
{
    private const float RealScale = 0.00019931793212890625f;   // strata.ple.scale of Edge0-145B-q3k's ple-fp8.gguf
    private readonly List<IDisposable> _cleanup = [];
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-ple-fp8-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        foreach (var d in _cleanup) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>
    /// The scalar reference, written from the definition and sharing nothing with <see cref="PleRowCodec"/> (which builds floats from
    /// bit fields): value = (-1)^s * 2^(e-7) * (1 + m/8) for e &gt; 0, (-1)^s * 2^-6 * (m/8) for e = 0; S.1111.111 is NaN (read as 0).
    /// </summary>
    private static float Reference(byte b, float scale)
    {
        int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
        if (e == 15 && m == 7) return (s == 1 ? -0f : 0f) * scale;   // strata: f = 0, then v = s ? -f : f (so 0xFF is -0)
        double mag = e == 0 ? Math.Pow(2, -6) * (m / 8.0) : Math.Pow(2, e - 7) * (1.0 + m / 8.0);
        return (float)(s == 1 ? -mag : mag) * scale;
    }

    private static void AssertBitEqual(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{what}: element {i} differs ({expected[i]:R} vs {actual[i]:R})");
    }

    private sealed class MappedBytes : IDisposable
    {
        private readonly MemoryMappedFile _mmf;
        private readonly MemoryMappedViewAccessor _view;
        public nint Ptr { get; }
        public byte[] Bytes { get; }
        public MappedBytes(string path, int length, int seed)
        {
            Bytes = new byte[length];
            new Random(seed).NextBytes(Bytes);
            File.WriteAllBytes(path, Bytes);
            _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            Ptr = (nint)p;
        }
        public void Dispose() { _view.SafeMemoryMappedViewHandle.ReleasePointer(); _view.Dispose(); _mmf.Dispose(); }
    }

    [Fact]
    public void Anchors_FromTheOcpDefinition()
    {
        // Literal values (not computed by either implementation): bias 7, 3 mantissa bits, max finite 448, min subnormal 2^-9.
        var anchors = new (byte Code, float Value)[]
        {
            (0x00, 0f), (0x38, 1f), (0x40, 2f), (0x30, 0.5f), (0x39, 1.125f), (0x3C, 1.5f), (0x7E, 448f), (0x7D, 416f),
            (0x08, 0.015625f),                // smallest normal 2^-6
            (0x01, 0.001953125f),             // smallest subnormal 2^-9
            (0x07, 0.013671875f),             // largest subnormal 7 * 2^-9
            (0xB8, -1f), (0xFE, -448f), (0x7F, 0f), (0xFF, 0f),   // NaN codes read as 0 (edge0-strata ngram.cpp kFp8)
        };
        foreach (var (code, value) in anchors)
        {
            Assert.Equal(value, PleRowCodec.E4M3FnToSingle(code));
            Assert.Equal(value, Reference(code, 1f));   // the reference agrees with the literals too, so it is a usable oracle below
        }
        Assert.True(float.IsNegative(PleRowCodec.E4M3FnToSingle(0x80)) && PleRowCodec.E4M3FnToSingle(0x80) == 0f, "0x80 is -0");
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(RealScale)]
    [InlineData(3.7e-3f)]
    public void AllCodes_MatchTheScalarReference_BitForBit(float scale)
    {
        var bytes = new byte[256];
        for (int i = 0; i < 256; i++) bytes[i] = (byte)i;
        var got = new float[256];
        fixed (byte* p = bytes) PleRowCodec.Decode((nint)p, 256, QuantizationType.I8, scale, got);
        var want = new float[256];
        for (int i = 0; i < 256; i++) want[i] = Reference((byte)i, scale);
        AssertBitEqual(want, got, $"all 256 codes, scale {scale:R}");
        // the scale really is the table scale (a decode that ignored it would still pass the anchors at scale 1)
        Assert.Equal(1f * scale, got[0x38]);
    }

    [Fact]
    public void Fp8Table_NeedsAPositiveScale()
    {
        Assert.Throws<ArgumentException>(() => PleRowCodec.Validate(QuantizationType.I8, 0f));
        Assert.Throws<ArgumentException>(() => PleRowCodec.Validate(QuantizationType.I8, float.NaN));
        PleRowCodec.Validate(QuantizationType.I8, RealScale);
        PleRowCodec.Validate(QuantizationType.IQ4_NL, 0f);   // other formats ignore it
        Assert.Throws<ArgumentException>(() => new PleRowPrefetcher(0, QuantizationType.I8, 10, 160, PleTableOptions.Disabled));
    }

    [Theory]
    [InlineData(PleTableMode.Mmap, PleHintMode.Off)]
    [InlineData(PleTableMode.Mmap, PleHintMode.OsThenTouch)]
    [InlineData(PleTableMode.Ram, PleHintMode.Off)]
    [InlineData(PleTableMode.Ram, PleHintMode.Touch)]
    public void AllGatherSites_DecodeFp8_LikeTheReference_AndPrefetchChangesNothing(PleTableMode mode, PleHintMode hint)
    {
        const int rowDim = 160;
        const long tableRows = 4000;
        Directory.CreateDirectory(_dir);
        var table = new MappedBytes(Path.Combine(_dir, $"t-{mode}-{hint}.bin"), (int)(tableRows * rowDim), seed: 5);
        _cleanup.Add(table);
        var rng = new Random(9);
        var rows = new long[600];
        for (int i = 0; i < rows.Length; i++) rows[i] = rng.NextInt64(tableRows);
        rows[0] = 0; rows[^1] = tableRows - 1;
        // repeats so the Ram arm has cache hits on the second pass; a row gathered twice in one request too
        rows[10] = rows[3]; rows[11] = rows[3];

        var want = new float[rows.Length * rowDim];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rowDim; j++) want[i * rowDim + j] = Reference(table.Bytes[rows[i] * rowDim + j], RealScale);

        // site 1: plain GatherRows
        var plain = new float[want.Length];
        Qwen4ExpPle.GatherRows(table.Ptr, QuantizationType.I8, tableRows, rowDim, rows, plain, RealScale);
        AssertBitEqual(want, plain, "GatherRows");

        // sites 2 and 3 (and site 1 again when there is no cache): the prefetcher, first gather = read-through / plain, second = cache hits
        var opts = new PleTableOptions { Mode = mode, Hint = hint, CacheMegabytes = mode == PleTableMode.Ram ? 1 : 0, TouchThreads = 2 };
        using var pf = new PleRowPrefetcher(table.Ptr, QuantizationType.I8, tableRows, rowDim, opts, RealScale);
        var first = new float[want.Length];
        pf.Gather(pf.Begin(rows), rows, first);
        AssertBitEqual(want, first, $"prefetcher first gather ({mode}/{hint})");
        var second = new float[want.Length];
        pf.Gather(pf.Begin(rows), rows, second);
        AssertBitEqual(want, second, $"prefetcher second gather ({mode}/{hint})");
        if (mode == PleTableMode.Ram) Assert.True(pf.Stats.CacheHits > 0, "the cache-hit arm did not run");
    }

    [Fact]
    public void Fp8Table_IsNotWidened_RowBytesAreOneBytePerElement()
    {
        // The prefetcher's row cache stores RAW rows: an FP8 row costs 160 B (an F16 widening would be 320 B, F32 640 B).
        Assert.Equal(160, Dequantize.RowByteSize(160, QuantizationType.I8));
        using var pf = new PleRowPrefetcher(0, QuantizationType.I8, 100, 160, new PleTableOptions { Mode = PleTableMode.Ram, CacheMegabytes = 1 }, RealScale);
        Assert.Equal(1024 * 1024 / 160, pf.CacheCapacityRows);
    }

    // ── the real file ──────────────────────────────────────────────────────

    private static FixtureLocation LocateFile() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_PLE_FP8", "Edge0", "Edge0-145B-q3k", "ple-fp8.gguf");

    [SkippableFact]
    public void RealFile_HeaderEncodingAndRowsMatchTheReference()
    {
        var fx = LocateFile();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k/ple-fp8.gguf (51.2 GB)"));
        using var gguf = GgufFile.Open(fx.Path!);
        var (table, rows, scale) = Fp8PleTable.Open(gguf, 160);
        output.WriteLine($"rows {rows}, scale {scale:R}");
        Assert.Equal(320_001_536L, rows);
        Assert.Equal(RealScale, scale);

        // Read the raw bytes through a plain FileStream (no mapping, no dotLLM parser): this file is a one-tensor GGUF whose
        // tensor data runs to the end of the file, so row r starts at (length - rows * 160) + r * 160.
        using var fs = new FileStream(fx.Path!, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        long dataStart = fs.Length - rows * 160;
        Assert.Equal(0, dataStart % 32);   // GGUF default alignment: a sanity check on the arithmetic

        var rng = new Random(2026);
        var pick = new long[40];
        for (int i = 0; i < pick.Length; i++) pick[i] = rng.NextInt64(rows);
        pick[0] = 0; pick[1] = rows - 1;
        var want = new float[pick.Length * 160];
        var buf = new byte[160];
        long nonzero = 0, nanCodes = 0;
        for (int i = 0; i < pick.Length; i++)
        {
            fs.Position = dataStart + pick[i] * 160;
            fs.ReadExactly(buf);
            for (int j = 0; j < 160; j++)
            {
                want[i * 160 + j] = Reference(buf[j], scale);
                if (buf[j] != 0) nonzero++;
                if ((buf[j] & 0x7F) == 0x7F) nanCodes++;
            }
        }
        Assert.True(nonzero > pick.Length * 160 / 4, "the sampled rows are mostly zero bytes: this is not the table");
        Assert.Equal(0, nanCodes);   // "never in a real table"

        var got = new float[want.Length];
        Qwen4ExpPle.GatherRows(table, QuantizationType.I8, rows, 160, pick, got, scale);
        AssertBitEqual(want, got, "real rows");
        double rms = Math.Sqrt(got.Sum(v => (double)v * v) / got.Length);
        output.WriteLine($"sampled rows: rms {rms:G4}, max |v| {got.Max(MathF.Abs):G4}");
        Assert.InRange(rms, 1e-4, 1.0);
    }
}
