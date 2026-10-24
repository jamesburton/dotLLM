using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Models.Quantization;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>Issue #866: ggml I32 (26) and BF16 (30) tensors through the GGUF reader, plus the NanoQuant composite loader on a synthetic file.</summary>
public sealed class GgufI32Bf16Tests : IDisposable
{
    private readonly List<string> _files = [];
    public void Dispose() { foreach (var f in _files) try { File.Delete(f); } catch { } }

    private string Write(GgufWriter w)
    {
        string path = Path.Combine(Path.GetTempPath(), $"i32-{Guid.NewGuid():N}.gguf");
        File.WriteAllBytes(path, w.Build());
        _files.Add(path);
        return path;
    }

    private static byte[] Bytes(int[] v) { var b = new byte[v.Length * 4]; Buffer.BlockCopy(v, 0, b, 0, b.Length); return b; }
    private static byte[] Bf(float[] v)
    {
        var b = new byte[v.Length * 2];
        for (int i = 0; i < v.Length; i++) { ushort h = (ushort)(BitConverter.SingleToUInt32Bits(v[i]) >> 16); b[2 * i] = (byte)h; b[2 * i + 1] = (byte)(h >> 8); }
        return b;
    }

    [Fact]
    public void ElementSizes()
    {
        Assert.Equal(26, (int)QuantizationType.I32);
        Assert.Equal(400, QuantizationType.I32.ComputeByteCount(100));
        Assert.Equal(200, QuantizationType.BF16.ComputeByteCount(100));
        Assert.Equal(7 * 4, DotLLM.Cpu.Kernels.Dequantize.RowByteSize(7, QuantizationType.I32));
        Assert.Equal(7 * 2, DotLLM.Cpu.Kernels.Dequantize.RowByteSize(7, QuantizationType.BF16));   // odd counts are legal for both
    }

    [Fact]
    public unsafe void I32AndBf16_RoundTrip_WithDequantAndRawAccess()
    {
        int[] words = [int.MinValue, -1, 0, 1, 0x12345678, int.MaxValue, 42];
        float[] f = [1f, -2.5f, 0.15625f];
        using var file = GgufFile.Open(Write(new GgufWriter().AddString("general.architecture", "llama")
            .AddTensor("w", [7], 26, Bytes(words))
            .AddTensor("s", [3], 30, Bf(f))));
        var w = file.TensorsByName["w"]; var s = file.TensorsByName["s"];
        Assert.Equal(QuantizationType.I32, w.QuantizationType);
        Assert.Equal(QuantizationType.BF16, s.QuantizationType);
        var raw = new ReadOnlySpan<int>((void*)file.TensorDataPointer(w), 7);
        Assert.True(raw.SequenceEqual(words));
        var deq = new float[7];
        DotLLM.Cpu.Kernels.Dequantize.ToFloat32(file.TensorDataPointer(w), 7, QuantizationType.I32, deq);
        Assert.Equal(42f, deq[6]); Assert.Equal(-1f, deq[1]);
        var sd = new float[3];
        DotLLM.Cpu.Kernels.Dequantize.ToFloat32(file.TensorDataPointer(s), 3, QuantizationType.BF16, sd);
        Assert.Equal(f, sd);   // exactly representable in bf16
    }

    [Theory]
    [InlineData(26, 4)]   // I32: 4 elements need 16 bytes, only 15 present
    [InlineData(30, 4)]   // BF16: needs 8 bytes
    public void TruncatedPayload_IsRejected(uint type, int elements)
    {
        int need = (int)((QuantizationType)type).ComputeByteCount(elements);
        var path = Write(new GgufWriter().AddString("general.architecture", "llama").AddTensor("t", [elements], type, new byte[need - 1]));
        Assert.Throws<InvalidDataException>(() => GgufFile.Open(path));
    }

    [Theory]
    [InlineData(24)]   // I8
    [InlineData(25)]   // I16
    [InlineData(27)]   // I64
    [InlineData(28)]   // F64
    public void OtherIntegerTypes_StayRejected_WithClearMessage(uint type)
    {
        var path = Write(new GgufWriter().AddString("general.architecture", "llama").AddTensor("t", [4], type, new byte[64]));
        var ex = Assert.Throws<NotSupportedException>(() => GgufFile.Open(path));
        Assert.Contains("unrecognized", ex.Message);
    }

    // ---- loader on a synthetic NanoQuant GGUF (dOut=4, dIn=40, r=33, k=1)

    private static GgufWriter Synthetic(Action<GgufWriter>? mutate = null, int version = 2, string bitOrder = "lsb_first", bool withSalient = true)
    {
        int dOut = 4, dIn = 40, r = 33, uw = 2, vw = 2;
        var rng = new Random(7);
        int[] u = Enumerable.Range(0, dOut * uw).Select(_ => rng.Next()).ToArray();
        int[] v = Enumerable.Range(0, r * vw).Select(_ => rng.Next()).ToArray();
        var w = new GgufWriter().AddString("general.architecture", "qwen3")
            .AddUInt32("quantization.nanoquant.version", (uint)version)
            .AddString("quantization.nanoquant.bit_order", bitOrder)
            .AddUInt32("quantization.nanoquant.positive_bit", 0)
            .AddString("quantization.nanoquant.outlier_format", "column_side_path")
            .AddTensor("blk.0.ffn_up.nq_u", [uw, dOut], 26, Bytes(u))
            .AddTensor("blk.0.ffn_up.nq_v", [vw, r], 26, Bytes(v))
            .AddTensor("blk.0.ffn_up.nq_scale_pre", [dIn], 30, Bf(Enumerable.Range(0, dIn).Select(i => i == 5 ? 0f : 0.5f).ToArray()))
            .AddTensor("blk.0.ffn_up.nq_scale_mid", [r], 30, Bf(Enumerable.Repeat(0.25f, r).ToArray()))
            .AddTensor("blk.0.ffn_up.nq_scale_post", [dOut], 30, Bf(Enumerable.Repeat(2f, dOut).ToArray()));
        if (withSalient)
            w.AddTensor("blk.0.ffn_up.nq_salient_idx", [1], 26, Bytes([5]))
             .AddTensor("blk.0.ffn_up.nq_salient_weight", [1, dOut], 1, new byte[dOut * 2]);
        mutate?.Invoke(w);
        return w;
    }

    [Fact]
    public void Loader_BuildsLayer_FromSyntheticFile()
    {
        using var file = GgufFile.Open(Write(Synthetic()));
        Assert.Equal(["blk.0.ffn_up"], NanoQuantLoader.FindBases(file));
        using var layer = NanoQuantLoader.Load(file, "blk.0.ffn_up");
        Assert.Equal((4, 40, 33, 1), (layer.DOut, layer.DIn, layer.R, layer.SalientCount));
    }

    [Fact]
    public void Loader_WithoutSalient_Loads()
    {
        using var file = GgufFile.Open(Write(Synthetic(withSalient: false)));
        using var layer = NanoQuantLoader.Load(file, "blk.0.ffn_up");
        Assert.Equal(0, layer.SalientCount);
    }

    [Fact]
    public void Loader_RejectsUnsupportedLayoutDeclarations()
    {
        using (var f = GgufFile.Open(Write(Synthetic(version: 3))))
            Assert.Throws<NotSupportedException>(() => NanoQuantLoader.Load(f, "blk.0.ffn_up"));
        using (var f = GgufFile.Open(Write(Synthetic(bitOrder: "msb_first"))))
            Assert.Throws<NotSupportedException>(() => NanoQuantLoader.Load(f, "blk.0.ffn_up"));
    }

    [Fact]
    public void Loader_RejectsInconsistentShapes()
    {
        // rank 33 needs 2 words per U row; claim 3 columns of words instead
        var bad = new GgufWriter().AddString("general.architecture", "qwen3").AddUInt32("quantization.nanoquant.version", 2)
            .AddTensor("x.nq_u", [3, 4], 26, new byte[48]).AddTensor("x.nq_v", [2, 33], 26, new byte[33 * 8])
            .AddTensor("x.nq_scale_pre", [40], 30, new byte[80]).AddTensor("x.nq_scale_mid", [33], 30, new byte[66])
            .AddTensor("x.nq_scale_post", [4], 30, new byte[8]);
        using var f = GgufFile.Open(Write(bad));
        Assert.Throws<InvalidDataException>(() => NanoQuantLoader.Load(f, "x"));
    }

    [Fact]
    public void Loader_RejectsWrongDtypeForSignWords()
    {
        var bad = new GgufWriter().AddString("general.architecture", "qwen3").AddUInt32("quantization.nanoquant.version", 2)
            .AddTensor("x.nq_u", [1, 1], 0, new byte[4]);   // F32, not I32
        using var f = GgufFile.Open(Write(bad));
        Assert.Throws<NotSupportedException>(() => NanoQuantLoader.Load(f, "x"));
    }
}
