using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// The CPU MTP head stores its K/V in the trunk's KV element type, F16 by default (issue #1141, port of #1095): stored rows
/// are byte-identical to the format's own conversion, attention reads them in place and agrees with the F32 kernel run on the
/// dequantized rows, and the footprint shrinks by the format's ratio. Non-degenerate shape (2 KV heads, 8 query heads, hd 64).
/// </summary>
public sealed unsafe class CpuMtpKvFormatTests
{
    private const int Hidden = 8, KvHeads = 2, HeadDim = 64, NumHeads = 8, Stride = KvHeads * HeadDim, Rows = 37;

    private static float[] RandomRows(int seed, int rows)
    {
        var rng = new Random(seed);
        var a = new float[rows * Stride];
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(rng.NextDouble() * 4 - 2);
        return a;
    }

    private static float[] Decode(CpuMtpState st, ReadOnlySpan<byte> bytes)
    {
        var f = new float[Stride];
        fixed (byte* b = bytes)
        fixed (float* o = f)
        {
            switch (st.KvFormat)
            {
                case CpuMtpKvFormat.F32: MemoryMarshal.Cast<byte, float>(bytes).CopyTo(f); break;
                case CpuMtpKvFormat.F16:
                    for (int i = 0; i < Stride; i++) f[i] = (float)MemoryMarshal.Cast<byte, Half>(bytes)[i];
                    break;
                case CpuMtpKvFormat.Q8_0: KvQuantize.Q8_0ToF32(b, o, Stride); break;
                default: KvQuantize.Q4_0ToF32(b, o, Stride); break;
            }
        }
        return f;
    }

    [Theory]
    [InlineData(CpuMtpKvFormat.F16)]
    [InlineData(CpuMtpKvFormat.Q8_0)]
    [InlineData(CpuMtpKvFormat.Q4_0)]
    public void StoredRows_AreByteIdenticalToTheFormatsOwnConversion(CpuMtpKvFormat format)
    {
        using var st = new CpuMtpState(Hidden, KvHeads, HeadDim, 64, format);
        Assert.Equal(format, st.KvFormat);
        float[] k = RandomRows(1, Rows), v = RandomRows(2, Rows);
        st.BeginAbsorb(0, Rows);
        st.StoreKeyRows(0, k, Rows);
        st.StoreValueRows(0, v, Rows);
        st.EndAbsorb(Rows);

        var expected = new byte[st.RowBytes];
        for (int r = 0; r < Rows; r++)
        {
            fixed (float* src = &k[r * Stride])
            fixed (byte* dst = expected)
            {
                switch (format)
                {
                    case CpuMtpKvFormat.F16:
                    {
                        Span<Half> halves = MemoryMarshal.Cast<byte, Half>(expected.AsSpan());
                        for (int i = 0; i < Stride; i++)
                            halves[i] = (Half)src[i];
                    }
                        break;
                    case CpuMtpKvFormat.Q8_0: KvQuantize.F32ToQ8_0(src, dst, Stride); break;
                    default: KvQuantize.F32ToQ4_0(src, dst, Stride); break;
                }
            }
            Assert.True(expected.AsSpan().SequenceEqual(st.GetKeyRowBytes(r)), $"key row {r} differs from the format's own bytes");
        }
    }

    [Theory]
    [InlineData(CpuMtpKvFormat.F16)]
    [InlineData(CpuMtpKvFormat.Q8_0)]
    [InlineData(CpuMtpKvFormat.Q4_0)]
    public void Attention_ReadsTheStoredRowsInPlace_AndMatchesTheF32KernelOnTheDequantizedRows(CpuMtpKvFormat format)
    {
        using var st = new CpuMtpState(Hidden, KvHeads, HeadDim, 64, format);
        float[] k = RandomRows(3, Rows), v = RandomRows(4, Rows), q = RandomRows(5, NumHeads / KvHeads);
        // Reference: the same stored rows decoded to F32, through the original contiguous-span kernel.
        st.BeginAbsorb(0, Rows);
        st.StoreKeyRows(0, k, Rows);
        st.StoreValueRows(0, v, Rows);
        st.EndAbsorb(Rows);
        var kd = new float[Rows * Stride];
        var vd = new float[Rows * Stride];
        for (int r = 0; r < Rows; r++)
        {
            Decode(st, st.GetKeyRowBytes(r)).CopyTo(kd, r * Stride);
            Decode(st, st.GetValueRowBytes(r)).CopyTo(vd, r * Stride);
        }

        var expected = new float[NumHeads * HeadDim];
        Attention.Execute(q.AsSpan(0, NumHeads * HeadDim), kd, vd, expected, 1, Rows, NumHeads, KvHeads, HeadDim, Rows - 1, (int?)null);
        var actual = new float[NumHeads * HeadDim];
        st.AttendOver(q.AsSpan(0, NumHeads * HeadDim), actual, Rows, NumHeads, Rows - 1);

        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= 1e-5f, $"[{i}] ref {expected[i]} vs in-place {actual[i]}");
    }

    [Fact]
    public void Footprint_ShrinksByTheFormatsRatio()
    {
        long Bytes(CpuMtpKvFormat f) { using var s = new CpuMtpState(Hidden, KvHeads, HeadDim, 64, f); return s.KvBytesPerPosition; }
        long f32 = Bytes(CpuMtpKvFormat.F32), f16 = Bytes(CpuMtpKvFormat.F16), q8 = Bytes(CpuMtpKvFormat.Q8_0), q4 = Bytes(CpuMtpKvFormat.Q4_0);
        Assert.Equal(2L * Stride * 4, f32);
        Assert.Equal(f32 / 2, f16);
        Assert.Equal(2L * Stride / 32 * 34, q8);
        Assert.Equal(2L * Stride / 32 * 18, q4);
        Assert.True(f16 > q8 && q8 > q4);
    }

    [Fact]
    public void DefaultFormat_IsF16_NeverF32()
    {
        if (Environment.GetEnvironmentVariable(CpuMtpState.KvEnvVar) is not null) return;   // an A/B override is active
        using var st = new CpuMtpState(Hidden, KvHeads, HeadDim, 64);
        Assert.Equal(CpuMtpKvFormat.F16, st.KvFormat);
        Assert.Equal(CpuMtpKvFormat.F16, CpuMtpState.ResolveFormat(CpuMtpKvFormat.Q8_0, headDim: 48));   // q8_0 needs whole 32-blocks per head
        Assert.Equal(CpuMtpKvFormat.Q8_0, CpuMtpState.FromTrunk(KvCacheDType.Q8_0));
        Assert.Equal(CpuMtpKvFormat.Q4_0, CpuMtpState.FromTrunk(KvCacheDType.Q4_0));
        Assert.Null(CpuMtpState.FromTrunk(KvCacheDType.F32));
        Assert.Null(CpuMtpState.FromTrunk(null));
    }

    [Theory]
    [InlineData(CpuMtpKvFormat.F16)]
    [InlineData(CpuMtpKvFormat.Q8_0)]
    public void Growth_PreservesCommittedRows(CpuMtpKvFormat format)
    {
        using var st = new CpuMtpState(Hidden, KvHeads, HeadDim, 5000, format);
        Assert.Equal(1024, st.Capacity);
        float[] k = RandomRows(6, 1500);
        st.BeginAbsorb(0, 1500);   // grows past the first step
        Assert.True(st.Capacity >= 1500);
        st.StoreKeyRows(0, k, 1500);
        st.EndAbsorb(1500);
        var before = st.GetKeyRowBytes(1499).ToArray();
        st.EnsureCapacity(3000);
        Assert.True(before.AsSpan().SequenceEqual(st.GetKeyRowBytes(1499)));
        Assert.True(st.GetKeyRowBytes(2999).ToArray().All(b => b == 0), "grown rows start zeroed");
        Assert.Equal(2L * st.Capacity * st.RowBytes, st.AllocatedBytes);
    }
}
