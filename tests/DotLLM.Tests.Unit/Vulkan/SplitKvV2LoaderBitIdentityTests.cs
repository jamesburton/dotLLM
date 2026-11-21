using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using static DotLLM.Tests.Unit.Vulkan.VulkanDenseQ8KvKernelTests;
using static DotLLM.Tests.Unit.Vulkan.VulkanQ8KvAttentionKernelTests;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1926 E1: the v2 grouped split-KV decode pass with packed staged rows (E1a: Q8_0 / Q4_0 / mixed rows held as their words,
/// converted at use) and the group-size specialization constant (E1b) computes BIT-IDENTICAL output to the retained pre-change body
/// (<c>attention_f32_splitkv_gqa_v2*_pre1926.spv</c>) on the same buffers.
/// </summary>
/// <remarks>
/// <para>
/// Every shape has KV heads &gt; 1, a group &gt; 1 and distinct per-head, per-key data; the group sizes cover every GROUP value the host
/// can resolve (2, 3, 4, 5 = 9 split 5 + 4, 6 = 12 split 6 + 6, 7, 8), and the lengths cover one tile minus / plus one, a ragged last
/// split and a long multi-tile split. Each case runs the specialised pipeline AND the GROUP = 8 control, and asserts which GROUP the
/// dispatch bound. A pipeline built without specialization data must resolve to 8 (the safe default).
/// </para>
/// <para>
/// Bit identity is the right bar: E1 moves only where the same operations on the same values happen. Any drift is a bug. The data
/// blocks have distinct per-block scales (<see cref="VulkanQ8KvAttentionKernelTests.Rand"/>), so a neighbouring-block scale read moves
/// the result.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class SplitKvV2LoaderBitIdentityTests
{
    public static TheoryData<VulkanKvStorage, int, int, int, int> Cases()
    {
        var d = new TheoryData<VulkanKvStorage, int, int, int, int>();
        (int H, int Kv, int Hd)[] shapes = [(16, 8, 128), (24, 8, 128), (32, 8, 128), (72, 8, 128), (96, 8, 128), (14, 2, 64), (32, 4, 128)];
        int[] lengths = [65, 127, 128, 129, 1000, 4097];
        foreach (var st in new[] { VulkanKvStorage.Q8_0, VulkanKvStorage.Q4_0, VulkanKvStorage.Q8_0K_Q4_0V, VulkanKvStorage.Q4_0K_Q8_0V, VulkanKvStorage.F16 })
            foreach (var (h, kv, hd) in shapes)
                foreach (int n in lengths)
                {
                    // Keep the theory to a sane size: every storage x shape at two lengths, the long tail on the Qwen3-4B shape.
                    bool keep = n is 129 or 1000 || (h == 32 && kv == 8);
                    if (keep) d.Add(st, h, kv, hd, n);
                }
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void V2Pass_MatchesPreChangeBody_BitForBit(VulkanKvStorage storage, int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, storage.ShaderName("attention_f32_splitkv_gqa_v2") + "_pre1926.spv")), "pre-change control SPIR-V absent");
        using var device = VulkanDevice.Create();
        using var now = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage, null, "");
        using var pre = VulkanSplitKvAttentionKernel.Create(device, spvDir, storage, null, "_pre1926");
        now.PreferGqa = pre.PreferGqa = true;
        Skip.IfNot(now.UsesGqa2(numHeads, numKvHeads, headDim) && pre.UsesGqa2(numHeads, numKvHeads, headDim), "v2 pass unavailable on this device");
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, numHeads));

        var rng = new Random(0x1926 + seqKv * 31 + numHeads * 7 + (int)storage);
        int stride = numKvHeads * headDim;
        float[] q = Rand(rng, numHeads * headDim, 0.5f);
        float[] k = Rand(rng, seqKv * stride, 0.5f), v = Rand(rng, seqKv * stride, 0.5f);
        byte[] kb = Encode(storage.KeyFormat(), k), vb = Encode(storage.ValueFormat(), v);

        using var bQ = device.Allocate(q.Length * 4L);
        using var bK = device.Allocate(kb.Length);
        using var bV = device.Allocate(vb.Length);
        using var oNow = device.Allocate(q.Length * 4L);
        using var oCtl = device.Allocate(q.Length * 4L);
        using var oPre = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kb, bK); device.Upload(vb, bV);

        int g = numHeads / numKvHeads, subs = VulkanSplitKvAttentionKernel.GqaSubGroups(numHeads, numKvHeads);
        int expectGroup = (g + subs - 1) / subs;
        long before = now.GqaDispatches;
        now.PreferGroupSpec = true;
        now.Launch(bQ, bK, bV, oNow, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);
        Assert.Equal(expectGroup, now.LastGqa2Group);
        now.PreferGroupSpec = false;   // the GROUP = 8 control (the pipeline built at load)
        now.Launch(bQ, bK, bV, oCtl, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);
        Assert.Equal(VulkanSplitKvAttentionKernel.GqaMaxGroup, now.LastGqa2Group);
        Assert.Equal(before + 2, now.GqaDispatches);
        pre.Launch(bQ, bK, bV, oPre, 1, seqKv, numHeads, numKvHeads, headDim, seqKv - 1);

        uint[] a = new uint[q.Length], c = new uint[q.Length], r = new uint[q.Length];
        float[] tmp = new float[q.Length];
        device.Download(oNow, tmp); MemoryMarshal.Cast<float, uint>(tmp).CopyTo(a);
        device.Download(oCtl, tmp); MemoryMarshal.Cast<float, uint>(tmp).CopyTo(c);
        device.Download(oPre, tmp); MemoryMarshal.Cast<float, uint>(tmp).CopyTo(r);
        Assert.Contains(r, x => x != 0u);   // a degenerate (all-zero) output could match anything
        // E1a (the GROUP = 8 pipeline: packed rows, same body shape): bit for bit.
        for (int i = 0; i < q.Length; i++)
            Assert.True(c[i] == r[i], $"[{i}] GROUP=8 {BitConverter.UInt32BitsToSingle(c[i]):R} vs pre-change {BitConverter.UInt32BitsToSingle(r[i]):R}");
        // E1b (the specialised pipeline): the same operations per head, but a differently-unrolled module, and the driver compiler
        // contracts / schedules the GROUP = 4 module's dot products differently (measured: 1-2 ULP of the row amplitude on the Qwen3-4B
        // shape; every other group size is bit-identical). Bound it at 1e-6 of the row amplitude: a dropped head, wrong half-word or
        // neighbouring-block scale moves results by percent.
        float amp = 0;
        for (int i = 0; i < q.Length; i++) amp = MathF.Max(amp, MathF.Abs(BitConverter.UInt32BitsToSingle(r[i])));
        int differ = 0;
        for (int i = 0; i < q.Length; i++)
        {
            float x = BitConverter.UInt32BitsToSingle(a[i]), y = BitConverter.UInt32BitsToSingle(r[i]);
            if (a[i] != r[i]) differ++;
            Assert.True(MathF.Abs(x - y) <= 1e-6f * amp, $"[{i}] GROUP={expectGroup} {x:R} vs pre-change {y:R} (amplitude {amp:G3})");
        }
        if (expectGroup != 4)
            Assert.True(differ == 0, $"GROUP={expectGroup}: {differ} outputs not bit-identical (only GROUP=4 is known to drift)");
    }

    public static TheoryData<int, int, int, int, int> TqCases()
    {
        var d = new TheoryData<int, int, int, int, int>();
        for (int bits = 1; bits <= 8; bits++)
        {
            d.Add(bits, 32, 8, 128, 1000);
            d.Add(bits, 14, 2, 64, 129);
        }
        d.Add(4, 32, 8, 128, 4097);
        d.Add(3, 32, 8, 128, 4097);
        d.Add(7, 32, 4, 128, 777);
        return d;
    }

    /// <summary>
    /// #1926 E3 steps 1-2: the TurboQuant v2 twin with the centroid table in shared memory and format-sized code-word windows is
    /// bit-identical to the pre-change loader, for every code width (3, 5, 6, 7 straddle a word; 7 needs a 3-word window), both head
    /// dims, and a random guard word after the last vector (a window that took guard bits into a code would move the output).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(TqCases))]
    public void V2TqPass_MatchesPreChangeBody_BitForBit(int bits, int numHeads, int numKvHeads, int headDim, int seqKv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "attention_f32_splitkv_gqa_v2_kvtq_pre1926.spv")), "pre-change control SPIR-V absent");
        using var device = VulkanDevice.Create();
        var l = new DotLLM.Core.Attention.TurboQuantLayout(headDim, numKvHeads, bits, DotLLM.Core.Attention.TurboQuantLayout.DefaultNorm);
        using var now = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.Tq, l, "");
        using var pre = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.Tq, l, "_pre1926");
        now.PreferGqa = pre.PreferGqa = true;
        Skip.IfNot(now.UsesGqa2(numHeads, numKvHeads, headDim), "v2 pass unavailable on this device");
        var rng = new Random(0x19263 + bits * 17 + seqKv + headDim);
        float[] q = Rand(rng, numHeads * headDim, 2f);
        using var tq = TqFixture.Create(device, rng, l, seqKv);
        using var bQ = device.Allocate(q.Length * 4L);
        using var oNow = device.Allocate(q.Length * 4L);
        using var oCtl = device.Allocate(q.Length * 4L);
        using var oPre = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ);
        void Run(VulkanSplitKvAttentionKernel kern, VulkanDevice.Buffer o)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            kern.Record(ctx.CommandBuffer, bQ, tq.K, tq.V, o, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: seqKv - 1, tq: tq.Operands);
            ctx.SubmitAndWait();
        }
        now.PreferGroupSpec = false; Run(now, oCtl);
        now.PreferGroupSpec = true; Run(now, oNow);
        Run(pre, oPre);
        float[] a = new float[q.Length], c = new float[q.Length], r = new float[q.Length];
        device.Download(oNow, a); device.Download(oCtl, c); device.Download(oPre, r);
        float amp = r.Max(MathF.Abs);
        Assert.True(amp > 1e-3f, "degenerate reference");
        for (int i = 0; i < q.Length; i++)
            Assert.True(BitConverter.SingleToUInt32Bits(c[i]) == BitConverter.SingleToUInt32Bits(r[i]), $"[{i}] tq{bits} GROUP=8 {c[i]:R} vs pre-change {r[i]:R}");
        for (int i = 0; i < q.Length; i++)
            Assert.True(MathF.Abs(a[i] - r[i]) <= 1e-6f * amp, $"[{i}] tq{bits} GROUP={now.LastGqa2Group} {a[i]:R} vs pre-change {r[i]:R}");
    }

    /// <summary>The cache bytes of one side in <paramref name="format"/>: the C# reference quantisers (the layout contract of kv_load.glsl).</summary>
    internal static byte[] Encode(VulkanKvStorage format, float[] x) => format switch
    {
        VulkanKvStorage.Q8_0 => QuantizeRef(x),
        VulkanKvStorage.Q4_0 => QuantizeQ4Ref(x),
        VulkanKvStorage.F16 => MemoryMarshal.AsBytes(x.Select(f => BitConverter.HalfToUInt16Bits((Half)f)).ToArray().AsSpan()).ToArray(),
        VulkanKvStorage.F32 => MemoryMarshal.AsBytes(x.AsSpan()).ToArray(),
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    };
}

/// <summary>
/// Random TurboQuant cache buffers for kernel tests and benches (#1926): uniformly random code words (every code value is valid, so
/// every centroid index is used), per-vector F16 norms spread over 0.5-3 (distinct per key and per head), a random guard word, and a
/// sorted random centroid table. Not an encoder: the kernels under test only read codes in place.
/// </summary>
internal sealed class TqFixture : IDisposable
{
    public required VulkanDevice.Buffer K { get; init; }
    public required VulkanDevice.Buffer V { get; init; }
    public required VulkanDevice.Buffer KNorm { get; init; }
    public required VulkanDevice.Buffer VNorm { get; init; }
    public required VulkanDevice.Buffer Cent { get; init; }
    public TqKvOperands Operands => new(KNorm, VNorm, Cent);
    public long CodeBytes { get; init; }

    public static TqFixture Create(VulkanDevice device, Random rng, DotLLM.Core.Attention.TurboQuantLayout l, int seqKv, bool deviceLocal = false)
    {
        long vecs = (long)seqKv * l.NumKvHeads;
        long words = vecs * l.CodeUintsPerVec + 1;
        VulkanDevice.Buffer Codes()
        {
            var w = new uint[words];
            for (long i = 0; i < words; i++) w[i] = (uint)rng.NextInt64(0, 1L << 32);
            return Up(MemoryMarshal.AsBytes(w.AsSpan()));
        }
        VulkanDevice.Buffer Norms()
        {
            var h = new ushort[(vecs + 1) & ~1L];
            for (long i = 0; i < vecs; i++) h[i] = BitConverter.HalfToUInt16Bits((Half)(0.5 + 2.5 * rng.NextDouble()));
            return Up(MemoryMarshal.AsBytes(h.AsSpan()));
        }
        var cent = Enumerable.Range(0, 1 << l.MseBits).Select(_ => (float)((rng.NextDouble() * 2 - 1) * 0.25)).Order().ToArray();
        VulkanDevice.Buffer Up(ReadOnlySpan<byte> data)
        {
            long n = (data.Length + 3) / 4 * 4;
            if (!deviceLocal)
            {
                var b = device.Allocate(n);
                device.Upload(data, b);
                return b;
            }
            using var staging = device.Allocate(n);
            device.Upload(data, staging);
            var dst = device.AllocateDeviceLocal(n);
            device.CopyBufferSynchronous(staging, dst, (ulong)data.Length);
            return dst;
        }
        return new TqFixture
        {
            K = Codes(), V = Codes(), KNorm = Norms(), VNorm = Norms(), Cent = Up(MemoryMarshal.AsBytes(cent.AsSpan())),
            CodeBytes = words * 4,
        };
    }

    public void Dispose()
    {
        K.Dispose(); V.Dispose(); KNorm.Dispose(); VNorm.Dispose(); Cent.Dispose();
    }
}
