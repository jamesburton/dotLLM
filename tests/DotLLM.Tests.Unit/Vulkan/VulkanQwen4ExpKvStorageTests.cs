using System.Runtime.InteropServices;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #437: lazy K/V + indexer growth and the F16 K/V (and optionally F16 raw indexer key) storage of the Vulkan Qwen4-Exp QSA layers.
/// Kernel-level parity of the F16 twins against the shipping F32 kernels (on F16-rounded inputs, so the only difference is summation order),
/// cache-level growth correctness, and whole-model checks: lazy is bit-identical to eager across many growths (the descriptor-cache aliasing trap),
/// F16 tracks the CPU oracle, and every fast path is proven to have run by dispatch counters.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpKvStorageTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpKvStorageTests(ITestOutputHelper output) => _out = output;

    private static float[] Rand(Random rng, long n, float scale = 1f)
    {
        var a = new float[n];
        for (long i = 0; i < n; i++) a[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static float[] RoundToHalf(float[] x) => x.Select(v => (float)(Half)v).ToArray();

    /// <summary>Packs floats as halves, two per word (the device layout).</summary>
    private static uint[] PackHalves(float[] x)
    {
        var w = new uint[x.Length / 2];
        for (int i = 0; i < w.Length; i++)
            w[i] = BitConverter.HalfToUInt16Bits((Half)x[2 * i]) | ((uint)BitConverter.HalfToUInt16Bits((Half)x[2 * i + 1]) << 16);
        return w;
    }

    private static void UploadWords(VulkanDevice d, uint[] w, VulkanDevice.Buffer b) => d.Upload(MemoryMarshal.AsBytes<uint>(w), b);

    private static uint[] DownloadWords(VulkanDevice d, VulkanDevice.Buffer b, int words)
    {
        var w = new uint[words];
        d.Download(b, MemoryMarshal.Cast<uint, float>(w.AsSpan()));
        return w;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double n = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; n += d * d; den += (double)a[i] * a[i]; }
        return Math.Sqrt(n / Math.Max(den, 1e-30));
    }

    private static void Run(VulkanDevice d, Action<nint> record)
    {
        using var ctx = d.CreateSubmitContext();
        ctx.Begin();
        KernelSupport.HostToComputeBarrier(ctx.CommandBuffer);
        record(ctx.CommandBuffer);
        KernelSupport.ComputeToHostBarrier(ctx.CommandBuffer);
        ctx.SubmitAndWait();
    }

    // ───────────────────────── kernels ─────────────────────────

    [SkippableFact]
    public void ConvertKernel_MatchesHostHalfRounding_WithOffsetAndSaturation()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        Skip.IfNot(k.F16Supported, "F16 SPIR-V twins not built");
        var rng = new Random(3);
        var src = Rand(rng, 1000, 30f);
        src[10] = 1e9f; src[11] = -1e9f; src[12] = 65504f; src[13] = 1e-9f;   // saturation, the largest half, a subnormal-or-flush
        const int Offset = 64;
        using var bSrc = device.AllocateDeviceLocal(src.Length * 4L);
        using var bDst = device.AllocateDeviceLocal((Offset + src.Length) * 2L);
        device.Upload(src, bSrc);
        device.Upload(MemoryMarshal.AsBytes<uint>(new uint[(Offset + src.Length) / 2]), bDst);
        Run(device, cmd => k.RecordF32ToF16(cmd, bSrc, bDst, Offset, src.Length));
        var got = DownloadWords(device, bDst, (Offset + src.Length) / 2);
        int bad = 0;
        for (int i = 0; i < src.Length; i++)
        {
            int e = Offset + i;
            ushort bits = (ushort)(got[e / 2] >> ((e & 1) * 16));
            float g = (float)BitConverter.UInt16BitsToHalf(bits);
            float want = Math.Clamp(src[i], -65504f, 65504f);
            float wantHalf = (float)(Half)want;
            // The shader rounds to nearest-even itself (AMD's packHalf2x16 truncates): exact for the normal range, a hair of slack for subnormals.
            if (g != wantHalf && Math.Abs(g - wantHalf) > (Math.Abs(want) < 6.2e-5f ? 1e-7 : 0.0)) bad++;
        }
        Assert.Equal(0, bad);
        for (int e = 0; e < Offset; e++) Assert.Equal(0u, (got[e / 2] >> ((e & 1) * 16)) & 0xFFFFu);   // below the offset untouched
        Assert.True(k.ConvertDispatches > 0);
        _out.WriteLine($"convert {src.Length} floats at offset {Offset}: all match the host half conversion");
    }

    [SkippableTheory]
    [InlineData(4, 2, 32, 4, 4, 256, 3)]      // small geometry, dense + sparse queries mixed
    [InlineData(12, 2, 256, 64, 8, 700, 2)]   // released head geometry (hd256, 2 KV heads), tiny budget so queries are sparse
    public void F16GatherAttention_MatchesF32Kernel_OnHalfRoundedKv(int heads, int kvHeads, int headDim, int blockSize, int budget, int tokens, int splits)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        Skip.IfNot(k.F16Supported, "F16 SPIR-V twins not built");
        var rng = new Random(11);
        int kvStride = kvHeads * headDim, Q = 6, first = tokens - Q;
        var q = Rand(rng, (long)Q * heads * headDim);
        var kv32 = RoundToHalf(Rand(rng, (long)tokens * kvStride, 2f));
        var vv32 = RoundToHalf(Rand(rng, (long)tokens * kvStride, 2f));
        var sel = new int[Q * budget];
        for (int tq = 0; tq < Q; tq++)
        {
            int nb = (first + tq + 1) / blockSize;
            var ids = Enumerable.Range(0, nb).OrderBy(_ => rng.Next()).Take(budget).OrderBy(x => x).ToArray();
            for (int i = 0; i < budget; i++) sel[tq * budget + i] = i < ids.Length ? ids[i] : 0;
        }
        long partRows = (long)Q * heads * splits;
        using var bQ = device.AllocateDeviceLocal(q.Length * 4L);
        using var bK32 = device.AllocateDeviceLocal(kv32.Length * 4L);
        using var bV32 = device.AllocateDeviceLocal(vv32.Length * 4L);
        using var bK16 = device.AllocateDeviceLocal(kv32.Length * 2L);
        using var bV16 = device.AllocateDeviceLocal(vv32.Length * 2L);
        using var bSel = device.AllocateDeviceLocal(sel.Length * 4L);
        using var bPart = device.AllocateDeviceLocal(partRows * headDim * 4);
        using var bMs = device.AllocateDeviceLocal(partRows * 2 * 4);
        using var bOut32 = device.AllocateDeviceLocal((long)Q * heads * headDim * 4);
        using var bOut16 = device.AllocateDeviceLocal((long)Q * heads * headDim * 4);
        device.Upload(q, bQ); device.Upload(kv32, bK32); device.Upload(vv32, bV32);
        UploadWords(device, PackHalves(kv32), bK16); UploadWords(device, PackHalves(vv32), bV16);
        device.Upload(MemoryMarshal.AsBytes<int>(sel), bSel);
        void Attn(VulkanDevice.Buffer kb, VulkanDevice.Buffer vb, VulkanDevice.Buffer o, bool f16)
            => Run(device, cmd => k.RecordAttention(cmd, bQ, kb, vb, bSel, bPart, bMs, o, 0, Q, first, heads, kvHeads, headDim, blockSize, budget, splits, f16));
        long before = k.F16AttentionDispatches;
        Attn(bK32, bV32, bOut32, false);
        Attn(bK16, bV16, bOut16, true);
        Assert.Equal(before + 1, k.F16AttentionDispatches);
        var a = new float[Q * heads * headDim]; var b = new float[a.Length];
        device.Download(bOut32, a); device.Download(bOut16, b);
        double err = RelL2(a, b);
        _out.WriteLine($"heads {heads} hd {headDim} tokens {tokens}: F16-KV vs F32 kernel on the same rounded values relL2 {err:E2}");
        Assert.True(err < 2e-5, $"F16 gather diverges from the F32 gather on identical values: {err:E2}");
        Assert.True(a.Any(v => Math.Abs(v) > 1e-3), "control: output is all ~0, the comparison proves nothing");
    }

    [SkippableFact]
    public void F16RawPool_MatchesF32Pool_OnHalfRoundedKeys()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var k = Qwen4ExpQsaKernels.Create(device, spvDir);
        Skip.IfNot(k.F16Supported, "F16 SPIR-V twins not built");
        const int Dim = 128, Rope = 64, Block = 4, Tokens = 400;
        var rng = new Random(5);
        var raw = RoundToHalf(Rand(rng, (long)Tokens * Dim));
        var gamma = Rand(rng, Dim, 0.5f); for (int i = 0; i < Dim; i++) gamma[i] += 1f;
        int blocks = Tokens / Block;
        using var bRaw32 = device.AllocateDeviceLocal(raw.Length * 4L);
        using var bRaw16 = device.AllocateDeviceLocal(raw.Length * 2L);
        using var bG = device.AllocateDeviceLocal(Dim * 4L);
        using var bP32 = device.AllocateDeviceLocal((blocks + 1L) * Dim * 4);
        using var bP16 = device.AllocateDeviceLocal((blocks + 1L) * Dim * 4);
        device.Upload(raw, bRaw32); UploadWords(device, PackHalves(raw), bRaw16); device.Upload(gamma, bG);
        Run(device, cmd =>
        {
            k.RecordPool(cmd, bRaw32, bG, bP32, 0, blocks, Dim, Rope, Block, 1e-6f, 1.0e7f);
            k.RecordPool(cmd, bRaw16, bG, bP16, 0, blocks, Dim, Rope, Block, 1e-6f, 1.0e7f, rawF16: true);
        });
        var a = new float[blocks * Dim]; var b = new float[a.Length];
        device.Download(bP32, a); device.Download(bP16, b);
        Assert.Equal(0.0, RelL2(a, b), 6);
    }

    // ───────────────────────── caches ─────────────────────────

    [SkippableTheory]
    [InlineData(VulkanKvStorage.F32)]
    [InlineData(VulkanKvStorage.F16)]
    public void LazyCache_GrowsPreservingRows_AndStopsAtTheLimit(VulkanKvStorage storage)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var qk = Qwen4ExpQsaKernels.Create(device, spvDir);
        Skip.If(storage == VulkanKvStorage.F16 && !qk.F16Supported, "F16 SPIR-V twins not built");
        int old = VulkanNemotronHKvCache.LazyGrowthQuantum;
        VulkanNemotronHKvCache.LazyGrowthQuantum = 8;
        try
        {
            const int Stride = 16, Max = 100;
            using var cache = new VulkanNemotronHKvCache(device, [0], 1, 2, 8, Max, storage, lazy: true);
            Assert.Equal(0, cache.AllocatedLength);
            Assert.Equal(Max, cache.MaxLength);
            var rng = new Random(2);
            var rows = new List<float[]>();
            int written = 0, growths = 0;
            while (written < Max)
            {
                int n = Math.Min(rng.Next(1, 9), Max - written);
                if (cache.EnsureCapacity(written + n)) growths++;
                var k = Rand(rng, n * Stride, 3f); var v = Rand(rng, n * Stride, 3f);
                using var bk = device.AllocateDeviceLocal(k.Length * 4L); using var bv = device.AllocateDeviceLocal(v.Length * 4L);
                device.Upload(k, bk); device.Upload(v, bv);
                var pos = Enumerable.Range(written, n).ToArray();
                Run(device, cmd =>
                {
                    cache.RecordUpdate(cmd, bk, bv, pos, n, 0, qk);
                    KernelSupport.ComputeToComputeBarrier(cmd);
                });
                rows.Add(k); rows.Add(v);
                written += n;
            }
            Assert.True(growths >= 8, $"only {growths} growths: the test does not exercise repeated growth");
            Assert.Equal(Max, cache.AllocatedLength);
            // Every row (written across many reallocations) must read back intact.
            var wantK = rows.Where((_, i) => i % 2 == 0).SelectMany(x => x).ToArray();
            var wantV = rows.Where((_, i) => i % 2 == 1).SelectMany(x => x).ToArray();
            float[] ReadBack(VulkanDevice.Buffer b)
            {
                if (storage == VulkanKvStorage.F32) { var f = new float[Max * Stride]; device.Download(b, f); return f; }
                var w = DownloadWords(device, b, Max * Stride / 2);
                var o = new float[Max * Stride];
                for (int i = 0; i < o.Length; i++) o[i] = (float)BitConverter.UInt16BitsToHalf((ushort)(w[i / 2] >> ((i & 1) * 16)));
                return o;
            }
            var gotK = ReadBack(cache.GetKeysBuffer(0)); var gotV = ReadBack(cache.GetValuesBuffer(0));
            for (int i = 0; i < wantK.Length; i++)
            {
                float ek = storage == VulkanKvStorage.F32 ? wantK[i] : (float)(Half)wantK[i], ev = storage == VulkanKvStorage.F32 ? wantV[i] : (float)(Half)wantV[i];
                Assert.True(Math.Abs(gotK[i] - ek) <= (Math.Abs(wantK[i]) < 6.2e-5f ? 1e-7f : 0f), $"K element {i}: {gotK[i]} != {ek}");
                Assert.True(Math.Abs(gotV[i] - ev) <= (Math.Abs(wantV[i]) < 6.2e-5f ? 1e-7f : 0f), $"V element {i}: {gotV[i]} != {ev}");
            }
            Assert.Throws<ArgumentOutOfRangeException>(() => cache.EnsureCapacity(Max + 1));
        }
        finally { VulkanNemotronHKvCache.LazyGrowthQuantum = old; }
    }

    [SkippableFact]
    public void F16Cache_RefusesToBeWrittenWithoutTheConvertKernel_AndHalvesTheFootprint()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        using var f32 = new VulkanNemotronHKvCache(device, [0, -1, 1], 2, 2, 8, 64);
        using var f16 = new VulkanNemotronHKvCache(device, [0, -1, 1], 2, 2, 8, 64, VulkanKvStorage.F16);
        Assert.Equal(f32.AllocatedBytes / 2, f16.AllocatedBytes);
        Assert.Equal(f32.BytesPerPosition / 2, f16.BytesPerPosition);
        using var b = device.AllocateDeviceLocal(16 * 4);
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        Assert.Throws<InvalidOperationException>(() => f16.RecordUpdate(ctx.CommandBuffer, b, b, [0], 1, 0, null));
    }

    // ───────────────────────── whole model ─────────────────────────

    private static byte[] SmallBudgetModel(int budgetTokens = 16, Q4eGeometry? g = null)
        => Qwen4ExpRandomGguf.Build((g ?? Qwen4ExpRandomGguf.Tiny) with { Budget = budgetTokens, Context = 512 }, Q4eQuant.F32);

    private sealed class Flags : IDisposable
    {
        private readonly VulkanKvStorage _kv; private readonly bool _lazy, _idx; private readonly int _q;
        public Flags(VulkanKvStorage kv, bool lazy, bool idxF16 = false, int quantum = 16)
        {
            _kv = VulkanQwen4ExpTransformerModel.KvStorage; _lazy = VulkanQwen4ExpTransformerModel.LazyKv; _idx = VulkanQwen4ExpTransformerModel.IndexerRawF16;
            _q = VulkanNemotronHKvCache.LazyGrowthQuantum;
            VulkanQwen4ExpTransformerModel.KvStorage = kv; VulkanQwen4ExpTransformerModel.LazyKv = lazy; VulkanQwen4ExpTransformerModel.IndexerRawF16 = idxF16;
            VulkanNemotronHKvCache.LazyGrowthQuantum = quantum;
        }
        public void Dispose()
        {
            VulkanQwen4ExpTransformerModel.KvStorage = _kv; VulkanQwen4ExpTransformerModel.LazyKv = _lazy; VulkanQwen4ExpTransformerModel.IndexerRawF16 = _idx;
            VulkanNemotronHKvCache.LazyGrowthQuantum = _q;
        }
    }

    /// <summary>Chunked prefill then token-by-token decode on a fresh state; returns the logits of every call.</summary>
    private static List<float[]> Trace(VulkanQwen4ExpTransformerModel m, VulkanQwen4ExpSequenceState st, int[] ids, int[] chunks)
    {
        var all = new List<float[]>();
        int at = 0;
        foreach (int c in chunks)
        {
            using var t = m.Forward(ids.AsSpan(at, c), Enumerable.Range(at, c).ToArray(), -1, st);
            all.Add(Q4eRig.Row(t, t.Shape[0] - 1));
            at += c;
        }
        for (; at < ids.Length; at++)
        {
            using var t = m.Forward(ids.AsSpan(at, 1), [at], -1, st);
            all.Add(Q4eRig.Row(t, 0));
        }
        return all;
    }

    [SkippableFact]
    public void LazyGrowth_IsBitIdenticalToEager_AcrossManyGrowths_AndDecodesCorrectlyAfterTheSecond()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        var ids = VulkanQwen4ExpParityTests.Ids(140, rig.Config.VocabSize, seed: 17);
        int[] chunks = [13, 38, 1, 1, 24, 7, 1, 1, 1];   // dense -> sparse (limit 19), chunks and single-row decode, odd boundaries
        List<float[]> eager, lazy;
        long growths;
        using (new Flags(VulkanKvStorage.F32, lazy: false))
        {
            using var st = rig.Vk.CreateState();
            eager = Trace(rig.Vk, st, ids, chunks);
            Assert.Equal(rig.Vk.ContextCapacity, st.OwnKvAllocatedForTests);
        }
        using (new Flags(VulkanKvStorage.F32, lazy: true))
        {
            long g0 = rig.Vk.CacheGrowths;
            using var st = rig.Vk.CreateState();
            lazy = Trace(rig.Vk, st, ids, chunks);
            growths = rig.Vk.CacheGrowths - g0;
            Assert.True(st.OwnKvAllocatedForTests < rig.Vk.ContextCapacity, "a 140-token sequence must not hold the full capacity");
        }
        Assert.True(growths >= 5, $"only {growths} growths");
        _out.WriteLine($"{growths} growths over {eager.Count} calls");
        for (int i = 0; i < eager.Count; i++)
            Assert.True(eager[i].AsSpan().SequenceEqual(lazy[i]), $"call {i}: lazy logits differ from eager (descriptor cache or copy bug)");
        // Control: the eager trace really is sensitive to the KV contents (a different prompt differs).
        Assert.False(eager[^1].AsSpan().SequenceEqual(eager[2]));
    }

    [SkippableFact]
    public void F16Kv_TracksTheOracleAsCloselyAsF32_AndTheF16PathRan()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        Skip.IfNot(rig.Vk.Qsa.F16Supported, "F16 SPIR-V twins not built");
        const int T = 100;
        var ids = VulkanQwen4ExpParityTests.Ids(T, rig.Config.VocabSize, seed: 4);
        var pos = Enumerable.Range(0, T).ToArray();
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), T - 1);

        float[] f32, f16;
        using (new Flags(VulkanKvStorage.F32, lazy: true))
        {
            using var st = rig.Vk.CreateState();
            using var t = rig.Vk.Forward(ids, pos, -1, st);
            f32 = Q4eRig.Row(t, 0);
        }
        long f16Before = rig.Vk.Qsa.QsaKernels.F16AttentionDispatches, convBefore = rig.Vk.Qsa.QsaKernels.ConvertDispatches;
        using (new Flags(VulkanKvStorage.F16, lazy: true))
        {
            using var st = rig.Vk.CreateState();
            using var t = rig.Vk.Forward(ids, pos, -1, st);
            f16 = Q4eRig.Row(t, 0);
        }
        Assert.True(rig.Vk.Qsa.QsaKernels.F16AttentionDispatches > f16Before, "the F16 gather kernel never ran");
        Assert.True(rig.Vk.Qsa.QsaKernels.ConvertDispatches > convBefore, "the F16 convert kernel never ran");
        var (r32, k32, _) = VulkanQwen4ExpParityTests.Compare(cpu, f32);
        var (r16, k16, top16) = VulkanQwen4ExpParityTests.Compare(cpu, f16);
        var (rx, kx, _) = VulkanQwen4ExpParityTests.Compare(f32, f16);
        _out.WriteLine($"vs oracle: F32 relL2 {r32:E2} KL {k32:E2} | F16 relL2 {r16:E2} KL {k16:E2} top1 {top16} | F16 vs F32 relL2 {rx:E2} KL {kx:E2}");
        Assert.True(r16 <= r32 + 3e-3 && k16 <= k32 + 2e-4 && top16, "F16 KV diverges from the oracle by more than the F32 path does");
        Assert.True(rx < 3e-3 && kx < 1e-4, "F16 KV diverges from the F32 KV");
        // Control: F16 really quantised (a bit-identical result would mean the F32 path ran).
        Assert.False(f32.AsSpan().SequenceEqual(f16), "F16 KV produced bit-identical logits: the quantised path did not run");
    }

    [SkippableFact]
    public void F16Kv_DenseRegime_And_Decode_AndRawKeysF16_TrackTheOracle()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        Skip.IfNot(rig.Vk.Qsa.F16Supported, "F16 SPIR-V twins not built");
        var ids = VulkanQwen4ExpParityTests.Ids(70, rig.Config.VocabSize, seed: 21);
        foreach (bool idx in new[] { false, true })
        {
            using var flags = new Flags(VulkanKvStorage.F16, lazy: true, idxF16: idx);
            using var st = rig.Vk.CreateState();
            double worst = 0;
            float[]? vk = null, cpu = null;
            for (int i = 0; i < ids.Length; i++)   // token by token: dense below 19, sparse beyond, growths in between
            {
                cpu = Q4eRig.Row(rig.Cpu.Forward([ids[i]], [i], -1), 0);
                using var t = rig.Vk.Forward([ids[i]], [i], -1, st);
                vk = Q4eRig.Row(t, 0);
                if (i >= 3) worst = Math.Max(worst, VulkanQwen4ExpParityTests.Compare(cpu, vk).RelL2);
            }
            _out.WriteLine($"raw keys F16 = {idx}: token-by-token worst relL2 vs oracle {worst:E2}");
            Assert.True(worst < 6e-3, $"worst relL2 {worst:E2}");
            for (int s = 0; s < 8; s++)
            {
                int next = Array.IndexOf(cpu!, cpu!.Max());
                Assert.Equal(next, Array.IndexOf(vk!, vk!.Max()));
                int p = ids.Length + s;
                cpu = Q4eRig.Row(rig.Cpu.Forward([next], [p], -1), 0);
                using var t = rig.Vk.Forward([next], [p], -1, st);
                vk = Q4eRig.Row(t, 0);
            }
        }
    }

    [SkippableFact]
    public void F16Kv_RefusesToRunWithoutQsa()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(SmallBudgetModel(), spvDir);
        Skip.IfNot(rig.Vk.Qsa.F16Supported, "F16 SPIR-V twins not built");
        using var flags = new Flags(VulkanKvStorage.F16, lazy: true);
        using var st = rig.Vk.CreateState();
        VulkanQwen4ExpQsa.Enabled = false;
        try { Assert.Throws<NotSupportedException>(() => rig.Vk.Forward([5, 6, 7], [0, 1, 2], -1, st)); }
        finally { VulkanQwen4ExpQsa.Enabled = true; }
    }
}
