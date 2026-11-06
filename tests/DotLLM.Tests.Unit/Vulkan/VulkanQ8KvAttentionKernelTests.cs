using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Flint Q8_0 KV cache (opt-in): the quantise kernel and the <c>_kvq8</c> attention twins that read the 36-byte blocks in place.
/// </summary>
/// <remarks>
/// <para>Oracles. (1) The writer's bytes equal a C# reference quantiser exactly (F32 scale = amax / 127, int8 = roundEven(x * (1 / scale))).
/// (2) The split-KV decode and the reference sink kernel read the dequantised values (<c>int8 * scale</c>, exact in F32), so their Q8 twins must be
/// <b>bit-identical</b> to the F32 kernels run on rows dequantised by the C# reference: any difference is a block/word/byte indexing bug.
/// (3) The coopmat sink-flash prefill rounds K / V to F16 in LDS, so it is compared with the F16-KV twin on the F16-rounded dequantised rows
/// (coopmat tolerance).</para>
/// <para>Controls: every case also asserts the Q8 output differs from the attention over the UNQUANTISED rows (so a kernel that ignored the
/// quantisation, or read stale F32, cannot pass), and that this difference is small (the Q8 error budget).</para>
/// <para>Mutants (applied to the .spv in bin, 2026-11-05, all killed): kv_load.glsl K <c>blk * 9</c> -> <c>blk * 8</c> (3 cases fail) and V read with
/// the K scale (3); writer ignores <c>dstBlockOffset</c> (1) and truncates instead of roundEven (3); coopmat flash data word off by one (3) and V
/// with the K scale (3); GQA v2 split pass K scale from the wrong word (3), V with the K scale (3), V second word skipped (3). The GQA cases run the
/// v2 pass (its own Q8 loader) and the per-query-head case runs kv_load.glsl, so each loader has its own killing case.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ8KvAttentionKernelTests
{
    private const int BlockBytes = 36;

    // ── C# reference quantiser / dequantiser (the layout contract of kv_load.glsl) ──

    internal static byte[] QuantizeRef(ReadOnlySpan<float> x)
    {
        Assert.Equal(0, x.Length % 32);
        var dst = new byte[x.Length / 32 * BlockBytes];
        for (int b = 0; b < x.Length / 32; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(x[b * 32 + i]));
            float d = amax / 127f;
            float id = d != 0f ? 1f / d : 0f;
            BitConverter.TryWriteBytes(dst.AsSpan(b * BlockBytes, 4), d);
            for (int i = 0; i < 32; i++)
            {
                int q = Math.Clamp((int)MathF.Round(x[b * 32 + i] * id, MidpointRounding.ToEven), -127, 127);
                dst[b * BlockBytes + 4 + i] = (byte)(sbyte)q;
            }
        }
        return dst;
    }

    internal static float[] DequantizeRef(ReadOnlySpan<byte> blocks)
    {
        var x = new float[blocks.Length / BlockBytes * 32];
        for (int b = 0; b < blocks.Length / BlockBytes; b++)
        {
            float d = BitConverter.ToSingle(blocks.Slice(b * BlockBytes, 4));
            for (int i = 0; i < 32; i++) x[b * 32 + i] = (sbyte)blocks[b * BlockBytes + 4 + i] * d;
        }
        return x;
    }

    private static float[] Rand(Random rng, int n, float amp)
    {
        var r = new float[n];
        // A spread of magnitudes per block (so scales differ), plus the occasional outlier that sets a block's scale.
        for (int i = 0; i < n; i++)
        {
            float a = amp * (1f + 3f * ((i / 32) % 5));
            r[i] = (float)((rng.NextDouble() * 2 - 1) * a) * (i % 97 == 0 ? 4f : 1f);
        }
        return r;
    }

    // ── 1. writer ──

    [SkippableTheory]
    [InlineData(512, 7, 0)]       // a Flint row (4 KV heads x 128) x 7 positions, from block 0
    [InlineData(512, 5, 3)]       // a sliding-window slot written mid-ring: non-zero destination block offset
    [InlineData(256, 1, 0)]       // one position, a narrow row
    public void Writer_MatchesReferenceQuantiser_ByteForByte(int stride, int positions, int dstRow)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var writer = VulkanKvQ8WriteKernel.Create(device, spvDir);
        var rng = new Random(0xA8 + stride + positions);
        float[] src = Rand(rng, positions * stride, 0.5f);
        // Hand-made blocks: all zero (scale 0) and a lone spike (every other element quantises to 0). Exact rounding ties are deliberately not
        // asserted: x * (1 / scale) is not correctly rounded on every driver, so a tie may land on either neighbour (the dequantised error is the same).
        Array.Clear(src, 0, 32);
        Array.Clear(src, 32, 32); src[40] = 12.5f;

        int totalRows = dstRow + positions;
        long dstBytes = (long)totalRows * stride / 32 * BlockBytes;
        using var bSrc = device.Allocate((long)src.Length * 4);
        using var bDst = device.Allocate(dstBytes);
        device.Upload(new byte[dstBytes], bDst);   // sentinel zeros: bytes outside the written range must stay untouched
        device.Upload(src, bSrc);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            writer.RecordF32ToQ8(ctx.CommandBuffer, bSrc, bDst, (long)dstRow * stride, (long)positions * stride);
            ctx.SubmitAndWait();
        }
        var gotF = new float[dstBytes / 4];
        device.Download(bDst, gotF.AsSpan());
        byte[] got = MemoryMarshal.AsBytes(gotF.AsSpan()).ToArray();
        int head = dstRow * stride / 32 * BlockBytes;
        Assert.True(got.AsSpan(0, head).ToArray().All(b => b == 0), "bytes before the destination offset were touched");
        // The scale is amax / 127 and the quants are roundEven(x / scale), but the driver's division / reciprocal is not correctly rounded, so the
        // stored scale may sit a few ULP from the C# reference and a value within ~1e-4 of a rounding tie may land on either neighbour. The contract
        // that matters is relative to the scale the GPU STORED: |q| <= 127 and q = roundEven(x / storedScale) (double precision) up to those ties.
        int blocks = src.Length / 32, tieCount = 0;
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int i = 0; i < 32; i++) amax = MathF.Max(amax, MathF.Abs(src[b * 32 + i]));
            float d = BitConverter.ToSingle(got, head + b * BlockBytes);
            float refD = amax / 127f;
            Assert.True(MathF.Abs(d - refD) <= 4e-7f * refD, $"block {b}: stored scale {d:R} vs amax/127 {refD:R}");
            for (int i = 0; i < 32; i++)
            {
                sbyte q = (sbyte)got[head + b * BlockBytes + 4 + i];
                double exact = d == 0 ? 0 : src[b * 32 + i] / (double)d;
                double nearest = Math.Round(exact, MidpointRounding.ToEven);
                bool nearTie = Math.Abs(Math.Abs(exact - Math.Truncate(exact)) - 0.5) < 2e-3;
                if (nearTie) tieCount++;
                Assert.True(q == (int)nearest || (nearTie && Math.Abs(q - exact) <= 0.5 + 2e-3), $"block {b} elem {i}: stored {q} for x/scale = {exact:R}");
                Assert.InRange((int)q, -127, 127);
            }
        }
        Assert.True(tieCount < blocks * 32 / 50, "test data degenerate: too many rounding ties");
    }

    // ── 2. split-KV decode + reference sink kernel (bit-identical to F32 on dequantised rows) ──

    [SkippableTheory]
    [InlineData(4000, 0, 32, 4, 128, 2.0f, true)]     // Flint full (NoPE) layer at depth, group of 8, GQA-grouped v2 pass
    [InlineData(2600, 513, 32, 4, 128, 1.0f, true)]   // Flint sliding layer, clipped to the window
    [InlineData(777, 0, 32, 4, 128, 0.0f, true)]      // ragged last split
    [InlineData(2600, 513, 32, 4, 128, 1.0f, false)]  // the per-query-head pass
    public void SplitKvDecode_ReadsQ8InPlace_MatchesF32OnDequantisedRows(
        int seqKv, int window, int numHeads, int numKvHeads, int headDim, float sinkBias, bool gqa)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var split32 = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var splitQ8 = VulkanSplitKvAttentionKernel.Create(device, spvDir, VulkanKvStorage.Q8_0);
        split32.PreferGqa = gqa; splitQ8.PreferGqa = gqa;
        Skip.If(gqa && !splitQ8.UsesGqa(numHeads, numKvHeads, headDim), "GQA-grouped split pass unavailable");

        var rng = new Random(0xA8 + seqKv + window);
        int stride = numKvHeads * headDim, kvElems = seqKv * stride;
        float[] q = Rand(rng, numHeads * headDim, 0.5f);
        float[] k = Rand(rng, kvElems, 0.5f), v = Rand(rng, kvElems, 0.5f);
        float[] sinks = new float[numHeads];
        for (int h = 0; h < numHeads; h++) sinks[h] = sinkBias + (float)(rng.NextDouble() * 2 - 1);
        byte[] kq = QuantizeRef(k), vq = QuantizeRef(v);
        float[] kd = DequantizeRef(kq), vd = DequantizeRef(vq);
        int pos = seqKv - 1, kvStart = window > 0 ? Math.Max(0, pos - window + 1) : 0;
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv - kvStart, numHeads));

        using var bQ = device.Allocate(q.Length * 4L);
        using var bK8 = device.Allocate(kq.Length); using var bV8 = device.Allocate(vq.Length);
        using var bKd = device.Allocate(kd.Length * 4L); using var bVd = device.Allocate(vd.Length * 4L);
        using var bKr = device.Allocate(k.Length * 4L); using var bVr = device.Allocate(v.Length * 4L);
        using var bS = device.Allocate(sinks.Length * 4L);
        using var oQ8 = device.Allocate(q.Length * 4L); using var oRef = device.Allocate(q.Length * 4L); using var oRaw = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kq, bK8); device.Upload(vq, bV8); device.Upload(kd, bKd); device.Upload(vd, bVd);
        device.Upload(k, bKr); device.Upload(v, bVr); device.Upload(sinks, bS);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            splitQ8.Record(ctx.CommandBuffer, bQ, bK8, bV8, oQ8, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart);
            split32.Record(ctx.CommandBuffer, bQ, bKd, bVd, oRef, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart);
            split32.Record(ctx.CommandBuffer, bQ, bKr, bVr, oRaw, 1, seqKv, numHeads, numKvHeads, headDim, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart);
            ctx.SubmitAndWait();
        }
        AssertMatchesAndControl(device, oQ8, oRef, oRaw, q.Length);
    }

    [SkippableTheory]
    [InlineData(20, 0, 32, 4, 128)]    // below the split threshold: the reference sink kernel serves the decode step
    [InlineData(300, 513, 32, 4, 128)]
    public void ReferenceSinkKernel_ReadsQ8InPlace_MatchesF32OnDequantisedRows(int seqKv, int window, int numHeads, int numKvHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        var rng = new Random(0xA9 + seqKv);
        int stride = numKvHeads * headDim, kvElems = seqKv * stride;
        float[] q = Rand(rng, numHeads * headDim, 0.5f), k = Rand(rng, kvElems, 0.5f), v = Rand(rng, kvElems, 0.5f);
        float[] sinks = Enumerable.Range(0, numHeads).Select(h => 0.5f + 0.01f * h).ToArray();
        byte[] kq = QuantizeRef(k), vq = QuantizeRef(v);
        float[] kd = DequantizeRef(kq), vd = DequantizeRef(vq);
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK8 = device.Allocate(kq.Length); using var bV8 = device.Allocate(vq.Length);
        using var bKd = device.Allocate(kd.Length * 4L); using var bVd = device.Allocate(vd.Length * 4L);
        using var bKr = device.Allocate(k.Length * 4L); using var bVr = device.Allocate(v.Length * 4L);
        using var bS = device.Allocate(sinks.Length * 4L);
        using var oQ8 = device.Allocate(q.Length * 4L); using var oRef = device.Allocate(q.Length * 4L); using var oRaw = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kq, bK8); device.Upload(vq, bV8); device.Upload(kd, bKd); device.Upload(vd, bVd);
        device.Upload(k, bKr); device.Upload(v, bVr); device.Upload(sinks, bS);
        int pos = seqKv - 1;
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            gpt.RecordAttentionSinks(ctx.CommandBuffer, bQ, bK8, bV8, oQ8, bS, 1, seqKv, numHeads, numKvHeads, headDim, pos, window, 0f, VulkanKvStorage.Q8_0);
            gpt.RecordAttentionSinks(ctx.CommandBuffer, bQ, bKd, bVd, oRef, bS, 1, seqKv, numHeads, numKvHeads, headDim, pos, window);
            gpt.RecordAttentionSinks(ctx.CommandBuffer, bQ, bKr, bVr, oRaw, bS, 1, seqKv, numHeads, numKvHeads, headDim, pos, window);
            ctx.SubmitAndWait();
        }
        AssertMatchesAndControl(device, oQ8, oRef, oRaw, q.Length);
    }

    private static void AssertMatchesAndControl(VulkanDevice device, VulkanDevice.Buffer oQ8, VulkanDevice.Buffer oRef, VulkanDevice.Buffer oRaw, int n)
    {
        float[] a = new float[n], r = new float[n], raw = new float[n];
        device.Download(oQ8, a); device.Download(oRef, r); device.Download(oRaw, raw);
        // Same operands (int8 * scale is exact in F32); only the compiler's FMA contraction of that product into the dot / accumulate differs, so
        // agreement is a few ULP. An indexing bug (block, word, byte, scale) moves a result by percent, five orders of magnitude more.
        float refMax = r.Max(MathF.Abs);
        for (int i = 0; i < n; i++)
            Assert.True(MathF.Abs(a[i] - r[i]) <= 2e-6f * refMax + 1e-9f, $"[{i}] Q8 kernel {a[i]:R} vs F32 kernel on dequantised rows {r[i]:R}");
        // Control 1: the quantisation is visible (the Q8 result is NOT the unquantised attention) ...
        float maxAbs = raw.Max(MathF.Abs), worst = 0;
        for (int i = 0; i < n; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - raw[i]));
        Assert.True(worst > 1e-6f * maxAbs, $"Q8 output is indistinguishable from the unquantised attention (worst {worst:G3}); the control cannot discriminate");
        // ... control 2: and within the Q8_0 error budget (about 0.4% of a block's amplitude per element, averaged by the softmax).
        Assert.True(worst < 0.05f * maxAbs, $"Q8 error {worst:G3} exceeds 5% of the attention output amplitude {maxAbs:G3}");
    }

    // ── 3. coopmat sink-flash prefill ──

    [SkippableTheory]
    [InlineData(100, 612, 32, 4, 128, 512, 0)]      // chunked prefill at an offset, dense layer
    [InlineData(77, 1205, 32, 4, 128, 1128, 513)]   // sliding layer (window clip), partial Q/KV tiles
    [InlineData(301, 333, 32, 4, 128, 0, 0)]        // fresh prefill, partial tiles both ways
    public void SinkFlashPrefill_ReadsQ8InPlace_MatchesF16TwinOnDequantisedRows(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int posOffset, int window)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var q8 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.Q8_0, sinks: true, splitQ: null);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F16, sinks: true, splitQ: null);
        Skip.If(q8 is null || f16 is null || !q8.CanDispatch(headDim, numHeads, numKvHeads), "no coopmat GQA sink flash on this device");
        using var _q8 = q8!; using var _f16 = f16!;

        var rng = new Random(0xAA + seqQ + seqKv);
        int stride = numKvHeads * headDim, kvElems = seqKv * stride;
        float[] q = Rand(rng, seqQ * numHeads * headDim, 0.5f), k = Rand(rng, kvElems, 0.5f), v = Rand(rng, kvElems, 0.5f);
        float[] sinks = Enumerable.Range(0, numHeads).Select(h => 0.5f + 0.01f * h).ToArray();
        byte[] kq = QuantizeRef(k), vq = QuantizeRef(v);
        float[] kd = DequantizeRef(kq), vd = DequantizeRef(vq);
        ushort[] kh = kd.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray(), vh = vd.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray();
        ushort[] krh = k.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray(), vrh = v.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray();

        using var bQ = device.Allocate(q.Length * 4L);
        using var bK8 = device.Allocate(kq.Length); using var bV8 = device.Allocate(vq.Length);
        using var bK16 = device.Allocate(kh.Length * 2L); using var bV16 = device.Allocate(vh.Length * 2L);
        using var bKr = device.Allocate(krh.Length * 2L); using var bVr = device.Allocate(vrh.Length * 2L);
        using var bS = device.Allocate(sinks.Length * 4L);
        using var oQ8 = device.Allocate(q.Length * 4L); using var oF16 = device.Allocate(q.Length * 4L); using var oRaw = device.Allocate(q.Length * 4L);
        device.Upload(q, bQ); device.Upload(kq, bK8); device.Upload(vq, bV8); device.Upload(sinks, bS);
        device.Upload(MemoryMarshal.AsBytes(kh.AsSpan()), bK16); device.Upload(MemoryMarshal.AsBytes(vh.AsSpan()), bV16);
        device.Upload(MemoryMarshal.AsBytes(krh.AsSpan()), bKr); device.Upload(MemoryMarshal.AsBytes(vrh.AsSpan()), bVr);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q8.Record(ctx.CommandBuffer, bQ, bK8, bV8, oQ8, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOffset, slidingWindow: window, sinks: bS);
            f16.Record(ctx.CommandBuffer, bQ, bK16, bV16, oF16, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOffset, slidingWindow: window, sinks: bS);
            f16.Record(ctx.CommandBuffer, bQ, bKr, bVr, oRaw, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset: posOffset, slidingWindow: window, sinks: bS);
            ctx.SubmitAndWait();
        }
        int n = q.Length;
        float[] a = new float[n], r = new float[n], raw = new float[n];
        device.Download(oQ8, a); device.Download(oF16, r); device.Download(oRaw, raw);
        float maxAbs = r.Max(MathF.Abs), worst = 0, worstRaw = 0;
        for (int i = 0; i < n; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(a[i] - r[i]));
            worstRaw = MathF.Max(worstRaw, MathF.Abs(a[i] - raw[i]));
        }
        Assert.True(maxAbs > 1e-3f, "degenerate reference");
        // Same operands up to the F32 -> F16 rounding of the dequantised values inside the LDS tile (hardware rounding mode), then identical coopmat math.
        Assert.True(worst <= 2e-3f * maxAbs + 1e-5f, $"Q8 prefill differs from the F16 twin on the dequantised rows by {worst:G3} (amplitude {maxAbs:G3})");
        Assert.True(worstRaw > 1e-5f * maxAbs, "control: the Q8 prefill must differ from attention over the unquantised rows");
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) { num += (double)(a[i] - raw[i]) * (a[i] - raw[i]); den += (double)raw[i] * raw[i]; }
        double relL2 = Math.Sqrt(num / den);
        Assert.True(relL2 < 0.03, $"Q8 prefill relative L2 error {relL2:G3} vs attention over the unquantised rows exceeds the 3% Q8_0 budget (worst element {worstRaw:G3} of amplitude {maxAbs:G3})");
    }
}
