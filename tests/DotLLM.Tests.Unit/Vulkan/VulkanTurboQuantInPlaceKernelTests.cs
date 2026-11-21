using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Engine;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.ChunkReuse;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tests.Unit.Engine.KvCache;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1892 (TurboQuant step 2): every attention arm the dense chain and Flint dispatch has a <c>_kvtq</c> twin that reads the
/// packed codes and norms of a <see cref="VulkanKvStorage.Tq"/> <see cref="VulkanKvCache"/> in place, in code space. No decoded
/// history exists in any format.
/// </summary>
/// <remarks>
/// <para>Each arm case checks two things on the same data (K and V with different sign seeds, head vectors of varied norm, GQA
/// with more than one KV head and a group above one, bit widths 2 / 3 / 4 / 8: 3 makes codes straddle 32-bit words, 8 uses the
/// 256-entry centroid table):</para>
/// <list type="number">
/// <item><b>Loader</b>: the twin, given the rotated query, equals its own F32 kernel run on the code-space rows
/// (<c>centroid[code] x norm</c>) materialised on the host from the cache's stored words. Any difference beyond FMA contraction is a
/// word, bit, stride, norm or centroid indexing bug. Cooperative-matrix twins round K / V to F16 in LDS, so they are compared with
/// their F16 twin on the F16-rounded code-space rows, at coopmat tolerance.</item>
/// <item><b>Oracle</b> (the merge gate): rotate Q in, twin, rotate O out equals the same arm's F32 kernel over the scratch-dequant
/// oracle <see cref="VulkanTurboQuantKvCache"/> built with the same norm format, within a tolerance that scales with
/// <c>sqrt(seqKv x headDim)</c>.</item>
/// </list>
/// <para>Controls: the output differs from attention over the unquantised rows, and its error against them is the oracle's own.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanTurboQuantInPlaceKernelTests
{
    private const ulong Seed = 0x1892_7C0DEUL;
    private const ulong VSeedXor = DotLLM.Engine.KvCache.Codecs.TurboQuantCodec.VSeedMask;   // TurboQuantKvCache's V-seed derivation

    private readonly ITestOutputHelper _out;
    public VulkanTurboQuantInPlaceKernelTests(ITestOutputHelper output) => _out = output;

    /// <summary>One arm: records attention reading <c>k</c> / <c>v</c> (codes for a twin, rows for a reference) into <c>o</c>.</summary>
    private delegate void Rec(nint cmd, VulkanDevice.Buffer q, VulkanDevice.Buffer k, VulkanDevice.Buffer v, VulkanDevice.Buffer o, TqKvOperands? tq);

    public static TheoryData<int, int> BitsAndHeadDims()
    {
        var d = new TheoryData<int, int>();
        foreach (int bits in new[] { 2, 3, 4, 8 })
            foreach (int hd in new[] { 64, 128, 256 })
                d.Add(bits, hd);
        return d;
    }

    // Prefill at an offset with partial tiles. hd64: Qwen2.5-0.5B (14 / 2); hd128: 32 / 8; hd256: 16 / 4.
    private static (int SeqQ, int SeqKv, int Heads, int KvHeads) PrefillShape(int hd) => hd switch
    {
        64 => (37, 290, 14, 2),
        128 => (45, 333, 32, 8),
        _ => (21, 150, 16, 4),
    };

    private static TurboQuantLayout Layout(int bits, int hd, int kvHeads) => new(hd, kvHeads, bits, TurboQuantLayout.DefaultNorm);

    private static float[] Uniform(Random rng, int n, float amp)
    {
        var r = new float[n];
        for (int i = 0; i < n; i++) r[i] = (float)((rng.NextDouble() * 2 - 1) * amp);
        return r;
    }

    /// <summary>K / V rows whose head vectors have norms spread over ~6x, so a dropped or misindexed norm cannot pass.</summary>
    private static float[] KvRows(Random rng, int seqKv, int kvHeads, int hd)
    {
        float[] r = Uniform(rng, seqKv * kvHeads * hd, 1f);
        for (int hv = 0; hv < seqKv * kvHeads; hv++)
        {
            float s = 0.5f + 2.5f * (float)rng.NextDouble();
            for (int c = 0; c < hd; c++) r[hv * hd + c] *= s;
        }
        return r;
    }

    private static VulkanDevice.Buffer UploadF32(VulkanDevice device, float[] x)
    {
        var b = device.Allocate(x.Length * 4L);
        device.Upload(x, b);
        return b;
    }

    private static VulkanDevice.Buffer UploadF16(VulkanDevice device, float[] x)
    {
        ushort[] h = x.Select(f => BitConverter.HalfToUInt16Bits((Half)f)).ToArray();
        var b = device.Allocate((h.Length * 2L + 3) / 4 * 4);
        device.Upload(MemoryMarshal.AsBytes(h.AsSpan()), b);
        return b;
    }

    /// <summary>The cache's stored rows in code space, <c>centroid[code] x norm</c>, decoded on the host by an independent bit reader.</summary>
    private static float[] CodeSpaceRows(VulkanKvCache cache, bool values, TurboQuantLayout l, int seqKv, ReadOnlySpan<float> centroids)
    {
        int vecs = seqKv * l.NumKvHeads, cu = l.CodeUintsPerVec;
        var words = new uint[vecs * cu + 1];
        cache.DownloadTqCodes(0, values, words);
        var normBytes = new byte[TqKvTwin.NormBufferBytes(vecs, l.Norm)];
        cache.DownloadTqNormBytes(0, values, normBytes);
        var rows = new float[vecs * l.HeadDim];
        for (int hv = 0; hv < vecs; hv++)
        {
            float norm = TurboQuantNormCodec.Decode(l.Norm, normBytes, hv);
            for (int c = 0; c < l.HeadDim; c++)
            {
                int code = 0;
                for (int b = 0; b < l.MseBits; b++)
                {
                    long bit = (long)c * l.MseBits + b;
                    code |= (int)((words[hv * cu + (bit >> 5)] >> (int)(bit & 31)) & 1u) << b;
                }
                rows[hv * l.HeadDim + c] = centroids[code] * norm;
            }
        }
        return rows;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / den);
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float w = 0;
        for (int i = 0; i < a.Length; i++) w = MathF.Max(w, MathF.Abs(a[i] - b[i]));
        return w;
    }

    /// <summary>
    /// Runs one arm against the code-space reference and the scratch oracle. <paramref name="reference"/> is the arm's F32-KV kernel.
    /// <paramref name="f16Reference"/> is given for the cooperative-matrix arms that stage K / V as one F16 tile: their F16-KV twin on
    /// the F16-rounded code-space rows is the same-operand comparison for the loader check. A cooperative-matrix arm that keeps the
    /// F32 value (the head-dim-256 body's hi / lo tiles) passes <paramref name="coopmat"/> alone and is checked against its F32 kernel.
    /// </summary>
    private void Case(VulkanDevice device, string spvDir, TurboQuantLayout l, int seqQ, int seqKv, int numHeads,
        Rec twin, Rec reference, Rec? f16Reference = null, bool coopmat = false)
    {
        int hd = l.HeadDim, kvHeads = l.NumKvHeads, bits = l.MseBits;
        var rng = new Random(0x1892 + bits * 131 + hd * 7 + seqQ * 3 + seqKv);
        float[] q = Uniform(rng, seqQ * numHeads * hd, 8f);   // scores of a few units: a peaked, non-degenerate softmax
        float[] k = KvRows(rng, seqKv, kvHeads, hd), v = KvRows(rng, seqKv, kvHeads, hd);

        var codecK = new TurboQuantCodec(hd, bits, Seed, useQjl: false);
        var codecV = new TurboQuantCodec(hd, bits, Seed ^ VSeedXor, useQjl: false);
        // K and V rotate with different sign sets: a twin or a rotation that took the other side's signs cannot pass.
        Assert.False(codecK.RotationSigns.SequenceEqual(codecV.RotationSigns), "K and V sign seeds must differ");
        using var enc = TurboQuantEncodeF32Kernel.Create(device, spvDir);
        using var rot = TqRotateKernel.Create(device, spvDir);
        using var cache = new VulkanKvCache(device, 1, seqKv, l, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD);
        using var oracle = new VulkanTurboQuantKvCache(device, spvDir, 1, kvHeads, hd, seqKv, bits,
            codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD, l.Norm);
        Assert.Equal(VulkanKvStorage.Tq, cache.Storage);
        Assert.Equal(l.Norm, oracle.NormFormat);

        using var bQ = UploadF32(device, q);
        using var bQrot = UploadF32(device, q);   // rotated in place into K code space; the oracle arm keeps the original
        using var bK = UploadF32(device, k);
        using var bV = UploadF32(device, v);
        int[] positions = Enumerable.Range(0, seqKv).ToArray();
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            cache.RecordUpdate(ctx.CommandBuffer, bK, bV, positions, seqKv, 0, tqWriter: enc);
            oracle.RecordUpdate(ctx.CommandBuffer, bK, bV, positions, seqKv, 0);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            oracle.RecordDequant(ctx.CommandBuffer, 0);
            ctx.SubmitAndWait();
        }
        Assert.Equal(seqKv, cache.CurrentLength);

        float[] kCode = CodeSpaceRows(cache, false, l, seqKv, codecK.Centroids), vCode = CodeSpaceRows(cache, true, l, seqKv, codecK.Centroids);
        bool cm = coopmat || f16Reference is not null;
        using var bKc = f16Reference is not null ? UploadF16(device, kCode) : UploadF32(device, kCode);
        using var bVc = f16Reference is not null ? UploadF16(device, vCode) : UploadF32(device, vCode);

        long outBytes = q.Length * 4L;
        using var oTwin = device.Allocate(outBytes); using var oCode = device.Allocate(outBytes);
        using var oRef = device.Allocate(outBytes); using var oRaw = device.Allocate(outBytes);
        var ops = cache.TqOperands(0);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            nint c = ctx.CommandBuffer;
            rot.RecordQuery(c, bQrot, seqQ, numHeads, hd, cache.TqSignsK);
            KernelSupport.ComputeToComputeBarrier(c);
            twin(c, bQrot, cache.GetKeysBuffer(0), cache.GetValuesBuffer(0), oTwin, ops);
            (f16Reference ?? reference)(c, bQrot, bKc, bVc, oCode, null);
            reference(c, bQ, oracle.GetKeysBuffer(), oracle.GetValuesBuffer(), oRef, null);
            reference(c, bQ, bK, bV, oRaw, null);
            ctx.SubmitAndWait();
        }
        int n = q.Length;
        float[] twinCode = new float[n], refCode = new float[n], r = new float[n], raw = new float[n], a = new float[n];
        device.Download(oTwin, twinCode); device.Download(oCode, refCode); device.Download(oRef, r); device.Download(oRaw, raw);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            rot.RecordOutput(ctx.CommandBuffer, oTwin, seqQ, numHeads, hd, cache.TqSignsV);   // once, after the arm (split-KV: after its combine pass)
            ctx.SubmitAndWait();
        }
        device.Download(oTwin, a);

        // 1. Loader: same operands as the reference on the materialised code-space rows.
        float codeMax = refCode.Max(MathF.Abs), codeWorst = MaxAbsDiff(twinCode, refCode);
        float codeTol = cm ? 2e-3f : 1e-5f;
        Assert.True(codeMax > 1e-3f, "degenerate code-space reference");
        // 2. Oracle: in place == scratch-dequant path, same norm format. F32 arms differ only by the rounding of the two rotations and
        //    the dot / accumulate order, which grows with sqrt(seqKv * headDim); cooperative-matrix arms round K / V to F16 in different
        //    spaces (code space here, original space in the oracle's kernel), so they get the coopmat budget.
        float refMax = r.Max(MathF.Abs), worst = MaxAbsDiff(a, r);
        float tol = cm ? 1.5e-2f : 2f * 1.1920929e-7f * MathF.Sqrt((float)seqKv * hd);
        double relOracle = RelL2(a, r);
        _out.WriteLine($"bits {bits} hd {hd} seqQ {seqQ} seqKv {seqKv}: loader {codeWorst / codeMax:E2} (tol {codeTol:E1}); oracle max {worst / refMax:E2} (tol {tol:E2}) relL2 {relOracle:E2}");
        Assert.True(codeWorst <= codeTol * codeMax + (cm ? 1e-5f : 0f),
            $"loader: twin differs from its {(f16Reference is not null ? "F16" : "F32")} kernel on the code-space rows by {codeWorst:G3} (amplitude {codeMax:G3}, bits {bits}, hd {hd})");
        Assert.True(worst <= tol * refMax, $"in-place differs from the scratch oracle by {worst:G3} (amplitude {refMax:G3}, tolerance {tol * refMax:G3}; bits {bits}, hd {hd})");
        if (cm) Assert.True(relOracle < 5e-3, $"in-place vs oracle relative L2 {relOracle:G3}");

        // Controls: TurboQuant really happened, and the twin adds nothing to the codec's own error.
        float worstRaw = MaxAbsDiff(a, raw);
        Assert.True(worstRaw > 1e-4f * refMax, $"output is indistinguishable from attention over the unquantised rows (worst {worstRaw:G3})");
        double twinErr = RelL2(a, raw), own = RelL2(r, raw);
        Assert.True(twinErr <= own * 1.02 + 1e-3, $"relative L2 error {twinErr:G4} vs unquantised exceeds the oracle's own {own:G4}");
    }

    private static VulkanDevice.Buffer Sinks(VulkanDevice device, int numHeads)
    {
        var b = device.Allocate(numHeads * 4L);
        device.Upload(Enumerable.Range(0, numHeads).Select(h => 0.5f + 0.01f * h).ToArray(), b);
        return b;
    }

    private static VulkanDevice.Buffer QueryPositions(VulkanDevice device, int[] positions)
    {
        var b = device.Allocate(positions.Length * 4L);
        device.Upload(MemoryMarshal.AsBytes(positions.AsSpan()), b);
        return b;
    }

    // ── dense chain ─────────────────────────────────────────────────────────────────────────────────────────────

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void PerToken_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        using var tq = AttentionF32Kernel.Create(device, spvDir, l);
        using var f32 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.F32);
        Assert.Equal(f32.Mode, tq.Mode);   // the subgroup twin when the F32 kernel takes the subgroup shader
        foreach (int rows in new[] { seqQ, 1 })   // a prefill chunk at an offset, then one decode row
        {
            int off = seqKv - rows;
            Case(device, spvDir, l, rows, seqKv, heads,
                (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, rows, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
                (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, rows, seqKv, heads, kvHeads, hd, positionOffset: off));
        }
    }

    /// <summary>
    /// Head vectors whose codes do not fill their last word (head dim 16: 3 bits = 48 of 64 bits, 5 bits = 80 of 96, 7 bits = 112 of
    /// 128, 1 bit = 16 of 32): the next vector starts at the next WORD, not at the next bit. At head dims 64 / 128 / 256 every code
    /// length is a whole number of words, so only these shapes can tell the per-vector word stride from tight packing.
    /// </summary>
    [SkippableTheory]
    [InlineData(3, 16)]
    [InlineData(5, 16)]
    [InlineData(7, 16)]
    [InlineData(1, 16)]
    public void PerToken_CodesNotFillingTheirLastWord_UseTheWordStride(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        const int seqQ = 9, seqKv = 50, heads = 6, kvHeads = 3;
        var l = Layout(bits, hd, kvHeads);
        Assert.NotEqual(0, hd * bits % 32);
        using var tq = AttentionF32Kernel.Create(device, spvDir, l);
        using var f32 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.F32);
        int off = seqKv - seqQ;
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off));
    }

    /// <summary>The shared-memory per-token body too (the subgroup one is what this device picks): both have a twin.</summary>
    [SkippableTheory]
    [InlineData(3, 64)]
    [InlineData(8, 128)]
    public void PerTokenSharedMemory_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? old = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_FORCE_SHARED_REDUCE");
        Environment.SetEnvironmentVariable("DOTLLM_VULKAN_FORCE_SHARED_REDUCE", "1");
        try
        {
            using var device = VulkanDevice.Create();
            var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
            var l = Layout(bits, hd, kvHeads);
            using var tq = AttentionF32Kernel.Create(device, spvDir, l);
            using var f32 = AttentionF32Kernel.Create(device, spvDir, VulkanKvStorage.F32);
            Assert.Equal(AttentionF32Kernel.DispatchMode.SharedMem, tq.Mode);
            int off = seqKv - seqQ;
            Case(device, spvDir, l, seqQ, seqKv, heads,
                (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
                (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off));
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_FORCE_SHARED_REDUCE", old);
        }
    }

    // Query rows at arbitrary positions inside an already-filled cache (selective recompute's read side).
    private static int[] ScatteredPositions(int seqQ, int seqKv)
    {
        var p = new int[seqQ];
        for (int i = 0; i < seqQ; i++) p[i] = (int)(((long)i * 7919 + 13) % seqKv);
        p[0] = seqKv - 1; if (seqQ > 1) p[1] = 0;
        return p;
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void QPos_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        using var tq = AttentionQueryPositionsKernel.Create(device, spvDir, l);
        using var f32 = AttentionQueryPositionsKernel.Create(device, spvDir);
        using var bPos = QueryPositions(device, ScatteredPositions(seqQ, seqKv));
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, bPos, seqQ, seqKv, heads, kvHeads, hd, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, bPos, seqQ, seqKv, heads, kvHeads, hd));
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void QPosTiled_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        using var bPos = QueryPositions(device, ScatteredPositions(seqQ, seqKv));
        // The implementation the model would pick (coopmat here), then the scalar one: both have a twin.
        foreach (bool coopmat in new[] { true, false })
        {
            var f32 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32, allowCoopmat: coopmat, queryPositions: true);
            if (f32 is null || f32.IsCoopmat != coopmat) { f32?.Dispose(); continue; }   // no such implementation on this device
            using var _f = f32;
            using var tq = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.Tq, allowCoopmat: coopmat, queryPositions: true, tq: l);
            Assert.True(tq is not null && tq.IsCoopmat == coopmat, $"no TurboQuant twin of the {(coopmat ? "coopmat" : "scalar")} per-position GQA flash kernel");
            using var f16 = coopmat ? VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16, queryPositions: true) : null;
            Assert.True(!coopmat || f16 is not null);
            Assert.True(tq!.CanDispatch(hd, heads, kvHeads));
            Rec refRec = (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, queryPositions: bPos);
            Case(device, spvDir, l, seqQ, seqKv, heads,
                (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, queryPositions: bPos, tq: t),
                refRec,
                coopmat ? (c, q, k, v, o, _) => f16!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, queryPositions: bPos) : null);
        }
    }

    private void SplitKv(int bits, int hd, int seqKv, int heads, int kvHeads, bool gqa, bool gqa2, int window = 0, bool sinks = false)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var l = Layout(bits, hd, kvHeads);
        using var f32 = VulkanSplitKvAttentionKernel.Create(device, spvDir);
        using var tq = VulkanSplitKvAttentionKernel.TryCreate(device, spvDir, l);
        Assert.True(tq is not null, "no TurboQuant split-KV twin");
        f32.PreferGqa = gqa; tq!.PreferGqa = gqa;
        f32.PreferGqa2 = gqa2; tq.PreferGqa2 = gqa2;
        // F32 partials (#1926 E4 control): these tests hold the LOADER to the scratch oracle at reduction-order tolerance; the F16
        // partial format has its own stated-tolerance tests (SplitKvPartialF16Tests) and model gate.
        f32.PreferPartF16 = false; tq.PreferPartF16 = false;
        Assert.Equal(f32.UsesGqa(heads, kvHeads, hd), tq.UsesGqa(heads, kvHeads, hd));        // same pass as the F32 kernel takes
        Assert.Equal(f32.UsesGqa2(heads, kvHeads, hd), tq.UsesGqa2(heads, kvHeads, hd));
        Skip.If(gqa && !f32.UsesGqa(heads, kvHeads, hd), "GQA-grouped split pass unavailable for this shape");
        Skip.If(gqa2 && !f32.UsesGqa2(heads, kvHeads, hd), "the v2 grouped pass serves head dims up to 128 on a wave64 device");
        Assert.True(VulkanSplitKvAttentionKernel.WouldSplit(seqKv, heads));
        using var bS = sinks ? Sinks(device, heads) : null;
        int pos = seqKv - 1, kvStart = window > 0 ? Math.Max(0, pos - window + 1) : 0;
        long before = tq.GqaDispatches;
        Case(device, spvDir, l, 1, seqKv, heads,
            (c, q, k, v, o, t) => tq.Record(c, q, k, v, o, 1, seqKv, heads, kvHeads, hd, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, 1, seqKv, heads, kvHeads, hd, positionOffset: pos, slidingWindow: window, sinks: bS, kvStart: kvStart));
        Assert.Equal(gqa, tq.GqaDispatches > before);
    }

    /// <summary>Split-KV decode, the per-query-head pass and the GQA-grouped v1 pass.</summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void SplitKvDecode_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        var (_, _, heads, kvHeads) = PrefillShape(hd);
        SplitKv(bits, hd, 777, heads, kvHeads, gqa: false, gqa2: false);
        SplitKv(bits, hd, 900, heads, kvHeads, gqa: true, gqa2: false);
    }

    /// <summary>Split-KV decode, the GQA-grouped v2 pass (its own 8-element loader); head dim up to 128.</summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void SplitKvV2Decode_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        var (_, _, heads, kvHeads) = PrefillShape(hd);
        SplitKv(bits, hd, 2600, heads, kvHeads, gqa: true, gqa2: true);
    }

    /// <summary>
    /// 4K context, the shape decode runs at once a conversation is under way: many splits per head, and the output rotation applied
    /// once to the combined result. (A rotation recorded before the combine pass would rotate a stale output buffer that the combine
    /// then overwrites, leaving the result in code space.)
    /// </summary>
    [SkippableTheory]
    [InlineData(3, 64, true, true)]
    [InlineData(4, 128, true, true)]
    [InlineData(4, 128, true, false)]
    [InlineData(8, 256, false, false)]
    public void SplitKvDecode_At4KContext_MatchesOracle(int bits, int hd, bool gqa, bool gqa2)
    {
        var (_, _, heads, kvHeads) = PrefillShape(hd);
        SplitKv(bits, hd, 4096, heads, kvHeads, gqa, gqa2);
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void ScalarFlash_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        Skip.If(hd > VulkanFlashAttentionF32Kernel.MaxHeadDim, "the dense chain's scalar flash body serves head dims up to 128");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        using var f32 = VulkanFlashAttentionF32Kernel.Create(device, spvDir, FlashAttentionWideVariant.None);
        using var tq = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir, l);
        Assert.True(tq is not null, "no TurboQuant scalar flash twin");
        int off = seqKv - seqQ;
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void CoopmatFlash_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        var f32 = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir);
        var f16 = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, VulkanKvStorage.F16);
        Skip.If(f32 is null || f16 is null || hd > f16.SupportedMaxHeadDim, "no coopmat flash on this device / shape");
        using var _a = f32!; using var _b = f16!;
        using var tq = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir, l);
        Assert.True(tq is not null && hd <= tq.SupportedMaxHeadDim, "no TurboQuant coopmat flash twin for this head dim");
        int off = seqKv - seqQ;
        // Head dim 256 runs the hi / lo body, which keeps the F32 code-space value: its same-operand reference is the F32 kernel.
        Rec f16Rec = (c, q, k, v, o, _) => f16.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off);
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off),
            hd > VulkanFlashAttentionCoopmatKernel.MaxHeadDim ? null : f16Rec, coopmat: true);
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void ScalarGqaFlash_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        var f32 = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.F32);
        Skip.If(f32 is null || !f32.CanDispatch(hd, heads, kvHeads), "no scalar GQA flash for this shape / device");
        using var _f = f32!;
        using var tq = VulkanFlashAttentionGqaKernel.TryCreateScalar(device, spvDir, VulkanKvStorage.Tq, tq: l);
        Assert.True(tq is not null, "no TurboQuant scalar GQA flash twin");
        int off = seqKv - seqQ;
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off));
    }

    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void CoopmatGqaFlash_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (seqQ, seqKv, heads, kvHeads) = PrefillShape(hd);
        var l = Layout(bits, hd, kvHeads);
        var f32 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F32);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.F16);
        Skip.If(f32 is null || f16 is null || !f16.CanDispatch(hd, heads, kvHeads), "no coopmat GQA flash on this device / shape");
        using var _a = f32!; using var _b = f16!;
        using var tq = VulkanFlashAttentionGqaKernel.TryCreateCoopmat(device, spvDir, VulkanKvStorage.Tq, tq: l);
        Assert.True(tq is not null, "no TurboQuant coopmat GQA flash twin");
        int off = seqKv - seqQ;
        Case(device, spvDir, l, seqQ, seqKv, heads,
            (c, q, k, v, o, t) => tq!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, tq: t),
            (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off),
            (c, q, k, v, o, _) => f16.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off));
    }

    // ── Flint sink attention ────────────────────────────────────────────────────────────────────────────────────

    // A Flint layer is 32 / 4 at head dim 128; the other head dims keep the twins honest about the head-dim constant.
    private static (int Heads, int KvHeads) SinkShape(int hd) => hd == 128 ? (32, 4) : hd == 64 ? (16, 4) : (8, 2);

    /// <summary>Sink decode: split-KV with the sink folded into the combine pass, full and window-clipped.</summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void SinkSplitKv_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        var (heads, kvHeads) = SinkShape(hd);
        SplitKv(bits, hd, 2600, heads, kvHeads, gqa: true, gqa2: hd <= VulkanSplitKvAttentionKernel.Gqa2MaxHeadDim, window: 513, sinks: true);
        SplitKv(bits, hd, 777, heads, kvHeads, gqa: false, gqa2: false, window: 0, sinks: true);
    }

    /// <summary>Sink prefill: the GQA flash kernel's sink twin, from zero and window-clipped at an offset.</summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void SinkFlash_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (heads, kvHeads) = SinkShape(hd);
        var l = Layout(bits, hd, kvHeads);
        var f32 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F32, sinks: true, splitQ: null);
        var f16 = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.F16, sinks: true, splitQ: null);
        Skip.If(f32 is null || f16 is null || !f16.CanDispatch(hd, heads, kvHeads), "no GQA sink flash on this device");
        using var _a = f32!; using var _b = f16!;
        using var tq = VulkanFlashAttentionGqaKernel.TryCreate(device, spvDir, VulkanKvStorage.Tq, sinks: true, splitQ: null, tq: l);
        Assert.True(tq is not null && tq.IsCoopmat == f16.IsCoopmat, "no TurboQuant twin of the GQA sink flash kernel the model picks");
        using var bS = Sinks(device, heads);
        foreach (var (seqQ, seqKv, off, window) in new[] { (60, 60, 0, 0), (41, 700, 659, 257) })
        {
            Rec f16Rec = (c, q, k, v, o, _) => f16.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, slidingWindow: window, sinks: bS);
            Case(device, spvDir, l, seqQ, seqKv, heads,
                (c, q, k, v, o, t) => tq!.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, slidingWindow: window, sinks: bS, tq: t),
                (c, q, k, v, o, _) => f32.Record(c, q, k, v, o, seqQ, seqKv, heads, kvHeads, hd, positionOffset: off, slidingWindow: window, sinks: bS),
                tq!.IsCoopmat ? f16Rec : null);
        }
    }

    /// <summary>The reference sink kernel (what a sink dispatch falls to when neither split-KV nor the flash twin serves it).</summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void ReferenceSink_ReadsTqInPlace_MatchesOracle(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var (heads, kvHeads) = SinkShape(hd);
        var l = Layout(bits, hd, kvHeads);
        using var gpt = VulkanGptOssKernels.Create(device, spvDir);
        Assert.True(gpt.TryEnsureTurboQuantSinks(device, spvDir, l), "no TurboQuant twin of the reference sink kernel");
        using var bS = Sinks(device, heads);
        foreach (var (seqQ, seqKv, window) in new[] { (1, 14, 0), (5, 300, 0), (3, 290, 100) })
        {
            int off = seqKv - seqQ;
            Case(device, spvDir, l, seqQ, seqKv, heads,
                (c, q, k, v, o, t) => gpt.RecordAttentionSinks(c, q, k, v, o, bS, seqQ, seqKv, heads, kvHeads, hd, off, window, kvStorage: VulkanKvStorage.Tq, tq: t),
                (c, q, k, v, o, _) => gpt.RecordAttentionSinks(c, q, k, v, o, bS, seqQ, seqKv, heads, kvHeads, hd, off, window, kvStorage: VulkanKvStorage.F32));
        }
    }

    // ── writer ──────────────────────────────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// The in-place cache stores exactly the CPU codec's bytes: codes (per-vector word stride, zero-padded) and norms in the
    /// layout's format, written at a non-zero position, leaving the rows before it and the other layer untouched. The corpus has
    /// exact centroid ties, F16-saturating and subnormal norms, and zero vectors (<see cref="TurboQuantByteExactCorpus"/>).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(BitsAndHeadDims))]
    public void Writer_MatchesCpuEncode_ByteForByte(int bits, int hd)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        var corpus = TurboQuantByteExactCorpus.Build(bits, hd, halfInputs: false);
        Assert.False(corpus.K.Codec.RotationSigns.SequenceEqual(corpus.V.Codec.RotationSigns), "K and V sign seeds must differ");
        const int kvHeads = TurboQuantByteExactCorpus.NumKvHeads, seqLen = TurboQuantByteExactCorpus.SeqLen, startPos = 3, layer = 1;
        var l = Layout(bits, hd, kvHeads);
        int maxSeq = startPos + seqLen;
        using var enc = TurboQuantEncodeF32Kernel.Create(device, spvDir);
        using var cache = new VulkanKvCache(device, 2, maxSeq, l,
            corpus.K.Codec.Centroids, corpus.K.Codec.RotationSigns, corpus.V.Codec.RotationSigns, corpus.K.Codec.InvSqrtD);
        using var bK = UploadF32(device, corpus.K.Data);
        using var bV = UploadF32(device, corpus.V.Data);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            cache.RecordUpdate(ctx.CommandBuffer, bK, bV, Enumerable.Range(startPos, seqLen).ToArray(), seqLen, layer, tqWriter: enc);
            ctx.SubmitAndWait();
        }
        Assert.Equal(maxSeq, cache.CurrentLength);

        int cu = l.CodeUintsPerVec, vecs = maxSeq * kvHeads, skip = startPos * kvHeads;
        foreach (bool values in new[] { false, true })
        {
            var side = values ? corpus.V : corpus.K;
            var (cpuCodes, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, l.Norm);
            var words = new uint[vecs * cu + 1];
            cache.DownloadTqCodes(layer, values, words);
            var norms = new byte[TqKvTwin.NormBufferBytes(vecs, l.Norm)];
            cache.DownloadTqNormBytes(layer, values, norms);
            ReadOnlySpan<byte> gpuCodes = MemoryMarshal.AsBytes(words.AsSpan(skip * cu, seqLen * kvHeads * cu));
            var (codeBad, normBad, total) = TurboQuantByteExactCorpus.Compare(cpuCodes, cpuNorms, gpuCodes, norms.AsSpan(skip * l.NormBytes), cu * 4, l.NormBytes);
            Assert.True(codeBad == 0 && normBad == 0, $"{(values ? "V" : "K")} bits {bits} hd {hd}: {codeBad} code vectors and {normBad} norms of {total} differ from the CPU codec");
        }
        Assert.True(corpus.K.TieCoords > 0, "the corpus has no exact centroid ties: the tie policy is unobservable");
        // DeviceBytes counts every buffer: codes (with the guard word) and norms for both sides and layers, the centroids and both sign vectors.
        long expected = 2 * 2 * (l.CodeBytesPerLayer(maxSeq) + TqKvTwin.NormBufferBytes(vecs, l.Norm)) + (1L << bits) * 4 + 2L * hd * 4;
        Assert.Equal(expected, cache.DeviceBytes);
    }

    // ── static shader facts ─────────────────────────────────────────────────────────────────────────────────────

    private static (Dictionary<uint, List<uint>> Bindings, long WorkgroupBytes) Reflect(string spvPath)
    {
        uint[] w = MemoryMarshal.Cast<byte, uint>(File.ReadAllBytes(spvPath)).ToArray();
        var size = new Dictionary<uint, long>();          // type id -> bytes
        var constant = new Dictionary<uint, long>();      // scalar constant id -> value
        var pointee = new Dictionary<uint, uint>();       // pointer type id -> pointee type id
        var bindings = new Dictionary<uint, List<uint>>();
        long workgroup = 0;
        for (int i = 5; i < w.Length;)
        {
            int count = (int)(w[i] >> 16), op = (int)(w[i] & 0xFFFF);
            switch (op)
            {
                case 21 or 22: size[w[i + 1]] = w[i + 2] / 8; break;                                   // OpTypeInt / OpTypeFloat
                case 23 or 24: size[w[i + 1]] = size.GetValueOrDefault(w[i + 2]) * w[i + 3]; break;                      // OpTypeVector / OpTypeMatrix
                case 28: size[w[i + 1]] = size.GetValueOrDefault(w[i + 2]) * constant.GetValueOrDefault(w[i + 3]); break;                  // OpTypeArray
                case 32: pointee[w[i + 1]] = w[i + 3]; break;                                          // OpTypePointer
                case 43 or 50: constant[w[i + 2]] = w[i + 3]; break;                                   // OpConstant / OpSpecConstant
                case 59 when w[i + 3] == 4: workgroup += size.GetValueOrDefault(pointee[w[i + 1]]); break;               // OpVariable, StorageClass Workgroup
                case 71 when w[i + 2] == 33:                                                           // OpDecorate Binding
                    if (!bindings.TryGetValue(w[i + 3], out var list)) bindings[w[i + 3]] = list = new List<uint>();
                    list.Add(w[i + 1]);
                    break;
            }
            i += count;
        }
        return (bindings, workgroup);
    }

    /// <summary>
    /// Every <c>_kvtq</c> SPIR-V binds the K norms, V norms and centroids at 12, 13 and 14, each binding number names exactly one
    /// buffer (a body using 12..14 for something else would alias the loader's operands), and the twin's shared memory is byte for
    /// byte its Q8_0 twin's: TurboQuant adds no LDS to any body (the centroid table is a buffer binding; the hd64 coopmat body has
    /// 341 bytes of headroom and is not given a twin at all).
    /// </summary>
    [SkippableFact]
    public void TqTwins_BindOperandsAt12To14_AndAddNoSharedMemory()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        // The retained #1926 pre-change controls (*_pre1926) are A/B and bit-identity arms, never dispatched by a model.
        string[] twins = Directory.GetFiles(spvDir, "*_kvtq*.spv").Where(p => !p.EndsWith("_pre1926.spv", StringComparison.Ordinal)).ToArray();
        // One twin per body the dense chain and Flint can dispatch (VulkanTransformerModel.RecordAttentionArm and the per-position arm).
        string[] expected =
        [
            "attention_f32_kvtq", "attention_f32_sg_kvtq", "attention_f32_qpos_kvtq",
            "attention_f32_splitkv_kvtq", "attention_f32_splitkv_gqa_kvtq", "attention_f32_splitkv_gqa_v2_kvtq",
            "attention_flash_f32_kvtq", "attention_flash_f32_coopmat_kvtq", "attention_flash_f32_coopmat_kvtq_hd256",
            "attention_flash_gqa_kvtq", "attention_flash_gqa_qpos_kvtq", "attention_flash_gqa_sinks_kvtq",
            "attention_flash_gqa_cm_kvtq", "attention_flash_gqa_cm_kvtq_hd128", "attention_flash_gqa_cm_qpos_kvtq", "attention_flash_gqa_cm_qpos_kvtq_hd128",
            "attention_flash_gqa_cm_sinks_kvtq", "attention_flash_gqa_cm_sinks_kvtq_hd128", "attention_sinks_f32_kvtq",
        ];
        Assert.Equal(expected.Order(), twins.Select(p => Path.GetFileNameWithoutExtension(p)!).Order());
        foreach (string path in twins)
        {
            string name = Path.GetFileNameWithoutExtension(path);
            var (bindings, lds) = Reflect(path);
            foreach (var (binding, vars) in bindings)
                Assert.True(vars.Count == 1, $"{name}: binding {binding} names {vars.Count} buffers");
            foreach (uint b in new uint[] { 12, 13, 14 })
                Assert.True(bindings.ContainsKey(b), $"{name}: no buffer at binding {b}");
            Assert.True(bindings.Keys.Where(b => b < 12).Max() <= 4, $"{name}: a body binding above 4 would need the layout table updated");
            string q8 = Path.Combine(spvDir, name.Replace("_kvtq", "_kvq8") + ".spv");
            Assert.True(File.Exists(q8), $"{name}: no Q8_0 twin to compare shared memory with");
            long q8Lds = Reflect(q8).WorkgroupBytes;
            _out.WriteLine($"{name}: shared memory {lds} B (Q8_0 twin {q8Lds} B)");
            // #1926 E3: the v2 split-KV twin alone copies the 2^bits centroid table into shared memory (tqCentSh, 4 << bits bytes:
            // 64 B at tq4). The array is sized by the TQ_MSE_BITS specialization constant, which this SPIR-V reflection does not count,
            // so its fixed arrays still equal the Q8_0 twin's; the driver's figure is in Probe1926SplitKvV2ShaderInfoTests.
            Assert.True(lds == q8Lds, $"{name}: shared memory {lds} B differs from its Q8_0 twin's {q8Lds} B");
        }
        Assert.DoesNotContain(twins, p => Path.GetFileName(p).Contains("hd64"));
    }

    // ── model level: the real dispatch ──────────────────────────────────────────────────────────────────────────

    private static float[] LastRow(DotLLM.Core.Tensors.ITensor logits, int vocab)
    {
        var row = new float[vocab];
        new ReadOnlySpan<float>((float*)logits.DataPointer + (logits.Shape[0] - 1) * vocab, vocab).CopyTo(row);
        return row;
    }

    private static (double MeanAbs, double Cos, bool SameArgmax) Compare(float[] a, float[] b)
    {
        double sum = 0, dot = 0, na = 0, nb = 0;
        int ia = 0, ib = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += Math.Abs((double)a[i] - b[i]); dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i];
            if (a[i] > a[ia]) ia = i;
            if (b[i] > b[ib]) ib = i;
        }
        return (sum / a.Length, dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-30), ia == ib);
    }

    /// <summary>Prefills <paramref name="tokens"/> in chunks, then decodes one more token; returns the logits after the last prefill chunk and after the decode step.</summary>
    private static (float[] Prefill, float[] Decode) Run(VulkanTransformerModel model, DotLLM.Core.Attention.IKvCache cache, int[] tokens, int chunk, int vocab)
    {
        float[] last = [];
        for (int start = 0; start < tokens.Length - 1; start += chunk)
        {
            int n = Math.Min(chunk, tokens.Length - 1 - start);
            using var logits = model.Forward(tokens.AsSpan(start, n), Enumerable.Range(start, n).ToArray(), -1, cache);
            last = LastRow(logits, vocab);
        }
        using var dec = model.Forward(tokens.AsSpan(tokens.Length - 1, 1), [tokens.Length - 1], -1, cache);
        return (last, LastRow(dec, vocab));
    }

    /// <summary>
    /// Runs the real forward three ways on Qwen2.5-0.5B (dense, hd64, GQA 7): the model's plain KV, the scratch-dequant oracle, and
    /// the in-place TurboQuant cache, and requires in place to agree with the oracle to within <paramref name="maxRatio"/> of the
    /// codec's own effect on the logits (oracle vs plain KV).
    /// </summary>
    /// <remarks>
    /// <paramref name="layers"/> = 1 loads the first layer only. That is what makes a TIGHT comparison possible: layer 0's K / V depend
    /// only on the embeddings, so both paths store bit-identical codes and differ by kernel rounding alone. At full depth every layer
    /// re-quantises K / V that already carry the previous layers' rounding differences, and a coordinate that lands on the other side
    /// of a centroid boundary moves by a whole quantisation step, so the two paths drift apart by a fraction of the codec's own noise
    /// (measured here with every arm in F32: tq4 0.5, tq8 0.05 to 0.13 of the codec effect). A full-depth comparison is therefore a
    /// loose statistical check; it is kept, at 8 bits, because only it exercises per-layer buffers.
    /// </remarks>
    private void ModelCase(int bits, int promptTokens, int chunk, bool expectSplitKv, int layers, double maxRatio, double minCos)
    {
        string? path = TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (layers > 0) config = config with { NumLayers = layers };
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] text = tokenizer.Encode(string.Concat(Enumerable.Repeat(
            "The quick brown fox jumps over the lazy dog while seven wizards quietly judge a boxing match in Zurich. ", 1 + promptTokens / 12)));
        Assert.True(text.Length >= promptTokens, $"prompt too short: {text.Length}");
        int[] tokens = text[..promptTokens];
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        CompareWithOracle(model, config, tokens, spvDir, bits, chunk, expectSplitKv, maxRatio, minCos);
    }

    /// <summary>Per-coordinate codes of the first <paramref name="vecs"/> head vectors of a layer side, from its stored words.</summary>
    private static byte[] Codes(uint[] words, TurboQuantLayout l, int vecs)
    {
        var codes = new byte[vecs * l.HeadDim];
        for (int hv = 0; hv < vecs; hv++)
            for (int c = 0; c < l.HeadDim; c++)
            {
                int code = 0;
                for (int b = 0; b < l.MseBits; b++)
                {
                    long bit = (long)c * l.MseBits + b;
                    code |= (int)((words[hv * l.CodeUintsPerVec + (bit >> 5)] >> (int)(bit & 31)) & 1u) << b;
                }
                codes[hv * l.HeadDim + c] = (byte)code;
            }
        return codes;
    }

    private static double Agreement(byte[] a, byte[] b)
    {
        int same = 0;
        for (int i = 0; i < a.Length; i++) if (a[i] == b[i]) same++;
        return (double)same / a.Length;
    }

    /// <summary>
    /// The three-way comparison on an already loaded model (real GGUF or synthetic): plain KV, scratch oracle, in place, all through
    /// the model's own <c>Forward</c> (so through <c>RecordAttention</c>: rotate Q in, arm, rotate the output out).
    /// <paramref name="minLayerCodeAgreement"/> &gt; 0 also compares, layer by layer, the K and V codes the two TurboQuant caches end
    /// up holding.
    /// </summary>
    private void CompareWithOracle(VulkanTransformerModel model, ModelConfig config, int[] tokens, string spvDir, int bits, int chunk,
        bool expectSplitKv, double maxRatio, double minCos, double minLayerCodeAgreement = 0)
    {
        int promptTokens = tokens.Length;
        int hd = config.HeadDim, vocab = config.VocabSize;

        var codecK = new TurboQuantCodec(hd, bits, Seed, useQjl: false);
        var codecV = new TurboQuantCodec(hd, bits, Seed ^ VSeedXor, useQjl: false);
        Assert.False(codecK.RotationSigns.SequenceEqual(codecV.RotationSigns), "K and V sign seeds must differ");
        var layout = new TurboQuantLayout(hd, config.NumKvHeads, bits, TurboQuantLayout.DefaultNorm);
        int vecs = promptTokens * config.NumKvHeads;
        var oracleCodes = new byte[config.NumLayers][][];
        var inPlaceCodes = new byte[config.NumLayers][][];

        // Baseline (the model's default KV) and the scratch oracle first: the switch below changes what CreateKvCache makes.
        (float[] Prefill, float[] Decode) plain, oracle, inPlace;
        using (var c = model.CreateKvCache(promptTokens + 4)) plain = Run(model, c, tokens, chunk, vocab);
        using (var c = model.CreateTurboQuantKvCache(spvDir, promptTokens + 4, bits, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD))
        {
            Assert.Equal(TurboQuantLayout.DefaultNorm, c.NormFormat);
            oracle = Run(model, c, tokens, chunk, vocab);
            for (int layer = 0; layer < config.NumLayers && minLayerCodeAgreement > 0; layer++)
            {
                oracleCodes[layer] = new byte[2][];
                for (int side = 0; side < 2; side++)
                {
                    var words = new uint[vecs * layout.CodeUintsPerVec];
                    c.DownloadCodeUints(layer, side == 1, words);
                    oracleCodes[layer][side] = Codes(words, layout, vecs);
                }
            }
        }

        Assert.True(model.TryUseInPlaceTurboQuantKv(spvDir, bits, codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD));
        long attn0 = model.TurboQuantAttentionCount, split0 = model.TurboQuantSplitKvCount;
        using (var c = model.CreateKvCache(promptTokens + 4))
        {
            Assert.Equal(VulkanKvStorage.Tq, c.Storage);
            var l = new TurboQuantLayout(hd, config.NumKvHeads, bits, TurboQuantLayout.DefaultNorm);
            Assert.Equal(l, c.TurboQuantLayout);
            inPlace = Run(model, c, tokens, chunk, vocab);
            for (int layer = 0; layer < config.NumLayers && minLayerCodeAgreement > 0; layer++)
            {
                inPlaceCodes[layer] = new byte[2][];
                for (int side = 0; side < 2; side++)
                {
                    var words = new uint[vecs * layout.CodeUintsPerVec];
                    c.DownloadTqCodes(layer, side == 1, words);
                    inPlaceCodes[layer][side] = Codes(words, layout, vecs);
                }
            }
            // The cache holds codes (plus a guard word per layer side) and F16 norms, the centroids and two sign vectors: nothing wider.
            long vectors = (long)(promptTokens + 4) * config.NumKvHeads;
            Assert.Equal(2L * config.NumLayers * (l.CodeBytesPerLayer(promptTokens + 4) + TqKvTwin.NormBufferBytes(vectors, l.Norm)) + (1L << bits) * 4 + 2L * hd * 4,
                c.DeviceBytes);
        }
        int forwards = (promptTokens - 1 + chunk - 1) / chunk + 1;
        Assert.Equal((long)forwards * config.NumLayers, model.TurboQuantAttentionCount - attn0);   // every layer of every forward ran in code space
        Assert.Equal(expectSplitKv ? (long)config.NumLayers : 0L, model.TurboQuantSplitKvCount - split0);

        // One forward writes at most 65535 head vectors (the encode's and the rotations' dispatch bound): a longer one is refused, not clipped.
        int tooMany = VulkanTransformerModel.MaxTurboQuantVectorsPerForward / config.NumKvHeads + 1;
        using (var c = model.CreateKvCache(tooMany))
            Assert.Throws<NotSupportedException>(() => model.Forward(new int[tooMany], Enumerable.Range(0, tooMany).ToArray(), -1, c));

        var rows = new[] { ("prefill", inPlace.Prefill, oracle.Prefill, plain.Prefill), ("decode", inPlace.Decode, oracle.Decode, plain.Decode) };
        foreach (var (name, a, o, p) in rows)
        {
            var vsOracle = Compare(a, o);
            var effect = Compare(o, p);
            _out.WriteLine($"tq{bits} layers {config.NumLayers} {name} @ {promptTokens}: in-place vs oracle mean|d| {vsOracle.MeanAbs:E3} cos {vsOracle.Cos:F6}; "
                + $"codec effect (oracle vs plain KV) {effect.MeanAbs:E3}; ratio {vsOracle.MeanAbs / effect.MeanAbs:F4}");
        }
        foreach (var (name, a, o, p) in rows)
        {
            var vsOracle = Compare(a, o);
            var effect = Compare(o, p);
            // The sensitivity control: TurboQuant moved the logits, so a forward that read anything but the codes could not agree with the oracle ...
            Assert.True(effect.MeanAbs > 1e-2, $"{name}: tq{bits} is indistinguishable from the plain KV; the comparison cannot fail");
            // ... and in place agrees with the oracle far more closely than TurboQuant differs from the plain KV.
            Assert.True(vsOracle.MeanAbs < maxRatio * effect.MeanAbs,
                $"{name}: in-place vs oracle {vsOracle.MeanAbs:G4} exceeds {maxRatio} x the codec's own effect {effect.MeanAbs:G4}");
            Assert.True(vsOracle.Cos > minCos, $"{name}: logits diverged from the oracle, cos {vsOracle.Cos:F6}");
        }

        if (minLayerCodeAgreement <= 0) return;
        // Per layer: the codes the in-place cache holds against the codes the oracle cache holds. Layer 0's K / V depend only on the
        // embeddings, so they are bit-identical. A deeper layer's codes agree as long as every earlier layer read ITS OWN codes and
        // norms; a layer that read another layer's buffers hands the next layer different K / V, and agreement drops to chance
        // (compare the cross-layer control below: the same metric between different layers).
        for (int layer = 0; layer < config.NumLayers; layer++)
            for (int side = 0; side < 2; side++)
            {
                double own = Agreement(inPlaceCodes[layer][side], oracleCodes[layer][side]);
                int other = (layer + 1) % config.NumLayers;
                double cross = Agreement(inPlaceCodes[layer][side], oracleCodes[other][side]);
                _out.WriteLine($"layer {layer} {(side == 0 ? "K" : "V")}: code agreement with the oracle's layer {layer} {own:P2}; with its layer {other} {cross:P2}");
                if (layer == 0) Assert.True(own == 1.0, $"layer 0 {(side == 0 ? "K" : "V")} codes are not bit-identical to the oracle's ({own:P3})");
                Assert.True(own >= minLayerCodeAgreement, $"layer {layer} {(side == 0 ? "K" : "V")}: only {own:P2} of the codes match the oracle's");
                if (config.NumLayers > 1)
                    Assert.True(cross < minLayerCodeAgreement / 2, $"control: layer {layer} agrees {cross:P2} with the oracle's layer {other}; the metric cannot tell layers apart");
            }
    }

    /// <summary>
    /// The real dispatch at 4K context (first layer, so both paths hold identical codes): prefill in 512-row chunks, then one decode
    /// step, which the split-KV twin serves. The decode logits match the scratch oracle tightly, so the output rotation ran once on
    /// the combined result. A rotation recorded anywhere else (before the arm, hence before the combine pass) leaves the attention
    /// output in code space and fails this.
    /// </summary>
    [SkippableFact]
    public void SplitKv_LongContext_RotatesAfterCombine()
        => ModelCase(bits: 3, promptTokens: 4096, chunk: 512, expectSplitKv: true, layers: 1, maxRatio: 0.05, minCos: 0.9999);

    /// <summary>
    /// A fresh prefill chunk attends its own rows, and those rows come from the cache's codes: every row is "current" here (an
    /// empty cache, one chunk), at 2 bits where the codes differ grossly from the fresh K / V. The result matches the oracle (which
    /// decodes the same codes) tightly while differing from the plain-KV forward by the codec's full effect.
    /// </summary>
    [SkippableFact]
    public void CurrentChunkRows_ComeFromTheCache()
        => ModelCase(bits: 2, promptTokens: 13, chunk: 12, expectSplitKv: false, layers: 1, maxRatio: 0.05, minCos: 0.9999);

    /// <summary>
    /// Real model, full depth, 8 bits. <b>This is NOT an oracle-parity gate.</b> Its one job is to notice a layer reading or writing
    /// another layer's code / norm buffers on a real 24-layer model, which moves the logits by more than the codec's whole effect.
    /// It cannot be a parity gate: at full depth every layer re-quantises K / V that already carry the earlier layers' rounding
    /// differences, so two correct arithmetic paths legitimately drift apart by a fraction of the codec's own noise (measured: 0.10
    /// to 0.12 of it at 8 bits, about 0.5 at 4 bits, 0.68 at 3 bits). Hence the loose bounds (0.4, cos 0.995). The deterministic
    /// per-layer check is <see cref="Synthetic_EachLayerWritesAndReadsItsOwnBuffers"/>; parity is gated per attention at kernel level
    /// and on one-layer models.
    /// </summary>
    [SkippableFact]
    public void FullDepth_EachLayerReadsItsOwnBuffers_Statistical()
        => ModelCase(bits: 8, promptTokens: 300, chunk: 128, expectSplitKv: true, layers: 0, maxRatio: 0.4, minCos: 0.995);

    // ── model-free: a synthetic Llama-shape model, so these never skip for a missing GGUF ──────────────────────────────────────

    /// <summary>Writes a random Llama-shape safetensors model (RMSNorm, GQA, SwiGLU) and returns its config.</summary>
    private static ModelConfig WriteSyntheticModel(string path, int layers, int heads, int kvHeads, int hd, int hidden, int vocab, int maxSeq)
    {
        var rng = new Random(0x7157 + layers * 31 + heads * 7 + kvHeads);
        float[] R(int n, float amp) { var r = new float[n]; for (int i = 0; i < n; i++) r[i] = (float)((rng.NextDouble() * 2 - 1) * amp); return r; }
        float[] Ones(int n) { var r = new float[n]; Array.Fill(r, 1f); return r; }
        int inter = hidden * 2;
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [vocab, hidden], R(vocab * hidden, 1.0f));
        bld.AddFloat32("model.norm.weight", [hidden], Ones(hidden));
        for (int i = 0; i < layers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [hidden], Ones(hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [hidden], Ones(hidden));
            // Q / K large enough for a peaked softmax, so a wrong K-side rotation moves the output as much as a wrong V-side one.
            bld.AddFloat32($"{p}.self_attn.q_proj.weight", [heads * hd, hidden], R(heads * hd * hidden, 0.35f));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight", [kvHeads * hd, hidden], R(kvHeads * hd * hidden, 0.35f));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight", [kvHeads * hd, hidden], R(kvHeads * hd * hidden, 0.2f));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight", [hidden, heads * hd], R(hidden * heads * hd, 0.2f));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight", [inter, hidden], R(inter * hidden, 0.1f));
            bld.AddFloat32($"{p}.mlp.up_proj.weight", [inter, hidden], R(inter * hidden, 0.1f));
            bld.AddFloat32($"{p}.mlp.down_proj.weight", [hidden, inter], R(hidden * inter, 0.1f));
        }
        bld.AddFloat32("lm_head.weight", [vocab, hidden], R(vocab * hidden, 0.5f));
        bld.WriteTo(path);
        return new ModelConfig
        {
            Architecture = DotLLM.Core.Configuration.Architecture.Llama,
            VocabSize = vocab,
            HiddenSize = hidden,
            IntermediateSize = inter,
            NumLayers = layers,
            NumAttentionHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = hd,
            MaxSequenceLength = maxSeq,
            NormEpsilon = 1e-5f,
            RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: hd, Type: RoPEType.Norm),
        };
    }

    private void WithSyntheticModel(int layers, int heads, int kvHeads, int hd, int hidden, int vocab, int maxSeq,
        Action<VulkanTransformerModel, ModelConfig, string> body)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string dir = Path.Combine(Path.GetTempPath(), $"dotllm-tq-inplace-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "model.safetensors");
            var config = WriteSyntheticModel(path, layers, heads, kvHeads, hd, hidden, vocab, maxSeq);
            using var file = SafetensorsFile.Open(path);
            using var model = VulkanTransformerModel.LoadFromSafetensors(file, config, spvDir);
            body(model, config, spvDir);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }

    private static int[] SyntheticTokens(int n, int vocab)
    {
        var rng = new Random(0x70C + n);
        return Enumerable.Range(0, n).Select(_ => rng.Next(vocab)).ToArray();
    }

    /// <summary>
    /// The real dispatch wrapper (<c>RecordAttention</c>: K-sign rotation of Q, the arm, V-sign rotation of the output) on a
    /// one-layer synthetic model: a 12-row prefill chunk into an empty cache, whose rows attend each other from the codes just
    /// written, then one decode row on the per-token arm (13 keys, below the split-KV threshold). Needs no model file.
    /// </summary>
    [SkippableFact]
    public void Synthetic_PrefillChunkAndPerTokenDecode_ThroughTheRealDispatch_MatchOracle()
        => WithSyntheticModel(layers: 1, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 1024, (model, config, spvDir)
            => CompareWithOracle(model, config, SyntheticTokens(13, config.VocabSize), spvDir, bits: 2, chunk: 12, expectSplitKv: false, maxRatio: 0.05, minCos: 0.9999));

    /// <summary>
    /// The real dispatch wrapper with the split-KV decode arm: 600 keys, so the decode row is split and combined, and the output
    /// rotation recorded by the wrapper must land after the combine pass. Prefill runs in 128-row chunks at growing offsets. 3 bits
    /// (codes straddle words). Needs no model file.
    /// </summary>
    [SkippableFact]
    public void Synthetic_SplitKvDecode_ThroughTheRealDispatch_RotatesAfterCombine()
        => WithSyntheticModel(layers: 1, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 1024, (model, config, spvDir)
            => CompareWithOracle(model, config, SyntheticTokens(600, config.VocabSize), spvDir, bits: 3, chunk: 128, expectSplitKv: true, maxRatio: 0.05, minCos: 0.9999));

    /// <summary>
    /// Four layers: after the same prompt, each layer's K and V codes in the in-place cache are compared with the oracle cache's
    /// codes for that layer. Layer 0 must be bit-identical; deeper layers must agree far above chance, which they only do when
    /// every earlier layer wrote and read its own code and norm buffers; and the same metric against a DIFFERENT layer is at chance.
    /// Deterministic (codes, not logits) and model-free.
    /// </summary>
    [SkippableFact]
    public void Synthetic_EachLayerWritesAndReadsItsOwnBuffers()
        => WithSyntheticModel(layers: 4, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 1024, (model, config, spvDir)
            => CompareWithOracle(model, config, SyntheticTokens(200, config.VocabSize), spvDir, bits: 4, chunk: 64, expectSplitKv: true,
                maxRatio: double.MaxValue, minCos: -1.0, minLayerCodeAgreement: 0.6));   // the codes decide here, not the logits

    /// <summary>
    /// With in-place TurboQuant active the model asks for prefill chunks of at most floor(65535 / numKvHeads) rows (one forward may
    /// write 65535 head vectors), and for one pass (0) otherwise.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 2, 32, 32767)]    // the encode grid binds: floor(65535 / 2)
    [InlineData(32, 32, 16, 2047)]   // the encode grid binds: floor(65535 / 32)
    [InlineData(32, 1, 32, 32752)]   // GQA ratio 32: the flash twin's ceil(rows / 16) x 32 groups bind, not the encode (65535 / 1)
    public void PreferredPrefillChunk_IsTheTurboQuantForwardBound(int heads, int kvHeads, int hd, int expected)
        => WithSyntheticModel(layers: 1, heads, kvHeads, hd, hidden: 64, vocab: 128, maxSeq: 256, (model, config, spvDir) =>
        {
            Assert.Equal(0, model.PreferredPrefillChunkTokens);
            Assert.Equal(0, model.MaxPrefillForwardTokens);
            var ck = new TurboQuantCodec(hd, 4, Seed, useQjl: false);
            var cv = new TurboQuantCodec(hd, 4, Seed ^ VSeedXor, useQjl: false);
            Assert.True(model.TryUseInPlaceTurboQuantKv(spvDir, 4, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD));
            Assert.Equal(expected, model.PreferredPrefillChunkTokens);
            Assert.Equal(expected, model.MaxPrefillForwardTokens);   // the same bound is the hard limit the engine clamps to
            Assert.True((long)model.PreferredPrefillChunkTokens * kvHeads <= VulkanTransformerModel.MaxTurboQuantVectorsPerForward);
            // The flash twin's X groups, ceil(rows / 16) x heads, stay inside the dispatch limit at the bound and not one row above it.
            Assert.True((long)((model.PreferredPrefillChunkTokens + 15) / 16) * heads <= 65535);
            Assert.True(((long)(model.PreferredPrefillChunkTokens + 1) * kvHeads > 65535)
                        || (long)((model.PreferredPrefillChunkTokens + 16) / 16) * heads > 65535, "the bound is not tight");
        });

    /// <summary>
    /// #1892 Task 8: the real capability. False until the shared helper has built the twin set for the request, then true for exactly that
    /// request (its bits; not tq*q; not a window), and the helper's factory builds the Vulkan in-place cache, which then runs a forward.
    /// A host whose twin set lacks an arm its device runs would answer false here and refuse at startup.
    /// </summary>
    [SkippableFact]
    public void Capability_IsFalseUntilTheTwinSetIsBuilt_ThenTrueOnlyForThatRequest()
        => WithSyntheticModel(layers: 2, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 256, (model, config, spvDir) =>
        {
            var kv4 = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4);
            var kv3 = kv4 with { TurboQuantBits = 3 };
            Assert.False(model.SupportsInPlaceTurboQuant(kv4), "nothing is built yet");
            Assert.Equal(0, model.MaxPrefillForwardTokens);

            Assert.Null(DotLLM.Server.TurboQuantKv.PostLoadRefusal(kv4, null, DotLLM.Server.KvCommand.Serve, paged: false, config, model,
                hasDraftModel: false, mtpWillAttach: false));
            Assert.True(model.SupportsInPlaceTurboQuant(kv4), "the twin set is built and complete");
            Assert.False(model.SupportsInPlaceTurboQuant(kv3), "the twins are specialised for 4 bits");
            Assert.False(model.SupportsInPlaceTurboQuant(kv4 with { TurboQuantUseQjl = true }));
            Assert.False(model.SupportsInPlaceTurboQuant(kv4 with { MixedPrecisionWindowSize = 16 }));
            Assert.False(model.SupportsInPlaceTurboQuant(new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.Q8_0, TurboQuantBits: 4)));
            Assert.Equal(VulkanTransformerModel.MaxTurboQuantRowsPerForward(config.NumAttentionHeads, config.NumKvHeads), model.MaxPrefillForwardTokens);

            using (var cache = DotLLM.Server.TurboQuantKv.CreateFactory(model, kv4)(config, 64))
            {
                var vk = Assert.IsType<VulkanKvCache>(cache);
                Assert.Equal(VulkanKvStorage.Tq, vk.Storage);
                using var logits = model.Forward(SyntheticTokens(8, config.VocabSize), Enumerable.Range(0, 8).ToArray(), -1, cache);
                Assert.True(logits.Shape[0] >= 1);
            }

            // Asking for another width through the helper rebuilds the set for it: the capability follows the request.
            Assert.Null(DotLLM.Server.TurboQuantKv.PostLoadRefusal(kv3, null, DotLLM.Server.KvCommand.Run, paged: false, config, model, false, false));
            Assert.True(model.SupportsInPlaceTurboQuant(kv3));
            Assert.False(model.SupportsInPlaceTurboQuant(kv4));
        });

    /// <summary>
    /// #1892: a refusal path and a finally may both dispose the model (run's post-load refusal used to); a second Dispose must be a no-op.
    /// The synthetic model's own <c>using</c> is a third.
    /// </summary>
    [SkippableFact]
    public void Dispose_IsIdempotent()
        => WithSyntheticModel(layers: 1, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 64, (model, config, spvDir) =>
        {
            model.Dispose();
            model.Dispose();
        });

    /// <summary>
    /// #1892 Task 8: the shared factory is self-sufficient on Vulkan. It builds the twin set itself, so a caller that skipped the post-load
    /// pass still gets the in-place cache, and a request the host cannot serve is refused with the Vulkan reason, not a #1900 host-gap message.
    /// </summary>
    [SkippableFact]
    public void CreateFactory_BuildsTheTwinSetItself_AndRefusesWithAnAccurateReason()
        => WithSyntheticModel(layers: 1, heads: 4, kvHeads: 2, hd: 32, hidden: 128, vocab: 96, maxSeq: 128, (model, config, spvDir) =>
        {
            var kv = new KvCacheConfig(KvCacheDType.TurboQuant, KvCacheDType.TurboQuant, TurboQuantBits: 4);
            using (var cache = DotLLM.Server.TurboQuantKv.CreateFactory(model, kv)(config, 32))   // no PostLoadRefusal first
                Assert.Equal(VulkanKvStorage.Tq, Assert.IsType<VulkanKvCache>(cache).Storage);
            var e = Assert.Throws<NotSupportedException>(() => DotLLM.Server.TurboQuantKv.CreateFactory(model, kv with { TurboQuantUseQjl = true }));
            Assert.Contains("Vulkan host", e.Message);
            Assert.DoesNotContain("there is no other TurboQuant cache", e.Message);
        });

    /// <summary>
    /// A prompt longer than one TurboQuant forward may take (32 KV heads: 2047 rows) through <see cref="TextGenerator"/> with no
    /// chunk size configured: the generator chunks by the model's preference instead of meeting the guard in <c>Forward</c>, and
    /// the result equals a run with an explicit small chunk size (one layer, so both hold identical codes).
    /// </summary>
    [SkippableFact]
    public void Generator_PromptLongerThanOneTurboQuantForward_IsChunkedByTheModelPreference()
        => WithSyntheticModel(layers: 1, heads: 32, kvHeads: 32, hd: 16, hidden: 64, vocab: 128, maxSeq: 4096, (model, config, spvDir) =>
        {
            const int bits = 4;
            var ck = new TurboQuantCodec(16, bits, Seed, useQjl: false);
            var cv = new TurboQuantCodec(16, bits, Seed ^ VSeedXor, useQjl: false);
            Assert.True(model.TryUseInPlaceTurboQuantKv(spvDir, bits, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD));
            const int cap = 65535 / 32;   // 2047 rows
            var rng = new Random(0x6E);
            string prompt = new(Enumerable.Range(0, 2600).Select(_ => (char)rng.Next(32, 127)).ToArray());
            Assert.True(prompt.Length > cap);
            var opts = new InferenceOptions { Temperature = 0, MaxTokens = 4, Logprobs = true };

            var byPreference = new TextGenerator(model, new CharTokenizer(), (_, size) => model.CreateKvCache(size), mtpEnabled: false);
            long attn0 = model.TurboQuantAttentionCount;
            var a = byPreference.Generate(prompt, opts);   // a model that asked for one pass would meet the guard in Forward here
            Assert.Equal(cap, model.PreferredPrefillChunkTokens);
            Assert.Equal(cap, byPreference.EffectivePrefillChunkSize);
            Assert.Equal(prompt.Length, a.PromptTokenCount);
            Assert.True(model.TurboQuantAttentionCount - attn0 >= 2 + 1, "the prefill did not run as at least two forwards");

            var explicitChunks = new TextGenerator(model, new CharTokenizer(), (_, size) => model.CreateKvCache(size), mtpEnabled: false, prefillChunkSize: 512);
            var b = explicitChunks.Generate(prompt, opts);
            Assert.Equal(b.GeneratedTokenIds, a.GeneratedTokenIds);
            for (int i = 0; i < a.Logprobs!.Length; i++)
                Assert.True(Math.Abs(a.Logprobs[i].Logprob - b.Logprobs![i].Logprob) < 2e-2,
                    $"token {i}: logprob {a.Logprobs[i].Logprob:R} by preference vs {b.Logprobs[i].Logprob:R} with 512-row chunks");

            // #1892: an explicit chunk size above the bound is CLAMPED to it (with one log line), not refused: it used to override the
            // preference and meet the guard in Forward. 0 (one pass) becomes the bound too, so no engine path hands the model a whole prompt.
            var tooBig = new TextGenerator(model, new CharTokenizer(), (_, size) => model.CreateKvCache(size), mtpEnabled: false, prefillChunkSize: cap + 1);
            Assert.Equal(cap, tooBig.EffectivePrefillChunkSize);
            var c = tooBig.Generate(prompt, opts);
            Assert.Equal(a.GeneratedTokenIds, c.GeneratedTokenIds);
            // The last-resort guard in Forward is still there for a caller that bypasses the generator.
            using var direct = model.CreateKvCache(prompt.Length + 8);
            var ids = new CharTokenizer().Encode(prompt).ToArray();
            Assert.Throws<NotSupportedException>(() => model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, direct));
        });
}
