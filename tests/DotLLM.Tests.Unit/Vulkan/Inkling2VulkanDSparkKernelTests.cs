using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Op-level parity of the Inkling 2 Vulkan DSpark shaders (#2046) against scalar double-precision references built from the CPU drafter's own math
/// (<c>Inkling2DSparkDrafter</c>). Inputs are BF16 / F16 exactly as the GPU stores them; outputs are compared at BF16 resolution. Shader mutants are loaded through
/// <c>DOTLLM_INKLING2_SPV_OVERRIDE</c> (a directory holding a mutated copy of one .spv; the tracked .spv are never edited).
/// </summary>
public unsafe class Inkling2VulkanDSparkKernelTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly VulkanDevice? _dev;
    private readonly string _spv = Inkling2VulkanTests.SpvDir();
    private readonly string? _ovr = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");
    private readonly Random _rnd = new(2046);

    /// <summary>True: the activation buffers hold F16 words and the <c>_f16act</c> shader twins run (the F16 default, #2075); false: BF16.</summary>
    protected virtual bool F16Act => false;

    public Inkling2VulkanDSparkKernelTests(ITestOutputHelper output)
    {
        _output = output;
        if (Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") != "1" && VulkanDevice.IsAvailable()) _dev = VulkanDevice.Create();
    }

    public void Dispose() => _dev?.Dispose();

    private VulkanDevice Dev { get { Skip.If(_dev is null, "No Vulkan device"); return _dev!; } }

    private static readonly HashSet<string> ActTwins = ["inkling2_dspark_eltwise", "inkling2_dspark_ctxmix", "inkling2_dspark_headnorm", "inkling2_dspark_attn", "inkling2_dspark_markov_pre", "inkling2_dspark_markov_step"];

    private OlalaKernel K(string name, int buffers, int pushWords, params uint[] spec) => KIn(null, F16Act && ActTwins.Contains(name) ? name + "_f16act" : name, buffers, pushWords, spec);

    private OlalaKernel KIn(string? dirOverride, string name, int buffers, int pushWords, params uint[] spec)
    {
        string dir = dirOverride ?? (!string.IsNullOrEmpty(_ovr) && File.Exists(Path.Combine(_ovr, name + ".spv")) ? _ovr : _spv);
        return new OlalaKernel(Dev, dir, name, buffers, pushWords, spec);
    }

    private static uint F(float f) => BitConverter.SingleToUInt32Bits(f);
    private float Rnd(float x) => F16Act ? (float)(Half)x : Bf(x);   // round to the ACTIVATION format
    private static float Bf(float x) => BitConverter.Int32BitsToSingle(Inkling2ConvState.ToBf16(x) << 16);
    private static float BfBits(ushort b) => BitConverter.Int32BitsToSingle(b << 16);
    private float[] Rand(int n, float s = 1f) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(_rnd.NextDouble() * 2 - 1) * s; return a; }

    private VulkanDevice.Buffer UpBytes(byte[] a) { var b = Dev.AllocateDeviceLocal(Math.Max(16, (a.Length + 3L) / 4 * 4)); Dev.Upload(a.Length % 4 == 0 ? a : a.Concat(new byte[4 - a.Length % 4]).ToArray(), b); return b; }
    private VulkanDevice.Buffer UpF(float[] a) { var b = Dev.AllocateDeviceLocal(Math.Max(16, a.Length * 4L)); Dev.Upload(a, b); return b; }
    private VulkanDevice.Buffer UpAct(float[] a) => F16Act ? UpF16(a) : UpBf16(a);
    private VulkanDevice.Buffer UpBf16(float[] a) => UpBytes(MemoryMarshal.AsBytes(a.Select(Inkling2ConvState.ToBf16).ToArray().AsSpan()).ToArray());
    private VulkanDevice.Buffer UpF16(float[] a) => UpBytes(MemoryMarshal.AsBytes(a.Select(x => BitConverter.HalfToUInt16Bits((Half)x)).ToArray().AsSpan()).ToArray());
    private VulkanDevice.Buffer Zero(long bytes) { var b = Dev.AllocateDeviceLocal(Math.Max(16, (bytes + 3) / 4 * 4)); Dev.FillBufferSynchronous(b, 0); return b; }
    private float[] DownAct(VulkanDevice.Buffer b, int n) => F16Act ? DownF16(b, n) : DownBf16(b, n);
    private float[] DownBf16(VulkanDevice.Buffer b, int n) { var raw = new float[(n + 1) / 2]; Dev.Download(b, raw); var u = MemoryMarshal.Cast<float, ushort>(raw.AsSpan()); var r = new float[n]; for (int i = 0; i < n; i++) r[i] = BfBits(u[i]); return r; }
    private float[] DownF16(VulkanDevice.Buffer b, int n) { var raw = new float[(n + 1) / 2]; Dev.Download(b, raw); var u = MemoryMarshal.Cast<float, ushort>(raw.AsSpan()); var r = new float[n]; for (int i = 0; i < n; i++) r[i] = (float)BitConverter.UInt16BitsToHalf(u[i]); return r; }
    private float[] DownF(VulkanDevice.Buffer b, int n) { var r = new float[n]; Dev.Download(b, r); return r; }

    private readonly record struct Step(OlalaKernel Kernel, uint[] Push, uint Gx, uint Gy, uint Gz, VulkanDevice.Buffer[] Bufs);

    private void Run(params Step[] steps)
    {
        using var sub = Dev.CreateSubmitContext();
        sub.Begin();
        KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
        foreach (var s in steps)
        {
            s.Kernel.Record(sub.CommandBuffer, s.Bufs, s.Push, s.Gx, s.Gy, s.Gz);
            KernelSupport.ComputeToComputeBarrier(sub.CommandBuffer);
        }
        sub.SubmitAndWait();
        foreach (var s in steps) s.Kernel.Dispose();
    }

    private double Check(string what, float[] want, float[] got, double tol, bool soft = false)
    {
        double worst = 0; int at = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = Math.Abs(want[i] - got[i]) / (1.0 + Math.Abs(want[i]));
            if (double.IsNaN(got[i]) || d > worst) { worst = double.IsNaN(got[i]) ? 1e9 : d; at = i; }
        }
        _output.WriteLine($"{what}: worst normalised error {worst:E2} at {at}");
        if (!soft && !(worst < tol)) Assert.Fail($"{what}: worst normalised error {worst:E3} at {at} (want {want[at]}, got {got[at]})");
        return worst;
    }

    // ───────────── copy / eltwise ─────────────

    [SkippableFact]
    public void Copy_StridedRows_IsVerbatim()
    {
        int W = 300, rows = 3;
        var src = Rand(rows * W).Select(Rnd).ToArray();
        var dst = Zero(rows * 4 * W * 2L);
        Run(new Step(K("inkling2_dspark_copy", 2, 6), [(uint)W, (uint)rows, 0, (uint)W, (uint)(2 * W), (uint)(4 * W)], 2, (uint)rows, 1, [UpAct(src), dst]));
        var got = DownAct(dst, rows * 4 * W);
        for (int r = 0; r < rows; r++) for (int c = 0; c < W; c++) Assert.Equal(src[r * W + c], got[r * 4 * W + 2 * W + c]);
        Assert.Equal(0f, got[0]);   // the other slots are untouched
    }

    [SkippableTheory]
    [InlineData(0u)]
    [InlineData(1u)]
    public void Eltwise_MatchesReference(uint mode)
    {
        int n = 1000;
        var a = Rand(n, 3).Select(Rnd).ToArray(); var b = Rand(n, 3).Select(Rnd).ToArray();
        var want = new float[n];
        for (int i = 0; i < n; i++) want[i] = mode == 0 ? a[i] + b[i] : (float)(a[i] / (1.0 + Math.Exp(-a[i])) * b[i]);
        var ba = UpAct(a);
        Run(new Step(K("inkling2_dspark_eltwise", 2, 1, mode), [(uint)n], 4, 1, 1, [ba, UpAct(b)]));
        Check("eltwise" + mode, want, DownAct(ba, n), 6e-3);
    }

    // ───────────── ctx mix (K/V injection input and f_rnn) ─────────────

    [SkippableTheory]
    [InlineData(0u)]
    [InlineData(1u)]
    public void CtxMix_MatchesReference(uint mode)
    {
        int H = 2304, rows = mode == 1 ? 45 : 3, posBase = 14;
        var taps = Rand(rows * 4 * H, 4).Select(Rnd).ToArray();
        var mix = new float[] { 0.1f, 0.2f, 0.3f, 0.4f };
        var gamma = Rand(H).Select(x => 1 + x).ToArray();
        var want = new float[rows * H];
        for (int r = 0; r < rows; r++)
        {
            var m = new float[H];
            for (int k = 0; k < 4; k++) for (int i = 0; i < H; i++) m[i] += mix[k] * taps[(r * 4 + k) * H + i];
            Inkling2Ops.RmsNorm(m, gamma, 1e-6f, want.AsSpan(r * H, H));
        }
        var y = Zero(mode == 0 ? rows * H * 2L : 16L * H * 2);
        Run(new Step(K("inkling2_dspark_ctxmix", 4, 3, mode), [(uint)H, (uint)posBase, F(1e-6f)], (uint)rows, 1, 1, [UpAct(taps), UpF(mix), UpF(gamma), y]));
        if (mode == 0) Check("ctxmix bf16", want, DownAct(y, rows * H), 8e-3);
        else
        {
            var ring = DownF16(y, 16 * H);
            var got = new float[rows * H];
            for (int r = 0; r < rows; r++) Array.Copy(ring, ((posBase + r) % 16) * H, got, r * H, H);
            // rows r and r + 16 share a slot: the slot holds the LAST row written, so only the last 16 rows can be read back
            Check("ctxmix f16 ring", want.AsSpan((rows - 16) * H).ToArray(), got.AsSpan((rows - 16) * H).ToArray(), 2e-3);
        }
    }

    // ───────────── per-head norm + RoPE + cache store ─────────────

    private static void RefHeadNorm(float[] x, int heads, int row, float[] gamma, bool norm, bool rope, int pos, float[] dst, int dstOff)
    {
        const int hd = 128;
        var cos = new float[hd / 2]; var sin = new float[hd / 2];
        for (int i = 0; i < hd / 2; i++) { double a = (double)pos * Math.Pow(1e6, -2.0 * i / hd); cos[i] = (float)Math.Cos(a); sin[i] = (float)Math.Sin(a); }
        for (int h = 0; h < heads; h++)
        {
            var v = x.AsSpan((row * heads + h) * hd, hd).ToArray();
            if (norm) Inkling2Ops.RmsNorm(v, gamma, 1e-6f, v);
            if (rope) RoPE.ApplyRotationNeoX(v, cos, sin, hd);
            v.CopyTo(dst, dstOff + h * hd);
        }
    }

    private float[] RopeTab()
    {
        var t = new float[128];
        for (int i = 0; i < 64; i++) { double f = Math.Pow(1e6, -2.0 * i / 128) / (2 * Math.PI); float hi = (float)f; t[2 * i] = hi; t[2 * i + 1] = (float)(f - hi); }
        return t;
    }

    [SkippableTheory]
    [InlineData(5, 0)]
    [InlineData(3000, 0)]
    [InlineData(100000, 0)]
    [InlineData(1000, 40)]
    public void HeadNormRope_BF16Out_MatchesReference(int posBase, int unusedRing)
    {
        _ = unusedRing;
        int heads = 3, rows = 4;
        var x = Rand(rows * heads * 128, 3).Select(Rnd).ToArray(); var gamma = Rand(128).Select(v => 1 + v).ToArray();
        var want = new float[rows * heads * 128];
        for (int r = 0; r < rows; r++) RefHeadNorm(x, heads, r, gamma, true, true, posBase + r, want, r * heads * 128);
        var y = Zero(rows * heads * 128 * 2L);
        Run(new Step(K("inkling2_dspark_headnorm", 4, 7, 1, 1, 0), [(uint)heads, (uint)posBase, 0, 0, 0, 0, F(1e-6f)], (uint)heads, (uint)rows, 1, [UpAct(x), UpF(gamma), UpF(RopeTab()), y]));
        Check($"headnorm+rope pos {posBase}", want, DownAct(y, want.Length), 1.5e-2);
    }

    /// <summary>K/V store into a ring: row r lands in slot (slotBase + r) % ring, F16; the RoPE position is independent of the slot. A ring of the wrong size lands rows in other slots.</summary>
    [SkippableTheory]
    [InlineData(1153, 1100, 1)]
    [InlineData(10, 7, 1)]
    [InlineData(0, 5, 1)]
    [InlineData(10, 7, 0)]
    public void HeadNormRope_F16RingStore_PlacesRowsInTheirSlots(int ring, int basePos, int rope)
    {
        int heads = 8, rows = 6;
        var x = Rand(rows * heads * 128, 3).Select(Rnd).ToArray(); var gamma = Rand(128).Select(v => 1 + v).ToArray();
        int slots = ring == 0 ? basePos + rows : ring;
        var y = Zero((long)slots * heads * 128 * 2);
        Run(new Step(K("inkling2_dspark_headnorm", 4, 7, 1, (uint)rope, 1), [(uint)heads, (uint)basePos, (uint)basePos, (uint)ring, 0, 0, F(1e-6f)], (uint)heads, (uint)rows, 1,
            [UpAct(x), UpF(gamma), UpF(RopeTab()), y]));
        var got = DownF16(y, slots * heads * 128);
        double worst = 0;
        for (int r = 0; r < rows; r++)
        {
            var want = new float[heads * 128];
            RefHeadNorm(x, heads, r, gamma, true, rope == 1, basePos + r, want, 0);
            int slot = ring == 0 ? basePos + r : (basePos + r) % ring;
            for (int i = 0; i < want.Length; i++) worst = Math.Max(worst, Math.Abs(want[i] - got[slot * heads * 128 + i]) / (1 + Math.Abs(want[i])));
        }
        _output.WriteLine($"ring {ring} basePos {basePos} rope {rope}: worst {worst:E2}");
        Assert.True(worst < 2e-3, $"worst {worst}");
    }

    // ───────────── block attention ─────────────

    private (float[] q, float[] kc, float[] vc, float[] kb, float[] vb) AttnData(int b, int slots)
    {
        const int heads = 24, kv = 8;
        return (Rand(b * heads * 128, 2).Select(Rnd).ToArray(), Rand(slots * kv * 128, 2).Select(x => (float)(Half)x).ToArray(), Rand(slots * kv * 128).Select(x => (float)(Half)x).ToArray(),
                Rand(b * kv * 128, 2).Select(x => (float)(Half)x).ToArray(), Rand(b * kv * 128).Select(x => (float)(Half)x).ToArray());
    }

    /// <summary>Double-precision reference. Context key at position p lives at slot (ring == 0 ? p : p % ring).</summary>
    private static float[] RefAttn(float[] q, float[] kc, float[] vc, float[] kb, float[] vb, int b, int lo, int pos0, int ring)
    {
        const int heads = 24, kv = 8, hd = 128;
        var o = new float[b * heads * hd];
        int nCtx = pos0 - lo;
        for (int h = 0; h < heads; h++)
        {
            int g = h / 3;
            for (int i = 0; i < b; i++)
            {
                var sc = new double[nCtx + b];
                for (int j = 0; j < nCtx + b; j++)
                {
                    double s = 0;
                    for (int d = 0; d < hd; d++)
                    {
                        double kk = j < nCtx ? kc[((ring == 0 ? lo + j : (lo + j) % ring) * kv + g) * hd + d] : kb[((j - nCtx) * kv + g) * hd + d];
                        s += q[(i * heads + h) * hd + d] * kk;
                    }
                    sc[j] = s / Math.Sqrt(hd);
                }
                double m = sc.Max(), den = 0;
                for (int j = 0; j < sc.Length; j++) { sc[j] = Math.Exp(sc[j] - m); den += sc[j]; }
                for (int d = 0; d < hd; d++)
                {
                    double acc = 0;
                    for (int j = 0; j < sc.Length; j++)
                        acc += sc[j] * (j < nCtx ? vc[((ring == 0 ? lo + j : (lo + j) % ring) * kv + g) * hd + d] : vb[((j - nCtx) * kv + g) * hd + d]);
                    o[(i * heads + h) * hd + d] = (float)(acc / den);
                }
            }
        }
        return o;
    }

    private float[] RunAttn(float[] q, float[] kc, float[] vc, float[] kb, float[] vb, int b, int lo, int pos0, int ring)
    {
        var o = Zero(b * 24 * 128 * 2L);
        Run(new Step(K("inkling2_dspark_attn", 6, 7), [(uint)b, (uint)lo, (uint)pos0, (uint)ring, 24, 8, F(1f / MathF.Sqrt(128))], 8, 1, 1,
            [UpAct(q), UpF16(kc), UpF16(vc), UpF16(kb), UpF16(vb), o]));
        return DownAct(o, b * 24 * 128);
    }

    [SkippableTheory]
    [InlineData(4, 0, 0, 0)]       // no context
    [InlineData(4, 0, 63, 0)]      // a tile boundary
    [InlineData(4, 0, 65, 0)]
    [InlineData(8, 0, 300, 0)]
    [InlineData(4, 976, 2000, 0)]  // a 1024-key window over a linear cache
    [InlineData(4, 976, 2000, 1153)]   // the same window over the ring
    [InlineData(2, 0, 1500, 0)]    // global layer
    public void BlockAttention_MatchesReference(int b, int lo, int pos0, int ring)
    {
        int slots = ring == 0 ? Math.Max(pos0, 1) : ring;
        var (q, kc, vc, kb, vb) = AttnData(b, slots);
        var want = RefAttn(q, kc, vc, kb, vb, b, lo, pos0, ring);
        var got = RunAttn(q, kc, vc, kb, vb, b, lo, pos0, ring);
        Check($"attn b={b} lo={lo} pos0={pos0} ring={ring}", want, got, 1.2e-2);
    }

    /// <summary>
    /// The window edge is observable: the OLDEST visible key is made decisive (a key identical to the query direction with a distinctive value), so dropping it, or letting
    /// a too-short ring overwrite it, changes the output by an amount far above the BF16 noise. This is the op-level guard behind the 'ring one row too short' mutant of #2060.
    /// </summary>
    [SkippableFact]
    public void BlockAttention_OldestKeyIsDecisive_AndARingOneSlotTooShortLosesIt()
    {
        int b = 4, window = 1024, pos0 = 3000, lo = pos0 - window;
        int ring = window + 4;                        // the minimum the verify needs: window + the rows a rejected verify may have written ahead
        var (q, kc, vc, kb, vb) = AttnData(b, ring);
        // every key weak, except the oldest visible one (position lo) which equals the first query vector of every head's group
        for (int i = 0; i < kc.Length; i++) kc[i] *= 0.05f;
        for (int g = 0; g < 8; g++)
            for (int d = 0; d < 128; d++)
            {
                kc[((lo % ring) * 8 + g) * 128 + d] = (float)(Half)(q[(0 * 24 + g * 3) * 128 + d] * 4);
                vc[((lo % ring) * 8 + g) * 128 + d] = (float)(Half)(10f * (d % 2 == 0 ? 1 : -1));
            }
        var good = RunAttn(q, kc, vc, kb, vb, b, lo, pos0, ring);
        var wantGood = RefAttn(q, kc, vc, kb, vb, b, lo, pos0, ring);
        Check("oldest key (correct ring)", wantGood, good, 1.5e-2);

        var wantDropped = RefAttn(q, kc, vc, kb, vb, b, lo + 1, pos0, ring);   // window one key too narrow
        double dropEffect = 0;
        for (int i = 0; i < wantGood.Length; i++) dropEffect = Math.Max(dropEffect, Math.Abs(wantGood[i] - wantDropped[i]));
        _output.WriteLine($"dropping the oldest key moves the output by up to {dropEffect:F3}");
        Assert.True(dropEffect > 0.5, "the oldest key must matter on this data or the test cannot see a window/ring error");

        // The same data in a ring one slot too short: position pos0 + 3 (a rejected row written ahead) aliases the oldest key's slot.
        var kShort = (float[])kc.Clone(); var vShort = (float[])vc.Clone();
        int shortRing = ring - 1;
        // Rebuild the cache at the short ring size, as the injection would: every position of the window, then the rows written ahead.
        var k2 = new float[shortRing * 8 * 128]; var v2 = new float[shortRing * 8 * 128];
        for (int p = lo; p < pos0 + 4; p++)
        {
            int srcSlot = p % ring, dstSlot = p % shortRing;
            bool ahead = p >= pos0;   // rows written ahead hold different data
            for (int i = 0; i < 8 * 128; i++) { k2[dstSlot * 8 * 128 + i] = ahead ? 0.05f : kShort[srcSlot * 8 * 128 + i]; v2[dstSlot * 8 * 128 + i] = ahead ? 3f : vShort[srcSlot * 8 * 128 + i]; }
        }
        var gotShort = RunAttn(q, k2, v2, kb, vb, b, lo, pos0, shortRing);
        double shortEffect = 0;
        for (int i = 0; i < wantGood.Length; i++) shortEffect = Math.Max(shortEffect, Math.Abs(wantGood[i] - gotShort[i]));
        _output.WriteLine($"a ring one slot too short moves the output by up to {shortEffect:F3}");
        Assert.True(shortEffect > 0.5, "the too-short ring must change the attention output");
    }

    // ───────────── Markov head ─────────────

    private sealed class MarkovRef
    {
        public int H, R = 256, V;
        public float[] J = null!, W1 = null!, W2 = null!, Jb = null!, B2 = null!;
    }

    /// <summary>Double reference of <c>MarkovWalk</c> over the GPU's BF16 weights and BF16-rounded pre-activations; returns tokens and the softcapped logits rows.</summary>
    private (int[] tokens, float[][] cols) RefWalk(MarkovRef m, float[] hid, float[] frnn, int[] anchorAndNone, int n, float cap, float[][] baseLogits, bool noState = false, bool capFirst = false)
    {
        int H = m.H, R = m.R, V = m.V, stride = 2 * R + 2 * H;
        var pre = new float[n][];
        for (int r = 0; r < n; r++)
        {
            pre[r] = new float[3 * R];
            for (int j = 0; j < 3 * R; j++)
            {
                double s = m.Jb[j];
                for (int k = 0; k < H; k++) s += m.J[j * stride + 2 * R + k] * (double)hid[r * H + k] + m.J[j * stride + 2 * R + H + k] * (double)frnn[k];
                pre[r][j] = Rnd((float)s);
            }
        }
        var tokens = new int[n]; var cols = new float[n][];
        var state = new double[R];
        int prev = anchorAndNone[0];
        for (int i = 1; i <= n; i++)
        {
            if (noState) Array.Clear(state);
            var jp = new double[3 * R];
            for (int j = 0; j < 3 * R; j++)
            {
                double s = pre[i - 1][j];
                for (int k = 0; k < R; k++) s += m.J[j * stride + k] * state[k] + m.J[j * stride + R + k] * (double)m.W1[prev * R + k];
                jp[j] = s;
            }
            var c2 = new float[R];
            for (int j = 0; j < R; j++)
            {
                double gate = 1 / (1 + Math.Exp(-jp[j])), cand = Math.Tanh(jp[R + j]);
                state[j] = cand + gate * (state[j] - cand);
                c2[j] = Rnd((float)Math.Tanh(jp[2 * R + j]));
            }
            var col = new float[V]; int best = 0;
            for (int v = 0; v < V; v++)
            {
                double bias = m.B2[v];
                for (int k = 0; k < R; k++) bias += m.W2[v * R + k] * (double)c2[k];
                double x = capFirst ? cap * Math.Tanh(baseLogits[i - 1][v] / cap) + bias : baseLogits[i - 1][v] + bias;
                if (!capFirst && cap > 0) x = cap * Math.Tanh(x / cap);
                col[v] = (float)x;
                if (col[v] > col[best]) best = v;
            }
            cols[i - 1] = col; tokens[i - 1] = best; prev = best;
        }
        return (tokens, cols);
    }

    private sealed record MarkovRun(int[] Tokens, float[][] Cols, MarkovRef M, float[] Hid, float[] Frnn, float[][] BaseL, int Anchor);

    private MarkovRun RunMarkov(int n, int V, bool noStateMutantGpu = false, bool capFirstGpu = false, bool logitsF16 = false)
    {
        int H = 2304, R = 256, stride = 2 * R + 2 * H;
        var m = new MarkovRef { H = H, V = V };
        m.J = Rand(768 * stride, 0.05f).Select(Bf).ToArray(); m.W1 = Rand(V * R, 1).Select(Bf).ToArray(); m.W2 = Rand(V * R, 0.3f).Select(Bf).ToArray();
        m.Jb = Rand(768, 0.5f); m.B2 = Rand(V, 0.5f);
        var hid = Rand(n * H, 1).Select(Rnd).ToArray();
        var frnnRing = Rand(16 * H, 1).Select(x => (float)(Half)x).ToArray();
        int slot = 5;
        var frnn = frnnRing.AsSpan(slot * H, H).ToArray();
        var baseL = Enumerable.Range(0, n).Select(_ => Rand(V, 12)).ToArray();
        float cap = 30f;
        int anchor = 17 % V;

        var tok = Dev.Allocate(64);
        Dev.Upload(MemoryMarshal.AsBytes(new ReadOnlySpan<int>([anchor])), tok);
        var baseFlat = baseL.SelectMany(x => x).ToArray();
        if (logitsF16) for (int i = 0; i < baseFlat.Length; i++) { baseFlat[i] = (float)(Half)baseFlat[i]; baseL[i / V][i % V] = baseFlat[i]; }   // the GPU's base logits are the F16-rounded GEMV outputs
        var logits = logitsF16 ? UpF16(baseFlat) : UpF(baseFlat);
        var jb = UpBf16(m.J); var hb = UpAct(hid); var fb = UpF16(frnnRing); var jbias = UpF(m.Jb);
        var pre = Zero(n * 768 * 2L); var w1 = UpBf16(m.W1); var w2 = UpBf16(m.W2); var b2 = UpF(m.B2); var c2 = Zero(256 * 2);
        int parts = (V + 31) / 32;
        var part = Zero(parts * 8L);
        var steps = new List<Step> { new(K("inkling2_dspark_markov_pre", 5, 4), [(uint)H, (uint)R, (uint)n, (uint)slot], 768, 1, 1, [jb, hb, fb, jbias, pre]) };
        for (int i = 1; i <= n; i++)
        {
            steps.Add(new(K("inkling2_dspark_markov_step", 5, 5), [(uint)H, (uint)R, (uint)i, (uint)V, noStateMutantGpu ? (uint)i : 1u], 1, 1, 1, [jb, w1, pre, tok, c2]));
            steps.Add(new(K(logitsF16 ? "inkling2_dspark_markov_logits_f16" : "inkling2_dspark_markov_logits", 5, 4), [(uint)V, (uint)((long)(i - 1) * V), 1u, F(capFirstGpu ? 0f : cap)], (uint)parts, 1, 1, [logits, w2, b2, c2, part]));
            steps.Add(new(K("inkling2_dspark_argmax", 2, 2), [(uint)parts, (uint)i], 1, 1, 1, [part, tok]));
        }
        Run(steps.ToArray());
        var ids = new float[8]; Dev.Download(tok, ids);
        var gotTok = MemoryMarshal.Cast<float, int>(ids.AsSpan()).Slice(1, n).ToArray();
        var all = logitsF16 ? DownF16(logits, n * V) : DownF(logits, n * V);
        var cols = Enumerable.Range(0, n).Select(r => all.AsSpan(r * V, V).ToArray()).ToArray();
        return new MarkovRun(gotTok, cols, m, hid, frnn, baseL, anchor);
    }

    [SkippableTheory]
    [InlineData(1, 500)]
    [InlineData(3, 1000)]
    [InlineData(7, 777)]
    public void MarkovHead_MatchesReferenceWalk(int n, int V)
    {
        var r = RunMarkov(n, V);
        var (refTok, refCols) = RefWalk(r.M, r.Hid, r.Frnn, [r.Anchor], n, 30f, r.BaseL);
        double worst = 0;
        for (int i = 0; i < n; i++) worst = Math.Max(worst, Check($"markov col {i}", refCols[i], r.Cols[i], 0.05, soft: true));
        _output.WriteLine($"tokens gpu [{string.Join(",", r.Tokens)}] ref [{string.Join(",", refTok)}]");
        Assert.True(worst < 0.05, $"worst normalised logit error {worst:E2}");
        // the GPU token chain equals the reference chain unless a near-tie (margin < 0.05) flips it
        for (int i = 0; i < n; i++)
        {
            var sorted = refCols[i].OrderByDescending(x => x).Take(2).ToArray();
            if (sorted[0] - sorted[1] > 0.05 && r.Tokens[i] != refTok[i]) { Assert.Equal(refTok[i], r.Tokens[i]); }
            if (r.Tokens[i] != refTok[i]) break;   // later steps condition on a different token: stop comparing chains
        }
        for (int i = 0; i < n; i++) Assert.Equal(TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)r.Cols[i]), r.Tokens[i]);
    }

    /// <summary>Mutants of the walk (state reset every step; softcap before the bias) must move the logits far beyond the reference tolerance.</summary>
    [SkippableTheory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void MarkovHead_Mutants_AreFarFromTheReference(bool noState, bool capFirst)
    {
        int n = 4, V = 600;
        var r = RunMarkov(n, V, noState, capFirst);
        var (_, refCols) = RefWalk(r.M, r.Hid, r.Frnn, [r.Anchor], n, 30f, r.BaseL);
        double worst = 0;
        for (int i = 1; i < n; i++) worst = Math.Max(worst, Check($"mutant col {i}", refCols[i], r.Cols[i], 0, soft: true));
        _output.WriteLine($"mutant noState={noState} capFirst={capFirst}: worst normalised logit error {worst:E2}");
        Assert.True(worst > 0.05, "the mutant must be visible");
    }

    // ----- #2065: the conv rounds its in-batch history exactly like Inkling2ConvState.Run -----

    private static string ShadersDir()
    {
        string d = AppContext.BaseDirectory;
        for (int i = 0; i < 10 && d is not null; i++, d = Path.GetDirectoryName(d)!)
            if (Directory.Exists(Path.Combine(d, "native", "vulkan", "shaders"))) return Path.Combine(d, "native", "vulkan", "shaders");
        throw new DirectoryNotFoundException("native/vulkan/shaders");
    }

    /// <summary>Compiles the F32-activation conv shader with its source text edited (a mutant), into a fresh directory; null when glslc is unavailable.</summary>
    private static string? CompileConvMutant(string find, string replace)
    {
        string src = File.ReadAllText(Path.Combine(ShadersDir(), "inkling2_conv.glsl"));
        Assert.Contains(find, src);
        string dir = Path.Combine(Path.GetTempPath(), "conv-mutant-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "inkling2_conv.glsl"), src.Replace(find, replace));
        File.WriteAllText(Path.Combine(dir, "inkling2_conv_f32act.comp"), "#version 450\n#extension GL_GOOGLE_include_directive : require\n#define ACT_F32 1\n#include \"inkling2_conv.glsl\"\n");
        try
        {
            var psi = new System.Diagnostics.ProcessStartInfo("glslc", $"--target-env=vulkan1.2 -O -I \"{ShadersDir()}\" -o \"{Path.Combine(dir, "inkling2_conv_f32act.spv")}\" \"{Path.Combine(dir, "inkling2_conv_f32act.comp")}\"")
            { RedirectStandardError = true, UseShellExecute = false };
            using var pr = System.Diagnostics.Process.Start(psi)!;
            string err = pr.StandardError.ReadToEnd();
            pr.WaitForExit();
            Assert.True(pr.ExitCode == 0, err);
            return dir;
        }
        catch (System.ComponentModel.Win32Exception) { return null; }
    }

    private (float[] y8, float[] y1, byte[] st8, byte[] st1) ConvEightVsSingles(string? dir, bool dyn)
    {
        int C = 300, T = 8, heads = 3;
        var x = Rand(T * C, 3f);                 // F32 activations that are NOT BF16-exact (the reference arm)
        var w = Rand(C * 4);
        var dynv = Rand(T * heads * 4);
        var st0 = Zero(C * 3 * 2L);
        var y8b = Zero(T * C * 4L);
        var xb = UpF(x); var wb = UpF(w); var db = UpF(dynv);
        uint dy = dyn ? 1u : 0u;
        Run(new Step(KIn(dir, "inkling2_conv_f32act", 7, 6, dy, 0, 0), [(uint)C, (uint)T, (uint)heads, 0, 0, 0], (uint)((C + 63) / 64), 1, 1, [xb, wb, db, st0, db, y8b, db]));
        var y8 = DownF(y8b, T * C);
        var st8 = new float[(C * 3 * 2 + 3) / 4]; Dev.Download(st0, st8);

        var st1b = Zero(C * 3 * 2L);
        var y1 = new float[T * C];
        for (int t = 0; t < T; t++)
        {
            var yb = Zero(C * 4L);
            var dyn1 = UpF(dynv.AsSpan(t * heads * 4, heads * 4).ToArray());
            Run(new Step(KIn(dir, "inkling2_conv_f32act", 7, 6, dy, 0, 0), [(uint)C, 1, (uint)heads, 0, 0, 0], (uint)((C + 63) / 64), 1, 1, [UpF(x.AsSpan(t * C, C).ToArray()), wb, dyn1, st1b, dyn1, yb, dyn1]));
            Array.Copy(DownF(yb, C), 0, y1, t * C, C);
        }
        var s1 = new float[(C * 3 * 2 + 3) / 4]; Dev.Download(st1b, s1);
        return (y8, y1, MemoryMarshal.AsBytes(st8.AsSpan()).ToArray(), MemoryMarshal.AsBytes(s1.AsSpan()).ToArray());
    }

    /// <summary>An 8-row conv forward equals 8 single-row forwards BIT FOR BIT in the F32-activation reference arm too (#2065), with and without dynamic taps.</summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Conv_EightRows_EqualEightSingleRows_BitForBit(bool dyn)
    {
        var (y8, y1, st8, st1) = ConvEightVsSingles(null, dyn);
        for (int i = 0; i < y8.Length; i++) Assert.True(BitConverter.SingleToInt32Bits(y8[i]) == BitConverter.SingleToInt32Bits(y1[i]), $"dyn={dyn}: element {i} differs ({y8[i]} vs {y1[i]})");
        Assert.True(st8.AsSpan().SequenceEqual(st1), "carried state differs");
    }

    /// <summary>Mutant: the in-batch history is NOT rounded (the shader before #2065); the 8-row forward must then differ from the single-row forwards.</summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void Conv_WithoutInBatchRounding_Mutant_IsCaught(bool dyn)
    {
        string? dir = CompileConvMutant("s2 = bf16ToF(fToBf16(x));", "s2 = x;");
        Skip.If(dir is null, "glslc not on PATH");
        var (y8, y1, _, _) = ConvEightVsSingles(dir, dyn);
        int differ = 0;
        for (int i = 0; i < y8.Length; i++) if (BitConverter.SingleToInt32Bits(y8[i]) != BitConverter.SingleToInt32Bits(y1[i])) differ++;
        _output.WriteLine($"mutant (history unrounded), dyn={dyn}: {differ} of {y8.Length} outputs differ from the single-row forwards");
        Assert.True(differ > 0, "the unrounded-history mutant survived");
    }

    /// <summary>The F16-logits Markov pass (base logits stored as F16, softcapped columns written back as F16) against the double reference walk: error at F16 resolution, the token chain equals the reference chain except at near-ties.</summary>
    [SkippableTheory]
    [InlineData(3, 1000)]
    [InlineData(7, 777)]
    public void MarkovHead_F16Logits_MatchesReferenceWalk(int n, int V)
    {
        var r = RunMarkov(n, V, logitsF16: true);
        var (refTok, refCols) = RefWalk(r.M, r.Hid, r.Frnn, [r.Anchor], n, 30f, r.BaseL);
        double worst = 0;
        for (int i = 0; i < n; i++) worst = Math.Max(worst, Check($"markov f16 col {i}", refCols[i], r.Cols[i], 0.05, soft: true));
        _output.WriteLine($"tokens gpu [{string.Join(",", r.Tokens)}] ref [{string.Join(",", refTok)}]");
        Assert.True(worst < 0.01, $"worst normalised error {worst:E2} (F16 storage is about 5e-4 relative)");
        for (int i = 0; i < n; i++)
        {
            var sorted = refCols[i].OrderByDescending(x => x).Take(2).ToArray();
            if (sorted[0] - sorted[1] > 0.05 && r.Tokens[i] != refTok[i]) Assert.Equal(refTok[i], r.Tokens[i]);
            if (r.Tokens[i] != refTok[i]) break;
        }
    }
}

/// <summary>The same op-level parity with F16 activation buffers and the <c>_f16act</c> shader twins (the default activation format since #2075). A BF16-reading shader under F16 words (the mutant of the PR) fails every test of this class that reads activations.</summary>
public sealed unsafe class Inkling2VulkanDSparkKernelF16ActTests : Inkling2VulkanDSparkKernelTests
{
    public Inkling2VulkanDSparkKernelF16ActTests(ITestOutputHelper output) : base(output) { }

    /// <inheritdoc/>
    protected override bool F16Act => true;
}
