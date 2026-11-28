using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Op-level parity of every new Inkling 2 shader (#2034) against a scalar double-precision reference built from the CPU reference's own ops
/// (<see cref="Inkling2Ops"/>, <c>Inkling2ConvState</c>, <see cref="Dequantize"/>). The F32-activation arm is checked tightly; mutated shaders are loaded
/// through <c>DOTLLM_INKLING2_SPV_OVERRIDE</c>-style directories by the mutation script (the tracked .spv are never edited).
/// </summary>
public sealed unsafe class Inkling2VulkanKernelTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly VulkanDevice? _dev;
    private readonly string _spv = Inkling2VulkanTests.SpvDir();
    private readonly string? _ovr = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");
    private readonly Random _rnd = new(1234);

    public Inkling2VulkanKernelTests(ITestOutputHelper output)
    {
        _output = output;
        if (Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") != "1" && VulkanDevice.IsAvailable()) _dev = VulkanDevice.Create();
    }

    public void Dispose() => _dev?.Dispose();

    private VulkanDevice Dev { get { Skip.If(_dev is null, "No Vulkan device"); return _dev!; } }

    private OlalaKernel K(string name, int buffers, int pushWords, params uint[] spec)
    {
        string dir = !string.IsNullOrEmpty(_ovr) && File.Exists(Path.Combine(_ovr, name + ".spv")) ? _ovr : _spv;
        return new OlalaKernel(Dev, dir, name, buffers, pushWords, spec);
    }

    private VulkanDevice.Buffer Up(float[] a) { var b = Dev.AllocateDeviceLocal(Math.Max(16, a.Length * 4L)); Dev.Upload(a, b); return b; }
    private VulkanDevice.Buffer UpBytes(byte[] a) { var b = Dev.AllocateDeviceLocal(Math.Max(16, (a.Length + 3L) / 4 * 4)); Dev.Upload(a.Length % 4 == 0 ? a : a.Concat(new byte[4 - a.Length % 4]).ToArray(), b); return b; }
    private VulkanDevice.Buffer UpU(uint[] a) => UpBytes(MemoryMarshal.AsBytes(a.AsSpan()).ToArray());
    private VulkanDevice.Buffer Zero(long bytes) { var b = Dev.AllocateDeviceLocal(Math.Max(16, bytes)); Dev.FillBufferSynchronous(b, 0); return b; }
    private float[] Down(VulkanDevice.Buffer b, int n) { var r = new float[n]; Dev.Download(b, r); return r; }
    private float[] Rand(int n, float s = 1f) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(_rnd.NextDouble() * 2 - 1) * s; return a; }
    private static uint F(float f) => BitConverter.SingleToUInt32Bits(f);

    private void Dispatch(OlalaKernel k, ReadOnlySpan<uint> push, uint gx, uint gy, uint gz, params VulkanDevice.Buffer[] bufs)
    {
        using var sub = Dev.CreateSubmitContext();
        sub.Begin();
        KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
        k.Record(sub.CommandBuffer, bufs, push, gx, gy, gz);
        sub.SubmitAndWait();
        k.Dispose();
    }

    private void Check(string what, float[] want, float[] got, double tol)
    {
        double worst = 0; int at = 0;
        for (int i = 0; i < want.Length; i++)
        {
            double d = Math.Abs(want[i] - got[i]) / (1.0 + Math.Abs(want[i]));
            if (double.IsNaN(got[i]) || d > worst) { worst = double.IsNaN(got[i]) ? 1e9 : d; at = i; }
        }
        _output.WriteLine($"{what}: worst normalised error {worst:E2} at {at}");
        if (!(worst < tol)) Assert.Fail($"{what}: worst normalised error {worst:E3} at {at} (want {want[at]}, got {got[at]})");
    }

    [SkippableFact]
    public void RmsNorm_MatchesReference()
    {
        int T = 3, H = 300;
        var x = Rand(T * H, 3f); var g = Rand(H);
        var want = new float[T * H];
        for (int t = 0; t < T; t++) Inkling2Ops.RmsNorm(x.AsSpan(t * H, H), g, 1e-6f, want.AsSpan(t * H, H));
        var y = Zero(T * H * 4L);
        Dispatch(K("inkling2_rmsnorm_f32act", 3, 5), [(uint)H, 0, 0, 1, F(1e-6f)], (uint)T, 1, 1, Up(x), Up(g), y);
        Check("rmsnorm", want, Down(y, T * H), 1e-5);
    }

    [SkippableFact]
    public void Embed_Q8_0_GainRmsNorm_MatchesReference()
    {
        int H = 256, V = 6;
        var table = Rand(V * H, 2f);
        var q = new byte[(int)Dequantize.RowByteSize(H, QuantizationType.Q8_0) * V];
        Quantize.FromFloat32(table, V * H, QuantizationType.Q8_0, q);
        var g = Rand(H); int[] ids = [4, 1, 5];
        var want = new float[ids.Length * H]; var row = new float[H];
        fixed (byte* p = q)
            for (int t = 0; t < ids.Length; t++)
            {
                Dequantize.ToFloat32((nint)p + (nint)(ids[t] * Dequantize.RowByteSize(H, QuantizationType.Q8_0)), H, QuantizationType.Q8_0, row);
                Inkling2Ops.RmsNorm(row, g, 1e-6f, want.AsSpan(t * H, H));
            }
        var y = Zero(ids.Length * H * 4L);
        Dispatch(K("inkling2_embed_f32act", 4, 3), [(uint)H, (uint)V, F(1e-6f)], (uint)ids.Length, 1, 1, UpBytes(q), UpU(ids.Select(i => (uint)i).ToArray()), Up(g), y);
        Check("embed", want, Down(y, want.Length), 1e-5);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void DepthMix_MatchesReference(bool bf16Query)
    {
        int T = 3, H = 200, nb = 3, nq = 3, slotStride = T * H;
        var blocks = new float[6 * slotStride];
        for (int s = 0; s < 6; s++) for (int t = 0; t < T; t++) { var r = Rand(H, 2f); Inkling2Ops.RmsNorm(r, default, 1e-6f, blocks.AsSpan(s * slotStride + t * H, H)); }
        var cur = Rand(T * H, 4f);
        uint[] slots = [4, 1, 5, 0, 0, 0];
        var qf = Rand(nq * H);
        // BF16 queries are rounded to bf16 first so the reference sees the same values.
        var qBits = new ushort[nq * H]; for (int i = 0; i < qf.Length; i++) { qBits[i] = Inkling2ConvState.ToBf16(qf[i]); }
        if (bf16Query) for (int i = 0; i < qf.Length; i++) qf[i] = BitConverter.Int32BitsToSingle(qBits[i] << 16);
        var want = new float[nq * T * H];
        for (int t = 0; t < T; t++)
        {
            var states = new float[(nb + 1) * H];
            for (int b = 0; b < nb; b++) blocks.AsSpan((int)slots[b] * slotStride + t * H, H).CopyTo(states.AsSpan(b * H, H));
            Inkling2Ops.RmsNorm(cur.AsSpan(t * H, H), default, 1e-6f, states.AsSpan(nb * H, H));
            var o = new float[nq * H];
            Inkling2Ops.DepthMixToken(states, nb + 1, qf, nq, H, o);
            for (int q = 0; q < nq; q++) o.AsSpan(q * H, H).CopyTo(want.AsSpan((q * T + t) * H, H));
        }
        uint[] qw = bf16Query ? PackBf16(qBits) : qf.Select(F).ToArray();
        var y = Zero(want.Length * 4L);
        var push = new List<uint> { (uint)H, (uint)T, (uint)nq, (uint)nb, (uint)slotStride, 0, F(1e-6f) }; push.AddRange(slots);
        Dispatch(K("inkling2_mix_f32act", 4, 13, bf16Query ? 1u : 0u), push.ToArray(), (uint)T, 1, 1, Up(blocks), Up(cur), UpU(qw), y);
        Check("depth mix", want, Down(y, want.Length), 1e-5);
    }

    private static uint[] PackBf16(ushort[] b)
    {
        var w = new uint[(b.Length + 1) / 2];
        for (int i = 0; i < b.Length; i++) w[i / 2] |= (uint)b[i] << (16 * (i & 1));
        return w;
    }

    [SkippableFact]
    public void Control_InAndOut_MatchReference()
    {
        int T = 2, H = 180;
        var cur = Rand(T * H, 2f); var msg = Rand(4 * T * H, 2f); var gain = Rand(T * H, 3f); var forget = Rand(T * H, 3f); var gamma = Rand(H);
        int ctrl = 2; long msgOff = (long)ctrl * T * H;
        var wantIn = new float[T * 2 * H]; var wantOut = new float[T * H]; var o = new float[H];
        for (int t = 0; t < T; t++)
        {
            var m = msg.AsSpan((int)msgOff + t * H, H);
            Inkling2Ops.RmsNorm(cur.AsSpan(t * H, H), default, 1e-6f, wantIn.AsSpan(t * 2 * H, H));
            Inkling2Ops.RmsNorm(m, default, 1e-6f, wantIn.AsSpan(t * 2 * H + H, H));
            Inkling2Ops.ControlCombine(cur.AsSpan(t * H, H), m, gain.AsSpan(t * H, H), forget.AsSpan(t * H, H), o);
            Inkling2Ops.RmsNorm(o, gamma, 1e-6f, wantOut.AsSpan(t * H, H));
        }
        var bc = Up(cur); var bm = Up(msg);
        var yin = Zero(wantIn.Length * 4L);
        Dispatch(K("inkling2_ctrl_in_f32act", 3, 4), [(uint)H, 0, (uint)msgOff, F(1e-6f)], (uint)T, 1, 1, bc, bm, yin);
        Check("ctrl_in", wantIn, Down(yin, wantIn.Length), 1e-5);
        var yo = Zero(wantOut.Length * 4L);
        Dispatch(K("inkling2_ctrl_out_f32act", 6, 4), [(uint)H, 0, (uint)msgOff, F(1e-6f)], (uint)T, 1, 1, bc, bm, Up(gain), Up(forget), Up(gamma), yo);
        Check("ctrl_out", wantOut, Down(yo, wantOut.Length), 1e-5);
    }

    [SkippableTheory]
    [InlineData(true, true)]
    [InlineData(false, false)]
    public void Conv_CarriedState_MatchesReference(bool dyn, bool add)
    {
        int C = 192, heads = 3, T = 5;
        var w = Rand(C * 4);
        var cpu = new Inkling2ConvState(C, 4, Inkling2ConvStateStorage.BF16);
        var state = Zero(C * 3 * 2L);
        for (int call = 0; call < 2; call++)   // the second call reads the state the first one wrote
        {
            var x = Rand(T * C, 2f); var d = Rand(T * heads * 4); var a = Rand(T * C);
            var want = new float[T * C]; var scratch = new float[cpu.ScratchFloats(T)];
            cpu.Run(T, x, w, dyn ? d : default, heads, want, scratch);
            if (add) for (int i = 0; i < want.Length; i++) want[i] += a[i];
            var y = Zero(want.Length * 4L);
            Dispatch(K("inkling2_conv_f32act", 6, 5, dyn ? 1u : 0u, add ? 1u : 0u), [(uint)C, (uint)T, (uint)heads, 0, 0], (uint)((C + 63) / 64), 1, 1, Up(x), Up(w), Up(d), state, Up(a), y);
            Check($"conv call {call}", want, Down(y, want.Length), 1e-5);
        }
        cpu.Dispose();
    }

    private void ReferenceAttention(int T, int pos0, int nh, int nkv, bool local, int window, int extent, float[] q, float[] rel, float[] sink, float[] gate, float[][] kAll, float[][] vAll, float[] relBias, float[] want)
    {
        int group = nh / nkv;
        for (int t = 0; t < T; t++)
        {
            int p = pos0 + t, lo = local ? Math.Max(0, p - window + 1) : 0;
            for (int h = 0; h < nh; h++)
            {
                int g = h / group;
                var l = new double[p - lo + 1];
                for (int j = lo; j <= p; j++)
                {
                    double dot = 0; for (int e = 0; e < 128; e++) dot += q[t * nh * 128 + h * 128 + e] * kAll[j][g * 128 + e];
                    int d = p - j; double bias = 0;
                    if (d < extent) for (int e = 0; e < 16; e++) bias += relBias[d * 16 + e] * rel[t * nh * 16 + h * 16 + e];
                    l[j - lo] = dot / Math.Sqrt(128) + bias - sink[t * nh + h];
                }
                double M = Math.Max(0, l.Max()), S = Math.Exp(-M);
                foreach (var v in l) S += Math.Exp(v - M);
                for (int e = 0; e < 128; e++)
                {
                    double o = 0; for (int j = lo; j <= p; j++) o += Math.Exp(l[j - lo] - M) / S * vAll[j][g * 128 + e];
                    double gt = gate[t * nh * 128 + h * 128 + e];
                    want[t * nh * 128 + h * 128 + e] = (float)(o / (1 + Math.Exp(-gt)));
                }
            }
        }
    }

    /// <summary>
    /// Attention with a ring (capacity window + chunk) wrapped past its end, T larger than the window, GQA with 2 kv heads (3 q heads each: a wrong head mapping shows),
    /// relative extent smaller than the key range (the bias-clamp), and the zero sink.
    /// </summary>
    [SkippableTheory]
    [InlineData(true, 0)]
    [InlineData(false, 0)]
    [InlineData(true, 1)]
    public void Attention_RingWindowExtentSink_MatchesReference(bool local, int kvq8)
    {
        int nh = 6, nkv = 2, window = 8, extent = 8, T = 10, pos0 = 14, ring = window + T;
        int total = pos0 + T;
        var kAll = new float[total][]; var vAll = new float[total][];
        for (int j = 0; j < total; j++) { kAll[j] = Rand(nkv * 128, 2f); vAll[j] = Rand(nkv * 128); }
        // Cache contents as the GPU sees them: F16 or q8_0, in the position-major layout, ring-slotted for local layers.
        int cap = local ? ring : total;
        long rowBytes = kvq8 == 0 ? 256 : 136;
        var kb = new byte[cap * nkv * rowBytes]; var vb = new byte[cap * nkv * rowBytes];
        var kDec = new float[total][]; var vDec = new float[total][];
        for (int j = 0; j < total; j++)
        {
            kDec[j] = new float[nkv * 128]; vDec[j] = new float[nkv * 128];
            if (local && j < pos0 - ring + 1 + 0 && j < pos0 - window) { }
            int slot = local ? j % ring : j;
            for (int g = 0; g < nkv; g++)
            {
                StoreRow(kb, (slot * nkv + g) * rowBytes, kAll[j].AsSpan(g * 128, 128), kvq8, kDec[j].AsSpan(g * 128, 128));
                StoreRow(vb, (slot * nkv + g) * rowBytes, vAll[j].AsSpan(g * 128, 128), kvq8, vDec[j].AsSpan(g * 128, 128));
            }
        }
        var q = Rand(T * nh * 128); var rel = Rand(T * nh * 16); var sink = Rand(T * nh, 2f); var gate = Rand(T * nh * 128, 2f); var relBias = Rand(extent * 16);
        var want = new float[T * nh * 128];
        ReferenceAttention(T, pos0, nh, nkv, local, window, extent, q, rel, sink, gate, kDec, vDec, relBias, want);
        var y = Zero(want.Length * 4L);
        Dispatch(K("inkling2_attn_f32act", 8, 8, (uint)kvq8), [(uint)T, (uint)pos0, (uint)nh, (uint)nkv, local ? 1u : 0u, (uint)window, local ? (uint)ring : 0u, (uint)extent],
                 (uint)nkv, (uint)T, 1, Up(q), Up(rel), Up(sink), Up(gate), UpBytes(kb), UpBytes(vb), Up(relBias), y);
        Check($"attention local={local} kvq8={kvq8}", want, Down(y, want.Length), 2e-4);
    }

    private static void StoreRow(byte[] buf, long at, ReadOnlySpan<float> x, int kvq8, Span<float> decoded)
    {
        if (kvq8 == 0)
        {
            for (int i = 0; i < 128; i++) { var h = (Half)x[i]; BitConverter.TryWriteBytes(buf.AsSpan((int)at + 2 * i), h); decoded[i] = (float)h; }
            return;
        }
        for (int b = 0; b < 4; b++)
        {
            float amax = 0; for (int i = 0; i < 32; i++) amax = Math.Max(amax, Math.Abs(x[b * 32 + i]));
            float d = amax / 127f; var dh = (Half)d; float id = d != 0 ? 1f / d : 0f; long bo = at + b * 34;
            BitConverter.TryWriteBytes(buf.AsSpan((int)bo), dh);
            for (int i = 0; i < 32; i++)
            {
                int qv = (int)MathF.Round(x[b * 32 + i] * id, MidpointRounding.AwayFromZero);
                buf[bo + 2 + i] = (byte)(sbyte)qv; decoded[b * 32 + i] = (float)dh * qv;
            }
        }
    }

    [SkippableTheory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(true, 1)]
    public void QkPrep_ClipNormTauAndCacheStore_MatchReference(bool global, int kvq8)
    {
        int nh = 6, nkv = 2, T = 4, pos0 = 9, ring = global ? 0 : 12;
        var q = Rand(T * nh * 128, 5f); var rel = Rand(T * nh * 16, 4f); var kin = Rand(T * nkv * 128, 6f); var vin = Rand(T * nkv * 128, 3f);
        var qclip = Rand(nh * 128); var relclip = Rand(nh * 16);
        float qk = 22.627417f, rc = 8f, eps = 1e-6f; int floorN = 4; float alpha = 0.1f;
        var wq = (float[])q.Clone(); var wr = (float[])rel.Clone();
        long rowBytes = kvq8 == 0 ? 256 : 136; int cap = global ? pos0 + T : ring;
        var wantK = new float[T][]; var wantV = new float[T][];
        var kb = new byte[cap * nkv * rowBytes]; var vb = new byte[cap * nkv * rowBytes];
        for (int t = 0; t < T; t++)
        {
            float tau = Inkling2Ops.LogScale(pos0 + t, floorN, alpha);
            for (int h = 0; h < nh; h++)
            {
                Inkling2Ops.ClipL2(wq.AsSpan(t * nh * 128 + h * 128, 128), qclip.AsSpan(h * 128, 128), qk);
                Inkling2Ops.ClipL2(wr.AsSpan((t * nh + h) * 16, 16), relclip.AsSpan(h * 16, 16), rc);
            }
            if (global) { for (int i = 0; i < nh * 128; i++) wq[t * nh * 128 + i] *= tau; for (int i = 0; i < nh * 16; i++) wr[t * nh * 16 + i] *= tau; }
            wantK[t] = new float[nkv * 128]; wantV[t] = new float[nkv * 128];
            int slot = global ? pos0 + t : (pos0 + t) % ring;
            for (int g = 0; g < nkv; g++)
            {
                var k = kin.AsSpan(t * nkv * 128 + g * 128, 128).ToArray(); Inkling2Ops.ClipL2(k, default, qk);
                var v = vin.AsSpan(t * nkv * 128 + g * 128, 128).ToArray(); Inkling2Ops.RmsNorm(v, default, eps, v);
                StoreRow(kb, (slot * nkv + g) * rowBytes, k, kvq8, wantK[t].AsSpan(g * 128, 128));
                StoreRow(vb, (slot * nkv + g) * rowBytes, v, kvq8, wantV[t].AsSpan(g * 128, 128));
            }
        }
        var bq = Up(q); var br = Up(rel);
        var kc = Zero(kb.Length); var vc = Zero(vb.Length);
        Dispatch(K("inkling2_qkprep_f32act", 8, 12, (uint)kvq8), [(uint)T, (uint)pos0, (uint)ring, global ? 1u : 0u, F(qk), F(rc), F(eps), F(floorN), F(alpha), (uint)nh, (uint)nkv, 16],
                 (uint)(nh + 2 * nkv + nh), (uint)T, 1, bq, br, Up(kin), Up(vin), Up(qclip), Up(relclip), kc, vc);
        Check("q", wq, Down(bq, wq.Length), 1e-5);
        Check("rel", wr, Down(br, wr.Length), 1e-5);
        var gotK = new byte[kb.Length]; var gotV = new byte[vb.Length];
        DownBytes(kc, gotK); DownBytes(vc, gotV);
        // F16 rows: compare decoded halves (summation-order differences flip the last mantissa bit of ~8% of the values: a half ulp is 2^-10 relative).
        // q8_0 rows: a rounding tie may move one int8 by 1; a wrong block scale, slot or sign differs in far more than 1% of the bytes.
        int bad = 0, total = kb.Length + vb.Length;
        if (kvq8 == 0)
        {
            foreach (var (want, got) in new[] { (kb, gotK), (vb, gotV) })
                for (int i = 0; i < want.Length; i += 2)
                {
                    float a = (float)BitConverter.ToHalf(want, i), b = (float)BitConverter.ToHalf(got, i);
                    if (Math.Abs(a - b) > 2e-3 * Math.Abs(a) + 1e-6) bad++;
                }
            total /= 2;
        }
        else
        {
            for (int i = 0; i < kb.Length; i++) if (kb[i] != gotK[i]) bad++;
            for (int i = 0; i < vb.Length; i++) if (vb[i] != gotV[i]) bad++;
        }
        _output.WriteLine($"cache entries off: {bad} of {total}");
        Assert.True(bad <= total / 100, $"{bad} of {total} cache entries differ");
    }

    private void DownBytes(VulkanDevice.Buffer b, byte[] dst)
    {
        var f = new float[(dst.Length + 3) / 4]; Dev.Download(b, f);
        MemoryMarshal.AsBytes(f.AsSpan()).Slice(0, dst.Length).CopyTo(dst);
    }

    private (byte[] Bank, float[] Dec) MakeBank(QuantizationType qt, int rows, int K)
    {
        long rb = Dequantize.RowByteSize(K, qt);
        var bytes = new byte[rb * rows];
        if (qt == QuantizationType.Q8_0)
        {
            var w = Rand(rows * K, 1.5f);
            Quantize.FromFloat32(w, rows * K, qt, bytes);
        }
        else
        {
            _rnd.NextBytes(bytes);
            for (long b = 0; b < bytes.Length / 36; b++) for (int s = 0; s < 4; s++) bytes[b * 36 + s] = (byte)_rnd.Next(0x28, 0x48);   // sane UE4M3 scales
        }
        var dec = new float[rows * K];
        fixed (byte* p = bytes) for (int r = 0; r < rows; r++) Dequantize.ToFloat32((nint)p + (nint)(r * rb), K, qt, dec.AsSpan(r * K, K));
        return (bytes, dec);
    }

    [SkippableTheory]
    [InlineData(QuantizationType.NVFP4)]
    [InlineData(QuantizationType.Q8_0)]
    public void MoeGateUpAndDown_MatchReference(QuantizationType qt)
    {
        int E = 5, H = 192, ff = 64, topk = 2, T = 3;
        var (gu, guDec) = MakeBank(qt, E * 2 * ff, H);
        var (dn, dnDec) = MakeBank(qt, E * H, ff);
        bool scaled = qt == QuantizationType.NVFP4;
        var gus = Rand(E, 0.5f).Select(v => 0.7f + v).ToArray(); var dns = Rand(E, 0.5f).Select(v => 0.7f + v).ToArray();
        var x = Rand(T * H, 2f);
        uint[] sel = [3, 0, 1, 4, 4, 2]; var selw = Rand(T * topk).Select(v => v + 1.2f).ToArray();
        float beta = 4f, lb = 25f;
        var h = new float[T * topk * ff]; var want = new float[T * H];
        for (int p = 0; p < T * topk; p++)
        {
            int e = (int)sel[p], t = p / topk;
            for (int j = 0; j < ff; j++)
            {
                double g = 0, u = 0;
                for (int k = 0; k < H; k++) { g += (double)guDec[(e * 2 * ff + j) * H + k] * x[t * H + k]; u += (double)guDec[(e * 2 * ff + ff + j) * H + k] * x[t * H + k]; }
                if (scaled) { g *= gus[e]; u *= gus[e]; }
                h[p * ff + j] = Inkling2Ops.Situ((float)g, (float)u, beta, lb);
            }
        }
        for (int t = 0; t < T; t++)
            for (int m = 0; m < H; m++)
            {
                double tot = 0;
                for (int s = 0; s < topk; s++)
                {
                    int p = t * topk + s, e = (int)sel[p]; double d = 0;
                    for (int k = 0; k < ff; k++) d += (double)dnDec[(e * H + m) * ff + k] * h[p * ff + k];
                    if (scaled) d *= dns[e];
                    tot += selw[p] * d;
                }
                want[t * H + m] = (float)tot;
            }
        string n = qt == QuantizationType.NVFP4 ? "nvfp4" : "q8_0";
        var hb = Zero(h.Length * 4L);
        var bsel = UpU(sel);
        Dispatch(K($"inkling2_moe_gate_up_{n}_f32act", 5, 5, scaled ? 1u : 0u), [(uint)ff, (uint)H, (uint)topk, F(beta), F(lb)], (uint)ff, (uint)(T * topk), 1, UpBytes(gu), Up(x), hb, bsel, Up(gus));
        Check($"gate_up+SITU {n}", h, Down(hb, h.Length), 2e-4);
        var y = Zero(want.Length * 4L);
        Dispatch(K($"inkling2_moe_down_{n}_f32act", 6, 3, scaled ? 1u : 0u), [(uint)H, (uint)ff, (uint)topk], (uint)H, (uint)T, 1, UpBytes(dn), Up(h), y, bsel, Up(selw), Up(dns));
        Check($"down {n}", want, Down(y, want.Length), 2e-4);
    }

    [SkippableFact]
    public void Softcap_MatchesReference()
    {
        var x = Rand(1000, 90f);
        var want = x.Select(v => (float)(30.0 * Math.Tanh(v / 30.0))).ToArray();
        var b = Up(x);
        Dispatch(K("inkling2_softcap", 1, 2), [1000u, F(30f)], 4, 1, 1, b);
        Check("softcap", want, Down(b, 1000), 1e-5);
    }

    /// <summary>
    /// Image-row injection (#2062): selected tokens of a pre-filled activation buffer are overwritten with the F16 rows (rounded once to the activation type),
    /// unselected tokens are untouched. Discriminates the compact-row addressing (sel -> row index, not the token index), the F16 unpacking order (low half first)
    /// and the sentinel.
    /// </summary>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void EmbedRows_OverwritesSelectedTokens(bool f32act)
    {
        int T = 5, H = 300, nRows = 2;   // H even; rows 2 -> tokens 3 and 1 (row order differs from token order)
        var rows = new Half[nRows * H];
        for (int i = 0; i < rows.Length; i++) rows[i] = (Half)(float)(_rnd.NextDouble() * 20 - 10);
        uint none = 0xFFFFFFFFu;
        uint[] sel = [none, 1u, none, 0u, none];
        var prefill = Rand(T * H, 3f);
        var wantF = new float[T * H];
        for (int t = 0; t < T; t++)
            for (int k = 0; k < H; k++)
                wantF[t * H + k] = sel[t] == none ? prefill[t * H + k] : (float)rows[(int)sel[t] * H + k];
        var rowBytes = MemoryMarshal.AsBytes(rows.AsSpan()).ToArray();
        string name = f32act ? "inkling2_embed_rows_f32act" : "inkling2_embed_rows";
        if (f32act)
        {
            var y = Up(prefill);
            Dispatch(K(name, 3, 1), [(uint)H], (uint)T, 1, 1, UpBytes(rowBytes), UpU(sel), y);
            Check("embed_rows f32", wantF, Down(y, T * H), 1e-7);
        }
        else
        {
            static ushort Bf(float f) { uint b = BitConverter.SingleToUInt32Bits(f); b += 0x7fffu + ((b >> 16) & 1u); return (ushort)(b >> 16); }
            var init = prefill.Select(Bf).ToArray();
            var y = UpBytes(MemoryMarshal.AsBytes(init.AsSpan()).ToArray());
            Dispatch(K(name, 3, 1), [(uint)H], (uint)T, 1, 1, UpBytes(rowBytes), UpU(sel), y);
            var raw = new float[(T * H + 1) / 2];
            Dev.Download(y, raw);
            var got = MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
            int bad = 0;
            for (int i = 0; i < T * H; i++) if (got[i] != Bf(wantF[i])) bad++;
            _output.WriteLine($"embed_rows bf16: {bad} of {T * H} elements differ from the RNE-rounded reference");
            Assert.Equal(0, bad);
        }
    }
}
