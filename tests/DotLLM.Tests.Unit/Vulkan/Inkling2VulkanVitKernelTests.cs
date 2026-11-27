using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Op-level parity of the Vulkan ViT kernels (#2062 part B) against double-precision references written independently of the shaders. Mutated shader copies are loaded through
/// <c>DOTLLM_INKLING2_SPV_OVERRIDE</c> by the mutation script (the tracked .spv are never edited), see docs/INKLING2.md section 12b.
/// </summary>
public sealed unsafe class Inkling2VulkanVitKernelTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly VulkanDevice? _dev;
    private readonly string _spv = Inkling2VulkanTests.SpvDir();
    private readonly string? _ovr = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");
    private readonly Random _rnd = new(4321);

    public Inkling2VulkanVitKernelTests(ITestOutputHelper output)
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

    private VulkanDevice.Buffer UpBytes(byte[] a) { var b = Dev.AllocateDeviceLocal(Math.Max(256, (a.Length + 3L) / 4 * 4)); Dev.Upload(a.Length % 4 == 0 ? a : a.Concat(new byte[4 - a.Length % 4]).ToArray(), b); return b; }
    private VulkanDevice.Buffer UpF32(float[] a) => UpBytes(MemoryMarshal.AsBytes(a.AsSpan()).ToArray());
    private VulkanDevice.Buffer UpU16(ushort[] a) => UpBytes(MemoryMarshal.AsBytes(a.AsSpan()).ToArray());
    private VulkanDevice.Buffer Zero(long bytes) { var b = Dev.AllocateDeviceLocal(Math.Max(256, bytes)); Dev.FillBufferSynchronous(b, 0); return b; }
    private VulkanDevice.Buffer Dummy() => Zero(256);

    private ushort[] DownU16(VulkanDevice.Buffer b, int n)
    {
        var raw = new float[(n + 1) / 2];
        Dev.Download(b, raw);
        return MemoryMarshal.Cast<float, ushort>(raw.AsSpan()).Slice(0, n).ToArray();
    }

    private static ushort Bf(float f) { uint b = BitConverter.SingleToUInt32Bits(f); b += 0x7fffu + ((b >> 16) & 1u); return (ushort)(b >> 16); }
    private static float FromBf(ushort u) => BitConverter.UInt32BitsToSingle((uint)u << 16);
    private static float RoundBf(float f) => FromBf(Bf(f));
    private static float[] DownF(ushort[] u) => u.Select(FromBf).ToArray();
    private static uint F(float f) => BitConverter.SingleToUInt32Bits(f);

    private float[] Rand(int n, float s = 1f) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(_rnd.NextDouble() * 2 - 1) * s; return a; }
    private float[] RandBf(int n, float s = 1f) => Rand(n, s).Select(RoundBf).ToArray();
    private ushort[] ToBf(float[] a) => a.Select(Bf).ToArray();

    private void Dispatch(OlalaKernel k, ReadOnlySpan<uint> push, uint gx, uint gy, uint gz, params VulkanDevice.Buffer[] bufs)
    {
        using var sub = Dev.CreateSubmitContext();
        sub.Begin();
        KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
        k.Record(sub.CommandBuffer, bufs, push, gx, gy, gz);
        sub.SubmitAndWait();
        k.Dispose();
    }

    /// <summary>Worst |got - want| / (1 + |want|); BF16 outputs carry 2^-9 relative rounding, so the bound is a few of those.</summary>
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

    private static double Erf(double x)
    {
        // high-precision erf (Numerical Recipes erfc Chebyshev, ~1.2e-7 -> refined by one series/continued step is unnecessary: the reference only needs 1e-7)
        double z = Math.Abs(x), t = 1.0 / (1.0 + 0.5 * z);
        double r = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - r : r - 1;
    }
    private static double Gelu(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    // ───────────── LayerNorm ─────────────

    [SkippableFact]
    public void LayerNorm_MatchesReference()
    {
        int rows = 5; const int N = 1024;
        var x = RandBf(rows * N, 4f); for (int i = 0; i < x.Length; i++) x[i] += 1.5f;   // a mean away from zero
        x = x.Select(RoundBf).ToArray();
        var g = Rand(N); var b = Rand(N); float eps = 1e-5f;
        var want = new float[rows * N];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0; for (int i = 0; i < N; i++) mean += x[r * N + i]; mean /= N;
            double var = 0; for (int i = 0; i < N; i++) { double d = x[r * N + i] - mean; var += d * d; } var /= N;
            for (int i = 0; i < N; i++) want[r * N + i] = (float)((x[r * N + i] - mean) / Math.Sqrt(var + eps) * g[i] + b[i]);
        }
        var y = Zero(rows * N * 2L);
        Dispatch(K("vit_layernorm", 4, 2), [(uint)rows, F(eps)], (uint)rows, 1, 1, UpU16(ToBf(x)), UpF32(g), UpF32(b), y);
        Check("layernorm", want, DownF(DownU16(y, rows * N)), 1.5e-2);
    }

    // ───────────── Q8_0 GEMM and epilogues ─────────────

    private byte[] Quant(float[] w, int M, int Kd, out float[] deq)
    {
        var q = new byte[(int)Dequantize.RowByteSize(Kd, QuantizationType.Q8_0) * M];
        Quantize.FromFloat32(w, M * Kd, QuantizationType.Q8_0, q);
        deq = new float[M * Kd]; var row = new float[Kd];
        long rb = Dequantize.RowByteSize(Kd, QuantizationType.Q8_0);
        fixed (byte* p = q)
            for (int m = 0; m < M; m++) { Dequantize.ToFloat32((nint)p + (nint)(m * rb), Kd, QuantizationType.Q8_0, row); row.CopyTo(deq, m * Kd); }
        return q;
    }

    private static ushort Half(float f) => BitConverter.HalfToUInt16Bits((Half)f);
    private static float FromHalf(ushort u) => (float)BitConverter.UInt16BitsToHalf(u);

    /// <summary>F16 RoPE table of random angles: cos_w sin_w cos_h sin_h per token (the rounded halves are what the reference rotates with).</summary>
    private ushort[] RopeTable(int T)
    {
        var tab = new ushort[T * 64];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < 16; i++)
            {
                double aw = _rnd.NextDouble() * 6.28, ah = _rnd.NextDouble() * 6.28;
                tab[t * 64 + i] = Half((float)Math.Cos(aw)); tab[t * 64 + 16 + i] = Half((float)Math.Sin(aw));
                tab[t * 64 + 32 + i] = Half((float)Math.Cos(ah)); tab[t * 64 + 48 + i] = Half((float)Math.Sin(ah));
            }
        return tab;
    }

    [SkippableTheory]
    [InlineData(0u)]   // bias
    [InlineData(1u)]   // bias + erf GELU
    [InlineData(2u)]   // Y += layerscale * (bias + acc)
    [InlineData(3u)]   // bias + 2D RoPE
    [InlineData(4u)]   // GELU, no bias
    [InlineData(5u)]   // F16 out
    public void GemmQ8_Epilogues_MatchReference(uint epi)
    {
        int M = 192, Kd = 96, T = 70;   // M multiple of 64; T has a tail tile; K = 3 Q8_0 blocks
        var w = Rand(M * Kd, 0.7f); var q = Quant(w, M, Kd, out var wd);
        var x = RandBf(T * Kd, 2f); var bias = Rand(M); var ls = Rand(M, 0.3f); var y0 = RandBf(T * M, 3f);
        var rope = RopeTable(T);
        var want = new float[T * M];
        for (int t = 0; t < T; t++)
        {
            var acc = new double[M];
            for (int m = 0; m < M; m++) { double s = 0; for (int k = 0; k < Kd; k++) s += (double)wd[m * Kd + k] * x[t * Kd + k]; acc[m] = s; }
            for (int m = 0; m < M; m++)
            {
                double v = acc[m] + (epi <= 3 ? bias[m] : 0);
                if (epi == 1 || epi == 4) v = Gelu(v);
                want[t * M + m] = (float)v;
            }
            if (epi == 3)
                for (int hh = 0; hh < M / 64; hh++)
                    for (int half = 0; half < 2; half++)
                        for (int i = 0; i < 16; i++)
                        {
                            int a = t * M + hh * 64 + half * 32 + 2 * i, b2 = a + 1;
                            double c = FromHalf(rope[t * 64 + half * 32 + i]), s = FromHalf(rope[t * 64 + half * 32 + 16 + i]);
                            double x0 = want[a], x1 = want[b2];
                            want[a] = (float)(x0 * c - x1 * s); want[b2] = (float)(x0 * s + x1 * c);
                        }
            if (epi == 2) for (int m = 0; m < M; m++) want[t * M + m] = (float)(y0[t * M + m] + ls[m] * want[t * M + m]);
        }
        var yBuf = epi == 2 ? UpU16(ToBf(y0)) : Zero(T * M * 2L);
        var hBuf = Zero(T * M * 2L);
        Dispatch(K("vit_gemm_q8", 7, 7, epi, 0u), [(uint)M, (uint)Kd, (uint)T, (uint)Kd, 0, 0, 0], (uint)(M / 64), (uint)((T + 63) / 64), 1,
            UpBytes(q), UpU16(ToBf(x)), epi == 5 ? Dummy() : yBuf, UpF32(bias), UpF32(ls), UpU16(rope), epi == 5 ? hBuf : Dummy());
        var got = epi == 5 ? DownU16(hBuf, T * M).Select(FromHalf).ToArray() : DownF(DownU16(yBuf, T * M));
        Check($"gemm q8 epi {epi}", want, got, epi == 5 ? 2e-3 : 1.2e-2);
    }

    /// <summary>The 2x2 merge as the mm.1 GEMM's activation addressing: token (j, i) reads [(2i,2j), (2i+1,2j), (2i,2j+1), (2i+1,2j+1)] of the patch rows (CLS row first).</summary>
    [SkippableFact]
    public void GemmQ8_Merge_MatchesReference()
    {
        int gw = 6, gh = 4, mw = gw / 2, mh = gh / 2, T = mw * mh, M = 64, Kd = 4096, E = 1024, posOff = 1;
        var w = Rand(M * Kd, 0.5f); var q = Quant(w, M, Kd, out var wd);
        var x = RandBf((1 + gw * gh) * E, 2f);
        var want = new float[T * M];
        for (int j = 0; j < mh; j++)
            for (int i = 0; i < mw; i++)
            {
                int[] patch = [(2 * j) * gw + 2 * i, (2 * j) * gw + 2 * i + 1, (2 * j + 1) * gw + 2 * i, (2 * j + 1) * gw + 2 * i + 1];
                for (int m = 0; m < M; m++)
                {
                    double s = 0;
                    for (int qd = 0; qd < 4; qd++)
                        for (int c = 0; c < E; c++) s += (double)wd[m * Kd + qd * E + c] * x[(patch[qd] + posOff) * E + c];
                    want[(j * mw + i) * M + m] = (float)Gelu(s);
                }
            }
        var y = Zero(T * M * 2L);
        Dispatch(K("vit_gemm_q8", 7, 7, 4u, 1u), [(uint)M, (uint)Kd, (uint)T, (uint)Kd, (uint)mw, (uint)gw, (uint)posOff], 1, 1, 1,
            UpBytes(q), UpU16(ToBf(x)), y, Dummy(), Dummy(), Dummy(), Dummy());
        Check("gemm q8 merge + gelu", want, DownF(DownU16(y, T * M)), 1.2e-2);
    }

    // ───────────── attention ─────────────

    private float[] RefAttention(float[] q, float[] k, float[] v, int npos)
    {
        var o = new float[npos * 1024];
        for (int h = 0; h < 16; h++)
            for (int i = 0; i < npos; i++)
            {
                var s = new double[npos]; double mx = double.NegativeInfinity;
                for (int j = 0; j < npos; j++) { double d = 0; for (int c = 0; c < 64; c++) d += (double)q[i * 1024 + h * 64 + c] * k[j * 1024 + h * 64 + c]; s[j] = d * 0.125; mx = Math.Max(mx, s[j]); }
                double sum = 0; for (int j = 0; j < npos; j++) { s[j] = Math.Exp(s[j] - mx); sum += s[j]; }
                for (int c = 0; c < 64; c++) { double a = 0; for (int j = 0; j < npos; j++) a += s[j] * v[j * 1024 + h * 64 + c]; o[i * 1024 + h * 64 + c] = (float)(a / sum); }
            }
        return o;
    }

    [SkippableTheory]
    [InlineData(53, 0)]     // one dispatch, a key tail of 11 and a query tail of 5
    [InlineData(150, 48)]   // the same sequence in two dispatches (qStart 48): the chunked path of a long image
    public void Attention_MatchesReference(int npos, int split)
    {
        var q = RandBf(npos * 1024, 3f); var k = RandBf(npos * 1024, 3f); var v = RandBf(npos * 1024, 2f);
        var want = RefAttention(q, k, v, npos);
        var o = Zero(npos * 1024 * 2L);
        var kern = K("vit_attn", 4, 2);
        var bq = UpU16(ToBf(q)); var bk = UpU16(ToBf(k)); var bv = UpU16(ToBf(v));
        if (split == 0) Dispatch(kern, [(uint)npos, 0u], (uint)((npos + 15) / 16), 16, 1, bq, bk, bv, o);
        else
        {
            using var sub = Dev.CreateSubmitContext();
            sub.Begin();
            KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
            kern.Record(sub.CommandBuffer, [bq, bk, bv, o], [(uint)npos, 0u], (uint)(split / 16), 16, 1);
            kern.Record(sub.CommandBuffer, [bq, bk, bv, o], [(uint)npos, (uint)split], (uint)((npos - split + 15) / 16), 16, 1);
            sub.SubmitAndWait();
            kern.Dispose();
        }
        Check($"attention npos {npos} split {split}", want, DownF(DownU16(o, npos * 1024)), 1.2e-2);
    }

    // ───────────── patch + position embedding ─────────────

    private static float PosRef(float[] table, int ns, int gw, int gh, int px, int py, int c)
    {
        // ggml bilinear upscale, half-pixel centres, clamped, no antialias (independent re-statement in double)
        double sf0 = (double)gw / ns, sf1 = (double)gh / ns;
        double fx = (px + 0.5) / sf0 - 0.5, fy = (py + 0.5) / sf1 - 0.5;
        int x0 = Math.Clamp((int)Math.Floor(fx), 0, ns - 1), x1 = Math.Clamp((int)Math.Floor(fx) + 1, 0, ns - 1);
        int y0 = Math.Clamp((int)Math.Floor(fy), 0, ns - 1), y1 = Math.Clamp((int)Math.Floor(fy) + 1, 0, ns - 1);
        double dx = Math.Clamp(fx - x0, 0, 1), dy = Math.Clamp(fy - y0, 0, 1);
        double T(int xx, int yy) => table[(yy * ns + xx) * 1024 + c];
        return (float)(T(x0, y0) * (1 - dx) * (1 - dy) + T(x1, y0) * dx * (1 - dy) + T(x0, y1) * (1 - dx) * dy + T(x1, y1) * dx * dy);
    }

    [SkippableTheory]
    [InlineData(6, 4, 2u)]     // interpolated grid, BF16 tables (the default)
    [InlineData(6, 4, 0u)]     // F32 arm
    [InlineData(6, 4, 1u)]     // F16 arm
    [InlineData(32, 32, 2u)]   // the table's own grid (identity)
    [InlineData(9, 14, 2u)]    // non-square, upscale in one axis and downscale... (9 < 32): both axes interpolated
    public void PatchEmbd_PositionInterpolation_MatchesReference(int gw, int gh, uint wt)
    {
        const int ns = 32, E = 1024, Kp = 1200;
        int n = gw * gh, W = gw * 20, H = gh * 20;
        var wRaw = Rand(E * Kp, 0.1f); var posRaw = Rand(ns * ns * E, 1f);
        float Conv(float f) => wt == 0 ? f : wt == 1 ? FromHalf(Half(f)) : RoundBf(f);
        var wDec = wRaw.Select(Conv).ToArray(); var posDec = posRaw.Select(Conv).ToArray();
        var pix = Rand(W * H * 3, 1f).Select(f => FromHalf(Half(f))).ToArray();
        var want = new float[(n + 1) * E];
        for (int p = 0; p < n; p++)
        {
            int px = p % gw, py = p / gw;
            var patch = new double[Kp];
            for (int k = 0; k < Kp; k++)
            {
                int c = k / 400, r = k % 400, ky = r / 20, kx = r % 20;
                patch[k] = pix[((py * 20 + ky) * W + px * 20 + kx) * 3 + c];
            }
            for (int c = 0; c < E; c++)
            {
                double s = 0; for (int k = 0; k < Kp; k++) s += wDec[c * Kp + k] * patch[k];
                want[(p + 1) * E + c] = (float)(s + PosRef(posDec, ns, gw, gh, px, py, c));
            }
        }
        VulkanDevice.Buffer Table(float[] raw) => wt == 0 ? UpF32(raw) : UpU16(raw.Select(f => wt == 1 ? Half(f) : Bf(f)).ToArray());
        var x = Zero((n + 1) * E * 2L);
        Dispatch(K("vit_patch_embd", 4, 6, wt), [(uint)gw, (uint)gh, (uint)W, 1u, (uint)ns, (uint)n], (uint)(E / 64), (uint)((n + 63) / 64), 1,
            UpU16(pix.Select(Half).ToArray()), Table(wRaw), Table(posRaw), x);
        var got = DownF(DownU16(x, (n + 1) * E));
        Check($"patch_embd+posemb {gw}x{gh} wt {wt}", want.Skip(E).ToArray(), got.Skip(E).ToArray(), 1.5e-2);
    }

    [SkippableFact]
    public void Cls_MatchesReference()
    {
        var c = Rand(1024); var p = Rand(1024);
        var want = c.Zip(p, (a, b) => a + b).ToArray();
        var x = Zero(2048);
        Dispatch(K("vit_cls", 3, 0), [], 4, 1, 1, UpF32(c), UpF32(p), x);
        Check("cls", want, DownF(DownU16(x, 1024)), 1e-2);
    }
}
