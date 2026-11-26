using DotLLM.Models.Edge0;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// Edge0 latent-MoE CPU forward (#1967, G2) against an independent float64 oracle on a structurally faithful synthetic pack.
/// The tolerance (relative L2) covers activation quantization (Q8_K / Q8_1 / Q8_0 inputs of the integer dot kernels); the group
/// mapping and column-scale mutants move the output by O(1), so they cannot hide inside it (see the PR body for the mutant runs).
/// </summary>
public sealed unsafe class Edge0LatentMoeTests
{
    private const int H = SyntheticEdge0Pack.Hidden, K = SyntheticEdge0Pack.TopK;

    private static float[] Random(int n, int seed, float scale)
    {
        var r = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1) * scale;
        return a;
    }

    private static double RelL2(float[] got, float[] want)
    {
        double num = 0, den = 0;
        for (int i = 0; i < got.Length; i++) { double d = got[i] - want[i]; num += d * d; den += (double)want[i] * want[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static (float[] Router, float[] X) Inputs(int tokens)
        => (Random(SyntheticEdge0Pack.Experts * H, 11, 0.6f), Random(tokens * H, 12, 1f));

    [Theory]
    [InlineData(false, true, 1)]
    [InlineData(true, true, 1)]
    [InlineData(false, false, 1)]
    [InlineData(false, true, 4)]   // pooled: Parallel.For over experts, pooled D/U GEMMs
    public void Forward_MatchesIndependentOracle(bool q4Latent, bool withScales, int threads)
    {
        using var pool = threads > 1 ? new DotLLM.Cpu.Threading.ComputeThreadPool(threads) : null;
        using var syn = new SyntheticEdge0Pack(q4Latent: q4Latent, withScales: withScales);
        using var pack = syn.Open();
        const int T = 5;
        var (router, x) = Inputs(T);
        for (int layer = 0; layer < pack.Layers; layer++)
        {
            var y = new float[T * H];
            Array.Fill(y, float.NaN);   // y is "fully overwritten": a forward that accumulates into it instead must fail here
            Edge0LatentMoe.Forward(pack, layer, router, K, normTopKProb: true, shared: null, x, y, T, pool);
            for (int t = 0; t < T; t++)
            {
                var want = Edge0LatentMoeReference.Routed(pack, layer, router, K, x[(t * H)..((t + 1) * H)],
                    syn.Labels[(layer * pack.Experts)..((layer + 1) * pack.Experts)], withScales ? syn.Scales[layer] : null, out _, out _);
                double err = RelL2(y[(t * H)..((t + 1) * H)], want);
                // Activation quantization to 8-bit blocks costs about 1e-2; a wrong group or scale costs O(1).
                Assert.True(err < 0.05, $"layer {layer} token {t} (q4Latent={q4Latent}, scales={withScales}): rel L2 {err:E3}");
            }
        }
    }

    /// <summary>Round-to-nearest-even F32 to BF16 bits, and the exact widening back (the router fixture is BF16-representable, so the oracle is exact).</summary>
    private static ushort ToBf16(float v)
    {
        uint b = BitConverter.SingleToUInt32Bits(v);
        return (ushort)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    }

    private static float FromBf16(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    /// <summary>A native BF16 copy of an F32 router (stands in for the mapped GGUF tensor) and its exact F32 widening for the oracle.</summary>
    private sealed class Bf16Router : IDisposable
    {
        public nint Ptr { get; }
        public float[] Widened { get; }
        public Bf16Router(float[] f32)
        {
            Widened = new float[f32.Length];
            Ptr = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(f32.Length * 2), 64);
            var dst = new Span<ushort>((void*)Ptr, f32.Length);
            for (int i = 0; i < f32.Length; i++) { dst[i] = ToBf16(f32[i]); Widened[i] = FromBf16(dst[i]); }
        }
        public void Dispose() => System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)Ptr);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(5, 1)]
    [InlineData(70, 4)]   // two chunks, pooled router logits (GemmBf16 with a pool)
    public void Bf16Router_ReadInPlace_MatchesOracleAndTheWidenedF32Path(int tokens, int threads)
    {
        using var pool = threads > 1 ? new DotLLM.Cpu.Threading.ComputeThreadPool(threads) : null;
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        var (routerF32, x) = Inputs(tokens);
        using var bf = new Bf16Router(routerF32);
        var viaBf16 = new float[tokens * H]; var viaF32 = new float[tokens * H];
        Array.Fill(viaBf16, float.NaN);
        Edge0LatentMoe.ForwardBf16Router(pack, 1, bf.Ptr, K, true, null, x, viaBf16, tokens, pool);
        Edge0LatentMoe.Forward(pack, 1, bf.Widened, K, true, null, x, viaF32, tokens, pool);
        // the oracle sees exactly the BF16 values (rounding of the fixture is not part of what is under test)
        for (int t = 0; t < Math.Min(tokens, 6); t++)
        {
            var want = Edge0LatentMoeReference.Routed(pack, 1, bf.Widened, K, x[(t * H)..((t + 1) * H)],
                syn.Labels[pack.Experts..(2 * pack.Experts)], syn.Scales[1], out _, out _);
            Assert.True(RelL2(viaBf16[(t * H)..((t + 1) * H)], want) < 0.05, $"token {t}: BF16 router output differs from the oracle");
        }
        // Same routing decisions, so the same output up to the GEMM tile order of the logits: far tighter than the oracle bound.
        Assert.True(RelL2(viaBf16, viaF32) < 1e-3, $"BF16-in-place vs widened F32 router: rel L2 {RelL2(viaBf16, viaF32):E3}");
    }

    [Fact]
    public void Bf16Router_SteadyState_AllocatesNothing()
    {
        // The in-place read means no widened [experts][hidden] copy per call (here 16 x 64 x 4 B; on the real model 5.2 MB per layer).
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        var (routerF32, x) = Inputs(3);
        using var bf = new Bf16Router(routerF32);
        var y = new float[3 * H];
        for (int i = 0; i < 3; i++) Edge0LatentMoe.ForwardBf16Router(pack, 0, bf.Ptr, K, true, null, x, y, 3);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 5; i++) Edge0LatentMoe.ForwardBf16Router(pack, 0, bf.Ptr, K, true, null, x, y, 3);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void Forward_IsChunkAndBatchInvariant()
    {
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        const int T = 70;   // crosses the 64-token chunk boundary
        var (router, x) = Inputs(T);
        var all = new float[T * H];
        Edge0LatentMoe.Forward(pack, 1, router, K, true, null, x, all, T);
        for (int t = 0; t < T; t += 17)
        {
            var one = new float[H];
            Edge0LatentMoe.Forward(pack, 1, router, K, true, null, x.AsSpan(t * H, H), one, 1);
            Assert.True(RelL2(all[(t * H)..((t + 1) * H)], one) < 1e-5, $"token {t} differs between batch and single-token runs");
        }
    }

    [Fact]
    public void Routing_UsesMoreThanOneGroup_SoGroupMappingIsObservable()
    {
        // A fixture where every selected expert sat in one group could not tell a right label->group map from a wrong one.
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        var (router, x) = Inputs(8);
        var groups = new HashSet<int>();
        for (int t = 0; t < 8; t++)
        {
            Edge0LatentMoeReference.Routed(pack, 0, router, K, x[(t * H)..((t + 1) * H)],
                syn.Labels[..pack.Experts], syn.Scales[0], out var ids, out _);
            foreach (int e in ids) groups.Add(syn.Labels[e]);
        }
        Assert.True(groups.Count >= 2, $"only groups {{{string.Join(",", groups)}}} were routed");
    }

    [Fact]
    public void ColumnScale_IsObservable()
    {
        // With scales present, the oracle with and without the 1/s factor must differ materially: otherwise the colscale mutant could survive.
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        var (router, x) = Inputs(1);
        var lab = syn.Labels[..pack.Experts];
        var with = Edge0LatentMoeReference.Routed(pack, 0, router, K, x, lab, syn.Scales[0], out _, out _);
        var without = Edge0LatentMoeReference.Routed(pack, 0, router, K, x, lab, null, out _, out _);
        Assert.True(RelL2(without, with) > 0.2);
        // and the production path agrees with the scaled oracle, not the unscaled one
        var y = new float[H];
        Edge0LatentMoe.Forward(pack, 0, router, K, true, null, x, y, 1);
        Assert.True(RelL2(y, with) < RelL2(y, without) / 4);
    }

    [Fact]
    public void SharedExpert_IsGatedAndAdded()
    {
        using var syn = new SyntheticEdge0Pack();
        using var pack = syn.Open();
        const int I = 32;
        // Q8_0 shared expert built by hand: gate/up [I rows x 64], down [64 rows x I]
        static unsafe nint Q8Block(int rows, int cols, int seed, List<nint> owned)
        {
            var r = new Random(seed);
            long bytes = rows * (cols / 32) * 34L;
            nint p = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 64);
            owned.Add(p);
            var span = new Span<byte>((void*)p, (int)bytes);
            for (int b = 0; b < bytes / 34; b++)
            {
                System.Buffers.Binary.BinaryPrimitives.WriteHalfLittleEndian(span.Slice(b * 34, 2), (Half)0.02f);
                for (int j = 0; j < 32; j++) span[b * 34 + 2 + j] = (byte)(sbyte)r.Next(-30, 31);
            }
            return p;
        }
        var owned = new List<nint>();
        try
        {
            var sh = new Edge0LatentMoe.SharedExpert(Q8Block(I, H, 1, owned), Q8Block(I, H, 2, owned), Q8Block(H, I, 3, owned),
                DotLLM.Core.Configuration.QuantizationType.Q8_0, DotLLM.Core.Configuration.QuantizationType.Q8_0, I, Random(H, 4, 0.5f));
            var (router, x) = Inputs(1);
            var routed = new float[H]; var both = new float[H];
            Edge0LatentMoe.Forward(pack, 0, router, K, true, null, x, routed, 1);
            Edge0LatentMoe.Forward(pack, 0, router, K, true, sh, x, both, 1);

            // oracle for the shared branch
            var g = Edge0LatentMoeReference.Dequant(sh.Gate, I * H, sh.GateUpType);
            var u = Edge0LatentMoeReference.Dequant(sh.Up, I * H, sh.GateUpType);
            var d = Edge0LatentMoeReference.Dequant(sh.Down, H * I, sh.DownType);
            var hid = new double[I];
            for (int r = 0; r < I; r++)
            {
                double a = 0, b = 0;
                for (int i = 0; i < H; i++) { a += (double)g[r * H + i] * x[i]; b += (double)u[r * H + i] * x[i]; }
                hid[r] = a / (1 + Math.Exp(-a)) * b;
            }
            double logit = 0;
            for (int i = 0; i < H; i++) logit += (double)sh.InputGate![i] * x[i];
            double gate = 1 / (1 + Math.Exp(-logit));
            var want = new float[H];
            for (int r = 0; r < H; r++)
            {
                double s = 0;
                for (int i = 0; i < I; i++) s += (double)d[r * I + i] * hid[i];
                want[r] = routed[r] + (float)(gate * s);
            }
            // Compare the shared contribution itself (both - routed vs want - routed), so the large routed term cannot mask it.
            var gotSh = new float[H]; var wantSh = new float[H];
            for (int r = 0; r < H; r++) { gotSh[r] = both[r] - routed[r]; wantSh[r] = want[r] - routed[r]; }
            double mag = Math.Sqrt(wantSh.Sum(v => (double)v * v));
            Assert.True(mag > 1e-3, $"the shared branch must contribute (|shared| = {mag:E2})");
            Assert.True(RelL2(gotSh, wantSh) < 0.03, $"shared expert: rel L2 {RelL2(gotSh, wantSh):E3}");
        }
        finally { foreach (nint p in owned) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)p); }
    }
}
