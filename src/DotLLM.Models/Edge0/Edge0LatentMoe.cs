using System.Buffers;
using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Edge0;

/// <summary>
/// CPU reference forward of one Edge0 latent-MoE layer (the MoE half of a block, shared expert included). Semantics are those of
/// edge0-strata <c>latent_cpu_verify.cpp</c> / the <c>andynoodles/edge0-vllm</c> converter oracle (<c>latent_moe_reference</c>):
/// <code>
/// p = softmax(router · x); top-k by p (ties to the lower index); w = p / sum(p of the top-k)
/// xl_g = (D_g · x) ⊙ (1/s)                       (g = D/U group of an expert; 1/s only for the Q3_K build)
/// h_j  = silu(gate_e · xl_g) ⊙ (up_e · xl_g)       (Q3_K rows, read in place)
/// z_g  = Σ_{j: g_j = g} w_j · down_e · h_j         (IQ4_NL rows, read in place)
/// out  = Σ_g U_g · z_g  +  sigmoid(w_sh · x) · down_sh(silu(gate_sh x) ⊙ up_sh x)
/// </code>
/// Quantized weights are never widened: gate/up/down run through the existing quantized GEMM kernels on the mmap'd pack bytes.
/// </summary>
public static unsafe class Edge0LatentMoe
{
    /// <summary>The shared expert (<c>ffn_{gate,up,down}_shexp</c> + the sigmoid gate <c>ffn_gate_inp_shexp</c>) of a layer, raw GGUF views.</summary>
    /// <param name="Gate">Pointer to the gate projection (<paramref name="Intermediate"/> rows of <c>Hidden</c> inputs).</param>
    /// <param name="Up">Pointer to the up projection.</param>
    /// <param name="Down">Pointer to the down projection (<c>Hidden</c> rows of <paramref name="Intermediate"/> inputs).</param>
    /// <param name="GateUpType">Storage type of gate and up.</param>
    /// <param name="DownType">Storage type of down.</param>
    /// <param name="Intermediate">Shared-expert FFN width (640).</param>
    /// <param name="InputGate">F32 <c>[Hidden]</c> sigmoid-gate weights; <see langword="null"/> = ungated.</param>
    public readonly record struct SharedExpert(nint Gate, nint Up, nint Down, QuantizationType GateUpType, QuantizationType DownType,
        int Intermediate, float[]? InputGate);

    /// <summary>Tokens processed per chunk (bounds the latent scratch; chunking does not change a token's result).</summary>
    private const int ChunkTokens = 64;

    /// <summary>
    /// Runs the MoE half of <paramref name="layer"/> for <paramref name="tokens"/> rows.
    /// </summary>
    /// <param name="pack">The opened pack.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="router">F32 router matrix <c>[experts][hidden]</c> (<c>ffn_gate_inp</c>).</param>
    /// <param name="topK">Experts per token (10).</param>
    /// <param name="normTopKProb">Renormalise the top-k probabilities (true for Edge0).</param>
    /// <param name="shared">Shared expert, or null to skip it.</param>
    /// <param name="x">Input <c>[tokens][hidden]</c>.</param>
    /// <param name="y">Output <c>[tokens][hidden]</c>, fully overwritten.</param>
    /// <param name="tokens">Row count.</param>
    /// <param name="pool">Optional thread pool.</param>
    public static void Forward(Edge0Pack pack, int layer, ReadOnlySpan<float> router, int topK, bool normTopKProb,
        in SharedExpert? shared, ReadOnlySpan<float> x, Span<float> y, int tokens, ComputeThreadPool? pool = null)
    {
        ArgumentNullException.ThrowIfNull(pack);
        int H = pack.Hidden;
        if (x.Length < (long)tokens * H) throw new ArgumentException("x too small.", nameof(x));
        if (y.Length < (long)tokens * H) throw new ArgumentException("y too small.", nameof(y));
        if (router.Length < (long)pack.Experts * H) throw new ArgumentException("router too small.", nameof(router));
        for (int t0 = 0; t0 < tokens; t0 += ChunkTokens)
        {
            int tc = Math.Min(ChunkTokens, tokens - t0);
            ForwardChunk(pack, layer, router, topK, normTopKProb, x.Slice(t0 * H, tc * H), y.Slice(t0 * H, tc * H), tc, pool);
            if (shared is { } sh) AddShared(sh, H, x.Slice(t0 * H, tc * H), y.Slice(t0 * H, tc * H), tc, pool);
        }
    }

    private static void ForwardChunk(Edge0Pack pack, int layer, ReadOnlySpan<float> router, int k, bool norm,
        ReadOnlySpan<float> x, Span<float> y, int T, ComputeThreadPool? pool)
    {
        int H = pack.Hidden, L = pack.Latent, G = pack.Groups, E = pack.Experts, V = T * k;
        var rent = new Rented();
        try
        {
            int[] assignExpert = rent.Ints(V), cursors = rent.Ints(E + 1), bucketTokens = rent.Ints(V), bucketSlots = rent.Ints(V),
                unique = rent.Ints(Math.Min(E, V) + 1);
            float[] assignWeight = rent.Floats(V);
            MoeSwiGluMlp.Route(x, router, assignExpert, assignWeight, cursors, bucketTokens, bucketSlots, unique, E, k, H, T, norm);

            // Which groups does this chunk touch? Only those D/U are applied.
            ReadOnlySpan<int> labels = pack.Labels(layer);
            Span<int> slotOf = stackalloc int[G];
            slotOf.Fill(-1);
            int used = 0;
            for (int v = 0; v < V; v++)
            {
                int g = labels[assignExpert[v]];
                if (slotOf[g] < 0) slotOf[g] = used++;
            }

            // ── D, then the column scale: xl[g][t] = (D_g x_t) * (1/s), one T x L slab per used group ──
            float[] xlBuf = rent.Floats(used * T * L);
            ReadOnlySpan<float> inv = pack.InverseColumnScale(layer);
            for (int g = 0; g < G; g++)
            {
                if (slotOf[g] < 0) continue;
                Span<float> slab = xlBuf.AsSpan(slotOf[g] * T * L, T * L);
                fixed (float* xp = x) fixed (float* op = slab)
                    GemmLatentDense(pack.D(layer, g), pack.LatentDenseType, xp, op, L, H, T, pool);
                if (!inv.IsEmpty)
                    for (int t = 0; t < T; t++)
                        TensorPrimitives.Multiply(slab.Slice(t * L, L), inv, slab.Slice(t * L, L));
            }

            // ── virtual tokens: one (token, expert) pair per row, k' = 1; hidden' row = that expert's group latent ──
            float[] hidden = rent.Floats(V * L), yv = rent.Floats(V * L);
            int[] vCursors = rent.Ints(E + 1), vTokens = rent.Ints(V), vSlots = rent.Ints(V), vUnique = rent.Ints(Math.Min(E, V) + 1);
            Array.Clear(vCursors, 0, E + 1);
            for (int v = 0; v < V; v++)
            {
                int e = assignExpert[v], g = labels[e], t = v / k;
                xlBuf.AsSpan((slotOf[g] * T + t) * L, L).CopyTo(hidden.AsSpan(v * L, L));
                vCursors[e]++;
            }
            int run = 0;
            for (int e = 0; e <= E; e++) { int c = vCursors[e]; vCursors[e] = run; run += c; }
            Span<int> wr = E <= 1024 ? stackalloc int[E] : new int[E];
            for (int e = 0; e < E; e++) wr[e] = vCursors[e];
            for (int v = 0; v < V; v++) { int pos = wr[assignExpert[v]]++; vTokens[pos] = v; vSlots[pos] = 0; }
            int uc = 0;
            for (int e = 0; e < E; e++) if (vCursors[e + 1] > vCursors[e]) vUnique[uc++] = e;

            long stride = pack.ExpertRecordBytes;
            MoeSwiGluMlp.ExecuteRoutedFromAssignments(
                hidden.AsSpan(0, V * L),
                pack.GateBank(layer), pack.GateUpType, stride, ReadOnlySpan<nint>.Empty,
                pack.UpBank(layer), pack.GateUpType, stride, ReadOnlySpan<nint>.Empty,
                pack.DownBank(layer), pack.DownType, stride, ReadOnlySpan<nint>.Empty,
                assignExpert.AsSpan(0, V), assignWeight.AsSpan(0, V), vCursors, vTokens, vSlots, vUnique, uc,
                yv.AsSpan(0, V * L),
                E, 1, L, pack.ExpertFf, V,
                ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, ReadOnlySpan<nint>.Empty, 0, ReadOnlySpan<float>.Empty,
                loraAdapter: null, loraLayer: layer, threadPool: pool, packedLegacyExperts: true);

            // ── z[g][t] = Σ_{j in group g} w_j · down_j h_j (the weight was applied by the kernel); out = Σ_g U_g z_g ──
            y.Slice(0, T * H).Clear();
            float[] tmp = rent.Floats(T * H);
            for (int g = 0; g < G; g++)
            {
                if (slotOf[g] < 0) continue;
                float[] z = rent.Floats(T * L);
                Array.Clear(z, 0, T * L);
                for (int t = 0; t < T; t++)
                    for (int j = 0; j < k; j++)
                    {
                        int v = t * k + j;
                        if (labels[assignExpert[v]] != g) continue;
                        TensorPrimitives.Add(z.AsSpan(t * L, L), yv.AsSpan(v * L, L), z.AsSpan(t * L, L));
                    }
                fixed (float* zp = z) fixed (float* tp = tmp)
                    GemmLatentDense(pack.U(layer, g), pack.LatentDenseType, zp, tp, H, L, T, pool);
                TensorPrimitives.Add(y.Slice(0, T * H), tmp.AsSpan(0, T * H), y.Slice(0, T * H));
            }
        }
        finally { rent.Dispose(); }
    }

    private static void AddShared(in SharedExpert s, int H, ReadOnlySpan<float> x, Span<float> y, int T, ComputeThreadPool? pool)
    {
        int I = s.Intermediate;
        float[] g = ArrayPool<float>.Shared.Rent(T * I), u = ArrayPool<float>.Shared.Rent(T * I), h = ArrayPool<float>.Shared.Rent(T * I),
            d = ArrayPool<float>.Shared.Rent(T * H);
        try
        {
            fixed (float* xp = x) fixed (float* gp = g) fixed (float* up = u) fixed (float* hp = h) fixed (float* dp = d)
            {
                GemmAny(s.Gate, s.GateUpType, xp, gp, I, H, T, pool);
                GemmAny(s.Up, s.GateUpType, xp, up, I, H, T, pool);
                for (int t = 0; t < T; t++) FusedOps.SwiGLU(g.AsSpan(t * I, I), u.AsSpan(t * I, I), h.AsSpan(t * I, I));
                GemmAny(s.Down, s.DownType, hp, dp, H, I, T, pool);
            }
            for (int t = 0; t < T; t++)
            {
                float scale = 1f;
                if (s.InputGate is { } w)
                {
                    float logit = TensorPrimitives.Dot(w.AsSpan(0, H), x.Slice(t * H, H));
                    scale = 1f / (1f + MathF.Exp(-logit));
                }
                TensorPrimitives.MultiplyAdd(d.AsSpan(t * H, H), scale, y.Slice(t * H, H), y.Slice(t * H, H));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(g); ArrayPool<float>.Shared.Return(u); ArrayPool<float>.Shared.Return(h); ArrayPool<float>.Shared.Return(d);
        }
    }

    private static void GemmLatentDense(nint w, QuantizationType qt, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
        => GemmAny(w, qt, b, c, m, k, n, pool);

    private static void GemmAny(nint w, QuantizationType qt, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.Q8_0: MatMul.GemmQ8_0((byte*)w, b, c, m, k, n, pool, null); return;
            case QuantizationType.Q4_0: MatMul.GemmLegacyQuant((byte*)w, qt, b, c, m, k, n, pool, null); return;
            default: MatMul.GemmDequantRows((byte*)w, qt, b, c, m, k, n, pool); return;
        }
    }

    private sealed class Rented : IDisposable
    {
        private readonly List<int[]> _i = [];
        private readonly List<float[]> _f = [];
        public int[] Ints(int n) { var a = ArrayPool<int>.Shared.Rent(Math.Max(n, 1)); _i.Add(a); return a; }
        public float[] Floats(int n) { var a = ArrayPool<float>.Shared.Rent(Math.Max(n, 1)); _f.Add(a); return a; }
        public void Dispose()
        {
            foreach (var a in _i) ArrayPool<int>.Shared.Return(a);
            foreach (var a in _f) ArrayPool<float>.Shared.Return(a);
        }
    }
}
