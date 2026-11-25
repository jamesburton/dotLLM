using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>Weight storage kinds the Olala kernels read in place (never widened).</summary>
internal enum OlalaWeightKind
{
    /// <summary>Q8_0: 34-byte blocks of 32 (F16 scale + int8).</summary>
    Q8_0,
    /// <summary>Q6_K: 210-byte superblocks of 256.</summary>
    Q6_K,
    /// <summary>BF16.</summary>
    BF16,
    /// <summary>F32 (router).</summary>
    F32,
}

/// <summary>Launch wrappers over <c>native/ptx/olala.ptx</c> (<c>native/kernels/olala.cu</c>); activations are BF16, state is F32.</summary>
internal sealed unsafe class CudaOlalaKernels : IDisposable
{
    private readonly CudaModule _module;
    private readonly nint[] _gemv1 = new nint[4], _gemv8 = new nint[4], _moeUp = new nint[4], _moeDown = new nint[4], _embed = new nint[4];
    private readonly nint _route, _geodesic, _mamba, _tpaKv, _tpaCarry, _attn, _combine;

    private static readonly string[] Suffix = ["q8_0", "q6_k", "bf16", "f32"];

    /// <param name="ptxDir">PTX directory.</param>
    /// <param name="f32Activations">Diagnostic arm: load <c>olala_f32act.ptx</c> (F32 stored activations) instead of the shipped BF16 module.</param>
    public CudaOlalaKernels(string ptxDir, bool f32Activations = false)
    {
        _module = CudaModule.LoadFromFile(Path.Combine(ptxDir, f32Activations ? "olala_f32act.ptx" : "olala.ptx"));
        try
        {
            for (int i = 0; i < 4; i++)
            {
                _gemv1[i] = _module.GetFunction($"olala_gemv_{Suffix[i]}_t1");
                _gemv8[i] = _module.GetFunction($"olala_gemv_{Suffix[i]}_t8");
                _moeUp[i] = _module.GetFunction($"olala_moe_up_{Suffix[i]}");
                _moeDown[i] = _module.GetFunction($"olala_moe_down_{Suffix[i]}");
                if (i != (int)OlalaWeightKind.Q6_K) _embed[i] = _module.GetFunction($"olala_embed_{Suffix[i]}");
            }
            _route = _module.GetFunction("olala_route");
            _geodesic = _module.GetFunction("olala_geodesic");
            _mamba = _module.GetFunction("olala_mamba3_mimo");
            _tpaKv = _module.GetFunction("olala_tpa_kv");
            _tpaCarry = _module.GetFunction("olala_tpa_carry");
            _attn = _module.GetFunction("olala_attn");
            _combine = _module.GetFunction("olala_attn_combine");
        }
        catch { _module.Dispose(); throw; }
    }

    private static void Launch(nint f, uint gx, uint gy, uint bx, uint smem, nint stream, void** args)
        => CudaDriverApi.cuLaunchKernel(f, gx, gy, 1, bx, 1, 1, smem, stream, (nint)args, 0).ThrowOnError();

    /// <summary>Row byte size of <paramref name="inDim"/> elements of <paramref name="kind"/>.</summary>
    public static long RowBytes(OlalaWeightKind kind, int inDim) => kind switch
    {
        OlalaWeightKind.Q8_0 => inDim / 32L * 34,
        OlalaWeightKind.Q6_K => inDim / 256L * 210,
        OlalaWeightKind.BF16 => inDim * 2L,
        _ => inDim * 4L,
    };

    /// <summary>Whether a row of <paramref name="inDim"/> elements can be read by the kernels (block / vector granularity).</summary>
    public static bool SupportsInDim(OlalaWeightKind kind, int inDim) => kind switch
    {
        OlalaWeightKind.Q8_0 => inDim % 32 == 0,
        OlalaWeightKind.Q6_K => inDim % 256 == 0,
        _ => inDim % 8 == 0,
    };

    /// <summary><c>Y[t][row] = W[row] . X[t]</c>. <paramref name="epi"/>: 0 store, 1 ReLU^2, 2 add <paramref name="aux"/> (BF16 [t][row]).</summary>
    public void Gemv(OlalaWeightKind kind, nint w, int inDim, int outDim, nint x, nint y, int tokens, bool outF32, int epi, nint aux, nint stream)
    {
        ulong rowBytes = (ulong)RowBytes(kind, inDim);
        int f32 = outF32 ? 1 : 0;
        bool single = tokens == 1;
        nint fn = single ? _gemv1[(int)kind] : _gemv8[(int)kind];
        void** args = stackalloc void*[] { &w, &rowBytes, &x, &y, &inDim, &outDim, &tokens, &f32, &epi, &aux };
        Launch(fn, (uint)((outDim + 7) / 8), single ? 1u : (uint)((tokens + 7) / 8), 256, 0, stream, args);
    }

    /// <summary>Routed-expert up projection with ReLU^2: <c>Y[t*K+s] = relu(Up[e_ts] . X[t])^2</c>.</summary>
    public void MoeUp(OlalaWeightKind kind, nint bank, long expertBytes, int inDim, int outDim, nint idx, nint x, nint y, int tokens, int k, nint stream)
    {
        ulong eb = (ulong)expertBytes, rb = (ulong)RowBytes(kind, inDim);
        void** args = stackalloc void*[] { &bank, &eb, &rb, &idx, &x, &y, &inDim, &outDim, &k };
        Launch(_moeUp[(int)kind], (uint)((outDim + 7) / 8), (uint)(tokens * k), 256, 0, stream, args);
    }

    /// <summary>Routed-expert down projection with the routing-weight sum: <c>Y[t] = sum_s w[t][s] Down[e_ts] . Hid[t*K+s]</c>.</summary>
    public void MoeDown(OlalaWeightKind kind, nint bank, long expertBytes, int inDim, int outDim, nint idx, nint wts, nint hid, nint y, int tokens, int k, nint stream)
    {
        ulong eb = (ulong)expertBytes, rb = (ulong)RowBytes(kind, inDim);
        void** args = stackalloc void*[] { &bank, &eb, &rb, &idx, &wts, &hid, &y, &inDim, &outDim, &k };
        Launch(_moeDown[(int)kind], (uint)((outDim + 7) / 8), (uint)tokens, 256, 0, stream, args);
    }

    /// <summary>Router + sigmoid + top-K of (score + bias) + renormalisation, fused so scores are never rounded.</summary>
    public void Route(nint router, nint bias, nint x, nint idx, nint wts, int hidden, int experts, int k, float scale, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &router, &bias, &x, &idx, &wts, &hidden, &experts, &k, &scale };
        Launch(_route, (uint)tokens, 1, 256, (uint)((hidden + 2 * experts) * 4), stream, args);
    }

    /// <summary>Geodesic residual update <c>x &lt;- rotate(x, g)</c>, in place.</summary>
    public void Geodesic(nint x, nint g, int hidden, float scale, float bias, int layer, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &x, &g, &hidden, &scale, &bias, &layer };
        Launch(_geodesic, (uint)tokens, 1, 256, 0, stream, args);
    }

    /// <summary>Embedding lookup + L2 normalisation.</summary>
    public void Embed(OlalaWeightKind kind, nint w, int hidden, nint ids, nint output, int tokens, nint stream)
    {
        ulong rb = (ulong)RowBytes(kind, hidden);
        void** args = stackalloc void*[] { &w, &rb, &ids, &output, &hidden };
        Launch(_embed[(int)kind], (uint)tokens, 1, 256, (uint)(hidden * 4), stream, args);
    }

    /// <summary>Mamba-3 MIMO step over <paramref name="tokens"/> tokens, one block per head; F32 state updated in place.</summary>
    public void Mamba(nint zxdt, nint dyn, int tokens, nint ssm, nint angle, nint kst, nint vst, nint bnorm, nint cnorm, nint bbias, nint cbias,
                      nint dtbias, nint dskip, nint mimoX, nint mimoZ, nint mimoO, nint y, int heads, int headDim, int state, int rank, int angles,
                      float eps, nint stream)
    {
        void** args = stackalloc void*[] { &zxdt, &dyn, &tokens, &ssm, &angle, &kst, &vst, &bnorm, &cnorm, &bbias, &cbias, &dtbias, &dskip,
                                           &mimoX, &mimoZ, &mimoO, &y, &heads, &headDim, &state, &rank, &angles, &eps };
        Launch(_mamba, (uint)heads, 1, 512, (uint)((3 * rank * state + 3 * angles) * 4), stream, args);
    }

    /// <summary>TPA K/V reconstruction, token shift, K RMSNorm and BF16 cache write.</summary>
    public void TpaKv(nint ak, nint av, nint bk, nint bv, nint sk, nint sv, nint lastK, nint lastV, nint kcache, nint vcache, nint kgain,
                      int posBase, int tokens, int kvHeads, int rank, int headDim, float eps, nint stream)
    {
        void** args = stackalloc void*[] { &ak, &av, &bk, &bv, &sk, &sv, &lastK, &lastV, &kcache, &vcache, &kgain, &posBase, &tokens, &kvHeads, &rank, &headDim, &eps };
        Launch(_tpaKv, (uint)kvHeads, (uint)tokens, 128, 0, stream, args);
    }

    /// <summary>Stores the last token's raw K/V as the F32 token-shift carry.</summary>
    public void TpaCarry(nint ak, nint av, nint bk, nint bv, nint lastK, nint lastV, int tokens, int kvHeads, int rank, int headDim, nint stream)
    {
        void** args = stackalloc void*[] { &ak, &av, &bk, &bv, &lastK, &lastV, &tokens, &kvHeads, &rank, &headDim };
        Launch(_tpaCarry, (uint)kvHeads, 1, 128, 0, stream, args);
    }

    /// <summary>Attention of <paramref name="tokens"/> new tokens over the BF16 history.</summary>
    public void Attention(nint q, nint kc, nint vc, nint output, nint qgain, nint scaler, int posBase, int tokens, int qHeads, int kvHeads, int headDim,
                          int window, float softcap, float eps, nint stream)
    {
        void** args = stackalloc void*[] { &q, &kc, &vc, &output, &qgain, &scaler, &posBase, &qHeads, &kvHeads, &headDim, &window, &softcap, &eps };
        Launch(_attn, (uint)qHeads, (uint)tokens, 256, 0, stream, args);
    }

    /// <summary>Differential combine and output gate.</summary>
    public void Combine(nint attn, nint lam, nint gate, nint output, int tokens, int kvHeads, int group, int headDim, float gateBias, nint stream)
    {
        void** args = stackalloc void*[] { &attn, &lam, &gate, &output, &kvHeads, &group, &headDim, &gateBias };
        Launch(_combine, (uint)tokens, 1, 256, 0, stream, args);
    }

    public void Dispose() => _module.Dispose();
}
