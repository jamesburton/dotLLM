using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>The DFlash drafter's own kernels (<c>dflash_f32.ptx</c>, issue #1175): the DFlash2 convolution and the ring attention.</summary>
internal sealed unsafe class CudaDFlashKernels : IDisposable
{
    /// <summary>PTX file name.</summary>
    public const string PtxFileName = "dflash_f32.ptx";

    private CudaModule? _module;
    private readonly nint _conv, _attention, _q4k, _q6k;

    /// <summary>Most rows one multi-column GEMV launch covers.</summary>
    public const int MaxGemvColumns = 8;

    private CudaDFlashKernels(CudaModule module, nint conv, nint attention, nint q4k, nint q6k)
    {
        _module = module;
        _conv = conv;
        _attention = attention;
        _q4k = q4k;
        _q6k = q6k;
    }

    /// <summary>Loads the module from <paramref name="ptxDir"/>, or null when it is missing or incomplete.</summary>
    /// <param name="ptxDir">PTX directory.</param>
    /// <returns>The kernels, or null.</returns>
    public static CudaDFlashKernels? TryLoad(string ptxDir)
    {
        string path = Path.Combine(ptxDir, PtxFileName);
        if (!File.Exists(path)) return null;
        var module = CudaModule.LoadFromFile(path);
        nint conv = module.TryGetFunction("dflash_conv_f32"), attn = module.TryGetFunction("dflash_attention_f32");
        nint q4k = module.TryGetFunction("dflash_gemv_q4_k_multi_f32"), q6k = module.TryGetFunction("dflash_gemv_q6_k_multi_f32");
        if (conv == 0 || attn == 0 || q4k == 0 || q6k == 0) { module.Dispose(); return null; }
        return new CudaDFlashKernels(module, conv, attn, q4k, q6k);
    }

    /// <summary>DFlash2 grouped dynamic causal convolution (out of place).</summary>
    public void LaunchConv(nint x, nint dyn, nint baseKernel, nint y, int rows, int hidden, int kernel, int group, int side, nint stream)
    {
        nint xA = x, dA = dyn, bA = baseKernel, yA = y;
        int rA = rows, hA = hidden, kA = kernel, gA = group, sA = side;
        void** args = stackalloc void*[] { &xA, &dA, &bA, &yA, &rA, &hA, &kA, &gA, &sA };
        uint grid = (uint)((rows * hidden + 255) / 256);
        CudaDriverApi.cuLaunchKernel(_conv, grid, 1, 1, 256, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>Non-causal block attention over the F16 context ring and the block's own K/V.</summary>
    public void LaunchAttention(nint q, nint kb, nint vb, nint ringK, nint ringV, nint output,
        int b, int heads, int kvHeads, int headDim, int ringCap, int lo, int start, int window, float scale, nint stream)
    {
        nint qA = q, kA = kb, vA = vb, rkA = ringK, rvA = ringV, oA = output;
        int bA = b, hA = heads, kvA = kvHeads, hdA = headDim, capA = ringCap, loA = lo, stA = start, wA = window;
        float scA = scale;
        void** args = stackalloc void*[] { &qA, &kA, &vA, &rkA, &rvA, &oA, &bA, &hA, &kvA, &hdA, &capA, &loA, &stA, &wA, &scA };
        int nKeys = start - lo + b;
        uint shared = (uint)((headDim + nKeys + 32) * sizeof(float));
        CudaDriverApi.cuLaunchKernel(_attention, (uint)(b * heads), 1, 1, 128, 1, 1, shared, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>True when <see cref="LaunchGemvMulti"/> covers <paramref name="qt"/>.</summary>
    public static bool HasGemvMulti(DotLLM.Core.Configuration.QuantizationType qt)
        => qt is DotLLM.Core.Configuration.QuantizationType.Q4_K or DotLLM.Core.Configuration.QuantizationType.Q6_K;

    /// <summary><c>y[c][n] = W[n,k] · x[c][k]</c> for up to <see cref="MaxGemvColumns"/> F32 rows, reading each weight block once.</summary>
    public void LaunchGemvMulti(DotLLM.Core.Configuration.QuantizationType qt, nint w, nint x, nint y, int n, int k, int cols, nint stream)
    {
        nint wA = w, xA = x, yA = y;
        int nA = n, kA = k, cA = cols;
        void** args = stackalloc void*[] { &wA, &xA, &yA, &nA, &kA, &cA };
        nint func = qt == DotLLM.Core.Configuration.QuantizationType.Q4_K ? _q4k : _q6k;
        CudaDriverApi.cuLaunchKernel(func, (uint)((n + 3) / 4), 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _module?.Dispose();
        _module = null;
    }
}
