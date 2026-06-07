using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Synthetic equivalence tests for the GPU MLA Phase A attention kernel
/// (<c>attention_mla_f32</c>). Constructs random Q/K_nope/K_pe/V on the host,
/// runs the GPU kernel, and compares against a step-by-step CPU reference that
/// reproduces the per-head causal-masked softmax-weighted-V loop. Tolerance is
/// generous on this F32 path — the kernel and the reference are
/// algorithmically identical, so the only source of drift is the order of
/// floating-point reductions in the online softmax / V accumulation.
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaMlaAttentionTests : IDisposable
{
    private const float Tolerance = 1e-3f;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    public CudaMlaAttentionTests()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableFact]
    public void AttentionMla_SingleToken_SingleHead_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel, "MLA attention PTX not built");

        Run(seqQ: 1, seqKv: 1, numHeads: 1, qkNope: 4, qkRope: 2, vHead: 4, positionOffset: 0, seed: 42);
    }

    [SkippableFact]
    public void AttentionMla_Decode_MultipleHeads_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel, "MLA attention PTX not built");

        // seqQ=1 (decode), seqKv=8 (history+current). Mirrors the tightest hot
        // path on the GPU: one new token attending over a populated cache.
        Run(seqQ: 1, seqKv: 8, numHeads: 4, qkNope: 8, qkRope: 4, vHead: 6, positionOffset: 7, seed: 7);
    }

    [SkippableFact]
    public void AttentionMla_Prefill_MultipleHeads_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel, "MLA attention PTX not built");

        // Prefill: seqQ=seqKv=4. Causal mask gates each query token to its own
        // prefix — ensures the kernel masks tk > tq positions correctly.
        Run(seqQ: 4, seqKv: 4, numHeads: 3, qkNope: 8, qkRope: 4, vHead: 8, positionOffset: 0, seed: 123);
    }

    [SkippableFact]
    public void AttentionMla_DeepSeekV2LiteShapes_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel, "MLA attention PTX not built");

        // DeepSeek-V2-Lite production shapes: 16 heads, qkNope=128, qkRope=64,
        // vHead=128. seqKv=2 keeps the test fast while still exercising the
        // multi-token attention loop (and an interesting tile scan since
        // TILE_KV=128 in the kernel).
        Run(seqQ: 1, seqKv: 2, numHeads: 16, qkNope: 128, qkRope: 64, vHead: 128,
            positionOffset: 1, seed: 99);
    }

    [SkippableFact]
    public void AttentionMla_LargeKvLength_TilingHandled()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasMlaAttentionKernel, "MLA attention PTX not built");

        // seqKv > TILE_KV (128) — exercises the online-softmax rescale path
        // across multiple tiles. positionOffset=199 makes the single new
        // query attend over all 200 cached positions.
        Run(seqQ: 1, seqKv: 200, numHeads: 2, qkNope: 16, qkRope: 8, vHead: 16,
            positionOffset: 199, seed: 555);
    }

    private unsafe void Run(int seqQ, int seqKv, int numHeads,
        int qkNope, int qkRope, int vHead, int positionOffset, int seed)
    {
        int qkHead = qkNope + qkRope;
        var rng = new Random(seed);

        float[] q = RandomArr(rng, seqQ * numHeads * qkHead, 0.5f);
        float[] kNope = RandomArr(rng, seqKv * numHeads * qkNope, 0.5f);
        float[] kPe = RandomArr(rng, seqKv * qkRope, 0.5f);
        float[] v = RandomArr(rng, seqKv * numHeads * vHead, 0.5f);

        float softmaxScale = 1.0f / MathF.Sqrt(qkHead);

        // ── CPU reference ──
        float[] expected = new float[seqQ * numHeads * vHead];
        ComputeReference(q, kNope, kPe, v, expected,
            seqQ, seqKv, numHeads, qkNope, qkRope, vHead, positionOffset, softmaxScale);

        // ── GPU kernel ──
        long qBytes = (long)q.Length * sizeof(float);
        long kNopeBytes = (long)kNope.Length * sizeof(float);
        long kPeBytes = (long)kPe.Length * sizeof(float);
        long vBytes = (long)v.Length * sizeof(float);
        long outBytes = (long)expected.Length * sizeof(float);

        CudaDriverApi.cuMemAlloc_v2(out nint dQ, (nuint)qBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKnope, (nuint)kNopeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dKpe, (nuint)kPeBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dV, (nuint)vBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOut, (nuint)outBytes).ThrowOnError();

        try
        {
            fixed (float* p = q) CudaDriverApi.cuMemcpyHtoD_v2(dQ, (nint)p, (nuint)qBytes).ThrowOnError();
            fixed (float* p = kNope) CudaDriverApi.cuMemcpyHtoD_v2(dKnope, (nint)p, (nuint)kNopeBytes).ThrowOnError();
            fixed (float* p = kPe) CudaDriverApi.cuMemcpyHtoD_v2(dKpe, (nint)p, (nuint)kPeBytes).ThrowOnError();
            fixed (float* p = v) CudaDriverApi.cuMemcpyHtoD_v2(dV, (nint)p, (nuint)vBytes).ThrowOnError();

            _kernels!.LaunchAttentionMla(
                dQ, dKnope, dKpe, dV, dOut,
                seqQ, seqKv, numHeads, qkNope, qkRope, vHead,
                positionOffset, softmaxScale, _stream!.Handle);
            _stream.Synchronize();

            float[] actual = new float[expected.Length];
            fixed (float* p = actual)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOut, (nuint)outBytes).ThrowOnError();

            for (int i = 0; i < expected.Length; i++)
            {
                float diff = MathF.Abs(expected[i] - actual[i]);
                Assert.True(diff <= Tolerance,
                    $"index {i}: expected={expected[i]} actual={actual[i]} diff={diff} (tol={Tolerance})");
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dQ);
            CudaDriverApi.cuMemFree_v2(dKnope);
            CudaDriverApi.cuMemFree_v2(dKpe);
            CudaDriverApi.cuMemFree_v2(dV);
            CudaDriverApi.cuMemFree_v2(dOut);
        }
    }

    /// <summary>
    /// CPU reference: per-head causal-masked softmax-weighted-V — a stripped
    /// transcription of the attention loop in <c>MlaAttention.Execute</c>
    /// without the projections (already applied) or the cache writes.
    /// </summary>
    private static void ComputeReference(
        float[] q, float[] kNope, float[] kPe, float[] v, float[] output,
        int seqQ, int seqKv, int numHeads, int qkNope, int qkRope, int vHead,
        int positionOffset, float softmaxScale)
    {
        int qkHead = qkNope + qkRope;
        int qStride = numHeads * qkHead;
        int kNopeStride = numHeads * qkNope;
        int vStride = numHeads * vHead;

        for (int h = 0; h < numHeads; h++)
        {
            for (int tq = 0; tq < seqQ; tq++)
            {
                int posQ = positionOffset + tq;
                float[] scores = new float[seqKv];
                for (int s = 0; s < seqKv; s++)
                {
                    if (s > posQ) { scores[s] = float.NegativeInfinity; continue; }
                    float dot = 0f;
                    int qNopeOff = tq * qStride + h * qkHead;
                    int qPeOff = qNopeOff + qkNope;
                    int kNopeOff = s * kNopeStride + h * qkNope;
                    int kPeOff = s * qkRope;
                    for (int d = 0; d < qkNope; d++)
                        dot += q[qNopeOff + d] * kNope[kNopeOff + d];
                    for (int d = 0; d < qkRope; d++)
                        dot += q[qPeOff + d] * kPe[kPeOff + d];
                    scores[s] = dot * softmaxScale;
                }

                float mx = float.NegativeInfinity;
                for (int i = 0; i < scores.Length; i++) if (scores[i] > mx) mx = scores[i];
                float sum = 0f;
                for (int i = 0; i < scores.Length; i++)
                {
                    if (float.IsNegativeInfinity(scores[i])) { scores[i] = 0f; continue; }
                    scores[i] = MathF.Exp(scores[i] - mx);
                    sum += scores[i];
                }
                if (sum > 0f) for (int i = 0; i < scores.Length; i++) scores[i] /= sum;

                int outOff = tq * vStride + h * vHead;
                for (int d = 0; d < vHead; d++)
                {
                    float acc = 0f;
                    for (int s = 0; s <= posQ && s < seqKv; s++)
                        acc += scores[s] * v[s * vStride + h * vHead + d];
                    output[outOff + d] = acc;
                }
            }
        }
    }

    private static float[] RandomArr(Random rng, int n, float scale)
    {
        float[] arr = new float[n];
        for (int i = 0; i < n; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return arr;
    }
}
