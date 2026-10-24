using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures.DFlash;

/// <summary>
/// The pieces of a target model a DFlash drafter borrows (issue #1175): the drafter has no token embedding or output head of
/// its own, it embeds its block with the target's <c>token_embd</c> and projects its hidden rows with the target's lm-head
/// (by reference, never copied). Buffers are in the drafter backend's memory (host for <see cref="DFlashCpuBackend"/>, device
/// for a GPU backend).
/// </summary>
internal unsafe interface IDFlashTargetOps
{
    /// <summary>Embeds <paramref name="tokens"/> into <paramref name="dst"/> (<c>[tokens, hidden]</c> F32), exactly as the target's own forward does.</summary>
    /// <param name="tokens">Token ids.</param>
    /// <param name="dst">Destination rows, in backend memory.</param>
    void EmbedForDraft(ReadOnlySpan<int> tokens, nint dst);

    /// <summary>Projects final-normed hidden rows to logits with the target's output head (and any activation rotation it folds in).</summary>
    /// <param name="hidden">Rows <c>[rows, hidden]</c> in backend memory; may be overwritten.</param>
    /// <param name="rows">Row count.</param>
    /// <param name="logits">Host destination <c>[rows, vocab]</c>.</param>
    void ProjectForDraft(nint hidden, int rows, float* logits);

    /// <summary>
    /// Embeds <paramref name="tokens"/> into HOST rows <c>[tokens, hidden]</c> F32 with the target's <c>token_embd</c>, read in place
    /// from its stored type. The LiLiCorr head (#1237) embeds its candidate tokens on the host.
    /// </summary>
    /// <param name="tokens">Token ids.</param>
    /// <param name="dst">Host destination rows.</param>
    void EmbedHost(ReadOnlySpan<int> tokens, float* dst)
        => throw new NotSupportedException($"{GetType().Name} cannot embed on the host; LiLiCorr drafters need it.");
}

/// <summary>
/// The kernels a <see cref="DFlashDrafter"/> runs on (issue #1175). Buffers are opaque handles in the backend's memory; weights
/// are named by the drafter's own <see cref="DFlashDrafter.Matrix"/> values and norm arrays, which a GPU backend maps to the
/// device copies it uploaded at bind time.
/// </summary>
internal unsafe interface IDFlashBackend : IDisposable
{
    /// <summary>Called once with the drafter whose weights the backend will be asked to use.</summary>
    /// <param name="drafter">The drafter.</param>
    void Bind(DFlashDrafter drafter);

    /// <summary>Allocates <paramref name="floats"/> F32 scratch.</summary>
    nint Alloc(long floats);

    /// <summary>Allocates <paramref name="halves"/> F16 (the context ring).</summary>
    nint AllocHalf(long halves);

    /// <summary>Frees a buffer from <see cref="Alloc"/> / <see cref="AllocHalf"/>.</summary>
    void Free(nint buffer);

    /// <summary>Copies <paramref name="count"/> floats from backend memory to the host.</summary>
    void Download(nint src, float* dst, int count);

    /// <summary>Widens host F16 tap rows into backend F32.</summary>
    void LoadTaps(Half* taps, int count, nint dst);

    /// <summary><c>y[rows, w.Out] = x[rows, w.In] · wᵀ</c> with the weight in its stored type.</summary>
    void Gemm(in DFlashDrafter.Matrix w, nint x, nint y, int rows);

    /// <summary>Per-row RMSNorm of <paramref name="rows"/> rows of <paramref name="weight"/>.Length.</summary>
    void RmsNorm(nint src, nint dst, int rows, float[] weight);

    /// <summary>Per-head RMSNorm then NeoX RoPE (full head) of rows at <paramref name="positions"/>.</summary>
    void NormRopeHeads(nint x, int heads, float[] norm, ReadOnlySpan<int> positions);

    /// <summary>DFlash2 grouped dynamic causal convolution, out of place (see <see cref="DFlashDrafter"/>).</summary>
    void Conv(nint src, nint dyn, float[] baseKernel, int side, int rows, nint dst);

    /// <summary><c>gate = silu(gate) * up</c> over <paramref name="n"/> elements.</summary>
    void SwiGlu(nint gate, nint up, int n);

    /// <summary><c>x += y</c> over <paramref name="n"/> elements.</summary>
    void Add(nint x, nint y, int n);

    /// <summary>Writes K/V rows (F32, <c>[rows, kvDim]</c>) into the F16 ring at <c>pos % capacity</c>.</summary>
    void WriteRing(nint ringK, nint ringV, int capacity, ReadOnlySpan<int> positions, nint k, nint v);

    /// <summary>
    /// Non-causal attention of <paramref name="b"/> block queries over the ring positions <c>[lo, start)</c> within the sliding
    /// window of each query, then the block's own K/V. Writes <paramref name="output"/>. With <paramref name="sinks"/> (one logit
    /// per query head, #1237) the softmax denominator also holds <c>exp(sink)</c>, as llama.cpp's <c>soft_max_ext</c> with sinks.
    /// </summary>
    void Attend(nint ringK, nint ringV, int capacity, int lo, int start, int b, nint q, nint k, nint v, nint output, float[]? sinks);

    /// <summary>Copies <paramref name="floats"/> F32 values between backend buffers (any offsets).</summary>
    void Copy(nint src, nint dst, int floats);
}

/// <summary>The CPU drafter backend: host buffers, the CPU GEMM dispatch, head-parallel attention on the pool.</summary>
internal sealed unsafe class DFlashCpuBackend(ComputeThreadPool? pool, bool ownsPool = false) : IDFlashBackend
{
    private DFlashDrafter _d = null!;
    private float[] _invFreq = [];
    private float _eps;

    /// <inheritdoc/>
    public void Bind(DFlashDrafter drafter)
    {
        _d = drafter;
        _eps = drafter.Eps;
        _invFreq = new float[drafter.HeadDim / 2];
        for (int i = 0; i < _invFreq.Length; i++)
            _invFreq[i] = (float)(1.0 / Math.Pow(drafter.RopeTheta, 2.0 * i / drafter.HeadDim));
    }

    /// <inheritdoc/>
    public nint Alloc(long floats) => (nint)NativeMemory.AlignedAlloc((nuint)(Math.Max(floats, 1) * sizeof(float)), 64);

    /// <inheritdoc/>
    public nint AllocHalf(long halves) => (nint)NativeMemory.AlignedAlloc((nuint)(Math.Max(halves, 1) * sizeof(ushort)), 64);

    /// <inheritdoc/>
    public void Free(nint buffer) { if (buffer != 0) NativeMemory.AlignedFree((void*)buffer); }

    /// <inheritdoc/>
    public void Download(nint src, float* dst, int count) => new ReadOnlySpan<float>((void*)src, count).CopyTo(new Span<float>(dst, count));

    /// <inheritdoc/>
    public void LoadTaps(Half* taps, int count, nint dst)
        => TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(taps, count), new Span<float>((void*)dst, count));

    /// <inheritdoc/>
    public void Gemm(in DFlashDrafter.Matrix w, nint x, nint y, int rows)
    {
        NemotronHTransformerModel.GemmDispatch(w.Ptr, w.Qt, (float*)x, (float*)y, w.Out, w.In, rows, pool, null);
        if (w.Scale != 1f)
        {
            var o = new Span<float>((void*)y, checked(rows * w.Out));
            TensorPrimitives.Multiply(o, w.Scale, o);
        }
    }

    /// <inheritdoc/>
    public void Copy(nint src, nint dst, int floats)
        => new ReadOnlySpan<float>((void*)src, floats).CopyTo(new Span<float>((void*)dst, floats));

    /// <inheritdoc/>
    public void RmsNorm(nint src, nint dst, int rows, float[] weight)
    {
        int n = weight.Length;
        for (int t = 0; t < rows; t++)
            Cpu.Kernels.RmsNorm.Execute(new ReadOnlySpan<float>((float*)src + (long)t * n, n), weight, _eps,
                new Span<float>((float*)dst + (long)t * n, n));
    }

    /// <inheritdoc/>
    public void NormRopeHeads(nint x, int heads, float[] norm, ReadOnlySpan<int> positions)
    {
        int hd = _d.HeadDim;
        Span<float> cos = stackalloc float[hd / 2];
        Span<float> sin = stackalloc float[hd / 2];
        for (int t = 0; t < positions.Length; t++)
        {
            for (int i = 0; i < cos.Length; i++)
            {
                double a = (double)positions[t] * _invFreq[i];
                cos[i] = (float)Math.Cos(a);
                sin[i] = (float)Math.Sin(a);
            }
            for (int h = 0; h < heads; h++)
            {
                var v = new Span<float>((float*)x + ((long)t * heads + h) * hd, hd);
                Cpu.Kernels.RmsNorm.Execute(v, norm, _eps, v);
                RoPE.ApplyRotationNeoX(v, cos, sin, hd);
            }
        }
    }

    /// <inheritdoc/>
    public void Conv(nint src, nint dyn, float[] baseKernel, int side, int rows, nint dst)
    {
        float* x = (float*)src, y = (float*)dst, dy = (float*)dyn;
        int h = _d.HiddenSize, k = _d.ConvKernel, group = _d.ConvGroup, groups = h / group, dynWidth = 2 * k * groups;
        for (int p = 0; p < rows; p++)
        {
            float* d = dy + (long)p * dynWidth + side * k * groups;
            for (int c = 0; c < h; c++)
            {
                float s = 0f;
                for (int tap = 0; tap < k && tap <= p; tap++)
                    s += (baseKernel[(side * k + tap) * h + c] + d[tap * groups + c / group]) * x[(long)(p - tap) * h + c];
                y[(long)p * h + c] = s;
            }
        }
    }

    /// <inheritdoc/>
    public void SwiGlu(nint gate, nint up, int n)
    {
        var g = new Span<float>((void*)gate, n);
        SiLu.Execute(g, g);
        TensorPrimitives.Multiply(g, new ReadOnlySpan<float>((void*)up, n), g);
    }

    /// <inheritdoc/>
    public void Add(nint x, nint y, int n)
        => TensorPrimitives.Add(new ReadOnlySpan<float>((void*)x, n), new ReadOnlySpan<float>((void*)y, n), new Span<float>((void*)x, n));

    /// <inheritdoc/>
    public void WriteRing(nint ringK, nint ringV, int capacity, ReadOnlySpan<int> positions, nint k, nint v)
    {
        int kvDim = _d.NumKvHeads * _d.HeadDim;
        for (int t = 0; t < positions.Length; t++)
        {
            long off = (long)(positions[t] % capacity) * kvDim;
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>((float*)k + (long)t * kvDim, kvDim), new Span<Half>((Half*)ringK + off, kvDim));
            TensorPrimitives.ConvertToHalf(new ReadOnlySpan<float>((float*)v + (long)t * kvDim, kvDim), new Span<Half>((Half*)ringV + off, kvDim));
        }
    }

    /// <inheritdoc/>
    public void Attend(nint ringK, nint ringV, int capacity, int lo, int start, int b, nint q, nint k, nint v, nint output, float[]? sinks)
    {
        if (sinks is not null && sinks.Length != _d.NumHeads)
            throw new ArgumentException($"Expected {_d.NumHeads} sink logits, got {sinks.Length}.", nameof(sinks));
        fixed (float* sp = sinks)
            AttendCore(ringK, ringV, capacity, lo, start, b, q, k, v, output, sp);
    }

    private void AttendCore(nint ringK, nint ringV, int capacity, int lo, int start, int b, nint q, nint k, nint v, nint output, float* sinks)
    {
        var ctx = new AttendCtx
        {
            Sinks = sinks,
            Q = (float*)q, K = (float*)k, V = (float*)v, Out = (float*)output,
            RingK = (Half*)ringK, RingV = (Half*)ringV, RingCap = capacity,
            Lo = lo, Start = start, B = b, Window = _d.Window,
            Heads = _d.NumHeads, KvHeads = _d.NumKvHeads, HeadDim = _d.HeadDim,
            Scale = 1f / MathF.Sqrt(_d.HeadDim),
        };
        if (pool is null)
        {
            for (int hh = 0; hh < ctx.Heads; hh++) AttendHead(&ctx, hh);
            return;
        }
        pool.Dispatch((nint)(&ctx), &AttendWorker);
    }

    private struct AttendCtx
    {
        public float* Q, K, V, Out, Sinks;
        public Half* RingK, RingV;
        public int RingCap, Lo, Start, B, Window, Heads, KvHeads, HeadDim;
        public float Scale;
    }

    private static void AttendWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        var ctx = (AttendCtx*)ctxPtr;
        ComputeThreadPool.PartitionRange(ctx->Heads, threadIdx, threadCount, out int s, out int e);
        for (int hh = s; hh < e; hh++) AttendHead(ctx, hh);
    }

    [SkipLocalsInit]
    private static void AttendHead(AttendCtx* c, int head)
    {
        int d = c->HeadDim, kvh = head / (c->Heads / c->KvHeads), b = c->B;
        int qStride = c->Heads * d, kvStride = c->KvHeads * d;
        int nCtx = c->Start - c->Lo, nKeys = nCtx + b;
        float* scores = (float*)NativeMemory.Alloc((nuint)((long)b * nKeys * sizeof(float)));
        try
        {
            Span<float> kf = stackalloc float[d];
            // Context keys (F16 ring): convert each once, score every query row against it.
            for (int j = 0; j < nCtx; j++)
            {
                int pos = c->Lo + j;
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingK + (long)(pos % c->RingCap) * kvStride + kvh * d, d), kf);
                for (int i = 0; i < b; i++)
                {
                    // Per-query window: positions at or beyond Window behind the query are masked.
                    scores[(long)i * nKeys + j] = c->Start + i - pos >= c->Window
                        ? float.NegativeInfinity
                        : TensorPrimitives.Dot(new ReadOnlySpan<float>(c->Q + (long)i * qStride + head * d, d), kf) * c->Scale;
                }
            }
            for (int j = 0; j < b; j++)
            {
                var kr = new ReadOnlySpan<float>(c->K + (long)j * kvStride + kvh * d, d);
                for (int i = 0; i < b; i++)
                    scores[(long)i * nKeys + nCtx + j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(c->Q + (long)i * qStride + head * d, d), kr) * c->Scale;
            }
            for (int i = 0; i < b; i++)
            {
                var row = new Span<float>(scores + (long)i * nKeys, nKeys);
                float m = TensorPrimitives.Max((ReadOnlySpan<float>)row);
                // A sink is one more logit in the softmax with no value row: it only enlarges the denominator.
                if (c->Sinks != null) m = MathF.Max(m, c->Sinks[head]);
                TensorPrimitives.Add((ReadOnlySpan<float>)row, -m, row);
                TensorPrimitives.Exp((ReadOnlySpan<float>)row, row);
                float denom = TensorPrimitives.Sum((ReadOnlySpan<float>)row);
                if (c->Sinks != null) denom += MathF.Exp(c->Sinks[head] - m);
                TensorPrimitives.Multiply((ReadOnlySpan<float>)row, 1f / denom, row);
                new Span<float>(c->Out + (long)i * qStride + head * d, d).Clear();
            }
            float* vfp = stackalloc float[d];
            var vf = new Span<float>(vfp, d);
            for (int j = 0; j < nKeys; j++)
            {
                ReadOnlySpan<float> vr;
                if (j < nCtx)
                {
                    int pos = c->Lo + j;
                    TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(c->RingV + (long)(pos % c->RingCap) * kvStride + kvh * d, d), vf);
                    vr = new ReadOnlySpan<float>(vfp, d);
                }
                else
                {
                    vr = new ReadOnlySpan<float>(c->V + (long)(j - nCtx) * kvStride + kvh * d, d);
                }
                for (int i = 0; i < b; i++)
                {
                    float w = scores[(long)i * nKeys + j];
                    if (w == 0f) continue;
                    var o = new Span<float>(c->Out + (long)i * qStride + head * d, d);
                    TensorPrimitives.MultiplyAdd(vr, w, o, o);
                }
            }
        }
        finally
        {
            NativeMemory.Free(scores);
        }
    }

    /// <inheritdoc/>
    public void Dispose() { if (ownsPool) pool?.Dispose(); }
}
