using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Vision;

/// <summary>Deliberately wrong implementations used to prove the Qwen-VL vision gates can fail (CLAUDE.md "Prove the Test Can Fail"). Test-only.</summary>
public enum QwenVlVisionMutant
{
    /// <summary>The correct implementation.</summary>
    None,
    /// <summary>Position-embedding interpolation with half-pixel centres instead of <c>align_corners</c>.</summary>
    PosembHalfPixel,
    /// <summary>Tokens kept in raster order instead of the 2x2 merge-block order (positions, pos-embd and merge all follow).</summary>
    RasterOrder,
    /// <summary>The y and x rope positions swapped.</summary>
    RopeAxesSwapped,
    /// <summary>Vision RoPE pairs adjacent elements instead of (i, i + 36).</summary>
    RopeAdjacentPairs,
    /// <summary>Only the first of the two patch-embedding kernels is applied.</summary>
    FirstPatchKernelOnly,
    /// <summary>No post-LayerNorm.</summary>
    NoPostLn,
    /// <summary>Exact erf GELU instead of the tanh approximation.</summary>
    GeluErf,
    /// <summary>Attention scale 1/sqrt(64) instead of 1/sqrt(72).</summary>
    WrongAttnScale,
}

/// <summary>
/// CPU vision tower + projector of the Qwen3-VL family as shipped for Bonsai 2.5 27B
/// (<c>Ternary-Bonsai-2.5-27B-mmproj-Q8_0.gguf</c>, <c>general.architecture = clip</c>, projector <c>qwen3vl_merger</c>; llama.cpp b12780
/// <c>tools/mtmd/models/qwen3vl.cpp</c>; issue #2073).
/// </summary>
/// <remarks>
/// <para>Graph: two 16x16x3 patch kernels summed (the temporal-2 Conv3D split into two Conv2D; for a still image both see the same frame, so they fold into
/// one kernel at load), patch bias, tokens reordered into 2x2 merge blocks, bilinear (<c>align_corners</c>) 48x48 position embeddings added in the same order,
/// then 27 x (LayerNorm eps 1e-6, fused QKV + bias, vision RoPE (head 72: pairs (i, i + 36), the first 18 rotated by the row, the next 18 by the column,
/// theta 1e4), full attention scale 1/sqrt(72), out-proj + bias, residual, LayerNorm, up + bias, tanh GELU, down + bias, residual), post-LayerNorm, four consecutive
/// tokens (one merge block) concatenated to 4608, mm.0 + bias, tanh GELU, mm.2 + bias to the 5120-wide text embedding. No CLS, no deepstack (this checkpoint has none).</para>
/// <para>Weights are read in place from the mmap'd GGUF (Q8_0 matrices through the Q8_0 GEMM; <c>ffn_down</c> is F16 because 4304 is not a multiple of 32, read through
/// the F16 GEMM) and never expanded. Parity details kept on purpose: ggml quantises the activations of a Q8_0 matmul to Q8_0 and those of an F16 matmul to F16, and
/// its CPU GELU goes through an F16 lookup table; both are replicated.</para>
/// </remarks>
public sealed unsafe class QwenVlVisionEncoder : IImageEncoder
{
    private const int P = QwenVlImagePreprocessor.PatchSize;
    private const int GridPos = 48;

    private readonly record struct Mat(nint Ptr, QuantizationType Qt, int In, int Out);

    private sealed class Block
    {
        public required nint Ln1W, Ln1B, Ln2W, Ln2B, QkvB, OB, UpB, DownB;
        public required Mat Qkv, O, Up, Down;
    }

    private readonly GgufFile _gguf;
    private readonly bool _ownsGguf;
    private readonly ComputeThreadPool? _pool;
    private readonly Block[] _blocks;
    private readonly nint _patchBias, _postLnW, _postLnB;
    private readonly nint _patchFold;   // folded kernel [E][3*P*P], F32 / F16 / BF16 per _embedPrecision
    private readonly nint _posConv;     // position table [2304][E] in _embedPrecision (F32: the file pointer)
    private readonly Mat _mm0, _mm1;
    private readonly nint _mm0B, _mm1B;
    private readonly int _nLayer, _hidden, _heads, _headDim, _ff, _outDim;
    private readonly float _eps;
    private readonly Inkling2Arena _arena = new();
    private readonly object _lock = new();
    private readonly ImageOverLimit _overLimit;
    private readonly VisionEmbedPrecision _embedPrecision;

    /// <summary>Native scratch kept resident after a call (default 128 MB); a larger working set is released when the call ends.</summary>
    internal long ScratchKeepBytes { get; set; } = 128L << 20;

    /// <summary>Native scratch bytes the last call needed at its peak.</summary>
    internal long LastPeakScratchBytes { get; private set; }

    /// <summary>Storage format of the patch kernel and position table in force.</summary>
    public VisionEmbedPrecision EmbedPrecision => _embedPrecision;

    /// <summary>Test seam: deliberately wrong behaviour.</summary>
    internal QwenVlVisionMutant Mutant { get; set; }

    /// <summary>Test seam: receives named intermediate tensors.</summary>
    internal Inkling2TraceSink? Trace { get; set; }

    /// <summary>When true the F16 rounding of the im2col input is skipped (A/B arm).</summary>
    internal bool SkipIm2colF16Rounding { get; set; }

    /// <summary>When true GELU is the exact tanh form instead of the F16-lookup-table form of the reference CPU (A/B arm).</summary>
    internal bool ExactGelu { get; set; }

    /// <summary>Test seam: BGR instead of RGB channel order (preprocessing mutant).</summary>
    internal bool SwapRgb { get; set; }

    /// <inheritdoc/>
    public int EmbeddingDimension => _outDim;

    /// <summary>Opens <paramref name="mmprojPath"/> and loads the encoder.</summary>
    public static QwenVlVisionEncoder Load(string mmprojPath, ThreadingConfig threading,
        ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
        => Load(GgufFile.Open(mmprojPath), threading, ownsGguf: true, overLimit, embedPrecision);

    /// <summary>Loads from an opened mmproj GGUF.</summary>
    public static QwenVlVisionEncoder Load(GgufFile gguf, ThreadingConfig threading, bool ownsGguf = false,
        ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
    {
        var md = gguf.Metadata;
        if (md.GetStringOrDefault("general.architecture") != "clip" || md.GetStringOrDefault("clip.projector_type") != "qwen3vl_merger")
            throw new NotSupportedException("not a qwen3vl_merger mmproj (general.architecture clip, clip.projector_type qwen3vl_merger)");
        if (md.GetUInt32("clip.vision.patch_size") != P || md.GetUInt32("clip.vision.spatial_merge_size") != QwenVlImagePreprocessor.Merge)
            throw new NotSupportedException("qwen3vl_merger mmproj with a patch size / merge other than 16 / 2 is not supported");
        if (md.GetBoolArray("clip.vision.is_deepstack_layers").Any(b => b))
            throw new NotSupportedException("a qwen3vl_merger mmproj with deepstack layers is not supported (the Bonsai 2.5 checkpoint has none)");
        return new QwenVlVisionEncoder(gguf, ownsGguf, threading, overLimit, embedPrecision ?? Inkling2VisionEncoder.EmbedPrecisionFromEnvironment());
    }

    private QwenVlVisionEncoder(GgufFile gguf, bool ownsGguf, ThreadingConfig threading, ImageOverLimit overLimit, VisionEmbedPrecision embedPrecision)
    {
        _gguf = gguf; _ownsGguf = ownsGguf; _overLimit = overLimit; _embedPrecision = embedPrecision;
        var md = gguf.Metadata;
        _nLayer = (int)md.GetUInt32("clip.vision.block_count");
        _hidden = (int)md.GetUInt32("clip.vision.embedding_length");
        _heads = (int)md.GetUInt32("clip.vision.attention.head_count");
        _ff = (int)md.GetUInt32("clip.vision.feed_forward_length");
        _outDim = (int)md.GetUInt32("clip.vision.projection_dim");
        _eps = md.GetFloat32("clip.vision.attention.layer_norm_epsilon");
        _headDim = _hidden / _heads;
        if (_headDim != 72 || _hidden != 1152) throw new NotSupportedException("unexpected qwen3vl_merger vision geometry (head 72 x 16 expected)");
        var t = gguf.TensorsByName;

        nint F32(string n)
        {
            if (!t.TryGetValue(n, out var d)) throw new InvalidDataException($"mmproj tensor {n} missing");
            if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"{n} is {d.QuantizationType}; the vision contract keeps it F32");
            return gguf.TensorDataPointer(d);
        }
        Mat M(string n)
        {
            if (!t.TryGetValue(n, out var d)) throw new InvalidDataException($"mmproj tensor {n} missing");
            if (d.QuantizationType is not (QuantizationType.Q8_0 or QuantizationType.F16 or QuantizationType.F32))
                throw new NotSupportedException($"{n} is {d.QuantizationType}; only Q8_0 / F16 / F32 vision matrices are implemented");
            return new Mat(gguf.TensorDataPointer(d), d.QuantizationType, (int)d.Shape[0], (int)d.Shape[1]);
        }

        _patchBias = F32("v.patch_embd.bias");
        _postLnW = F32("v.post_ln.weight"); _postLnB = F32("v.post_ln.bias");
        if (t.ContainsKey("v.class_embd") || t.ContainsKey("v.pre_ln.weight"))
            throw new NotSupportedException("a CLS token / pre-LayerNorm is not part of the qwen3vl_merger graph");
        if (t["v.position_embd.weight"].Shape[1] != GridPos * GridPos) throw new NotSupportedException("position table is not 48x48");
        _blocks = new Block[_nLayer];
        for (int il = 0; il < _nLayer; il++)
        {
            string B(string s) => $"v.blk.{il}.{s}";
            _blocks[il] = new Block
            {
                Ln1W = F32(B("ln1.weight")), Ln1B = F32(B("ln1.bias")), Ln2W = F32(B("ln2.weight")), Ln2B = F32(B("ln2.bias")),
                QkvB = F32(B("attn_qkv.bias")), OB = F32(B("attn_out.bias")), UpB = F32(B("ffn_up.bias")), DownB = F32(B("ffn_down.bias")),
                Qkv = M(B("attn_qkv.weight")), O = M(B("attn_out.weight")), Up = M(B("ffn_up.weight")), Down = M(B("ffn_down.weight")),
            };
        }
        _mm0 = M("mm.0.weight"); _mm1 = M("mm.2.weight"); _mm0B = F32("mm.0.bias"); _mm1B = F32("mm.2.bias");
        _pool = threading.IsParallel ? new ComputeThreadPool(threading.EffectiveThreadCount) : null;

        // Fold the two temporal kernels (a still image feeds the same frame to both): W = W0 + W1, one pass, then store per the embed precision.
        long kCount = (long)_hidden * 3 * P * P;
        nint w0 = F32("v.patch_embd.weight"), w1 = F32("v.patch_embd.weight.1");
        var fold = (float*)NativeMemory.AlignedAlloc((nuint)(kCount * 4), 64);
        for (long i = 0; i < kCount; i++) fold[i] = ((float*)w0)[i] + ((float*)w1)[i];
        _firstKernel = w0;
        if (embedPrecision == VisionEmbedPrecision.F32) { _patchFold = (nint)fold; }
        else { _patchFold = ConvertTo((nint)fold, kCount, embedPrecision); NativeMemory.AlignedFree(fold); }
        nint pos = F32("v.position_embd.weight");
        _posConv = embedPrecision == VisionEmbedPrecision.F32 ? pos : ConvertTo(pos, (long)GridPos * GridPos * _hidden, embedPrecision);
    }

    private readonly nint _firstKernel;

    private static nint ConvertTo(nint f32, long count, VisionEmbedPrecision to)
    {
        var dst = (ushort*)NativeMemory.AlignedAlloc((nuint)(count * 2), 64);
        var src = (float*)f32;
        if (to == VisionEmbedPrecision.F16)
            for (long i = 0; i < count; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
        else
            for (long i = 0; i < count; i++) dst[i] = Inkling2VisionEncoder.FloatToBf16(src[i]);
        return (nint)dst;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float PosAt(long i) => _embedPrecision switch
    {
        VisionEmbedPrecision.F16 => (float)BitConverter.UInt16BitsToHalf(((ushort*)_posConv)[i]),
        VisionEmbedPrecision.BF16 => BitConverter.UInt32BitsToSingle((uint)((ushort*)_posConv)[i] << 16),
        _ => ((float*)_posConv)[i],
    };

    /// <inheritdoc/>
    public int CountTokens(int width, int height) => QwenVlImagePreprocessor.CountTokens(width, height, _overLimit);

    /// <inheritdoc/>
    public ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        lock (_lock)
        {
            try
            {
                _arena.Reset();
                var (w, h) = QwenVlImagePreprocessor.PlanSize(image.Width, image.Height, _overLimit);
                var hwc = _arena.Alloc<float>(w * h * 3);
                QwenVlImagePreprocessor.Normalize(image, hwc, SwapRgb, _overLimit);
                return RunLocked(hwc, w, h, image.ContentHash, cancellationToken);
            }
            finally { LastPeakScratchBytes = _arena.Capacity; _arena.Trim(ScratchKeepBytes); }
        }
    }

    /// <summary>Encodes an already-normalised interleaved-RGB float image whose sides are multiples of 32 (test seam).</summary>
    internal ImageEmbeddings EncodeNormalized(ReadOnlySpan<float> hwc, int width, int height, string contentHash = "")
    {
        lock (_lock)
        {
            try
            {
                _arena.Reset();
                var copy = _arena.Alloc<float>(hwc.Length);
                hwc.CopyTo(copy);
                return RunLocked(copy, width, height, contentHash, default);
            }
            finally { LastPeakScratchBytes = _arena.Capacity; _arena.Trim(ScratchKeepBytes); }
        }
    }

    /// <summary>Native bytes of the transient scratch held after the last call.</summary>
    internal long ScratchBytes => _arena.Capacity;

    // ───────────────────────────── the tower ─────────────────────────────

    private ImageEmbeddings RunLocked(Span<float> hwc, int W, int H, string hash, CancellationToken ct)
    {
        if (W % (P * 2) != 0 || H % (P * 2) != 0) throw new ArgumentException("image sides must be multiples of 32");
        int gw = W / P, gh = H / P, n = gw * gh, E = _hidden;
        int mw = gw / 2, mh = gh / 2, nm = mw * mh;
        bool raster = Mutant == QwenVlVisionMutant.RasterOrder;

        // token t -> patch (x, y): 2x2 merge blocks in raster order of blocks, (ky, kx) row-major inside a block (ggml permute chain of qwen3vl.cpp).
        var tokX = _arena.Alloc<int>(n); var tokY = _arena.Alloc<int>(n);
        for (int t = 0; t < n; t++)
        {
            if (raster) { tokX[t] = t % gw; tokY[t] = t / gw; continue; }
            int blk = t / 4, inner = t % 4;
            tokX[t] = (blk % mw) * 2 + (inner & 1);
            tokY[t] = (blk / mw) * 2 + (inner >> 1);
        }

        // 1. patch embedding: F16-rounded im2col rows [c][ky][kx] x the folded kernel, in token order.
        var patches = _arena.Alloc<float>(n * 3 * P * P);
        fixed (float* hp = hwc) fixed (float* pp = patches) fixed (int* tx = tokX) fixed (int* ty = tokY)
        {
            var job = new Job { A = hp, C = pp, I0 = W, I1 = H, I5 = (nint)tx, I6 = (nint)ty, I4 = SkipIm2colF16Rounding ? 1 : 0 };
            Run(n, &Im2colWorker, &job);
        }
        var x = _arena.Alloc<float>(n * E);
        fixed (float* a = patches) fixed (float* c = x)
        {
            nint kernel = Mutant == QwenVlVisionMutant.FirstPatchKernelOnly ? _firstKernel : _patchFold;
            var prec = Mutant == QwenVlVisionMutant.FirstPatchKernelOnly ? VisionEmbedPrecision.F32 : _embedPrecision;
            switch (prec)
            {
                case VisionEmbedPrecision.F16: MatMul.GemmF16(kernel, a, c, E, 3 * P * P, n, _pool); break;
                case VisionEmbedPrecision.BF16: MatMul.GemmBf16(kernel, a, c, E, 3 * P * P, n, _pool); break;
                default: MatMul.GemmF32((float*)kernel, a, c, E, 3 * P * P, n, _pool); break;
            }
        }
        AddBias(x, (float*)_patchBias, n, E);
        Trace?.Invoke("patch_embd", x, n, E);

        // 2. position embeddings: bilinear (align_corners) to the patch grid, added per token.
        bool interp = !(gh == GridPos && gw == GridPos);
        var posRows = interp ? _arena.Alloc<float>(n * E) : default;   // [y * gw + x][E]
        if (interp) InterpolatePosemb(posRows, gw, gh);
        for (int t = 0; t < n; t++)
        {
            int p = tokY[t] * gw + tokX[t];
            var dst = x.Slice(t * E, E);
            if (interp) { var pe = posRows.Slice(p * E, E); for (int c = 0; c < E; c++) dst[c] += pe[c]; }
            else for (int c = 0; c < E; c++) dst[c] += PosAt((long)p * E + c);
        }
        Trace?.Invoke("inp_pos_emb", x, n, E);

        // 3. the blocks.
        var xn = _arena.Alloc<float>(n * E);
        var qkv = _arena.Alloc<float>(n * 3 * E);
        var q = _arena.Alloc<float>(n * E); var k = _arena.Alloc<float>(n * E); var v = _arena.Alloc<float>(n * E);
        var att = _arena.Alloc<float>(n * E); var proj = _arena.Alloc<float>(n * E);
        if ((long)n * _ff > int.MaxValue) throw new InvalidOperationException("image too large");
        var hid = _arena.Alloc<float>(n * _ff);
        var cosY = _arena.Alloc<float>(n * 18); var sinY = _arena.Alloc<float>(n * 18);
        var cosX = _arena.Alloc<float>(n * 18); var sinX = _arena.Alloc<float>(n * 18);
        BuildRopeTables(cosY, sinY, cosX, sinX, tokX, tokY);

        for (int il = 0; il < _nLayer; il++)
        {
            ct.ThrowIfCancellationRequested();
            var b = _blocks[il];
            LayerNorm(x, xn, (float*)b.Ln1W, (float*)b.Ln1B, n);
            Linear(b.Qkv, (float*)b.QkvB, xn, qkv, n);
            for (int t = 0; t < n; t++)
            {
                qkv.Slice(t * 3 * E, E).CopyTo(q.Slice(t * E, E));
                qkv.Slice(t * 3 * E + E, E).CopyTo(k.Slice(t * E, E));
                qkv.Slice(t * 3 * E + 2 * E, E).CopyTo(v.Slice(t * E, E));
            }
            Rope(q, cosY, sinY, cosX, sinX, n);
            Rope(k, cosY, sinY, cosX, sinX, n);
            Attention(q, k, v, att, n);
            Linear(b.O, (float*)b.OB, att, proj, n);
            TensorPrimitives.Add(proj.Slice(0, n * E), x.Slice(0, n * E), x.Slice(0, n * E));
            LayerNorm(x, xn, (float*)b.Ln2W, (float*)b.Ln2B, n);
            Linear(b.Up, (float*)b.UpB, xn, hid, n);
            Gelu(hid, n * _ff);
            Linear(b.Down, (float*)b.DownB, hid, proj, n);
            TensorPrimitives.Add(x.Slice(0, n * E), proj.Slice(0, n * E), x.Slice(0, n * E));
            Trace?.Invoke($"layer_out-{il}", x, n, E);
        }

        // 4. post-LayerNorm, merge (4 consecutive tokens), projector.
        if (Mutant != QwenVlVisionMutant.NoPostLn)
        {
            LayerNorm(x, xn, (float*)_postLnW, (float*)_postLnB, n);
            xn.CopyTo(x);
        }
        Trace?.Invoke("post_ln", x, n, E);
        // x [n][E] is already [nm][4E] row-major: block t holds tokens 4t..4t+3 in (ky, kx) order.
        var merged = x.Slice(0, nm * 4 * E);
        var mid = _arena.Alloc<float>(nm * _mm0.Out);
        Linear(_mm0, (float*)_mm0B, merged, mid, nm);
        Gelu(mid, nm * _mm0.Out);
        var outF = _arena.Alloc<float>(nm * _outDim);
        Linear(_mm1, (float*)_mm1B, mid, outF, nm);
        Trace?.Invoke("projected", outF, nm, _outDim);

        var result = new ImageEmbeddings(mw, mh, _outDim, hash);
        var rows = result.Rows;
        for (int i = 0; i < nm * _outDim; i++) rows[i] = (Half)outF[i];
        return result;
    }

    // ───────────────────────────── ops ─────────────────────────────

    private void GemmMat(Mat w, Span<float> input, Span<float> output, int n)
    {
        fixed (float* a = input) fixed (float* c = output)
        {
            switch (w.Qt)
            {
                case QuantizationType.Q8_0:
                    MatMul.GemmQ8_0((byte*)w.Ptr, a, c, w.Out, w.In, n, _pool, null);
                    break;
                case QuantizationType.F16:
                    // ggml quantises the activations of an F16 matmul to F16 (vec_dot_type) before the dot product.
                    for (long i = 0; i < (long)n * w.In; i++) a[i] = (float)(Half)a[i];
                    MatMul.GemmF16(w.Ptr, a, c, w.Out, w.In, n, _pool);
                    break;
                default:
                    MatMul.GemmF32((float*)w.Ptr, a, c, w.Out, w.In, n, _pool);
                    break;
            }
        }
    }

    private void Linear(Mat w, float* bias, Span<float> input, Span<float> output, int n)
    {
        GemmMat(w, input, output, n);
        AddBias(output, bias, n, w.Out);
    }

    private void AddBias(Span<float> data, float* bias, int rows, int cols)
    {
        var job = new Job { A = bias, C = (float*)Unsafe.AsPointer(ref data[0]), I0 = cols, I1 = rows };
        Run(rows, &BiasWorker, &job);
    }

    private static void BiasWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        for (int r = s; r < e; r++)
            TensorPrimitives.Add(new Span<float>(j.C + (long)r * j.I0, j.I0), new ReadOnlySpan<float>(j.A, j.I0), new Span<float>(j.C + (long)r * j.I0, j.I0));
    }

    private void LayerNorm(Span<float> input, Span<float> output, float* w, float* b, int rows)
    {
        var job = new Job { A = (float*)Unsafe.AsPointer(ref input[0]), B = w, D = b, C = (float*)Unsafe.AsPointer(ref output[0]), I0 = _hidden, I1 = rows, F0 = _eps };
        Run(rows, &LayerNormWorker, &job);
    }

    private static void LayerNormWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        int n = j.I0;
        for (int r = s; r < e; r++)
        {
            var x = new ReadOnlySpan<float>(j.A + (long)r * n, n);
            var y = new Span<float>(j.C + (long)r * n, n);
            double sum = 0;
            for (int i = 0; i < n; i++) sum += x[i];
            float mean = (float)(sum / n);
            double sq = 0;
            for (int i = 0; i < n; i++) { float d = x[i] - mean; sq += d * d; }
            float scale = 1.0f / MathF.Sqrt((float)(sq / n) + j.F0);
            for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * scale * j.B[i] + j.D[i];
        }
    }

    private void Gelu(Span<float> data, int count)
    {
        int mode = Mutant == QwenVlVisionMutant.GeluErf ? 2 : ExactGelu ? 1 : 0;
        var job = new Job { C = (float*)Unsafe.AsPointer(ref data[0]), I0 = count, I1 = mode };
        Run((count + 4095) / 4096, &GeluWorker, &job);
    }

    private static void GeluWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        int chunks = (j.I0 + 4095) / 4096;
        Part(chunks, ti, tn, out int s, out int e);
        for (int c = s; c < e; c++)
        {
            int a = c * 4096, z = Math.Min(j.I0, a + 4096);
            for (int i = a; i < z; i++)
            {
                float x = j.C[i];
                j.C[i] = j.I1 switch { 2 => GeluErfScalar(x), 1 => GeluTanhScalar(x), _ => GeluTanhLut(x) };
            }
        }
    }

    /// <summary>Tanh-approximate GELU (<c>ggml_gelu</c> in exact arithmetic).</summary>
    internal static float GeluTanhScalar(float x) => 0.5f * x * (1.0f + MathF.Tanh(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));

    /// <summary>
    /// <c>ggml_vec_gelu_f32</c> of the CPU backend: values at or below -10 give 0, at or above 10 give x, and in between the F16-rounded input indexes a table of
    /// F16 GELU values (so the result is <c>f16(gelu(f16(x)))</c>).
    /// </summary>
    internal static float GeluTanhLut(float x)
    {
        if (x <= -10.0f) return 0.0f;
        if (x >= 10.0f) return x;
        float xr = (float)(Half)x;
        return (float)(Half)GeluTanhScalar(xr);
    }

    /// <summary>Exact (erf) GELU (the wrong variant for this checkpoint; test seam).</summary>
    internal static float GeluErfScalar(float x) => 0.5f * x * (1.0f + (float)Erf(x * 0.70710678118654752440));

    private static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * ax);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-ax * ax);
        return x < 0 ? -y : y;
    }

    /// <summary>ggml upscale, BILINEAR | ALIGN_CORNERS, from the 48x48 table to (width, height); destination layout <c>[y * width + x][E]</c>.</summary>
    private void InterpolatePosemb(Span<float> dst, int width, int height)
    {
        bool half = Mutant == QwenVlVisionMutant.PosembHalfPixel;
        const int ns = GridPos;
        float sf0 = (float)width / ns, sf1 = (float)height / ns;
        float off = 0.5f;
        if (!half)
        {
            off = 0.0f;
            if (width > 1) sf0 = (float)(width - 1) / (ns - 1);
            if (height > 1) sf1 = (float)(height - 1) / (ns - 1);
        }
        for (int y = 0; y < height; y++)
        {
            float fy = (y + off) / sf1 - off;
            int y0 = (int)MathF.Floor(fy), y1 = y0 + 1;
            y0 = Math.Clamp(y0, 0, ns - 1); y1 = Math.Clamp(y1, 0, ns - 1);
            float dy = Math.Clamp(fy - y0, 0f, 1f);
            for (int xx = 0; xx < width; xx++)
            {
                float fx = (xx + off) / sf0 - off;
                int x0 = (int)MathF.Floor(fx), x1 = x0 + 1;
                x0 = Math.Clamp(x0, 0, ns - 1); x1 = Math.Clamp(x1, 0, ns - 1);
                float dx = Math.Clamp(fx - x0, 0f, 1f);
                var row = dst.Slice((y * width + xx) * _hidden, _hidden);
                long i00 = ((long)y0 * ns + x0) * _hidden, i10 = ((long)y0 * ns + x1) * _hidden, i01 = ((long)y1 * ns + x0) * _hidden, i11 = ((long)y1 * ns + x1) * _hidden;
                float w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy), w01 = (1 - dx) * dy, w11 = dx * dy;
                for (int c = 0; c < _hidden; c++)
                    row[c] = PosAt(i00 + c) * w00 + PosAt(i10 + c) * w10 + PosAt(i01 + c) * w01 + PosAt(i11 + c) * w11;
            }
        }
    }

    /// <summary>cos/sin of <c>pos * theta_scale^i</c>, i in [0, 18), theta_scale = 10000^(-2/36), for the row (y) and column (x) positions of every token (ggml <c>rope_cache_init</c>: repeated multiplication).</summary>
    private void BuildRopeTables(Span<float> cy, Span<float> sy, Span<float> cx, Span<float> sx, Span<int> tokX, Span<int> tokY)
    {
        float thetaScale = MathF.Pow(10000.0f, -2.0f / 36);
        bool swap = Mutant == QwenVlVisionMutant.RopeAxesSwapped;
        for (int t = 0; t < cy.Length / 18; t++)
        {
            float ty = swap ? tokX[t] : tokY[t], tx = swap ? tokY[t] : tokX[t];
            for (int i = 0; i < 18; i++)
            {
                cy[t * 18 + i] = MathF.Cos(ty); sy[t * 18 + i] = MathF.Sin(ty);
                cx[t * 18 + i] = MathF.Cos(tx); sx[t * 18 + i] = MathF.Sin(tx);
                ty *= thetaScale; tx *= thetaScale;
            }
        }
    }

    private void Rope(Span<float> data, Span<float> cy, Span<float> sy, Span<float> cx, Span<float> sx, int n)
    {
        var job = new Job
        {
            C = (float*)Unsafe.AsPointer(ref data[0]),
            A = (float*)Unsafe.AsPointer(ref cy[0]), B = (float*)Unsafe.AsPointer(ref sy[0]),
            D = (float*)Unsafe.AsPointer(ref cx[0]), E = (float*)Unsafe.AsPointer(ref sx[0]),
            I0 = _heads, I1 = n, I2 = Mutant == QwenVlVisionMutant.RopeAdjacentPairs ? 1 : 0,
        };
        Run(n, &RopeWorker, &job);
    }

    private static void RopeWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        int heads = j.I0;
        for (int t = s; t < e; t++)
            for (int h = 0; h < heads; h++)
            {
                float* v = j.C + ((long)t * heads + h) * 72;
                for (int ic = 0; ic < 36; ic++)
                {
                    float c = ic < 18 ? j.A[t * 18 + ic] : j.D[t * 18 + ic - 18];
                    float sn = ic < 18 ? j.B[t * 18 + ic] : j.E[t * 18 + ic - 18];
                    int i0 = j.I2 == 1 ? 2 * ic : ic, i1 = j.I2 == 1 ? 2 * ic + 1 : ic + 36;
                    float x0 = v[i0], x1 = v[i1];
                    v[i0] = x0 * c - x1 * sn;
                    v[i1] = x0 * sn + x1 * c;
                }
            }
    }

    /// <summary>Full (non-causal) attention, scale 1/sqrt(72); per (head, 16-query block) work item with a native per-worker score scratch.</summary>
    private void Attention(Span<float> q, Span<float> k, Span<float> v, Span<float> o, int npos)
    {
        const int QB = 16;
        int blocks = (npos + QB - 1) / QB;
        int workers = _pool?.ThreadCount ?? 1;
        var scratch = _arena.Alloc<float>(workers * QB * npos);
        var job = new Job
        {
            A = (float*)Unsafe.AsPointer(ref q[0]), B = (float*)Unsafe.AsPointer(ref k[0]), D = (float*)Unsafe.AsPointer(ref v[0]), C = (float*)Unsafe.AsPointer(ref o[0]),
            I0 = _heads, I1 = npos, I2 = blocks,
            E = (float*)Unsafe.AsPointer(ref scratch[0]),
            F0 = Mutant == QwenVlVisionMutant.WrongAttnScale ? 0.125f : 1.0f / MathF.Sqrt(72.0f),
        };
        Run(_heads * blocks, &AttentionWorker, &job);
    }

    private static void AttentionWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        int heads = j.I0, npos = j.I1, blocks = j.I2;
        Part(heads * blocks, ti, tn, out int s, out int e);
        const int QB = 16;
        float* scores = j.E + (long)ti * QB * npos;
        int E = heads * 72;
        for (int item = s; item < e; item++)
        {
            int h = item / blocks, qb = item % blocks;
            int q0 = qb * QB, q1 = Math.Min(npos, q0 + QB);
            for (int qi = q0; qi < q1; qi++)
            {
                var qv = new ReadOnlySpan<float>(j.A + (long)qi * E + h * 72, 72);
                float* sc = scores + (qi - q0) * npos;
                float max = float.NegativeInfinity;
                for (int ki = 0; ki < npos; ki++)
                {
                    float d = TensorPrimitives.Dot(qv, new ReadOnlySpan<float>(j.B + (long)ki * E + h * 72, 72)) * j.F0;
                    sc[ki] = d;
                    if (d > max) max = d;
                }
                float sum = 0;
                for (int ki = 0; ki < npos; ki++) { float p = MathF.Exp(sc[ki] - max); sc[ki] = p; sum += p; }
                float inv = 1.0f / sum;
                var acc = new Span<float>(j.C + (long)qi * E + h * 72, 72);
                acc.Clear();
                for (int ki = 0; ki < npos; ki++)
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(j.D + (long)ki * E + h * 72, 72), sc[ki] * inv, acc, acc);
            }
        }
    }

    private static void Im2colWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        int W = j.I0;
        bool exact = j.I4 == 1;
        int* tx = (int*)j.I5; int* ty = (int*)j.I6;
        int total = ((int)(j.I1 / P) * (W / P));
        Part(total, ti, tn, out int s, out int e);
        for (int t = s; t < e; t++)
        {
            int px = tx[t], py = ty[t];
            float* dst = j.C + (long)t * 3 * P * P;
            for (int c = 0; c < 3; c++)
                for (int ky = 0; ky < P; ky++)
                    for (int kx = 0; kx < P; kx++)
                    {
                        float v = j.A[((long)(py * P + ky) * W + px * P + kx) * 3 + c];
                        dst[(c * P + ky) * P + kx] = exact ? v : (float)(Half)v;
                    }
        }
    }

    // ───────────────────────────── dispatch ─────────────────────────────

    private struct Job
    {
        public float* A, B, C, D, E;
        public int I0, I1, I2, I3, I4;
        public nint I5, I6;
        public float F0;
    }

    private void Run(int items, delegate*<nint, int, int, void> fn, Job* job)
    {
        if (_pool is null || items < 2) fn((nint)job, 0, 1);
        else _pool.Dispatch((nint)job, fn);
    }

    private static void Part(int total, int ti, int tn, out int start, out int end)
    {
        int b = total / tn, r = total % tn;
        start = ti * b + Math.Min(ti, r);
        end = start + b + (ti < r ? 1 : 0);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _pool?.Dispose();
        _arena.Dispose();
        if (_patchFold != 0) NativeMemory.AlignedFree((void*)_patchFold);
        if (_embedPrecision != VisionEmbedPrecision.F32 && _posConv != 0) NativeMemory.AlignedFree((void*)_posConv);
        if (_ownsGguf) _gguf.Dispose();
    }
}
