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

/// <summary>Deliberately wrong implementations used to prove the vision gates can fail (CLAUDE.md "Prove the Test Can Fail"). Test-only.</summary>
public enum Inkling2VisionMutant
{
    /// <summary>The correct implementation.</summary>
    None,
    /// <summary>Position-embedding interpolation with <c>align_corners</c> (pixel offset 0) instead of half-pixel centres.</summary>
    PosembAlignCorners,
    /// <summary>Position-embedding grid transposed (x and y swapped) before interpolation.</summary>
    PosembTransposed,
    /// <summary>Row/column RoPE positions swapped (<c>pos_w</c> drives the second half of the head).</summary>
    RopeAxesSwapped,
    /// <summary>RoPE rotates the (i, i + n/2) pairs (NEOX) instead of adjacent pairs.</summary>
    RopeNeox,
    /// <summary>2x2 merge concatenates (x, y) in the wrong order.</summary>
    MergeOrder,
    /// <summary>No 2x2 merge: the first 4096 values of each... (rows are truncated/reshaped without the spatial grouping).</summary>
    MergeSkipped,
    /// <summary>LayerScale (ls1/ls2) not applied.</summary>
    NoLayerScale,
    /// <summary>Tanh-approximate GELU instead of erf GELU.</summary>
    GeluTanh,
    /// <summary>The CLS token is not prepended (patches start at position 0).</summary>
    NoCls,
}

/// <summary>Storage format of the two large F32 tensors of the mmproj that are not quantized: <c>v.patch_embd.weight</c> and <c>v.position_embd.weight</c>.</summary>
public enum VisionEmbedPrecision
{
    /// <summary>As stored in the GGUF (F32, read in place). An A/B arm.</summary>
    F32,
    /// <summary>Converted once at load to IEEE half (the F16 GEMM reads it directly).</summary>
    F16,
    /// <summary>Converted once at load to bfloat16 (round to nearest even; the BF16 GEMM reads it directly). The default.</summary>
    BF16,
}

/// <summary>
/// CPU vision tower + projector of Inkling-2 (<c>mmproj-Inkling-2-Mini-Q8_0.gguf</c>, <c>general.architecture = clip</c>, projector
/// <c>inkling2v</c>; llama.cpp b12780 <c>tools/mtmd/models/inkling2v.cpp</c> and <c>mtmd-graph.cpp</c> <c>build_vit</c>; issue #2035).
/// </summary>
/// <remarks>
/// <para>Graph: patch conv (20x20x3, F32, no bias; the temporal-2 kernel is already folded into one frame) + bilinear-interpolated 32x32 position
/// embeddings, CLS token (class_embd + class_position_embd) prepended, LayerNorm, 23 x (LN, QKV+bias, 2D RoPE on Q/K, full attention, o-proj+bias,
/// LayerScale, residual, LN, up+bias, erf-GELU, down+bias, LayerScale, residual), drop CLS, 2x2 patch merge, mm.1 (no bias), erf-GELU, mm.2.
/// Weights are read in place from the mmap'd GGUF (Q8_0 matrices through the Q8_0 GEMM, never expanded); activations are per-call F32 transients in
/// native scratch (CLAUDE.md: no managed tensor arrays); the stored output is F16 (<see cref="ImageEmbeddings"/>).</para>
/// <para>Parity details kept on purpose: the reference computes the patch conv through an F16 <c>im2col</c>, so the normalised pixels are rounded to
/// F16 before the F32 conv weights see them; position-embedding interpolation is ggml's non-antialiased half-pixel bilinear.</para>
/// </remarks>
public sealed unsafe class Inkling2VisionEncoder : IImageEncoder
{
    private readonly record struct Mat(nint Ptr, QuantizationType Qt, int In, int Out);

    private sealed class Block
    {
        public required nint Ln1W, Ln1B, Ln2W, Ln2B, Ls1, Ls2, QB, KB, VB, OB, UpB, DownB;
        public required Mat Q, K, V, O, Up, Down;
    }

    private readonly GgufFile _gguf;
    private readonly bool _ownsGguf;
    private readonly ComputeThreadPool? _pool;
    private readonly Block[] _blocks;
    private readonly nint _patchW, _cls, _clsPos, _posemb, _preLnW, _preLnB;
    private readonly Mat _mm1, _mm2;
    private readonly int _nLayer, _hidden, _heads, _headDim, _ff, _outDim, _gridPos;
    private readonly float _eps;
    private readonly Inkling2Arena _arena = new();
    private readonly object _lock = new();
    private readonly ImageOverLimit _overLimit;
    private readonly VisionEmbedPrecision _embedPrecision;
    private nint _patchConv, _posConv;   // converted copies (native) when _embedPrecision != F32; the F32 file pointers stay for the sizes

    /// <summary>Environment variable that selects the <see cref="VisionEmbedPrecision"/> when the caller does not (<c>f32</c>, <c>f16</c>, <c>bf16</c>).</summary>
    public const string EmbedPrecisionEnvVar = "DOTLLM_VISION_EMBD_PRECISION";

    /// <summary>
    /// Native scratch kept resident after a call (default 128 MB). A larger working set (a 4,096-token image holds about 1 GB of F32 activations) is
    /// released when the call ends, so one big image does not pin its scratch for the life of the server; the next call regrows it.
    /// </summary>
    internal long ScratchKeepBytes { get; set; } = 128L << 20;

    /// <summary>Native scratch bytes the last call needed at its peak (before <see cref="ScratchKeepBytes"/> released it).</summary>
    internal long LastPeakScratchBytes { get; private set; }

    /// <summary>Storage format of the patch and position embeddings in force.</summary>
    public VisionEmbedPrecision EmbedPrecision => _embedPrecision;

    /// <summary>The default storage of <c>patch_embd</c> / <c>position_embd</c>: BF16 (user ruling 2026-11-27, after the F32 / F16 / BF16 evaluation in docs/INKLING2.md); F32 and F16 are A/B arms.</summary>
    public const VisionEmbedPrecision DefaultEmbedPrecision = VisionEmbedPrecision.BF16;

    /// <summary>Reads <see cref="EmbedPrecisionEnvVar"/>; unset means <see cref="DefaultEmbedPrecision"/> (BF16). A malformed value is an error (a typo must not silently run the other arm).</summary>
    public static VisionEmbedPrecision EmbedPrecisionFromEnvironment(Func<string, string?>? env = null)
    {
        string? text = (env ?? Environment.GetEnvironmentVariable)(EmbedPrecisionEnvVar);
        if (string.IsNullOrWhiteSpace(text)) return DefaultEmbedPrecision;
        return text.Trim().ToLowerInvariant() switch
        {
            "f32" => VisionEmbedPrecision.F32,
            "f16" => VisionEmbedPrecision.F16,
            "bf16" => VisionEmbedPrecision.BF16,
            _ => throw new ArgumentException($"{EmbedPrecisionEnvVar} must be f32, f16 or bf16, got '{text}'."),
        };
    }

    /// <summary>Test seam: deliberately wrong behaviour.</summary>
    internal Inkling2VisionMutant Mutant { get; set; }

    /// <summary>Test seam: receives named intermediate tensors using llama.cpp's checkpoint names.</summary>
    internal Inkling2TraceSink? Trace { get; set; }

    /// <summary>When true the F16 rounding of the im2col input is skipped (parity A/B arm).</summary>
    internal bool SkipIm2colF16Rounding { get; set; }

    /// <inheritdoc/>
    public int EmbeddingDimension => _outDim;

    /// <summary>Opens <paramref name="mmprojPath"/> and loads the encoder.</summary>
    public static Inkling2VisionEncoder Load(string mmprojPath, ThreadingConfig threading,
        ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
        => Load(GgufFile.Open(mmprojPath), threading, ownsGguf: true, overLimit, embedPrecision);

    /// <summary>Loads from an opened mmproj GGUF.</summary>
    public static Inkling2VisionEncoder Load(GgufFile gguf, ThreadingConfig threading, bool ownsGguf = false,
        ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
    {
        var md = gguf.Metadata;
        if (md.GetStringOrDefault("general.architecture") != "clip" || md.GetStringOrDefault("clip.vision.projector_type") != "inkling2v")
            throw new NotSupportedException("not an inkling2v mmproj (general.architecture clip, clip.vision.projector_type inkling2v)");
        if (md.GetUInt32("clip.vision.patch_size") != Inkling2ImagePreprocessor.PatchSize
            || md.GetUInt32("clip.vision.spatial_merge_size") != Inkling2ImagePreprocessor.Merge)
            throw new NotSupportedException("inkling2v mmproj with a patch size / merge other than 20 / 2 is not supported");
        return new Inkling2VisionEncoder(gguf, ownsGguf, threading, overLimit, embedPrecision ?? EmbedPrecisionFromEnvironment());
    }

    private Inkling2VisionEncoder(GgufFile gguf, bool ownsGguf, ThreadingConfig threading, ImageOverLimit overLimit, VisionEmbedPrecision embedPrecision)
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
        if (_headDim != 64 || _hidden != 1024) throw new NotSupportedException("unexpected inkling2v vision geometry");
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
            if (d.QuantizationType is not (QuantizationType.Q8_0 or QuantizationType.F32))
                throw new NotSupportedException($"{n} is {d.QuantizationType}; only Q8_0 / F32 vision matrices are implemented");
            return new Mat(gguf.TensorDataPointer(d), d.QuantizationType, (int)d.Shape[0], (int)d.Shape[1]);
        }

        _patchW = F32("v.patch_embd.weight");
        _cls = F32("v.class_embd"); _clsPos = F32("v.class_position_embd.weight");
        _posemb = F32("v.position_embd.weight");
        _gridPos = (int)Math.Sqrt(t["v.position_embd.weight"].Shape[1]);
        _preLnW = F32("v.pre_ln.weight"); _preLnB = F32("v.pre_ln.bias");
        if (t.ContainsKey("v.post_ln.weight")) throw new NotSupportedException("a post_ln is not part of the inkling2v graph");
        _blocks = new Block[_nLayer];
        for (int il = 0; il < _nLayer; il++)
        {
            string B(string s) => $"v.blk.{il}.{s}";
            _blocks[il] = new Block
            {
                Ln1W = F32(B("ln1.weight")), Ln1B = F32(B("ln1.bias")), Ln2W = F32(B("ln2.weight")), Ln2B = F32(B("ln2.bias")),
                Ls1 = F32(B("ls1.weight")), Ls2 = F32(B("ls2.weight")),
                QB = F32(B("attn_q.bias")), KB = F32(B("attn_k.bias")), VB = F32(B("attn_v.bias")), OB = F32(B("attn_out.bias")),
                UpB = F32(B("ffn_up.bias")), DownB = F32(B("ffn_down.bias")),
                Q = M(B("attn_q.weight")), K = M(B("attn_k.weight")), V = M(B("attn_v.weight")), O = M(B("attn_out.weight")),
                Up = M(B("ffn_up.weight")), Down = M(B("ffn_down.weight")),
            };
        }
        _mm1 = M("mm.1.weight"); _mm2 = M("mm.2.weight");
        if (t.ContainsKey("mm.1.bias") || t.ContainsKey("mm.2.bias")) throw new NotSupportedException("projector biases are not part of this checkpoint's contract");
        _pool = threading.IsParallel ? new ComputeThreadPool(threading.EffectiveThreadCount) : null;
        if (_embedPrecision != VisionEmbedPrecision.F32)
        {
            // One-time conversion at load (user ruling 2026-11-27): the F16 / BF16 copies replace the F32 tensors in every kernel; the F32
            // file pages are never touched again. 2 bytes per element instead of 4.
            _patchConv = Convert(_patchW, (long)_hidden * 3 * Inkling2ImagePreprocessor.PatchSize * Inkling2ImagePreprocessor.PatchSize, _embedPrecision);
            _posConv = Convert(_posemb, (long)_gridPos * _gridPos * _hidden, _embedPrecision);
        }
    }

    private static nint Convert(nint f32, long count, VisionEmbedPrecision to)
    {
        var dst = (ushort*)NativeMemory.AlignedAlloc((nuint)(count * 2), 64);
        var src = (float*)f32;
        if (to == VisionEmbedPrecision.F16)
            for (long i = 0; i < count; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
        else
            for (long i = 0; i < count; i++) dst[i] = FloatToBf16(src[i]);
        return (nint)dst;
    }

    /// <summary>Round-to-nearest-even float to bfloat16 bits (finite values; NaN keeps a quiet NaN).</summary>
    internal static ushort FloatToBf16(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if (float.IsNaN(f)) return (ushort)((bits >> 16) | 0x0040);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    /// <summary>Element <paramref name="i"/> of the position-embedding table in whatever format it is stored.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private float PosAt(long i) => _embedPrecision switch
    {
        VisionEmbedPrecision.F16 => (float)BitConverter.UInt16BitsToHalf(((ushort*)_posConv)[i]),
        VisionEmbedPrecision.BF16 => BitConverter.UInt32BitsToSingle((uint)((ushort*)_posConv)[i] << 16),
        _ => ((float*)_posemb)[i],
    };

    /// <inheritdoc/>
    public int CountTokens(int width, int height) => Inkling2ImagePreprocessor.CountTokens(width, height, _overLimit);

    /// <inheritdoc/>
    public ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        lock (_lock)
        {
            try
            {
                _arena.Reset();
                var (w, h, _, _) = Inkling2ImagePreprocessor.PlanSize(image.Width, image.Height, _overLimit);
                var hwc = _arena.Alloc<float>(w * h * 3);
                Inkling2ImagePreprocessor.Normalize(image, hwc, swapRgb: SwapRgb, overLimit: _overLimit);
                return RunLocked(hwc, w, h, image.ContentHash, cancellationToken);
            }
            finally { LastPeakScratchBytes = _arena.Capacity; _arena.Trim(ScratchKeepBytes); }
        }
    }

    /// <summary>Test seam: BGR instead of RGB channel order (preprocessing mutant).</summary>
    internal bool SwapRgb { get; set; }

    /// <summary>Encodes an already-normalised interleaved-RGB float image whose sides are multiples of 40 (what <c>llama-mtmd-debug -p encode</c> feeds).</summary>
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
        const int P = Inkling2ImagePreprocessor.PatchSize;
        if (W % (P * 2) != 0 || H % (P * 2) != 0) throw new ArgumentException("image sides must be multiples of 40");
        int gw = W / P, gh = H / P, n = gw * gh, npos = n + 1, E = _hidden;
        bool cls = Mutant != Inkling2VisionMutant.NoCls;
        if (!cls) npos = n;
        int posOff = cls ? 1 : 0;

        // 1. patch embedding: F16-rounded im2col rows [c][ky][kx] x F32 conv weights.
        var patches = _arena.Alloc<float>(n * 3 * P * P);
        fixed (float* hp = hwc) fixed (float* pp = patches)
        {
            var job = new Job { A = hp, C = pp, I0 = W, I1 = H, I2 = gw, I3 = gh, I4 = SkipIm2colF16Rounding ? 1 : 0 };
            Run(n, &Im2colWorker, &job);
        }
        var patchEmb = _arena.Alloc<float>(n * E);
        fixed (float* a = patches) fixed (float* c = patchEmb)
        {
            switch (_embedPrecision)
            {
                case VisionEmbedPrecision.F16: MatMul.GemmF16(_patchConv, a, c, E, 3 * P * P, n, _pool); break;
                case VisionEmbedPrecision.BF16: MatMul.GemmBf16(_patchConv, a, c, E, 3 * P * P, n, _pool); break;
                default: MatMul.GemmF32((float*)_patchW, a, c, E, 3 * P * P, n, _pool); break;
            }
        }
        Trace?.Invoke("patch_embd", patchEmb, n, E);

        // 2. position embeddings (bilinear to the patch grid), CLS, add.
        var x = _arena.Alloc<float>(npos * E);
        Span<float> pos = default;
        bool interp = !(gh == _gridPos && gw == _gridPos);
        if (interp)
        {
            pos = _arena.Alloc<float>(E * n);   // layout [chan][y][x] first (ggml order), then reordered to [y*gw+x][chan]
            InterpolatePosemb(pos, gw, gh);
            Trace?.Invoke("posemb_interp", pos, E * gh, gw);
        }
        var posRows = interp ? _arena.Alloc<float>(n * E) : default;
        if (interp)
        {
            for (int c = 0; c < E; c++)
                for (int p = 0; p < n; p++) posRows[p * E + c] = pos[c * n + p];
        }
        var clsRow = new ReadOnlySpan<float>((void*)_cls, E);
        var clsPos = new ReadOnlySpan<float>((void*)_clsPos, E);
        if (cls)
            for (int c = 0; c < E; c++) x[c] = clsRow[c] + clsPos[c];
        for (int p = 0; p < n; p++)
        {
            var dst = x.Slice((p + posOff) * E, E);
            var src = patchEmb.Slice(p * E, E);
            if (interp)
            {
                var pe = posRows.Slice(p * E, E);
                for (int c = 0; c < E; c++) dst[c] = src[c] + pe[c];
            }
            else
                for (int c = 0; c < E; c++) dst[c] = src[c] + PosAt((long)p * E + c);   // un-interpolated: the table row of this patch
        }
        Trace?.Invoke("embeddings", x, npos, E);

        // 3. pre-LN and the 23 blocks.
        var xn = _arena.Alloc<float>(npos * E);
        LayerNorm(x, xn, (float*)_preLnW, (float*)_preLnB, npos);
        xn.CopyTo(x);
        Trace?.Invoke("pre_ln", x, npos, E);

        var q = _arena.Alloc<float>(npos * E);
        var k = _arena.Alloc<float>(npos * E);
        var v = _arena.Alloc<float>(npos * E);
        var att = _arena.Alloc<float>(npos * E);
        var proj = _arena.Alloc<float>(npos * E);
        if ((long)npos * _ff > int.MaxValue) throw new InvalidOperationException("image too large");
        var hid = _arena.Alloc<float>(npos * _ff);
        var cosw = _arena.Alloc<float>(npos * 16); var sinw = _arena.Alloc<float>(npos * 16);
        var cosh = _arena.Alloc<float>(npos * 16); var sinh = _arena.Alloc<float>(npos * 16);
        BuildRopeTables(cosw, sinw, cosh, sinh, gw, gh, cls);

        for (int il = 0; il < _nLayer; il++)
        {
            ct.ThrowIfCancellationRequested();
            var b = _blocks[il];
            LayerNorm(x, xn, (float*)b.Ln1W, (float*)b.Ln1B, npos);
            Trace?.Invoke($"layer_inp_normed-{il}", xn, npos, E);
            Linear(b.Q, (float*)b.QB, xn, q, npos);
            Linear(b.K, (float*)b.KB, xn, k, npos);
            Linear(b.V, (float*)b.VB, xn, v, npos);
            Trace?.Invoke($"Vcur-{il}", v, npos, E);
            Rope(q, cosw, sinw, cosh, sinh, npos);
            Rope(k, cosw, sinw, cosh, sinh, npos);
            Trace?.Invoke($"Qcur_pos-{il}", q, npos, E);
            Trace?.Invoke($"Kcur_pos-{il}", k, npos, E);
            Attention(q, k, v, att, npos);
            Linear(b.O, (float*)b.OB, att, proj, npos);
            Trace?.Invoke($"attn_out-{il}", proj, npos, E);
            if (Mutant != Inkling2VisionMutant.NoLayerScale) ScaleRows(proj, (float*)b.Ls1, npos);
            TensorPrimitives.Add(proj.Slice(0, npos * E), x.Slice(0, npos * E), x.Slice(0, npos * E));   // x = attn + x
            Trace?.Invoke($"ffn_inp-{il}", x, npos, E);
            LayerNorm(x, xn, (float*)b.Ln2W, (float*)b.Ln2B, npos);
            Linear(b.Up, (float*)b.UpB, xn, hid, npos);
            Gelu(hid, npos * _ff);
            Linear(b.Down, (float*)b.DownB, hid, proj, npos);
            Trace?.Invoke($"ffn_out-{il}", proj, npos, E);
            if (Mutant != Inkling2VisionMutant.NoLayerScale) ScaleRows(proj, (float*)b.Ls2, npos);
            TensorPrimitives.Add(x.Slice(0, npos * E), proj.Slice(0, npos * E), x.Slice(0, npos * E));
            Trace?.Invoke($"layer_out-{il}", x, npos, E);
        }
        Trace?.Invoke("encoder_out", x, npos, E);

        // 4. drop CLS, 2x2 patch merge: token (j, i) = [(2i,2j), (2i+1,2j), (2i,2j+1), (2i+1,2j+1)] (build_patch_merge_permute).
        int mw = gw / 2, mh = gh / 2, nm = mw * mh;
        var merged = _arena.Alloc<float>(nm * 4 * E);
        var feat = x.Slice(posOff * E, n * E);
        if (Mutant == Inkling2VisionMutant.MergeSkipped)
            feat.Slice(0, nm * 4 * E).CopyTo(merged);   // consecutive patches, no 2D grouping
        else
        {
            for (int j = 0; j < mh; j++)
                for (int i = 0; i < mw; i++)
                {
                    var dst = merged.Slice((j * mw + i) * 4 * E, 4 * E);
                    int x0 = 2 * i, y0 = 2 * j;
                    int a0 = y0 * gw + x0, a1 = y0 * gw + x0 + 1, a2 = (y0 + 1) * gw + x0, a3 = (y0 + 1) * gw + x0 + 1;
                    if (Mutant == Inkling2VisionMutant.MergeOrder) (a1, a2) = (a2, a1);
                    feat.Slice(a0 * E, E).CopyTo(dst); feat.Slice(a1 * E, E).CopyTo(dst.Slice(E)); feat.Slice(a2 * E, E).CopyTo(dst.Slice(2 * E)); feat.Slice(a3 * E, E).CopyTo(dst.Slice(3 * E));
                }
        }
        Trace?.Invoke("pixel_shuffle", merged, nm, 4 * E);

        // 5. projector: mm.1 (no bias) -> erf GELU -> mm.2 (no bias).
        var mid = _arena.Alloc<float>(nm * _mm1.Out);
        GemmMat(_mm1, merged, mid, nm);
        Trace?.Invoke("ffn_up", mid, nm, _mm1.Out);
        Gelu(mid, nm * _mm1.Out);
        Trace?.Invoke("ffn_gelu_erf", mid, nm, _mm1.Out);
        var outF = _arena.Alloc<float>(nm * _outDim);
        GemmMat(_mm2, mid, outF, nm);
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
            if (w.Qt == QuantizationType.Q8_0) MatMul.GemmQ8_0((byte*)w.Ptr, a, c, w.Out, w.In, n, _pool, null);
            else MatMul.GemmF32((float*)w.Ptr, a, c, w.Out, w.In, n, _pool);
        }
    }

    private void Linear(Mat w, float* bias, Span<float> input, Span<float> output, int n)
    {
        GemmMat(w, input, output, n);
        int m = w.Out;
        var job = new Job { A = bias, C = (float*)Unsafe.AsPointer(ref output[0]), I0 = m, I1 = n };
        Run(n, &BiasWorker, &job);
    }

    private static void BiasWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        for (int r = s; r < e; r++)
            TensorPrimitives.Add(new Span<float>(j.C + (long)r * j.I0, j.I0), new ReadOnlySpan<float>(j.A, j.I0), new Span<float>(j.C + (long)r * j.I0, j.I0));
    }

    private void ScaleRows(Span<float> data, float* scale, int rows)
    {
        var job = new Job { A = scale, C = (float*)Unsafe.AsPointer(ref data[0]), I0 = _hidden, I1 = rows };
        Run(rows, &ScaleWorker, &job);
    }

    private static void ScaleWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        for (int r = s; r < e; r++)
            TensorPrimitives.Multiply(new Span<float>(j.C + (long)r * j.I0, j.I0), new ReadOnlySpan<float>(j.A, j.I0), new Span<float>(j.C + (long)r * j.I0, j.I0));
    }

    /// <summary><c>ggml_norm</c> + gain + bias: double-precision mean and variance per row, like the reference.</summary>
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
        bool tanh = Mutant == Inkling2VisionMutant.GeluTanh;
        var job = new Job { C = (float*)Unsafe.AsPointer(ref data[0]), I0 = count, I1 = tanh ? 1 : 0 };
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
            if (j.I1 == 1)
                for (int i = a; i < z; i++) { float x = j.C[i]; j.C[i] = 0.5f * x * (1.0f + MathF.Tanh(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x))); }
            else
                for (int i = a; i < z; i++) { float x = j.C[i]; j.C[i] = 0.5f * x * (1.0f + Erf(x * 0.70710678118654752440f)); }
        }
    }

    /// <summary>Exact (erf) GELU of one value, as the layers and the projector apply it (test seam for the op-level gate).</summary>
    internal static float GeluErfScalar(float x) => 0.5f * x * (1.0f + Erf(x * 0.70710678118654752440f));

    /// <summary>Tanh-approximate GELU (the wrong variant; test seam).</summary>
    internal static float GeluTanhScalar(float x) => 0.5f * x * (1.0f + MathF.Tanh(0.79788456080286535588f * x * (1.0f + 0.044715f * x * x)));

    /// <summary>erf, Abramowitz &amp; Stegun 7.1.26: |error| below 1.5e-7, far below the Q8_0 activation noise.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Erf(float x)
    {
        float ax = MathF.Abs(x);
        float t = 1.0f / (1.0f + 0.3275911f * ax);
        float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * MathF.Exp(-ax * ax);
        return x < 0 ? -y : y;
    }

    private void InterpolatePosemb(Span<float> dst, int width, int height)
    {
        // ggml upscale (GGML_SCALE_MODE_BILINEAR, no flags): source (a = width-ish 32, b = 32, embd), dst (width, height, embd).
        int ns = _gridPos;
        bool alignCorners = Mutant == Inkling2VisionMutant.PosembAlignCorners;
        bool transposed = Mutant == Inkling2VisionMutant.PosembTransposed;
        float sf0 = (float)width / ns, sf1 = (float)height / ns;
        float off = 0.5f;
        if (alignCorners)
        {
            off = 0.0f;
            sf0 = width > 1 && ns > 1 ? (float)(width - 1) / (ns - 1) : sf0;
            sf1 = height > 1 && ns > 1 ? (float)(height - 1) / (ns - 1) : sf1;
        }
        // src[(b * ns + a) * E + c]; position index = y * ns + x (PosAt reads the stored format)
        for (int c = 0; c < _hidden; c++)
            for (int y = 0; y < height; y++)
            {
                float fy = (y + off) / sf1 - off;
                int y0 = (int)MathF.Floor(fy), y1 = y0 + 1;
                y0 = Math.Max(0, Math.Min(y0, ns - 1)); y1 = Math.Max(0, Math.Min(y1, ns - 1));
                float dy = Math.Max(0f, Math.Min(fy - y0, 1f));
                for (int xx = 0; xx < width; xx++)
                {
                    float fx = (xx + off) / sf0 - off;
                    int x0 = (int)MathF.Floor(fx), x1 = x0 + 1;
                    x0 = Math.Max(0, Math.Min(x0, ns - 1)); x1 = Math.Max(0, Math.Min(x1, ns - 1));
                    float dx = Math.Max(0f, Math.Min(fx - x0, 1f));
                    
                    int Idx(int px, int py) => transposed ? (px * ns + py) * _hidden + c : (py * ns + px) * _hidden + c;
                    float a = PosAt(Idx(x0, y0)), b = PosAt(Idx(x1, y0)), cc = PosAt(Idx(x0, y1)), d = PosAt(Idx(x1, y1));
                    dst[(c * height + y) * width + xx] = a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + cc * (1 - dx) * dy + d * dx * dy;
                }
            }
    }

    /// <summary>cos/sin of <c>pos * theta_scale^i</c> (float, repeated multiplication as <c>ggml_rope_cache_init</c>), i in [0,16), for pos_w and pos_h.</summary>
    private static void BuildRopeTables(Span<float> cw, Span<float> sw, Span<float> ch, Span<float> sh, int gw, int gh, bool cls)
    {
        float thetaScale = MathF.Pow(10000.0f, -2.0f / 32);
        int off = cls ? 1 : 0;
        for (int t = 0; t < cw.Length / 16; t++)
        {
            int px, py;
            if (cls && t == 0) { px = 0; py = 0; }
            else { int p = t - off; px = p % gw + 1; py = p / gw + 1; }
            float tw = px, th = py;
            for (int i = 0; i < 16; i++)
            {
                cw[t * 16 + i] = MathF.Cos(tw); sw[t * 16 + i] = MathF.Sin(tw);
                ch[t * 16 + i] = MathF.Cos(th); sh[t * 16 + i] = MathF.Sin(th);
                tw *= thetaScale; th *= thetaScale;
            }
        }
    }

    private void Rope(Span<float> data, Span<float> cw, Span<float> sw, Span<float> ch, Span<float> sh, int npos)
    {
        bool swap = Mutant == Inkling2VisionMutant.RopeAxesSwapped, neox = Mutant == Inkling2VisionMutant.RopeNeox;
        var job = new Job
        {
            C = (float*)Unsafe.AsPointer(ref data[0]), A = (float*)Unsafe.AsPointer(ref (swap ? ref ch[0] : ref cw[0])), B = (float*)Unsafe.AsPointer(ref (swap ? ref sh[0] : ref sw[0])),
            D = (float*)Unsafe.AsPointer(ref (swap ? ref cw[0] : ref ch[0])), E = (float*)Unsafe.AsPointer(ref (swap ? ref sw[0] : ref sh[0])),
            I0 = _heads, I1 = npos, I2 = neox ? 1 : 0,
        };
        Run(npos, &RopeWorker, &job);
    }

    private static void RopeWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        Part(j.I1, ti, tn, out int s, out int e);
        int heads = j.I0;
        for (int t = s; t < e; t++)
            for (int h = 0; h < heads; h++)
            {
                float* v = j.C + ((long)t * heads + h) * 64;
                for (int half = 0; half < 2; half++)
                {
                    float* hv = v + half * 32;
                    float* c = (half == 0 ? j.A : j.D) + t * 16;
                    float* sn = (half == 0 ? j.B : j.E) + t * 16;
                    if (j.I2 == 0)
                        for (int i = 0; i < 16; i++)
                        {
                            float x0 = hv[2 * i], x1 = hv[2 * i + 1];
                            hv[2 * i] = x0 * c[i] - x1 * sn[i];
                            hv[2 * i + 1] = x0 * sn[i] + x1 * c[i];
                        }
                    else
                        for (int i = 0; i < 16; i++)
                        {
                            float x0 = hv[i], x1 = hv[i + 16];
                            hv[i] = x0 * c[i] - x1 * sn[i];
                            hv[i + 16] = x0 * sn[i] + x1 * c[i];
                        }
                }
            }
    }

    /// <summary>Full (non-causal) attention, scale 1/sqrt(64); per (head, 16-query block) work item with a native per-worker score scratch.</summary>
    private void Attention(Span<float> q, Span<float> k, Span<float> v, Span<float> o, int npos)
    {
        const int QB = 16;
        int blocks = (npos + QB - 1) / QB;
        int workers = _pool?.ThreadCount ?? 1;
        var scratch = _arena.Alloc<float>(workers * QB * npos);   // per-worker score tiles, native
        var job = new Job
        {
            A = (float*)Unsafe.AsPointer(ref q[0]), B = (float*)Unsafe.AsPointer(ref k[0]), D = (float*)Unsafe.AsPointer(ref v[0]), C = (float*)Unsafe.AsPointer(ref o[0]),
            I0 = _heads, I1 = npos, I2 = blocks,
            E = (float*)Unsafe.AsPointer(ref scratch[0]),
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
        {
            int E = heads * 64;
            for (int item = s; item < e; item++)
            {
                int h = item / blocks, qb = item % blocks;
                int q0 = qb * QB, q1 = Math.Min(npos, q0 + QB);
                for (int qi = q0; qi < q1; qi++)
                {
                    var qv = new ReadOnlySpan<float>(j.A + (long)qi * E + h * 64, 64);
                    float* sc = scores + (qi - q0) * npos;
                    float max = float.NegativeInfinity;
                    for (int ki = 0; ki < npos; ki++)
                    {
                        float d = TensorPrimitives.Dot(qv, new ReadOnlySpan<float>(j.B + (long)ki * E + h * 64, 64)) * 0.125f;
                        sc[ki] = d;
                        if (d > max) max = d;
                    }
                    float sum = 0;
                    for (int ki = 0; ki < npos; ki++) { float p = MathF.Exp(sc[ki] - max); sc[ki] = p; sum += p; }
                    float inv = 1.0f / sum;
                    var acc = new Span<float>(j.C + (long)qi * E + h * 64, 64);
                    acc.Clear();
                    for (int ki = 0; ki < npos; ki++)
                        TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(j.D + (long)ki * E + h * 64, 64), sc[ki] * inv, acc, acc);
                }
            }
        }
    }

    private static void Im2colWorker(nint ctx, int ti, int tn)
    {
        ref var j = ref Unsafe.AsRef<Job>((void*)ctx);
        const int P = 20;
        int W = j.I0, H = j.I1, gw = j.I2, gh = j.I3;
        bool exact = j.I4 == 1;
        Part(gw * gh, ti, tn, out int s, out int e);
        for (int p = s; p < e; p++)
        {
            int px = p % gw, py = p / gw;
            float* dst = j.C + (long)p * 3 * P * P;
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
        if (_patchConv != 0) NativeMemory.AlignedFree((void*)_patchConv);
        if (_posConv != 0) NativeMemory.AlignedFree((void*)_posConv);
        _patchConv = _posConv = 0;
        if (_ownsGguf) _gguf.Dispose();
    }
}
