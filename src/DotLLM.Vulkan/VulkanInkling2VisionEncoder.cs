using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using DotLLM.Models.Vision;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan vision tower + projector of Inkling-2 (<c>mmproj-Inkling-2-Mini-Q8_0.gguf</c>, projector <c>inkling2v</c>; issue #2062 part B). Same graph and parity details as the CPU
/// <see cref="Inkling2VisionEncoder"/> (docs/INKLING2.md section 12): patch conv through F16-rounded pixels, bilinear position embeddings, CLS, pre-LN, 23 x (LN, QKV+bias, 2D RoPE, full
/// attention, o-proj, LayerScale, residual, LN, up, erf-GELU, down, LayerScale, residual), 2x2 merge, mm.1, erf-GELU, mm.2.
/// </summary>
/// <remarks>
/// <para>Precision (user rulings): BF16 activations (rounded once per store; F32 only in registers and shared memory: GEMM accumulators, LayerNorm statistics, the flash attention softmax),
/// Q8_0 matrices read in place (imported zero-copy / DirectStorage through <see cref="VulkanWeightImportPolicy"/>, decoded inside the GEMM tile load), <c>patch_embd</c> / <c>position_embd</c>
/// BF16 by default (<see cref="VisionEmbedPrecision"/>, <c>DOTLLM_VISION_EMBD_PRECISION</c>), biases / LayerNorm gains / LayerScale vectors F32 as in the file. The RoPE cos/sin table is F16
/// (<c>npos x 64</c> halves), the image pixels are held as F16 (as the reference's F16 im2col rounds them anyway). The result is the F16 <see cref="ImageEmbeddings"/>. Attention never stores a score matrix.</para>
/// <para>Fusions: patch GEMM + position interpolation + CLS row in one kernel pair; RoPE in the Q/K GEMM epilogue; LayerScale + residual add in the o-proj / down GEMM epilogue; GELU in the up GEMM
/// epilogue; the 2x2 merge is the mm.1 GEMM's activation addressing; the final GEMM stores F16.</para>
/// </remarks>
public sealed unsafe class VulkanInkling2VisionEncoder : IImageEncoder
{
    private sealed class Mat { public required VulkanDevice.Buffer Buf; public required int In, Out; }

    private sealed class Block
    {
        public required VulkanDevice.Buffer Ln1W, Ln1B, Ln2W, Ln2B, Ls1, Ls2, QB, KB, VB, OB, UpB, DownB;
        public required Mat Q, K, V, O, Up, Down;
    }

    private const int E = 1024, Heads = 16, Ff = 4096, P = Inkling2ImagePreprocessor.PatchSize;

    private readonly VulkanDevice _device;
    private readonly string _spvDir;
    private readonly string? _spvOverride;
    private readonly GgufFile _gguf;
    private readonly bool _ownsGguf;
    private readonly Block[] _blocks;
    private readonly VulkanDevice.Buffer _patchW, _posW, _cls, _clsPos, _preLnW, _preLnB, _dummy;
    private readonly Mat _mm1, _mm2;
    private readonly int _nLayer, _outDim, _ns;
    private readonly float _eps;
    private readonly ImageOverLimit _overLimit;
    private readonly VisionEmbedPrecision _embedPrecision;
    private readonly List<VulkanDevice.Buffer> _owned = new();
    private readonly Dictionary<string, OlalaKernel> _kernels = new();
    private readonly VulkanDevice.SubmitContext _submit;
    private readonly object _lock = new();
    private nint _cmd;
    private long _weightBytes;
    private bool _recorded;

    // per-call scratch, grown on demand and trimmed after a call above ScratchKeepBytes
    private readonly List<VulkanDevice.Buffer> _scratch = new();
    private VulkanDevice.Buffer _pix = null!, _rope = null!, _x = null!, _xn = null!, _q = null!, _k = null!, _v = null!, _att = null!, _hid = null!, _out = null!;
    private long _pixCap, _npos, _outCap;

    /// <summary>
    /// Storage format of the tower's activations (the residual stream, norm outputs, Q/K/V, attention output, the up-projection). Default <see cref="Inkling2ActFormat.F16"/> since #2075 (user ruling
    /// 2026-11-28, after the activation-precision study, docs/INKLING2.md 12c); env <c>DOTLLM_VISION_ACT=bf16|f16|f32</c> (BF16 = the previous default, F32 = measurement arm). Accumulation stays F32 in every
    /// arm. Set before encoding; changing it drops the scratch. A non-finite F16 result (overflow, never seen: the residual stream peaks at 49 against 65,504) re-encodes that image in BF16, see <see cref="Bf16Fallbacks"/>.
    /// </summary>
    public Inkling2ActFormat ActFormat
    {
        get => _act;
        set { if (value == _act) return; lock (_lock) { ReleaseScratch(); _act = value; } }
    }

    /// <summary>MEASUREMENT arm (env <c>DOTLLM_VISION_ROPE_F32=1</c>): the RoPE cos/sin table in F32 instead of F16 (isolates the table's contribution to the error). Set before encoding.</summary>
    public bool RopeF32
    {
        get => _ropeF32;
        set { if (value == _ropeF32) return; lock (_lock) { ReleaseScratch(); _ropeF32 = value; foreach (var t in _ropeTables.Values) NativeMemory.AlignedFree((void*)t); _ropeTables.Clear(); _ropeOrder.Clear(); } }
    }

    private Inkling2ActFormat _act = Inkling2ActFormat.F16;

    /// <summary>Test seam: runs on the downloaded embeddings just before the finite check (poisons them to prove the guard's BF16 fallback).</summary>
    internal Action<Span<Half>>? AfterDownload { get; set; }

    /// <summary>Number of encodes whose F16 result was non-finite and was recomputed in BF16 (the runtime guard; 0 in every measured run).</summary>
    public int Bf16Fallbacks { get; private set; }
    private bool _ropeF32;
    private int ActBytes => _act == Inkling2ActFormat.F32 ? 4 : 2;

    /// <summary>Scratch kept resident after a call (default 128 MB); a larger working set (a 4,096-token image holds about 500 MB) is released when the call ends.</summary>
    public long ScratchKeepBytes { get; set; } = 128L << 20;

    /// <summary>Device scratch bytes the last call needed at its peak.</summary>
    public long LastPeakScratchBytes { get; private set; }

    /// <summary>Resident weight bytes on the device (Q8_0 matrices count their imported size; the converted embeddings their converted size).</summary>
    public long WeightBytes => _weightBytes;

    /// <summary>Storage format of the patch and position embeddings in force.</summary>
    public VisionEmbedPrecision EmbedPrecision => _embedPrecision;

    /// <summary>Device bytes of the patch + position embeddings in the format in force.</summary>
    public long EmbedBytes { get; private set; }

    /// <summary>Test seam: receives named intermediate activations (BF16 widened to float) using the CPU encoder's checkpoint names. Forces a submit per checkpoint.</summary>
    internal Action<string, float[], int, int>? Trace { get; set; }

    /// <summary>Wall time of the last call's GPU part (upload, all dispatches, download), seconds.</summary>
    public double LastGpuSeconds { get; private set; }

    /// <inheritdoc/>
    public int EmbeddingDimension => _outDim;

    /// <summary>Opens <paramref name="mmprojPath"/> and loads the encoder onto <paramref name="device"/>.</summary>
    public static VulkanInkling2VisionEncoder Load(VulkanDevice device, string mmprojPath, string spvDir, ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
    {
        var gguf = GgufFile.Open(mmprojPath);
        try { return Load(device, gguf, spvDir, ownsGguf: true, overLimit, embedPrecision); }
        catch { gguf.Dispose(); throw; }   // a refused mmproj / device must not leak the mapping (the caller may fall back to the CPU encoder)
    }

    /// <summary>Loads from an opened mmproj GGUF (which must outlive the encoder: Q8_0 weights are imported in place).</summary>
    public static VulkanInkling2VisionEncoder Load(VulkanDevice device, GgufFile gguf, string spvDir, bool ownsGguf = false, ImageOverLimit overLimit = ImageOverLimit.Refuse, VisionEmbedPrecision? embedPrecision = null)
    {
        ArgumentNullException.ThrowIfNull(device); ArgumentNullException.ThrowIfNull(gguf);
        var md = gguf.Metadata;
        if (md.GetStringOrDefault("general.architecture") != "clip" || md.GetStringOrDefault("clip.vision.projector_type") != "inkling2v")
            throw new NotSupportedException("not an inkling2v mmproj (general.architecture clip, clip.vision.projector_type inkling2v)");
        if (md.GetUInt32("clip.vision.patch_size") != P || md.GetUInt32("clip.vision.spatial_merge_size") != Inkling2ImagePreprocessor.Merge)
            throw new NotSupportedException("inkling2v mmproj with a patch size / merge other than 20 / 2 is not supported");
        if (!device.HasSubgroupArithmetic) throw new NotSupportedException("The Inkling 2 Vulkan vision kernels need subgroup arithmetic (VK_KHR_shader_subgroup_arithmetic).");
        if (device.SubgroupSize != 0 && device.SubgroupSize < 16) throw new NotSupportedException($"The Inkling 2 Vulkan vision kernels need a subgroup of at least 16 lanes; this device reports {device.SubgroupSize}.");
        if (!device.NumericFeatures.StorageBuffer16BitAccess) throw new NotSupportedException("The Inkling 2 Vulkan vision kernels need storageBuffer16BitAccess.");
        return new VulkanInkling2VisionEncoder(device, gguf, spvDir, ownsGguf, overLimit, embedPrecision ?? Inkling2VisionEncoder.EmbedPrecisionFromEnvironment());
    }

    private VulkanInkling2VisionEncoder(VulkanDevice device, GgufFile gguf, string spvDir, bool ownsGguf, ImageOverLimit overLimit, VisionEmbedPrecision embedPrecision)
    {
        _device = device; _gguf = gguf; _spvDir = spvDir; _ownsGguf = ownsGguf; _overLimit = overLimit; _embedPrecision = embedPrecision;
        _spvOverride = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");
        string? actEnv = Environment.GetEnvironmentVariable("DOTLLM_VISION_ACT");
        if (!string.IsNullOrEmpty(actEnv))
            _act = actEnv.ToLowerInvariant() switch { "bf16" => Inkling2ActFormat.Bf16, "f16" => Inkling2ActFormat.F16, "f32" => Inkling2ActFormat.F32, _ => throw new ArgumentException($"DOTLLM_VISION_ACT='{actEnv}': expected bf16, f16 or f32.") };
        _ropeF32 = Environment.GetEnvironmentVariable("DOTLLM_VISION_ROPE_F32") == "1";
        var md = gguf.Metadata;
        _nLayer = (int)md.GetUInt32("clip.vision.block_count");
        int hidden = (int)md.GetUInt32("clip.vision.embedding_length"), heads = (int)md.GetUInt32("clip.vision.attention.head_count"), ff = (int)md.GetUInt32("clip.vision.feed_forward_length");
        _outDim = (int)md.GetUInt32("clip.vision.projection_dim");
        _eps = md.GetFloat32("clip.vision.attention.layer_norm_epsilon");
        if (hidden != E || heads != Heads || ff != Ff || hidden / heads != 64) throw new NotSupportedException("unexpected inkling2v vision geometry");
        var t = gguf.TensorsByName;
        if (t.ContainsKey("v.post_ln.weight")) throw new NotSupportedException("a post_ln is not part of the inkling2v graph");
        if (t.ContainsKey("mm.1.bias") || t.ContainsKey("mm.2.bias")) throw new NotSupportedException("projector biases are not part of this checkpoint's contract");
        _ns = (int)Math.Sqrt(t["v.position_embd.weight"].Shape[1]);

        _dummy = Own(device.AllocateDeviceLocal(256));
        _submit = device.CreateSubmitContext();
        try
        {
            using var staging = VulkanStagingBuffer.Create(device, 64L * 1024 * 1024);

            VulkanDevice.Buffer F32(string n)
            {
                if (!t.TryGetValue(n, out var d)) throw new InvalidDataException($"mmproj tensor {n} missing");
                if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"{n} is {d.QuantizationType}; the vision contract keeps it F32");
                long cnt = 1; for (int i = 0; i < d.Shape.Rank; i++) cnt *= d.Shape[i];
                var b = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), QuantizationType.F32, 1, checked((int)cnt), out long up);
                _weightBytes += up;
                return Own(b);
            }
            Mat M(string n)
            {
                if (!t.TryGetValue(n, out var d)) throw new InvalidDataException($"mmproj tensor {n} missing");
                if (d.QuantizationType != QuantizationType.Q8_0)
                    throw new NotSupportedException($"{n} is {d.QuantizationType}; the Vulkan vision tower reads Q8_0 matrices in place and refuses to widen anything else (CLAUDE.md precision rule 2).");
                int inDim = (int)d.Shape[0], outDim = (int)d.Shape[1];
                if (inDim % 32 != 0) throw new NotSupportedException($"{n}: K = {inDim} is not a multiple of 32");
                var b = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), QuantizationType.Q8_0, outDim, inDim, out long up);
                _weightBytes += up;
                return new Mat { Buf = Own(b), In = inDim, Out = outDim };
            }
            // patch_embd / position_embd: F32 in the file; F16 / BF16 are converted once through a transient native buffer (the only copy), F32 is read in place.
            VulkanDevice.Buffer Embed(string n)
            {
                var d = t[n];
                if (d.QuantizationType != QuantizationType.F32) throw new InvalidDataException($"{n} is {d.QuantizationType}; the vision contract keeps it F32");
                long cnt = 1; for (int i = 0; i < d.Shape.Rank; i++) cnt *= d.Shape[i];
                if (embedPrecision == VisionEmbedPrecision.F32)
                {
                    var b = VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(device, staging, gguf.TensorDataPointer(d), QuantizationType.F32, 1, checked((int)cnt), out long up);
                    _weightBytes += up; EmbedBytes += up;
                    return Own(b);
                }
                long nb = cnt * 2;
                nint tmp = (nint)NativeMemory.AlignedAlloc((nuint)nb, 64);
                try
                {
                    var src = new ReadOnlySpan<float>((void*)gguf.TensorDataPointer(d), checked((int)cnt));
                    var dst = new Span<ushort>((void*)tmp, checked((int)cnt));
                    if (embedPrecision == VisionEmbedPrecision.F16) for (int i = 0; i < dst.Length; i++) dst[i] = BitConverter.HalfToUInt16Bits((Half)src[i]);
                    else for (int i = 0; i < dst.Length; i++) dst[i] = Bf16(src[i]);
                    var buf = device.AllocateDeviceLocal(nb);
                    staging.UploadBytes(tmp, nb, buf);
                    _weightBytes += nb; EmbedBytes += nb;
                    return Own(buf);
                }
                finally { NativeMemory.AlignedFree((void*)tmp); }
            }

            _patchW = Embed("v.patch_embd.weight"); _posW = Embed("v.position_embd.weight");
            _cls = F32("v.class_embd"); _clsPos = F32("v.class_position_embd.weight");
            _preLnW = F32("v.pre_ln.weight"); _preLnB = F32("v.pre_ln.bias");
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
            if (_mm1.In != 4 * E || _mm1.Out != 4096 || _mm2.In != 4096 || _mm2.Out != _outDim) throw new NotSupportedException("unexpected inkling2v projector geometry");
        }
        catch
        {
            // a matrix refused part-way (e.g. an F16 mmproj) must not leak the device buffers already uploaded
            _submit.Dispose();
            foreach (var b in _owned) b.Dispose();
            _owned.Clear();
            throw;
        }
    }

    private VulkanDevice.Buffer Own(VulkanDevice.Buffer b) { _owned.Add(b); return b; }

    /// <summary>Round-to-nearest-even float to bfloat16 bits.</summary>
    private static ushort Bf16(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if (float.IsNaN(f)) return (ushort)((bits >> 16) | 0x0040);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }

    /// <inheritdoc/>
    public int CountTokens(int width, int height) => Inkling2ImagePreprocessor.CountTokens(width, height, _overLimit);

    // ───────────────────────────── scratch ─────────────────────────────

    private VulkanDevice.Buffer Scratch(long bytes) { var b = _device.AllocateDeviceLocal(Math.Max(256, bytes)); _scratch.Add(b); return b; }

    private void ReleaseScratch()
    {
        foreach (var b in _scratch) b.Dispose();
        _scratch.Clear();
        _pix = _rope = _x = _xn = _q = _k = _v = _att = _hid = _out = null!;
        _npos = 0; _pixCap = 0; _outCap = 0;
        _ropeOnDevice = (0, 0);
    }

    // RoPE tables depend only on the patch grid width and the position count: built once per grid into native memory (npos x 128 bytes, F16) and kept (a handful of grids, evicted
    // oldest-first), and not re-uploaded while the device scratch still holds the same grid. No managed array per call.
    private readonly Dictionary<(int Gw, int Npos), nint> _ropeTables = new();
    private readonly Queue<(int Gw, int Npos)> _ropeOrder = new();
    private (int Gw, int Npos) _ropeOnDevice;
    private const int MaxCachedRopeGrids = 8;

    private void UploadRopeTable(int gw, int npos)
    {
        if (_ropeOnDevice == (gw, npos)) return;
        if (!_ropeTables.TryGetValue((gw, npos), out nint tab))
        {
            if (_ropeTables.Count >= MaxCachedRopeGrids)
            {
                var old = _ropeOrder.Dequeue();
                NativeMemory.AlignedFree((void*)_ropeTables[old]);
                _ropeTables.Remove(old);
            }
            tab = (nint)NativeMemory.AlignedAlloc((nuint)((long)npos * (_ropeF32 ? 256 : 128)), 64);
            if (_ropeF32) FillRopeTable(new Span<float>((void*)tab, npos * 64), gw, npos);
            else FillRopeTable(new Span<ushort>((void*)tab, npos * 64), gw, npos);
            _ropeTables[(gw, npos)] = tab; _ropeOrder.Enqueue((gw, npos));
        }
        _device.Upload(new ReadOnlySpan<byte>((void*)tab, npos * (_ropeF32 ? 256 : 128)), _rope);
        _ropeOnDevice = (gw, npos);
    }

    /// <summary>Number of RoPE tables currently cached (test seam).</summary>
    internal int CachedRopeTables => _ropeTables.Count;

    private long ScratchBytes() => _scratch.Sum(b => b.Size);

    private void EnsureScratch(long npos, long pixBytes, long outBytes)
    {
        if (_npos >= npos && _pixCap >= pixBytes && _outCap >= outBytes && _x is not null) return;
        ReleaseScratch();
        long rowsB = npos * E * ActBytes;
        _x = Scratch(rowsB); _xn = Scratch(rowsB); _q = Scratch(rowsB); _k = Scratch(rowsB); _v = Scratch(rowsB); _att = Scratch(rowsB);
        _hid = Scratch(npos * Ff * ActBytes);          // up-projection output; reused as the projector's mm.1 output (n/4 rows x 4096) at the end
        _rope = Scratch(npos * 64 * (_ropeF32 ? 4 : 2));
        _pix = Scratch(pixBytes);
        _out = Scratch(outBytes);
        _npos = npos; _pixCap = pixBytes; _outCap = outBytes;
        foreach (var k in _kernels.Values) k.InvalidateDescriptorCache();
    }

    // ───────────────────────────── kernels & recording ─────────────────────────────

    private OlalaKernel Kernel(string name, int buffers, int pushWords, params uint[] spec)
    {
        // activation-format twins (gen_vit_wrappers.py): vit_X[_f16act|_f32act]; the GEMM also has the F32-RoPE-table measurement variant (_rope32)
        string suffix = _act == Inkling2ActFormat.F16 ? "_f16act" : _act == Inkling2ActFormat.F32 ? "_f32act" : "";
        name += suffix + (_ropeF32 && name == "vit_gemm_q8" ? "_rope32" : "");
        string key = name + "|" + string.Join(",", spec);
        if (!_kernels.TryGetValue(key, out var k))
        {
            string dir = _spvDir;
            if (!string.IsNullOrEmpty(_spvOverride) && File.Exists(Path.Combine(_spvOverride, name + ".spv"))) dir = _spvOverride;
            _kernels[key] = k = new OlalaKernel(_device, dir, name, buffers, pushWords, spec);
        }
        return k;
    }

    private void Run(OlalaKernel k, ReadOnlySpan<uint> push, uint gx, uint gy, uint gz, params VulkanDevice.Buffer[] bufs)
    {
        if (_recorded) KernelSupport.ComputeToComputeBarrier(_cmd);
        _recorded = true;
        k.Record(_cmd, bufs, push, gx, gy, gz);
    }

    private void Begin()
    {
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(_cmd);
        _recorded = false;
    }

    private void Flush()
    {
        _submit.SubmitAndWait();
        Begin();
    }

    private static uint F(float f) => BitConverter.SingleToUInt32Bits(f);

    /// <summary><c>Y = epi(W X)</c>: <paramref name="x"/> [T][K] -> <paramref name="y"/> [T][W.Out].</summary>
    private void Gemm(Mat w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int T, uint epi, VulkanDevice.Buffer? bias = null, VulkanDevice.Buffer? ls = null, bool merge = false, int mw = 0, int gw = 0, int posOff = 0)
    {
        var k = Kernel("vit_gemm_q8", 7, 7, epi, merge ? 1u : 0u);
        uint gx = (uint)((w.Out + 63) / 64), gy = (uint)((T + 63) / 64);
        Run(k, [(uint)w.Out, (uint)w.In, (uint)T, (uint)w.In, (uint)mw, (uint)gw, (uint)posOff], gx, gy, 1,
            w.Buf, x, epi == 5 ? _dummy : y, bias ?? _dummy, ls ?? _dummy, epi == 3 ? _rope : _dummy, epi == 5 ? y : _dummy);
    }

    private void LayerNorm(VulkanDevice.Buffer x, VulkanDevice.Buffer gamma, VulkanDevice.Buffer beta, VulkanDevice.Buffer y, int rows)
        => Run(Kernel("vit_layernorm", 4, 2), [(uint)rows, F(_eps)], (uint)rows, 1, 1, x, gamma, beta, y);

    private float[] DownloadBf16(VulkanDevice.Buffer b, int rows, int cols)
    {
        long count = (long)rows * cols;
        var res = new float[count];
        if (_act == Inkling2ActFormat.F32) { _device.Download(b, res); return res; }
        var raw = new float[(count + 1) / 2];
        _device.Download(b, raw);
        var u = MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
        if (_act == Inkling2ActFormat.F16) for (long i = 0; i < count; i++) res[i] = (float)BitConverter.UInt16BitsToHalf(u[(int)i]);
        else for (long i = 0; i < count; i++) res[i] = BitConverter.UInt32BitsToSingle((uint)u[(int)i] << 16);
        return res;
    }

    private void TraceBuf(string name, VulkanDevice.Buffer b, int rows, int cols)
    {
        if (Trace is null) return;
        Flush();
        Trace(name, DownloadBf16(b, rows, cols), rows, cols);
    }

    // ───────────────────────────── encode ─────────────────────────────

    /// <inheritdoc/>
    public ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        lock (_lock)
        {
            var (w, h, _, _) = Inkling2ImagePreprocessor.PlanSize(image.Width, image.Height, _overLimit);
            // Normalised pixels -> F16 (the reference's im2col rounds them to F16), through a transient native F32 buffer.
            long px = (long)w * h * 3;
            nint f32 = (nint)NativeMemory.AlignedAlloc((nuint)(px * 4), 64);
            nint f16 = (nint)NativeMemory.AlignedAlloc((nuint)(px * 2), 64);
            try
            {
                Inkling2ImagePreprocessor.Normalize(image, new Span<float>((void*)f32, checked((int)px)), overLimit: _overLimit);
                var src = new ReadOnlySpan<float>((void*)f32, checked((int)px));
                var dst = new Span<Half>((void*)f16, checked((int)px));
                for (int i = 0; i < src.Length; i++) dst[i] = (Half)src[i];
                return EncodeHalfPixels(new ReadOnlySpan<byte>((void*)f16, checked((int)(px * 2))), w, h, image.ContentHash, cancellationToken);
            }
            finally { NativeMemory.AlignedFree((void*)f32); NativeMemory.AlignedFree((void*)f16); }
        }
    }

    /// <summary>Encodes an already-normalised, F16-rounded HWC image whose sides are multiples of 40 (what the CPU encoder's <c>EncodeNormalized</c> takes, rounded).</summary>
    internal ImageEmbeddings EncodeHalfPixels(ReadOnlySpan<byte> halfPixels, int W, int H, string hash, CancellationToken ct = default)
    {
        lock (_lock)
        {
            if (W % (P * 2) != 0 || H % (P * 2) != 0) throw new ArgumentException("image sides must be multiples of 40");
            int gw = W / P, gh = H / P, n = gw * gh, npos = n + 1;
            int mw = gw / 2, mh = gh / 2, nm = mw * mh;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                EnsureScratch(npos, (long)W * H * 3 * 2, (long)nm * _outDim * 2);
                LastPeakScratchBytes = ScratchBytes();
                var result = new ImageEmbeddings(mw, mh, _outDim, hash);
                try
                {
                    _device.Upload(halfPixels, _pix);
                    UploadRopeTable(gw, npos);
                    Begin();
                    RecordTower(W, gw, gh, n, npos, nm, mw, ct);
                    _submit.SubmitAndWait();
                    _device.Download(_out, MemoryMarshal.Cast<Half, float>(result.Rows));
                    AfterDownload?.Invoke(result.Rows);
                    if (_act == Inkling2ActFormat.F16 && !AllFinite(result.Rows))
                    {
                        // Runtime guard (#2075): the tower is stateless, so a non-finite F16 result is recomputed in BF16 (full F32 exponent range) instead of returning NaN embeddings.
                        Bf16Fallbacks++;
                        _act = Inkling2ActFormat.Bf16;
                        try
                        {
                            Begin();
                            RecordTower(W, gw, gh, n, npos, nm, mw, ct);
                            _submit.SubmitAndWait();
                            _device.Download(_out, MemoryMarshal.Cast<Half, float>(result.Rows));
                        }
                        finally { _act = Inkling2ActFormat.F16; }
                    }
                    return result;
                }
                catch { result.Dispose(); throw; }
            }
            finally
            {
                LastGpuSeconds = sw.Elapsed.TotalSeconds;
                if (ScratchBytes() > ScratchKeepBytes) ReleaseScratch();
            }
        }
    }

    /// <summary>True when every element is finite (the guard's check: one pass over the <c>[n/4][2304]</c> F16 output).</summary>
    internal static bool AllFinite(ReadOnlySpan<Half> rows)
    {
        foreach (var h in rows) if (!Half.IsFinite(h)) return false;
        return true;
    }

    private void RecordTower(int W, int gw, int gh, int n, int npos, int nm, int mw, CancellationToken ct)
    {
        // 1. CLS row + patch embedding + position embedding -> x
        Run(Kernel("vit_cls", 3, 0), [], (uint)((E + 255) / 256), 1, 1, _cls, _clsPos, _x);
        uint wt = _embedPrecision switch { VisionEmbedPrecision.F32 => 0u, VisionEmbedPrecision.F16 => 1u, _ => 2u };
        Run(Kernel("vit_patch_embd", 4, 6, wt), [(uint)gw, (uint)gh, (uint)W, 1u, (uint)_ns, (uint)n], (uint)(E / 64), (uint)((n + 63) / 64), 1, _pix, _patchW, _posW, _x);
        TraceBuf("embeddings", _x, npos, E);

        // 2. pre-LN (the result is the residual stream: ping-pong the two row buffers)
        var x = _x; var xn = _xn;
        LayerNorm(x, _preLnW, _preLnB, xn, npos);
        (x, xn) = (xn, x);
        TraceBuf("pre_ln", x, npos, E);

        bool bigSeq = npos > 2048;
        for (int il = 0; il < _nLayer; il++)
        {
            ct.ThrowIfCancellationRequested();
            var b = _blocks[il];
            LayerNorm(x, b.Ln1W, b.Ln1B, xn, npos);
            Gemm(b.Q, xn, _q, npos, 3, b.QB);
            Gemm(b.K, xn, _k, npos, 3, b.KB);
            Gemm(b.V, xn, _v, npos, 0, b.VB);
            if (Trace is not null) { TraceBuf($"q-{il}", _q, npos, E); TraceBuf($"k-{il}", _k, npos, E); TraceBuf($"v-{il}", _v, npos, E); }
            // attention in query chunks so a long sequence is several short submits (GPU watchdog)
            int qChunk = bigSeq ? 2048 : ((npos + 15) / 16) * 16;
            var attn = Kernel("vit_attn", 4, 2);
            for (int q0 = 0; q0 < npos; q0 += qChunk)
            {
                int blocks = (Math.Min(qChunk, npos - q0) + 15) / 16;
                Run(attn, [(uint)npos, (uint)q0], (uint)blocks, Heads, 1, _q, _k, _v, _att);
                if (bigSeq) Flush();
            }
            if (Trace is not null) TraceBuf($"att-{il}", _att, npos, E);
            Gemm(b.O, _att, x, npos, 2, b.OB, b.Ls1);
            LayerNorm(x, b.Ln2W, b.Ln2B, xn, npos);
            Gemm(b.Up, xn, _hid, npos, 1, b.UpB);
            if (Trace is not null) TraceBuf($"hid-{il}", _hid, npos, Ff);
            Gemm(b.Down, _hid, x, npos, 2, b.DownB, b.Ls2);
            if (Trace is not null) TraceBuf($"layer_out-{il}", x, npos, E);
            else if (bigSeq) Flush();
        }
        TraceBuf("encoder_out", x, npos, E);

        // 3. 2x2 merge (the mm.1 activation addressing), mm.1 + erf GELU, mm.2 -> F16 output. The CLS row is the first row of x (posOff 1).
        Gemm(_mm1, x, _hid, nm, 4, merge: true, mw: mw, gw: gw, posOff: 1);
        Gemm(_mm2, _hid, _out, nm, 5);
    }

    /// <summary>Rope table (F16 <c>npos x 64</c>): per token cos_w[16] sin_w[16] cos_h[16] sin_h[16]; the CPU's float angles (repeated multiplication as <c>ggml_rope_cache_init</c>), rounded to F16.</summary>
    internal static ushort[] BuildRopeTable(int gw, int npos)
    {
        var tab = new ushort[(long)npos * 64];
        FillRopeTable(tab, gw, npos);
        return tab;
    }

    private static void FillRopeTable(Span<ushort> tab, int gw, int npos) => FillRopeTable(tab, default, gw, npos);

    private static void FillRopeTable(Span<float> tab, int gw, int npos) => FillRopeTable(default, tab, gw, npos);

    private static void FillRopeTable(Span<ushort> tab, Span<float> tabF, int gw, int npos)
    {
        bool f32 = !tabF.IsEmpty;
        float thetaScale = MathF.Pow(10000.0f, -2.0f / 32);
        for (int t = 0; t < npos; t++)
        {
            int px = 0, py = 0;
            if (t > 0) { int p = t - 1; px = p % gw + 1; py = p / gw + 1; }
            float tw = px, th = py;
            for (int i = 0; i < 16; i++)
            {
                float c0 = MathF.Cos(tw), s0 = MathF.Sin(tw), c1 = MathF.Cos(th), s1 = MathF.Sin(th);
                if (f32) { tabF[t * 64 + i] = c0; tabF[t * 64 + 16 + i] = s0; tabF[t * 64 + 32 + i] = c1; tabF[t * 64 + 48 + i] = s1; }
                else
                {
                    tab[t * 64 + i] = BitConverter.HalfToUInt16Bits((Half)c0); tab[t * 64 + 16 + i] = BitConverter.HalfToUInt16Bits((Half)s0);
                    tab[t * 64 + 32 + i] = BitConverter.HalfToUInt16Bits((Half)c1); tab[t * 64 + 48 + i] = BitConverter.HalfToUInt16Bits((Half)s1);
                }
                tw *= thetaScale; th *= thetaScale;
            }
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (var k in _kernels.Values) k.Dispose();
        _kernels.Clear();
        _submit.Dispose();
        ReleaseScratch();
        foreach (var t in _ropeTables.Values) NativeMemory.AlignedFree((void*)t);
        _ropeTables.Clear(); _ropeOrder.Clear();
        foreach (var b in _owned) b.Dispose();
        _owned.Clear();
        if (_ownsGguf) _gguf.Dispose();
    }
}
