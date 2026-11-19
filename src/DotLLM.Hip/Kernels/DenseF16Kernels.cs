using DotLLM.Core.Configuration;
using DotLLM.Hip.Interop;

namespace DotLLM.Hip.Kernels;

/// <summary>Single-token F16 activations, direct packed weights and paged KV with register reductions.</summary>
/// <remarks>Device pointers must belong to the supplied context and cover the declared dimensions.</remarks>
public sealed unsafe class DenseF16Kernels : IDisposable
{
    private readonly HipContext _context;
    private readonly HipStream _stream;
    private readonly HipModule _module;
    private readonly nint _embed, _convertF32, _rmsNorm, _linear, _add, _swiGlu, _rope, _writeKv, _attention;
    private readonly nint _embedPacked, _linearPacked, _writeKvPacked, _attentionPacked;
    private bool _disposed;

    /// <summary>Loads the complete dense primitive manifest from <c>dense_f16.co</c>.</summary>
    public DenseF16Kernels(HipContext context, HipStream stream, string? coDirectory = null)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(stream);
        if (context.DeviceId != stream.DeviceId)
            throw new ArgumentException("HIP context and stream must use the same device.", nameof(stream));
        _context = context;
        _stream = stream;
        context.MakeCurrent();
        string path = Path.Combine(coDirectory ?? Path.Combine(AppContext.BaseDirectory, "co"), "dense_f16.co");
        if (!File.Exists(path))
            throw new FileNotFoundException("Build the HIP kernels with native/hip/build.ps1 or build.sh for this GPU before loading the dense backend.", path);
        _module = HipModule.LoadFromFile(path);
        try
        {
            _embed = _module.GetFunction("dense_embed_f16");
            _convertF32 = _module.GetFunction("dense_convert_f32_f16");
            _rmsNorm = _module.GetFunction("dense_rmsnorm_f16");
            _linear = _module.GetFunction("dense_linear_f16");
            _add = _module.GetFunction("dense_add_f16");
            _swiGlu = _module.GetFunction("dense_swiglu_f16");
            _rope = _module.GetFunction("dense_rope_f16");
            _writeKv = _module.GetFunction("dense_write_kv_f16");
            _attention = _module.GetFunction("dense_attention_f16");
            _embedPacked = _module.GetFunction("dense_embed_packed");
            _linearPacked = _module.GetFunction("dense_linear_packed");
            _writeKvPacked = _module.GetFunction("dense_write_kv_packed");
            _attentionPacked = _module.GetFunction("dense_attention_packed");
        }
        catch
        {
            _module.Dispose();
            throw;
        }
    }

    /// <summary>Copies the selected row of a row-major F16 embedding table.</summary>
    public void Embed(nint output, nint weights, int token, int dim)
    {
        Pointer(output); Pointer(weights); Positive(dim);
        ArgumentOutOfRangeException.ThrowIfNegative(token);
        void** args = stackalloc void*[] { &output, &weights, &token, &dim };
        Launch(_embed, Blocks(dim), 256, args);
    }

    /// <summary>Reads one F16/Q4_0/Q8_0 embedding row directly into an F16 activation row.</summary>
    public void Embed(nint output, nint weights, int token, int dim, QuantizationType qt)
    {
        if (qt == QuantizationType.F16) { Embed(output, weights, token, dim); return; }
        int type = PackedWeightType(qt, dim);
        Pointer(output); Pointer(weights);
        ArgumentOutOfRangeException.ThrowIfNegative(token);
        void** args = stackalloc void*[] { &output, &weights, &token, &dim, &type };
        Launch(_embedPacked, Blocks(dim), 256, args);
    }

    /// <summary>Narrows file-backed F32 tensors during loading; the caller releases the source after synchronization.</summary>
    public void ConvertF32ToF16(nint output, nint input, int count)
    {
        Pointer(output); Pointer(input); Positive(count);
        void** args = stackalloc void*[] { &output, &input, &count };
        Launch(_convertF32, Blocks(count), 256, args);
    }

    /// <summary>Normalizes one F16 row using an F16 scale vector.</summary>
    public void RmsNorm(nint output, nint input, nint weight, int dim, float epsilon)
    {
        Pointer(output); Pointer(input); Pointer(weight); Positive(dim); PositiveFinite(epsilon);
        void** args = stackalloc void*[] { &output, &input, &weight, &dim, &epsilon };
        Launch(_rmsNorm, 1, 32, args);
    }

    /// <summary>Computes output[outDim] = weights[outDim,inDim] × input[inDim], with F32 register accumulation.</summary>
    public void Linear(nint output, nint input, nint weights, int outDim, int inDim)
    {
        Pointer(output); Pointer(input); Pointer(weights); Positive(outDim); Positive(inDim);
        void** args = stackalloc void*[] { &output, &input, &weights, &outDim, &inDim };
        Launch(_linear, (uint)outDim, 32, args);
    }

    /// <summary>Projects an F16 activation row using resident F16/Q4_0/Q8_0 weights without expanding the matrix.</summary>
    public void Linear(nint output, nint input, nint weights, int outDim, int inDim, QuantizationType qt)
    {
        if (qt == QuantizationType.F16) { Linear(output, input, weights, outDim, inDim); return; }
        int type = PackedWeightType(qt, inDim);
        Pointer(output); Pointer(input); Pointer(weights); Positive(outDim);
        void** args = stackalloc void*[] { &output, &input, &weights, &outDim, &inDim, &type };
        Launch(_linearPacked, (uint)outDim, 32, args);
    }

    /// <summary>Adds two F16 vectors; output may alias either input.</summary>
    public void Add(nint output, nint left, nint right, int count)
    {
        Pointer(output); Pointer(left); Pointer(right); Positive(count);
        void** args = stackalloc void*[] { &output, &left, &right, &count };
        Launch(_add, Blocks(count), 256, args);
    }

    /// <summary>Computes SiLU(gate) × up in F16; output may alias either input.</summary>
    public void SwiGlu(nint output, nint gate, nint up, int count)
    {
        Pointer(output); Pointer(gate); Pointer(up); Positive(count);
        void** args = stackalloc void*[] { &output, &gate, &up, &count };
        Launch(_swiGlu, Blocks(count), 256, args);
    }

    /// <summary>Applies standard or NeoX partial RoPE in place; scale multiplies the position.</summary>
    public void Rope(nint q, nint k, int qHeads, int kvHeads, int headDim, int rotaryDim,
        int position, float theta, float scale, bool neox)
    {
        Pointer(q); Pointer(k); Heads(qHeads, kvHeads, headDim);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (rotaryDim < 0 || rotaryDim > headDim || (rotaryDim & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(rotaryDim), "RoPE dimension must be even and fit the head dimension.");
        PositiveFinite(theta); PositiveFinite(scale);
        if (rotaryDim == 0) return;
        int mode = neox ? 1 : 0;
        int pairs = checked((qHeads + kvHeads) * (rotaryDim / 2));
        void** args = stackalloc void*[] { &q, &k, &qHeads, &kvHeads, &headDim, &rotaryDim, &position, &theta, &scale, &mode };
        Launch(_rope, Blocks(pairs), 256, args);
    }

    /// <summary>Writes one token into F16 pages laid out as [pageTokens, kvDim]. Tables hold device pointers.</summary>
    public void WriteKv(nint kPages, nint vPages, nint k, nint v, int pageTokens, int kvDim, int position)
    {
        Pointer(kPages); Pointer(vPages); Pointer(k); Pointer(v); Positive(pageTokens); Positive(kvDim);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        void** args = stackalloc void*[] { &kPages, &vPages, &k, &v, &pageTokens, &kvDim, &position };
        Launch(_writeKv, Blocks(kvDim), 256, args);
    }

    /// <summary>Attends one query to exactly length cached tokens without allocating contiguous KV scratch.</summary>
    public void Attention(nint output, nint q, nint kPages, nint vPages,
        int pageTokens, int qHeads, int kvHeads, int headDim, int length, float scale)
    {
        Pointer(output); Pointer(q); Pointer(kPages); Pointer(vPages);
        Positive(pageTokens); Heads(qHeads, kvHeads, headDim); Positive(length); PositiveFinite(scale);
        if (headDim > 256)
            throw new NotSupportedException("HIP dense attention currently supports head dimensions up to 256.");
        void** args = stackalloc void*[] { &output, &q, &kPages, &vPages, &pageTokens, &qHeads, &kvHeads, &headDim, &length, &scale };
        Launch(_attention, (uint)qHeads, 32, args);
    }

    /// <summary>Writes a token into independently typed F16/Q8_0 K/V pages; Q8_0 blocks never cross a head.</summary>
    public void WriteKv(nint kPages, nint vPages, nint k, nint v, int pageTokens,
        int kvHeads, int headDim, int position, KvCacheDType keyType, KvCacheDType valueType)
    {
        Heads(kvHeads, kvHeads, headDim);
        ValidateKvTypes(headDim, keyType, valueType);
        int kvDim = checked(kvHeads * headDim);
        if (keyType == KvCacheDType.F32 && valueType == KvCacheDType.F32)
        {
            WriteKv(kPages, vPages, k, v, pageTokens, kvDim, position);
            return;
        }
        Pointer(kPages); Pointer(vPages); Pointer(k); Pointer(v); Positive(pageTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        int kt = (int)keyType, vt = (int)valueType;
        void** args = stackalloc void*[] { &kPages, &vPages, &k, &v, &pageTokens, &kvDim, &position, &kt, &vt };
        Launch(_writeKvPacked, (uint)(kvDim / 32), 32, args);
    }

    /// <summary>Reads mixed F16/Q8_0 paged history directly, keeping online softmax state in registers.</summary>
    public void Attention(nint output, nint q, nint kPages, nint vPages,
        int pageTokens, int qHeads, int kvHeads, int headDim, int length, float scale,
        KvCacheDType keyType, KvCacheDType valueType)
    {
        ValidateKvTypes(headDim, keyType, valueType);
        if (keyType == KvCacheDType.F32 && valueType == KvCacheDType.F32)
        {
            Attention(output, q, kPages, vPages, pageTokens, qHeads, kvHeads, headDim, length, scale);
            return;
        }
        Pointer(output); Pointer(q); Pointer(kPages); Pointer(vPages);
        Positive(pageTokens); Heads(qHeads, kvHeads, headDim); Positive(length); PositiveFinite(scale);
        int kt = (int)keyType, vt = (int)valueType;
        void** args = stackalloc void*[] { &output, &q, &kPages, &vPages, &pageTokens, &qHeads, &kvHeads, &headDim, &length, &scale, &kt, &vt };
        Launch(_attentionPacked, (uint)qHeads, 32, args);
    }

    private static int PackedWeightType(QuantizationType type, int width)
    {
        if (type is not (QuantizationType.Q4_0 or QuantizationType.Q8_0))
            throw new NotSupportedException($"HIP dense weights support F16, Q4_0 and Q8_0, not {type}.");
        Positive(width);
        if (width % 32 != 0)
            throw new ArgumentException("Packed HIP weight rows must contain a multiple of 32 elements.", nameof(width));
        return (int)type;
    }

    private static void ValidateKvTypes(int headDim, KvCacheDType keyType, KvCacheDType valueType)
    {
        if (keyType is not (KvCacheDType.F32 or KvCacheDType.Q8_0)
            || valueType is not (KvCacheDType.F32 or KvCacheDType.Q8_0))
            throw new NotSupportedException("HIP KV supports F16 (KvCacheDType.F32) and Q8_0 independently for keys and values.");
        Positive(headDim);
        if (headDim > 256 || ((keyType == KvCacheDType.Q8_0 || valueType == KvCacheDType.Q8_0) && headDim % 32 != 0))
            throw new NotSupportedException("HIP Q8_0 KV requires head dimensions divisible by 32 and no larger than 256.");
    }

    private void Launch(nint function, uint blocks, uint threads, void** args)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _context.MakeCurrent();
        HipDriverApi.hipModuleLaunchKernel(function, blocks, 1, 1, threads, 1, 1,
            0, _stream.Handle, (nint)args, 0).ThrowOnError();
    }

    private static uint Blocks(int count) => (uint)(((long)count + 255) / 256);
    private static void Positive(int value) => ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
    private static void Pointer(nint value)
    {
        if (value == 0) throw new ArgumentException("A non-null device pointer is required.");
    }
    private static void PositiveFinite(float value)
    {
        if (!float.IsFinite(value) || value <= 0) throw new ArgumentOutOfRangeException(nameof(value));
    }
    private static void Heads(int qHeads, int kvHeads, int headDim)
    {
        Positive(qHeads); Positive(kvHeads); Positive(headDim);
        if (qHeads % kvHeads != 0) throw new ArgumentException("Query heads must be divisible by KV heads.");
        _ = checked(qHeads * headDim);
        _ = checked(kvHeads * headDim);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _module.Dispose();
    }
}
