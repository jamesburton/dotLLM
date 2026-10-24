using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures.DFlash;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// DFlash on the CUDA <c>qwen35</c> trunk (issue #1175), all on the GPU: the verify forward copies the tapped layer inputs to
/// pinned host memory (F16 there), and the drafter (<see cref="CudaDFlashBackend"/>) runs on the device with its weights
/// uploaded as stored, its context ring in device F16, and the target's own lm-head.
/// </summary>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private int[] _dflashSlotForLayerInput = [];
    private DFlashMtpState? _dflashCall;     // the DFlash state of the forward in flight, null otherwise
    private nint _dflashTapPinned;           // [slot][row][hidden] F32, page-locked
    private long _dflashTapPinnedBytes;

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize, CreateDFlashBackend());
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        _dflashSlotForLayerInput = slots;
        _dflash = drafter;
    }

    /// <summary>A drafter backend on this model's device and stream (also the tests' handle on the CUDA kernels).</summary>
    /// <returns>The backend.</returns>
    internal IDFlashBackend CreateDFlashBackend()
    {
        _context.MakeCurrent();
        var kernels = CudaDFlashKernels.TryLoad(_kernels.PtxDirectory)
            ?? throw new NotSupportedException($"{CudaDFlashKernels.PtxFileName} is missing from {_kernels.PtxDirectory}: the CUDA DFlash drafter needs it.");
        return new CudaDFlashBackend(this, kernels);
    }

    /// <summary>Draft tokens per round (see the CPU host).</summary>
    public int MtpPreferredDraftTokens => _dflash?.DraftTokensPerRound ?? 0;

    private int ForwardDFlashArgMax(DFlashMtpState state, int tokenId, int position)
    {
        _context.MakeCurrent();
        return state.NextDraft(this, tokenId, position, state.Drafter.DraftTokensPerRound);
    }

    private ITensor ForwardDFlashDraft(DFlashMtpState state, int tokenId, int position)
    {
        ForwardDFlashArgMax(state, tokenId, position);
        var row = state.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    /// <summary>Rows of this tile the drafter's ring can hold: <c>[first, seqLen)</c>.</summary>
    private int DFlashTapFirst(int seqLen) => Math.Max(0, seqLen - _dflashCall!.RingCapacity);

    private void EnsureDFlashTapPinned(int rows)
    {
        long bytes = (long)_dflash!.TargetLayers.Length * rows * Config.HiddenSize * sizeof(float);
        if (_dflashTapPinned != 0 && _dflashTapPinnedBytes >= bytes) return;
        FreeDFlashTapPinned();
        CudaDriverApi.cuMemHostAlloc(out _dflashTapPinned, (nuint)bytes, 0).ThrowOnError();
        _dflashTapPinnedBytes = bytes;
    }

    private void FreeDFlashTapPinned()
    {
        if (_dflashTapPinned != 0) CudaDriverApi.cuMemFreeHost(_dflashTapPinned);
        _dflashTapPinned = 0;
        _dflashTapPinnedBytes = 0;
    }

    /// <summary>Queues the D2H copy of <c>HiddenState</c> rows <c>[first, seqLen)</c> into the tap slot of <paramref name="layerInput"/>.</summary>
    private void QueueDFlashTap(int layerInput, int seqLen)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        int first = DFlashTapFirst(seqLen), rows = seqLen - first;
        long rowBytes = (long)Config.HiddenSize * sizeof(float);
        CudaDriverApi.cuMemcpyDtoHAsync_v2(_dflashTapPinned + (nint)((long)slot * rows * rowBytes),
            _state.HiddenState + (nint)(first * rowBytes), (nuint)(rows * rowBytes), _stream.Handle).ThrowOnError();
    }

    /// <summary>After the tile's stream synchronize: hands the taps to the state and injects them.</summary>
    private void FinishDFlashTaps(ReadOnlySpan<int> positions)
    {
        var state = _dflashCall!;
        int first = DFlashTapFirst(positions.Length), rows = positions.Length - first, h = Config.HiddenSize;
        state.EnsureTapRows(rows);
        float* host = (float*)_dflashTapPinned;
        for (int s = 0; s < _dflash!.TargetLayers.Length; s++)
            for (int r = 0; r < rows; r++)
                state.CaptureTap(r, s, new ReadOnlySpan<float>(host + ((long)s * rows + r) * h, h));
        state.InjectCaptured(positions.Slice(first));
        state.TrimTapRows();
        if (rows > 4 * _dflash.BlockSize) FreeDFlashTapPinned();   // a long prefill tile grew it
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
    {
        // The trunk's own host-side lookup (rows from the mapped table, Hadamard-latent rows restored), then one upload.
        int h = Config.HiddenSize;
        var host = new float[tokens.Length * h];
        fixed (float* p = host)
        {
            for (int t = 0; t < tokens.Length; t++)
            {
                nint rowSrc = _embedDataBase + (nint)(_embedDataOffset + (ulong)tokens[t] * (ulong)_embedRowBytes);
                Dequantize.ToFloat32(rowSrc, h, _tokenEmbedQt, host.AsSpan(t * h, h));
            }
            _hadamard?.RotateInverseInPlaceHost(p, tokens.Length, h);
            CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(host.Length * sizeof(float))).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.ProjectForDraft(nint hidden, int rows, float* logits)
    {
        nint streamH = _stream.Handle;
        _state.EnsureCapacity(rows);
        _state.EnsureLogitsCapacity(rows);
        nint headIn = hidden;
        if (_hadamard is { } headRot)
        {
            headIn = _state.FfnGate;   // #1222: FFN scratch is dead after the trunk
            headRot.RotateForward(hidden, headIn, rows, _outputInputDim, permuteGdnValueHeads: false, streamH);
        }
        GemmLmHead(_outputDevice, _outputQt, headIn, _state.Logits, _outputOutputDim, _outputInputDim, rows);
        _stream.Synchronize();
        CudaDriverApi.cuMemcpyDtoH_v2((nint)logits, _state.Logits,
            (nuint)((long)rows * Config.VocabSize * sizeof(float))).ThrowOnError();
    }

    private void DisposeDFlash()
    {
        FreeDFlashTapPinned();
        _dflash?.Dispose();
    }

    /// <summary>
    /// The DFlash drafter on this model's device and stream (issue #1175). Weights are uploaded as stored and run through the
    /// trunk's own projection dispatcher one row at a time (the quantised GEMV: no F16 expansion of the drafter's weights);
    /// norms, RoPE, SwiGLU and adds reuse the trunk's kernels; the convolution and the ring attention are
    /// <c>dflash_f32.ptx</c>.
    /// </summary>
    private sealed class CudaDFlashBackend(CudaQwen3HybridDenseTransformerModel m, CudaDFlashKernels k) : IDFlashBackend
    {
        private readonly Dictionary<nint, nint> _weights = new();                          // host mmap ptr -> device
        private readonly Dictionary<float[], nint> _vec32 = new(ReferenceEqualityComparer.Instance);
        private readonly List<nint> _owned = [];
        private DFlashDrafter _d = null!;
        private nint _positions;          // device int[cap]
        private int _positionsCap;
        private int _uploadedFirst = -1, _uploadedRows;
        private nint _stage;              // pinned F32 staging for tap uploads
        private long _stageFloats;

        /// <summary>Device bytes held by the drafter's weights.</summary>
        public long WeightBytes { get; private set; }

        public void Bind(DFlashDrafter drafter)
        {
            _d = drafter;
            foreach (var w in drafter.DeviceMatrices)
            {
                if (_weights.ContainsKey(w.Ptr)) continue;
                long bytes = Dequantize.RowByteSize(w.In, w.Qt) * w.Out;
                nint dev = Upload((void*)w.Ptr, bytes);
                _weights[w.Ptr] = dev;
                WeightBytes += bytes;
            }
            foreach (var v in drafter.DeviceVectors)
            {
                fixed (float* p = v) _vec32[v] = Upload(p, (long)v.Length * sizeof(float));
            }
        }

        private nint Upload(void* src, long bytes)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)bytes).ThrowOnError();
            _owned.Add(dev);
            CudaDriverApi.cuMemcpyHtoD_v2(dev, (nint)src, (nuint)bytes).ThrowOnError();
            return dev;
        }

        public nint Alloc(long floats)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)(Math.Max(floats, 1) * sizeof(float))).ThrowOnError();
            return dev;
        }

        public nint AllocHalf(long halves)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint dev, (nuint)(Math.Max(halves, 1) * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(dev, 0, (nuint)(Math.Max(halves, 1) * sizeof(ushort))).ThrowOnError();
            return dev;
        }

        public void Free(nint buffer) { if (buffer != 0) CudaDriverApi.cuMemFree_v2(buffer); }

        public void Download(nint src, float* dst, int count)
        {
            m._stream.Synchronize();
            CudaDriverApi.cuMemcpyDtoH_v2((nint)dst, src, (nuint)((long)count * sizeof(float))).ThrowOnError();
        }

        public void LoadTaps(Half* taps, int count, nint dst)
        {
            if (_stageFloats < count)
            {
                if (_stage != 0) { m._stream.Synchronize(); CudaDriverApi.cuMemFreeHost(_stage); }
                CudaDriverApi.cuMemHostAlloc(out _stage, (nuint)((long)count * sizeof(float)), 0).ThrowOnError();
                _stageFloats = count;
            }
            m._stream.Synchronize();   // the previous chunk's upload may still read the staging rows
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(taps, count), new Span<float>((void*)_stage, count));
            CudaDriverApi.cuMemcpyHtoDAsync_v2(dst, _stage, (nuint)((long)count * sizeof(float)), m._stream.Handle).ThrowOnError();
        }

        public void Gemm(in DFlashDrafter.Matrix w, nint x, nint y, int rows)
        {
            if (w.Scale != 1f)
                throw new NotSupportedException("The CUDA DFlash drafter does not apply per-tensor weight scales (X.scale) yet.");
            nint dev = _weights[w.Ptr];
            long xRow = (long)w.In * sizeof(float), yRow = (long)w.Out * sizeof(float);
            if (CudaDFlashKernels.HasGemvMulti(w.Qt))
            {
                // K-quants: every weight block read once per 8 rows, straight from the stored blocks.
                for (int r = 0; r < rows; r += CudaDFlashKernels.MaxGemvColumns)
                    k.LaunchGemvMulti(w.Qt, dev, x + (nint)(r * xRow), y + (nint)(r * yRow), w.Out, w.In,
                        Math.Min(CudaDFlashKernels.MaxGemvColumns, rows - r), m._stream.Handle);
                return;
            }
            // Anything else one row at a time: the quantised GEMV reads the stored blocks directly, where a multi-row call
            // would dequantize the whole matrix to F16 first (#495).
            for (int r = 0; r < rows; r++)
                m.Gemm(dev, w.Qt, x + (nint)(r * xRow), y + (nint)(r * yRow), w.Out, w.In, 1);
        }

        public void RmsNorm(nint src, nint dst, int rows, float[] weight)
            => m._kernels.LaunchRmsNormF32(src, _vec32[weight], dst, weight.Length, _d.Eps, rows, m._stream.Handle);

        public void NormRopeHeads(nint x, int heads, float[] norm, ReadOnlySpan<int> positions)
        {
            int rows = positions.Length;
            if (_positionsCap < rows)
            {
                if (_positions != 0) { m._stream.Synchronize(); CudaDriverApi.cuMemFree_v2(_positions); }
                _uploadedFirst = -1;
                CudaDriverApi.cuMemAlloc_v2(out _positions, (nuint)(Math.Max(rows, 64) * sizeof(int))).ThrowOnError();
                _positionsCap = Math.Max(rows, 64);
            }
            // q and k of a block (and every layer) share the positions: upload once per distinct run.
            if (positions[0] != _uploadedFirst || rows != _uploadedRows || positions[rows - 1] != positions[0] + rows - 1)
            {
                fixed (int* p = positions)
                    CudaDriverApi.cuMemcpyHtoD_v2(_positions, (nint)p, (nuint)(rows * sizeof(int))).ThrowOnError();
                bool contiguous = positions[rows - 1] == positions[0] + rows - 1;
                _uploadedFirst = contiguous ? positions[0] : -1;
                _uploadedRows = rows;
            }
            nint s = m._stream.Handle;
            m._kernels.LaunchPerHeadRmsNormF32(x, _vec32[norm],   // the kernel's weight is F32 despite the parameter name
                _d.Eps, heads, _d.HeadDim, rows, s);
            m._kernels.LaunchRoPEF32PartialNeoX(x, x, _positions, rows, heads, 0, _d.HeadDim, _d.HeadDim / 2, _d.RopeTheta, s);
        }

        public void Conv(nint src, nint dyn, float[] baseKernel, int side, int rows, nint dst)
            => k.LaunchConv(src, dyn, _vec32[baseKernel], dst, rows, _d.HiddenSize, _d.ConvKernel, _d.ConvGroup, side, m._stream.Handle);

        public void SwiGlu(nint gate, nint up, int n) => m._kernels.LaunchSwiGLUF32(gate, up, gate, n, 1, m._stream.Handle);

        public void Add(nint x, nint y, int n) => m._kernels.LaunchAddF32(x, y, x, n, m._stream.Handle);

        public void WriteRing(nint ringK, nint ringV, int capacity, ReadOnlySpan<int> positions, nint kRows, nint vRows)
        {
            int kvDim = _d.NumKvHeads * _d.HeadDim;
            long rowHalf = (long)kvDim * sizeof(ushort), rowF = (long)kvDim * sizeof(float);
            // Runs of consecutive slots convert in one launch.
            for (int i = 0; i < positions.Length;)
            {
                int slot = positions[i] % capacity, j = i + 1;
                while (j < positions.Length && positions[j] % capacity == slot + (j - i)) j++;
                int n = (j - i) * kvDim;
                m._kernels.LaunchConvertF32ToF16(kRows + (nint)(i * rowF), ringK + (nint)(slot * rowHalf), n, m._stream.Handle);
                m._kernels.LaunchConvertF32ToF16(vRows + (nint)(i * rowF), ringV + (nint)(slot * rowHalf), n, m._stream.Handle);
                i = j;
            }
        }

        public void Copy(nint src, nint dst, int floats)
            => CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)((long)floats * sizeof(float)), m._stream.Handle).ThrowOnError();

        public void Attend(nint ringK, nint ringV, int capacity, int lo, int start, int b, nint q, nint kb, nint vb, nint output, float[]? sinks)
        {
            if (sinks is not null) throw new NotSupportedException("The CUDA DFlash drafter has no attention sinks yet.");
            k.LaunchAttention(q, kb, vb, ringK, ringV, output, b, _d.NumHeads, _d.NumKvHeads, _d.HeadDim,
                capacity, lo, start, _d.Window, 1f / MathF.Sqrt(_d.HeadDim), m._stream.Handle);
        }

        public void Dispose()
        {
            m._stream.Synchronize();
            foreach (nint p in _owned) CudaDriverApi.cuMemFree_v2(p);
            _owned.Clear();
            if (_positions != 0) CudaDriverApi.cuMemFree_v2(_positions);
            if (_stage != 0) CudaDriverApi.cuMemFreeHost(_stage);
            _positions = _stage = 0;
            k.Dispose();
        }
    }
}
