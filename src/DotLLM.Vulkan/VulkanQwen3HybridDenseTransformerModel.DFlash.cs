using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures.DFlash;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// DFlash on the Vulkan <c>qwen35</c> trunk (issues #1175, #1193), all on the GPU: the verify forward copies the tapped layer
/// inputs into a host-readback buffer (F16 in the state), and the drafter (<see cref="VulkanDFlashBackend"/>) runs on the device
/// with its weights in their stored quantisation (imported zero-copy on UMA), its context ring in device F16, and the target's
/// own embedding table and lm-head. The injection of a verify's taps and the next draft record into one command buffer that is
/// submitted once, when the draft's logits are read.
/// </summary>
public sealed partial class VulkanQwen3HybridDenseTransformerModel : IDFlashDraftAttachable, IDFlashTargetOps
{
    private DFlashDrafter? _dflash;
    private VulkanDFlashBackend? _dflashGpu;
    private string _spvDir = string.Empty;
    private int[] _dflashSlotForLayerInput = [];
    private VulkanDevice.Buffer? _dflashTapBuffer;   // [slot][row][hidden] F32, host readback, transient
    private long _dflashTapBufferBytes;
    private float[] _dflashTapHost = [];

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dflash is not null;

    /// <summary>Device bytes held by the attached drafter's weights and norm vectors (0 without one).</summary>
    public long DFlashWeightBytes => _dflashGpu?.WeightBytes ?? 0;

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        if (_dflash is not null) throw new InvalidOperationException("A DFlash drafter is already attached.");
        var drafter = DFlashDrafter.Open(path, Config.HiddenSize, Config.NumLayers, Config.VocabSize, CreateDFlashBackend());
        int[] slots;
        try { slots = drafter.SlotForLayerInput(Config.NumLayers); }
        catch { drafter.Dispose(); throw; }
        _dflashSlotForLayerInput = slots;
        _dflashGpu = (VulkanDFlashBackend)drafter.Backend;
        _dflash = drafter;
        // Sized now so a draft never grows them mid-recording (a grow flushes the recorded draft early).
        if (_state.EnsureCapacity(drafter.BlockSize)) InvalidateDescriptorCaches();
        EnsureMultiRowLogits(drafter.BlockSize, Config.VocabSize);
    }

    /// <summary>A drafter backend on this model's device (also the tests' handle on the Vulkan drafter kernels).</summary>
    /// <returns>The backend.</returns>
    internal IDFlashBackend CreateDFlashBackend() => new VulkanDFlashBackend(this, VulkanDFlashKernels.Create(_device, _spvDir));

    /// <summary>Draft tokens per round (see the CPU host).</summary>
    public int MtpPreferredDraftTokens => _dflash is null ? 0 : _dflash.DraftTokensPerRound;

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dflash is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        if (state is not DFlashMtpState dflash)
            throw new NotSupportedException($"{nameof(VulkanQwen3HybridDenseTransformerModel)} has a native draft argmax only for DFlash states.");
        return dflash.NextDraft(this, tokenId, position, dflash.Drafter.DraftTokensPerRound);
    }

    private ITensor ForwardDFlashDraft(DFlashMtpState state, int tokenId, int position)
    {
        state.NextDraft(this, tokenId, position, state.Drafter.DraftTokensPerRound);
        var row = state.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        unsafe { row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length)); }
        return result;
    }

    /// <summary>Allocates the tap buffer for <paramref name="rows"/> before any recording (a new buffer invalidates descriptor caches).</summary>
    private void EnsureDFlashTapBuffer(int rows)
    {
        long bytes = (long)_dflash!.TargetLayers.Length * rows * Config.HiddenSize * sizeof(float);
        if (_dflashTapBuffer is not null && _dflashTapBufferBytes >= bytes) return;
        _dflashTapBuffer?.Dispose();
        _dflashTapBuffer = _device.AllocateHostReadback(bytes);
        _dflashTapBufferBytes = bytes;
        InvalidateDescriptorCaches();
    }

    /// <summary>Records the copy of <c>HiddenState</c> rows <c>[first, first + rows)</c> into tap slot of <paramref name="layerInput"/>.</summary>
    private void RecordDFlashTap(nint cmdBuf, int layerInput, int first, int rows)
    {
        int slot = _dflashSlotForLayerInput[layerInput];
        if (slot < 0) return;
        long rowBytes = (long)Config.HiddenSize * sizeof(float);
        KernelSupport.ComputeTransferFullBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _dflashTapBuffer!,
            srcOffset: (ulong)(first * rowBytes), dstOffset: (ulong)((long)slot * rows * rowBytes), size: (ulong)(rows * rowBytes));
    }

    /// <summary>Hands the downloaded taps to the state and injects them (after the forward's last submit).</summary>
    private unsafe void FinishDFlashTaps(DFlashMtpState state, ReadOnlySpan<int> positions, int rows)
    {
        int h = Config.HiddenSize, slots = _dflash!.TargetLayers.Length;
        int n = slots * rows * h;
        if (_dflashTapHost.Length < n) _dflashTapHost = new float[n];
        var host = _dflashTapHost.AsSpan(0, n);
        _device.Download(_dflashTapBuffer!, host);
        state.EnsureTapRows(rows);
        for (int s = 0; s < slots; s++)
            for (int r = 0; r < rows; r++)
                state.CaptureTap(r, s, host.Slice((s * rows + r) * h, h));
        state.InjectCaptured(positions);   // recorded on the drafter's command buffer; the next draft (or any invalidation) submits it
        state.TrimTapRows();
        if (rows > 4 * _dflash.BlockSize)
        {
            // A long prefill grew it: give the transient buffer back.
            _dflashTapBuffer!.Dispose();
            _dflashTapBuffer = null;
            _dflashTapBufferBytes = 0;
            _dflashTapHost = [];
            InvalidateDescriptorCaches();
        }
    }

    /// <inheritdoc/>
    void IDFlashTargetOps.EmbedForDraft(ReadOnlySpan<int> tokens, nint dst)
    {
        var g = _dflashGpu!;
        if (_state.EnsureCapacity(tokens.Length)) InvalidateDescriptorCaches();
        nint cmd = g.Command();
        RecordEmbeddingGather(cmd, tokens);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var (buf, off) = g.Resolve(dst);
        RecordCopyBufferRange(cmd, _state.HiddenState, buf, 0, (ulong)off, (ulong)((long)tokens.Length * Config.HiddenSize * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
    }

    /// <inheritdoc/>
    unsafe void IDFlashTargetOps.ProjectForDraft(nint hidden, int rows, float* logits)
    {
        var g = _dflashGpu!;
        int h = Config.HiddenSize, vocab = Config.VocabSize;
        bool resized = _state.EnsureCapacity(rows);
        var logitsBuf = rows == 1 ? SingleRowLogits : EnsureMultiRowLogits(rows, vocab);
        if (resized) InvalidateDescriptorCaches();
        nint cmd = g.Command();
        var (buf, off) = g.Resolve(hidden);
        RecordCopyBufferRange(cmd, buf, _state.NormOutput, (ulong)off, 0, (ulong)((long)rows * h * sizeof(float)));
        KernelSupport.ComputeTransferFullBarrier(cmd);
        var headIn = _state.NormOutput;
        if (_hadamard is { } headRot)
        {
            headIn = _state.HadamardScratch!;
            headRot.RecordForward(cmd, _state.NormOutput, headIn, rows, _weights.OutputInputDim);
            KernelSupport.ComputeToComputeBarrier(cmd);
        }
        RecordMatmul(cmd, _weights.OutputWeight, _weights.OutputDeviceQuantType, headIn, logitsBuf,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: rows);
        g.Flush();
        _device.Download(logitsBuf, new Span<float>(logits, rows * vocab));
    }

    private void DisposeDFlash()
    {
        _dflashTapBuffer?.Dispose();
        _dflash?.Dispose();
        _dflashGpu = null;
    }

    /// <summary>
    /// The DFlash drafter on this model's device (issue #1193), the Vulkan twin of the CUDA backend. Buffers are handed to the
    /// drafter as synthetic addresses (<c>id &lt;&lt; 32 | byte offset</c>) so its pointer arithmetic (row offsets into a scratch
    /// buffer, per-layer offsets into the ring) keeps working: the drafter's own kernels take ring offsets as push constants, and
    /// the trunk's kernels (which bind whole buffers) see an offset view through a shift buffer. Every op records into one
    /// command buffer, a full barrier after each; <see cref="Flush"/> submits it (the head's logits, a download, a free, or any
    /// descriptor-cache invalidation of the model).
    /// </summary>
    internal sealed unsafe class VulkanDFlashBackend(VulkanQwen3HybridDenseTransformerModel m, VulkanDFlashKernels k) : IDFlashBackend
    {
        private readonly VulkanDevice.SubmitContext _ctx = m._device.CreateSubmitContext();
        private readonly Dictionary<long, VulkanDevice.Buffer> _buffers = [];
        private readonly Dictionary<nint, (VulkanDevice.Buffer Buffer, QuantizationType Qt)> _weights = [];
        private readonly Dictionary<float[], VulkanDevice.Buffer> _vectors = new(ReferenceEqualityComparer.Instance);
        private readonly VulkanDevice.Buffer?[] _shift = new VulkanDevice.Buffer?[2];
        private VulkanDevice.Buffer? _cosSin;
        private float[] _cosSinHost = [];
        private int[] _cosSinPositions = [];   // the positions the table holds for the open command buffer
        private float[] _tapHost = [];
        private VulkanDevice.Buffer? _upload, _readback;   // host-visible staging for taps in and selector rows out
        private long _flushTicks;
        private int _flushes;
        private double[] _invFreq = [];
        private DFlashDrafter _d = null!;
        private long _nextId = 1;
        private bool _open;
        private bool _disposed;

        /// <summary>Device bytes held by the drafter's weights and vectors.</summary>
        public long WeightBytes { get; private set; }

        public void Bind(DFlashDrafter drafter)
        {
            if (drafter.HeadDim > VulkanDFlashKernels.MaxHeadDim || (drafter.HeadDim & 1) != 0)
                throw new NotSupportedException($"The Vulkan DFlash drafter covers even head dims up to {VulkanDFlashKernels.MaxHeadDim}; this drafter has {drafter.HeadDim}.");
            _d = drafter;
            _invFreq = new double[drafter.HeadDim / 2];
            for (int i = 0; i < _invFreq.Length; i++)
                _invFreq[i] = (float)(1.0 / Math.Pow(drafter.RopeTheta, 2.0 * i / drafter.HeadDim));   // float, as the CPU's table

            long staging = 64;
            foreach (var w in drafter.DeviceMatrices)
                staging = Math.Max(staging, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(w.Out, w.In, w.Qt));
            foreach (var v in drafter.DeviceVectors)
                staging = Math.Max(staging, (long)v.Length * sizeof(float));
            using var stage = VulkanStagingBuffer.Create(m._device, staging);
            foreach (var w in drafter.DeviceMatrices)
            {
                if (_weights.ContainsKey(w.Ptr)) continue;
                // The stored blocks as they are (imported zero-copy where the device allows); K-quant drafters are never widened.
                var buf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(m._device, stage, w.Ptr, w.Qt, w.Out, w.In,
                    forceF32: false, out var qt, out long bytes);
                _weights[w.Ptr] = (buf, qt);
                WeightBytes += bytes;
            }
            foreach (var v in drafter.DeviceVectors)
            {
                if (_vectors.ContainsKey(v)) continue;
                _vectors[v] = VulkanQwen3MoeHybridWeights.UploadFloatArray(m._device, stage, v);
                WeightBytes += (long)v.Length * sizeof(float);
            }
        }

        // ── Memory ──────────────────────────────────────────────────────

        private nint Track(VulkanDevice.Buffer buffer)
        {
            long id = _nextId++;
            _buffers[id] = buffer;
            return (nint)(id << 32);
        }

        /// <summary>The buffer and byte offset a handle (plus any offset the drafter added to it) names.</summary>
        public (VulkanDevice.Buffer Buffer, long Offset) Resolve(nint handle)
            => (_buffers[(long)handle >> 32], (long)handle & 0xFFFFFFFFL);

        public nint Alloc(long floats) => Track(m._device.AllocateDeviceLocal(Math.Max(floats, 1) * sizeof(float)));

        public nint AllocHalf(long halves)
        {
            long bytes = (Math.Max(halves, 2) + 1) / 2 * 2 * sizeof(ushort);
            if (bytes > uint.MaxValue) throw new NotSupportedException("DFlash ring offsets are 32-bit.");
            return Track(m._device.AllocateDeviceLocal(bytes));
        }

        public void Free(nint buffer)
        {
            if (buffer == 0 || _disposed) return;
            Flush();
            if (!_buffers.Remove((long)buffer >> 32, out var b)) return;
            b.Dispose();   // descriptor sets naming it are evicted through the device's destroy log (#467)
        }

        public void Download(nint src, float* dst, int count)
        {
            // Through a persistent host-readback buffer, copied at the end of the recorded work: one submit, no allocation.
            long bytes = (long)count * sizeof(float);
            _readback = Grow(_readback, bytes, readback: true);
            var (b, off) = Resolve(src);
            RecordCopy(b, off, _readback, 0, bytes);
            Flush();
            m._device.Download(_readback, new Span<float>(dst, count));
        }

        public void LoadTaps(Half* taps, int count, nint dst)
        {
            Flush();   // an earlier chunk's recorded projections may still read dst, or its copy the staging
            long bytes = (long)count * sizeof(float);
            _upload = Grow(_upload, bytes, readback: false);
            if (_tapHost.Length < count) _tapHost = new float[count];
            TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(taps, count), _tapHost.AsSpan(0, count));
            m._device.Upload(_tapHost.AsSpan(0, count), _upload);   // host-visible: a mapped write, no submit
            var (b, off) = Resolve(dst);
            RecordCopy(_upload, 0, b, off, bytes);
        }

        /// <summary>A host-visible buffer of at least <paramref name="bytes"/>, replacing <paramref name="current"/> when smaller.</summary>
        private VulkanDevice.Buffer Grow(VulkanDevice.Buffer? current, long bytes, bool readback)
        {
            if (current is not null && current.Size >= bytes) return current;
            Flush();
            current?.Dispose();
            return readback ? m._device.AllocateHostReadback(bytes) : m._device.Allocate(bytes);
        }

        /// <summary>Test hook: copies host floats to <paramref name="dst"/> (any offset), synchronously.</summary>
        internal void UploadForTest(float* src, nint dst, int count)
        {
            Flush();
            using var tmp = m._device.Allocate((long)count * sizeof(float));
            m._device.Upload(new ReadOnlySpan<float>(src, count), tmp);
            var (b, off) = Resolve(dst);
            RecordCopy(tmp, 0, b, off, (long)count * sizeof(float));
            Flush();
        }

        /// <summary>Test hook: copies floats at <paramref name="src"/> (any offset) to the host, synchronously.</summary>
        internal void DownloadForTest(nint src, float* dst, int count)
        {
            using var tmp = m._device.AllocateHostReadback((long)count * sizeof(float));
            var (b, off) = Resolve(src);
            RecordCopy(b, off, tmp, 0, (long)count * sizeof(float));
            Flush();
            m._device.Download(tmp, new Span<float>(dst, count));
        }

        // ── Recording ───────────────────────────────────────────────────

        /// <summary>The open command buffer (begun on first use).</summary>
        public nint Command()
        {
            if (!_open)
            {
                _ctx.Begin();
                KernelSupport.HostToComputeBarrier(_ctx.CommandBuffer);
                _open = true;
                _cosSinPositions = [];
            }
            return _ctx.CommandBuffer;
        }

        /// <summary>Submits whatever is recorded and waits.</summary>
        public void Flush()
        {
            if (!_open) return;
            _open = false;   // first: the submit cannot re-enter
            KernelSupport.ComputeToHostBarrier(_ctx.CommandBuffer);
            KernelSupport.TransferToHostBarrier(_ctx.CommandBuffer);
            long t = System.Diagnostics.Stopwatch.GetTimestamp();
            _ctx.SubmitAndWait();
            _flushTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t;
            _flushes++;
        }

        private void Done() => KernelSupport.ComputeTransferFullBarrier(_ctx.CommandBuffer);

        /// <summary>A whole buffer holding the <paramref name="bytes"/> at <paramref name="handle"/>: itself, or a shift slot it is copied into.</summary>
        private VulkanDevice.Buffer In(nint handle, long bytes, int slot)
        {
            var (b, off) = Resolve(handle);
            if (off == 0) return b;
            var s = Shift(slot, bytes);
            RecordCopy(b, off, s, 0, bytes);
            return s;
        }

        /// <summary>A whole buffer to write <paramref name="bytes"/> for <paramref name="handle"/> into; <see cref="Commit"/> copies it back.</summary>
        private VulkanDevice.Buffer Out(nint handle, long bytes, int slot)
        {
            var (b, off) = Resolve(handle);
            return off == 0 ? b : Shift(slot, bytes);
        }

        private void Commit(nint handle, long bytes, int slot)
        {
            var (b, off) = Resolve(handle);
            if (off != 0) RecordCopy(_shift[slot]!, 0, b, off, bytes);
        }

        private void RecordCopy(VulkanDevice.Buffer src, long srcOff, VulkanDevice.Buffer dst, long dstOff, long bytes)
        {
            RecordCopyBufferRange(Command(), src, dst, (ulong)srcOff, (ulong)dstOff, (ulong)bytes);
            Done();
        }

        private VulkanDevice.Buffer Shift(int slot, long bytes)
        {
            if (_shift[slot] is { } s && s.Size >= bytes) return s;
            // Sized for the drafter's largest offset view (its drafted rows). Grown before anything that reads it is recorded.
            Flush();
            _shift[slot]?.Dispose();
            _shift[slot] = m._device.AllocateDeviceLocal(Math.Max(bytes, (long)_d.BlockSize * _d.HiddenSize * sizeof(float)));
            return _shift[slot]!;
        }

        public void Gemm(in DFlashDrafter.Matrix w, nint x, nint y, int rows)
        {
            var (wb, qt) = _weights[w.Ptr];
            long xBytes = (long)rows * w.In * sizeof(float), yBytes = (long)rows * w.Out * sizeof(float);
            var yb = Out(y, yBytes, 1);
            var xb = In(x, xBytes, 0);
            m.RecordMatmul(Command(), wb, qt, xb, yb, outputDim: w.Out, inputDim: w.In, seqLen: rows);
            Done();
            Commit(y, yBytes, 1);
        }

        public void RmsNorm(nint src, nint dst, int rows, float[] weight)
        {
            long bytes = (long)rows * weight.Length * sizeof(float);
            var db = src == dst ? null : Out(dst, bytes, 1);
            var sb = In(src, bytes, 0);
            m._kernels.RmsNorm.Record(Command(), sb, _vectors[weight], db ?? sb, rows, weight.Length, _d.Eps);
            Done();
            Commit(dst, bytes, src == dst ? 0 : 1);
        }

        public void NormRopeHeads(nint x, int heads, float[] norm, ReadOnlySpan<int> positions)
        {
            int rows = positions.Length, half = _d.HeadDim / 2;
            var (xb, off) = Resolve(x);
            if (off != 0) throw new NotSupportedException("DFlash norm/RoPE runs on whole buffers.");
            long tableBytes = (long)rows * half * 2 * sizeof(float);
            if (_cosSin is null || _cosSin.Size < tableBytes)
            {
                Flush();
                _cosSin?.Dispose();
                _cosSin = m._device.AllocateDeviceLocal(Math.Max(tableBytes, 64L * half * 2 * sizeof(float)));
                _cosSinPositions = [];
            }
            nint cmd = Command();
            if (!positions.SequenceEqual(_cosSinPositions))
            {
                // cos/sin in double from the float frequencies, exactly the CPU's table; a block's q and k (all layers) share it.
                int n = rows * half * 2;
                if (_cosSinHost.Length < n) _cosSinHost = new float[n];
                for (int r = 0; r < rows; r++)
                    for (int i = 0; i < half; i++)
                    {
                        double a = (double)positions[r] * _invFreq[i];
                        _cosSinHost[(r * half + i) * 2] = (float)Math.Cos(a);
                        _cosSinHost[(r * half + i) * 2 + 1] = (float)Math.Sin(a);
                    }
                fixed (float* p = _cosSinHost)
                {
                    for (long o = 0; o < tableBytes; o += 65536)   // vkCmdUpdateBuffer takes at most 64 KiB
                        VulkanApi.vkCmdUpdateBuffer(cmd, _cosSin.Handle, (ulong)o, (ulong)Math.Min(65536, tableBytes - o), (nint)((byte*)p + o));
                }
                _cosSinPositions = positions.ToArray();
                Done();
            }
            k.RecordNormRope(cmd, xb, _vectors[norm], _cosSin, rows, heads, _d.HeadDim, _d.Eps);
            Done();
        }

        public void Conv(nint src, nint dyn, float[] baseKernel, int side, int rows, nint dst)
        {
            var (xb, xo) = Resolve(src);
            var (db, dOff) = Resolve(dyn);
            var (yb, yo) = Resolve(dst);
            if ((xo | dOff | yo) != 0) throw new NotSupportedException("DFlash convolution runs on whole buffers.");
            k.RecordConv(Command(), xb, db, _vectors[baseKernel], yb, rows, _d.HiddenSize, _d.ConvKernel, _d.ConvGroup, side);
            Done();
        }

        public void SwiGlu(nint gate, nint up, int n)
        {
            long bytes = (long)n * sizeof(float);
            var g = In(gate, bytes, 0);
            var u = In(up, bytes, 1);
            m._kernels.SwiGlu.Record(Command(), g, u, g, n);
            Done();
            Commit(gate, bytes, 0);
        }

        public void Add(nint x, nint y, int n)
        {
            long bytes = (long)n * sizeof(float);
            var xb = In(x, bytes, 0);
            var yb = In(y, bytes, 1);
            m._kernels.Add.Record(Command(), xb, yb, xb, n);
            Done();
            Commit(x, bytes, 0);
        }

        public void WriteRing(nint ringK, nint ringV, int capacity, ReadOnlySpan<int> positions, nint kRows, nint vRows)
        {
            int kvDim = _d.NumKvHeads * _d.HeadDim;
            var (kb, ko) = Resolve(kRows);
            var (vb, vo) = Resolve(vRows);
            if ((ko | vo) != 0) throw new NotSupportedException("DFlash ring writes read whole buffers.");
            var (rk, rko) = Resolve(ringK);
            var (rv, rvo) = Resolve(ringV);
            nint cmd = Command();
            // Runs of consecutive positions go in one dispatch each (the shader wraps the slot).
            for (int i = 0; i < positions.Length;)
            {
                int j = i + 1;
                while (j < positions.Length && positions[j] == positions[i] + (j - i)) j++;
                k.RecordRingWrite(cmd, kb, vb, rk, rv, j - i, kvDim, i, positions[i], capacity, rko, rvo);
                i = j;
            }
            Done();
        }

        public void Attend(nint ringK, nint ringV, int capacity, int lo, int start, int b, nint q, nint kb, nint vb, nint output)
        {
            var (qb, qo) = Resolve(q);
            var (kbb, kbo) = Resolve(kb);
            var (vbb, vbo) = Resolve(vb);
            var (ob, oo) = Resolve(output);
            if ((qo | kbo | vbo | oo) != 0) throw new NotSupportedException("DFlash attention runs on whole buffers.");
            var (rk, rko) = Resolve(ringK);
            var (rv, rvo) = Resolve(ringV);
            k.RecordAttention(Command(), qb, kbb, vbb, rk, rv, ob, b, _d.NumHeads, _d.NumKvHeads, _d.HeadDim,
                capacity, lo, start, _d.Window, 1f / MathF.Sqrt(_d.HeadDim), rko, rvo);
            Done();
        }

        public void Dispose()
        {
            if (_disposed) return;
            Flush();
            _disposed = true;
            if (Environment.GetEnvironmentVariable("DOTLLM_DFLASH_TRACE") == "1" && _flushes > 0)
                Console.Error.WriteLine($"[dflash-vk] submits={_flushes} submit_wait_ms={_flushTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} weights_mb={WeightBytes >> 20}");
            _upload?.Dispose();
            _readback?.Dispose();
            _ctx.Dispose();
            foreach (var b in _buffers.Values) b.Dispose();
            _buffers.Clear();
            foreach (var (b, _) in _weights.Values) b.Dispose();
            _weights.Clear();
            foreach (var b in _vectors.Values) b.Dispose();
            _vectors.Clear();
            foreach (var b in _shift) b?.Dispose();
            _cosSin?.Dispose();
            k.Dispose();
            if (ReferenceEquals(m._dflashGpu, this)) m._dflashGpu = null;
        }
    }
}
