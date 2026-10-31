using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// The <c>qwen35moe</c> MTP head on Vulkan (issue #1188): the device mirror of the CPU oracle in
/// <c>Qwen3MoeHybridTransformerModel.Mtp.cs</c> and of the dense <c>qwen35</c> head (<see cref="VulkanQwen3HybridDenseTransformerModel"/>, #435),
/// with the trunk's routed + shared-expert MoE kernels in place of the dense SwiGLU.
/// </summary>
/// <remarks>
/// <para>
/// Every trunk forward carrying a <see cref="VulkanMtpState"/> downloads its post-final-norm rows (normalised on the host, as the dense head
/// does) and absorbs the batch into the head's KV in one submit (K/V only). A draft step is one submit: embed, <c>eh_proj</c>, gated attention
/// over the head's KV, MoE, shared-head norm and the trunk's LM head. The head's routed banks are uploaded once, like the trunk's resident
/// banks, so MTP needs resident MoE (the streaming mode re-uploads a layer's experts per forward).
/// </para>
/// <para>
/// The head KV lives in <see cref="VulkanMtpState"/>, which stores F32 rows (4 KiB per position here) — the same store the dense Vulkan
/// head uses and #1107 tracks; converting that shared state to the trunk KV type converts both heads.
/// </para>
/// </remarks>
public sealed partial class VulkanQwen3MoeHybridTransformerModel
{
    /// <summary>Device copy of the MTP block. Embedding and LM head are the trunk's.</summary>
    private sealed class MtpHeadDevice : IDisposable
    {
        public required VulkanDevice.Buffer AttnNorm { get; init; }
        public required VulkanDevice.Buffer PostAttnNorm { get; init; }
        public required VulkanQwen3MoeHybridWeights.FullAttnLayerBuffers Attention { get; init; }
        public required VulkanQwen3MoeMoeUpload.LayerBundle Moe { get; init; }
        public required VulkanDevice.Buffer EhProj { get; init; }
        public required QuantizationType EhProjQt { get; init; }
        public required int EhProjIn { get; init; }
        public required int EhProjOut { get; init; }
        public required VulkanDevice.Buffer Enorm { get; init; }
        public required VulkanDevice.Buffer Hnorm { get; init; }
        public VulkanDevice.Buffer? SharedHeadNorm { get; init; }

        public void Dispose()
        {
            AttnNorm.Dispose(); PostAttnNorm.Dispose(); Attention.Dispose(); Moe.Dispose();
            EhProj.Dispose(); Enorm.Dispose(); Hnorm.Dispose(); SharedHeadNorm?.Dispose();
        }
    }

    /// <summary>One-row device scratch for <see cref="DraftStep"/>.</summary>
    private sealed class MtpScratch : IDisposable
    {
        public required VulkanDevice.Buffer ENorm { get; init; }
        public required VulkanDevice.Buffer HNorm { get; init; }
        public required VulkanDevice.Buffer Concat { get; init; }
        public required VulkanDevice.Buffer Cur { get; init; }
        public required VulkanDevice.Buffer Residual { get; init; }
        public required VulkanDevice.Buffer Normed { get; init; }
        public required VulkanDevice.Buffer AddOut { get; init; }

        public static MtpScratch Allocate(VulkanDevice device, int rows, int hidden)
        {
            long h = (long)rows * hidden * sizeof(float);
            return new MtpScratch
            {
                ENorm = device.AllocateDeviceLocal(h),
                HNorm = device.AllocateDeviceLocal(h),
                Concat = device.AllocateDeviceLocal(2 * h),
                Cur = device.AllocateDeviceLocal(h),
                Residual = device.AllocateDeviceLocal(h),
                Normed = device.AllocateDeviceLocal(h),
                AddOut = device.AllocateDeviceLocal(h),
            };
        }

        public void Dispose()
        {
            ENorm.Dispose(); HNorm.Dispose(); Concat.Dispose(); Cur.Dispose();
            Residual.Dispose(); Normed.Dispose(); AddOut.Dispose();
        }
    }

    private MtpHeadDevice? _mtpHead;
    private MtpScratch? _mtpScratch;
    private MtpScratch? _mtpAbsorbScratch;
    private int _mtpAbsorbRows;
    private VulkanDevice.Buffer? _mtpPairUpload;   // host-visible [rows, hidden] pairing rows for the absorb
    private float[] _mtpRows = [];
    private float[] _outputNormHost = [];
    private VulkanMtpState? _lastMtpState;

    /// <inheritdoc/>
    public bool SupportsMtp => _mtpHead is not null;

    /// <inheritdoc/>
    /// <remarks>
    /// <see cref="ComputeMemoryBytes"/> is only the scratch (~3.7 GiB on Qwen3.6-35B-A3B), which would put this 22 GiB model on the
    /// plain-first side of the gate's size prior. With the head opted in (<see cref="MtpOptIn"/>) MTP is tried first and the gate's own
    /// measurements decide from there.
    /// </remarks>
    public long MtpGatePriorBytes => long.MaxValue;

    /// <summary>Default head-KV depth when no context length is given.</summary>
    public const int MtpDefaultMaxSequenceLength = 4096;

    /// <summary>
    /// Uploads the CPU model's MTP block (issue #1188). Resident MoE only, and never with CPU-placed MoE layers: the head drafts every round,
    /// so a per-forward upload or a host round trip would cost more than it saves.
    /// </summary>
    /// <summary>
    /// Opt-in (<c>DOTLLM_VK_QWEN35MOE_MTP=1</c>) until the verify path is exact and a clear win: on Qwen3.6-35B-A3B (Strix) the multi-row MoE
    /// kernels are not bit-identical to the fused 1-row decode chain (greedy MTP diverged from plain decode on 10 of 12 runs, at near-ties) and
    /// the speed-up was 0.97-1.09x (K=2..4) against llama.cpp's 1.46x. Without the opt-in the head is not uploaded (no memory cost).
    /// </summary>
    internal static bool MtpOptIn { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN35MOE_MTP") == "1";

    private void UploadMtpHeadIfPresent()
    {
        if (!MtpOptIn || _cpuModel?.MtpHead is not { } cpu || !_residentMoeEnabled || _cpuMoeLayer.Any(c => c))
            return;
        int h = Config.HiddenSize;
        long staging = 64;
        foreach (var (o, i, qt) in new[]
                 {
                     (cpu.Attn.QOutputDim, cpu.Attn.QInputDim, cpu.Attn.QQuantType), (cpu.Attn.KOutputDim, cpu.Attn.KInputDim, cpu.Attn.KQuantType),
                     (cpu.Attn.VOutputDim, cpu.Attn.VInputDim, cpu.Attn.VQuantType), (cpu.Attn.OOutputDim, cpu.Attn.OInputDim, cpu.Attn.OQuantType),
                     (cpu.EhProjOut, cpu.EhProjIn, cpu.EhProjQt),
                 })
            staging = Math.Max(staging, VulkanQwen3MoeHybridWeights.ProjectionUploadBytes(o, i, qt));
        staging = Math.Max(staging, (long)Math.Max(h, cpu.Attn.QNormWeight.Length) * sizeof(float));
        using var st = VulkanStagingBuffer.Create(_device, staging);
        var attn = VulkanQwen3MoeHybridWeights.UploadFullAttnLayer(_device, st, cpu.Attn, out _);
        var eh = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, st, cpu.EhProj, cpu.EhProjQt, cpu.EhProjOut, cpu.EhProjIn,
            forceF32: false, out var ehQt, out _);
        _mtpHead = new MtpHeadDevice
        {
            AttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, st, cpu.AttnNorm),
            PostAttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, st, cpu.PostAttnNorm),
            Attention = attn,
            Moe = VulkanQwen3MoeMoeUpload.UploadLayer(_device, cpu.Moe, h, residentQuant: true),
            EhProj = eh,
            EhProjQt = ehQt,
            EhProjIn = cpu.EhProjIn,
            EhProjOut = cpu.EhProjOut,
            Enorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, st, cpu.Enorm),
            Hnorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, st, cpu.Hnorm),
            SharedHeadNorm = cpu.SharedHeadNorm is { } shn ? VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, st, shn) : null,
        };
    }

    private void DisposeMtp()
    {
        _mtpHead?.Dispose();
        _mtpHead = null;
        _mtpScratch?.Dispose();
        _mtpAbsorbScratch?.Dispose();
        _mtpPairUpload?.Dispose();
    }

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => CreateMtpState(Math.Max(MtpDefaultMaxSequenceLength, Config.MaxSequenceLength));

    /// <summary>
    /// Stores <paramref name="rows"/> F32 K/V rows from the model's scratch into the head's cache at rows <c>[firstRow, +rows)</c>
    /// in the head's storage (issue #1141): a plain copy for the F32 reference arm, else the F32 to F16 convert kernel. Brackets
    /// the writes with the barriers (compute before, compute-or-transfer to compute after).
    /// </summary>
    private void RecordMtpKvStore(nint cmd, VulkanMtpState state, int firstRow, int rows, int kvStride)
    {
        if (state.Storage == VulkanKvStorage.F32)
        {
            KernelSupport.ComputeToTransferBarrier(cmd);
            ulong rowBytes = (ulong)state.KvRowBytes;
            RecordCopyBufferRange(cmd, _state.K, state.KeyCache, 0, (ulong)firstRow * rowBytes, (ulong)rows * rowBytes);
            RecordCopyBufferRange(cmd, _state.V, state.ValueCache, 0, (ulong)firstRow * rowBytes, (ulong)rows * rowBytes);
            KernelSupport.TransferToComputeBarrier(cmd);
            return;
        }

        var writer = _kernels.EnsureNarrowKv(VulkanKvStorage.F16, Config.HeadDim).Writer;
        KernelSupport.ComputeToComputeBarrier(cmd);
        long first = (long)firstRow * kvStride, count = (long)rows * kvStride;
        writer.RecordF32ToF16(cmd, _state.K, state.KeyCache, first, count);
        writer.RecordF32ToF16(cmd, _state.V, state.ValueCache, first, count);
        KernelSupport.ComputeToComputeBarrier(cmd);
    }

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
        => _mtpHead is null ? null : new VulkanMtpState(_device, Config.HiddenSize, _mtpHead.Attention.NumKvHeads, Config.HeadDim, maxSequenceLength);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                           ILoraAdapter? adapter, IMtpState? mtpState)
    {
        if (adapter is not null)
            throw new NotSupportedException("VulkanQwen3MoeHybridTransformerModel has no LoRA path.");
        return Forward(tokenIds, positions, deviceId, kvCache, gdnState: null, mtpState);
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var head = _mtpHead ?? throw new NotSupportedException($"{nameof(VulkanQwen3MoeHybridTransformerModel)} has no MTP head (SupportsMtp=false).");
        if (state is not VulkanMtpState mtp)
            throw new ArgumentException($"{nameof(VulkanQwen3MoeHybridTransformerModel)} requires a VulkanMtpState; got {state.GetType().Name}.", nameof(state));
        if ((uint)tokenId >= (uint)Config.VocabSize)
            throw new ArgumentOutOfRangeException(nameof(tokenId));
        BindMtpState(mtp);
        return DraftStep(head, mtp, tokenId, position);
    }

    // A disposed state's buffer handles can be recycled into a new one's; the kernels key descriptor sets on handles.
    private void BindMtpState(VulkanMtpState mtp)
    {
        if (ReferenceEquals(_lastMtpState, mtp))
            return;
        _kernels.InvalidateAll();
        _iqF16Prefill?.InvalidateDescriptorCache();
        _lastMtpState = mtp;
    }

    /// <summary>
    /// After the trunk's layers: downloads <c>HiddenState</c>'s <paramref name="seqLen"/> rows and applies the output norm on the host
    /// (llama.cpp's <c>h_nextn</c>, every row — the device final norm covers only the logit rows), then hands them to the state.
    /// </summary>
    private void CaptureMtpRows(VulkanMtpState state, int seqLen)
    {
        int h = Config.HiddenSize;
        int n = checked(seqLen * h);
        if (_mtpRows.Length < n) _mtpRows = new float[n];
        var rows = _mtpRows.AsSpan(0, n);
        _device.Download(_state.HiddenState, rows);
        if (_outputNormHost.Length == 0)
        {
            _outputNormHost = new float[h];
            _device.Download(_weights.OutputNormWeight, _outputNormHost);
        }
        for (int r = 0; r < seqLen; r++)
        {
            var row = rows.Slice(r * h, h);
            DotLLM.Cpu.Kernels.RmsNorm.Execute(row, _outputNormHost, Config.NormEpsilon, row);
        }
        state.SetCapturedRows(rows, seqLen);
    }

    /// <summary>
    /// K/V-only absorb of a contiguous trunk batch into the head's cache slots, one submit (the dense head's #472 batched absorb). Token
    /// <c>i</c> pairs with <c>h_{p-1}</c>: the carried row for <c>i == 0</c>, captured row <c>i - 1</c> otherwise.
    /// </summary>
    private void AbsorbMtp(VulkanMtpState state, ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions)
    {
        var head = _mtpHead!;
        BindMtpState(state);
        int s = tokenIds.Length, h = Config.HiddenSize, hd = Config.HeadDim, kvHeads = head.Attention.NumKvHeads;
        for (int i = 1; i < s; i++)
        {
            if (positions[i] != positions[0] + i)
                throw new NotSupportedException("The qwen35moe MTP absorb needs contiguous positions (prefill and verify batches always are).");
        }
        float eps = Config.NormEpsilon;
        int first = positions[0];
        state.BeginAbsorb(first, s);

        // Allocations before recording: a grow invalidates the handle-keyed descriptor caches.
        bool resized = _state.EnsureCapacity(s);
        if (_mtpAbsorbScratch is null || _mtpAbsorbRows < s)
        {
            _mtpAbsorbScratch?.Dispose();
            _mtpPairUpload?.Dispose();
            _mtpAbsorbScratch = MtpScratch.Allocate(_device, s, h);
            _mtpPairUpload = _device.Allocate((long)s * h * sizeof(float));
            _mtpAbsorbRows = s;
            resized = true;
        }
        if (resized) { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }
        var sc = _mtpAbsorbScratch;

        int n = checked(s * h);
        if (_mtpRows.Length < n) _mtpRows = new float[n];
        var pair = _mtpRows.AsSpan(0, n);
        state.CopyAbsorbPairingRows(s, pair);
        _device.Upload((ReadOnlySpan<float>)pair, _mtpPairUpload!);
        UploadPositions(positions);

        long rowBytes = (long)h * sizeof(float);
        _submit.Begin();
        nint cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        RecordEmbeddingGather(cmd, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, _state.HiddenState, head.Enorm, sc.ENorm, rowCount: s, n: h, eps: eps);
        _kernels.RmsNorm.Record(cmd, _mtpPairUpload!, head.Hnorm, sc.HNorm, rowCount: s, n: h, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmd);
        for (int i = 0; i < s; i++)
        {
            ulong rowOff = (ulong)(i * rowBytes), catOff = (ulong)(2 * i * rowBytes);
            RecordCopyBufferRange(cmd, sc.ENorm, sc.Concat, rowOff, catOff, (ulong)rowBytes);
            RecordCopyBufferRange(cmd, sc.HNorm, sc.Concat, rowOff, catOff + (ulong)rowBytes, (ulong)rowBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmd);
        RecordMatmul(cmd, head.EhProj, head.EhProjQt, sc.Concat, sc.Cur, outputDim: head.EhProjOut, inputDim: head.EhProjIn, seqLen: s);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, sc.Cur, head.AttnNorm, sc.Normed, rowCount: s, n: h, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        var a = head.Attention;
        RecordMatmul(cmd, a.KWeight, a.KDeviceQuantType, sc.Normed, _state.K, outputDim: a.KOutputDim, inputDim: a.KInputDim, seqLen: s);
        RecordMatmul(cmd, a.VWeight, a.VDeviceQuantType, sc.Normed, _state.V, outputDim: a.VOutputDim, inputDim: a.VInputDim, seqLen: s);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, _state.K, a.KNormWeight, _state.K, rowCount: s * kvHeads, n: hd, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        // The RoPE kernel always rotates a Q operand too; give it one head of the (dead) Q scratch.
        _kernels.Rope.Record(cmd, _state.Q, _state.K, _state.PositionsBuffer, seqLen: s, numHeads: 1, numKvHeads: kvHeads,
            headDim: hd, ropeDim: _ropeDim, theta: _ropeTheta, variant: RopeF32Kernel.Variant.NeoX);
        RecordMtpKvStore(cmd, state, first, s, kvHeads * hd);
        KernelSupport.ComputeTransferFullBarrier(cmd);
        KernelSupport.ComputeToHostBarrier(cmd);
        _submit.SubmitAndWait();
        state.EndAbsorb(first + s);
        state.SeedFromCapturedRow(s - 1);
    }

    /// <summary>One head step at <paramref name="position"/> from the state's pending hidden; returns [1, vocab] logits. One submit.</summary>
    private ITensor DraftStep(MtpHeadDevice head, VulkanMtpState state, int tokenId, int position)
    {
        int h = Config.HiddenSize, vocab = Config.VocabSize, heads = Config.NumAttentionHeads, hd = Config.HeadDim;
        var a = head.Attention;
        int kvHeads = a.NumKvHeads, qElems = heads * hd;
        float eps = Config.NormEpsilon;

        // Slot p holds the pair (h_{p-1}, x_p): writing at p discards anything speculative beyond it; a gap is a protocol error.
        if (state.CurrentLength > position)
            state.Rollback(position);
        else if (state.CurrentLength < position)
            throw new InvalidOperationException(
                $"MTP step at position {position} but the head KV covers {state.CurrentLength} positions: every trunk Forward of the sequence must pass the MTP state.");
        if (position >= state.MaxSteps || position >= Config.MaxSequenceLength)
            throw new InvalidOperationException($"MTP head KV exhausted at position {position} (MaxSteps={state.MaxSteps}).");

        bool resized = _state.EnsureCapacity(1);
        if (_mtpScratch is null) { _mtpScratch = MtpScratch.Allocate(_device, 1, h); resized = true; }
        if (resized) { _kernels.InvalidateAll(); _iqF16Prefill?.InvalidateDescriptorCache(); }
        var sc = _mtpScratch;
        UploadPositions([position]);
        long rowBytes = (long)h * sizeof(float);

        _submit.Begin();
        nint cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        RecordEmbeddingGather(cmd, [tokenId]);
        KernelSupport.TransferToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, _state.HiddenState, head.Enorm, sc.ENorm, rowCount: 1, n: h, eps: eps);
        _kernels.RmsNorm.Record(cmd, state.PendingHidden, head.Hnorm, sc.HNorm, rowCount: 1, n: h, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmd);
        RecordCopyBufferRange(cmd, sc.ENorm, sc.Concat, 0, 0, (ulong)rowBytes);
        RecordCopyBufferRange(cmd, sc.HNorm, sc.Concat, 0, (ulong)rowBytes, (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmd);
        RecordMatmul(cmd, head.EhProj, head.EhProjQt, sc.Concat, sc.Residual, outputDim: head.EhProjOut, inputDim: head.EhProjIn, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmd);   // Residual = inpSA (the eh_proj output)

        // Gated attention over the head's own KV.
        _kernels.RmsNorm.Record(cmd, sc.Residual, head.AttnNorm, sc.Normed, rowCount: 1, n: h, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        RecordMatmul(cmd, a.QWeight, a.QDeviceQuantType, sc.Normed, _state.QGateScratch, outputDim: a.QOutputDim, inputDim: a.QInputDim, seqLen: 1);
        RecordMatmul(cmd, a.KWeight, a.KDeviceQuantType, sc.Normed, _state.K, outputDim: a.KOutputDim, inputDim: a.KInputDim, seqLen: 1);
        RecordMatmul(cmd, a.VWeight, a.VDeviceQuantType, sc.Normed, _state.V, outputDim: a.VOutputDim, inputDim: a.VInputDim, seqLen: 1);
        KernelSupport.ComputeToTransferBarrier(cmd);
        long headBytes = (long)hd * sizeof(float);
        for (int i = 0; i < heads; i++)
        {
            ulong qg = (ulong)(2 * i * headBytes), q = (ulong)(i * headBytes);
            RecordCopyBufferRange(cmd, _state.QGateScratch, _state.Q, qg, q, (ulong)headBytes);
            RecordCopyBufferRange(cmd, _state.QGateScratch, _state.GateScratch, qg + (ulong)headBytes, q, (ulong)headBytes);
        }
        KernelSupport.TransferToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, _state.Q, a.QNormWeight, _state.Q, rowCount: heads, n: hd, eps: eps);
        _kernels.RmsNorm.Record(cmd, _state.K, a.KNormWeight, _state.K, rowCount: kvHeads, n: hd, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.Rope.Record(cmd, _state.Q, _state.K, _state.PositionsBuffer, seqLen: 1, numHeads: heads, numKvHeads: kvHeads,
            headDim: hd, ropeDim: _ropeDim, theta: _ropeTheta, variant: RopeF32Kernel.Variant.NeoX);
        RecordMtpKvStore(cmd, state, position, 1, kvHeads * hd);
        var headAttention = state.Storage == VulkanKvStorage.F32
            ? _kernels.Attention : _kernels.EnsureNarrowKv(VulkanKvStorage.F16, hd).Attention;
        headAttention.Record(cmd, _state.Q, state.KeyCache, state.ValueCache, _state.AttnOutput,
            seqQ: 1, seqKv: position + 1, numHeads: heads, numKvHeads: kvHeads, headDim: hd, positionOffset: position, slidingWindow: 0);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.SigmoidGateMul.Record(cmd, _state.AttnOutput, _state.GateScratch, nTotal: qElems);
        KernelSupport.ComputeToComputeBarrier(cmd);
        RecordMatmul(cmd, a.OWeight, a.ODeviceQuantType, _state.AttnOutput, sc.Cur, outputDim: a.OOutputDim, inputDim: a.OInputDim, seqLen: 1);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.Add.Record(cmd, sc.Residual, sc.Cur, sc.AddOut, h);   // AddOut = attention block output = FFN residual
        KernelSupport.ComputeToComputeBarrier(cmd);

        // MoE FFN through the trunk recorder, which takes the normed row in NormOutput (routed experts) AND re-norms HiddenState for the
        // shared expert, exactly as a trunk layer leaves them; the expert sum lands in NormOutput.
        KernelSupport.ComputeToTransferBarrier(cmd);
        RecordCopyBufferRange(cmd, sc.AddOut, _state.HiddenState, 0, 0, (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmd);
        _kernels.RmsNorm.Record(cmd, _state.HiddenState, head.PostAttnNorm, _state.NormOutput, rowCount: 1, n: h, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        RecordMoeLayer(cmd, head.Moe, head.PostAttnNorm, 1, h, eps);
        KernelSupport.ComputeToComputeBarrier(cmd);
        _kernels.Add.Record(cmd, sc.AddOut, _state.NormOutput, sc.Cur, h);
        KernelSupport.ComputeToComputeBarrier(cmd);

        // Shared-head norm (the next step chains this post-norm row, #469), then the trunk's LM head.
        _kernels.RmsNorm.Record(cmd, sc.Cur, head.SharedHeadNorm ?? _weights.OutputNormWeight, _state.NormOutput, rowCount: 1, n: h, eps: eps);
        KernelSupport.ComputeToTransferBarrier(cmd);
        RecordCopyBufferRange(cmd, _state.NormOutput, state.PendingHidden, 0, 0, (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmd);
        RecordMatmul(cmd, _weights.OutputWeight, _weights.OutputDeviceQuantType, _state.NormOutput, _state.Logits,
            outputDim: _weights.OutputOutputDim, inputDim: _weights.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmd);
        _submit.SubmitAndWait();
        state.Advance();

        var result = UnmanagedTensor.Allocate(new TensorShape(1, vocab), DType.Float32, deviceId: -1);
        unsafe { _device.Download(_state.Logits, new Span<float>((void*)result.DataPointer, vocab)); }
        return result;
    }
}
