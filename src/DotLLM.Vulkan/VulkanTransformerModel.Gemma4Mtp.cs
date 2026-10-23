using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;

using QuantType = DotLLM.Core.Configuration.QuantizationType;

namespace DotLLM.Vulkan;

/// <summary>
/// Gemma 4 MTP speculative decoding support for <see cref="VulkanTransformerModel"/> (issue #959).
/// The drafter is a standalone 4-layer mini-transformer (78M params for E4B) with Q-only cross-
/// attention into the trunk's KV-cache — no separate drafter KV-cache. All draft steps use a
/// constant RoPE position (the last trunk token's position), and the <c>post_projection</c> output
/// feeds back as the next step's hidden-state input.
/// </summary>
public sealed partial class VulkanTransformerModel : IMtpHeadAttachable
{
    private Gemma4MtpDeviceHead? _g4mtp;
    private float[]? _g4mtpLogits;

    /// <summary>
    /// Device-resident drafter weights and scratch buffers for a single seqLen=1 draft step.
    /// All buffers are allocated at <see cref="AttachMtpHead"/> time and live until the model is disposed.
    /// </summary>
    private sealed class Gemma4MtpDeviceHead : IDisposable
    {
        public required Gemma4MtpHeadWeights CpuWeights;
        public required GgufFile OwnedFile;

        // ── Global weight buffers ──
        public required VulkanDevice.Buffer TokenEmbd;
        public required QuantType TokenEmbdQt;
        public required VulkanDevice.Buffer PreProjection;
        public required QuantType PreProjectionQt;
        public required VulkanDevice.Buffer PostProjection;
        public required QuantType PostProjectionQt;
        public required VulkanDevice.Buffer OutputNorm;

        // ── Per-layer weight buffers ──
        public required Gemma4MtpDeviceLayer[] Layers;

        /// <summary>Maps drafter layer index → trunk layer index for KV-cache reads (cross-attention donor).</summary>
        public required int[] TrunkDonorLayers;

        // ── Scratch buffers (seqLen=1, drafter-sized) ──
        /// <summary>[1, drafterHidden] — working hidden state inside the drafter.</summary>
        public required VulkanDevice.Buffer Hidden;
        /// <summary>[1, drafterHidden] — alternate hidden slot for residual adds.</summary>
        public required VulkanDevice.Buffer HiddenAlt;
        /// <summary>[1, drafterHidden] — normalized hidden for projections.</summary>
        public required VulkanDevice.Buffer NormOut;
        /// <summary>[1, max(qDim)] — Q projection output.</summary>
        public required VulkanDevice.Buffer Q;
        /// <summary>[1, max(qDim)] — attention output.</summary>
        public required VulkanDevice.Buffer AttnOut;
        /// <summary>[1, intermediateSize] — FFN gate output.</summary>
        public required VulkanDevice.Buffer FfnGate;
        /// <summary>[1, intermediateSize] — FFN up output.</summary>
        public required VulkanDevice.Buffer FfnUp;
        /// <summary>[1, intermediateSize] — SiLU/GeGLU output.</summary>
        public required VulkanDevice.Buffer SiluOut;
        /// <summary>[1, trunkHidden] — pre-projection input (concat of scaled embed + trunk hidden).</summary>
        public required VulkanDevice.Buffer PreProjInput;
        /// <summary>[1, trunkHidden] — post-projection output (drafter → trunk space).</summary>
        public required VulkanDevice.Buffer PostProjOut;
        /// <summary>[1, vocab] — logits (drafter's <c>token_embd</c> used as output projection).</summary>
        public required VulkanDevice.Buffer Logits;

        public void Dispose()
        {
            TokenEmbd.Dispose(); PreProjection.Dispose(); PostProjection.Dispose(); OutputNorm.Dispose();
            foreach (var l in Layers) l.Dispose();
            Hidden.Dispose(); HiddenAlt.Dispose(); NormOut.Dispose(); Q.Dispose(); AttnOut.Dispose();
            FfnGate.Dispose(); FfnUp.Dispose(); SiluOut.Dispose();
            PreProjInput.Dispose(); PostProjOut.Dispose(); Logits.Dispose();
            OwnedFile.Dispose();
        }
    }

    private sealed class Gemma4MtpDeviceLayer : IDisposable
    {
        public required VulkanDevice.Buffer AttnNorm;
        public required VulkanDevice.Buffer AttnQ;
        public required QuantType AttnQQt;
        public required int AttnQOutputDim;
        public required VulkanDevice.Buffer AttnQNorm;
        public required VulkanDevice.Buffer AttnOutput;
        public required QuantType AttnOutputQt;
        public required VulkanDevice.Buffer PostAttnNorm;
        public required VulkanDevice.Buffer FfnNorm;
        public required VulkanDevice.Buffer FfnGate;
        public required QuantType FfnGateQt;
        public required VulkanDevice.Buffer FfnUp;
        public required QuantType FfnUpQt;
        public required VulkanDevice.Buffer FfnDown;
        public required QuantType FfnDownQt;
        public required VulkanDevice.Buffer PostFfwNorm;
        public required float LayerOutputScale;
        public required bool IsSlidingWindow;
        public required int HeadDim;

        public void Dispose()
        {
            AttnNorm.Dispose(); AttnQ.Dispose(); AttnQNorm.Dispose();
            AttnOutput.Dispose(); PostAttnNorm.Dispose();
            FfnNorm.Dispose(); FfnGate.Dispose(); FfnUp.Dispose(); FfnDown.Dispose();
            PostFfwNorm.Dispose();
        }
    }

    // ───────────────────────────── IMtpHeadAttachable ─────────────────────────────

    /// <inheritdoc/>
    public bool HasMtpHead => _g4mtp is not null;

    /// <inheritdoc/>
    public void AttachMtpHead(string path)
    {
        if (_g4mtp is not null) throw new InvalidOperationException("An MTP head is already attached.");
        var file = GgufFile.Open(path);
        try
        {
            var cpuWeights = Gemma4MtpHeadLoader.Load(file, Config.HiddenSize);
            UploadGemma4MtpHead(file, cpuWeights);
        }
        catch { file.Dispose(); throw; }
    }

    private unsafe void UploadGemma4MtpHead(GgufFile file, Gemma4MtpHeadWeights w)
    {
        int dH = w.HiddenSize;
        int tH = w.TrunkHiddenSize;
        int inter = w.IntermediateSize;
        int maxQDim = 0;
        for (int i = 0; i < w.NumLayers; i++)
            maxQDim = Math.Max(maxQDim, w.Layers[i].AttnQOutputDim);

        var deviceLayers = new Gemma4MtpDeviceLayer[w.NumLayers];
        using (var staging = VulkanStagingBuffer.Create(_device, 64L << 20))
        {
            var embd = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                _device, staging, w.TokenEmbdWeight, w.TokenEmbdQuantType,
                w.VocabSize, dH, forceF32: false, out var embdQt, out _);

            var preProjBuf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                _device, staging, w.PreProjectionWeight, w.PreProjectionQuantType,
                dH, tH * 2, forceF32: false, out var preProjQt, out _);

            var postProjBuf = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                _device, staging, w.PostProjectionWeight, w.PostProjectionQuantType,
                tH, dH, forceF32: false, out var postProjQt, out _);

            var outNormBuf = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, w.OutputNormWeight);

            for (int i = 0; i < w.NumLayers; i++)
            {
                ref var lw = ref w.Layers[i];
                deviceLayers[i] = new Gemma4MtpDeviceLayer
                {
                    AttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, lw.AttnNormWeight),
                    AttnQ = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                        _device, staging, lw.AttnQWeight, lw.AttnQQuantType,
                        lw.AttnQOutputDim, dH, forceF32: false, out var qQt, out _),
                    AttnQQt = qQt,
                    AttnQOutputDim = lw.AttnQOutputDim,
                    AttnQNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, lw.AttnQNormWeight),
                    AttnOutput = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                        _device, staging, lw.AttnOutputWeight, lw.AttnOutputQuantType,
                        dH, lw.AttnQOutputDim, forceF32: false, out var oQt, out _),
                    AttnOutputQt = oQt,
                    PostAttnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, lw.PostAttentionNormWeight),
                    FfnNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, lw.FfnNormWeight),
                    FfnGate = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                        _device, staging, lw.FfnGateWeight, lw.FfnGateQuantType,
                        inter, dH, forceF32: false, out var gateQt, out _),
                    FfnGateQt = gateQt,
                    FfnUp = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                        _device, staging, lw.FfnUpWeight, lw.FfnUpQuantType,
                        inter, dH, forceF32: false, out var upQt, out _),
                    FfnUpQt = upQt,
                    FfnDown = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(
                        _device, staging, lw.FfnDownWeight, lw.FfnDownQuantType,
                        dH, inter, forceF32: false, out var downQt, out _),
                    FfnDownQt = downQt,
                    PostFfwNorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, lw.PostFfwNormWeight),
                    LayerOutputScale = lw.LayerOutputScale,
                    IsSlidingWindow = lw.IsSlidingWindow,
                    HeadDim = lw.HeadDim,
                };
            }

            // Compute trunk donor layer for each drafter layer: SWA → last trunk own-KV SWA layer,
            // full → last trunk own-KV full layer. Mirrors Config.SharedKvDonorLayer for shared-KV trunk layers.
            int kvFromStart = Config.NumLayers - Config.NumSharedKvLayers;
            int swaDonor = kvFromStart - 2;
            int fullDonor = kvFromStart - 1;
            var donorLayers = new int[w.NumLayers];
            for (int i = 0; i < w.NumLayers; i++)
                donorLayers[i] = w.Layers[i].IsSlidingWindow ? swaDonor : fullDonor;

            _g4mtp = new Gemma4MtpDeviceHead
            {
                CpuWeights = w,
                OwnedFile = file,
                TokenEmbd = embd,
                TokenEmbdQt = embdQt,
                PreProjection = preProjBuf,
                PreProjectionQt = preProjQt,
                PostProjection = postProjBuf,
                PostProjectionQt = postProjQt,
                OutputNorm = outNormBuf,
                Layers = deviceLayers,
                TrunkDonorLayers = donorLayers,
                // Scratch buffers (all seqLen=1)
                Hidden = _device.AllocateDeviceLocal(dH * sizeof(float)),
                HiddenAlt = _device.AllocateDeviceLocal(dH * sizeof(float)),
                NormOut = _device.AllocateDeviceLocal(dH * sizeof(float)),
                Q = _device.AllocateDeviceLocal(maxQDim * sizeof(float)),
                AttnOut = _device.AllocateDeviceLocal(maxQDim * sizeof(float)),
                FfnGate = _device.AllocateDeviceLocal(inter * sizeof(float)),
                FfnUp = _device.AllocateDeviceLocal(inter * sizeof(float)),
                SiluOut = _device.AllocateDeviceLocal(inter * sizeof(float)),
                PreProjInput = _device.AllocateDeviceLocal(tH * 2 * sizeof(float)),
                PostProjOut = _device.AllocateDeviceLocal(tH * sizeof(float)),
                Logits = _device.AllocateDeviceLocal((long)Config.VocabSize * sizeof(float)),
            };
        }
    }

    // ───────────────────────────── IModel MTP surface ─────────────────────────────

    /// <inheritdoc/>
    public bool SupportsMtp => _g4mtp is not null;

    /// <inheritdoc/>
    public long MtpGatePriorBytes => long.MaxValue;

    /// <inheritdoc/>
    public int MtpPreferredDraftTokens => _g4mtp?.CpuWeights.NextnPredictLayers ?? 0;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => CreateMtpState(Config.MaxSequenceLength);

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
    {
        if (_g4mtp is null) return null;
        return new VulkanGemma4MtpState(Config.HiddenSize, _g4mtp.CpuWeights.HiddenSize);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
    {
        var result = Forward(tokenIds, positions, deviceId, kvCache, adapter);
        if (mtpState is VulkanGemma4MtpState g4state && _g4mtp is not null)
        {
            g4state.TrunkKvCache = kvCache as VulkanKvCache;
            CaptureHiddenForGemma4Mtp(g4state, tokenIds.Length, positions);
        }
        return result;
    }

    /// <summary>
    /// After a trunk forward, downloads the post-final-norm hidden state into <paramref name="state"/>'s
    /// captured rows. The forward has already submitted and waited, so the GPU state is quiescent.
    /// </summary>
    private unsafe void CaptureHiddenForGemma4Mtp(VulkanGemma4MtpState state, int seqLen, ReadOnlySpan<int> positions)
    {
        int h = Config.HiddenSize;
        int n = Math.Min(seqLen, VulkanGemma4MtpState.MaxCaptureRows);
        float[] buf = new float[n * h];

        long rowBytes = (long)h * sizeof(float);
        int offset = seqLen - n;
        long srcOffset = (long)offset * rowBytes;

        float* raw = (float*)NativeMemory.AlignedAlloc((nuint)(n * rowBytes), 64);
        try
        {
            using var staging = _device.Allocate(n * rowBytes);
            _device.CopyBufferRangeSynchronous(_state.HiddenState, staging,
                srcOffset: (ulong)srcOffset, dstOffset: 0, size: (ulong)(n * rowBytes));
            _device.Download(staging, new Span<float>(raw, n * h));

            var normWeight = _cpuWeights.OutputNormWeight.AsSpan();
            float eps = Config.NormEpsilon;
            for (int r = 0; r < n; r++)
            {
                var row = new Span<float>(raw + r * h, h);
                RmsNormCpu(row, normWeight, eps, buf.AsSpan(r * h, h));
            }
        }
        finally
        {
            NativeMemory.AlignedFree(raw);
        }

        state.CaptureRows(buf, n, positions[0]);
        state.SeedFromCapturedRow(n - 1);
    }

    private static void RmsNormCpu(ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float eps, Span<float> output)
    {
        float sumSq = 0;
        for (int i = 0; i < input.Length; i++)
            sumSq += input[i] * input[i];
        float scale = 1.0f / MathF.Sqrt(sumSq / input.Length + eps);
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] * scale * weight[i];
    }

    // ───────────────────────────── draft step ─────────────────────────────

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _g4mtp is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        var g4 = state as VulkanGemma4MtpState
            ?? throw new ArgumentException("Expected VulkanGemma4MtpState.", nameof(state));
        return RunGemma4DraftStep(g4, tokenId, position, default);
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var g4 = state as VulkanGemma4MtpState
            ?? throw new ArgumentException("Expected VulkanGemma4MtpState.", nameof(state));
        int vocab = Config.VocabSize;
        var result = UnmanagedTensor.Allocate(new TensorShape(1, vocab), DType.Float32, deviceId: -1);
        try
        {
            unsafe
            {
                RunGemma4DraftStep(g4, tokenId, position, new Span<float>((void*)result.DataPointer, vocab));
            }
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    /// <summary>
    /// Runs one Gemma 4 MTP draft step on the GPU. Flow (llama.cpp gemma4-assistant):
    /// <c>concat(trunkEmbed(token) * √tH, trunkHidden)</c> [5120] → <c>pre_projection</c> → [256] →
    /// 4 decoder layers (Q-only cross-attention into trunk KV + GeGLU FFN, Gemma-4 4-norm residual) →
    /// <c>output_norm</c> → drafter's <c>token_embd</c> as LM head → logits; <c>post_projection</c> [256→2560]
    /// feeds back as the next step's trunk hidden input. Returns the argmax token. When <paramref name="logits"/>
    /// is non-empty, copies the full logit row into it.
    /// </summary>
    private unsafe int RunGemma4DraftStep(VulkanGemma4MtpState st, int tokenId, int position, Span<float> logits)
    {
        var h = _g4mtp ?? throw new NotSupportedException("No Gemma 4 MTP head attached.");
        var w = h.CpuWeights;
        if (!st.HasSeed)
            throw new InvalidOperationException("MTP state has not been seeded (call SeedFromCapturedRow after the trunk forward).");
        var kvCache = st.TrunkKvCache
            ?? throw new InvalidOperationException("Trunk KV-cache reference not set (was Forward called with mtpState first?).");

        int dH = w.HiddenSize;       // 256
        int tH = w.TrunkHiddenSize;  // 2560
        int vocab = Config.VocabSize;
        float eps = Config.NormEpsilon;
        float embScale = MathF.Sqrt(tH);
        int numHeads = w.NumHeads;    // 4
        int numKvHeads = w.NumKvHeads; // 2

        // ── 1. Host: build pre_projection input [tH*2] = concat(trunkEmbed(token)*√tH, trunkHidden) ──
        float* preProjHost = (float*)NativeMemory.AlignedAlloc((nuint)(tH * 2 * sizeof(float)), 64);
        try
        {
            var embedHalf = new Span<float>(preProjHost, tH);
            DequantTokenEmbedRow(tokenId, embedHalf, tH);
            TensorPrimitives.Multiply(embedHalf, embScale, embedHalf);
            st.PendingTrunkHidden.AsSpan(0, tH).CopyTo(new Span<float>(preProjHost + tH, tH));
            _device.Upload(new ReadOnlySpan<float>(preProjHost, tH * 2), h.PreProjInput);
        }
        finally
        {
            NativeMemory.AlignedFree(preProjHost);
        }

        // Gemma 4 MTP: all draft steps use the SAME RoPE position — the last trunk token.
        int seqKv = kvCache.CurrentLength;
        int ropePos = seqKv - 1;
        Span<int> draftPos = [ropePos];
        UploadPositions(draftPos);

        // ── 2. Record GPU commands ──
        _submit.Begin();
        nint cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);

        // Pre-projection: concat [tH*2] → drafter space [dH]
        RecordMatmul(cmd, h.PreProjection, h.PreProjectionQt, h.PreProjInput, h.Hidden,
            outputDim: dH, inputDim: tH * 2, seqLen: 1);
        BarrierComputeToCompute(cmd);

        // ── 3. Decoder layers (Q-only cross-attention + GeGLU FFN, Gemma-4 4-norm residual) ──
        var curHidden = h.Hidden;
        var altHidden = h.HiddenAlt;

        for (int i = 0; i < w.NumLayers; i++)
        {
            var dl = h.Layers[i];
            int headDim = dl.HeadDim;
            int qDim = dl.AttnQOutputDim;
            int trunkDonor = h.TrunkDonorLayers[i];
            int slidingWindow = dl.IsSlidingWindow ? (Config.SlidingWindowSize ?? 0) : 0;
            var (ropeTheta, ropeDim) = GemmaLayerRope(trunkDonor);

            // Gemma-4 partial-rotary globals: freqDim=headDim, neoxPairOffset=headDim/2.
            bool partialGlobal = !dl.IsSlidingWindow && ropeDim < headDim;
            if (!dl.IsSlidingWindow && _proportionalRopePairs > 0)
            {
                ropeDim = 2 * _proportionalRopePairs;
                partialGlobal = ropeDim < headDim;
            }

            // attn_norm(hidden) → NormOut
            _rmsnorm.Record(cmd, curHidden, dl.AttnNorm, h.NormOut,
                rowCount: 1, n: dH, eps: eps);
            BarrierComputeToCompute(cmd);

            // Q projection: [dH] → [qDim]
            RecordMatmul(cmd, dl.AttnQ, dl.AttnQQt, h.NormOut, h.Q,
                outputDim: qDim, inputDim: dH, seqLen: 1);
            BarrierComputeToCompute(cmd);

            // Per-head Q RMSNorm
            _rmsnorm.Record(cmd, h.Q, dl.AttnQNorm, h.Q,
                rowCount: numHeads, n: headDim, eps: eps);
            BarrierComputeToCompute(cmd);

            // RoPE on Q only. K is already roped in the trunk's KV-cache; pass AttnOut
            // as a dummy K buffer with numKvHeads=1 (mirrors trunk shared-KV path).
            _rope.Record(cmd, h.Q, h.AttnOut, _state.PositionsBuffer,
                seqLen: 1, numHeads: numHeads, numKvHeads: 1,
                headDim: headDim, ropeDim: ropeDim, theta: ropeTheta,
                variant: RopeF32Kernel.Variant.NeoX,
                freqDim: partialGlobal ? headDim : 0,
                neoxPairOffset: partialGlobal ? headDim / 2 : (int?)null);
            BarrierComputeToCompute(cmd);

            // Cross-attention into trunk's KV-cache. positionOffset = ropePos (last trunk position)
            // satisfies the tight-span invariant and gives correct causal masking.
            var kSrc = kvCache.GetKeysBuffer(trunkDonor);
            var vSrc = kvCache.GetValuesBuffer(trunkDonor);
            RecordAttention(cmd, h.Q, kSrc, vSrc, h.AttnOut,
                seqQ: 1, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: ropePos, slidingWindow: slidingWindow,
                softCap: 0.0f, scaleOverride: GetAttentionScaleOverride());
            BarrierComputeToCompute(cmd);

            // O projection: [qDim] → [dH]
            RecordMatmul(cmd, dl.AttnOutput, dl.AttnOutputQt, h.AttnOut, h.NormOut,
                outputDim: dH, inputDim: qDim, seqLen: 1);
            BarrierComputeToCompute(cmd);

            // post_attention_norm
            _rmsnorm.Record(cmd, h.NormOut, dl.PostAttnNorm, h.NormOut,
                rowCount: 1, n: dH, eps: eps);
            BarrierComputeToCompute(cmd);

            // Residual #1
            _add.Record(cmd, curHidden, h.NormOut, altHidden, dH);
            BarrierComputeToCompute(cmd);
            (curHidden, altHidden) = (altHidden, curHidden);

            // ffn_norm(hidden) → NormOut
            _rmsnorm.Record(cmd, curHidden, dl.FfnNorm, h.NormOut,
                rowCount: 1, n: dH, eps: eps);
            BarrierComputeToCompute(cmd);

            // GeGLU FFN: gate + up → GeGLU → down
            RecordMatmul(cmd, dl.FfnGate, dl.FfnGateQt, h.NormOut, h.FfnGate,
                outputDim: w.IntermediateSize, inputDim: dH, seqLen: 1);
            RecordMatmul(cmd, dl.FfnUp, dl.FfnUpQt, h.NormOut, h.FfnUp,
                outputDim: w.IntermediateSize, inputDim: dH, seqLen: 1);
            BarrierComputeToCompute(cmd);
            _geglu!.Record(cmd, h.FfnGate, h.FfnUp, h.SiluOut, w.IntermediateSize);
            BarrierComputeToCompute(cmd);
            RecordMatmul(cmd, dl.FfnDown, dl.FfnDownQt, h.SiluOut, h.NormOut,
                outputDim: dH, inputDim: w.IntermediateSize, seqLen: 1);
            BarrierComputeToCompute(cmd);

            // post_ffw_norm
            _rmsnorm.Record(cmd, h.NormOut, dl.PostFfwNorm, h.NormOut,
                rowCount: 1, n: dH, eps: eps);
            BarrierComputeToCompute(cmd);

            // Residual #2
            _add.Record(cmd, curHidden, h.NormOut, altHidden, dH);
            BarrierComputeToCompute(cmd);
            (curHidden, altHidden) = (altHidden, curHidden);

            // Layer output scale (in-place on curHidden)
            if (dl.LayerOutputScale != 1.0f)
            {
                _embedScale!.Record(cmd, curHidden, dH, dl.LayerOutputScale);
                BarrierComputeToCompute(cmd);
            }
        }

        // ── 4. Output norm → LM head → post-projection ──
        _rmsnorm.Record(cmd, curHidden, h.OutputNorm, h.NormOut,
            rowCount: 1, n: dH, eps: eps);
        BarrierComputeToCompute(cmd);

        // LM head: drafter's token_embd [dH→vocab] as output projection
        RecordMatmul(cmd, h.TokenEmbd, h.TokenEmbdQt, h.NormOut, h.Logits,
            outputDim: vocab, inputDim: dH, seqLen: 1);

        // Post-projection: normed hidden → trunk space [dH→tH] for next draft step
        RecordMatmul(cmd, h.PostProjection, h.PostProjectionQt, h.NormOut, h.PostProjOut,
            outputDim: tH, inputDim: dH, seqLen: 1);

        KernelSupport.ComputeToHostBarrier(cmd);

        // ── 5. Submit, download logits + post-projection ──
        _submit.SubmitAndWait();

        _g4mtpLogits ??= new float[vocab];
        _device.Download(h.Logits, _g4mtpLogits);
        if (!logits.IsEmpty)
            _g4mtpLogits.AsSpan().CopyTo(logits);
        int argmax = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)_g4mtpLogits);

        // Feed post-projection output back as the next step's trunk hidden
        _device.Download(h.PostProjOut, st.PendingTrunkHidden);

        return argmax;
    }
}
