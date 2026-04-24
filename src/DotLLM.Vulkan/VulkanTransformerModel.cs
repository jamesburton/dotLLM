using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

using QuantType = DotLLM.Core.Configuration.QuantizationType;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for Llama-family transformer models.
/// Implements <see cref="IModel"/> using the wave-1/wave-2 Vulkan compute
/// kernels: <see cref="MatMulF32Kernel"/> plus the Q8_0 matmul kernels
/// (<see cref="MatMulQ8_0Kernel"/> for decode-path GEMV,
/// <see cref="MatMulQ8_0GemmKernel"/> for batched prefill),
/// <see cref="RmsNormF32Kernel"/>, <see cref="RopeF32Kernel"/>,
/// <see cref="AttentionF32Kernel"/>, <see cref="SwiGluF32Kernel"/>, and
/// <see cref="AddKernel"/> for residuals.
/// </summary>
/// <remarks>
/// <para>
/// Q8_0 weights stay on device as 34-byte blocks and are consumed directly
/// by the Q8_0 matmul kernels — 4× less VRAM and 4× less bytes-per-forward
/// on the weight read vs the legacy dequantise-at-load path. Other quant
/// types (F16, K-quants) still dequantise to FP32 at load. All non-matmul
/// kernels remain F32; only the weight storage changes. The model assumes
/// a pure-Transformer Llama-family architecture — MLA, MoE, and SSM layers
/// are rejected at load time.
/// </para>
/// <para>
/// Forward pass is fence-pipelined: a single persistent command buffer
/// records every kernel dispatch + inter-kernel pipeline barrier for the
/// whole forward, submits once per forward, and waits on a single fence
/// before downloading logits. Legacy synchronous kernel launches (one
/// <c>vkQueueWaitIdle</c> per kernel) are only used by the standalone
/// unit tests.
/// </para>
/// <para>
/// Architectural parallel with <c>DotLLM.Cuda.CudaTransformerModel</c>:
/// upload weights once at construction, reuse a single
/// <see cref="VulkanForwardState"/> for scratch. Each linear projection
/// dispatches through <see cref="RecordMatmul"/> which picks
/// <c>matmul_q8_0</c> / <c>matmul_q8_0_gemm</c> / <c>matmul_f32</c> based on
/// the weight's device-side quant type and <c>seqLen</c>. Logits come back
/// as a single <see cref="UnmanagedTensor"/> of shape <c>[1, vocabSize]</c>
/// matching the CUDA return convention.
/// </para>
/// </remarks>
public sealed class VulkanTransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanWeights _weights;
    private readonly VulkanForwardState _state;

    // Kernels — one instance each, pipelines are reused across all launches.
    private readonly MatMulF32Kernel _matmul;
    private readonly MatMulQ8_0Kernel _matmulQ8;
    private readonly MatMulQ8_0GemmKernel _matmulQ8Gemm;
    // Optional: coopmat Q8_0 GEMM for prefill (seqLen>1) on devices that
    // advertise VK_KHR_cooperative_matrix. ~3.8× over the scalar GEMM on AMD
    // RDNA3.5 iGPU at Llama-3 4096² N=64 (790 vs 209 GFLOPS). Null on devices
    // without coopmat — the router falls back to _matmulQ8Gemm then.
    private readonly MatMulQ8_0GemmCoopmatKernel? _matmulQ8GemmCoopmat;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly RopeF32Kernel _rope;
    private readonly AttentionF32Kernel _attention;
    private readonly SwiGluF32Kernel _swiglu;
    private readonly AddKernel _add;

    // Persistent command buffer + fence used by Forward. One SubmitContext
    // per model — reset+begin at the start of each forward, submit+wait at
    // the end. Bias host-side steps split the forward into multiple submits
    // but each submit still batches many dispatches behind one fence.
    private readonly VulkanDevice.SubmitContext _submit;

    private readonly TransformerWeights _cpuWeights; // retained for embedding lookup
    private readonly GgufFile? _gguf;
    private readonly float _ropeTheta;
    private readonly int _ropeDim;
    private readonly RopeF32Kernel.Variant _ropeVariant;
    private readonly int _slidingWindow;
    private readonly bool _ownsDevice;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes;

    /// <summary>Creates a <see cref="VulkanKvCache"/> sized for this model.</summary>
    public VulkanKvCache CreateKvCache(int maxSeqLen)
        => new(_device, Config.NumLayers, Config.NumKvHeads, Config.HeadDim, maxSeqLen);

    private VulkanTransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, VulkanWeights weights, TransformerWeights cpuWeights,
        VulkanForwardState state,
        MatMulF32Kernel matmul, MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        RmsNormF32Kernel rmsnorm, RopeF32Kernel rope,
        AttentionF32Kernel attention, SwiGluF32Kernel swiglu, AddKernel add,
        VulkanDevice.SubmitContext submit,
        GgufFile? gguf,
        float ropeTheta, int ropeDim, RopeF32Kernel.Variant ropeVariant, int slidingWindow)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _cpuWeights = cpuWeights;
        _state = state;
        _matmul = matmul;
        _matmulQ8 = matmulQ8;
        _matmulQ8Gemm = matmulQ8Gemm;
        _matmulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        _rmsnorm = rmsnorm;
        _rope = rope;
        _attention = attention;
        _swiglu = swiglu;
        _add = add;
        _submit = submit;
        _gguf = gguf;
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
        _ropeVariant = ropeVariant;
        _slidingWindow = slidingWindow;
    }

    /// <summary>
    /// Loads a model from an opened GGUF file onto a new Vulkan device.
    /// The caller owns the returned model; disposing it tears down the
    /// device, pipelines, and weight buffers.
    /// </summary>
    /// <param name="gguf">Opened GGUF file. Must remain alive for the model's lifetime.</param>
    /// <param name="config">Model configuration extracted from the GGUF metadata.</param>
    /// <param name="spvDir">
    /// Directory containing the compiled Vulkan SPIR-V blobs. When null,
    /// falls back to <c>spv/</c> next to the running assembly (matches the
    /// MSBuild <c>Content</c> copy pattern used by the Vulkan project).
    /// </param>
    public static VulkanTransformerModel LoadFromGguf(GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        var device = VulkanDevice.Create();
        try
        {
            spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
            var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
            return BuildModel(device, ownsDevice: true, config, cpuWeights, spvDir, gguf);
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Loads a model onto an existing <see cref="VulkanDevice"/>. The device
    /// is NOT disposed when the model is disposed — the caller retains
    /// ownership. Useful when the device is shared with other Vulkan
    /// components (e.g. a diagnostic hook that wants to launch its own
    /// kernels on the same queue).
    /// </summary>
    public static VulkanTransformerModel LoadFromGguf(
        VulkanDevice device, GgufFile gguf, ModelConfig config, string? spvDir = null)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);

        RejectUnsupportedArchitecture(config);

        spvDir ??= Path.Combine(AppContext.BaseDirectory, "spv");
        var cpuWeights = TransformerWeights.LoadFromGguf(gguf, config);
        return BuildModel(device, ownsDevice: false, config, cpuWeights, spvDir, gguf);
    }

    private static VulkanTransformerModel BuildModel(
        VulkanDevice device, bool ownsDevice, ModelConfig config,
        TransformerWeights cpuWeights, string spvDir, GgufFile? gguf)
    {
        // Q8_0 matrices stay on device as 34-byte blocks — the forward pass
        // below dispatches them through the Q8_0 GEMV / GEMM kernels. Other
        // quant types are still dequantised to FP32 at upload.
        var weights = VulkanWeights.Upload(device, cpuWeights, config.NumLayers);

        var state = new VulkanForwardState(device,
            config.HiddenSize, config.NumAttentionHeads, config.NumKvHeads,
            config.HeadDim, config.IntermediateSize, config.VocabSize,
            initialSeqLen: 1);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        // Optional coopmat prefill GEMM — 3.8× over scalar on AMD RDNA3.5 at
        // Llama-3 4096² N=64. Null on devices without KHR_cooperative_matrix;
        // router falls back to the scalar GEMM. Tolerance: abs 5e-3 / rel 5e-3
        // end-to-end (looser than the 1e-4 / 1e-3 of the scalar path because
        // KHR_coopmat only offers F16 operands — see the coopmat kernel tests).
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* Kernel threw: no usable tile shape. Stay on scalar. */ }
        }
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var rope = RopeF32Kernel.Create(device, spvDir);
        var attention = AttentionF32Kernel.Create(device, spvDir);
        var swiglu = SwiGluF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);

        var submit = device.CreateSubmitContext();

        int ropeDim = config.RoPEConfig?.DimensionCount ?? config.HeadDim;
        if (ropeDim == 0) ropeDim = config.HeadDim;
        float ropeTheta = config.RoPEConfig?.Theta ?? 10000.0f;
        RoPEType ropeType = config.RoPEConfig?.Type ?? RoPEType.Norm;
        var ropeVariant = ropeType == RoPEType.NeoX ? RopeF32Kernel.Variant.NeoX : RopeF32Kernel.Variant.Norm;

        int slidingWindow = config.SlidingWindowSize ?? 0;

        return new VulkanTransformerModel(
            device, ownsDevice,
            config, weights, cpuWeights, state,
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            rmsnorm, rope, attention, swiglu, add,
            submit,
            gguf,
            ropeTheta, ropeDim, ropeVariant, slidingWindow);
    }

    private static void RejectUnsupportedArchitecture(ModelConfig config)
    {
        if (config.MlaConfig is not null)
            throw new NotSupportedException("MLA (DeepSeek-V2/V3) is not supported on the Vulkan backend yet.");
        if (config.Moe is not null)
            throw new NotSupportedException("MoE is not supported on the Vulkan backend yet.");
        if (config.HybridLayout is not null || config.SsmConfig is not null || config.Mamba3Config is not null)
            throw new NotSupportedException("Hybrid SSM / Mamba architectures are not supported on the Vulkan backend yet.");
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");

        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int intermediateSize = Config.IntermediateSize;
        int vocabSize = Config.VocabSize;
        float eps = Config.NormEpsilon;

        bool scratchResized = _state.EnsureCapacity(seqLen);

        // Descriptor sets cache buffer handles. When scratch is re-allocated
        // every cached set becomes stale and must be dropped — otherwise the
        // next dispatch binds a dangling VkBuffer. In steady-state decode
        // (seqLen = 1 after the initial prefill) scratch never grows, so the
        // cache stays warm across forwards.
        if (scratchResized)
            InvalidateKernelCaches();

        // 1. Validate token IDs (done host-side; cheap), then upload only
        //    positions host→device. The embedding table is device-local and
        //    populated once at construction; per-token rows are gathered into
        //    HiddenState via vkCmdCopyBuffer recorded on the same command
        //    buffer (see RecordEmbeddingGather below).
        ValidateTokenIds(tokenIds);
        UploadPositions(positions);

        // 2. Begin the single per-forward command buffer and record the
        //    whole transformer. Bias-add host steps split the forward into
        //    multiple submits (one per distinct set of biases we need to
        //    pause for); everything else stays inside the pipelined path.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);

        // Gather one embedding row per token from the device-local
        // TokenEmbedding buffer into HiddenState[t, :]. The first consumer
        // is the TRANSFER copy (HiddenState → Residual) at the top of the
        // layer loop; the second is the first RMSNorm's COMPUTE read — so
        // we need both TransferRead and ShaderRead visibility.
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToTransferAndComputeBarrier(cmdBuf);

        for (int layer = 0; layer < Config.NumLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];
            ref readonly var cpuLw = ref _cpuWeights.Layers[layer];

            // Residual snapshot (pre-attention): HiddenState → Residual.
            RecordCopyBuffer(cmdBuf, _state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf); // TRANSFER→COMPUTE would be tighter; COMPUTE→COMPUTE covers both paths

            // Attn RMSNorm
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.AttnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Q/K/V projections
            RecordMatmul(cmdBuf, lw.Q, lw.QDeviceQuantType, _state.NormOutput, _state.Q,
                lw.QOutputDim, lw.QInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.K, lw.KDeviceQuantType, _state.NormOutput, _state.K,
                lw.KOutputDim, lw.KInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.V, lw.VDeviceQuantType, _state.NormOutput, _state.V,
                lw.VOutputDim, lw.VInputDim, seqLen);

            // Optional QKV biases — host path. Submit, wait, write, re-begin.
            if (cpuLw.QBias is not null || cpuLw.KBias is not null || cpuLw.VBias is not null)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                if (cpuLw.QBias is { } qb) AddBiasRows(_state.Q, qb, lw.QOutputDim, seqLen);
                if (cpuLw.KBias is { } kb) AddBiasRows(_state.K, kb, lw.KOutputDim, seqLen);
                if (cpuLw.VBias is { } vb) AddBiasRows(_state.V, vb, lw.VOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // RoPE on Q and K
            _rope.Record(cmdBuf, _state.Q, _state.K, _state.PositionsBuffer,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: _ropeDim, theta: _ropeTheta,
                variant: _ropeVariant);

            // Attention input buffers: either the uncached K/V window or the full KV cache.
            VulkanDevice.Buffer kSrc, vSrc;
            int seqKv;
            int positionOffset;
            if (kvCache is VulkanKvCache vkCache)
            {
                // RoPE writes K; attention (via the cache buffers) reads K.
                // Barrier the RoPE → KV copy, then the KV copy → attention.
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
                KernelSupport.TransferToComputeBarrier(cmdBuf);
                kSrc = vkCache.GetKeysBuffer(layer);
                vSrc = vkCache.GetValuesBuffer(layer);
                seqKv = vkCache.CurrentLength;
                positionOffset = positions[0];
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
                kSrc = _state.K;
                vSrc = _state.V;
                seqKv = seqLen;
                positionOffset = 0;
            }

            _attention.Record(cmdBuf, _state.Q, kSrc, vSrc, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv,
                numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
                positionOffset: positionOffset, slidingWindow: _slidingWindow);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Output projection → NormOutput (reuse slot).
            RecordMatmul(cmdBuf, lw.O, lw.ODeviceQuantType, _state.AttnOutput, _state.NormOutput,
                lw.OOutputDim, lw.OInputDim, seqLen);

            if (cpuLw.OBias is { } ob)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                AddBiasRows(_state.NormOutput, ob, lw.OOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #1: AddScratch = Residual + NormOutput; then AddScratch → HiddenState.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordCopyBuffer(cmdBuf, _state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Residual snapshot (pre-FFN): HiddenState → Residual.
            RecordCopyBuffer(cmdBuf, _state.HiddenState, _state.Residual, (long)seqLen * hiddenSize * sizeof(float));
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // FFN RMSNorm
            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.FfnNormWeight, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Gate/Up projections
            RecordMatmul(cmdBuf, lw.Gate, lw.GateDeviceQuantType, _state.NormOutput, _state.FfnGate,
                lw.GateOutputDim, lw.GateInputDim, seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordMatmul(cmdBuf, lw.Up, lw.UpDeviceQuantType, _state.NormOutput, _state.FfnUp,
                lw.UpOutputDim, lw.UpInputDim, seqLen);

            if (cpuLw.GateBias is not null || cpuLw.UpBias is not null)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                if (cpuLw.GateBias is { } gb) AddBiasRows(_state.FfnGate, gb, lw.GateOutputDim, seqLen);
                if (cpuLw.UpBias is { } ub) AddBiasRows(_state.FfnUp, ub, lw.UpOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // SwiGLU
            _swiglu.Record(cmdBuf, _state.FfnGate, _state.FfnUp, _state.SiluOutput, seqLen * intermediateSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Down projection
            RecordMatmul(cmdBuf, lw.Down, lw.DownDeviceQuantType, _state.SiluOutput, _state.NormOutput,
                lw.DownOutputDim, lw.DownInputDim, seqLen);

            if (cpuLw.DownBias is { } db)
            {
                KernelSupport.ComputeToHostBarrier(cmdBuf);
                _submit.SubmitAndWait();
                AddBiasRows(_state.NormOutput, db, lw.DownOutputDim, seqLen);
                _submit.Begin();
                cmdBuf = _submit.CommandBuffer;
                KernelSupport.HostToComputeBarrier(cmdBuf);
            }
            else
            {
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            // Residual add #2: AddScratch = Residual + NormOutput; then AddScratch → HiddenState.
            _add.Record(cmdBuf, _state.Residual, _state.NormOutput, _state.AddScratch, seqLen * hiddenSize);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            RecordCopyBuffer(cmdBuf, _state.AddScratch, _state.HiddenState, (long)seqLen * hiddenSize * sizeof(float));

            // COMPUTE→COMPUTE between layers — next iteration's first op is the HiddenState→Residual copy.
            if (layer < Config.NumLayers - 1)
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
        }

        // 3. Final RMSNorm on the last token only, then LM head.
        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        KernelSupport.ComputeToComputeBarrier(cmdBuf);
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.NormOutput,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.NormOutput, _weights.OutputNormWeight, _state.NormOutput,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        RecordMatmul(cmdBuf, _weights.OutputWeight, _weights.OutputDeviceQuantType,
            _state.NormOutput, _state.Logits,
            _weights.OutputOutputDim, _weights.OutputInputDim, seqLen: 1);

        // 4. COMPUTE→HOST barrier for the vocab-row download that follows, submit, wait.
        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // 5. Return logits as a host-resident UnmanagedTensor [1, vocabSize].
        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        unsafe
        {
            var dest = new Span<float>((void*)result.DataPointer, vocabSize);
            _device.Download(_state.Logits, dest);
        }
        return result;
    }

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _matmulQ8.InvalidateDescriptorCache();
        _matmulQ8Gemm.InvalidateDescriptorCache();
        _matmulQ8GemmCoopmat?.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _rope.InvalidateDescriptorCache();
        _attention.InvalidateDescriptorCache();
        _swiglu.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Dispatches a matmul for a single linear projection: chooses
    /// <see cref="MatMulQ8_0Kernel"/> (decode-path GEMV) when the device-side
    /// weight is Q8_0 and <paramref name="seqLen"/>==1, the batched
    /// <see cref="MatMulQ8_0GemmKernel"/> when Q8_0 and <paramref name="seqLen"/>&gt;1,
    /// and <see cref="MatMulF32Kernel"/> for every non-Q8_0 weight.
    /// </summary>
    /// <remarks>
    /// All Q8_0 kernels require <paramref name="inputDim"/> to be a multiple
    /// of 32 (the Q8_0 group size). Llama-family projections satisfy this by
    /// construction; the Q8_0 kernels still validate at dispatch so a
    /// surprise non-aligned model fails loud.
    /// </remarks>
    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (weightQt == QuantType.Q8_0)
        {
            if (seqLen == 1)
            {
                _matmulQ8.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else if (_matmulQ8GemmCoopmat is not null)
            {
                // Prefill path on coopmat-capable devices — ~3.8× over scalar
                // at Llama-3 prefill shapes. See MatMulQ8_0GemmCoopmatKernel.
                _matmulQ8GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ8Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else
        {
            _matmul.Record(cmdBuf, weights, input, output,
                outputDim, inputDim, seqLen);
        }
    }

    /// <summary>
    /// Records a device-to-device <c>vkCmdCopyBuffer</c> of
    /// <paramref name="byteCount"/> bytes from the start of <paramref name="src"/>
    /// to the start of <paramref name="dst"/>. Replaces the scaffold's
    /// host-mapped memcpy which required a submit boundary on every call.
    /// </summary>
    private static void RecordCopyBuffer(nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long byteCount)
        => RecordCopyBufferRange(cmdBuf, src, dst, srcOffset: 0, dstOffset: 0, size: (ulong)byteCount);

    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    /// <summary>
    /// Adds a per-feature bias vector to every row of a
    /// <c>[seqLen, outputDim]</c> FP32 output buffer. Implemented in-place on
    /// the host via mapped memory — biases are tiny (hidden_size scale), and
    /// adding a dedicated "bias_add" compute kernel is out of scope for the
    /// correctness wave.
    /// </summary>
    private unsafe void AddBiasRows(VulkanDevice.Buffer output, float[] bias, int outputDim, int seqLen)
    {
        long biasBytes = (long)outputDim * sizeof(float);
        long outBytes = biasBytes * seqLen;

        VulkanApi.vkMapMemory(_device.Handle, output.Memory, 0, (ulong)outBytes, 0, out nint outMapped)
            .ThrowOnError("vkMapMemory AddBiasRows.output");
        try
        {
            float* o = (float*)outMapped;
            fixed (float* b = bias)
            {
                for (int t = 0; t < seqLen; t++)
                {
                    for (int i = 0; i < outputDim; i++)
                        o[t * outputDim + i] += b[i];
                }
            }
        }
        finally
        {
            VulkanApi.vkUnmapMemory(_device.Handle, output.Memory);
        }
    }

    /// <summary>
    /// Validates every token id is in range <c>[0, vocabSize)</c>. Separated
    /// from <see cref="RecordEmbeddingGather"/> so the check happens before
    /// we begin recording the command buffer — a bad id throws cleanly
    /// without leaving the submit context half-written.
    /// </summary>
    private void ValidateTokenIds(ReadOnlySpan<int> tokenIds)
    {
        int vocab = Config.VocabSize;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            if ((uint)id >= (uint)vocab)
                throw new ArgumentOutOfRangeException(nameof(tokenIds), $"Token id {id} is out of range");
        }
    }

    /// <summary>
    /// Records N device-local <c>vkCmdCopyBuffer</c> calls (one per input
    /// token) that gather per-token rows from the already-resident
    /// <see cref="VulkanWeights.TokenEmbedding"/> buffer into
    /// <see cref="VulkanForwardState.HiddenState"/>. The embedding table was
    /// dequantised to F32 and uploaded to device-local VRAM at construction
    /// time (see <see cref="VulkanWeights.Upload"/>), so the only
    /// per-forward cost here is <c>seqLen</c> cheap on-device copy commands
    /// — no host-mapped write, no host→device transfer bandwidth.
    /// </summary>
    /// <remarks>
    /// Vulkan's <c>vkCmdCopyBuffer</c> does accept a regions array, but the
    /// current P/Invoke surface takes a single region (matching the
    /// KV-cache-update path in <see cref="VulkanKvCache.RecordUpdate"/>).
    /// For <c>seqLen=1</c> decode this is one call; for prefill it's
    /// <c>promptLen</c> calls, still dwarfed by the per-layer matmul cost.
    /// </remarks>
    private void RecordEmbeddingGather(nint cmdBuf, ReadOnlySpan<int> tokenIds)
    {
        int hiddenSize = Config.HiddenSize;
        long rowBytes = (long)hiddenSize * sizeof(float);
        var srcBuf = _weights.TokenEmbedding.Handle;
        var dstBuf = _state.HiddenState.Handle;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int id = tokenIds[t];
            var region = new VkBufferCopy
            {
                srcOffset = (ulong)((long)id * rowBytes),
                dstOffset = (ulong)((long)t * rowBytes),
                size = (ulong)rowBytes,
            };
            VulkanApi.vkCmdCopyBuffer(cmdBuf, srcBuf, dstBuf, 1, region);
        }
    }

    private unsafe void UploadPositions(ReadOnlySpan<int> positions)
    {
        // The Allocate in EnsureCapacity already sized PositionsBuffer for seqLen;
        // delegate the mapped copy to device.Upload via a raw byte span.
        var posBytes = MemoryMarshal.AsBytes(positions);
        _device.Upload(posBytes, _state.PositionsBuffer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();

        _add.Dispose();
        _swiglu.Dispose();
        _attention.Dispose();
        _rope.Dispose();
        _rmsnorm.Dispose();
        _matmulQ8GemmCoopmat?.Dispose();
        _matmulQ8Gemm.Dispose();
        _matmulQ8.Dispose();
        _matmul.Dispose();

        _cpuWeights.Dispose();
        if (_ownsDevice)
            _device.Dispose();
    }
}
