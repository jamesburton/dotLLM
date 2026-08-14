// src/DotLLM.Cuda/Architectures/CudaNemotronHTransformerModel.cs
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// CUDA implementation of the NemotronH (<c>nemotron_h</c>) hybrid Mamba2-SSM + Transformer
/// model — e.g. NVIDIA's Nemotron-3-Nano-4B. F32 activations throughout, mirrors
/// <c>DotLLM.Models.Architectures.NemotronHTransformerModel</c> (CPU) and
/// <see cref="DotLLM.Vulkan.VulkanNemotronHTransformerModel"/> on the GPU.
/// </summary>
/// <remarks>
/// Each of the <c>config.NumLayers</c> layers is exactly one of Mamba2 SSM, GQA attention, or
/// squared-ReLU FFN (<see cref="ModelConfig.HybridLayout"/>'s <c>LayerKind</c> per layer), with a
/// single pre-sublayer RMSNorm and one residual add shared by all three kinds — see
/// <c>NemotronHTransformerModel.Forward</c> (CPU) for the authoritative per-layer sequence this
/// class's <c>Forward</c> (Task 11) mirrors.
/// </remarks>
public sealed unsafe class CudaNemotronHTransformerModel : IModel
{
    private readonly CudaNemotronHForwardState _state;
    private readonly CudaNemotronHSsmStateCache _ssmCache;
    private readonly CudaStream _stream;
    private readonly CudaCublasHandle _cublas;
    private readonly CudaContext _context;
    private readonly CudaKernels _kernels;
    private readonly int _deviceId;

    private readonly DeviceLayer[] _layers;

    // NOT device memory — the embedding LOOKUP is done as a per-call host-side row dequant in
    // Embed() (Task 8), reading directly from this retained HOST pointer (mmap'd GGUF data, or
    // a synthetic fixture's unmanaged host buffer), then H2D-copying only the seqLen rows
    // actually needed per call. This mirrors CudaQwen3HybridDenseTransformerModel's identical
    // "_embedDataBase" pattern and its documented rationale (only ever need seqLen rows, never
    // the whole vocab table) — see Task 8 for the full design note. Uniformly covers every GGUF
    // quant type (F32/F16/Q8_0/Q4_K/Q5_K/Q6_K/...) with one code path instead of a device-kernel
    // fast path for 3 formats plus a host fallback for the rest.
    private readonly nint _tokenEmbedHostPtr;
    private readonly QuantizationType _tokenEmbedQt;
    private readonly long _tokenEmbedRowBytes;
    private readonly nint _outputNormDevice;   // F32 [hiddenSize]
    private readonly nint _outputDevice;       // lm_head raw quant bytes (may alias _tokenEmbedDevice)
    private readonly QuantizationType _outputQt;
    private readonly int _outputOutputDim;     // vocab size
    private readonly int _outputInputDim;      // hidden size
    private readonly bool _ownsOutputDevice;

    private readonly HybridLayerLayout _layout;
    private readonly MambaSsmConfig _ssm;
    private readonly int[] _kvSlotForLayer;
    private readonly int _attentionLayerCount;
    private readonly int[] _ssmLayerOrdinal;
    private readonly int _numSsmLayers;


    // Prefill dequant-to-F16 + cuBLAS-HGEMM scratch, shared by every projection whose weight
    // has no native F32 CUDA kernel — see Task 8's Gemm dispatcher.
    private nint _dequantScratchF16Weight;
    private nint _activF16InScratch;
    private long _activF16InScratchElems;
    private nint _activF16OutScratch;
    private long _activF16OutScratchElems;

    // Caller-supplied per-sequence SSM state for the in-flight Forward — see Task 11.
    private CudaNemotronHSsmStateCache? _activeSsm;

    /// <summary>The CPU model <see cref="LoadFromGguf"/> reused to resolve GGUF tensor names;
    /// disposed with this model so its dequantised F32 norm arrays are released. The
    /// <see cref="GgufFile"/> itself stays caller-owned (mirrors
    /// <see cref="DotLLM.Vulkan.VulkanNemotronHTransformerModel"/>'s identical field). Null on
    /// the <see cref="BuildFromPrebuiltWeights"/> (synthetic-fixture) path.</summary>
    private NemotronHTransformerModel? _cpuModel;

    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _ssmCache.AllocatedBytes;

    /// <summary>Number of attention layers — the matching sparse KV-cache slot count.</summary>
    public int AttentionLayerCount => _attentionLayerCount;

    /// <summary>Creates a <see cref="CudaNemotronHKvCache"/> sized for this model.</summary>
    public CudaNemotronHKvCache CreateKvCache(int maxSeqLen)
    {
        _context.MakeCurrent();
        return new(_attentionLayerCount, Config.NumKvHeads, Config.HeadDim, maxSeqLen, _deviceId);
    }

    // ── Device-side per-layer weight structs ────────────────────────────────

    private readonly struct DeviceSsm
    {
        public required nint InWeight { get; init; }
        public required QuantizationType InQt { get; init; }
        public required int InInputDim { get; init; }
        public required int InOutputDim { get; init; }
        public required nint Conv1dWeightDevice { get; init; }  // F32 [dConv, convDim]
        public required nint Conv1dBiasDevice { get; init; }    // F32 [convDim]
        public required nint ADevice { get; init; }             // F32 [nHead]
        public required nint DDevice { get; init; }             // F32 [nHead]
        public required nint DtBiasDevice { get; init; }        // F32 [nHead]
        public required nint NormWeightDevice { get; init; }    // F32 [dInner]
        public required nint OutWeight { get; init; }
        public required QuantizationType OutQt { get; init; }
        public required int OutInputDim { get; init; }
        public required int OutOutputDim { get; init; }
    }

    private readonly struct DeviceAttn
    {
        public required nint QWeight { get; init; }
        public required QuantizationType QQt { get; init; }
        public required int QInputDim { get; init; }
        public required int QOutputDim { get; init; }
        public required nint KWeight { get; init; }
        public required QuantizationType KQt { get; init; }
        public required int KInputDim { get; init; }
        public required int KOutputDim { get; init; }
        public required nint VWeight { get; init; }
        public required QuantizationType VQt { get; init; }
        public required int VInputDim { get; init; }
        public required int VOutputDim { get; init; }
        public required nint OWeight { get; init; }
        public required QuantizationType OQt { get; init; }
        public required int OInputDim { get; init; }
        public required int OOutputDim { get; init; }
        public required int NumKvHeads { get; init; }
    }

    private readonly struct DeviceFfn
    {
        public required nint UpWeight { get; init; }
        public required QuantizationType UpQt { get; init; }
        public required int UpInputDim { get; init; }
        public required int UpOutputDim { get; init; }
        public required nint DownWeight { get; init; }
        public required QuantizationType DownQt { get; init; }
        public required int DownInputDim { get; init; }
        public required int DownOutputDim { get; init; }
    }

    private readonly struct DeviceLayer
    {
        public required nint AttnNormWeightDevice { get; init; } // F32 [hiddenSize]
        public required HybridLayerKind Kind { get; init; }
        public DeviceSsm? Ssm { get; init; }
        public DeviceAttn? Attention { get; init; }
        public DeviceFfn? Ffn { get; init; }
    }

    private CudaNemotronHTransformerModel(
        ModelConfig config,
        DeviceLayer[] layers,
        nint tokenEmbedHostPtr, QuantizationType tokenEmbedQt, long tokenEmbedRowBytes,
        nint outputNormDevice,
        nint outputDevice, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        bool ownsOutputDevice,
        int[] kvSlotForLayer, int attentionLayerCount,
        CudaNemotronHForwardState state, CudaNemotronHSsmStateCache ssmCache,
        CudaStream stream, CudaCublasHandle cublas, CudaContext context, CudaKernels kernels,
        int deviceId, nint dequantScratchDevice)
    {
        Config = config;
        _layers = layers;
        _tokenEmbedHostPtr = tokenEmbedHostPtr;
        _tokenEmbedQt = tokenEmbedQt;
        _tokenEmbedRowBytes = tokenEmbedRowBytes;
        _outputNormDevice = outputNormDevice;
        _outputDevice = outputDevice;
        _outputQt = outputQt;
        _outputOutputDim = outputOutputDim;
        _outputInputDim = outputInputDim;
        _ownsOutputDevice = ownsOutputDevice;
        _layout = config.HybridLayout!;
        _ssm = config.SsmConfig!.Value;
        _kvSlotForLayer = kvSlotForLayer;
        _attentionLayerCount = attentionLayerCount;
        _state = state;
        _ssmCache = ssmCache;
        _stream = stream;
        _cublas = cublas;
        _context = context;
        _kernels = kernels;
        _deviceId = deviceId;
        _dequantScratchF16Weight = dequantScratchDevice;

        _ssmLayerOrdinal = new int[config.NumLayers];
        int ssmOrdinal = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            _ssmLayerOrdinal[i] = _layout.LayerKind[i] == HybridLayerKind.Ssm
                ? ssmOrdinal++
                : -1;
        }
        _numSsmLayers = ssmOrdinal;
    }

    /// <summary>
    /// Loads a NemotronH model from an opened GGUF file onto the given CUDA device. Reuses
    /// <c>NemotronHTransformerModel.LoadFromGguf</c> (CPU) for all GGUF tensor-name
    /// resolution and shape validation, then uploads the resulting weights to device memory —
    /// see this task's "Design decision" note for why.
    /// </summary>
    public static CudaNemotronHTransformerModel LoadFromGguf(
        GgufFile gguf, ModelConfig config, int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"CudaNemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));

        var cpuModel = NemotronHTransformerModel.LoadFromGguf(gguf, config);
        try
        {
            var cpuLayers = ExtractCpuLayers(cpuModel);
            var outputNormWeight = ExtractOutputNormWeight(cpuModel);
            var (tokenEmbedPtr, tokenEmbedQt) = ExtractTokenEmbed(cpuModel);
            var (outputPtr, outputQt, outputM, outputK) = ExtractOutput(cpuModel);

            var model = BuildFromPrebuiltWeights(
                config, cpuLayers, outputNormWeight,
                outputPtr, outputQt, outputM, outputK,
                tokenEmbedPtr, tokenEmbedQt,
                deviceId, ptxDir);
            model._cpuModel = cpuModel;
            return model;
        }
        catch
        {
            cpuModel.Dispose();
            throw;
        }
    }

    // ── CPU-model field extraction via reflection (mirrors VulkanNemotronHTransformerModel's
    // identical Field()/Extract* helpers — the fields are `private`, not `internal`, so
    // InternalsVisibleTo alone doesn't expose them). ─────────────────────────────────────────

    private static NemotronHLayerWeights[] ExtractCpuLayers(NemotronHTransformerModel m)
        => (NemotronHLayerWeights[])Field("_layers").GetValue(m)!;

    private static float[] ExtractOutputNormWeight(NemotronHTransformerModel m)
        => (float[])Field("_outputNormWeight").GetValue(m)!;

    private static (nint ptr, QuantizationType qt) ExtractTokenEmbed(NemotronHTransformerModel m)
        => ((nint)Field("_tokenEmbedWeight").GetValue(m)!,
            (QuantizationType)Field("_tokenEmbedQuantType").GetValue(m)!);

    private static (nint ptr, QuantizationType qt, int outputDim, int inputDim) ExtractOutput(
        NemotronHTransformerModel m)
        => ((nint)Field("_outputWeight").GetValue(m)!,
            (QuantizationType)Field("_outputQuantType").GetValue(m)!,
            (int)Field("_outputOutputDim").GetValue(m)!,
            (int)Field("_outputInputDim").GetValue(m)!);

    private static System.Reflection.FieldInfo Field(string name)
        => typeof(NemotronHTransformerModel).GetField(
               name,
               System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)
           ?? throw new InvalidOperationException($"NemotronHTransformerModel.{name} field missing.");

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _context.MakeCurrent();
        _state.Dispose();
        _ssmCache.Dispose();
        FreeIfNonZero(ref _dequantScratchF16Weight);
        FreeIfNonZero(ref _activF16InScratch);
        FreeIfNonZero(ref _activF16OutScratch);

        for (int i = 0; i < _layers.Length; i++)
            FreeLayer(_layers[i]);

        // _tokenEmbedHostPtr is a borrowed host pointer (mmap'd GGUF data or a caller-owned
        // fixture buffer) — never device memory, never freed here. See its field doc.
        nint outputNorm = _outputNormDevice;
        FreeIfNonZero(ref outputNorm);
        if (_ownsOutputDevice)
        {
            nint outputDevice = _outputDevice;
            FreeIfNonZero(ref outputDevice);
        }

        _kernels.Dispose();
        _cublas.Dispose();
        _stream.Dispose();
        _context.Dispose();
        _cpuModel?.Dispose();

        _disposed = true;
        GC.SuppressFinalize(this);
    }

    private static void FreeLayer(in DeviceLayer layer)
    {
        nint p = layer.AttnNormWeightDevice; FreeIfNonZero(ref p);
        if (layer.Ssm is { } s)
        {
            nint a = s.InWeight; FreeIfNonZero(ref a);
            nint b = s.Conv1dWeightDevice; FreeIfNonZero(ref b);
            nint c = s.Conv1dBiasDevice; FreeIfNonZero(ref c);
            nint d = s.ADevice; FreeIfNonZero(ref d);
            nint e = s.DDevice; FreeIfNonZero(ref e);
            nint f = s.DtBiasDevice; FreeIfNonZero(ref f);
            nint g = s.NormWeightDevice; FreeIfNonZero(ref g);
            nint h = s.OutWeight; FreeIfNonZero(ref h);
        }
        if (layer.Attention is { } at)
        {
            nint q = at.QWeight; FreeIfNonZero(ref q);
            nint k = at.KWeight; FreeIfNonZero(ref k);
            nint v = at.VWeight; FreeIfNonZero(ref v);
            nint o = at.OWeight; FreeIfNonZero(ref o);
        }
        if (layer.Ffn is { } ff)
        {
            nint up = ff.UpWeight; FreeIfNonZero(ref up);
            nint down = ff.DownWeight; FreeIfNonZero(ref down);
        }
    }

    // ── Shared device-upload primitives (reused by BuildFromPrebuiltWeights, Task 8) ────────

    private static nint AllocDevice(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
        return ptr;
    }

    private static void CopyHtoD(nint dst, nint src, long bytes)
    {
        CudaDriverApi.cuMemcpyHtoD_v2(dst, src, (nuint)bytes).ThrowOnError();
    }

    private static void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { CudaDriverApi.cuMemFree_v2(ptr); ptr = 0; }
    }

    /// <summary>Uploads a projection's raw quantized (or F32/F16) bytes from a host pointer
    /// (mmap'd GGUF data, or a synthetic fixture's unmanaged buffer) to a fresh device buffer.
    /// No PQ2_0 repack (unlike <c>CudaQwen3HybridDenseTransformerModel.UploadRawTensor</c>) —
    /// NemotronH GGUFs never carry PQ2_0 tensors.</summary>
    private static nint UploadRawTensorFromHost(nint hostPtr, QuantizationType qt, int outputDim, int inputDim)
    {
        long bytes = Dequantize.RowByteSize(inputDim, qt) * outputDim;
        nint device = AllocDevice(bytes);
        CopyHtoD(device, hostPtr, bytes);
        return device;
    }

    /// <summary>Uploads an already-dequantised managed float array (e.g. <c>NormWeight</c>,
    /// <c>Conv1dWeight</c>, <c>AttnNormWeight</c> — every small per-layer F32 array
    /// <see cref="NemotronHLayerWeights"/>'s CPU loader already materialised) to device memory.</summary>
    private static nint UploadF32ArrayFrom(float[] data)
    {
        long bytes = (long)data.Length * sizeof(float);
        nint device = AllocDevice(bytes);
        fixed (float* p = data)
        {
            CopyHtoD(device, (nint)p, bytes);
        }
        return device;
    }

    /// <summary>
    /// Builds a NemotronH CUDA model from caller-owned, pre-built <see cref="NemotronHLayerWeights"/> —
    /// the shared entry point for <see cref="LoadFromGguf"/> (Task 7) and the synthetic-fixture
    /// parity test (Task 13). Caller retains ownership of every host <see cref="nint"/> pointer
    /// (token embed, output, plus every projection inside <paramref name="cpuLayers"/>) — this
    /// method only reads from them to build fresh device buffers plus the retained
    /// <c>_tokenEmbedHostPtr</c> for per-call embedding dequant.
    /// </summary>
    internal static CudaNemotronHTransformerModel BuildFromPrebuiltWeights(
        ModelConfig config,
        NemotronHLayerWeights[] cpuLayers,
        float[] outputNormWeight,
        nint outputWeight, QuantizationType outputQt, int outputOutputDim, int outputInputDim,
        nint tokenEmbedWeight, QuantizationType tokenEmbedQt,
        int deviceId = 0, string? ptxDir = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(cpuLayers);
        ArgumentNullException.ThrowIfNull(outputNormWeight);
        if (config.Architecture != Architecture.NemotronH)
            throw new ArgumentException(
                $"CudaNemotronHTransformerModel requires Architecture.NemotronH, got {config.Architecture}.",
                nameof(config));
        if (config.HybridLayout is null)
            throw new ArgumentException("NemotronH config must have HybridLayout populated.", nameof(config));
        if (config.SsmConfig is null)
            throw new ArgumentException("NemotronH config must have SsmConfig populated.", nameof(config));
        if (cpuLayers.Length != config.NumLayers)
            throw new ArgumentException(
                $"cpuLayers length {cpuLayers.Length} != config.NumLayers {config.NumLayers}.", nameof(cpuLayers));

        var layout = config.HybridLayout!;
        var ssm = config.SsmConfig!.Value;
        int hiddenSize = config.HiddenSize;

        var context = CudaContext.Create(deviceId);
        var stream = CudaStream.Create();
        var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);

        ptxDir ??= Path.Combine(AppContext.BaseDirectory, "ptx");
        var kernels = new CudaKernels(ptxDir);

        long maxTileFloats = 0;

        // Output norm — always F32 [hiddenSize].
        nint outputNormDevice = UploadF32ArrayFrom(outputNormWeight);

        // lm_head — always uploaded as its own fresh device buffer (no tied-embedding aliasing
        // optimization; NemotronH's CPU loader already resolves the tied case at the host-pointer
        // level, so outputWeight here is already the correct source regardless).
        nint outputDevice = UploadRawTensorFromHost(outputWeight, outputQt, outputOutputDim, outputInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)outputOutputDim * outputInputDim);

        long tokenEmbedRowBytes = Dequantize.RowByteSize(hiddenSize, tokenEmbedQt);

        // Per-layer upload.
        var layers = new DeviceLayer[config.NumLayers];
        var kvSlotForLayer = new int[config.NumLayers];
        int attentionLayerCount = 0;
        int numSsmLayers = 0;
        int maxIntermediate = 0;
        for (int i = 0; i < config.NumLayers; i++)
        {
            var cpuLayer = cpuLayers[i];
            nint attnNormDevice = UploadF32ArrayFrom(cpuLayer.AttnNormWeight);

            DeviceSsm? ssmDev = null;
            DeviceAttn? attnDev = null;
            DeviceFfn? ffnDev = null;

            switch (layout.LayerKind[i])
            {
                case HybridLayerKind.Ssm:
                    ssmDev = UploadDeviceSsmLayer(cpuLayer.Ssm!, ref maxTileFloats);
                    numSsmLayers++;
                    break;
                case HybridLayerKind.Attention:
                    attnDev = UploadDeviceAttentionLayer(cpuLayer.Attention!, ref maxTileFloats);
                    kvSlotForLayer[i] = attentionLayerCount++;
                    break;
                case HybridLayerKind.Ffn:
                    ffnDev = UploadDeviceFfnLayer(cpuLayer.Ffn!, ref maxTileFloats);
                    if (cpuLayer.Ffn!.UpOutputDim > maxIntermediate) maxIntermediate = cpuLayer.Ffn.UpOutputDim;
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {layout.LayerKind[i]} at layer {i}.");
            }
            if (layout.LayerKind[i] != HybridLayerKind.Attention) kvSlotForLayer[i] = -1;

            layers[i] = new DeviceLayer
            {
                AttnNormWeightDevice = attnNormDevice,
                Kind = layout.LayerKind[i],
                Ssm = ssmDev,
                Attention = attnDev,
                Ffn = ffnDev,
            };
        }
        if (maxIntermediate == 0) maxIntermediate = hiddenSize;

        // Any RoPEConfig is ignored — nemotron_h applies no position encoding on
        // attention (issue #372); mirrors the CPU model.

        var state = new CudaNemotronHForwardState(
            hiddenSize: hiddenSize,
            maxIntermediateSize: maxIntermediate,
            vocabSize: config.VocabSize,
            qElems: config.NumAttentionHeads * config.HeadDim,
            kvElems: config.NumKvHeads * config.HeadDim,
            inputProjectionDim: ssm.InputProjectionDim,
            convDim: ssm.ConvDim,
            dConv: ssm.DConv,
            dInner: ssm.DInner,
            nHead: ssm.NHead,
            nGroup: ssm.NGroup,
            dState: ssm.DState,
            maxSeqLen: config.MaxSequenceLength);

        var ssmCache = new CudaNemotronHSsmStateCache(ssm, numSsmLayers);

        UpdateMaxTile(ref maxTileFloats, maxIntermediate); // dequant scratch floor for tiny models
        nint dequantScratchDevice = AllocDevice(maxTileFloats * sizeof(ushort));

        return new CudaNemotronHTransformerModel(
            config, layers,
            tokenEmbedWeight, tokenEmbedQt, tokenEmbedRowBytes,
            outputNormDevice,
            outputDevice, outputQt, outputOutputDim, outputInputDim, ownsOutputDevice: true,
            kvSlotForLayer, attentionLayerCount,
            state, ssmCache, stream, cublas, context, kernels, deviceId, dequantScratchDevice);
    }

    private static void UpdateMaxTile(ref long max, long candidate)
    {
        if (candidate > max) max = candidate;
    }

    private static DeviceSsm UploadDeviceSsmLayer(NemotronHSsmWeights w, ref long maxTileFloats)
    {
        nint inDevice = UploadRawTensorFromHost(w.InWeight, w.InQuantType, w.InOutputDim, w.InInputDim);
        nint outDevice = UploadRawTensorFromHost(w.OutWeight, w.OutQuantType, w.OutOutputDim, w.OutInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.InOutputDim * w.InInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.OutOutputDim * w.OutInputDim);

        return new DeviceSsm
        {
            InWeight = inDevice, InQt = w.InQuantType, InInputDim = w.InInputDim, InOutputDim = w.InOutputDim,
            Conv1dWeightDevice = UploadF32ArrayFrom(w.Conv1dWeight),
            Conv1dBiasDevice = UploadF32ArrayFrom(w.Conv1dBias),
            ADevice = UploadF32ArrayFrom(w.A),
            DDevice = UploadF32ArrayFrom(w.D),
            DtBiasDevice = UploadF32ArrayFrom(w.DtBias),
            NormWeightDevice = UploadF32ArrayFrom(w.NormWeight),
            OutWeight = outDevice, OutQt = w.OutQuantType, OutInputDim = w.OutInputDim, OutOutputDim = w.OutOutputDim,
        };
    }

    private static DeviceAttn UploadDeviceAttentionLayer(NemotronHAttentionWeights w, ref long maxTileFloats)
    {
        nint qDevice = UploadRawTensorFromHost(w.QWeight, w.QQuantType, w.QOutputDim, w.QInputDim);
        nint kDevice = UploadRawTensorFromHost(w.KWeight, w.KQuantType, w.KOutputDim, w.KInputDim);
        nint vDevice = UploadRawTensorFromHost(w.VWeight, w.VQuantType, w.VOutputDim, w.VInputDim);
        nint oDevice = UploadRawTensorFromHost(w.OWeight, w.OQuantType, w.OOutputDim, w.OInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.QOutputDim * w.QInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.KOutputDim * w.KInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.VOutputDim * w.VInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.OOutputDim * w.OInputDim);

        return new DeviceAttn
        {
            QWeight = qDevice, QQt = w.QQuantType, QInputDim = w.QInputDim, QOutputDim = w.QOutputDim,
            KWeight = kDevice, KQt = w.KQuantType, KInputDim = w.KInputDim, KOutputDim = w.KOutputDim,
            VWeight = vDevice, VQt = w.VQuantType, VInputDim = w.VInputDim, VOutputDim = w.VOutputDim,
            OWeight = oDevice, OQt = w.OQuantType, OInputDim = w.OInputDim, OOutputDim = w.OOutputDim,
            NumKvHeads = w.NumKvHeads,
        };
    }

    private static DeviceFfn UploadDeviceFfnLayer(NemotronHFfnWeights w, ref long maxTileFloats)
    {
        nint upDevice = UploadRawTensorFromHost(w.UpWeight, w.UpQuantType, w.UpOutputDim, w.UpInputDim);
        nint downDevice = UploadRawTensorFromHost(w.DownWeight, w.DownQuantType, w.DownOutputDim, w.DownInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.UpOutputDim * w.UpInputDim);
        UpdateMaxTile(ref maxTileFloats, (long)w.DownOutputDim * w.DownInputDim);

        return new DeviceFfn
        {
            UpWeight = upDevice, UpQt = w.UpQuantType, UpInputDim = w.UpInputDim, UpOutputDim = w.UpOutputDim,
            DownWeight = downDevice, DownQt = w.DownQuantType, DownInputDim = w.DownInputDim, DownOutputDim = w.DownOutputDim,
        };
    }

    /// <summary>
    /// Dispatches one linear projection <c>Y[seqLen,m] = X[seqLen,k] @ W[m,k]^T</c> by quant
    /// type. F32 weights go straight through cuBLAS. Q8_0 decode (seqLen==1) uses the direct
    /// F32-in/F32-out quantized GEMV kernel. Every other quant format (F16, K-quants, IQ-quants)
    /// funnels through an F16 dequant-then-cuBLAS-HGEMM round trip. Verbatim copy of
    /// <c>CudaQwen3HybridDenseTransformerModel.Gemm</c> minus its I2_S/PQ2_0 branches (NemotronH
    /// GGUFs never carry ternary/PQ2_0 tensors).
    /// </summary>
    private void Gemm(nint weight, QuantizationType qt, nint x, nint y, int m, int k, int seqLen)
    {
        nint streamH = _stream.Handle;

        if (qt == QuantizationType.F32)
        {
            CudaGemm.LinearF32(_cublas.Handle, x, weight, y, seqLen, k, m, streamH);
            return;
        }

        if (seqLen == 1)
        {
            if (qt == QuantizationType.Q8_0)
            {
                _kernels.LaunchQuantizedGemvF32In(weight, x, y, m, k, streamH);
                return;
            }

            if (qt == QuantizationType.F16
                || _kernels.HasMmq(qt)
                || _kernels.HasQuantizedGemvKernel(qt))
            {
                EnsureActivF16InScratch(k);
                EnsureActivF16OutScratch(m);
                _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, k, streamH);

                if (qt == QuantizationType.F16)
                {
                    CudaGemm.GemvF16(_cublas.Handle, weight, _activF16InScratch,
                        _activF16OutScratch, m, k, streamH);
                }
                else if (_kernels.HasMmq(qt) && !CudaKernels.ForceDirectGemv)
                {
                    _kernels.LaunchQuantizedGemvMmq(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, preqScratch: 0, streamH);
                }
                else
                {
                    _kernels.LaunchQuantizedGemv(weight, qt,
                        _activF16InScratch, _activF16OutScratch, m, k, streamH);
                }

                _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, m, streamH);
                return;
            }
        }

        // Prefill (seqLen > 1) and decode fallback for any quant format without a direct path.
        long totalElems = (long)m * k;
        int totalElemsI = checked((int)totalElems);
        int activInElems = checked((int)((long)seqLen * k));
        int activOutElems = checked((int)((long)seqLen * m));
        EnsureActivF16InScratch(activInElems);
        EnsureActivF16OutScratch(activOutElems);

        _kernels.LaunchDequantToF16(weight, qt, _dequantScratchF16Weight, totalElemsI, streamH);
        _kernels.LaunchConvertF32ToF16(x, _activF16InScratch, activInElems, streamH);
        CudaGemm.LinearF16(_cublas.Handle, _activF16InScratch, _dequantScratchF16Weight,
            _activF16OutScratch, seqLen, k, m, streamH);
        _kernels.LaunchConvertF16ToF32(_activF16OutScratch, y, activOutElems, streamH);
    }

    private void EnsureActivF16InScratch(long halfs)
    {
        if (halfs <= _activF16InScratchElems) return;
        FreeIfNonZero(ref _activF16InScratch);
        CudaDriverApi.cuMemAlloc_v2(out _activF16InScratch, (nuint)(halfs * sizeof(ushort))).ThrowOnError();
        _activF16InScratchElems = halfs;
    }

    private void EnsureActivF16OutScratch(long halfs)
    {
        if (halfs <= _activF16OutScratchElems) return;
        FreeIfNonZero(ref _activF16OutScratch);
        CudaDriverApi.cuMemAlloc_v2(out _activF16OutScratch, (nuint)(halfs * sizeof(ushort))).ThrowOnError();
        _activF16OutScratchElems = halfs;
    }

    /// <summary>
    /// Embedding lookup via per-call host-side row dequant (see this task's design note) —
    /// writes <paramref name="tokenIds"/>.Length rows of <paramref name="hiddenSize"/> floats
    /// each into the device buffer <paramref name="hiddenDevice"/>. Mirrors
    /// <c>NemotronHTransformerModel.EmbedTokens</c> (CPU) exactly: F32 straight copy, F16
    /// widened via <c>TensorPrimitives.ConvertToSingle</c>, everything else via
    /// <c>Dequantize.ToFloat32</c>.
    /// </summary>
    private void Embed(ReadOnlySpan<int> tokenIds, nint hiddenDevice, int hiddenSize)
    {
        int seqLen = tokenIds.Length;
        float[] hostRows = new float[seqLen * hiddenSize];

        for (int t = 0; t < seqLen; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)Config.VocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {Config.VocabSize}).");

            Span<float> dst = hostRows.AsSpan(t * hiddenSize, hiddenSize);
            nint rowPtr = _tokenEmbedHostPtr + (nint)((long)tokenId * _tokenEmbedRowBytes);

            if (_tokenEmbedQt == QuantizationType.F32)
            {
                new ReadOnlySpan<float>((float*)rowPtr, hiddenSize).CopyTo(dst);
            }
            else if (_tokenEmbedQt == QuantizationType.F16)
            {
                var src = new ReadOnlySpan<Half>((Half*)rowPtr, hiddenSize);
                System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(src, dst);
            }
            else
            {
                Dequantize.ToFloat32(rowPtr, hiddenSize, _tokenEmbedQt, dst);
            }
        }

        fixed (float* p = hostRows)
        {
            CopyHtoD(hiddenDevice, (nint)p, (long)hostRows.Length * sizeof(float));
        }
    }

    /// <summary>
    /// Mamba2 SSM sub-layer forward — reads pre-normed activations from
    /// <c>_state.NormOutput</c> and writes the ssm_out projection back into the same buffer.
    /// Advances the per-layer conv/SSM recurrent state in place. See this task's reference note
    /// for how this 11-operation CUDA sequence maps onto the CPU/Vulkan 12-step numbering (two
    /// steps are fused into the Task 1 scan kernel).
    /// </summary>
    private void ForwardSsmBody(in DeviceSsm ssmW, int absoluteLayerIndex, int seqLen, int hiddenSize, float eps)
    {
        int dInner = _ssm.DInner;
        int dConv = _ssm.DConv;
        int nHead = _ssm.NHead;
        int headDim = _ssm.HeadDim;
        int dState = _ssm.DState;
        int nGroup = _ssm.NGroup;
        int convDim = _ssm.ConvDim;
        int groupDim = dInner / nGroup;
        int inProjDim = _ssm.InputProjectionDim;
        int bcDim = nGroup * dState;
        int dtOffset = 2 * dInner + 2 * nGroup * dState;
        nint streamH = _stream.Handle;

        int ssmOrdinal = _ssmLayerOrdinal[absoluteLayerIndex];
        var activeSsm = _activeSsm ?? _ssmCache;
        nint convStatePtr = activeSsm.GetConvStatePtr(ssmOrdinal);
        nint ssmStatePtr = activeSsm.GetSsmStatePtr(ssmOrdinal);

        // 1. ssm_in GEMM: NormOutput[seqLen, hiddenSize] -> Zxbcdt[seqLen, inProjDim].
        Gemm(ssmW.InWeight, ssmW.InQt, _state.NormOutput, _state.Zxbcdt, inProjDim, hiddenSize, seqLen);

        // 2. ConvInput = concat(conv_state, xBC rows sliced out of Zxbcdt).
        long convDimBytes = (long)convDim * sizeof(float);
        long inProjRowBytes = (long)inProjDim * sizeof(float);
        if (dConv > 1)
        {
            long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.ConvInput, convStatePtr, (nuint)convStateBytes, streamH).ThrowOnError();
        }
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes + (long)dInner * sizeof(float));
            nint dst = _state.ConvInput + (nint)((long)(dConv - 1 + t) * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)convDimBytes, streamH).ThrowOnError();
        }

        // 3. Conv1d causal -> XBC. Reuses the existing generic conv1d_causal_f32 kernel as-is.
        _kernels.LaunchConv1dCausalF32(_state.ConvInput, ssmW.Conv1dWeightDevice, ssmW.Conv1dBiasDevice, _state.XBC,
            dConv, convDim, seqLen, streamH);

        // 4. SiLU on XBC in place.
        _kernels.LaunchSiluF32(_state.XBC, (long)seqLen * convDim, streamH);

        // 5. Save the trailing (dConv-1) rows of ConvInput (pre-SiLU) back into conv_state.
        if (dConv > 1)
        {
            long convStateBytes = (long)(dConv - 1) * convDim * sizeof(float);
            nint src = _state.ConvInput + (nint)((long)seqLen * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(convStatePtr, src, (nuint)convStateBytes, streamH).ThrowOnError();
        }

        // 6. Extract the RAW dt slice (bias-add + guarded softplus are fused into the scan
        // kernel launched in step 8 — see Task 1).
        long dtRowBytes = (long)nHead * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes + (long)dtOffset * sizeof(float));
            nint dst = _state.DtBuffer + (nint)((long)t * dtRowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)dtRowBytes, streamH).ThrowOnError();
        }

        // 7. Split XBC[t,:] = [x | B | C] into SsmX / SsmB / SsmC.
        long xRowBytes = (long)dInner * sizeof(float);
        long bcRowBytes = (long)bcDim * sizeof(float);
        for (int t = 0; t < seqLen; t++)
        {
            nint rowBase = _state.XBC + (nint)((long)t * convDimBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmX + (nint)((long)t * xRowBytes), rowBase,
                (nuint)xRowBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmB + (nint)((long)t * bcRowBytes), rowBase + (nint)xRowBytes,
                (nuint)bcRowBytes, streamH).ThrowOnError();
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.SsmC + (nint)((long)t * bcRowBytes),
                rowBase + (nint)(xRowBytes + bcRowBytes), (nuint)bcRowBytes, streamH).ThrowOnError();
        }

        // 8. Mamba2 selective scan — dt bias-add, guarded softplus, decay, and the D-skip term
        // are ALL fused into this one launch (see Task 1's kernel documentation).
        _kernels.LaunchMamba2SelectiveScanF32(
            ssmStatePtr, _state.SsmX, _state.DtBuffer, ssmW.DtBiasDevice, ssmW.ADevice, ssmW.DDevice,
            _state.SsmB, _state.SsmC, _state.SsmY,
            nHead, headDim, dState, nGroup, seqLen, streamH);

        // 9. Extract z = Zxbcdt[t, 0..dInner) into SsmZ (strided source row, contiguous dest).
        for (int t = 0; t < seqLen; t++)
        {
            nint src = _state.Zxbcdt + (nint)((long)t * inProjRowBytes);
            nint dst = _state.SsmZ + (nint)((long)t * xRowBytes);
            CudaDriverApi.cuMemcpyDtoDAsync_v2(dst, src, (nuint)xRowBytes, streamH).ThrowOnError();
        }

        // 10. SwiGLU gating in place: SsmY = SiLU(SsmZ) * SsmY. Safe to alias up==output — see
        // this task's reference note.
        _kernels.LaunchSwiGLUF32(_state.SsmZ, _state.SsmY, _state.SsmY, dInner, seqLen, streamH);

        // 11. Group RMSNorm on SsmY in place.
        _kernels.LaunchGroupRmsNormF32(_state.SsmY, ssmW.NormWeightDevice, eps, seqLen, nGroup, groupDim, streamH);

        // 12. ssm_out projection into NormOutput.
        Gemm(ssmW.OutWeight, ssmW.OutQt, _state.SsmY, _state.NormOutput, hiddenSize, dInner, seqLen);
    }

    /// <summary>
    /// GQA attention sub-layer forward — reads pre-normed activations from
    /// <c>_state.NormOutput</c> and writes the o_proj result back into the same buffer.
    /// <paramref name="positions"/> is the HOST-side position span (needed by
    /// <see cref="IKvCache.Update(DotLLM.Core.Tensors.TensorRef, DotLLM.Core.Tensors.TensorRef, ReadOnlySpan{int}, int)"/>'s
    /// signature); <c>_state.PositionsDevice</c> (uploaded once
    /// per <c>Forward</c> call by Task 11) is the device-side copy RoPE reads from.
    /// </summary>
    private void ForwardAttentionBody(
        in DeviceAttn attn, int absoluteLayerIndex, int seqLen, ReadOnlySpan<int> positions,
        int numHeads, int numKvHeads, int headDim, IKvCache? kvCache)
    {
        int kvStride = numKvHeads * headDim;
        nint streamH = _stream.Handle;

        Gemm(attn.QWeight, attn.QQt, _state.NormOutput, _state.QScratch, attn.QOutputDim, attn.QInputDim, seqLen);
        Gemm(attn.KWeight, attn.KQt, _state.NormOutput, _state.KScratch, attn.KOutputDim, attn.KInputDim, seqLen);
        Gemm(attn.VWeight, attn.VQt, _state.NormOutput, _state.VScratch, attn.VOutputDim, attn.VInputDim, seqLen);

        // NO position encoding here (issue #372): llama.cpp's nemotron-h.cpp and HF's
        // NemotronHAttention rotate nothing — position information comes entirely
        // from the Mamba2 layers. Mirrors the CPU model's ForwardAttentionBody.

        if (kvCache is not null)
        {
            int kvSlot = _kvSlotForLayer[absoluteLayerIndex];
            if (kvSlot < 0)
                throw new InvalidOperationException(
                    $"Layer {absoluteLayerIndex} has no KV-cache slot (not an attention layer).");

            var kRef = new TensorRef(seqLen, kvStride, DType.Float32, _deviceId, _state.KScratch);
            var vRef = new TensorRef(seqLen, kvStride, DType.Float32, _deviceId, _state.VScratch);
            if (kvCache is CudaNemotronHKvCache cudaKvCache)
                cudaKvCache.UpdateDevice(kRef, vRef, positions, kvSlot, streamH);
            else
                kvCache.Update(kRef, vRef, positions, kvSlot);

            int seqKv = kvCache.CurrentLength;
            TensorRef cachedK = kvCache.GetKeysRef(kvSlot);
            TensorRef cachedV = kvCache.GetValuesRef(kvSlot);

            _kernels.LaunchAttentionF32(_state.QScratch, cachedK.DataPointer, cachedV.DataPointer, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqKv, numHeads, numKvHeads, headDim,
                positionOffset: positions[0], slidingWindow: 0, streamH);
        }
        else
        {
            _kernels.LaunchAttentionF32(_state.QScratch, _state.KScratch, _state.VScratch, _state.AttnOutput,
                seqQ: seqLen, seqKv: seqLen, numHeads, numKvHeads, headDim,
                positionOffset: 0, slidingWindow: 0, streamH);
        }

        Gemm(attn.OWeight, attn.OQt, _state.AttnOutput, _state.NormOutput, attn.OOutputDim, attn.OInputDim, seqLen);
    }

    /// <summary>Squared-ReLU FFN sub-layer forward (no gate) — up -> relu² -> down, reads/writes
    /// <c>_state.NormOutput</c>. Matches <c>NemotronHTransformerModel.ForwardFfnBody</c> exactly.</summary>
    private void ForwardFfnBody(in DeviceFfn ffn, int seqLen, int hiddenSize)
    {
        int intermediateSize = ffn.UpOutputDim;
        nint streamH = _stream.Handle;

        Gemm(ffn.UpWeight, ffn.UpQt, _state.NormOutput, _state.FfnIntermediate, ffn.UpOutputDim, ffn.UpInputDim, seqLen);
        _kernels.LaunchReluSquaredInplaceF32(_state.FfnIntermediate, seqLen * intermediateSize, streamH);
        Gemm(ffn.DownWeight, ffn.DownQt, _state.FfnIntermediate, _state.NormOutput, ffn.DownOutputDim, ffn.DownInputDim, seqLen);
    }

    /// <summary>Uploads the current call's host position span to <c>_state.PositionsDevice</c>
    /// (RoPE reads positions from device memory; <c>Embed</c> reads token ids from the host span
    /// directly, no upload needed there).</summary>
    /// <remarks>Unlike every other per-call buffer in <see cref="CudaNemotronHForwardState"/>,
    /// <c>PositionsDevice</c> is fixed-size (allocated once at <c>maxSeqLen</c> ints, never grown
    /// by <see cref="CudaNemotronHForwardState.EnsureCapacity"/>) — so a caller passing
    /// <c>positions.Length</c> greater than that fixed capacity would otherwise write past the
    /// buffer with no diagnostic. The per-element range check in <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>
    /// only catches monotonically-increasing overruns (a position value &gt;= maxSeq), not this
    /// case (e.g. duplicate/non-monotonic positions with <c>seqLen &gt; maxSeqLen</c>), so this
    /// guard checks the buffer's real allocated capacity directly instead.</remarks>
    private void UploadPositions(ReadOnlySpan<int> positions)
    {
        if (positions.Length > _state.MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(positions),
                $"positions.Length {positions.Length} exceeds the fixed PositionsDevice capacity " +
                $"{_state.MaxSeqLen}.");

        fixed (int* p = positions)
        {
            CudaDriverApi.cuMemcpyHtoD_v2(_state.PositionsDevice, (nint)p,
                (nuint)(positions.Length * sizeof(int))).ThrowOnError();
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numHeads = Config.NumAttentionHeads;
        int numKvHeads = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        float eps = Config.NormEpsilon;
        int maxSeq = Config.MaxSequenceLength;

        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        _context.MakeCurrent();
        _state.EnsureCapacity(seqLen);
        nint streamH = _stream.Handle;

        UploadPositions(positions);
        Embed(tokenIds, _state.HiddenState, hiddenSize);

        for (int layer = 0; layer < _layers.Length; layer++)
        {
            var lw = _layers[layer];

            // Save residual snapshot, then pre-sublayer RMSNorm into NormOutput — shared by all
            // three sub-layer kinds.
            CudaDriverApi.cuMemcpyDtoDAsync_v2(_state.Residual, _state.HiddenState,
                (nuint)((long)seqLen * hiddenSize * sizeof(float)), streamH).ThrowOnError();
            _kernels.LaunchRmsNormF32(_state.HiddenState, lw.AttnNormWeightDevice, _state.NormOutput,
                hiddenSize, eps, seqLen, streamH);

            switch (lw.Kind)
            {
                case HybridLayerKind.Ffn:
                    ForwardFfnBody(lw.Ffn!.Value, seqLen, hiddenSize);
                    break;
                case HybridLayerKind.Attention:
                    ForwardAttentionBody(lw.Attention!.Value, layer, seqLen, positions,
                        numHeads, numKvHeads, headDim, kvCache);
                    break;
                case HybridLayerKind.Ssm:
                    ForwardSsmBody(lw.Ssm!.Value, layer, seqLen, hiddenSize, eps);
                    break;
                default:
                    throw new InvalidOperationException(
                        $"Unknown HybridLayerKind {lw.Kind} at layer {layer}.");
            }

            // Residual add: HiddenState = Residual + NormOutput (NormOutput holds this
            // sub-layer's output — every ForwardXBody writes back into NormOutput).
            _kernels.LaunchAddF32(_state.Residual, _state.NormOutput, _state.HiddenState,
                seqLen * hiddenSize, streamH);
        }

        // Final RMSNorm over every row (matches the CPU host, which normalizes all seqLen rows
        // before the lm_head GEMM — unlike Qwen3HybridDense's optional lastTokenLogitsOnly this
        // model does not need since NemotronH's realistic vocab sizes don't hit the VRAM ceiling
        // that optimization exists for).
        _kernels.LaunchRmsNormF32(_state.HiddenState, _outputNormDevice, _state.HiddenState,
            hiddenSize, eps, seqLen, streamH);

        Gemm(_outputDevice, _outputQt, _state.HiddenState, _state.Logits,
             _outputOutputDim, _outputInputDim, seqLen);

        // cuMemcpyDtoH_v2 does not implicitly wait for this model's non-default _stream —
        // synchronize first (mirrors CudaQwen3HybridDenseTransformerModel's identical D2H tail).
        _stream.Synchronize();

        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        CudaDriverApi.cuMemcpyDtoH_v2(result.DataPointer, _state.Logits,
            (nuint)((long)seqLen * vocabSize * sizeof(float))).ThrowOnError();

        return result;
    }

    /// <summary>
    /// Forward with a caller-supplied per-sequence SSM state (the per-token recurrent state the
    /// continuous-batch scheduler threads so concurrent sequences don't share the model-owned
    /// default). Null falls back to the model-owned <c>_ssmCache</c> (single-sequence behaviour).
    /// Attention layers use <paramref name="kvCache"/> as usual. Mirrors
    /// <c>NemotronHTransformerModel.Forward(..., ISsmState?)</c> exactly.
    /// </summary>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache, ISsmState? ssmState)
    {
        _context.MakeCurrent();
        CudaNemotronHSsmStateCache? prev = _activeSsm;
        _activeSsm = ResolveSsm(ssmState);
        try { return Forward(tokenIds, positions, deviceId, kvCache); }
        finally { _activeSsm = prev; }
    }

    private CudaNemotronHSsmStateCache? ResolveSsm(ISsmState? ssmState)
    {
        if (ssmState is null) return null; // use _ssmCache
        if (ssmState is CudaNemotronHSsmStateCache cache)
        {
            if (cache.NumSsmLayers != _numSsmLayers)
                throw new ArgumentException(
                    $"SsmState covers {cache.NumSsmLayers} SSM layers but this model has {_numSsmLayers}.",
                    nameof(ssmState));
            return cache;
        }
        throw new ArgumentException(
            $"CudaNemotronHTransformerModel requires a {nameof(CudaNemotronHSsmStateCache)} for its " +
            $"SSM state; got {ssmState.GetType().Name}.",
            nameof(ssmState));
    }

    /// <inheritdoc/>
    /// <remarks>Re-zeroes the model-owned SSM cache (conv history + hidden state) used by every
    /// forward that does not carry a caller-supplied <see cref="ISsmState"/>. Callers that treat
    /// each forward as an independent sequence (perplexity windows, growing-context reprefill
    /// parity tests — see Task 14) must call this between sequences.</remarks>
    public void ResetSequenceState()
    {
        _context.MakeCurrent();
        _ssmCache.Reset();
    }

    /// <inheritdoc/>
    public bool RequiresPerSequenceState => true;

    /// <inheritdoc/>
    public bool SupportsThreadedSequenceState => true;

    /// <inheritdoc/>
    public IRecurrentSequenceState? CreateSequenceState()
    {
        _context.MakeCurrent();
        return new CudaNemotronHSsmStateCache(_ssm, _numSsmLayers);
    }

    /// <summary>
    /// Batched forward across sequences. NemotronH SSM state is per-token recurrent, so this
    /// threads each request's per-seq <see cref="SequenceForwardRequest.SsmState"/> through a
    /// per-sequence <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, ISsmState?)"/>
    /// call (no cross-sequence fusion). For 2+ requests every entry must supply a per-seq
    /// <c>SsmState</c> (a null would silently share the model-owned default and corrupt
    /// concurrent decode). LoRA adapters are not supported. Mirrors
    /// <c>NemotronHTransformerModel.ForwardBatch</c> exactly.
    /// </summary>
    public IReadOnlyList<ITensor> ForwardBatch(IReadOnlyList<SequenceForwardRequest> requests, int deviceId)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0) return Array.Empty<ITensor>();

        _context.MakeCurrent();

        for (int i = 0; i < requests.Count; i++)
        {
            if (requests[i].Adapter is not null)
                throw new NotSupportedException(
                    "CudaNemotronHTransformerModel.ForwardBatch does not support LoRA adapters.");
        }

        if (requests.Count >= 2)
        {
            for (int i = 0; i < requests.Count; i++)
            {
                if (requests[i].SsmState is null)
                    throw new ArgumentException(
                        $"CudaNemotronHTransformerModel.ForwardBatch with {requests.Count} requests requires " +
                        $"every request to supply a per-seq SsmState; request[{i}] has none.",
                        nameof(requests));
            }
        }

        var results = new ITensor[requests.Count];
        for (int i = 0; i < requests.Count; i++)
        {
            var r = requests[i];
            results[i] = Forward(r.TokenIds.Span, r.Positions.Span, deviceId, r.KvCache, r.SsmState);
        }
        return results;
    }
}
