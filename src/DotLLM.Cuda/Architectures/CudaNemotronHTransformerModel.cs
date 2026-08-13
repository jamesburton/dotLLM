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

    private readonly float _ropeTheta;
    private readonly int _ropeDim;

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
        => new(_attentionLayerCount, Config.NumKvHeads, Config.HeadDim, maxSeqLen, _deviceId);

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
        float ropeTheta, int ropeDim,
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
        _ropeTheta = ropeTheta;
        _ropeDim = ropeDim;
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
    /// <see cref="NemotronHTransformerModel.LoadFromGguf"/> (CPU) for all GGUF tensor-name
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
}
