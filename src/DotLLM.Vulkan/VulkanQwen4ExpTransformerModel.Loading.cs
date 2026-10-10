using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Vulkan;

public sealed unsafe partial class VulkanQwen4ExpTransformerModel
{
    /// <summary>Resident-capacity refusal override: <c>DOTLLM_VK_ALLOW_OVERCOMMIT=1</c> turns the pre-load refusal into a warning.</summary>
    private static bool AllowOvercommit
        => string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VK_ALLOW_OVERCOMMIT"), "1", StringComparison.Ordinal);

    /// <summary>
    /// Loads a Qwen4Exp model from an opened (possibly multi-shard) GGUF onto <paramref name="device"/>. The file must outlive the model.
    /// </summary>
    /// <param name="device">An initialized Vulkan device. Not owned.</param>
    /// <param name="gguf">The opened GGUF; must outlive the model.</param>
    /// <param name="config">Configuration extracted from <paramref name="gguf"/>.</param>
    /// <param name="spvDir">Directory containing compiled SPIR-V blobs.</param>
    /// <exception cref="NotSupportedException">The weights do not fit the device's resident capacity (and no overcommit override is set).</exception>
    public static VulkanQwen4ExpTransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir)
        => BuildFromGguf(device, gguf, config, spvDir, residentCapacityOverrideBytes: null);

    /// <summary>Test seam: <paramref name="residentCapacityOverrideBytes"/> replaces the device's resident capacity in the pre-load gate.</summary>
    internal static VulkanQwen4ExpTransformerModel BuildFromGguf(VulkanDevice device, GgufFile gguf, ModelConfig config, string spvDir,
        long? residentCapacityOverrideBytes)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(gguf);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(spvDir);
        if (config.Architecture != Architecture.Qwen4Exp || config.Qwen4Exp is not { } q4
            || config.GdnConfig is null || config.Moe is null || config.HybridLayout is null)
            throw new ArgumentException("VulkanQwen4ExpTransformerModel requires a fully populated Architecture.Qwen4Exp configuration.", nameof(config));
        var tensors = gguf.TensorsByName;
        var problems = Qwen4ExpTensors.FindProblems(tensors, config, includeTrunk: true, includeMtp: false);
        if (problems.Count > 0)
            throw new InvalidDataException("qwen4exp tensor table does not match the model contract: " +
                                           string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));
        if (q4.Ple is { } ple0 && ple0.Layers.Count > 1)
            throw new NotSupportedException("qwen4exp files carry one set of PLE hash constants; several PLE layers cannot be represented.");

        // Dense-attention capacity: exact while nb <= budget/blockSize, i.e. up to budget + block - 1 tokens.
        int denseLimit = q4.IndexerBlockSize > 0 ? q4.IndexerTopK + q4.IndexerBlockSize - 1 : config.MaxSequenceLength;
        int kvCapacity = Math.Min(config.MaxSequenceLength, denseLimit);

        // Residency gate BEFORE touching the device (the 122B WDDM-thrash class: refuse with numbers, do not page).
        var plan = Qwen4ExpResidencyPlan.Create(tensors, config, residentCapacityOverrideBytes ?? device.ResidentCapacityBytes(), kvCapacity,
            GC.GetGCMemoryInfo().TotalAvailableMemoryBytes);
        if (!plan.Fits)
        {
            string msg = "qwen4exp weights do not fit the Vulkan device's resident capacity: " + plan.Describe() + ". " +
                         "Quant types without a resident indexed-MoE kernel (everything but Q4_K/Q5_K/Q6_K/Q5_1/Q8_0 experts) are widened to F32 on upload.";
            if (!AllowOvercommit)
                throw new NotSupportedException(msg + " Set DOTLLM_VK_ALLOW_OVERCOMMIT=1 to load anyway (expect paging).");
            Console.Error.WriteLine("[dotLLM] WARNING: " + msg + " DOTLLM_VK_ALLOW_OVERCOMMIT=1: loading anyway.");
        }
        else if (plan.HostPressure)
            Console.Error.WriteLine("[dotLLM] WARNING: qwen4exp device weights plus the host-only n-gram table exceed physical RAM less headroom: " + plan.Describe());

        // The n-gram table: registered host-only BEFORE any upload so no staging/import path can ever take it.
        (nint, long)? hostOnly = null;
        if (tensors.TryGetValue(Qwen4ExpTensors.PerLayerTokenEmbd, out var tdesc))
        {
            nint tptr = gguf.TensorDataPointer(tdesc);
            long tbytes = Dequantize.RowByteSize(tdesc.Shape[0], tdesc.QuantizationType) * tdesc.Shape[1];
            VulkanWeightImportPolicy.RegisterHostOnly(tptr, tbytes);
            hostOnly = (tptr, tbytes);
        }

        var owned = new List<nint>();
        VulkanQwen3MoeHybridTransformerModel? core = null;
        var attnGr = new List<VulkanQwen4ExpGrWeights>();
        var ffnGr = new List<VulkanQwen4ExpGrWeights>();
        VulkanQwen4ExpGrWeights? head = null;
        var moeBundles = new List<VulkanQwen3MoeMoeUpload.LayerBundle>();
        Qwen4ExpGatedResidualKernel? gr = null;
        GroupRmsNormF32Kernel? groupRms = null;
        GdnPostScanGateF32Kernel? sigmoidGate = null;
        try
        {
            var phase = System.Diagnostics.Stopwatch.StartNew();
            var phases = new List<string>();
            void Mark(string what) { phases.Add($"{what}={phase.Elapsed.TotalSeconds:F1}s"); phase.Restart(); }
            var layers = BuildHybridLayers(gguf, tensors, config, owned);
            Mark("hybridLayers");
            var embDesc = tensors[Qwen4ExpTensors.TokenEmbd];
            nint embPtr = gguf.TensorDataPointer(embDesc);
            nint outPtr = embPtr; var outQt = embDesc.QuantizationType; int outM = embDesc.Shape[1], outK = embDesc.Shape[0];
            if (tensors.TryGetValue(Qwen4ExpTensors.Output, out var od))
            { outPtr = gguf.TensorDataPointer(od); outQt = od.QuantizationType; outM = od.Shape[1]; outK = od.Shape[0]; }

            // The hybrid's pre/post norms and output norm do not exist in qwen4exp (the gated residual replaces them): 1-element dummies.
            core = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                device, config, layers, outputNormWeight: [1f], outPtr, outQt, outM, outK, embPtr, embDesc.QuantizationType,
                spvDir, nCpuMoeLayers: 0);

            Mark("coreWeights");
            long weightBytes = 0;
            using (var staging = VulkanStagingBuffer.Create(device, 64L << 20))
            {
                for (int il = 0; il < config.NumLayers; il++)
                {
                    string b = $"blk.{il}.";
                    attnGr.Add(VulkanQwen4ExpGrWeights.Upload(device, staging, gguf, b + "hc_attn_norm.weight", b + "hc_attn_down.weight",
                        b + "hc_attn_up.weight", b + "hc_attn_inject.weight"));
                    ffnGr.Add(VulkanQwen4ExpGrWeights.Upload(device, staging, gguf, b + "hc_ffn_norm.weight", b + "hc_ffn_down.weight",
                        b + "hc_ffn_up.weight", b + "hc_ffn_inject.weight"));
                    weightBytes += attnGr[^1].Bytes + ffnGr[^1].Bytes;
                }
                head = VulkanQwen4ExpGrWeights.Upload(device, staging, gguf, Qwen4ExpTensors.OutputHcNorm, Qwen4ExpTensors.OutputHcDown,
                    Qwen4ExpTensors.OutputHcUp, inject: null);
                weightBytes += head.Bytes;
            }

            Mark("grWeights");
            // Routed experts are uploaded eagerly (resident): a model that cannot be resident was refused above, and the transient
            // per-forward upload path would be the 0.01 tok/s thrash class.
            // #874: one multi-threaded staging buffer for the whole load (parallel page-in + memcpy), and the next layers' banks
            // allocated on background threads while the current layer copies (vkAllocateMemory is ~0.45 s/GiB here). A/B switch:
            // DOTLLM_VULKAN_PARALLEL_UPLOAD=0 restores the serial per-layer staging and inline allocation.
            bool parallelUpload = !string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PARALLEL_UPLOAD"), "0", StringComparison.Ordinal);
            using (var upStaging = parallelUpload
                       ? VulkanStagingBuffer.CreateParallel(device, VulkanStagingBuffer.ParallelSlotBytes, VulkanStagingBuffer.ParallelSlotCount) : null)
            using (var prealloc = parallelUpload ? new VulkanBankPrealloc(device) : null)
            {
                const int AllocAhead = 2;
                long heapBytes = device.DeviceLocalHeapBytes();
                const long AheadMargin = 3L << 30;
                void ScheduleLayer(int il)
                {
                    if (prealloc is null || il >= config.NumLayers) return;
                    long[] sizes = VulkanQwen3MoeMoeUpload.BankAllocationSizes(layers[il].Moe, config.HiddenSize, residentQuant: true);
                    // Near the device-local heap boundary allocation ORDER decides what falls back to the slower heap: allocate those inline.
                    if (VulkanBankPrealloc.MayAllocateAhead(heapBytes, LiveDeviceBytes(device), prealloc.PendingBytes, sizes.Sum(), AheadMargin))
                        prealloc.Schedule(sizes);
                }
                for (int il = 0; il < AllocAhead; il++) ScheduleLayer(il);
                for (int il = 0; il < config.NumLayers; il++)
                {
                    moeBundles.Add(VulkanQwen3MoeMoeUpload.UploadLayer(device, layers[il].Moe, config.HiddenSize, residentQuant: true, upStaging, prealloc));
                    ScheduleLayer(il + AllocAhead);
                }
                upStaging?.WaitAll();   // the banks are consumed by compute right after the load: drain every queued copy first
            }

            Mark("moeBanks");
            Qwen4ExpPleBranch? ple = null;
            int pleLayer = -1;
            if (q4.Ple is { } pc)
            {
                pleLayer = pc.Layers[0];
                ple = BuildPleBranch(gguf, tensors, config, q4, pc, pleLayer, owned);
            }

            gr = Qwen4ExpGatedResidualKernel.Create(device, spvDir);
            groupRms = GroupRmsNormF32Kernel.Create(device, spvDir);
            sigmoidGate = GdnPostScanGateF32Kernel.Create(device, spvDir, sigmoidGate: true);

            Mark("ple+kernels");
            if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_MEM_TRACE"), "1", StringComparison.Ordinal))
                Console.Error.WriteLine($"[vulkan-load] phases: {string.Join(", ", phases)}; vkAllocateMemory={VulkanDevice.AllocateMemoryMilliseconds / 1000:F1}s, " +
                                        $"staging memcpy={VulkanStagingBuffer.MemcpyMilliseconds / 1000:F1}s/{VulkanStagingBuffer.MemcpyBytes / (1024 * 1024)} MiB (thread-sum), " +
                                        $"bank uploads={VulkanQwen3MoeMoeUpload.BanksMilliseconds / 1000:F1}s, submitter waited {VulkanStagingBuffer.CopyWaitMilliseconds / 1000:F1}s on memcpy and {VulkanBankPrealloc.WaitMilliseconds / 1000:F1}s on bank allocation; {device.MemorySnapshot()}");
            return new VulkanQwen4ExpTransformerModel(device, gguf, config, core, attnGr.ToArray(), ffnGr.ToArray(), head, moeBundles.ToArray(),
                ple, pleLayer, owned, hostOnly, gr, groupRms, sigmoidGate, kvCapacity, weightBytes);
        }
        catch
        {
            foreach (var m in moeBundles) m.Dispose();
            foreach (var g in attnGr) g.Dispose();
            foreach (var g in ffnGr) g.Dispose();
            head?.Dispose();
            gr?.Dispose(); groupRms?.Dispose(); sigmoidGate?.Dispose();
            core?.Dispose();
            foreach (nint p in owned) NativeMemory.AlignedFree((void*)p);
            if (hostOnly is { } h) VulkanWeightImportPolicy.UnregisterHostOnly(h.Item1);
            throw;
        }
    }

    /// <summary>Bytes this device object has allocated across all heaps (the sum is conservative once allocations fall back to the host heap).</summary>
    private static long LiveDeviceBytes(VulkanDevice device)
    {
        long sum = 0;
        for (int h = 0; h < 16; h++) sum += device.LiveBytesOnHeap(h);
        return sum;
    }

    private static Qwen3MoeLayerWeights[] BuildHybridLayers(GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t,
        ModelConfig config, List<nint> owned)
    {
        var layout = config.HybridLayout!;
        var gdn = config.GdnConfig!.Value;
        var layers = new Qwen3MoeLayerWeights[config.NumLayers];
        float[] dummyNorm = [1f];
        for (int il = 0; il < layers.Length; il++)
        {
            string b = $"blk.{il}.";
            bool isGdn = layout.LayerKind[il] == HybridLayerKind.GatedDeltaNet;
            layers[il] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = dummyNorm,
                PostAttnNormWeight = dummyNorm,
                Gdn = isGdn ? LoadGdn(b, gguf, t, gdn) : null,
                FullAttn = isGdn ? null : LoadAttention(b, gguf, t, layout.HeadCountKv[il], config.HeadDim),
                Moe = LoadMoe(il, gguf, t, config, owned),
            };
        }
        return layers;
    }

    private static float[] F32(GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, string name, long count)
    {
        var d = t[name];
        var r = new float[count];
        Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, r);
        return r;
    }

    private static GdnTokenMixingWeights LoadGdn(string b, GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, GatedDeltaNetConfig g)
    {
        int convDim = (2 * g.NKHead + g.NVHead) * g.DState;
        var qkv = t[b + "attn_qkv.weight"]; var gate = t[b + "attn_gate.weight"]; var alpha = t[b + "ssm_alpha.weight"];
        var beta = t[b + "ssm_beta.weight"]; var outp = t[b + "ssm_out.weight"];
        return new GdnTokenMixingWeights
        {
            QkvWeight = gguf.TensorDataPointer(qkv), QkvQuantType = qkv.QuantizationType, QkvInputDim = qkv.Shape[0], QkvOutputDim = qkv.Shape[1],
            GateWeight = gguf.TensorDataPointer(gate), GateQuantType = gate.QuantizationType, GateInputDim = gate.Shape[0], GateOutputDim = gate.Shape[1],
            A = F32(gguf, t, b + "ssm_a", g.NVHead),
            AlphaWeight = gguf.TensorDataPointer(alpha), AlphaQuantType = alpha.QuantizationType, AlphaInputDim = alpha.Shape[0], AlphaOutputDim = alpha.Shape[1],
            BetaWeight = gguf.TensorDataPointer(beta), BetaQuantType = beta.QuantizationType, BetaInputDim = beta.Shape[0], BetaOutputDim = beta.Shape[1],
            Conv1dWeight = F32(gguf, t, b + "ssm_conv1d.weight", (long)g.DConv * convDim),
            Conv1dBias = new float[convDim],
            DtBias = F32(gguf, t, b + "ssm_dt.bias", g.NVHead),
            SsmNormWeight = F32(gguf, t, b + "ssm_norm.weight", g.DState),
            OutWeight = gguf.TensorDataPointer(outp), OutQuantType = outp.QuantizationType, OutInputDim = outp.Shape[0], OutOutputDim = outp.Shape[1],
        };
    }

    private static Qwen3FullAttnWeights LoadAttention(string b, GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, int nKv, int headDim)
    {
        var q = t[b + "attn_q.weight"]; var k = t[b + "attn_k.weight"]; var v = t[b + "attn_v.weight"]; var o = t[b + "attn_output.weight"];
        return new Qwen3FullAttnWeights
        {
            QWeight = gguf.TensorDataPointer(q), QQuantType = q.QuantizationType, QInputDim = q.Shape[0], QOutputDim = q.Shape[1],
            KWeight = gguf.TensorDataPointer(k), KQuantType = k.QuantizationType, KInputDim = k.Shape[0], KOutputDim = k.Shape[1],
            VWeight = gguf.TensorDataPointer(v), VQuantType = v.QuantizationType, VInputDim = v.Shape[0], VOutputDim = v.Shape[1],
            OWeight = gguf.TensorDataPointer(o), OQuantType = o.QuantizationType, OInputDim = o.Shape[0], OOutputDim = o.Shape[1],
            NumKvHeads = nKv,
            QNormWeight = F32(gguf, t, b + "attn_q_norm.weight", headDim),
            KNormWeight = F32(gguf, t, b + "attn_k_norm.weight", headDim),
        };
    }

    private static MoeLayerWeights LoadMoe(int il, GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, ModelConfig config, List<nint> owned)
    {
        var moe = config.Moe!;
        string p = $"blk.{il}.";
        int hidden = config.HiddenSize, ne = moe.NumExperts, inter = moe.MoeIntermediateSize;
        int shared = moe.SharedExpertIntermediateSize ?? inter;

        nint SharedF32(string name, long count)
        {
            var d = t[name];
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            owned.Add(dst);
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, new Span<float>((void*)dst, (int)count));
            return dst;
        }

        var gateD = t[p + "ffn_gate_exps.weight"]; var upD = t[p + "ffn_up_exps.weight"]; var downD = t[p + "ffn_down_exps.weight"];
        var sg = t[p + "ffn_gate_shexp.weight"]; var su = t[p + "ffn_up_shexp.weight"]; var sd = t[p + "ffn_down_shexp.weight"];
        return new MoeLayerWeights(
            gate: F32(gguf, t, p + "ffn_gate_inp.weight", (long)ne * hidden),
            w1: new nint[ne], w2: new nint[ne], w3: new nint[ne],
            numExperts: ne, numExpertsPerTok: moe.NumExpertsPerTok, hiddenSize: hidden, intermediateSize: inter,
            normTopKProb: moe.NormTopKProb,
            sharedGateProj: [SharedF32(p + "ffn_gate_shexp.weight", (long)shared * hidden)],
            sharedUpProj: [SharedF32(p + "ffn_up_shexp.weight", (long)shared * hidden)],
            sharedDownProj: [SharedF32(p + "ffn_down_shexp.weight", (long)hidden * shared)],
            sharedIntermediateSize: shared,
            sharedExpertGate: F32(gguf, t, p + "ffn_gate_inp_shexp.weight", hidden),
            gateExpsRaw: gguf.TensorDataPointer(gateD), gateExpsRawQt: gateD.QuantizationType, gateExpsMDim: inter, gateExpsKDim: hidden,
            upExpsRaw: gguf.TensorDataPointer(upD), upExpsRawQt: upD.QuantizationType, upExpsMDim: inter, upExpsKDim: hidden,
            downExpsRaw: gguf.TensorDataPointer(downD), downExpsRawQt: downD.QuantizationType, downExpsMDim: hidden, downExpsKDim: inter,
            sharedGateRaw: [gguf.TensorDataPointer(sg)], sharedGateRawQt: sg.QuantizationType,
            sharedUpRaw: [gguf.TensorDataPointer(su)], sharedUpRawQt: su.QuantizationType,
            sharedDownRaw: [gguf.TensorDataPointer(sd)], sharedDownRawQt: sd.QuantizationType);
    }

    /// <summary>
    /// The n-gram branch exactly as the CPU oracle builds it (its own <see cref="Qwen4ExpPleBranch"/>); the key / value projections are
    /// dequantised to F32 once into host memory and run through the CPU GEMM. The table pointer is borrowed from the mmap and never copied.
    /// </summary>
    private static Qwen4ExpPleBranch BuildPleBranch(GgufFile gguf, IReadOnlyDictionary<string, GgufTensorDescriptor> t, ModelConfig config,
        Qwen4ExpConfig q4, Qwen4ExpPleConfig ple, int layer, List<nint> owned)
    {
        string b = $"blk.{layer}.";
        int hidden = config.HiddenSize, hc = q4.HyperConnectionCount, hcDim = hc * hidden;
        var tdesc = t[Qwen4ExpTensors.PerLayerTokenEmbd];

        Qwen4ExpProjection HostProj(string name)
        {
            var d = t[name];
            int inDim = d.Shape[0], outDim = d.Shape[1];
            long count = (long)inDim * outDim;
            nint w = (nint)NativeMemory.AlignedAlloc((nuint)(count * sizeof(float)), 64);
            owned.Add(w);
            Dequantize.ToFloat32(gguf.TensorDataPointer(d), count, d.QuantizationType, new Span<float>((void*)w, checked((int)count)));
            return (input, output, tokens) =>
            {
                fixed (float* x = input) fixed (float* y = output)
                    MatMul.GemmF32((float*)w, x, y, outDim, inDim, tokens);
            };
        }

        var convRaw = F32(gguf, t, b + "ple_conv1d.weight", (long)ple.ConvKernel * hcDim);   // [C][K] (K fastest)
        return new Qwen4ExpPleBranch(
            gguf.TensorDataPointer(tdesc), tdesc.QuantizationType, tdesc.Shape[1], ple.RowDim,
            ple.NgramSize, ple.HeadsPerNgram, ple.EosTokenId, ple.ConvKernel,
            ple.LayerMultipliers.Select(v => unchecked((long)v)).ToArray(),
            ple.HeadOffsets.Select(v => checked((long)v)).ToArray(),
            ple.HeadVocabSizes.Select(v => checked((long)v)).ToArray(),
            hc, hidden, config.NormEpsilon,
            F32(gguf, t, b + "ple_norm_key.weight", hcDim), F32(gguf, t, b + "ple_norm_query.weight", hcDim), F32(gguf, t, b + "ple_norm_conv.weight", hcDim),
            Qwen4ExpPle.TransposeConvWeight(convRaw, hcDim, ple.ConvKernel),
            HostProj(b + "ple_key.weight"), HostProj(b + "ple_value.weight"));
    }
}
