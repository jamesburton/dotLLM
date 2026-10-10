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

namespace DotLLM.Vulkan;

/// <summary>
/// Vulkan qwen4exp speculative decoding (issue #820, stage 2): the MTP draft head, per-row recurrent-state snapshots for the verify forward
/// and the (2..8-row) verify path. Mirrors the CPU oracle <c>Qwen4ExpTransformerModel</c> (<c>Qwen4ExpTransformerModel.Mtp.cs</c>).
/// </summary>
/// <remarks>
/// <para><b>Head.</b> One QSA + 512-expert MoE block on the 4-stream gated residual (<c>blk.48.*</c> of the separate <c>mtp-*.gguf</c>), fed by
/// the trunk's residual of the previous position and the next token: <c>R' = eh_proj([enorm(embed(t)) ; hnorm(R_s)])</c> per stream, GR-read,
/// QSA over the head's own dense K/V cells, GR-write, GR-read, MoE, GR-write, head mixer, the trunk's LM head. The block runs on a one-layer
/// <see cref="VulkanQwen3MoeHybridTransformerModel"/> (same kernels, own scratch, own K/V), the residual plumbing on this class's.
/// <c>eh_proj</c> is split once at load into its embedding and hidden halves (row-wise block copies of the Q8_0 matrix) so the front needs two
/// plain matmuls and one add instead of a concatenation.</para>
/// <para><b>Exactness.</b> Every emitted token is the verify row's argmax (the decoder accepts a draft only when it equals it), so the output is
/// the verify path's greedy, never a draft's. On Vulkan the multi-row kernels are not bit-identical to the 1-row decode (KL 2e-3..1.4e-2,
/// top-1 identical in the #876 measurements), so a near-tie can resolve differently from plain decode; that rate is measured, not assumed.</para>
/// </remarks>
public sealed unsafe partial class VulkanQwen4ExpTransformerModel : IMtpHeadAttachable
{
    /// <summary>Cells the head processes per command buffer when absorbing a trunk batch (bounds the head core's scratch).</summary>
    internal const int AbsorbChunk = 64;

    private string? _spvDir;
    private MtpHead? _mtp;

    // row snapshots
    private GdnScanMultiTokenSnapshotF32Kernel? _snapKernel;
    private Q4GdnRowSnapshots? _snapGdn;
    private int _snapCapacity;
    private bool _snapValid;
    private int _snapBase, _snapRowsRecorded;

    private float[]? _mtpLogits;

    /// <summary>Env var selecting the MTP <c>hnorm</c> convention: <c>stream</c> (default) or <c>joint</c> (same as the CPU oracle).</summary>
    public const string HnormEnvVar = Qwen4ExpTransformerModel.HnormEnvVar;

    /// <summary>True (default): <c>pre_fc_norm_hidden</c> normalises each residual stream on its own; false: one RMS over all streams.</summary>
    public bool MtpHnormPerStream { get; set; } = Environment.GetEnvironmentVariable(HnormEnvVar) != "joint";

    /// <summary>Path-proof counters (tests): draft steps, absorbed cells, verify forwards that recorded snapshots, restores.</summary>
    internal long DraftSteps, AbsorbedCells, SnapshotForwards, SnapshotRestores;

    /// <summary>Diagnostic: cumulative wall ms spent in the post-forward absorb, in snapshot restores and in draft steps (reset with <see cref="ResetMtpTimings"/>).</summary>
    public (double AbsorbMs, double RestoreMs, double DraftMs, long DraftSteps) MtpTimings
        => (_absorbTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency, _restoreTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency,
            _draftTicks * 1000.0 / System.Diagnostics.Stopwatch.Frequency, DraftSteps);

    /// <summary>Zeroes <see cref="MtpTimings"/>.</summary>
    public void ResetMtpTimings() { _absorbTicks = _restoreTicks = _draftTicks = 0; DraftSteps = 0; }

    private long _absorbTicks, _restoreTicks, _draftTicks;

    private sealed class MtpHead : IDisposable
    {
        public required VulkanQwen3MoeHybridTransformerModel Core;
        public required VulkanQwen4ExpGrWeights AttnGr, FfnGr, HeadGr;
        public required VulkanQwen3MoeMoeUpload.LayerBundle Moe;
        public required VulkanDevice.Buffer EhEmb, EhHid, Enorm, Hnorm;
        public required QuantizationType EhEmbQt, EhHidQt;
        public required VulkanDevice.Buffer E, A, HN, B, Pair, RFront, RWork;
        public required List<nint> Owned;
        public GgufFile? OwnedFile;
        /// <summary>Layer index of the head's attention block inside <see cref="Core"/> (0 for the real head's one-layer core).</summary>
        public int AttnLayer;
        /// <summary>Stand-in head (benchmarking): core, MoE bundle and GR weights belong to the trunk and are not disposed here.</summary>
        public bool SharedWithTrunk;

        public void Dispose()
        {
            if (SharedWithTrunk)
            {
                EhEmb.Dispose(); EhHid.Dispose(); Enorm.Dispose(); Hnorm.Dispose();
                E.Dispose(); A.Dispose(); HN.Dispose(); B.Dispose(); Pair.Dispose(); RFront.Dispose(); RWork.Dispose();
                foreach (nint p in Owned) NativeMemory.AlignedFree((void*)p);
                Owned.Clear();
                return;
            }
            Moe.Dispose(); AttnGr.Dispose(); FfnGr.Dispose(); HeadGr.Dispose();
            EhEmb.Dispose(); EhHid.Dispose(); Enorm.Dispose(); Hnorm.Dispose();
            E.Dispose(); A.Dispose(); HN.Dispose(); B.Dispose(); Pair.Dispose(); RFront.Dispose(); RWork.Dispose();
            Core.Dispose();
            foreach (nint p in Owned) NativeMemory.AlignedFree((void*)p);
            Owned.Clear();
            OwnedFile?.Dispose();
        }
    }

    // ───────────────────────────── head loading ─────────────────────────────

    /// <inheritdoc/>
    public bool HasMtpHead => _mtp is not null;

    /// <inheritdoc/>
    public void AttachMtpHead(string path)
    {
        var file = GgufFile.Open(path);
        try { AttachMtpHead(file, ownsFile: true); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>
    /// Attaches the MTP head found in <paramref name="headFile"/> (the <c>blk.{NumLayers}.*</c> QSA + MoE block, its <c>nextn.*</c> tensors and
    /// head mixer) and uploads it to the device. Embedding and LM head are the trunk's.
    /// </summary>
    /// <param name="headFile">An opened GGUF carrying the head; must outlive the model unless <paramref name="ownsFile"/>.</param>
    /// <param name="ownsFile">Dispose <paramref name="headFile"/> with the model.</param>
    public void AttachMtpHead(GgufFile headFile, bool ownsFile = false)
    {
        ArgumentNullException.ThrowIfNull(headFile);
        if (_mtp is not null) throw new InvalidOperationException("An MTP head is already attached.");
        if (_spvDir is null) throw new InvalidOperationException("The model was not built by BuildFromGguf (no SPIR-V directory).");
        var tensors = headFile.TensorsByName;
        var cfg = Config with { NextnPredictLayers = 1 };
        var problems = Qwen4ExpTensors.FindProblems(tensors, cfg, includeTrunk: false, includeMtp: true);
        if (problems.Count > 0)
            throw new InvalidDataException("qwen4exp MTP head tensor table does not match the contract: " +
                                           string.Join("; ", problems.Take(8)) + (problems.Count > 8 ? $"; (+{problems.Count - 8} more)" : ""));

        int il = Config.NumLayers, H = _hidden, S = _streams, hcDim = S * H;
        string b = $"blk.{il}.", n = b + "nextn.";
        long headBytes = 0;
        foreach (var kv in tensors)
        {
            if (kv.Key is Qwen4ExpTensors.TokenEmbd or Qwen4ExpTensors.Output) continue;   // the trunk's embedding / LM head are used
            long rows = 1;
            for (int d = 1; d < kv.Value.Shape.Rank; d++) rows *= kv.Value.Shape[d];
            headBytes += Dequantize.RowByteSize(kv.Value.Shape[0], kv.Value.QuantizationType) * rows;
        }
        EnsureDeviceHeadroom(headBytes + DefaultSnapshotBytes());
        var owned = new List<nint>();
        VulkanQwen3MoeHybridTransformerModel? core = null;
        VulkanQwen4ExpGrWeights? attnGr = null, ffnGr = null, headGr = null;
        VulkanQwen3MoeMoeUpload.LayerBundle? moe = null;
        VulkanDevice.Buffer? enorm = null, hnorm = null, ehEmb = null, ehHid = null;
        var bufs = new List<VulkanDevice.Buffer>();
        try
        {
            int nKv = tensors[b + "attn_k.weight"].Shape[1] / Config.HeadDim;
            var layers = new[]
            {
                new Qwen3MoeLayerWeights
                {
                    AttnNormWeight = [1f], PostAttnNormWeight = [1f], Gdn = null,
                    FullAttn = LoadAttention(b, headFile, tensors, nKv, Config.HeadDim),
                    Moe = LoadMoe(il, headFile, tensors, Config, owned),
                },
            };
            // A one-layer attention core: the head's block reuses the validated QSA / MoE kernels. Its embedding table and LM head are never
            // read (the trunk's are used), so they are 1-row stand-ins that cost nothing.
            var headCfg = Config with
            {
                NumLayers = 1, VocabSize = 1, NumKvHeads = nKv, NextnPredictLayers = 0,
                HybridLayout = new HybridLayerLayout
                {
                    LayerKind = [HybridLayerKind.Attention], HeadCountKv = [nKv], FeedForwardLength = [0],
                },
            };
            var tensorsT = _gguf.TensorsByName;
            var embDesc = tensorsT[Qwen4ExpTensors.TokenEmbd];
            nint embPtr = _gguf.TensorDataPointer(embDesc);
            core = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                _device, headCfg, layers, outputNormWeight: [1f], embPtr, embDesc.QuantizationType, 1, H, embPtr, embDesc.QuantizationType,
                _spvDir, nCpuMoeLayers: 0);
            if (core.Q4EnsureCapacity(AbsorbChunk)) core.Q4InvalidateCaches();

            long weightBytes = 0;
            QuantizationType embQt, hidQt;
            using (var staging = VulkanStagingBuffer.Create(_device, 64L << 20))
            {
                attnGr = VulkanQwen4ExpGrWeights.Upload(_device, staging, headFile, b + "hc_attn_norm.weight", b + "hc_attn_down.weight",
                    b + "hc_attn_up.weight", b + "hc_attn_inject.weight");
                ffnGr = VulkanQwen4ExpGrWeights.Upload(_device, staging, headFile, b + "hc_ffn_norm.weight", b + "hc_ffn_down.weight",
                    b + "hc_ffn_up.weight", b + "hc_ffn_inject.weight");
                headGr = VulkanQwen4ExpGrWeights.Upload(_device, staging, headFile, n + "hc_head_norm.weight", n + "hc_head_down.weight",
                    n + "hc_head_up.weight", inject: null);
                weightBytes += attnGr.Bytes + ffnGr.Bytes + headGr.Bytes;

                // eh_proj = cat(fc_embedding, fc_hidden) along the input axis: row o holds [fc_embedding(o, :), fc_hidden(o, :)], so each half of
                // every row is a whole number of quant blocks (H is a multiple of the block size) and the two halves split by row-wise byte copy.
                var eh = tensors[n + "eh_proj.weight"];
                if (eh.Shape[0] != 2 * H || eh.Shape[1] != H)
                    throw new InvalidDataException($"nextn.eh_proj is [{eh.Shape[0]}, {eh.Shape[1]}]; expected [{2 * H}, {H}].");
                long fullRow = Dequantize.RowByteSize(2 * H, eh.QuantizationType), halfRow = Dequantize.RowByteSize(H, eh.QuantizationType);
                if (fullRow != 2 * halfRow)
                    throw new NotSupportedException($"nextn.eh_proj quant {eh.QuantizationType} cannot be split at the embedding/hidden boundary.");
                nint src = headFile.TensorDataPointer(eh);
                nint pe = (nint)NativeMemory.AlignedAlloc((nuint)(halfRow * H), 64), ph = (nint)NativeMemory.AlignedAlloc((nuint)(halfRow * H), 64);
                owned.Add(pe); owned.Add(ph);
                for (int o = 0; o < H; o++)
                {
                    Buffer.MemoryCopy((void*)(src + (nint)(o * fullRow)), (void*)(pe + (nint)(o * halfRow)), halfRow, halfRow);
                    Buffer.MemoryCopy((void*)(src + (nint)(o * fullRow + halfRow)), (void*)(ph + (nint)(o * halfRow)), halfRow, halfRow);
                }
                ehEmb = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, staging, pe, eh.QuantizationType, H, H, false, out embQt, out long b1);
                ehHid = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, staging, ph, eh.QuantizationType, H, H, false, out hidQt, out long b2);
                weightBytes += b1 + b2;
                enorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, F32(headFile, tensors, n + "enorm.weight", H));
                hnorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, F32(headFile, tensors, n + "hnorm.weight", hcDim));
            }
            moe = VulkanQwen3MoeMoeUpload.UploadLayer(_device, layers[0].Moe, H, residentQuant: true);

            long rowBytes = (long)hcDim * 4;
            VulkanDevice.Buffer Scratch(long bytes) { var x = _device.AllocateDeviceLocal(bytes); bufs.Add(x); return x; }
            var head = new MtpHead
            {
                Core = core, AttnGr = attnGr, FfnGr = ffnGr, HeadGr = headGr, Moe = moe,
                EhEmb = ehEmb!, EhHid = ehHid!, Enorm = enorm!, Hnorm = hnorm!, EhEmbQt = embQt, EhHidQt = hidQt,
                E = Scratch((long)AbsorbChunk * H * 4), A = Scratch((long)AbsorbChunk * H * 4),
                HN = Scratch(AbsorbChunk * rowBytes), B = Scratch(AbsorbChunk * rowBytes),
                Pair = Scratch(AbsorbChunk * rowBytes), RFront = Scratch(AbsorbChunk * rowBytes), RWork = Scratch(AbsorbChunk * rowBytes),
                Owned = owned, OwnedFile = ownsFile ? headFile : null,
            };
            // The head's absorb chunk and the trunk's embedding gather / GR plumbing both run AbsorbChunk rows: size the trunk scratch up front.
            EnsureScratch(AbsorbChunk);
            if (_core.Q4EnsureCapacity(AbsorbChunk)) { _gr.InvalidateDescriptorCache(); _groupRms.InvalidateDescriptorCache(); _sigmoidGate.InvalidateDescriptorCache(); }
            _mtp = head;
            // Allocate the verify snapshots NOW: an out-of-memory belongs at attach time (where the caller declines speculation), not mid-decode.
            if (SupportsRecurrentRowSnapshots) EnsureRowSnapshots(DefaultSnapshotRows);
            if (Environment.GetEnvironmentVariable("DOTLLM_VK_Q4E_VERBOSE") == "1")
                Console.Error.WriteLine($"[dotLLM] qwen4exp MTP head attached: {weightBytes / (1024.0 * 1024.0):F0} MiB of dense head weights");
        }
        catch
        {
            moe?.Dispose(); attnGr?.Dispose(); ffnGr?.Dispose(); headGr?.Dispose(); core?.Dispose();
            enorm?.Dispose(); hnorm?.Dispose(); ehEmb?.Dispose(); ehHid?.Dispose();
            foreach (var x in bufs) x.Dispose();
            foreach (nint p in owned) NativeMemory.AlignedFree((void*)p);
            throw;
        }
    }


    /// <summary>
    /// BENCHMARKING AID, not a quality feature: attaches a stand-in head built from the trunk's own QSA layer <paramref name="trunkLayer"/>
    /// (its attention, MoE bank and gated-residual modules, shared with the trunk) with a random <c>eh_proj</c>. It costs the same per draft
    /// step as the real head (a QSA + 512-expert MoE block, the trunk's LM head) but needs only ~60 MB of extra device memory, so the round
    /// time of the whole speculative path can be measured on a checkpoint that leaves no room for the real head (UD-Q4_K_XL fills this
    /// box's device budget). Its drafts are noise: use the real head for acceptance.
    /// </summary>
    /// <param name="trunkLayer">Index of a QSA layer of the trunk.</param>
    public void AttachStandInMtpHead(int trunkLayer)
    {
        if (_mtp is not null) throw new InvalidOperationException("An MTP head is already attached.");
        if ((uint)trunkLayer >= (uint)Config.NumLayers || Config.HybridLayout!.LayerKind[trunkLayer] == HybridLayerKind.GatedDeltaNet)
            throw new ArgumentOutOfRangeException(nameof(trunkLayer), "The stand-in head needs a QSA layer.");
        int H = _hidden, S = _streams, hcDim = S * H;
        var owned = new List<nint>();
        var bufs = new List<VulkanDevice.Buffer>();
        VulkanDevice.Buffer Scratch(long bytes) { var x = _device.AllocateDeviceLocal(bytes); bufs.Add(x); return x; }
        try
        {
            long rowBytes = (long)hcDim * 4;
            long q8Bytes = (long)H * (H / 32) * 34;
            nint raw = (nint)NativeMemory.AlignedAlloc((nuint)q8Bytes, 64);
            owned.Add(raw);
            var rnd = new Random(1234);
            var blk = new byte[34];
            for (long i = 0; i < (long)H * (H / 32); i++)
            {
                rnd.NextBytes(blk);
                BitConverter.TryWriteBytes(blk.AsSpan(0, 2), (Half)0.004f);
                Marshal.Copy(blk, 0, raw + (nint)(i * 34), 34);
            }
            VulkanDevice.Buffer ehEmb, ehHid, enorm, hnorm;
            QuantizationType q1, q2;
            using (var staging = VulkanStagingBuffer.Create(_device, 64L << 20))
            {
                ehEmb = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, staging, raw, QuantizationType.Q8_0, H, H, false, out q1, out _);
                ehHid = VulkanQwen3MoeHybridWeights.UploadProjectionMatrix(_device, staging, raw, QuantizationType.Q8_0, H, H, false, out q2, out _);
                enorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, Enumerable.Repeat(1f, H).ToArray());
                hnorm = VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, Enumerable.Repeat(1f, hcDim).ToArray());
            }
            _mtp = new MtpHead
            {
                Core = _core, AttnGr = _attnGr[trunkLayer], FfnGr = _ffnGr[trunkLayer], HeadGr = _headGr, Moe = _moe[trunkLayer],
                EhEmb = ehEmb, EhHid = ehHid, Enorm = enorm, Hnorm = hnorm, EhEmbQt = q1, EhHidQt = q2,
                E = Scratch((long)AbsorbChunk * H * 4), A = Scratch((long)AbsorbChunk * H * 4),
                HN = Scratch(AbsorbChunk * rowBytes), B = Scratch(AbsorbChunk * rowBytes),
                Pair = Scratch(AbsorbChunk * rowBytes), RFront = Scratch(AbsorbChunk * rowBytes), RWork = Scratch(AbsorbChunk * rowBytes),
                Owned = owned, AttnLayer = trunkLayer, SharedWithTrunk = true,
            };
            EnsureScratch(AbsorbChunk);
            if (_core.Q4EnsureCapacity(AbsorbChunk)) { _gr.InvalidateDescriptorCache(); _groupRms.InvalidateDescriptorCache(); _sigmoidGate.InvalidateDescriptorCache(); }
            if (SupportsRecurrentRowSnapshots) EnsureRowSnapshots(DefaultSnapshotRows);
        }
        catch
        {
            foreach (var x in bufs) x.Dispose();
            foreach (nint p in owned) NativeMemory.AlignedFree((void*)p);
            throw;
        }
    }


    /// <summary>
    /// Refuses, with numbers, an attach that would not fit the device-local heap next to the resident trunk. This is a pure estimate and never
    /// touches the device on purpose: on this box the released trunk (UD-Q4_K_XL, ~69 GiB) fills the 69.8 GiB device-local heap, a further
    /// upload spills into the shared heap and then fails at the next submit with out-of-memory, and after one such failure EVERY later
    /// allocation fails (measured: even a 113 MiB sequence state) - so the only safe check is the one that runs before anything is allocated.
    /// <c>DOTLLM_VK_ALLOW_OVERCOMMIT=1</c> skips it (same switch as the load-time residency gate).
    /// </summary>
    private void EnsureDeviceHeadroom(long headBytes)
    {
        if (AllowOvercommit) return;
        // #880: compare like with like. The capacity is what the device can really hold resident (UMA: device-local heap PLUS the shared
        // heap the trunk already spills into - ~11 GiB of the real file lives there; discrete: VRAM only), the trunk is what this process
        // holds in those same heaps, and other processes' GPU memory is subtracted. The old check compared ALL live bytes with the
        // device-local heap alone, so it refused with ~30 GiB of the shared heap unused.
        // The usable capacity is the OS limit (UMA: ~0.64 x RAM, measured), not the sum of the advertised heaps, and the allocation that
        // crosses it does not fail - the NEXT submit does, and the device is then unusable. No extra headroom is subtracted: the cap
        // already is the point of failure and everything counted below is real.
        long local = VulkanMemoryCapacity.UsableCapacityBytes(_device.ResidentCapacityBytes(),
            GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, _device.PhysicalDeviceTypeValue);
        long trunk = _device.ResidentLiveBytes();
        long others = _device.ReadOtherProcessPressure()?.OtherBytes ?? 0;
        long scratch = Qwen4ExpResidencyPlan.KvBytes(Config, _kvCapacity) + (1L << 30) + (long)AbsorbChunk * _streams * _hidden * 4 * 8;
        long need = trunk + headBytes + scratch + others;
        if (need <= local) return;
        static string G(long b) => $"{b / (double)(1L << 30):F1} GiB";
        throw new NotSupportedException(
            $"The MTP head does not fit the device-local heap next to the resident trunk: resident {G(trunk)} + head {G(headBytes)} + KV/scratch {G(scratch)} = " +
            $"{G(need)} (incl. {G(others)} held by other processes) against a {G(local)} resident budget (a failed upload would leave the device unable to allocate at all). Use a smaller trunk quantisation, " +
            "or set DOTLLM_VK_ALLOW_OVERCOMMIT=1 to try anyway; decoding continues without speculation.");
    }


    /// <summary>Verify rows with snapshot scratch allocated at attach (a K = 4 round verifies 5 rows and snapshots 4); larger rounds grow it.</summary>
    internal const int DefaultSnapshotRows = 4;

    private long DefaultSnapshotBytes()
    {
        if (!SupportsRecurrentRowSnapshots) return 0;
        var g = _defaultState.Gdn;
        return (long)DefaultSnapshotRows * g.NumGdnLayers * ((long)g.GdnStateElements + g.ConvStateElements) * sizeof(float);
    }

    // ───────────────────────────── IModel MTP surface ─────────────────────────────

    /// <inheritdoc/>
    public bool SupportsMtp => _mtp is not null;

    /// <inheritdoc/>
    /// <remarks>A weight-bandwidth-bound 6B-active MoE: MTP is tried first and the gate still re-measures (same prior as the CPU oracle).</remarks>
    public long MtpGatePriorBytes => long.MaxValue;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => CreateMtpState(Config.MaxSequenceLength);

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength)
    {
        if (_mtp is not { } h) return null;
        int len = Math.Min(maxSequenceLength, _kvCapacity);
        return new VulkanQwen4ExpMtpState(_device, h.Core.Q4CreateKvCache(len), _streams * _hidden, len);
    }

    private VulkanQwen4ExpMtpState? RequireMtpState(IMtpState? state)
    {
        if (state is null || _mtp is null) return null;   // capability off: a state handed to a head-less model is a silent no-op
        return state as VulkanQwen4ExpMtpState
            ?? throw new ArgumentException($"qwen4exp on Vulkan needs a VulkanQwen4ExpMtpState; got {state.GetType().Name}.", nameof(state));
    }

    /// <inheritdoc/>
    /// <remarks>Model-owned state. Returns a logit row per input position while the batch is a verify-sized one (at most 32 rows), else the last row.</remarks>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
    {
        RejectAdapter(adapter);
        return ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, mtp: RequireMtpState(mtpState),
                           allRows: tokenIds.Length <= VulkanQwen4ExpMtpState.CaptureRows);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                           IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState, bool lastTokenLogitsOnly)
    {
        RejectAdapter(adapter);
        return ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly, mtp: RequireMtpState(mtpState),
                           allRows: !lastTokenLogitsOnly && tokenIds.Length <= VulkanQwen4ExpMtpState.CaptureRows);
    }

    private static void RejectAdapter(ILoraAdapter? adapter)
    {
        if (adapter is not null) throw new NotSupportedException("LoRA adapters are not supported by the Vulkan qwen4exp model (CPU only, #845).");
    }

    // ───────────────────────────── recurrent row snapshots ─────────────────────────────

    /// <inheritdoc/>
    public bool SupportsRecurrentRowSnapshots =>
        _spvDir is not null && Config.GdnConfig is { } g && g.DState <= 128 && GdnScanMultiTokenSnapshotF32Kernel.IsAvailable(_spvDir);

    /// <inheritdoc/>
    /// <remarks>
    /// The verify forward swaps the GDN scan for its snapshot twin (bit-identical state evolution), copies each row's conv window and records
    /// the n-gram branch's hash window / conv history per row. Device scratch is <c>(T-1) x layers x (NVHead x DState^2 + conv)</c> floats
    /// (~113 MiB per row on the released size), grown to the largest batch seen. A rejection then costs a state copy, not a replay forward.
    /// </remarks>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
                                                 IKvCache? kvCache, IMtpState? mtpState)
    {
        if (!SupportsRecurrentRowSnapshots)
            throw new NotSupportedException("The GDN snapshot scan kernel is unavailable (SupportsRecurrentRowSnapshots=false).");
        SnapshotForwards++;
        return ForwardCore(tokenIds, positions, deviceId, _defaultState, kvCache, lastTokenLogitsOnly: false, mtp: RequireMtpState(mtpState),
                           snapRows: Math.Max(tokenIds.Length - 1, 0), allRows: true);
    }

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if (!_snapValid || (uint)row > (uint)_snapRowsRecorded)
            throw new InvalidOperationException(
                $"No recurrent snapshot for row {row}: the last ForwardWithRecurrentSnapshots recorded {(_snapValid ? _snapRowsRecorded : 0)} " +
                "row(s), or a later forward invalidated them.");
        if (row == _snapRowsRecorded) return;   // the live state IS the state after the last row

        long tr0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var st = _defaultState;
        var snap = _snapGdn!;
        long stateBytes = (long)st.Gdn.GdnStateElements * sizeof(float), convBytes = (long)st.Gdn.ConvStateElements * sizeof(float);
        using (var ctx = _device.CreateSubmitContext())
        {
            ctx.Begin();
            nint cmd = ctx.CommandBuffer;
            KernelSupport.ComputeTransferFullBarrier(cmd);
            for (int l = 0; l < st.Gdn.NumGdnLayers; l++)
            {
                if (stateBytes > 0)
                    VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, snap.Gdn[l], st.Gdn.GetGdnStateBuffer(l), (ulong)(row * stateBytes), 0, (ulong)stateBytes);
                if (convBytes > 0)
                    VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, snap.Conv[l], st.Gdn.GetConvStateBuffer(l), (ulong)(row * convBytes), 0, (ulong)convBytes);
            }
            KernelSupport.ComputeTransferFullBarrier(cmd);
            ctx.SubmitAndWait();
        }
        st.Ple?.RestoreRow(row);
        st.Length = _snapBase + row + 1;
        _snapValid = false;
        SnapshotRestores++;
        _restoreTicks += System.Diagnostics.Stopwatch.GetTimestamp() - tr0;
    }

    private Q4GdnRowSnapshots EnsureRowSnapshots(int rows)
    {
        if (_snapGdn is { } cur && _snapCapacity >= rows) return cur;
        if (_spvDir is null) throw new NotSupportedException("No SPIR-V directory: row snapshots are unavailable.");
        _snapGdn?.Dispose();
        _snapGdn = null;
        _snapKernel ??= GdnScanMultiTokenSnapshotF32Kernel.Create(_device, _spvDir);
        var gdn = _defaultState.Gdn;
        int layers = gdn.NumGdnLayers;
        long sb = (long)gdn.GdnStateElements * sizeof(float), cb = (long)gdn.ConvStateElements * sizeof(float);
        var g = new VulkanDevice.Buffer[layers];
        var c = new VulkanDevice.Buffer[layers];
        for (int l = 0; l < layers; l++)
        {
            g[l] = _device.AllocateDeviceLocal(Math.Max(rows * sb, 4));
            c[l] = _device.AllocateDeviceLocal(Math.Max(rows * cb, 4));
        }
        _snapGdn = new Q4GdnRowSnapshots { Kernel = _snapKernel, Gdn = g, Conv = c };
        _snapCapacity = rows;
        _snapKernel.InvalidateDescriptorCache();   // freed handles can be recycled into the new buffers
        _core.Q4InvalidateCaches();
        return _snapGdn;
    }

    // ───────────────────────────── head graph ─────────────────────────────

    /// <summary>
    /// Front of <paramref name="n"/> head cells: <c>R' = eh_proj([enorm(embed(token)) ; hnorm(R_in)])</c> per stream, as
    /// <c>EhEmb x e</c> broadcast over the streams plus <c>EhHid x hnorm(R_in[s])</c>. <paramref name="rIn"/> and <paramref name="rOut"/> are
    /// <c>[n, hc * hidden]</c> device rows and must be distinct.
    /// </summary>
    private void RecordMtpFront(nint cmd, MtpHead h, ReadOnlySpan<int> tokens, VulkanDevice.Buffer rIn, VulkanDevice.Buffer rOut, int n)
    {
        int S = _streams, H = _hidden, row = S * H;
        var st = _core.Q4State;
        var k = _core.Q4Kernels;
        void Barrier() => KernelSupport.ComputeTransferFullBarrier(cmd);

        RecordEmbedding(cmd, tokens);
        Barrier();
        k.RmsNorm.Record(cmd, st.HiddenState, h.Enorm, h.E, rowCount: n, n: H, eps: _eps);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, rIn, h.HN, 0, 0, (ulong)((long)n * row * 4));
        Barrier();
        if (MtpHnormPerStream) _groupRms.Record(cmd, h.HN, h.Hnorm, n, S, H, _eps);
        else k.RmsNorm.Record(cmd, h.HN, h.Hnorm, h.HN, rowCount: n, n: row, eps: _eps);
        Barrier();
        _core.Q4RecordMatmul(cmd, h.EhEmb, h.EhEmbQt, h.E, h.A, outputDim: H, inputDim: H, seqLen: n);
        _core.Q4RecordMatmul(cmd, h.EhHid, h.EhHidQt, h.HN, h.B, outputDim: H, inputDim: H, seqLen: n * S);
        Barrier();
        _gr.RecordBroadcast(cmd, rOut, h.A, n, S, H);
        Barrier();
        k.Add.Record(cmd, rOut, h.B, rOut, n * row);
        Barrier();
    }

    /// <summary>One head draft step at <paramref name="position"/>; returns the argmax token, and the logits when <paramref name="logits"/> is given.</summary>
    private int RunDraftStep(VulkanQwen4ExpMtpState st, int tokenId, int position, Span<float> logits)
    {
        long td0 = System.Diagnostics.Stopwatch.GetTimestamp();
        var h = _mtp ?? throw new NotSupportedException("No MTP head is attached (SupportsMtp=false).");
        if (position < 1)
            throw new ArgumentOutOfRangeException(nameof(position), "An MTP cell pairs a token with the residual of the position before it: position must be >= 1.");
        if ((uint)tokenId >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(tokenId));
        int cell = position - 1;
        if (cell > st.Kv.CurrentLength)
            throw new InvalidOperationException($"MTP draft step at position {position} but the head holds {st.Kv.CurrentLength} cells (needs {cell}).");
        if (st.CurrentLength > position) st.Rollback(position);   // a smaller position discards the speculative steps beyond it

        int S = _streams, H = _hidden;
        ulong rowBytes = (ulong)((long)S * H * 4);
        Span<int> tok = [tokenId];
        Span<int> cpos = [cell];
        h.Core.Q4UploadPositions(cpos);
        var hst = h.Core.Q4State;
        var tst = _core.Q4State;
        var submit = _core.Q4Submit;
        submit.Begin();
        nint cmd = submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmd);
        void Barrier() => KernelSupport.ComputeTransferFullBarrier(cmd);

        st.RecordSeed(cmd);
        RecordMtpFront(cmd, h, tok, st.Pending, h.RWork, 1);

        RecordGrRead(cmd, h.AttnGr, h.RWork, hst.NormOutput, 1, inject: true);
        h.Core.Q4RecordAttention(cmd, h.AttnLayer, 1, cpos, st.Kv);
        Barrier();
        _gr.RecordWrite(cmd, h.RWork, hst.NormOutput, _gains, 1, S, H);
        Barrier();

        RecordGrRead(cmd, h.FfnGr, h.RWork, hst.NormOutput, 1, inject: true);
        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, hst.NormOutput, hst.MoeSharedInput, 0, 0, (ulong)H * 4);
        Barrier();
        h.Core.Q4RecordMoe(cmd, h.Moe, 1);
        Barrier();
        _gr.RecordWrite(cmd, h.RWork, hst.NormOutput, _gains, 1, S, H);
        Barrier();

        VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, h.RWork, st.Pending, 0, 0, rowBytes);   // the next draft step's input (before the head mixer)
        Barrier();
        RecordGrRead(cmd, h.HeadGr, h.RWork, tst.NormOutput, 1, inject: false);
        var w0 = _core.Q4Weights;
        _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, tst.NormOutput, tst.Logits,
            outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: 1);
        KernelSupport.ComputeToHostBarrier(cmd);
        submit.SubmitAndWait();
        st.MarkCovered(position);
        DraftSteps++;

        _mtpLogits ??= new float[_vocab];
        _device.Download(tst.Logits, _mtpLogits);
        if (!logits.IsEmpty) _mtpLogits.AsSpan().CopyTo(logits);
        int top = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)_mtpLogits);
        _draftTicks += System.Diagnostics.Stopwatch.GetTimestamp() - td0;

        // #822: the verify forward will be [last token, d1 .. dK]; its n-gram table rows depend only on those ids, and the prefix known so far
        // is enough to start paging them in while the remaining draft steps run (a later call wins; an abandoned request is harmless).
        if (_ple is not null && NgramPrefetch)
        {
            if (st.Run.Count == 0 || position != st.RunNextPosition) { st.Run.Clear(); st.RunStart = position; }
            st.Run.Add(tokenId);
            st.RunNextPosition = position + 1;
            if (_defaultState.Length == st.RunStart)
            {
                st.Run.Add(top);
                _ple.BeginPrefetch(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(st.Run), _defaultState.Ple!);
                st.Run.RemoveAt(st.Run.Count - 1);
            }
        }
        return top;
    }

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var st = RequireMtpState(state) ?? throw new ArgumentNullException(nameof(state));
        var result = UnmanagedTensor.Allocate(new TensorShape(1, _vocab), DType.Float32, deviceId: -1);
        try
        {
            RunDraftStep(st, tokenId, position, new Span<float>((void*)result.DataPointer, _vocab));
            return result;
        }
        catch { result.Dispose(); throw; }
    }

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _mtp is not null;

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
    {
        var st = RequireMtpState(state) ?? throw new ArgumentNullException(nameof(state));
        return RunDraftStep(st, tokenId, position, default);
    }

    /// <summary>
    /// Feeds a trunk batch to the head after the trunk forward: keeps the last trunk residual rows for seeding, then writes the K/V cells of
    /// every pair <c>(R_{p-1}, token_p)</c> (K/V only: an absorbed cell's own output is discarded, so its MoE and mixer are never run) and
    /// queues the seed from the batch's last row. Reads the residual rows the forward left in <c>_res</c>.
    /// </summary>
    private void AbsorbBatch(VulkanQwen4ExpMtpState mtp, ReadOnlySpan<int> tokenIds, int firstPosition)
    {
        var h = _mtp!;
        int T = tokenIds.Length, S = _streams, H = _hidden;
        mtp.BeginAbsorb(firstPosition, T);
        ulong rowBytes = (ulong)((long)S * H * 4);
        int skip = firstPosition == 0 ? 1 : 0;   // token 0 has no preceding residual and owns no cell
        int cap = Math.Min(T, VulkanQwen4ExpMtpState.CaptureRows);
        var hst = h.Core.Q4State;
        var submit = _core.Q4Submit;
        bool pre = true;
        int[] cposBuf = new int[AbsorbChunk];

        void Preamble(nint cmd)
        {
            mtp.RecordSeed(cmd);   // resolve the queued seed (old captured rows -> pending/carry) before the rows are overwritten
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, mtp.Captured, (ulong)(T - cap) * rowBytes, 0, (ulong)cap * rowBytes);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            mtp.CapturedFirst = T - cap;
            mtp.CapturedTotal = T;
            pre = false;
        }

        for (int c0 = skip; c0 < T; c0 += AbsorbChunk)
        {
            int m = Math.Min(AbsorbChunk, T - c0);
            Span<int> cpos = cposBuf.AsSpan(0, m);
            for (int i = 0; i < m; i++) cpos[i] = firstPosition + c0 - 1 + i;   // the cell of token j sits at position firstPosition + j - 1
            h.Core.Q4UploadPositions(cpos);
            submit.Begin();
            nint cmd = submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
            if (pre) Preamble(cmd);
            if (c0 == 0)
            {
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, mtp.Carry, h.Pair, 0, 0, rowBytes);
                if (m > 1) VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, h.Pair, 0, rowBytes, (ulong)(m - 1) * rowBytes);
            }
            else
                VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, h.Pair, (ulong)(c0 - 1) * rowBytes, 0, (ulong)m * rowBytes);
            KernelSupport.ComputeTransferFullBarrier(cmd);
            RecordMtpFront(cmd, h, tokenIds.Slice(c0, m), h.Pair, h.RFront, m);
            RecordGrRead(cmd, h.AttnGr, h.RFront, hst.NormOutput, m, inject: false);
            h.Core.Q4RecordAttention(cmd, h.AttnLayer, m, cpos, mtp.Kv);
            KernelSupport.ComputeToHostBarrier(cmd);
            submit.SubmitAndWait();
            AbsorbedCells += m;
        }
        if (pre)
        {
            submit.Begin();
            nint cmd = submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
            Preamble(cmd);
            KernelSupport.ComputeToHostBarrier(cmd);
            submit.SubmitAndWait();
        }
        mtp.SetCovered(firstPosition + T);
        mtp.SeedFromCapturedRow(T - 1);
    }

    // ───────────────────────────── diagnostics ─────────────────────────────

    /// <summary>
    /// Go/no-go diagnostic for the Vulkan MTP head (#820): wall ms of one draft-step proxy. The head block is a QSA + 512-expert MoE block
    /// on the gated residual plus a head mixer and the trunk's LM head; the trunk's FIRST QSA layer has exactly that geometry, so one
    /// 1-row recording of <c>GR-read, QSA, GR-write, GR-read, MoE, GR-write, head mixer, LM head</c> (one command buffer, one submit, logits
    /// downloaded and arg-maxed on the host) measures the draft step without the head weights. Omitted: <c>eh_proj</c> (a ~52 MB Q8_0 GEMV)
    /// and the head's Q8_0 expert banks (trunk banks are Q4_K).
    /// </summary>
    /// <param name="state">A state that has just run a forward (its last residual row seeds the proxy; its KV is read, not advanced).</param>
    /// <param name="reps">Steps to time.</param>
    /// <returns>Per-step wall ms.</returns>
    public double[] ProbeDraftStepMs(VulkanQwen4ExpSequenceState state, int reps)
    {
        int fq = -1;
        for (int il = 0; il < Config.NumLayers; il++)
            if (Config.HybridLayout!.LayerKind[il] != HybridLayerKind.GatedDeltaNet) { fq = il; break; }
        if (fq < 0) throw new InvalidOperationException("No QSA layer.");
        var kv = state.OwnKv;
        var st = _core.Q4State;
        int S = _streams, H = _hidden;
        int pos = state.Length;
        Span<int> positions = [pos];
        var submit = _core.Q4Submit;
        var w0 = _core.Q4Weights;
        var times = new double[reps];
        float[] logits = new float[_vocab];
        for (int r = 0; r < reps; r++)
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            if (kv.CurrentLength > pos) kv.Rollback(pos);
            _core.Q4UploadPositions(positions);
            submit.Begin();
            nint cmd = submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmd);
            void Barrier() => KernelSupport.ComputeTransferFullBarrier(cmd);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, _res, _headRes, (ulong)(((long)Math.Min(pos, _scratchCapacity) - 1) * S * H * 4), 0, (ulong)((long)S * H * 4));
            Barrier();
            RecordGrRead(cmd, _attnGr[fq], _headRes, st.NormOutput, 1, inject: true);
            _core.Q4RecordAttention(cmd, fq, 1, positions, kv);
            Barrier();
            _gr.RecordWrite(cmd, _headRes, st.NormOutput, _gains, 1, S, H);
            Barrier();
            RecordGrRead(cmd, _ffnGr[fq], _headRes, st.NormOutput, 1, inject: true);
            VulkanQwen3MoeHybridTransformerModel.Q4Copy(cmd, st.NormOutput, st.MoeSharedInput, 0, 0, (ulong)H * 4);
            Barrier();
            _core.Q4RecordMoe(cmd, _moe[fq], 1);
            Barrier();
            _gr.RecordWrite(cmd, _headRes, st.NormOutput, _gains, 1, S, H);
            Barrier();
            RecordGrRead(cmd, _headGr, _headRes, st.NormOutput, 1, inject: false);
            _core.Q4RecordMatmul(cmd, w0.OutputWeight, w0.OutputDeviceQuantType, st.NormOutput, st.Logits,
                outputDim: w0.OutputOutputDim, inputDim: w0.OutputInputDim, seqLen: 1);
            KernelSupport.ComputeToHostBarrier(cmd);
            submit.SubmitAndWait();
            _device.Download(st.Logits, logits.AsSpan());
            int top = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logits);
            if (top < 0) throw new InvalidOperationException();
            times[r] = System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        }
        if (kv.CurrentLength > pos) kv.Rollback(pos);
        return times;
    }
}
