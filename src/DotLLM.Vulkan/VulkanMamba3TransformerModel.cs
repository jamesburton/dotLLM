using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan.Interop;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// End-to-end Vulkan forward pass for the Mamba-3 architecture (SISO and MIMO). Mirrors
/// the CPU oracle <see cref="Mamba3Block"/> step-for-step:
/// <c>embed → N × (norm + in_proj + per-token prep + data-RoPE + SSD scan + out_proj +
/// residual) → norm_f → lm_head</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>SISO and MIMO.</b> The orchestrator branches on <see cref="Mamba3Config.IsMimo"/> per
/// layer: SISO routes through <see cref="Mamba3CanonicalSsdSisoF32Kernel"/> with pairwise-
/// rotated B/C, MIMO routes through <see cref="Mamba3CanonicalSsdMimoF32Kernel"/> with
/// rank-expanded B/C and halved-rotation RoPE plus the per-rank <c>mimo_z</c>/<c>mimo_o</c>
/// expansion / contraction weights. State plumbing (ssm_state, cum_angle) is identical
/// between the two paths — the rank axis lives entirely inside the per-token scratch.
/// </para>
/// <para>
/// <b>F32-only.</b> Every weight, every activation, every scratch is F32. Quantised ingest
/// is a follow-up — the CPU loader currently rejects non-F32 source dtypes too, so the
/// Vulkan side has no asymmetry to honour.
/// </para>
/// <para>
/// <b>Per-token preprocessing on the host.</b> The Mamba-3 SISO block needs softplus,
/// sigmoid, per-slice RMSNorm, group→head broadcast, bias add, qk_pre_dot, and a
/// shifted_γ-aware scale computation between the in_proj GEMM and the data-RoPE / SSD
/// scan. None of these have a dedicated Vulkan compute shader yet, and writing a fused
/// "scan_prep" shader is mechanically straightforward but not load-bearing for parity.
/// This first cut runs the per-token block host-side: download the in_proj output,
/// compute every prep table in C# (mirroring <see cref="Mamba3Block"/>.Forward step-for-step),
/// and upload the resulting tables before dispatching the data-RoPE + SSD scan. The
/// downloaded buffer is host-visible, so the round trip is one map+memcpy per layer
/// rather than a true PCIe transfer on UMA. Fused prep shaders are a perf follow-up;
/// correctness is bit-equal to the CPU oracle modulo F32 reduction-order noise in the
/// in_proj / out_proj matmuls.
/// </para>
/// <para>
/// <b>State threading.</b> Two persistent buffers per layer cross call boundaries:
/// <c>ssm_state</c> ([H, P, N], threaded into the SISO scan kernel) and <c>cum_angle</c>
/// ([H, S], threaded into the data-RoPE kernel via <c>hasCumPrev=writeCumOut=true</c>).
/// Both are owned by <see cref="VulkanMamba3State"/> and zero-initialised at construction;
/// the orchestrator never resets them between layers (each layer has its own state row).
/// The CPU oracle's <c>k_state</c> / <c>v_state</c> chunk-boundary buffers are NOT mirrored
/// — the Vulkan SISO path treats each Forward as one chunk and relies on the canonical
/// <c>shifted_γ[T-1] = 0</c> boundary inside the SSD scan. Cross-chunk parity is a follow-up
/// if streaming-decode parity becomes a regression bar.
/// </para>
/// </remarks>
public sealed class VulkanMamba3TransformerModel : IModel
{
    private readonly VulkanDevice _device;
    private readonly VulkanMamba3Weights _weights;
    private readonly VulkanMamba3ForwardScratch _state;
    private readonly VulkanMamba3State _recurrent;
    private readonly Mamba3Config _m3;
    private readonly bool _ownsDevice;

    // Kernels.
    private readonly MatMulF32Kernel _matmul;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly Mamba3DataRopeF32Kernel _dataRope;
    private readonly Mamba3CanonicalSsdSisoF32Kernel _sisoScan;
    // MIMO scan kernel — only created when Mamba3Config.IsMimo is true. Lazy creation
    // keeps the SPV file dependency optional for SISO-only deployments.
    private readonly Mamba3CanonicalSsdMimoF32Kernel? _mimoScan;
    private readonly AddKernel _add;

    private readonly VulkanDevice.SubmitContext _submit;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _state.AllocatedBytes + _weights.AllocatedBytes + _recurrent.AllocatedBytes;

    private VulkanMamba3TransformerModel(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config,
        VulkanMamba3Weights weights,
        VulkanMamba3ForwardScratch state,
        VulkanMamba3State recurrent,
        MatMulF32Kernel matmul, RmsNormF32Kernel rmsnorm,
        Mamba3DataRopeF32Kernel dataRope, Mamba3CanonicalSsdSisoF32Kernel sisoScan,
        Mamba3CanonicalSsdMimoF32Kernel? mimoScan,
        AddKernel add,
        VulkanDevice.SubmitContext submit)
    {
        _device = device;
        _ownsDevice = ownsDevice;
        Config = config;
        _weights = weights;
        _state = state;
        _recurrent = recurrent;
        _m3 = config.Mamba3Config!;

        _matmul = matmul;
        _rmsnorm = rmsnorm;
        _dataRope = dataRope;
        _sisoScan = sisoScan;
        _mimoScan = mimoScan;
        _add = add;
        _submit = submit;
    }

    /// <summary>
    /// Loads a Mamba-3 model (SISO or MIMO) from an opened safetensors file onto the
    /// Vulkan device. The <paramref name="file"/> must remain alive for the lifetime of
    /// the returned model.
    /// </summary>
    /// <param name="file">An opened safetensors file with Mamba-3 SISO/MIMO tensors.</param>
    /// <param name="config">Model config; <see cref="ModelConfig.Mamba3Config"/> must be populated.</param>
    /// <param name="spvDir">Directory containing the compiled SPIR-V kernel blobs.</param>
    /// <returns>A ready-to-forward Vulkan Mamba-3 model.</returns>
    public static VulkanMamba3TransformerModel LoadFromSafetensors(
        ISafetensorsTensorSource file, ModelConfig config, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(spvDir);
        if (config.Architecture != Architecture.Mamba3)
            throw new ArgumentException(
                $"VulkanMamba3TransformerModel requires Architecture.Mamba3, got {config.Architecture}.",
                nameof(config));

        Mamba3Weights cpuWeights = Mamba3WeightLoader.Load(config, file);
        try
        {
            var device = VulkanDevice.Create();
            try
            {
                return BuildInternal(device, ownsDevice: true, config, cpuWeights, spvDir);
            }
            catch
            {
                device.Dispose();
                throw;
            }
        }
        catch
        {
            cpuWeights.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Builds a Vulkan Mamba-3 model (SISO or MIMO) from a caller-owned device +
    /// already-loaded <see cref="Mamba3Weights"/>. The caller retains ownership of
    /// <paramref name="device"/>; the returned model does NOT dispose it.
    /// </summary>
    public static VulkanMamba3TransformerModel BuildOnDevice(
        VulkanDevice device, ModelConfig config, Mamba3Weights weights, string spvDir)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(spvDir);
        if (config.Architecture != Architecture.Mamba3)
            throw new ArgumentException(
                $"VulkanMamba3TransformerModel requires Architecture.Mamba3, got {config.Architecture}.",
                nameof(config));
        return BuildInternal(device, ownsDevice: false, config, weights, spvDir);
    }

    private static VulkanMamba3TransformerModel BuildInternal(
        VulkanDevice device, bool ownsDevice,
        ModelConfig config, Mamba3Weights cpuWeights, string spvDir)
    {
        var weights = VulkanMamba3Weights.Upload(device, config, cpuWeights);
        // Once uploaded to the device the CPU-side bundle is no longer needed, but the
        // caller owns its lifetime via the LoadFromSafetensors / BuildOnDevice contract:
        // we don't dispose it here. The CPU mmap anchor (the safetensors file itself)
        // is held by the caller separately.

        var state = new VulkanMamba3ForwardScratch(device, config, initialSeqLen: 1);
        var recurrent = new VulkanMamba3State(device, config);

        var matmul = MatMulF32Kernel.Create(device, spvDir);
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var dataRope = Mamba3DataRopeF32Kernel.Create(device, spvDir);
        var sisoScan = Mamba3CanonicalSsdSisoF32Kernel.Create(device, spvDir);
        Mamba3CanonicalSsdMimoF32Kernel? mimoScan = null;
        if (config.Mamba3Config!.IsMimo)
            mimoScan = Mamba3CanonicalSsdMimoF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);

        var submit = device.CreateSubmitContext();

        return new VulkanMamba3TransformerModel(
            device, ownsDevice,
            config, weights, state, recurrent,
            matmul, rmsnorm, dataRope, sisoScan, mimoScan, add,
            submit);
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
    {
        _ = kvCache; // Mamba-3 uses SSM state, not KV cache.

        if (tokenIds.Length != positions.Length)
            throw new ArgumentException("tokenIds and positions must have the same length.");
        int seqLen = tokenIds.Length;
        if (seqLen == 0) throw new ArgumentException("tokenIds must be non-empty.", nameof(tokenIds));

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numLayers = Config.NumLayers;
        int nHead = _m3.NumHeads;
        int headDim = _m3.HeadDim;
        int dState = _m3.StateSize;
        int dInner = _m3.DInner;
        int numBcHeads = _m3.NumGroups;
        int numRopeAngles = _m3.NumRopeAngles;
        int dInProj = _m3.InputProjectionDim;
        bool isMimo = _m3.IsMimo;
        int mimoRank = isMimo ? _m3.MimoRank : 1;
        float aFloor = _m3.AFloor;
        float eps = Config.NormEpsilon;

        bool resized = _state.EnsureCapacity(seqLen);
        if (resized) InvalidateKernelCaches();

        ValidateTokenIds(tokenIds);

        // 1. EMBEDDING GATHER (CPU records vkCmdCopyBuffer per token directly into
        //    HiddenState[t, :]). This is one submit on its own — keeps the host->device
        //    barriers simple.
        _submit.Begin();
        nint cmdBuf = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(cmdBuf);
        _state.ResetHiddenSlot();
        RecordEmbeddingGather(cmdBuf, tokenIds);
        KernelSupport.TransferToComputeBarrier(cmdBuf);
        _submit.SubmitAndWait();

        // Host-side scratch arrays for the per-token preprocessing — sized to seqLen and
        // reused across layers. Pooled here rather than in VulkanMamba3ForwardScratch
        // because they never cross the device boundary. Rank-aware sizing for B/C: SISO
        // collapses to T·H·N, MIMO uses the canonical T·R·H·N layout the MIMO scan kernel
        // expects (qkPreDotHost is shared — it stores the already-rank-summed Σ_r dot in
        // MIMO mode, matching the kernel's qkPreDotSum binding).
        float[] projHost = new float[seqLen * dInProj];
        float[] xHost = new float[seqLen * dInner];
        float[] zHost = new float[seqLen * dInner];
        float[] dtHost = new float[seqLen * nHead];
        float[] adtHost = new float[seqLen * nHead];
        float[] trapHost = new float[seqLen * nHead];
        float[] gammaHost = new float[seqLen * nHead];
        float[] scaleHost = new float[seqLen * nHead];
        float[] anglesRawHost = new float[seqLen * numRopeAngles];
        float[] bHost = new float[seqLen * mimoRank * nHead * dState];
        float[] cHost = new float[seqLen * mimoRank * nHead * dState];
        float[] qkPreDotHost = new float[seqLen * nHead];

        // 2. LAYERS — one Mamba3 SISO block per layer.
        for (int layer = 0; layer < numLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // ── 2a. PRE-NORM + IN_PROJ (single submit; everything device-side) ────
            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;

            _rmsnorm.Record(cmdBuf, _state.HiddenState, lw.Norm, _state.NormOutput,
                rowCount: seqLen, n: hiddenSize, eps: eps);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            _matmul.Record(cmdBuf, lw.InProj, _state.NormOutput, _state.Proj,
                m: lw.InProjOutputDim, k: lw.InProjInputDim, n: seqLen);

            // The matmul writes Proj (which is host-visible). We need a compute → host
            // barrier so the host map below sees the kernel's writes, then SubmitAndWait
            // to actually flush the queue.
            KernelSupport.ComputeToHostBarrier(cmdBuf);
            _submit.SubmitAndWait();

            // ── 2b. HOST PREP (mirrors Mamba3Block.Forward / ForwardMimo steps 2–4) ──
            _device.Download(_state.Proj, projHost.AsSpan(0, seqLen * dInProj));
            ComputeHostTables(
                projHost, lw, seqLen, dInProj, dInner, nHead, dState, headDim,
                numBcHeads, numRopeAngles, mimoRank, aFloor, eps,
                xHost, zHost, dtHost, adtHost, trapHost, gammaHost,
                anglesRawHost, bHost, cHost, qkPreDotHost, scaleHost);

            // Upload prepared tables. Each is a host-visible buffer, so this is a
            // map+memcpy. B / C are sized for [seqLen, R, H, N] in MIMO and [seqLen, H, N]
            // in SISO — same backing scratch, the rank slot collapses to 1 when SISO.
            int bcElems = seqLen * mimoRank * nHead * dState;
            _device.Upload(xHost.AsSpan(0, seqLen * dInner), _state.X);
            _device.Upload(zHost.AsSpan(0, seqLen * dInner), _state.Z);
            _device.Upload(dtHost.AsSpan(0, seqLen * nHead), _state.Dt);
            _device.Upload(adtHost.AsSpan(0, seqLen * nHead), _state.Adt);
            _device.Upload(gammaHost.AsSpan(0, seqLen * nHead), _state.Gamma);
            _device.Upload(scaleHost.AsSpan(0, seqLen * nHead), _state.Scale);
            _device.Upload(anglesRawHost.AsSpan(0, seqLen * numRopeAngles), _state.AnglesRaw);
            _device.Upload(bHost.AsSpan(0, bcElems), _state.B);
            _device.Upload(cHost.AsSpan(0, bcElems), _state.C);
            _device.Upload(qkPreDotHost.AsSpan(0, seqLen * nHead), _state.QkPreDot);

            // ── 2c. DATA-ROPE + SSD SCAN + OUT_PROJ + RESIDUAL (single submit) ───
            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmdBuf);

            VulkanDevice.Buffer cumAngle = _recurrent.GetCumAngleBuffer(layer);
            VulkanDevice.Buffer ssmState = _recurrent.GetSsmStateBuffer(layer);

            // data-RoPE: B and C are mutated in place (post-RoPE). cum_angle is read at
            // entry (hasCumPrev=true, even on first call — buffer is zero-initialised) and
            // written back at exit (writeCumOut=true) so subsequent decode chunks resume.
            // SISO uses pairwise rotation; MIMO uses halved-split rotation per the
            // canonical mamba3_mimo_fwd reference.
            _dataRope.Record(cmdBuf,
                b: _state.B, c: _state.C,
                anglesRaw: _state.AnglesRaw, dt: _state.Dt,
                cumPrev: cumAngle, cumOut: cumAngle,
                seqLen: seqLen, nRank: mimoRank, nHead: nHead, dState: dState,
                numRopeAngles: numRopeAngles,
                mode: isMimo ? Mamba3RopeMode.Halved : Mamba3RopeMode.Pairwise,
                hasCumPrev: true, writeCumOut: true);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            if (isMimo)
            {
                // SSD MIMO scan. qRoped = post-RoPE C, kRoped = post-RoPE B (matches the
                // CPU oracle's argument order to ExecuteMimo). qkPreDotHost is the
                // canonical Σ_r form, prepared host-side. mimo_z / mimo_o are uploaded
                // per-layer and bound here. Equivalent to CPU's ExecuteMimoStreaming with
                // empty kState/vState — the Vulkan path threads ssm_state and cum_angle
                // across calls but does not maintain k_state / v_state chunk-boundary
                // buffers (matches the SISO Vulkan path; one-shot semantics).
                Mamba3CanonicalSsdMimoF32Kernel mimo = _mimoScan
                    ?? throw new InvalidOperationException(
                        "MIMO scan kernel not initialised — Mamba3Config.IsMimo must be set when constructing the model.");
                if (lw.MimoZ is null || lw.MimoO is null)
                    throw new InvalidOperationException(
                        "MIMO layer is missing mimo_z/mimo_o device buffers — check VulkanMamba3Weights.Upload.");
                mimo.Record(cmdBuf,
                    state: ssmState,
                    v: _state.X,
                    qRoped: _state.C,
                    kRoped: _state.B,
                    qkPreDotSum: _state.QkPreDot,
                    scale: _state.Scale,
                    gamma: _state.Gamma,
                    adt: _state.Adt,
                    d: lw.D,
                    z: _state.Z,
                    mimoZ: lw.MimoZ,
                    mimoO: lw.MimoO,
                    y: _state.YScan,
                    seqLen: seqLen, nRank: mimoRank, nHead: nHead, headDim: headDim,
                    dState: dState, hasZ: true);
            }
            else
            {
                // SSD SISO scan. qRoped = post-RoPE C, kRoped = post-RoPE B (matches the
                // CPU oracle's argument order to ExecuteSiso).
                _sisoScan.Record(cmdBuf,
                    state: ssmState,
                    v: _state.X,
                    qRoped: _state.C,
                    kRoped: _state.B,
                    qkPreDot: _state.QkPreDot,
                    scale: _state.Scale,
                    gamma: _state.Gamma,
                    adt: _state.Adt,
                    d: lw.D,
                    z: _state.Z,
                    y: _state.YScan,
                    seqLen: seqLen, nHead: nHead, headDim: headDim, dState: dState,
                    hasZ: true);
            }
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // out_proj: YScan @ OutProj^T → BlockOut.
            _matmul.Record(cmdBuf, lw.OutProj, _state.YScan, _state.BlockOut,
                m: lw.OutProjOutputDim, k: lw.OutProjInputDim, n: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Residual add: NewHidden = OldHidden (Residual) + BlockOut, written into
            // AddScratch, then rotate the slot.
            _add.Record(cmdBuf, _state.Residual, _state.BlockOut, _state.AddScratch,
                seqLen * hiddenSize);

            _submit.SubmitAndWait();
            _state.RotateHiddenSlot();
        }

        // 3. FINAL RMSNORM + LM HEAD on last token only.
        _submit.Begin();
        cmdBuf = _submit.CommandBuffer;

        long rowBytes = (long)hiddenSize * sizeof(float);
        long lastRowOffset = (long)(seqLen - 1) * rowBytes;
        RecordCopyBufferRange(cmdBuf, _state.HiddenState, _state.LastTokenHidden,
            srcOffset: (ulong)lastRowOffset, dstOffset: 0, size: (ulong)rowBytes);
        KernelSupport.TransferToComputeBarrier(cmdBuf);

        _rmsnorm.Record(cmdBuf, _state.LastTokenHidden, _weights.FinalNormWeight, _state.LastTokenHidden,
            rowCount: 1, n: hiddenSize, eps: eps);
        KernelSupport.ComputeToComputeBarrier(cmdBuf);

        _matmul.Record(cmdBuf, _weights.LmHead, _state.LastTokenHidden, _state.Logits,
            m: _weights.LmHeadOutputDim, k: _weights.LmHeadInputDim, n: 1);

        KernelSupport.ComputeToHostBarrier(cmdBuf);
        _submit.SubmitAndWait();

        var shape = new TensorShape(1, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId: -1);
        try
        {
            unsafe
            {
                var dest = new Span<float>((void*)result.DataPointer, vocabSize);
                _device.Download(_state.Logits, dest);
            }
        }
        catch
        {
            result.Dispose();
            throw;
        }
        return result;
    }

    /// <summary>
    /// Runs the per-token preprocessing block on the host: split projection, compute
    /// DT/ADT/trap/gamma, RMSNorm B/C, broadcast G→H, add bias, qk_pre_dot, scale.
    /// Mirrors steps 2–4 of <see cref="Mamba3Block"/>.Forward (SISO,
    /// <paramref name="mimoRank"/>=1) or <see cref="Mamba3Block"/>.ForwardMimo
    /// (MIMO, <paramref name="mimoRank"/>>=2) step-for-step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Layout differences SISO ↔ MIMO.</b>
    /// </para>
    /// <list type="bullet">
    ///   <item><c>B_raw</c>/<c>C_raw</c> per-token slot is <c>[R, G, N]</c> row-major
    ///   (R=1 collapses to <c>[G, N]</c> for SISO).</item>
    ///   <item><c>B_bias</c>/<c>C_bias</c> are <c>[H, R, N]</c> row-major in MIMO;
    ///   for SISO the loader keeps the HF <c>[H, 1, N]</c> shape with R=1.</item>
    ///   <item>Output <c>b</c>/<c>c</c> are written as <c>[T, R, H, N]</c> for MIMO,
    ///   <c>[T, H, N]</c> for SISO. Dispatchers downstream slice <c>seqLen · R · H · N</c>
    ///   elements from the rank-aware scratch.</item>
    ///   <item><c>qkPreDot</c> stores the rank-summed <c>Σ_r Σ_n (C_biased · B_biased)</c>
    ///   for MIMO; SISO has no rank to sum so it simplifies to the SISO dot.</item>
    /// </list>
    /// </remarks>
    [SkipLocalsInit]
    private static void ComputeHostTables(
        ReadOnlySpan<float> proj,
        VulkanMamba3Weights.LayerBuffers lw,
        int seqLen, int dInProj, int dInner, int nHead, int dState, int headDim,
        int numBcHeads, int numRopeAngles, int mimoRank,
        float aFloor, float eps,
        Span<float> x, Span<float> z, Span<float> dt, Span<float> adt,
        Span<float> trap, Span<float> gamma,
        Span<float> anglesRaw, Span<float> b, Span<float> c, Span<float> qkPreDot,
        Span<float> scale)
    {
        // Norm weights and biases live on the device — but we need them host-side for the
        // per-token RMSNorm+bias step. Map and read them once per layer; the buffers are
        // device-local so we can't map directly, but they're tiny (d_state and
        // n_head*max(R,1)*d_state) — keep them on a host-readable mirror inside
        // VulkanMamba3Weights and read them straight here.
        ReadOnlySpan<float> bNormW = lw.BNormHost;
        ReadOnlySpan<float> cNormW = lw.CNormHost;
        ReadOnlySpan<float> bBias = lw.BBiasHost;
        ReadOnlySpan<float> cBias = lw.CBiasHost;
        ReadOnlySpan<float> dtBias = lw.DtBiasHost;

        int R = mimoRank;
        int bcPerToken = dState * numBcHeads * R;
        int ofsZ = 0;
        int ofsX = dInner;
        int ofsB = 2 * dInner;
        int ofsC = ofsB + bcPerToken;
        int ofsDdDt = ofsC + bcPerToken;
        int ofsDdA = ofsDdDt + nHead;
        int ofsTrap = ofsDdA + nHead;
        int ofsAngles = ofsTrap + nHead;

        int headsPerGroup = nHead / numBcHeads;

        for (int t = 0; t < seqLen; t++)
        {
            int src = t * dInProj;

            // z, x copies for the current token.
            proj.Slice(src + ofsZ, dInner).CopyTo(z.Slice(t * dInner, dInner));
            proj.Slice(src + ofsX, dInner).CopyTo(x.Slice(t * dInner, dInner));

            for (int h = 0; h < nHead; h++)
            {
                float ddDt = proj[src + ofsDdDt + h];
                float ddA = proj[src + ofsDdA + h];
                float trp = proj[src + ofsTrap + h];

                float dtv = SoftPlus(ddDt + dtBias[h]);
                float aVal = -SoftPlus(ddA);
                if (aVal > -aFloor) aVal = -aFloor;

                dt[t * nHead + h] = dtv;
                adt[t * nHead + h] = aVal * dtv;
                float tv = Sigmoid(trp);
                trap[t * nHead + h] = tv;
                gamma[t * nHead + h] = dtv * tv;
            }

            // angles_raw[t, :] (shared across heads).
            proj.Slice(src + ofsAngles, numRopeAngles)
                .CopyTo(anglesRaw.Slice(t * numRopeAngles, numRopeAngles));

            // B/C per-(R, G, N) slice RMSNorm + broadcast G→H + per-(H, R, N) bias add.
            // For SISO (R=1) this collapses to the original [H, 1, N] layout. For MIMO
            // the destination layout is [T, R, H, N] row-major (matches the canonical
            // SSD MIMO scan kernel binding).
            for (int r = 0; r < R; r++)
            {
                for (int g = 0; g < numBcHeads; g++)
                {
                    int bSrcBase = src + ofsB + (r * numBcHeads + g) * dState;
                    int cSrcBase = src + ofsC + (r * numBcHeads + g) * dState;
                    RmsNormFactor(proj.Slice(bSrcBase, dState), eps, out float bInvRms);
                    RmsNormFactor(proj.Slice(cSrcBase, dState), eps, out float cInvRms);

                    for (int hInGroup = 0; hInGroup < headsPerGroup; hInGroup++)
                    {
                        int h = g * headsPerGroup + hInGroup;
                        // Bias indexing: SISO uses [H, 1, N] (R=1, so equivalent to
                        // [H, G=1, N] via g==0), MIMO uses [H, R, N]. Both collapse to a
                        // simple flat [h * R + r] * dState base.
                        int biasBase = (h * R + r) * dState;
                        int dstBase = ((t * R + r) * nHead + h) * dState;

                        for (int n = 0; n < dState; n++)
                        {
                            float bv = proj[bSrcBase + n] * bInvRms * bNormW[n] + bBias[biasBase + n];
                            float cv = proj[cSrcBase + n] * cInvRms * cNormW[n] + cBias[biasBase + n];
                            b[dstBase + n] = bv;
                            c[dstBase + n] = cv;
                        }
                    }
                }
            }
        }

        // qk_pre_dot[t, h] = Σ_r Σ_n (C_biased · B_biased) — rank-summed in MIMO,
        // single-rank in SISO. Same as CPU oracle step 3.
        if (R == 1)
        {
            for (int t = 0; t < seqLen; t++)
            {
                int baseT = t * nHead * dState;
                for (int h = 0; h < nHead; h++)
                {
                    ReadOnlySpan<float> bh = b.Slice(baseT + h * dState, dState);
                    ReadOnlySpan<float> ch = c.Slice(baseT + h * dState, dState);
                    qkPreDot[t * nHead + h] = TensorPrimitives.Dot(ch, bh);
                }
            }
        }
        else
        {
            // MIMO: layout [T, R, H, N] — sum across the rank axis.
            for (int t = 0; t < seqLen; t++)
            {
                for (int h = 0; h < nHead; h++)
                {
                    float sum = 0f;
                    for (int r = 0; r < R; r++)
                    {
                        int baseIdx = ((t * R + r) * nHead + h) * dState;
                        ReadOnlySpan<float> bh = b.Slice(baseIdx, dState);
                        ReadOnlySpan<float> ch = c.Slice(baseIdx, dState);
                        sum += TensorPrimitives.Dot(ch, bh);
                    }
                    qkPreDot[t * nHead + h] = sum;
                }
            }
        }

        // scale[t, h] = γ[t, h] + shifted_γ[t, h]; shifted_γ[T-1, h] = 0.
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float sh = 0f;
                if (t + 1 < seqLen)
                {
                    int next = (t + 1) * nHead + h;
                    sh = dt[next] * (1f - trap[next]);
                }
                scale[t * nHead + h] = gamma[t * nHead + h] + sh;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RmsNormFactor(ReadOnlySpan<float> slice, float eps, out float invRms)
    {
        // F32 accumulator — matches CPU oracle's Mamba3Block.RmsNormInto (uses double
        // internally for parity with canonical rms_norm_ref's upcast semantics).
        double acc = 0.0;
        for (int i = 0; i < slice.Length; i++)
        {
            double v = slice[i];
            acc += v * v;
        }
        float mean = (float)(acc / slice.Length);
        invRms = 1f / MathF.Sqrt(mean + eps);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float SoftPlus(float x)
    {
        if (x > 20f) return x;
        if (x < -20f) return MathF.Exp(x);
        return MathF.Log(1f + MathF.Exp(x));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

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

    private static void RecordCopyBufferRange(
        nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst,
        ulong srcOffset, ulong dstOffset, ulong size)
    {
        var region = new VkBufferCopy { srcOffset = srcOffset, dstOffset = dstOffset, size = size };
        VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    private void InvalidateKernelCaches()
    {
        _matmul.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _dataRope.InvalidateDescriptorCache();
        _sisoScan.InvalidateDescriptorCache();
        _mimoScan?.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _recurrent.Dispose();

        _add.Dispose();
        _mimoScan?.Dispose();
        _sisoScan.Dispose();
        _dataRope.Dispose();
        _rmsnorm.Dispose();
        _matmul.Dispose();

        if (_ownsDevice)
            _device.Dispose();
    }
}
