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
using QuantizationType = DotLLM.Core.Configuration.QuantizationType;

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
/// <b>Activations stay F32; weights honour the Q8_0 overlay.</b> Every activation and scratch
/// buffer is F32. The matmul-target projections (<c>in_proj</c>, <c>out_proj</c>, <c>lm_head</c>)
/// honour the optional Q8_0 overlay on <see cref="Mamba3Weights"/> at upload time: when set,
/// the raw Q8_0 blocks are kept on device and dispatched through
/// <see cref="MatMulQ8_0Kernel"/> (decode GEMV) /
/// <see cref="MatMulQ8_0GemmKernel"/> / <see cref="MatMulQ8_0GemmCoopmatKernel"/> (prefill GEMM)
/// — same routing as <see cref="VulkanTransformerModel"/> /
/// <see cref="VulkanNemotronHTransformerModel"/>. Production load paths never set the overlay
/// (<see cref="Mamba3WeightLoader"/> is F32-only), so existing F32 forwards remain bit-equal.
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
/// <b>State threading.</b> Four persistent buffers per layer cross call boundaries:
/// <c>ssm_state</c> ([H, P, N], threaded into the SISO/MIMO scan kernel), <c>cum_angle</c>
/// ([H, S], threaded into the data-RoPE kernel via <c>hasCumPrev=writeCumOut=true</c>),
/// and the streaming-chunk boundary buffers <c>k_state</c> ([H, N] for SISO, [R, H, N] for
/// MIMO) and <c>v_state</c> ([H, P]). All four are owned by <see cref="VulkanMamba3State"/>
/// and zero-initialised at construction; the orchestrator never resets them between layers
/// (each layer has its own state row).
/// </para>
/// <para>
/// The <c>k_state</c> / <c>v_state</c> pair is consumed by
/// <see cref="Mamba3ChunkBoundaryF32Kernel"/> at the START of each chunk (after the data-
/// RoPE, before the SSD scan), applying the canonical
/// <c>ssm_state += v_state · (Σ_r k_state[r]) · DT[0] · (1-trap[0])</c> adjustment that
/// closes the <c>shifted_γ[T_prev-1]</c> lookahead gap a one-shot forward would have folded
/// in at the previous chunk's last token. The orchestrator skips the dispatch entirely on
/// the first chunk of a sequence (<see cref="VulkanMamba3State.HasBoundary"/> still
/// <c>false</c>) — bit-equal to a zero-buffered dispatch and matches the CPU oracle's
/// <c>if (kState.IsEmpty) skip</c> short-circuit. At the END of each chunk the orchestrator
/// copies the last token's post-RoPE (pre-scale) B slice to <c>k_state</c> (rank-aware) and
/// last token's V (= <c>x</c>) to <c>v_state</c> via <c>vkCmdCopyBuffer</c> regions, then
/// flips <see cref="VulkanMamba3State.MarkBoundaryPrimed"/> for the next call.
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
    // Q8_0 matmul kernels — always created so mixed-quant configs (some Q8_0 in_proj +
    // F32 out_proj, etc.) can dispatch correctly. RecordMatmul branches on the device-side
    // QuantizationType per call.
    private readonly MatMulQ8_0Kernel _matmulQ8;
    private readonly MatMulQ8_0GemmKernel _matmulQ8Gemm;
    // Optional coopmat-accelerated Q8_0 prefill GEMM — null on devices without
    // VK_KHR_cooperative_matrix support, in which case the scalar Q8_0 GEMM is used.
    private readonly MatMulQ8_0GemmCoopmatKernel? _matmulQ8GemmCoopmat;
    // Q4_K_M matmul kernels — Phase 1 of K-quant work. Always created; the dispatcher
    // in RecordMatmul branches on the device-side QuantizationType per call.
    private readonly MatMulQ4KGemvF32Kernel _matmulQ4K;
    private readonly MatMulQ4KGemmF32Kernel _matmulQ4KGemm;
    // Q5_K_M matmul kernels — Phase 1 sibling of Q4_K. Always created.
    private readonly MatMulQ5KGemvF32Kernel _matmulQ5K;
    private readonly MatMulQ5KGemmF32Kernel _matmulQ5KGemm;
    // Q6_K_M matmul kernels — Phase 1 sibling of Q4_K / Q5_K, completing the
    // K-quant matmul kernel coverage. Always created.
    private readonly MatMulQ6KGemvF32Kernel _matmulQ6K;
    private readonly MatMulQ6KGemmF32Kernel _matmulQ6KGemm;
    private readonly RmsNormF32Kernel _rmsnorm;
    private readonly Mamba3DataRopeF32Kernel _dataRope;
    private readonly Mamba3CanonicalSsdSisoF32Kernel _sisoScan;
    // MIMO scan kernel — only created when Mamba3Config.IsMimo is true. Lazy creation
    // keeps the SPV file dependency optional for SISO-only deployments.
    private readonly Mamba3CanonicalSsdMimoF32Kernel? _mimoScan;
    // Streaming-chunk boundary state-adjustment kernel. Used to close the canonical
    // shifted_γ[T_prev-1] lookahead gap when a Forward call resumes a previously
    // primed VulkanMamba3State.
    private readonly Mamba3ChunkBoundaryF32Kernel _boundary;
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
        MatMulF32Kernel matmul,
        MatMulQ8_0Kernel matmulQ8, MatMulQ8_0GemmKernel matmulQ8Gemm,
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat,
        MatMulQ4KGemvF32Kernel matmulQ4K, MatMulQ4KGemmF32Kernel matmulQ4KGemm,
        MatMulQ5KGemvF32Kernel matmulQ5K, MatMulQ5KGemmF32Kernel matmulQ5KGemm,
        MatMulQ6KGemvF32Kernel matmulQ6K, MatMulQ6KGemmF32Kernel matmulQ6KGemm,
        RmsNormF32Kernel rmsnorm,
        Mamba3DataRopeF32Kernel dataRope, Mamba3CanonicalSsdSisoF32Kernel sisoScan,
        Mamba3CanonicalSsdMimoF32Kernel? mimoScan,
        Mamba3ChunkBoundaryF32Kernel boundary,
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
        _matmulQ8 = matmulQ8;
        _matmulQ8Gemm = matmulQ8Gemm;
        _matmulQ8GemmCoopmat = matmulQ8GemmCoopmat;
        _matmulQ4K = matmulQ4K;
        _matmulQ4KGemm = matmulQ4KGemm;
        _matmulQ5K = matmulQ5K;
        _matmulQ5KGemm = matmulQ5KGemm;
        _matmulQ6K = matmulQ6K;
        _matmulQ6KGemm = matmulQ6KGemm;
        _rmsnorm = rmsnorm;
        _dataRope = dataRope;
        _sisoScan = sisoScan;
        _mimoScan = mimoScan;
        _boundary = boundary;
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
        // Q8_0 kernels are always created — production loaders never attach a Q8_0 overlay
        // so they remain unused on the F32 path, but mixed-quant test configs need them
        // bound on every device. Same policy as VulkanNemotronHTransformerModel.
        var matmulQ8 = MatMulQ8_0Kernel.Create(device, spvDir);
        var matmulQ8Gemm = MatMulQ8_0GemmKernel.Create(device, spvDir);
        MatMulQ8_0GemmCoopmatKernel? matmulQ8GemmCoopmat = null;
        if (device.HasCooperativeMatrix)
        {
            try { matmulQ8GemmCoopmat = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir); }
            catch (InvalidOperationException) { /* Kernel threw: no usable tile shape. Stay on scalar. */ }
        }
        // Q4_K_M GEMV + GEMM — Phase 1 of K-quant work. Always created.
        var matmulQ4K = MatMulQ4KGemvF32Kernel.Create(device, spvDir);
        var matmulQ4KGemm = MatMulQ4KGemmF32Kernel.Create(device, spvDir);
        // Q5_K_M GEMV + GEMM — Phase 1 sibling of Q4_K. Always created.
        var matmulQ5K = MatMulQ5KGemvF32Kernel.Create(device, spvDir);
        var matmulQ5KGemm = MatMulQ5KGemmF32Kernel.Create(device, spvDir);
        // Q6_K_M GEMV + GEMM — Phase 1 sibling of Q4_K / Q5_K. Always created.
        var matmulQ6K = MatMulQ6KGemvF32Kernel.Create(device, spvDir);
        var matmulQ6KGemm = MatMulQ6KGemmF32Kernel.Create(device, spvDir);
        var rmsnorm = RmsNormF32Kernel.Create(device, spvDir);
        var dataRope = Mamba3DataRopeF32Kernel.Create(device, spvDir);
        var sisoScan = Mamba3CanonicalSsdSisoF32Kernel.Create(device, spvDir);
        Mamba3CanonicalSsdMimoF32Kernel? mimoScan = null;
        if (config.Mamba3Config!.IsMimo)
            mimoScan = Mamba3CanonicalSsdMimoF32Kernel.Create(device, spvDir);
        var boundary = Mamba3ChunkBoundaryF32Kernel.Create(device, spvDir);
        var add = AddKernel.Create(device, spvDir);

        var submit = device.CreateSubmitContext();

        return new VulkanMamba3TransformerModel(
            device, ownsDevice,
            config, weights, state, recurrent,
            matmul, matmulQ8, matmulQ8Gemm, matmulQ8GemmCoopmat,
            matmulQ4K, matmulQ4KGemm,
            matmulQ5K, matmulQ5KGemm,
            matmulQ6K, matmulQ6KGemm,
            rmsnorm, dataRope, sisoScan, mimoScan, boundary, add,
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
        // Per-head boundary coefficient: coef[h] = dt[0, h] · (1 - trap[0, h]).
        // Shared across layers within a single Forward — every layer recomputes
        // dtHost/trapHost from its own in_proj output, so we recompute coef per
        // layer too. Holding the array outside the layer loop avoids re-allocation.
        float[] coefHost = new float[nHead];

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

            RecordMatmul(cmdBuf, lw.InProj, lw.InProjDeviceQuantType,
                _state.NormOutput, _state.Proj,
                outputDim: lw.InProjOutputDim, inputDim: lw.InProjInputDim, seqLen: seqLen);

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

            // Streaming-chunk boundary coefficient: coef[h] = dt[0, h] · (1 - trap[0, h]).
            // Only uploaded + the boundary kernel only dispatched when the recurrent state
            // is already primed by a prior Forward (HasBoundary == true). On the first
            // chunk of a sequence we skip both — matches the CPU oracle's empty-span
            // short-circuit in Mamba3Block.ApplyChunkBoundaryAdjustment.
            bool runBoundary = _recurrent.HasBoundary;
            if (runBoundary)
            {
                for (int h = 0; h < nHead; h++)
                    coefHost[h] = dtHost[h] * (1f - trapHost[h]);
                _device.Upload(coefHost.AsSpan(0, nHead), _state.BoundaryCoef);
            }

            // ── 2c. DATA-ROPE + SSD SCAN + OUT_PROJ + RESIDUAL (single submit) ───
            _submit.Begin();
            cmdBuf = _submit.CommandBuffer;
            KernelSupport.HostToComputeBarrier(cmdBuf);

            VulkanDevice.Buffer cumAngle = _recurrent.GetCumAngleBuffer(layer);
            VulkanDevice.Buffer ssmState = _recurrent.GetSsmStateBuffer(layer);
            VulkanDevice.Buffer kState = _recurrent.GetKStateBuffer(layer);
            VulkanDevice.Buffer vState = _recurrent.GetVStateBuffer(layer);

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

            // Streaming-chunk boundary state adjustment. Applied BEFORE the SSD scan
            // when the recurrent state was primed by a previous Forward — closes the
            // canonical shifted_γ[T_prev-1] lookahead gap that a one-shot forward would
            // have folded in at the previous chunk's last token. Reads the previous
            // chunk's k_state / v_state (written at the end of the previous Forward)
            // and updates ssm_state in place. SISO passes nRank=1 (kState is [H, N]);
            // MIMO passes the model's mimo_rank (kState is [R, H, N]). On the first
            // chunk of a sequence runBoundary == false → dispatch is skipped entirely
            // (matches CPU oracle's IsEmpty short-circuit; bit-equal to the existing
            // SISO/MIMO single-chunk path that landed at e40ada4 / 7142f31).
            if (runBoundary)
            {
                _boundary.Record(cmdBuf,
                    state: ssmState,
                    vState: vState,
                    kState: kState,
                    coef: _state.BoundaryCoef,
                    nHead: nHead, headDim: headDim, dState: dState,
                    nRank: _recurrent.KStateRank);
                KernelSupport.ComputeToComputeBarrier(cmdBuf);
            }

            if (isMimo)
            {
                // SSD MIMO scan. qRoped = post-RoPE C, kRoped = post-RoPE B (matches the
                // CPU oracle's argument order to ExecuteMimo). qkPreDotHost is the
                // canonical Σ_r form, prepared host-side. mimo_z / mimo_o are uploaded
                // per-layer and bound here. The streaming-chunk boundary adjustment that
                // CPU's ExecuteMimoStreaming applies inline before the scan is dispatched
                // separately above (Mamba3ChunkBoundaryF32Kernel) — by the time we reach
                // this scan, ssm_state already carries the boundary correction (or no-op
                // if HasBoundary was false), so the scan body is bit-equal to ExecuteMimo
                // with the canonical shifted_γ[T-1]=0 boundary inside.
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

            // ── 2d. PERSIST CHUNK-BOUNDARY BUFFERS ───────────────────────────────
            // Copy this chunk's last-token post-RoPE B → k_state, last-token V (= x)
            // → v_state. Both are on-device vkCmdCopyBuffer regions — kState's source
            // is the [T, R, H, N] post-RoPE B buffer (rank slot collapses to 1 for
            // SISO; one contiguous copy of R·H·N floats per layer covers both modes).
            // The next Forward call's boundary kernel will read these via the
            // GetKStateBuffer / GetVStateBuffer accessors. Issued before out_proj so
            // the kState/vState writes overlap with the residual GEMM in flight (the
            // copy targets are disjoint from out_proj's I/O).
            //
            // Compute → transfer barrier: data-RoPE wrote into _state.B as a compute
            // shader; we now read it as a transfer command. Single barrier covers both
            // the B and X reads (X was last touched by the scan, also compute-stage).
            KernelSupport.ComputeToTransferBarrier(cmdBuf);

            int kStateElems = _recurrent.KStateRank * nHead * dState;
            int vStateElems = nHead * headDim;
            // Last-token slice offsets in the post-RoPE B / X scratch.
            // _state.B layout: [seqLen, R, nHead, dState] row-major.
            // _state.X layout: [seqLen, nHead, headDim] row-major.
            ulong lastBOffset = (ulong)((long)(seqLen - 1) * kStateElems * sizeof(float));
            ulong lastXOffset = (ulong)((long)(seqLen - 1) * vStateElems * sizeof(float));
            ulong kStateBytes = (ulong)((long)kStateElems * sizeof(float));
            ulong vStateBytes = (ulong)((long)vStateElems * sizeof(float));
            RecordCopyBufferRange(cmdBuf, _state.B, kState,
                srcOffset: lastBOffset, dstOffset: 0, size: kStateBytes);
            RecordCopyBufferRange(cmdBuf, _state.X, vState,
                srcOffset: lastXOffset, dstOffset: 0, size: vStateBytes);
            // No transfer→compute barrier needed inside this submit — kState/vState
            // are not read again until the NEXT Forward's boundary dispatch, and the
            // SubmitAndWait below + the next call's HostToComputeBarrier serialise
            // those across submits.

            // out_proj: YScan @ OutProj^T → BlockOut.
            RecordMatmul(cmdBuf, lw.OutProj, lw.OutProjDeviceQuantType,
                _state.YScan, _state.BlockOut,
                outputDim: lw.OutProjOutputDim, inputDim: lw.OutProjInputDim, seqLen: seqLen);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);

            // Residual add: NewHidden = OldHidden (Residual) + BlockOut, written into
            // AddScratch, then rotate the slot.
            _add.Record(cmdBuf, _state.Residual, _state.BlockOut, _state.AddScratch,
                seqLen * hiddenSize);

            _submit.SubmitAndWait();
            _state.RotateHiddenSlot();
        }

        // After every layer has run, mark the recurrent state as primed so the NEXT
        // Forward call dispatches the boundary kernel. Idempotent — flipping the flag
        // a second time is a no-op. Stays sticky until VulkanMamba3State.Reset is
        // called by the caller.
        _recurrent.MarkBoundaryPrimed();

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

        RecordMatmul(cmdBuf, _weights.LmHead, _weights.LmHeadDeviceQuantType,
            _state.LastTokenHidden, _state.Logits,
            outputDim: _weights.LmHeadOutputDim, inputDim: _weights.LmHeadInputDim, seqLen: 1);

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
        _matmulQ8.InvalidateDescriptorCache();
        _matmulQ8Gemm.InvalidateDescriptorCache();
        _matmulQ8GemmCoopmat?.InvalidateDescriptorCache();
        _matmulQ4K.InvalidateDescriptorCache();
        _matmulQ4KGemm.InvalidateDescriptorCache();
        _matmulQ5K.InvalidateDescriptorCache();
        _matmulQ5KGemm.InvalidateDescriptorCache();
        _matmulQ6K.InvalidateDescriptorCache();
        _matmulQ6KGemm.InvalidateDescriptorCache();
        _rmsnorm.InvalidateDescriptorCache();
        _dataRope.InvalidateDescriptorCache();
        _sisoScan.InvalidateDescriptorCache();
        _mimoScan?.InvalidateDescriptorCache();
        _boundary.InvalidateDescriptorCache();
        _add.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Dispatches a matmul for one Mamba-3 projection: chooses
    /// <see cref="MatMulQ8_0Kernel"/> / <see cref="MatMulQ4KGemvF32Kernel"/> /
    /// <see cref="MatMulQ5KGemvF32Kernel"/> / <see cref="MatMulQ6KGemvF32Kernel"/>
    /// (decode-path GEMV) when <paramref name="seqLen"/>==1, the batched
    /// <see cref="MatMulQ8_0GemmKernel"/> / <see cref="MatMulQ4KGemmF32Kernel"/> /
    /// <see cref="MatMulQ5KGemmF32Kernel"/> / <see cref="MatMulQ6KGemmF32Kernel"/> (or
    /// the Q8_0 coopmat variant when available) when <paramref name="seqLen"/>&gt;1,
    /// and <see cref="MatMulF32Kernel"/> for every F32 weight. Same routing as
    /// <see cref="VulkanNemotronHTransformerModel"/> / <see cref="VulkanTransformerModel"/>.
    /// </summary>
    /// <remarks>
    /// Q8_0 kernels require <paramref name="inputDim"/> to be a multiple of 32; Q4_K /
    /// Q5_K / Q6_K kernels require it to be a multiple of 256. The upload path
    /// (<see cref="VulkanMamba3Weights"/>) only marks a projection as Q8_0 / Q4_K /
    /// Q5_K / Q6_K when the matching alignment constraint holds; otherwise the source
    /// is uploaded as F32 and lands here as F32, sidestepping the kernel alignment
    /// requirement entirely.
    /// </remarks>
    private void RecordMatmul(
        nint cmdBuf,
        VulkanDevice.Buffer weights, QuantizationType weightQt,
        VulkanDevice.Buffer input, VulkanDevice.Buffer output,
        int outputDim, int inputDim, int seqLen)
    {
        if (weightQt == QuantizationType.Q8_0)
        {
            if (seqLen == 1)
            {
                _matmulQ8.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else if (_matmulQ8GemmCoopmat is not null)
            {
                _matmulQ8GemmCoopmat.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
            else
            {
                _matmulQ8Gemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q4_K)
        {
            if (seqLen == 1)
            {
                _matmulQ4K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ4KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q5_K)
        {
            // Q5_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM. Same
            // alignment as Q4_K (inputDim % 256 == 0, enforced by upload path).
            if (seqLen == 1)
            {
                _matmulQ5K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ5KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else if (weightQt == QuantizationType.Q6_K)
        {
            // Q6_K_M decode-path GEMV (seqLen==1) or prefill-path tiled GEMM. Same
            // alignment as Q4_K / Q5_K (inputDim % 256 == 0, enforced by upload path).
            if (seqLen == 1)
            {
                _matmulQ6K.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim);
            }
            else
            {
                _matmulQ6KGemm.Record(cmdBuf, weights, input, output,
                    m: outputDim, k: inputDim, n: seqLen);
            }
        }
        else
        {
            _matmul.Record(cmdBuf, weights, input, output, outputDim, inputDim, seqLen);
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _submit.Dispose();
        _state.Dispose();
        _weights.Dispose();
        _recurrent.Dispose();

        _add.Dispose();
        _boundary.Dispose();
        _mimoScan?.Dispose();
        _sisoScan.Dispose();
        _dataRope.Dispose();
        _rmsnorm.Dispose();
        _matmulQ6KGemm.Dispose();
        _matmulQ6K.Dispose();
        _matmulQ5KGemm.Dispose();
        _matmulQ5K.Dispose();
        _matmulQ4KGemm.Dispose();
        _matmulQ4K.Dispose();
        _matmulQ8GemmCoopmat?.Dispose();
        _matmulQ8Gemm.Dispose();
        _matmulQ8.Dispose();
        _matmul.Dispose();

        if (_ownsDevice)
            _device.Dispose();
    }
}
