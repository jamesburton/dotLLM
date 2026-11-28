using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Models.Architectures.DSpark;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>Deliberately wrong target-side behaviours of the Vulkan Inkling 2 DSpark hooks (test-only mutants, issue #2046; the CPU's are <c>Inkling2Mutant</c>).</summary>
internal enum DSparkTargetMutant
{
    None,
    /// <summary>Core-layer taps keyed by cache slot instead of layer visit.</summary>
    TapBySlot,
    /// <summary>Prelude taps one visit early.</summary>
    TapEarlyPrelude,
    /// <summary>Every tap one visit late (the output of layer v instead of its input).</summary>
    TapLate,
    /// <summary>Visit 48 tapped before the output norm.</summary>
    TapPreNorm,
    /// <summary><c>RestoreRecurrentStateToRow</c> does nothing.</summary>
    SkipRestore,
    /// <summary>Restores row + 1.</summary>
    RestoreRowPlusOne,
    /// <summary>Restores only the 24 pass-0 slots (the pass-2/3 core slots 24..47 keep the rejected rows' state).</summary>
    RestorePass0Only,
}

/// <summary>
/// The DSpark speculative drafter on the Vulkan Inkling 2 target (issue #2046, epic #2025; CPU: <see cref="Inkling2TransformerModel"/>, semantics <c>docs/INKLING2.md</c> sections 13 to 15).
/// The drafter takes the model's MTP slot, so the unmodified <c>MtpSpeculativeDecoder</c>, the adaptive gate and streaming verify its drafts.
/// </summary>
/// <remarks>
/// <para><b>Precision (user rulings 2026-11-26/27).</b> Everything the drafter stores is BF16 (activations, taps, the Markov pre-activations) or F16 (context K/V, the <c>f_rnn</c> ring);
/// Q8_0 / BF16 weights are read in place. F32 is only an in-register / shared-memory accumulator. The Markov recurrent state is never stored: each walk step replays the
/// earlier steps from the stored BF16 pre-activations and token ids inside one workgroup. The only stored F32 are the drafter's logits (the same class as the target's logits;
/// <c>(block - 1) x vocab x 4</c> bytes), the 8-byte-per-workgroup argmax partials and the F32 weight vectors (norm gains, mix, bias) uploaded verbatim.</para>
/// <para><b>Windowed context K/V (#2060).</b> Drafter layers 0 and 2 are sliding-window layers: they keep a ring of <c>window + MaxChunk</c> positions (1025 + 128), flat in the
/// sequence length. Global layer 1 keeps the whole context in a buffer that doubles on demand, admitted against free device memory (never spilled).</para>
/// </remarks>
public sealed unsafe partial class VulkanInkling2TransformerModel : IContextGatedMtp
{
    // ── lazy attach (issue #2046, James 2026-11-28: the drafter never won below several thousand tokens, so an auto-detected one is attached per request above a context threshold) ──
    private string? _deferredPath;
    private bool _dsparkForced;
    private int? _minContextOverride;
    private readonly object _attachLock = new();

    /// <summary>Env var of the context threshold (prompt + generation budget, tokens) from which an auto-detected drafter is attached; <c>0</c> = every request, <c>off</c> = never (opt-in through <c>--dspark</c> only).</summary>
    public const string MinContextEnvVar = "DOTLLM_DSPARK_MIN_CONTEXT";

    /// <summary>Default of <see cref="DSparkMinContext"/>.</summary>
    public const int DefaultMinContext = 12288;

    /// <summary>Context (prompt + generation budget, tokens) from which an auto-detected drafter attaches; settable (tests, embedders), else <see cref="MinContextEnvVar"/>, else <see cref="DefaultMinContext"/>.</summary>
    public int DSparkMinContext
    {
        get
        {
            if (_minContextOverride is { } o) return o;
            return DSparkAttachPolicy.ParseMinContext(Environment.GetEnvironmentVariable(MinContextEnvVar), DefaultMinContext);
        }
        set => _minContextOverride = value;
    }

    /// <inheritdoc/>
    public bool DefersAutoAttach => true;

    /// <inheritdoc/>
    public void DeferAutoAttach(string path) { lock (_attachLock) _deferredPath = path; }

    /// <inheritdoc/>
    public (string Path, int MinContext)? DeferredDrafter => _deferredPath is { } p && _dsd is null ? (p, DSparkMinContext) : null;

    /// <inheritdoc/>
    public bool PrepareMtpFor(int contextTokens)
    {
        lock (_attachLock)
        {
            if (_dsd is null)
            {
                if (_deferredPath is not { } path || !DSparkAttachPolicy.ShouldAttach(true, contextTokens, DSparkMinContext)) return false;
                try { AttachCore(path); }
                catch (Exception e) when (e is not OutOfMemoryException)
                {
                    Console.Error.WriteLine($"[dspark] WARNING: drafter not attached ({e.GetType().Name}: {e.Message}); decoding without speculation.");
                    _deferredPath = null;
                    return false;
                }
                Console.Error.WriteLine($"[dspark] drafter attached for a context of {contextTokens} tokens (threshold {DSparkMinContext}).");
            }
            return DSparkAttachPolicy.MayUse(_dsparkForced, contextTokens, DSparkMinContext);
        }
    }

    // ── target-side state ──
    private VulkanDevice.Buffer? _snapBuf;
    private int _snapCapRows, _snapR;
    private bool _recordSnap, _rollbackPending;
    private int _recRows, _recPos0, _recR;
    private VulkanInkling2DSparkState? _tapState;
    private DSparkDrafter? _dsd;

    /// <summary>Test-only target mutant.</summary>
    internal DSparkTargetMutant TargetMutant { get; set; }

    /// <summary>Test-only drafter mutant (shared with the CPU drafter's enum).</summary>
    internal DSparkMutant DrafterMutant { get; set; }

    /// <summary>The attached drafter's spec (null when none).</summary>
    internal Inkling2DSparkSpec? DSparkSpec => _dsd?.Spec;

    /// <summary>Device bytes of the attached drafter: weights read in place are counted by their size, plus its scratch.</summary>
    public long DSparkBytes => _dsd?.Bytes ?? 0;

    /// <summary>Bytes of the recorded conv-window snapshots (BF16), 0 until the first <see cref="ForwardWithRecurrentSnapshots"/>.</summary>
    public long SnapshotBytes => _snapBuf?.Size ?? 0;

    internal VulkanDevice DeviceForTests => _device;

    /// <summary>Test hook: sets the sequence length without touching any state (the SkipRestore mutant test).</summary>
    internal void ForceLengthForTests(int length) { _length = length; _rollbackPending = false; }

    private int LayerVisits => _c.LoopPasses > 1 ? _nPhys + (_c.LoopPasses - 1) * (_c.LoopEnd - _c.LoopStart) : _nPhys;

    // ───────────────────────────── row snapshots (recurrent conv windows) ─────────────────────────────

    private long SnapOffset(int slotIx, int kind)
    {
        if (!_recordSnap) return -1;
        long r3 = 3 + _snapR, per = r3 * (2L * _kvW + 2L * _H);
        long b = slotIx * per;
        return kind switch { 0 => b, 1 => b + r3 * _kvW, 2 => b + 2 * r3 * _kvW, _ => b + 2 * r3 * _kvW + r3 * _H };
    }

    private void PrepareSnapshots(int rows)
    {
        if (rows > _chunk) throw new NotSupportedException($"inkling2: a recorded verify of {rows} rows exceeds the {_chunk}-token chunk.");
        _snapR = rows;
        if (_snapBuf is not null && rows <= _snapCapRows) return;
        int cap = Math.Max(rows, 8);
        long bytes = (long)_slots.Length * (3 + cap) * (2L * _kvW + 2L * _H) * 2;
        if (_snapBuf is not null) { _deviceBytes -= _snapBuf.Size; _snapBuf.Dispose(); _snapBuf = null; }
        _snapBuf = _device.AllocateDeviceLocal(bytes);
        _deviceBytes += _snapBuf.Size;
        _snapCapRows = cap;
        foreach (var k in _kernels.Values) k.InvalidateDescriptorCache();
    }

    /// <inheritdoc/>
    public bool SupportsRecurrentRowSnapshots => true;

    /// <inheritdoc/>
    public ITensor ForwardWithRecurrentSnapshots(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, IMtpState? mtpState)
        => ForwardCore(tokenIds, positions, deviceId, lastOnly: false, ds: AsState(mtpState), recordRows: true);

    /// <inheritdoc/>
    public void RestoreRecurrentStateToRow(int row)
    {
        if (TargetMutant == DSparkTargetMutant.SkipRestore) return;
        if (TargetMutant == DSparkTargetMutant.RestoreRowPlusOne) row++;
        if (_recRows == 0 || (uint)row >= (uint)_recRows || _snapBuf is null)
            throw new InvalidOperationException($"No recorded verify window for row {row} ({_recRows} rows recorded by the last ForwardWithRecurrentSnapshots).");
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        _pendingW.Clear(); _pendingR.Clear();
        var k = Kernel("conv_restore", "inkling2_conv_restore", 2, 3, false);
        long r3 = 3 + _recR, per = r3 * (2L * _kvW + 2L * _H);
        int nSlots = TargetMutant == DSparkTargetMutant.RestorePass0Only ? _nPhys : _slots.Length;
        for (int s = 0; s < nSlots; s++)
        {
            var sl = _slots[s];
            long b = s * per;
            Run(k, 1u << 1, [(uint)_kvW, (uint)row, (uint)b], (uint)((_kvW + 63) / 64), 1, 1, _snapBuf, sl.KConv);
            Run(k, 1u << 1, [(uint)_kvW, (uint)row, (uint)(b + r3 * _kvW)], (uint)((_kvW + 63) / 64), 1, 1, _snapBuf, sl.VConv);
            Run(k, 1u << 1, [(uint)_H, (uint)row, (uint)(b + 2 * r3 * _kvW)], (uint)((_H + 63) / 64), 1, 1, _snapBuf, sl.AttnConv);
            Run(k, 1u << 1, [(uint)_H, (uint)row, (uint)(b + 2 * r3 * _kvW + r3 * _H)], (uint)((_H + 63) / 64), 1, 1, _snapBuf, sl.FfnConv);
        }
        _submit.SubmitAndWait();
        _pendingW.Clear(); _pendingR.Clear();
        _length = _recPos0 + row + 1;
        _rollbackPending = false;
    }

    // ───────────────────────────── MTP surface (the drafter takes the MTP slot) ─────────────────────────────

    /// <inheritdoc/>
    public bool HasDFlashDrafter => _dsd is not null;

    /// <inheritdoc/>
    public bool SupportsMtp => _dsd is not null;

    /// <inheritdoc/>
    public int MtpPreferredDraftTokens => _dsd?.Spec.DraftTokensPerRound ?? 0;

    /// <inheritdoc/>
    public int MtpMaxDraftTokens => _dsd?.Spec.DraftTokensPerRound ?? int.MaxValue;

    /// <inheritdoc/>
    public bool SupportsMtpArgMax => _dsd is not null;

    /// <inheritdoc/>
    public IMtpState? CreateMtpState() => _dsd is null ? null : new VulkanInkling2DSparkState(this, RingPositionsFor());

    /// <inheritdoc/>
    public IMtpState? CreateMtpState(int maxSequenceLength) => CreateMtpState();

    internal int RingPositionsFor() => (_dsd?.Spec.Window ?? 1025) + _chunk;

    /// <summary>Opens a state with an explicit ring size (tests: the one-row-too-short mutant).</summary>
    internal VulkanInkling2DSparkState CreateDSparkState(int ringPositions, int globalInitialPositions = 0)
        => new(this, ringPositions, globalInitialPositions);

    private VulkanInkling2DSparkState? AsState(IMtpState? s)
        => s is null ? null : s as VulkanInkling2DSparkState ?? throw new NotSupportedException($"{nameof(VulkanInkling2TransformerModel)} drafts only with DSpark states.");

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState)
        => adapter is not null ? throw new NotSupportedException("inkling2 on Vulkan has no LoRA support.") : ForwardCore(tokenIds, positions, deviceId, lastOnly: false, ds: AsState(mtpState));

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache, ILoraAdapter? adapter, IMtpState? mtpState, bool lastTokenLogitsOnly)
        => adapter is not null ? throw new NotSupportedException("inkling2 on Vulkan has no LoRA support.") : ForwardCore(tokenIds, positions, deviceId, lastTokenLogitsOnly, ds: AsState(mtpState));

    /// <inheritdoc/>
    public int ForwardMtpArgMax(IMtpState state, int tokenId, int position)
        => (AsState(state) ?? throw new ArgumentNullException(nameof(state))).NextDraft(tokenId, position);

    /// <inheritdoc/>
    public ITensor ForwardMtp(IMtpState state, int tokenId, int position)
    {
        var s = AsState(state) ?? throw new ArgumentNullException(nameof(state));
        s.NextDraft(tokenId, position);
        if (s.Unusable) return Inkling2DSparkSpec.AnchorLogits(tokenId, _vocab);   // no block was drafted: a defined row whose argmax is the anchor (#2046 review)
        var row = s.DraftLogits(position);
        var result = UnmanagedTensor.Allocate(new TensorShape(1, row.Length), DType.Float32, -1);
        row.CopyTo(new Span<float>((void*)result.DataPointer, row.Length));
        return result;
    }

    // ───────────────────────────── attach ─────────────────────────────

    private sealed class DLayer
    {
        public required bool Local;
        public required Mat Q, K, V, O, KCtx, VCtx, Gate, Up, Down;
        public required VulkanDevice.Buffer AttnNorm, FfnNorm, QNorm, KNorm, KCtxNorm, CtxMix, CtxNorm;
    }

    private sealed class DSparkDrafter : IDisposable
    {
        public required Inkling2DSparkSpec Spec;
        public required GgufFile File;
        public required DLayer[] Layers;
        public required int[] SlotOfVisit;
        public required VulkanDevice.Buffer OutNorm, TokNorm, FrnnMix, FrnnNorm, W1, W2, Joint, JointBias, Mask, RopeTab;
        public VulkanDevice.Buffer? W2Bias;
        public bool LogitsF16;   // drafter logits stored as F16 (2 B per element) instead of F32
        public VulkanDevice.Buffer Taps = null!, Feat = null!, KTmp = null!, VTmp = null!, X = null!, Nrm = null!, O = null!, Q = null!, Q2 = null!, K = null!, V = null!,
            KBlk = null!, VBlk = null!, AttnO = null!, Gate = null!, Up = null!, Hid = null!, Logits = null!, Pre = null!, C2 = null!, Part = null!, Tok = null!;
        public readonly List<VulkanDevice.Buffer> Owned = new();
        public long Bytes;
        public int Parts;

        public void Dispose()
        {
            foreach (var b in Owned) b.Dispose();
            Owned.Clear();
            File.Dispose();
        }
    }

    private void DisposeDSpark()
    {
        _dsd?.Dispose();
        _dsd = null;
        _snapBuf?.Dispose();
        _snapBuf = null;
    }

    /// <inheritdoc/>
    public void AttachDFlashDrafter(string path)
    {
        lock (_attachLock)
        {
            if (_dsd is null) AttachCore(path);
            else throw new InvalidOperationException("A DSpark drafter is already attached.");
            _dsparkForced = true;   // an explicit attach (--dspark) forces the drafter on at every context length
        }
    }

    private void AttachCore(string path)
    {
        if (_dsd is not null) throw new InvalidOperationException("A DSpark drafter is already attached.");
        if (_opt.ActivationF32) throw new NotSupportedException("The DSpark drafter runs BF16 activations; the F32-activation reference arm is not supported.");
        var file = GgufFile.Open(path);
        try { _dsd = BuildDrafter(file, path); }
        catch { file.Dispose(); throw; }
    }

    private DSparkDrafter BuildDrafter(GgufFile file, string path)
    {
        var sp = Inkling2DSparkSpec.Read(file, _H, LayerVisits, _vocab);
        if (sp.TargetVisits.Length != 4 || sp.HeadDim != 128 || sp.Heads != 3 * sp.KvHeads || sp.KvHeads != _nkv || sp.Rank != 256 || sp.BlockSize > 8 || _H % 8 != 0 || sp.Ffn % 8 != 0
            || sp.Layers < 1 || sp.Heads * sp.HeadDim != _qW)
            throw new NotSupportedException($"The Inkling 2 Vulkan DSpark kernels are written for 4 taps, head_dim 128, 3 query heads per KV head, rank 256 and a block of at most 8; got {sp.TargetVisits.Length} taps, " +
                                            $"head_dim {sp.HeadDim}, {sp.Heads} q / {sp.KvHeads} kv heads, rank {sp.Rank}, block {sp.BlockSize}.");
        int kvDim = sp.KvHeads * sp.HeadDim, qDim = sp.Heads * sp.HeadDim, B = sp.BlockSize;
        long estimate = new FileInfo(path).Length + 96L * 1024 * 1024;
        if (!VulkanKvCache.AdmitGrowth(_device, estimate))
            throw new InvalidOperationException($"The DSpark drafter needs about {estimate / 1048576} MiB of device memory and the device does not have it free (the target holds {_device.ResidentLiveBytes() / 1048576} MiB); not attached.");

        var owned = new List<VulkanDevice.Buffer>();
        long bytes = 0;
        var tensors = file.TensorsByName;
        using var staging = VulkanStagingBuffer.Create(_device, 64L * 1024 * 1024);
        VulkanDevice.Buffer Keep(VulkanDevice.Buffer b) { owned.Add(b); return b; }

        Mat M(string name, int @in, int @out, bool bf16Only = false)
        {
            var d = tensors[name];
            var qt = d.QuantizationType;
            int inDim = d.Shape[0], outDim = d.Shape.Rank > 1 ? d.Shape[1] : 1;
            if (inDim != @in || outDim != @out) throw new InvalidOperationException($"Drafter tensor '{name}' is [{inDim}, {outDim}], expected [{@in}, {@out}].");
            bool ok = bf16Only ? qt == QuantizationType.BF16 : qt == QuantizationType.Q8_0 ? inDim % 32 == 0 : qt == QuantizationType.BF16;
            if (!ok) throw new NotSupportedException($"Drafter tensor '{name}' is {qt}: the Vulkan drafter reads Q8_0 / BF16 weights in place and refuses to widen anything else.");
            string stem = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^".weight".Length] : name;
            if (tensors.ContainsKey(stem + ".scale") || tensors.ContainsKey(stem + ".input_scale")) throw new NotSupportedException($"Drafter tensor '{name}' carries a scale; not supported.");
            var buf = Keep(VulkanQwen3MoeHybridWeights.UploadRawQuantMatrix(_device, staging, file.TensorDataPointer(d), qt, outDim, inDim, out long up));
            bytes += up;
            return new Mat { Buf = buf, Qt = qt, In = inDim, Out = outDim };
        }
        VulkanDevice.Buffer Vec(string name, int count)
        {
            if (!tensors.TryGetValue(name, out var d)) throw new InvalidOperationException($"Drafter is missing tensor '{name}'.");
            if (d.Shape.ElementCount != count) throw new InvalidOperationException($"Drafter tensor '{name}' has {d.Shape.ElementCount} elements, expected {count}.");
            var f = new float[count];
            Dequantize.ToFloat32(file.TensorDataPointer(d), count, d.QuantizationType, f);
            var buf = Keep(VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, f));
            bytes += buf.Size;
            return buf;
        }

        try
        {
            var layers = new DLayer[sp.Layers];
            for (int i = 0; i < layers.Length; i++)
            {
                string Bn(string s) => $"blk.{i}.{s}";
                layers[i] = new DLayer
                {
                    Local = sp.Local[i],
                    AttnNorm = Vec(Bn("attn_norm.weight"), _H), FfnNorm = Vec(Bn("ffn_norm.weight"), _H),
                    QNorm = Vec(Bn("attn_q_norm.weight"), sp.HeadDim), KNorm = Vec(Bn("attn_k_norm.weight"), sp.HeadDim), KCtxNorm = Vec(Bn("attn_k_ctx_norm.weight"), sp.HeadDim),
                    CtxMix = Vec(Bn("ctx_mix.weight"), 4), CtxNorm = Vec(Bn("ctx_norm.weight"), _H),
                    Q = M(Bn("attn_q.weight"), _H, qDim), K = M(Bn("attn_k.weight"), _H, kvDim), V = M(Bn("attn_v.weight"), _H, kvDim),
                    O = M(Bn("attn_output.weight"), qDim, _H), KCtx = M(Bn("attn_k_ctx.weight"), _H, kvDim), VCtx = M(Bn("attn_v_ctx.weight"), _H, kvDim),
                    Gate = M(Bn("ffn_gate.weight"), _H, sp.Ffn), Up = M(Bn("ffn_up.weight"), _H, sp.Ffn), Down = M(Bn("ffn_down.weight"), sp.Ffn, _H),
                };
            }
            var outNorm = Vec("output_norm.weight", _H);
            var tokNorm = Vec("token_embd_norm.weight", _H);
            var frnnMix = Vec("markov_ctx_mix.weight", 4);
            var frnnNorm = Vec("markov_ctx_norm.weight", _H);
            var w1 = M("markov_w1.weight", sp.Rank, _vocab, bf16Only: true);
            var w2 = M("markov_w2.weight", sp.Rank, _vocab, bf16Only: true);
            var joint = M("joint_proj.weight", 2 * _H + 2 * sp.Rank, 3 * sp.Rank, bf16Only: true);
            var jointBias = Vec("joint_proj.bias", 3 * sp.Rank);
            VulkanDevice.Buffer? w2Bias = sp.HasW2Bias ? Vec("markov_w2.bias", _vocab) : null;

            // mask_embd (a [H, rows] F32 weight in the file) is the block rows' INPUT activation: stored once as BF16, copied into the block with no further arithmetic.
            var maskD = tensors["mask_embd.weight"];
            int maskCount = (B - 1) * _H;
            var maskF = new float[sp.MaskRows * _H];
            Dequantize.ToFloat32(file.TensorDataPointer(maskD), maskF.Length, maskD.QuantizationType, maskF);
            var maskBits = new ushort[maskCount];
            for (int i = 0; i < maskCount; i++) maskBits[i] = Inkling2ConvState.ToBf16(maskF[i]);
            VulkanDevice.Buffer mask;
            fixed (ushort* mp = maskBits)
            {
                mask = Keep(_device.AllocateDeviceLocal(Math.Max(16, maskCount * 2L)));
                staging.UploadBytes((nint)mp, maskCount * 2L, mask);
            }
            bytes += mask.Size;

            // RoPE table: f_i = theta^(-2i/128) / (2 pi) as a hi/lo F32 pair, computed in double.
            var rope = new float[2 * (sp.HeadDim / 2)];
            for (int i = 0; i < sp.HeadDim / 2; i++)
            {
                double f = Math.Pow(sp.RopeTheta, -2.0 * i / sp.HeadDim) / (2.0 * Math.PI);
                float hi = (float)f;
                rope[2 * i] = hi; rope[2 * i + 1] = (float)(f - hi);
            }
            var ropeTab = Keep(VulkanQwen3MoeHybridWeights.UploadFloatArray(_device, staging, rope));

            var d = new DSparkDrafter
            {
                Spec = sp, File = file, Layers = layers, SlotOfVisit = SlotsFor(sp, LayerVisits),
                OutNorm = outNorm, TokNorm = tokNorm, FrnnMix = frnnMix, FrnnNorm = frnnNorm, W1 = w1.Buf, W2 = w2.Buf, Joint = joint.Buf, JointBias = jointBias, Mask = mask, RopeTab = ropeTab,
                W2Bias = w2Bias, LogitsF16 = DraftLogitsF16FromEnv(),
            };
            long scratch = 0;
            VulkanDevice.Buffer S(long bytes2, bool host = false)
            {
                var b = Keep(host ? _device.Allocate(Math.Max(16, bytes2)) : _device.AllocateDeviceLocal(Math.Max(16, bytes2)));   // in `owned`: a later failure frees it too
                scratch += b.Size;
                return b;
            }
            long bf = 2;
            d.Taps = S((long)_chunk * 4 * _H * bf); d.Feat = S((long)_chunk * _H * bf); d.KTmp = S((long)_chunk * kvDim * bf); d.VTmp = S((long)_chunk * kvDim * bf);
            d.X = S((long)B * _H * bf); d.Nrm = S((long)B * _H * bf); d.O = S((long)B * _H * bf);
            d.Q = S((long)B * qDim * bf); d.Q2 = S((long)B * qDim * bf); d.K = S((long)B * kvDim * bf); d.V = S((long)B * kvDim * bf);
            d.KBlk = S((long)B * kvDim * 2); d.VBlk = S((long)B * kvDim * 2); d.AttnO = S((long)B * qDim * bf);
            d.Gate = S((long)B * sp.Ffn * bf); d.Up = S((long)B * sp.Ffn * bf); d.Hid = S((long)B * _H * bf);
            d.Logits = S((long)(B - 1) * _vocab * (d.LogitsF16 ? 2 : 4), host: true);
            d.Pre = S((long)B * 768 * bf); d.C2 = S(sp.Rank * bf);
            d.Parts = (_vocab + 31) / 32;
            d.Part = S((long)d.Parts * 8);
            d.Tok = S(64, host: true);
            d.Owned.AddRange(owned);
            d.Bytes = bytes + scratch;
            _deviceBytes += d.Bytes;
            return d;
        }
        catch
        {
            foreach (var b in owned) b.Dispose();
            throw;
        }
    }

    private static int[] SlotsFor(Inkling2DSparkSpec sp, int visits)
    {
        var slots = new int[visits + 1];
        Array.Fill(slots, -1);
        for (int k = 0; k < sp.TargetVisits.Length; k++) slots[sp.TargetVisits[k]] = k;
        return slots;
    }

    // ───────────────────────────── taps and context injection (recorded into the target's command buffer) ─────────────────────────────

    private void TapVisit(int id, VulkanDevice.Buffer src, int rows)
    {
        if (_tapState is null || _dsd is not { } d) return;
        if ((uint)id >= (uint)d.SlotOfVisit.Length) return;
        int slot = d.SlotOfVisit[id];
        if (slot < 0) return;
        var k = Kernel("ds_copy", "inkling2_dspark_copy", 2, 6, false);
        Run(k, 1u << 1, [(uint)_H, (uint)rows, 0u, (uint)_H, (uint)(slot * _H), (uint)(4 * _H)], (uint)((_H + 255) / 256), (uint)rows, 1, src, d.Taps);
    }

    private void HeadNorm(VulkanDevice.Buffer src, VulkanDevice.Buffer? gamma, bool norm, bool rope, bool f16, int heads, int rows, int posBase, int slotBase, int ringCap, long outOff,
                          VulkanDevice.Buffer dst, float eps)
    {
        var k = Kernel($"ds_hn|{(norm ? 1 : 0)}|{(rope ? 1 : 0)}|{(f16 ? 1 : 0)}", "inkling2_dspark_headnorm", 4, 7, false, norm ? 1u : 0u, rope ? 1u : 0u, f16 ? 1u : 0u);
        Run(k, 1u << 3, [(uint)heads, (uint)posBase, (uint)slotBase, (uint)ringCap, (uint)outOff, 0u, F(eps)], (uint)heads, (uint)rows, 1, src, gamma ?? _dummy, _dsd!.RopeTab, dst);
    }

    private void CtxMix(VulkanDevice.Buffer mix, VulkanDevice.Buffer gamma, int rows, int posBase, bool frnn, VulkanDevice.Buffer dst, float eps)
    {
        var k = Kernel($"ds_cm|{(frnn ? 1 : 0)}", "inkling2_dspark_ctxmix", 4, 3, false, frnn ? 1u : 0u);
        Run(k, 1u << 3, [(uint)_H, (uint)posBase, F(eps)], (uint)rows, 1, 1, _dsd!.Taps, mix, gamma, dst);
    }

    private void Eltwise(int mode, VulkanDevice.Buffer a, VulkanDevice.Buffer b, long n)
    {
        var k = Kernel($"ds_el|{mode}", "inkling2_dspark_eltwise", 2, 1, false, (uint)mode);
        Run(k, 1u << 0, [(uint)n], (uint)((n + 255) / 256), 1, 1, a, b);
    }

    /// <summary>
    /// Projects the chunk's captured tap rows into the drafter's per-layer context K/V (llama.cpp <c>build_inject_ctx_mix</c>): per layer
    /// <c>rms(sum_k ctx_mix[k] tap_k) * ctx_norm</c> -> <c>attn_k_ctx</c> / <c>attn_v_ctx</c>; K gets a per-head RMSNorm and RoPE at its position, V neither; both are stored F16 at the
    /// position's slot (ring for the windowed layers). Also keeps each position's <c>f_rnn</c> row (F16 ring).
    /// </summary>
    private void RecordInject(VulkanInkling2DSparkState st, int rows, int basePos)
    {
        var d = _dsd!; var sp = d.Spec;
        CtxMix(d.FrnnMix, d.FrnnNorm, rows, basePos, frnn: true, st.Frnn, sp.Eps);
        for (int li = 0; li < d.Layers.Length; li++)
        {
            var l = d.Layers[li];
            CtxMix(l.CtxMix, l.CtxNorm, rows, basePos, frnn: false, d.Feat, sp.Eps);
            Gemv(l.KCtx, d.Feat, d.KTmp, rows);
            Gemv(l.VCtx, d.Feat, d.VTmp, rows);
            int ring = st.RingFor(li);
            HeadNorm(d.KTmp, l.KCtxNorm, norm: true, rope: true, f16: true, sp.KvHeads, rows, basePos, basePos, ring, 0, st.K[li], sp.Eps);
            HeadNorm(d.VTmp, null, norm: false, rope: false, f16: true, sp.KvHeads, rows, basePos, basePos, ring, 0, st.V[li], sp.Eps);
        }
        st.NoteInjected(basePos, rows);
    }

    // ───────────────────────────── drafting ─────────────────────────────

    /// <summary>
    /// Drafts <paramref name="n"/> tokens after <paramref name="anchor"/> (position <paramref name="pos0"/>): one pass over the block <c>[anchor, mask x n]</c> (non-causal, every row uses the
    /// anchor's window) and then the Markov walk, all in one command buffer; the n token ids are the only thing read back. False when the state has no context for the position
    /// before the block.
    /// </summary>
    internal bool DraftBlock(VulkanInkling2DSparkState st, int anchor, int pos0, int n, Span<int> tokens)
    {
        var d = _dsd!; var sp = d.Spec;
        int b = n + 1, H = _H, qDim = sp.Heads * sp.HeadDim, kvDim = sp.KvHeads * sp.HeadDim;
        if ((uint)anchor >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(anchor));
        int fpos = DrafterMutant == DSparkMutant.FrnnWrongPos ? pos0 - 2 : pos0 - 1;
        if (!st.HasFrnn(fpos)) return false;

        _device.Upload(MemoryMarshal.AsBytes(new ReadOnlySpan<int>(in anchor)), d.Tok);
        _submit.Begin();
        _cmd = _submit.CommandBuffer;
        KernelSupport.HostToComputeBarrier(_cmd);
        _pendingW.Clear(); _pendingR.Clear();

        // x row 0 = the drafter's token_embd_norm of the raw target embedding row (the embed kernel with the DRAFTER's gain); rows 1..n = mask_embd.
        var emb = Kernel("emb", "inkling2_embed", 4, 3);
        Run(emb, 1u << 3, [(uint)H, (uint)_vocab, F(sp.Eps)], 1, 1, 1, _embedMat.Buf, d.Tok, d.TokNorm, d.X);
        var cp = Kernel("ds_copy", "inkling2_dspark_copy", 2, 6, false);
        Run(cp, 1u << 1, [(uint)H, (uint)n, 0u, (uint)H, (uint)H, (uint)H], (uint)((H + 255) / 256), (uint)n, 1, d.Mask, d.X);

        var attn = Kernel("ds_attn", "inkling2_dspark_attn", 6, 7, false);
        for (int li = 0; li < d.Layers.Length; li++)
        {
            var l = d.Layers[li];
            Rms(d.X, 0, l.AttnNorm, d.Nrm, 0, b, sp.Eps);
            Gemv(l.Q, d.Nrm, d.Q, b); Gemv(l.K, d.Nrm, d.K, b); Gemv(l.V, d.Nrm, d.V, b);
            HeadNorm(d.Q, l.QNorm, norm: true, rope: true, f16: false, sp.Heads, b, pos0, 0, 0, 0, d.Q2, sp.Eps);
            // The block's K/V go through the F16 cache in llama.cpp before attention reads them: stored F16 here too.
            HeadNorm(d.K, l.KNorm, norm: true, rope: true, f16: true, sp.KvHeads, b, pos0, 0, 0, 0, d.KBlk, sp.Eps);
            HeadNorm(d.V, null, norm: false, rope: false, f16: true, sp.KvHeads, b, pos0, 0, 0, 0, d.VBlk, sp.Eps);
            bool windowed = l.Local && DrafterMutant != DSparkMutant.AllLayersGlobal || DrafterMutant == DSparkMutant.AllLayersLocal;
            int lo = windowed ? Math.Max(0, pos0 - sp.Window + 1) : 0;
            if (DrafterMutant == DSparkMutant.PerQueryWindow) lo = windowed ? Math.Max(0, pos0 + 1 - sp.Window + 1) : 0;
            int ring = st.RingFor(li);
            Run(attn, 1u << 5, [(uint)b, (uint)lo, (uint)pos0, (uint)ring, (uint)sp.Heads, (uint)sp.KvHeads, F(1f / MathF.Sqrt(sp.HeadDim))], (uint)sp.KvHeads, 1, 1,
                d.Q2, st.K[li], st.V[li], d.KBlk, d.VBlk, d.AttnO);
            Gemv(l.O, d.AttnO, d.O, b);
            Eltwise(0, d.X, d.O, (long)b * H);
            Rms(d.X, 0, l.FfnNorm, d.Nrm, 0, b, sp.Eps);
            Gemv(l.Gate, d.Nrm, d.Gate, b); Gemv(l.Up, d.Nrm, d.Up, b);
            Eltwise(1, d.Gate, d.Up, (long)b * sp.Ffn);
            Gemv(l.Down, d.Gate, d.O, b);
            Eltwise(0, d.X, d.O, (long)b * H);
        }

        // Rows 1..n: output_norm -> hidden_i, and the target LM head -> UNCAPPED base logits (F32, like the target's own).
        Rms(d.X, H, d.OutNorm, d.Hid, 0, n, sp.Eps);
        Gemv(_output, d.Hid, d.Logits, n, 0, f32Out: !d.LogitsF16, f16Out: d.LogitsF16);

        var pre = Kernel("ds_mpre", "inkling2_dspark_markov_pre", 5, 4, false);
        Run(pre, 1u << 4, [(uint)H, (uint)sp.Rank, (uint)n, (uint)(fpos % 16)], 768, 1, 1, d.Joint, d.Hid, st.Frnn, d.JointBias, d.Pre);
        var step = Kernel("ds_mstep", "inkling2_dspark_markov_step", 5, 5, false);
        var logits = Kernel(d.LogitsF16 ? "ds_mlog16" : "ds_mlog", d.LogitsF16 ? "inkling2_dspark_markov_logits_f16" : "inkling2_dspark_markov_logits", 5, 4, false);
        var amax = Kernel("ds_amax", "inkling2_dspark_argmax", 2, 2, false);
        for (int i = 1; i <= n; i++)
        {
            Run(step, 1u << 4, [(uint)H, (uint)sp.Rank, (uint)i, (uint)_vocab, DrafterMutant == DSparkMutant.NoMarkovState ? (uint)i : 1u], 1, 1, 1, d.Joint, d.W1, d.Pre, d.Tok, d.C2);
            bool capFirst = DrafterMutant == DSparkMutant.SoftcapBeforeBias;
            Run(logits, (1u << 0) | (1u << 4), [(uint)_vocab, (uint)((long)(i - 1) * _vocab), d.W2Bias is null ? 0u : 1u, F(capFirst ? 0f : sp.Softcap)], (uint)d.Parts, 1, 1,
                d.Logits, d.W2, d.W2Bias ?? _dummy, d.C2, d.Part);
            Run(amax, 1u << 1, [(uint)d.Parts, (uint)i], 1, 1, 1, d.Part, d.Tok);
        }
        _submit.SubmitAndWait();
        _pendingW.Clear(); _pendingR.Clear();
        var raw = new float[8];
        _device.Download(d.Tok, raw);
        var ids = MemoryMarshal.Cast<float, int>(raw.AsSpan());
        for (int i = 0; i < n; i++) tokens[i] = ids[1 + i];
        st.NoteDrafted(n);
        return true;
    }

    /// <summary>The softcapped logits rows <c>[n, vocab]</c> of the last <see cref="DraftBlock"/> (a download; only the non-greedy path asks).</summary>
    internal float[] DownloadDraftLogits(int rows)
    {
        long n = (long)rows * _vocab;
        if (!_dsd!.LogitsF16)
        {
            var all = new float[n];
            _device.Download(_dsd.Logits, all);
            return all;
        }
        var raw = new float[(n + 1) / 2];
        _device.Download(_dsd.Logits, raw);
        var bits = MemoryMarshal.Cast<float, ushort>(raw.AsSpan());
        var res = new float[n];
        for (long i = 0; i < n; i++) res[i] = (float)BitConverter.UInt16BitsToHalf(bits[(int)i]);
        return res;
    }

    /// <summary>Drafter logits storage: <c>DOTLLM_DSPARK_LOGITS=f32|f16</c>, default F16 (#2046: measured argmax-identical to F32 on the oracle rounds, see docs/INKLING2.md section 16).</summary>
    private static bool DraftLogitsF16FromEnv()
    {
        string? e = Environment.GetEnvironmentVariable("DOTLLM_DSPARK_LOGITS");
        if (string.IsNullOrEmpty(e)) return true;
        return e.ToLowerInvariant() switch { "f16" => true, "f32" => false, _ => throw new ArgumentException($"DOTLLM_DSPARK_LOGITS='{e}': expected f16 or f32.") };
    }

    /// <summary>True when the drafter's logits are stored as F16.</summary>
    internal bool DraftLogitsF16 => _dsd?.LogitsF16 ?? DraftLogitsF16FromEnv();

    /// <summary>Test/A-B hook: re-creates the drafter's logits buffer in the other storage format (the drafter must be idle).</summary>
    internal void SetDraftLogitsF16(bool f16)
    {
        var d = _dsd ?? throw new InvalidOperationException("No DSpark drafter attached.");
        if (d.LogitsF16 == f16) return;
        d.Owned.Remove(d.Logits);
        d.Logits.Dispose();
        d.Bytes -= (long)(d.Spec.BlockSize - 1) * _vocab * (d.LogitsF16 ? 2 : 4);
        _deviceBytes -= (long)(d.Spec.BlockSize - 1) * _vocab * (d.LogitsF16 ? 2 : 4);
        d.LogitsF16 = f16;
        long nb = (long)(d.Spec.BlockSize - 1) * _vocab * (f16 ? 2 : 4);
        d.Logits = _device.Allocate(nb);
        d.Owned.Add(d.Logits);
        d.Bytes += nb; _deviceBytes += nb;
    }

    /// <summary>Bytes of the drafter's logits buffer.</summary>
    internal long DraftLogitsBytes => _dsd?.Logits.Size ?? 0;

    internal VulkanDevice.Buffer? DSparkFrnnBuffer(VulkanInkling2DSparkState st) => st.Frnn;
}

/// <summary>
/// Per-sequence DSpark state of the Vulkan Inkling 2 model (issue #2046), driven through <see cref="IMtpState"/> like the CPU's: a target forward carrying it captures the tap rows and
/// injects them into the drafter's context K/V on the device; <see cref="NextDraft"/> drafts the whole block on the first call of a round and serves it afterwards.
/// </summary>
internal sealed class VulkanInkling2DSparkState : IMtpState
{
    private readonly VulkanInkling2TransformerModel _m;
    private readonly Inkling2DSparkSpec _spec;
    private readonly int _kvDim, _ringPositions;
    private readonly int[] _frnnPos = new int[16];
    private readonly int[] _draft;
    private int _length, _draftStart = -1, _draftCount;
    private float[]? _logitsHost;
    private bool _disposed;

    internal readonly VulkanDevice.Buffer[] K, V;
    internal readonly int[] Capacity;
    internal VulkanDevice.Buffer Frnn { get; }

    /// <summary>True once the drafter cannot follow the sequence (its global K/V would not fit in free device memory, or a gap in the injected positions); it then drafts garbage that the verify rejects.</summary>
    internal bool Unusable { get; private set; }
    internal string? UnusableReason { get; private set; }

    internal VulkanInkling2DSparkState(VulkanInkling2TransformerModel m, int ringPositions, int globalInitialPositions = 0)
    {
        _m = m;
        _spec = m.DSparkSpec ?? throw new InvalidOperationException("No DSpark drafter attached.");
        _kvDim = _spec.KvHeads * _spec.HeadDim;
        _ringPositions = ringPositions;
        _draft = new int[_spec.BlockSize - 1];
        Array.Fill(_frnnPos, -1);
        var dev = m.DeviceForTests;
        K = new VulkanDevice.Buffer[_spec.Layers]; V = new VulkanDevice.Buffer[_spec.Layers]; Capacity = new int[_spec.Layers];
        try
        {
            for (int i = 0; i < K.Length; i++)
            {
                int cap = _spec.Local[i] ? ringPositions : Math.Min(m.Config.MaxSequenceLength, globalInitialPositions > 0 ? globalInitialPositions : 1024);
                K[i] = dev.AllocateDeviceLocal((long)cap * _kvDim * 2); V[i] = dev.AllocateDeviceLocal((long)cap * _kvDim * 2); Capacity[i] = cap;
            }
            Frnn = dev.AllocateDeviceLocal(16L * _spec.Hidden * 2);
        }
        catch { Dispose(); throw; }
    }

    /// <summary>Positions the ring layers keep.</summary>
    internal int RingPositions => _ringPositions;

    /// <summary>Ring size of drafter layer <paramref name="layer"/> (0 = linear, the global layer).</summary>
    internal int RingFor(int layer) => _spec.Local[layer] ? _ringPositions : 0;

    /// <summary>Resident bytes of the context K/V and the f_rnn ring.</summary>
    internal long KvBytes { get { long b = Frnn.Size; for (int i = 0; i < K.Length; i++) b += K[i].Size + V[i].Size; return b; } }

    /// <inheritdoc/>
    public int CurrentLength => _length;
    /// <inheritdoc/>
    public ReadOnlySpan<float> CapturedHiddenRows => ReadOnlySpan<float>.Empty;
    /// <inheritdoc/>
    public int CapturedRowCount => 0;
    /// <inheritdoc/>
    public int HiddenSize => _spec.Hidden;

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        // An unusable state stopped injecting, so its length lags the target's; the decoder still rolls it back to the round's anchor every round (#2046 review).
        if (Unusable) { _draftStart = -1; if (length >= 0 && length < _length) _length = length; return; }
        if (length < 0 || length > _length) throw new ArgumentOutOfRangeException(nameof(length));
        _length = length;
        _draftStart = -1;
    }

    /// <inheritdoc/>
    public void SeedFromCapturedRow(int rowIndex) { }

    internal bool HasFrnn(int pos) => pos >= 0 && _frnnPos[pos % 16] == pos;

    internal void NoteInjected(int basePos, int rows)
    {
        for (int r = 0; r < rows; r++) _frnnPos[(basePos + r) % 16] = basePos + r;
        _length = Math.Max(_length, basePos + rows);
        _draftStart = -1;
    }

    internal void NoteDrafted(int n) { _draftCount = n; }

    /// <summary>
    /// Checks the coming forward (T rows at <paramref name="basePos"/>) can be injected and grows the global layer's buffer, admitted against free device memory (never spilled).
    /// False (and <see cref="Unusable"/>) when it cannot.
    /// </summary>
    internal bool PrepareFor(int basePos, int rows)
    {
        if (Unusable) return false;
        if (basePos > _length) { MarkUnusable($"gap: the target is at position {basePos} but the drafter context ends at {_length}"); return false; }
        int need = basePos + rows;
        var dev = _m.DeviceForTests;
        for (int i = 0; i < K.Length; i++)
        {
            if (_spec.Local[i] || need <= Capacity[i]) continue;
            int next = (int)Math.Min(_m.Config.MaxSequenceLength, Math.Max(need, (long)Capacity[i] * 2));
            long bytes = (long)next * _kvDim * 2;
            if (!VulkanKvCache.AdmitGrowth(dev, 2 * bytes))
            {
                MarkUnusable($"the drafter's global-layer context K/V would need {2 * bytes / 1048576} MiB for {next} positions and the device does not have it free");
                return false;
            }
            var nk = dev.AllocateDeviceLocal(bytes); var nv = dev.AllocateDeviceLocal(bytes);
            long used = (long)Capacity[i] * _kvDim * 2;
            dev.CopyBufferRangeSynchronous(K[i], nk, 0, 0, (ulong)used);
            dev.CopyBufferRangeSynchronous(V[i], nv, 0, 0, (ulong)used);
            K[i].Dispose(); V[i].Dispose();
            K[i] = nk; V[i] = nv; Capacity[i] = next;
        }
        return true;
    }

    private void MarkUnusable(string why)
    {
        Unusable = true; UnusableReason = why;
        Console.Error.WriteLine($"[dspark] drafter disabled for this sequence: {why}. Decoding continues without useful drafts.");
    }

    /// <summary>The DSpark side of <c>ForwardMtpArgMax</c>: at the context end drafts the block anchored at <paramref name="tokenId"/>, else serves the cached block.</summary>
    internal int NextDraft(int tokenId, int position)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        // An unusable state stopped injecting, so its length lags the target's: it must answer every position with the anchor (rejected by the verify), never throw.
        if (Unusable) return tokenId;
        if (position == _length)
        {
            int n = Math.Clamp(_spec.DraftTokensPerRound, 1, _draft.Length);
            _logitsHost = null;
            if (Unusable || !_m.DraftBlock(this, tokenId, position, n, _draft.AsSpan(0, n)))
            {
                // No context to condition on: draft the anchor repeated (the verify rejects it; correctness is unaffected, the adaptive gate sees the acceptance fall).
                Array.Fill(_draft, tokenId);
                _draftCount = n;
            }
            _draftStart = position;
            return _draft[0];
        }
        int i = position - _draftStart;
        if (_draftStart < 0 || i < 0 || i >= _draftCount)
            throw new InvalidOperationException($"DSpark draft step at position {position} has no block (context length {_length}, block at {_draftStart}, {_draftCount} drafts).");
        return _draft[i];
    }

    /// <summary>The (softcapped) logits row of the draft at <paramref name="position"/> + 1, after <see cref="NextDraft"/>.</summary>
    internal ReadOnlySpan<float> DraftLogits(int position)
    {
        _logitsHost ??= _m.DownloadDraftLogits(_draftCount);
        int vocab = _m.Config.VocabSize;
        return new ReadOnlySpan<float>(_logitsHost, (position - _draftStart) * vocab, vocab);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var b in K) b?.Dispose();
        foreach (var b in V) b?.Dispose();
        Frnn?.Dispose();
    }
}
