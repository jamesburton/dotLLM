using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Mamba-3 end-to-end <see cref="IModel"/> implementation backed by a loaded
/// <see cref="Mamba3Weights"/>. Wires the canonical <see cref="Mamba3Block"/>
/// into the standard Mamba-3 pipeline:
/// <c>embed → N × (norm + Mamba3Block + residual) → norm_f → lm_head</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Forward shapes.</b> Two flavours share the same implementation:
/// </para>
/// <list type="bullet">
///   <item><description>
///     <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int)"/> /
///     <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache)"/>
///     — one-shot prefill; the model allocates an ephemeral zero-initialised
///     <see cref="Mamba3State"/>, runs every layer, and disposes the state at
///     return.
///   </description></item>
///   <item><description>
///     <see cref="Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
///     — prefill-or-decode with persistent state. The caller owns a
///     <see cref="Mamba3State"/>; each layer reads its <c>ssm_state</c>
///     (<c>[n_head, head_dim, d_state]</c>) and <c>cum_angle</c>
///     (<c>[n_head, num_rope_angles]</c>) at entry and writes the updated
///     buffers back in place at exit, so subsequent calls resume the sequence.
///   </description></item>
/// </list>
/// <para>
/// <b>Weight ownership.</b> <see cref="Mamba3Weights"/> is owned by this model
/// and disposed on <see cref="Dispose"/>. The underlying
/// <see cref="SafetensorsFile"/> is not owned here — the caller holds it via
/// the <see cref="ModelLoader.LoadFromSafetensors(string, ThreadingConfig?)"/>
/// tuple return and disposes it after disposing the model. An opaque lifetime
/// anchor (<see cref="object"/>) is carried so the file can be kept alive by
/// consumers who only want to hold the model.
/// </para>
/// <para>
/// <b>Numeric layout.</b> Every tensor this model touches is F32 row-major.
/// This matches the Stage D2 <see cref="Mamba3WeightLoader"/> ingest path; a
/// bf16 / quantized path is a future stage.
/// </para>
/// </remarks>
public sealed unsafe class Mamba3TransformerModel : IModel
{
    private readonly Mamba3Weights _weights;
    private readonly Mamba3Config _m3;
    // Pooled per-Forward scratch — 64-byte-aligned unmanaged buffers sized to
    // the widest seqLen seen so far. Grown power-of-two on demand, freed on
    // Dispose. Shared across every layer of every Forward call.
    private readonly Mamba3ForwardScratch _scratch;
    // Lifetime anchor — prevents GC of the mmap-backed safetensors file while
    // weight pointers from _weights are in flight. Not null for any loaded
    // model. Consumers still must dispose the SafetensorsFile themselves.
#pragma warning disable IDE0052, CA1823
    private readonly object? _mmapAnchor;
#pragma warning restore IDE0052, CA1823
    private bool _disposed;

    /// <inheritdoc/>
    public ModelConfig Config { get; }

    /// <inheritdoc/>
    public long ComputeMemoryBytes => _scratch.AllocatedBytes;

    private Mamba3TransformerModel(ModelConfig config, Mamba3Weights weights, object? anchor)
    {
        Config = config;
        _weights = weights;
        _m3 = config.Mamba3Config
              ?? throw new ArgumentException(
                  "ModelConfig.Mamba3Config must be populated for Mamba3TransformerModel.",
                  nameof(config));
        _mmapAnchor = anchor;
        // Lazy-initial capacity: the first Forward call sizes the scratch to
        // its seqLen (rounded up to the next power of two), and subsequent
        // Forwards at that-or-smaller length reuse the same allocation.
        _scratch = new Mamba3ForwardScratch(config, initialCapacity: 0);
    }

    /// <summary>
    /// Constructs a Mamba-3 model from an already-loaded
    /// <see cref="Mamba3Weights"/> bundle. The weights' underlying memory (mmap
    /// view, aligned scratch, etc.) must outlive the returned model. If the
    /// weights wrap a <see cref="SafetensorsFile"/>, pass it as
    /// <paramref name="lifetimeAnchor"/> to prevent GC from collecting it
    /// prematurely.
    /// </summary>
    /// <param name="config">Model config with <see cref="ModelConfig.Mamba3Config"/> populated.</param>
    /// <param name="weights">Loaded Mamba-3 weights.</param>
    /// <param name="lifetimeAnchor">Optional strong reference to keep alive (typically the safetensors file).</param>
    /// <returns>A ready-to-forward model instance. Caller owns disposal.</returns>
    public static Mamba3TransformerModel FromLoadedWeights(
        ModelConfig config, Mamba3Weights weights, object? lifetimeAnchor = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(weights);
        if (config.Architecture != Architecture.Mamba3)
            throw new ArgumentException(
                $"Mamba3TransformerModel requires Architecture.Mamba3, got {config.Architecture}.",
                nameof(config));
        if (weights.Report.HasMissingRequired)
            throw new InvalidDataException(
                $"Mamba-3 weights are incomplete ({weights.Report.MissingRequiredCount} required tensors missing). "
                + "Inspect Mamba3Weights.Report.Problems before attempting a forward pass.");
        return new Mamba3TransformerModel(config, weights, lifetimeAnchor);
    }

    /// <summary>
    /// Loads a Mamba-3 model from an opened HuggingFace-convention safetensors
    /// file. The <paramref name="file"/> must remain alive for the lifetime of
    /// the returned model; it is held by a GC anchor here, but the caller still
    /// must dispose it after disposing the model.
    /// </summary>
    /// <param name="file">An opened safetensors file positioned at a Mamba-3 checkpoint.</param>
    /// <param name="config">Model config resolved via <see cref="Mamba3ConfigExtractor"/>.</param>
    /// <returns>A ready-to-forward Mamba-3 model.</returns>
    /// <exception cref="InvalidDataException">
    /// One or more required Mamba-3 tensors are missing or malformed in the
    /// safetensors file. Caller can introspect
    /// <c>Mamba3WeightLoader.Load(config, file).Report</c> for a pre-check.
    /// </exception>
    public static Mamba3TransformerModel LoadFromSafetensors(ISafetensorsTensorSource file, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(config);

        Mamba3Weights weights = Mamba3WeightLoader.Load(config, file);
        try
        {
            return FromLoadedWeights(config, weights, lifetimeAnchor: file);
        }
        catch
        {
            weights.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        => Forward(tokenIds, positions, deviceId, kvCache: null);

    /// <summary>
    /// Runs a prefill forward pass over <paramref name="tokenIds"/>. Returns a
    /// newly allocated logits tensor of shape <c>[seqLen, vocab_size]</c>.
    /// </summary>
    /// <param name="tokenIds">Input token IDs (prefill only — all prompt tokens).</param>
    /// <param name="positions">
    /// Position indices, same length as <paramref name="tokenIds"/>. Validated
    /// against <see cref="ModelConfig.MaxSequenceLength"/>. The Mamba-3
    /// recurrence encodes position implicitly through <c>cum_angle</c>
    /// accumulation, so <paramref name="positions"/> is not consumed by the
    /// block forward — the array is carried for interface parity with
    /// <see cref="IModel"/>.
    /// </param>
    /// <param name="deviceId">Target device for the output tensor (-1 for CPU).</param>
    /// <param name="kvCache">
    /// Unused — Mamba-3 maintains an SSM state, not a KV-cache. Passed as
    /// non-null when the caller reuses a shared <see cref="IKvCache"/>
    /// variable; the value is silently ignored here.
    /// </param>
    /// <returns>Logits <c>[seqLen, vocab_size]</c>.</returns>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, IKvCache? kvCache)
    {
        _ = kvCache; // Mamba-3 uses SSM state, not KV-cache. Ignore.
        // Ephemeral state: allocate, run, dispose. Equivalent to a fresh sequence
        // — zero SSM state and zero cum_angle at entry — exactly the previous
        // prefill-only behaviour.
        using var scratch = new Mamba3State(Config);
        return ForwardCore(tokenIds, positions, deviceId, scratch);
    }

    /// <summary>
    /// Runs a forward pass that reads and writes a persistent
    /// <see cref="Mamba3State"/>. Enables prefill-then-decode sequences: the
    /// state is updated in place by every layer, so a subsequent call resumes
    /// where the previous one left off. The first call on a freshly constructed
    /// (or <see cref="Mamba3State.Reset"/>ed) state is equivalent to a
    /// one-shot prefill over the same tokens.
    /// </summary>
    /// <param name="tokenIds">Input token IDs for this step (prefill chunk or single decode token).</param>
    /// <param name="positions">Position indices, same length as <paramref name="tokenIds"/>.</param>
    /// <param name="deviceId">Target device for the output tensor (-1 for CPU).</param>
    /// <param name="state">
    /// Persistent per-layer recurrent state. Allocated via
    /// <c>new Mamba3State(config)</c> by the caller; the model reads each
    /// layer's <c>ssm_state</c> / <c>cum_angle</c> at entry and writes the
    /// updated buffers at exit. Must have been constructed from a
    /// <see cref="ModelConfig"/> with matching layer count and Mamba-3 dims.
    /// </param>
    /// <returns>Logits <c>[seqLen, vocab_size]</c> for every token in this call.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="state"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="state"/> does not match this model's layer / head / d_state dims.
    /// </exception>
    [SkipLocalsInit]
    public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                           int deviceId, Mamba3State state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.NumLayers != Config.NumLayers)
            throw new ArgumentException(
                $"Mamba3State has {state.NumLayers} layers but model has {Config.NumLayers}.",
                nameof(state));
        int expectedSsm = _m3.NumHeads * _m3.HeadDim * _m3.StateSize;
        int expectedCum = _m3.NumHeads * _m3.NumRopeAngles;
        if (state.SsmStateElementsPerLayer != expectedSsm)
            throw new ArgumentException(
                $"Mamba3State SSM layout mismatch: state has {state.SsmStateElementsPerLayer} "
                + $"elements/layer, model expects {expectedSsm}.", nameof(state));
        if (state.CumAngleElementsPerLayer != expectedCum)
            throw new ArgumentException(
                $"Mamba3State cum_angle layout mismatch: state has {state.CumAngleElementsPerLayer} "
                + $"elements/layer, model expects {expectedCum}.", nameof(state));

        return ForwardCore(tokenIds, positions, deviceId, state);
    }

    [SkipLocalsInit]
    private ITensor ForwardCore(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions,
                                int deviceId, Mamba3State state)
    {
        int seqLen = tokenIds.Length;
        if (seqLen == 0 || seqLen != positions.Length)
            throw new ArgumentException("tokenIds and positions must have equal, non-zero length.");

        int maxSeq = Config.MaxSequenceLength;
        for (int i = 0; i < positions.Length; i++)
        {
            if ((uint)positions[i] >= (uint)maxSeq)
                throw new ArgumentOutOfRangeException(nameof(positions),
                    $"Position {positions[i]} at index {i} exceeds max sequence length {maxSeq}.");
        }

        int hiddenSize = Config.HiddenSize;
        int vocabSize = Config.VocabSize;
        int numLayers = Config.NumLayers;
        int nHead = _m3.NumHeads;
        int headDim = _m3.HeadDim;
        int dState = _m3.StateSize;
        int dInner = _m3.DInner;
        int numBcHeads = _m3.NumGroups;
        int numRopeAngles = _m3.NumRopeAngles;
        int mimoRank = _m3.MimoRank;
        bool isMimo = _m3.IsMimo;
        int effectiveRank = isMimo ? mimoRank : 1;
        int bcPerToken = dState * numBcHeads * effectiveRank;
        int dInProj = 2 * dInner + 2 * bcPerToken + 3 * nHead + numRopeAngles;
        float aFloor = _m3.AFloor;
        float eps = Config.NormEpsilon;

        // --- Scratch (managed F32, one per-call allocation per buffer) ---
        float[] hidden = new float[seqLen * hiddenSize];
        float[] residual = new float[seqLen * hiddenSize];
        float[] normOut = new float[seqLen * hiddenSize];
        float[] blockOut = new float[seqLen * hiddenSize];
        // Per-layer SSM state and cum_angle: ephemeral (prefill-only, zeroed).
        // Allocated per-layer inside the loop so large models don't hold
        // num_layers * state simultaneously. Each of shape [H, P, N] / [H, S].

        // 1. EMBEDDING LOOKUP — gather rows from [vocab, hidden].
        EmbeddingLookup(tokenIds, hidden, hiddenSize, vocabSize);

        // 2. LAYERS — pre-norm residual, one Mamba3Block per layer.
        for (int layer = 0; layer < numLayers; layer++)
        {
            ref readonly var lw = ref _weights.Layers[layer];

            // Snapshot residual and pre-norm into normOut (per-token RMSNorm).
            new Span<float>(hidden).CopyTo(residual);
            float[] normWeight = SpanFromHandle(lw.Norm, hiddenSize).ToArray();
            for (int t = 0; t < seqLen; t++)
            {
                RmsNorm.Execute(
                    new ReadOnlySpan<float>(hidden, t * hiddenSize, hiddenSize),
                    normWeight, eps,
                    new Span<float>(normOut, t * hiddenSize, hiddenSize));
            }

            // Per-layer SSM state & cum_angle & k_state & v_state — read/written
            // in place on the caller's persistent state. A freshly constructed
            // (or Reset()ed) state is all-zero, reproducing the prior
            // prefill-only behaviour (no chunk-boundary adjustment applied).
            // k_state + v_state close the canonical shifted_γ lookahead gap at
            // chunk edges — see Mamba3Block.Forward doc for the math.
            Span<float> ssmState = state.SsmState(layer);
            Span<float> cumAngle = state.CumAngle(layer);
            Span<float> kState = state.KState(layer);
            Span<float> vState = state.VState(layer);

            ReadOnlySpan<float> inProj = SpanFromHandle(lw.InProj, dInProj * hiddenSize);
            ReadOnlySpan<float> outProj = SpanFromHandle(lw.OutProj, hiddenSize * dInner);
            ReadOnlySpan<float> dtBias = SpanFromHandle(lw.DtBias, nHead);
            ReadOnlySpan<float> bNormW = SpanFromHandle(lw.BNorm, dState);
            ReadOnlySpan<float> cNormW = SpanFromHandle(lw.CNorm, dState);
            // B_bias / C_bias handle shape [H, {1 or R}, N] — element count == H·{1 or R}·N.
            ReadOnlySpan<float> bBias = SpanFromHandle(lw.BBias, nHead * effectiveRank * dState);
            ReadOnlySpan<float> cBias = SpanFromHandle(lw.CBias, nHead * effectiveRank * dState);
            ReadOnlySpan<float> dSkip = SpanFromHandle(lw.D, nHead);

            if (isMimo)
            {
                // Canonical MIMO has mimo_z / mimo_o tensors in the checkpoint
                // (shape [H, R, P]). Stage D2's loader does not yet carry those
                // fields on Mamba3LayerWeights — see risk notes in
                // mamba3_canonical_pivot.md. Surface a clear error instead of
                // silently degrading to SISO.
                throw new NotSupportedException(
                    "Mamba-3 MIMO checkpoints are not yet supported end-to-end: "
                    + "Mamba3WeightLoader does not carry mimo_z / mimo_o tensors or the "
                    + "[H, R, N] B_bias / C_bias shape. Track via the MIMO follow-up.");
            }

            Mamba3Block.Forward(
                scratch: _scratch,
                u: normOut,
                inProjWeight: inProj,
                outProjWeight: outProj,
                dtBias: dtBias,
                bNormWeight: bNormW,
                cNormWeight: cNormW,
                bBias: bBias,
                cBias: cBias,
                d: dSkip,
                y: blockOut,
                ssmState: ssmState,
                cumAngle: cumAngle,
                kState: kState,
                vState: vState,
                seqLen: seqLen,
                dModel: hiddenSize,
                dInner: dInner,
                nHead: nHead,
                headDim: headDim,
                dState: dState,
                numBcHeads: numBcHeads,
                numRopeAngles: numRopeAngles,
                aFloor: aFloor,
                normEps: eps);

            // Residual add: hidden = residual + blockOut (per token).
            for (int t = 0; t < seqLen; t++)
            {
                Add.Execute(
                    new ReadOnlySpan<float>(residual, t * hiddenSize, hiddenSize),
                    new ReadOnlySpan<float>(blockOut, t * hiddenSize, hiddenSize),
                    new Span<float>(hidden, t * hiddenSize, hiddenSize));
            }
        }

        // 3. FINAL RMSNORM (in-place)
        float[] finalNormWeight = SpanFromHandle(_weights.FinalNorm, hiddenSize).ToArray();
        for (int t = 0; t < seqLen; t++)
        {
            var slice = new Span<float>(hidden, t * hiddenSize, hiddenSize);
            RmsNorm.Execute(
                new ReadOnlySpan<float>(hidden, t * hiddenSize, hiddenSize),
                finalNormWeight, eps,
                slice);
        }

        // 4. LM HEAD — logits[n, m] = hidden[n, k] @ lm_head[m, k]^T,
        //    M=vocab, K=hidden, N=seqLen.
        //    When tie_word_embeddings=true, LmHead aliases TokenEmbedding (same pointer).
        var shape = new TensorShape(seqLen, vocabSize);
        var result = UnmanagedTensor.Allocate(shape, DType.Float32, deviceId);
        try
        {
            float* lmHeadPtr = (float*)_weights.LmHead.Pointer;
            if (lmHeadPtr == null)
                throw new InvalidOperationException(
                    "Mamba-3 LM head pointer is null — weight loader reported the tensor as populated "
                    + "but the pointer is unset. Check Mamba3Weights.Report for diagnostics.");

            fixed (float* hiddenPtr = hidden)
            {
                MatMul.GemmF32(
                    a: lmHeadPtr, b: hiddenPtr, c: (float*)result.DataPointer,
                    m: vocabSize, k: hiddenSize, n: seqLen);
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
    /// Copies a token-embedding row into the hidden buffer for each input
    /// token. Mamba-3 checkpoints are F32; the loader rejects other dtypes at
    /// Stage D2 so this method assumes F32 element layout.
    /// </summary>
    private void EmbeddingLookup(ReadOnlySpan<int> tokenIds, float[] hidden,
                                 int hiddenSize, int vocabSize)
    {
        if (!_weights.TokenEmbedding.IsPopulated)
            throw new InvalidOperationException(
                "Mamba-3 token embedding is not populated — see Mamba3Weights.Report.");
        if (_weights.TokenEmbedding.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"Mamba-3 token embedding dtype {_weights.TokenEmbedding.SourceDType} is not yet supported.");

        float* embPtr = (float*)_weights.TokenEmbedding.Pointer;
        for (int t = 0; t < tokenIds.Length; t++)
        {
            int tokenId = tokenIds[t];
            if ((uint)tokenId >= (uint)vocabSize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds),
                    $"Token ID {tokenId} at position {t} is out of range [0, {vocabSize}).");

            float* src = embPtr + (long)tokenId * hiddenSize;
            new ReadOnlySpan<float>(src, hiddenSize)
                .CopyTo(new Span<float>(hidden, t * hiddenSize, hiddenSize));
        }
    }

    /// <summary>
    /// Wraps a <see cref="Mamba3TensorHandle"/>'s F32 pointer as a typed span
    /// of the given length.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ReadOnlySpan<float> SpanFromHandle(Mamba3TensorHandle h, int length)
    {
        if (!h.IsPopulated)
            throw new InvalidOperationException("Mamba-3 tensor handle is empty.");
        if (h.SourceDType != SafetensorsDType.F32)
            throw new NotSupportedException(
                $"Mamba-3 tensor dtype {h.SourceDType} is not yet supported (expected F32).");
        return new ReadOnlySpan<float>((void*)h.Pointer, length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _scratch.Dispose();
        _weights.Dispose();
    }
}
