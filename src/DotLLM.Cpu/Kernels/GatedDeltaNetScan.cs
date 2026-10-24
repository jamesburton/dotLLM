using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Gated DeltaNet (GDN) recurrence kernel for Qwen3MoeHybrid models.
/// Scalar reference port of the delta-rule associative-memory update from
/// "Gated Linear Attention Transformers with Hardware-Efficient Training" (NVlabs, ICLR 2025,
/// arXiv:2412.06464) as implemented in llama.cpp's <c>ggml_gated_recurrence</c>.
/// </summary>
/// <remarks>
/// <para>
/// For each value head <c>vh</c> in <c>[0, NVHead)</c>, one decode step executes:
/// </para>
/// <code>
/// kh = vh % NKHead                     // K head: TILED broadcast (matches llama.cpp ggml_gated_delta_net)
/// S  ×= g                              // element-wise decay of [DState × DState] matrix
/// r   = S.T @ k[kh]                   // retrieve: what was previously associated with key k
/// d   = β × (v[vh] − r)               // prediction error, scaled by write gate
/// S  += outer(k[kh], d)               // delta-rule rank-1 update
/// y   = S.T @ q[kh] / √DState         // read output via query
/// </code>
/// <para>
/// <b>Caller responsibilities before <see cref="Execute"/>:</b> apply causal conv1d to K
/// (via <see cref="Conv1dCausal"/>), then L2-normalise per head
/// (via <see cref="L2NormalizeHeads"/>). The decay <c>g</c> and write gate <c>β</c>
/// must be pre-computed: <c>g = exp(softplus(α_proj + dt_bias) × A)</c>,
/// <c>β = sigmoid(β_proj)</c>.
/// </para>
/// </remarks>
public static class GatedDeltaNetScan
{
    /// <summary>
    /// Runs the GDN recurrence over <paramref name="seqLen"/> tokens, updating
    /// <paramref name="state"/> in place and writing to <paramref name="output"/>.
    /// </summary>
    /// <param name="state">
    /// Per-sequence associative-memory state, shape <c>[NVHead, DState, DState]</c>
    /// row-major (length <c>NVHead × DState²</c>). Updated in place.
    /// </param>
    /// <param name="q">Query vectors, shape <c>[seqLen, NKHead, DState]</c> row-major.</param>
    /// <param name="k">
    /// Key vectors, shape <c>[seqLen, NKHead, DState]</c> row-major. Must be L2-normalised
    /// per head before this call (see <see cref="L2NormalizeHeads"/>).
    /// </param>
    /// <param name="v">Value vectors, shape <c>[seqLen, NVHead, DState]</c> row-major.</param>
    /// <param name="g">
    /// Per-head decay scalars, shape <c>[seqLen, NVHead]</c>. Values in (0, 1] —
    /// the caller computes <c>exp(softplus(α_proj + dt_bias) × A)</c>.
    /// </param>
    /// <param name="beta">
    /// Per-head write-gate scalars, shape <c>[seqLen, NVHead]</c>. Values in [0, 1] —
    /// the caller computes <c>sigmoid(β_proj)</c>.
    /// </param>
    /// <param name="output">
    /// Output buffer, shape <c>[seqLen, NVHead, DState]</c> row-major. Overwritten.
    /// </param>
    /// <param name="nVHead">Number of value heads.</param>
    /// <param name="nKHead">Number of key heads. Must divide <paramref name="nVHead"/> evenly.</param>
    /// <param name="dState">Per-head state dimension (key and value share the same DState).</param>
    /// <param name="seqLen">Number of tokens to process.</param>
    /// <param name="rowSnapshots">
    /// Optional (issue #473): receives a copy of the whole <paramref name="state"/> after each of
    /// the first <paramref name="snapshotRows"/> tokens, token <c>t</c> at offset
    /// <c>t × NVHead × DState²</c>. Lets a speculative verify roll the recurrence back to any row
    /// without recomputing it.
    /// </param>
    /// <param name="snapshotRows">Rows to snapshot; 0 (default) records nothing.</param>
    /// <param name="stateSource">
    /// Optional (#840 fused checkpoint copy): when non-empty, the INITIAL state is read from this span instead of
    /// <paramref name="state"/> (whose prior content is ignored): token 0's decay pass computes
    /// <c>state = stateSource * g</c> instead of <c>state *= g</c>, which is bit-identical to copying the source into
    /// <paramref name="state"/> first and then decaying in place, minus the separate copy pass.
    /// </param>
    /// <param name="deltaRecord">
    /// Optional (#842): receives each value head's delta <c>d = beta * (v - S.T k)</c> (the vector of the rank-1 write
    /// <c>S += outer(k, d)</c>) for the first <paramref name="snapshotRows"/> tokens, token <c>t</c>, head <c>vh</c> at
    /// <c>(t * NVHead + vh) * DState</c>. With the decay <c>g</c> and the key <c>k</c> (already in the caller's hands) it is all
    /// <see cref="Replay"/> needs to rebuild the state after any of those tokens from the state BEFORE the chunk, instead of storing
    /// a full state per row. Independent of <paramref name="rowSnapshots"/>.
    /// </param>
    [SkipLocalsInit]
    public static void Execute(
        Span<float> state,
        ReadOnlySpan<float> q,
        ReadOnlySpan<float> k,
        ReadOnlySpan<float> v,
        ReadOnlySpan<float> g,
        ReadOnlySpan<float> beta,
        Span<float> output,
        int nVHead,
        int nKHead,
        int dState,
        int seqLen,
        Span<float> rowSnapshots = default,
        int snapshotRows = 0,
        ReadOnlySpan<float> stateSource = default,
        Span<float> deltaRecord = default)
    {
        if (nVHead <= 0) throw new ArgumentOutOfRangeException(nameof(nVHead));
        if (nKHead <= 0) throw new ArgumentOutOfRangeException(nameof(nKHead));
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead % nKHead != 0)
            throw new ArgumentException($"nVHead ({nVHead}) must be divisible by nKHead ({nKHead}).");

        // Per-vh → kh mapping is TILED (modulo), matching llama.cpp's ggml gated_delta_net op
        // (ggml-cpu/ops.cpp ggml_compute_forward_gated_delta_net_one_chunk: iq1 = iv1 % neq1,
        //  ik1 = iv1 % nek1). For Qwen3.6-A3B (NVHead=32, NKHead=16) this maps vh 0..15 to
        // kh 0..15 and vh 16..31 back to kh 0..15 (tiled), NOT vh/2 → kh (interleaved).
        // Using the wrong mapping silently produces garbage logits — verified against
        // llama-eval-callback per-tensor parity (attn_output diverges immediately when
        // wrong, matches when tiled).
        int statePerHead = dState * dState;
        int qkPerToken = nKHead * dState;
        int vPerToken = nVHead * dState;

        if (state.Length < (long)nVHead * statePerHead)
            throw new ArgumentException("state buffer too small.", nameof(state));
        if (q.Length < (long)seqLen * qkPerToken)
            throw new ArgumentException("q buffer too small.", nameof(q));
        if (k.Length < (long)seqLen * qkPerToken)
            throw new ArgumentException("k buffer too small.", nameof(k));
        if (v.Length < (long)seqLen * vPerToken)
            throw new ArgumentException("v buffer too small.", nameof(v));
        if (g.Length < (long)seqLen * nVHead)
            throw new ArgumentException("g buffer too small.", nameof(g));
        if (beta.Length < (long)seqLen * nVHead)
            throw new ArgumentException("beta buffer too small.", nameof(beta));
        if (output.Length < (long)seqLen * vPerToken)
            throw new ArgumentException("output buffer too small.", nameof(output));

        if ((uint)snapshotRows > (uint)seqLen)
            throw new ArgumentOutOfRangeException(nameof(snapshotRows));
        int stateLen = nVHead * statePerHead;
        bool recordStates = !rowSnapshots.IsEmpty;
        if (recordStates && rowSnapshots.Length < (long)snapshotRows * stateLen)
            throw new ArgumentException("rowSnapshots buffer too small.", nameof(rowSnapshots));
        bool recordDeltas = !deltaRecord.IsEmpty;
        if (recordDeltas && deltaRecord.Length < (long)snapshotRows * nVHead * dState)
            throw new ArgumentException("deltaRecord buffer too small.", nameof(deltaRecord));

        bool fromSource = !stateSource.IsEmpty;
        if (fromSource && stateSource.Length < (long)nVHead * statePerHead)
            throw new ArgumentException("stateSource too small.", nameof(stateSource));
        if (seqLen == 0)
        {
            if (fromSource) stateSource.Slice(0, stateLen).CopyTo(state);
            return;
        }

        // Temporary [DState] buffer reused across all (t, vh) iterations.
        // First used for "retrieved" (S.T @ k), then overwritten with delta (β*(v-r)).
        float[] tmpBuf = ArrayPool<float>.Shared.Rent(dState);
        try
        {
            for (int t = 0; t < seqLen; t++)
            {
                int qkOff = t * qkPerToken;
                int vOff = t * vPerToken;
                int gbOff = t * nVHead;

                for (int vh = 0; vh < nVHead; vh++)
                {
                    int kh = vh % nKHead;

                    ReadOnlySpan<float> qHead = q.Slice(qkOff + kh * dState, dState);
                    ReadOnlySpan<float> kHead = k.Slice(qkOff + kh * dState, dState);
                    ReadOnlySpan<float> vHead = v.Slice(vOff + vh * dState, dState);
                    float gHead = g[gbOff + vh];
                    float betaHead = beta[gbOff + vh];

                    Span<float> stateHead = state.Slice(vh * statePerHead, statePerHead);
                    // stateHead[row * dState + col]: row = key dim, col = value dim.

                    // 1. Decay: S_vh *= g_vh  (element-wise scalar on [DState, DState])
                    if (fromSource && t == 0)
                    {
                        ReadOnlySpan<float> srcHead = stateSource.Slice(vh * statePerHead, statePerHead);
                        for (int i = 0; i < statePerHead; i++)
                            stateHead[i] = srcHead[i] * gHead;
                    }
                    else
                    {
                        for (int i = 0; i < statePerHead; i++)
                            stateHead[i] *= gHead;
                    }

                    // 2. Retrieve: tmp[col] = Σ_row S[row,col] × k[row]  (= S.T @ k)
                    //    Row-outer traversal keeps state accesses sequential (cache-friendly).
                    Span<float> tmp = tmpBuf.AsSpan(0, dState);
                    tmp.Clear();
                    for (int row = 0; row < dState; row++)
                    {
                        float ki = kHead[row];
                        int rowBase = row * dState;
                        for (int col = 0; col < dState; col++)
                            tmp[col] += stateHead[rowBase + col] * ki;
                    }

                    // 3. Delta rule write:
                    //    tmp[col] = β × (v[col] − tmp[col])   // overwrite tmp with delta
                    //    S[row,col] += k[row] × tmp[col]       // rank-1 update
                    for (int col = 0; col < dState; col++)
                        tmp[col] = betaHead * (vHead[col] - tmp[col]);
                    if (recordDeltas && t < snapshotRows)
                        tmp.CopyTo(deltaRecord.Slice((t * nVHead + vh) * dState, dState));

                    for (int row = 0; row < dState; row++)
                    {
                        float ki = kHead[row];
                        int rowBase = row * dState;
                        for (int col = 0; col < dState; col++)
                            stateHead[rowBase + col] += ki * tmp[col];
                    }

                    // 4. Read output: out[col] = Σ_row S[row,col] × q[row] / √DState  (= S.T @ q / √d)
                    Span<float> outHead = output.Slice(vOff + vh * dState, dState);
                    outHead.Clear();
                    for (int row = 0; row < dState; row++)
                    {
                        float qi = qHead[row];
                        int rowBase = row * dState;
                        for (int col = 0; col < dState; col++)
                            outHead[col] += stateHead[rowBase + col] * qi;
                    }
                    float scale = 1f / MathF.Sqrt(dState);
                    for (int col = 0; col < dState; col++)
                        outHead[col] *= scale;
                }

                if (recordStates && t < snapshotRows)
                    state.Slice(0, stateLen).CopyTo(rowSnapshots.Slice(t * stateLen, stateLen));
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(tmpBuf);
        }
    }

    /// <summary>
    /// Rebuilds the state after the first <paramref name="rows"/> tokens of a chunk from the state BEFORE the chunk, using only the
    /// per-token decay, key and delta that <see cref="Execute"/> produced (#842): per token and head
    /// <c>S = S * g + outer(k, d)</c>. Performs exactly the floating-point operations of the scan's decay pass and rank-1 write
    /// (separate multiply and add, no FMA contraction), so the result is BIT-IDENTICAL to the state the scan leaves after that token,
    /// while the per-row cost of keeping it is <c>NVHead * (1 + 2 * DState)</c> floats instead of a whole <c>NVHead * DState^2</c> state.
    /// </summary>
    /// <param name="state">Receives the rebuilt state <c>[NVHead, DState, DState]</c> (prior content ignored when <paramref name="source"/> is given).</param>
    /// <param name="source">The state before the chunk; empty means <paramref name="state"/> already holds it.</param>
    /// <param name="k">The scan's L2-normalised keys <c>[tokens, NKHead * DState]</c>.</param>
    /// <param name="g">The scan's decays <c>[tokens, NVHead]</c>.</param>
    /// <param name="delta">The scan's recorded deltas <c>[tokens, NVHead, DState]</c> (<c>deltaRecord</c>).</param>
    /// <param name="nVHead">Value heads.</param>
    /// <param name="nKHead">Key heads (value head <c>vh</c> uses key head <c>vh % NKHead</c>).</param>
    /// <param name="dState">Per-head state dimension.</param>
    /// <param name="rows">Tokens to replay (&gt;= 1).</param>
    public static void Replay(Span<float> state, ReadOnlySpan<float> source, ReadOnlySpan<float> k, ReadOnlySpan<float> g,
                              ReadOnlySpan<float> delta, int nVHead, int nKHead, int dState, int rows)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        int per = dState * dState, stateLen = nVHead * per, qk = nKHead * dState;
        if (state.Length < stateLen) throw new ArgumentException("state too small.", nameof(state));
        if (!source.IsEmpty && source.Length < stateLen) throw new ArgumentException("source too small.", nameof(source));
        if (rows == 0)
        {
            if (!source.IsEmpty) source.Slice(0, stateLen).CopyTo(state);
            return;
        }
        if (k.Length < (long)rows * qk || g.Length < (long)rows * nVHead || delta.Length < (long)rows * nVHead * dState)
            throw new ArgumentException("replay inputs too small.");

        for (int t = 0; t < rows; t++)
        {
            for (int vh = 0; vh < nVHead; vh++)
            {
                int kh = vh % nKHead;
                float gHead = g[t * nVHead + vh];
                ReadOnlySpan<float> kHead = k.Slice(t * qk + kh * dState, dState);
                ReadOnlySpan<float> dHead = delta.Slice((t * nVHead + vh) * dState, dState);
                Span<float> sHead = state.Slice(vh * per, per);
                ReadOnlySpan<float> from = t == 0 && !source.IsEmpty ? source.Slice(vh * per, per) : sHead;
                for (int row = 0; row < dState; row++)
                    ReplayRow(from.Slice(row * dState, dState), sHead.Slice(row * dState, dState), dHead, kHead[row], gHead);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ReplayRow(ReadOnlySpan<float> src, Span<float> dst, ReadOnlySpan<float> d, float ki, float g)
    {
        int col = 0, n = dst.Length;
        if (Vector256.IsHardwareAccelerated)
        {
            var gv = Vector256.Create(g);
            var kv = Vector256.Create(ki);
            ref float sr = ref MemoryMarshal.GetReference(src);
            ref float dr = ref MemoryMarshal.GetReference(dst);
            ref float er = ref MemoryMarshal.GetReference(d);
            for (; col + Vector256<float>.Count <= n; col += Vector256<float>.Count)
            {
                // separate multiply / multiply / add: the scan's `S *= g` then `S += k * d`, rounded identically (no FMA)
                var scaled = Vector256.Multiply(Vector256.LoadUnsafe(ref sr, (nuint)col), gv);
                var write = Vector256.Multiply(kv, Vector256.LoadUnsafe(ref er, (nuint)col));
                Vector256.Add(scaled, write).StoreUnsafe(ref dr, (nuint)col);
            }
        }
        for (; col < n; col++)
        {
            float scaled = src[col] * g;
            float write = ki * d[col];
            dst[col] = scaled + write;
        }
    }

    /// <summary>
    /// In-place L2 normalisation applied independently to each <paramref name="dState"/>-element
    /// head slice. Call on the K tensor before <see cref="Execute"/> — the delta rule requires
    /// unit-norm keys for well-conditioned associative-memory writes.
    /// </summary>
    /// <param name="heads">
    /// Flat buffer of concatenated head vectors; length must be a multiple of
    /// <paramref name="dState"/>. Each <paramref name="dState"/>-element slice is normalised
    /// independently to unit norm.
    /// </param>
    /// <param name="dState">Dimension of each head vector.</param>
    /// <param name="eps">Epsilon added to the L2 norm for numerical stability (default 1e-6).</param>
    public static void L2NormalizeHeads(Span<float> heads, int dState, float eps = 1e-6f)
    {
        if (dState <= 0) throw new ArgumentOutOfRangeException(nameof(dState));
        int numHeads = heads.Length / dState;
        for (int h = 0; h < numHeads; h++)
        {
            Span<float> head = heads.Slice(h * dState, dState);
            float sumSq = 0f;
            for (int i = 0; i < dState; i++)
                sumSq += head[i] * head[i];
            float invNorm = 1f / (MathF.Sqrt(sumSq) + eps);
            for (int i = 0; i < dState; i++)
                head[i] *= invNorm;
        }
    }
}
