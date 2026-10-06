using DotLLM.Core.Models;

namespace DotLLM.Engine.Embeddings;

/// <summary>
/// Reduces a per-token hidden-state matrix <c>[seqLen, hiddenSize]</c> to a single embedding
/// vector <c>[hiddenSize]</c>, and optionally L2-normalises it.
/// </summary>
/// <remarks>
/// <para>Semantics follow llama.cpp's <c>llm_graph_context::build_pooling</c>
/// (<c>src/llama-graph.cpp</c>), which is the authoritative reference for GGUF op semantics
/// per the project guide:</para>
/// <list type="bullet">
///   <item><description><b>Mean</b> — <c>build_inp_mean</c> fills the weight matrix with a
///     uniform <c>1/n_seq_tokens</c> for every token of the sequence, so the pooled vector is
///     the plain arithmetic mean. No attention-mask weighting is involved (dotLLM runs one
///     sequence per forward, so there is no padding to mask).</description></item>
///   <item><description><b>Cls</b> / <b>Last</b> — <c>build_inp_cls</c> selects a single row:
///     the token with the smallest position (Cls) or the largest position (Last).</description></item>
/// </list>
/// <para>The hidden state passed in must be the post-final-norm state
/// (<see cref="IEmbeddingModel.ForwardHidden"/>), matching llama.cpp's <c>result_norm</c>.</para>
/// </remarks>
public static class EmbeddingPooler
{
    /// <summary>
    /// Pools <paramref name="hidden"/> (row-major <c>[seqLen, hiddenSize]</c>) into
    /// <paramref name="destination"/> (<c>[hiddenSize]</c>).
    /// </summary>
    /// <param name="hidden">Per-token hidden states, row-major, length <c>seqLen * hiddenSize</c>.</param>
    /// <param name="seqLen">Number of tokens. Must be &gt; 0.</param>
    /// <param name="hiddenSize">Hidden dimension. Must be &gt; 0.</param>
    /// <param name="pooling">Pooling strategy.</param>
    /// <param name="destination">Output vector of length <paramref name="hiddenSize"/>.</param>
    /// <exception cref="ArgumentOutOfRangeException">Dimensions are non-positive.</exception>
    /// <exception cref="ArgumentException">Buffer sizes do not match the stated dimensions.</exception>
    /// <exception cref="NotSupportedException">
    /// <paramref name="pooling"/> is <see cref="PoolingType.None"/> or <see cref="PoolingType.Rank"/>.
    /// </exception>
    public static void Pool(
        ReadOnlySpan<float> hidden,
        int seqLen,
        int hiddenSize,
        PoolingType pooling,
        Span<float> destination)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(seqLen, 0);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(hiddenSize, 0);

        if (hidden.Length < (long)seqLen * hiddenSize)
            throw new ArgumentException(
                $"Hidden state buffer holds {hidden.Length} floats, expected at least {(long)seqLen * hiddenSize}.",
                nameof(hidden));
        if (destination.Length < hiddenSize)
            throw new ArgumentException(
                $"Destination holds {destination.Length} floats, expected at least {hiddenSize}.",
                nameof(destination));

        destination = destination[..hiddenSize];

        switch (pooling)
        {
            case PoolingType.Last:
                hidden.Slice((seqLen - 1) * hiddenSize, hiddenSize).CopyTo(destination);
                return;

            case PoolingType.Cls:
                hidden[..hiddenSize].CopyTo(destination);
                return;

            case PoolingType.Mean:
                {
                    // Accumulate in double to keep the sum well-conditioned for long sequences;
                    // the divisor is the uniform 1/n llama.cpp's build_inp_mean writes.
                    Span<double> acc = hiddenSize <= 1024 ? stackalloc double[hiddenSize] : new double[hiddenSize];
                    acc.Clear();
                    for (int t = 0; t < seqLen; t++)
                    {
                        var row = hidden.Slice(t * hiddenSize, hiddenSize);
                        for (int i = 0; i < hiddenSize; i++)
                            acc[i] += row[i];
                    }

                    double inv = 1.0 / seqLen;
                    for (int i = 0; i < hiddenSize; i++)
                        destination[i] = (float)(acc[i] * inv);
                    return;
                }

            case PoolingType.None:
                throw new NotSupportedException(
                    "Pooling type 'none' produces one vector per token, which the OpenAI embeddings "
                    + "response shape cannot represent. Choose last, mean or cls.");

            case PoolingType.Rank:
                throw new NotSupportedException(
                    "Pooling type 'rank' requires a reranking classification head, which dotLLM does not load.");

            default:
                throw new NotSupportedException($"Unknown pooling type '{pooling}'.");
        }
    }

    /// <summary>
    /// L2-normalises <paramref name="vector"/> in place (Euclidean / p=2, matching llama.cpp's
    /// default <c>--embd-normalize 2</c>).
    /// </summary>
    /// <remarks>
    /// Arithmetic matches <c>common_embd_normalize</c> (<c>common/common.cpp</c>) step for step:
    /// the sum of squares and the square root are accumulated in <c>double</c>, the reciprocal is
    /// then narrowed to <c>float</c> and applied as a multiply. A zero vector yields a zero
    /// vector, exactly as llama.cpp's <c>sum &gt; 0.0 ? 1.0/sum : 0.0f</c> does.
    /// </remarks>
    public static void L2Normalize(Span<float> vector)
    {
        double sum = 0.0;
        for (int i = 0; i < vector.Length; i++)
        {
            float sq = vector[i] * vector[i];   // float product, as in C++ (float * float)
            sum += sq;
        }

        sum = Math.Sqrt(sum);

        float norm = sum > 0.0 ? (float)(1.0 / sum) : 0.0f;

        for (int i = 0; i < vector.Length; i++)
            vector[i] *= norm;
    }

    /// <summary>
    /// Chooses the pooling strategy for a model.
    /// </summary>
    /// <param name="requested">Explicit caller request, or <c>null</c> to use the model default.</param>
    /// <param name="declared">
    /// The checkpoint's declared pooling type (GGUF <c>{arch}.pooling_type</c>), or <c>null</c>
    /// when the checkpoint does not declare one.
    /// </param>
    /// <remarks>
    /// <para>Precedence: explicit request → checkpoint declaration → <see cref="PoolingType.Last"/>.</para>
    /// <para>The final fallback is a <b>deliberate deviation</b> from llama.cpp, whose
    /// <c>hparams.pooling_type</c> defaults to <c>LLAMA_POOLING_TYPE_NONE</c> when the key is
    /// absent. <c>NONE</c> yields one vector per token and is not representable in an OpenAI
    /// embeddings response, so it cannot be a usable default for this endpoint. <c>Last</c> is the
    /// correct choice for causal decoders — the last token is the only position whose hidden state
    /// has attended to the whole sequence — and is what llama.cpp's own tooling requires callers to
    /// pass (<c>--pooling last</c>) when running a generative model through
    /// <c>llama-embedding</c>/<c>llama-server --embeddings</c>.</para>
    /// <para>A checkpoint that declares <c>NONE</c> or <c>RANK</c> is honoured, not silently
    /// rewritten: <see cref="Pool"/> then throws, and the server turns that into an explicit error.</para>
    /// </remarks>
    public static PoolingType Resolve(PoolingType? requested, PoolingType? declared)
        => requested ?? declared ?? PoolingType.Last;
}
