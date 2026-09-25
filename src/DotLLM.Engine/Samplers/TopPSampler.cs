using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// Top-P (nucleus) sampling: keeps the smallest set of tokens whose cumulative probability
/// exceeds P, masking the rest to -infinity.
/// </summary>
/// <remarks>
/// <para>
/// <b>Tie-breaking is deterministic by token id.</b> At large vocabularies with near-uniform
/// logits, distinct logits routinely round to the <i>same</i> <see cref="float"/> probability after
/// softmax. <c>Array.Sort</c> is an unstable IntroSort that switches strategy by array size, so the
/// order of equal-probability entries — and therefore which token survives when the top-P cutoff
/// falls inside a tie — would otherwise be unspecified and could differ across runtime versions.
/// This implementation orders candidates by descending probability and, within a tie,
/// <i>ascending token id</i>, so a seeded run reproduces across environments.
/// </para>
/// <para>
/// The ordering is realised by sorting a single packed <see cref="ulong"/> key per token rather
/// than a (probability, index) array pair: the probability's raw bits occupy the high 32 bits and
/// the bitwise complement of the token id the low 32. Softmax outputs lie in <c>[0, 1]</c>, and the
/// IEEE-754 bit pattern of a non-negative <see cref="float"/> is monotonic when read as a
/// <see cref="uint"/>, so an ordinary ascending sort of the packed keys yields probability
/// ascending with the id descending inside each tie — i.e. the backwards (descending-probability)
/// walk below visits tied tokens lowest id first. Sorting one primitive array is also cheaper than
/// the paired key/item sort it replaces, so the stronger guarantee costs nothing.
/// </para>
/// </remarks>
public sealed class TopPSampler : ISamplerStep
{
    private readonly float? _topP;

    /// <summary>Creates a top-P step that reads from <see cref="SamplerContext"/>.</summary>
    public TopPSampler() { }

    /// <summary>Creates a self-configured top-P step.</summary>
    /// <param name="topP">Cumulative probability threshold (ignores context).</param>
    public TopPSampler(float topP) => _topP = topP;

    /// <inheritdoc/>
    [SkipLocalsInit]
    public void Apply(Span<float> logits, SamplerContext context)
    {
        float topP = _topP ?? context.TopP;
        if (topP >= 1.0f)
            return;

        int vocabSize = logits.Length;
        float[] rentedProbs = ArrayPool<float>.Shared.Rent(vocabSize);
        ulong[] rentedKeys = ArrayPool<ulong>.Shared.Rent(vocabSize);
        bool[] rentedKeep = ArrayPool<bool>.Shared.Rent(vocabSize);
        try
        {
            var probs = rentedProbs.AsSpan(0, vocabSize);
            var keys = rentedKeys.AsSpan(0, vocabSize);

            // Softmax to get probabilities
            TensorPrimitives.SoftMax(logits, probs);

            // Pack (probability, ~tokenId) so one ascending primitive sort orders by probability
            // ascending and, within equal probabilities, by token id descending.
            for (int i = 0; i < vocabSize; i++)
            {
                float p = probs[i];
                Debug.Assert(!float.IsNaN(p), "TopPSampler: probabilities must not contain NaN");
                keys[i] = ((ulong)BitConverter.SingleToUInt32Bits(p) << 32) | ~(uint)i;
            }

            Array.Sort(rentedKeys, 0, vocabSize);

            // Walk backwards (descending probability; ties ascending token id), accumulate until
            // we exceed topP.
            float cumulative = 0f;
            int keepStart = 0;
            for (int i = vocabSize - 1; i >= 0; i--)
            {
                cumulative += BitConverter.UInt32BitsToSingle((uint)(rentedKeys[i] >> 32));
                if (cumulative >= topP)
                {
                    keepStart = i; // keep [i, vocabSize)
                    break;
                }
            }

            // Build kept-indices set
            var keep = rentedKeep.AsSpan(0, vocabSize);
            keep.Clear();

            for (int i = keepStart; i < vocabSize; i++)
                keep[TokenIdOf(rentedKeys[i])] = true;

            for (int i = 0; i < vocabSize; i++)
            {
                if (!keep[i])
                    logits[i] = float.NegativeInfinity;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rentedProbs);
            ArrayPool<ulong>.Shared.Return(rentedKeys);
            ArrayPool<bool>.Shared.Return(rentedKeep);
        }
    }

    /// <summary>Recovers the token id packed into the low 32 bits of a sort key.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int TokenIdOf(ulong key) => (int)~(uint)key;
}
