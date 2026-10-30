using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Backend-agnostic drivers for the issue #966 long-context MTP tests: the exact TextGenerator shape
/// (batched prefill carrying the MTP state, then draft/verify/absorb rounds from the last token's own
/// slot) next to a plain one-token-at-a-time greedy decode of the same prompt.
/// </summary>
internal static class MtpLongContextHarness
{
    /// <summary>A deterministic non-degenerate prompt of <paramref name="length"/> tokens.</summary>
    public static int[] Prompt(int length, int vocab)
    {
        var ids = new int[length];
        for (int i = 0; i < length; i++)
            ids[i] = (int)(((long)i * 5 + 1 + i / 7) % vocab);
        return ids;
    }

    /// <summary>Plain greedy: batched prefill, then one token per forward. Returns <paramref name="newTokens"/> tokens.</summary>
    public static int[] PlainGreedy(IModel model, IKvCache kv, int[] prompt, int newTokens, int vocab)
    {
        var positions = Enumerable.Range(0, prompt.Length).ToArray();
        int token;
        using (ITensor t = model.Forward(prompt, positions, deviceId: -1, kv))
            token = ArgMax(LastRow(t, vocab));

        var result = new int[newTokens];
        for (int i = 0; i < newTokens; i++)
        {
            result[i] = token;
            if (i == newTokens - 1) break;
            using ITensor t = model.Forward([token], [prompt.Length + i], deviceId: -1, kv);
            token = ArgMax(LastRow(t, vocab));
        }
        return result;
    }

    /// <summary>
    /// MTP greedy: batched prefill with <paramref name="mtp"/>, then rounds of <paramref name="k"/>
    /// candidates until <paramref name="newTokens"/> tokens exist. Asserts nothing itself; the caller
    /// compares the returned tokens against <see cref="PlainGreedy"/>.
    /// </summary>
    public static int[] MtpGreedy(IModel model, IKvCache kv, IMtpState mtp, int[] prompt, int newTokens, int k, int vocab)
    {
        var positions = Enumerable.Range(0, prompt.Length).ToArray();
        var generated = new List<int>();
        using (ITensor t = model.Forward(prompt, positions, deviceId: -1, kv, adapter: null, mtp))
            generated.Add(ArgMax(LastRow(t, vocab)));

        var decoder = new MtpSpeculativeDecoder(greedy: true) { UseRecurrentRowSnapshots = false };
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        int position = prompt.Length;   // slot of generated[^1]
        var buffer = new int[k + 1];
        int guard = 0;
        while (generated.Count < newTokens && guard++ < newTokens * 4)
        {
            var result = decoder.DraftAndVerify(
                model, kv, mtp, pipeline, generated, constraint: null, position,
                vocabSize: vocab, numCandidates: k, buffer);
            if (result.AcceptedCount <= 0)
                throw new InvalidOperationException("A round must accept at least one token.");
            for (int i = 0; i < result.AcceptedCount && generated.Count < newTokens; i++)
                generated.Add(buffer[i]);
            position += result.AcceptedCount;
        }
        return generated.Take(newTokens).ToArray();
    }

    /// <summary>
    /// Batched prefill with <paramref name="mtp"/>, then <paramref name="steps"/> chained greedy draft
    /// steps from the last prompt token's follower; returns each step's full logits row. Unlike
    /// <see cref="MtpGreedy"/> this observes the head itself: verified output equals plain greedy no matter
    /// how bad the drafts are, so a corrupted head KV is only visible here.
    /// </summary>
    public static float[][] DraftLogits(IModel model, IKvCache kv, IMtpState mtp, int[] prompt, int steps, int vocab)
    {
        var positions = Enumerable.Range(0, prompt.Length).ToArray();
        int token;
        using (ITensor t = model.Forward(prompt, positions, deviceId: -1, kv, adapter: null, mtp))
            token = ArgMax(LastRow(t, vocab));

        var rows = new float[steps][];
        for (int i = 0; i < steps; i++)
        {
            using ITensor logits = model.ForwardMtp(mtp, token, prompt.Length + i);
            rows[i] = LastRow(logits, vocab);
            token = ArgMax(rows[i]);
        }
        return rows;
    }

    private static unsafe float[] LastRow(ITensor t, int vocab)
    {
        int rows = t.Shape[0];
        return new ReadOnlySpan<float>((float*)t.DataPointer + (long)(rows - 1) * vocab, vocab).ToArray();
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }
}
