using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Shared fixture and driver for the issue #1101 YaRN tests on the Qwen3.5/3.8 hybrid hosts (CPU oracle, Vulkan, CUDA).
/// </summary>
/// <remarks>
/// <para>
/// The fixture has Bonsai-2's attention shape: 256-wide heads with 64 rotary dims (partial rotary 0.25), so a host that ran
/// the YaRN ramp over the head width instead of the rotary dim would rotate differently. Two RoPE settings:
/// </para>
/// <list type="bullet">
/// <item><see cref="WriteDiscriminating"/>: base 1e4 and a 512-token native window, YaRN factor 4 (window 2048). HF's
/// correction range is (3, 16), so 13 of the 32 pairs sit on the ramp, and a prompt of a few hundred tokens turns those
/// pairs through whole radians. A wrong ramp, a wrong mscale or a missing YaRN path moves the logits by far more than the
/// cross-backend tolerance.</item>
/// <item><see cref="WriteBonsaiLike"/>: Bonsai-2's own RoPE (base 1e7, native 262,144, YaRN factor 4 → 1,048,576), run
/// cacheless at positions past 262,144 so the far positions are cheap.</item>
/// </list>
/// <para>
/// The trunk is teacher-forced (fixed tokens, not argmax) so a near-tie on one backend cannot fork the sequences.
/// </para>
/// </remarks>
internal static class HybridYarnParity
{
    /// <summary>Vocabulary size of the synthetic fixture.</summary>
    public const int Vocab = SyntheticQwen35HybridDenseMtpGguf.VocabSize;

    /// <summary>Native window of the discriminating fixture.</summary>
    public const int DiscriminatingNativeContext = 512;

    /// <summary>Native window of the Bonsai-like fixture (Bonsai-2's own).</summary>
    public const int BonsaiNativeContext = 262144;

    /// <summary>Cross-backend absolute tolerance for these tests (see the GPU parity tests for the measured gap).</summary>
    public const float AbsTol = 1e-3f;

    /// <summary>Cross-backend relative tolerance for these tests.</summary>
    public const float RelTol = 1e-3f;

    /// <summary>
    /// The smallest logit change a YaRN mistake must cause for the parity tests to see it: 10× <see cref="AbsTol"/>. The CPU tests
    /// check the fixture clears it for YaRN on/off and for the mscale alone.
    /// </summary>
    public const float SignalFloor = 10 * AbsTol;

    /// <summary>YaRN factor (Qwen3.8's).</summary>
    public const float Factor = 4.0f;

    /// <summary>Writes the discriminating fixture (base 1e4, native 512, rotary 64 of 256, with an MTP head).</summary>
    /// <param name="dir">Scratch directory.</param>
    /// <returns>The GGUF path.</returns>
    public static string WriteDiscriminating(string dir)
        => SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(dir, "yarn-discriminating.gguf"), withMtp: true, blockCount: 4,
            contextLength: DiscriminatingNativeContext, headDim: 256, ropeDim: 64, ropeFreqBase: 1e4f);

    /// <summary>Writes the Bonsai-like fixture (base 1e7, native 262,144, rotary 64 of 256, no MTP head).</summary>
    /// <param name="dir">Scratch directory.</param>
    /// <returns>The GGUF path.</returns>
    public static string WriteBonsaiLike(string dir)
        => SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(dir, "yarn-bonsai-like.gguf"), withMtp: false,
            contextLength: BonsaiNativeContext, headDim: 256, ropeDim: 64, ropeFreqBase: 1e7f);

    /// <summary>
    /// The model config as the server and <c>run</c> build it: GGUF metadata, then, with <paramref name="yarn"/>,
    /// <c>--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx &lt;native&gt;</c>.
    /// </summary>
    /// <param name="gguf">The opened fixture.</param>
    /// <param name="yarn">Whether to apply the YaRN override.</param>
    /// <returns>The config.</returns>
    /// <param name="attnFactor">Optional <c>--yarn-attn-factor</c>; <c>1 / 1.1386</c> cancels the mscale (a sensitivity probe).</param>
    public static ModelConfig Config(GgufFile gguf, bool yarn, float? attnFactor = null)
    {
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (!yarn)
            return config;
        return GgufModelConfigExtractor.ApplyRoPEOverride(config, new RoPEOverrideOptions
        {
            ScalingType = RoPEScalingType.YaRN,
            ScalingFactor = Factor,
            OrigMaxSeqLen = config.MaxSequenceLength,
            AttnFactor = attnFactor,
        });
    }

    /// <summary>Logits a cached run produced: the prefill's last row, each decode step, and each MTP draft step.</summary>
    /// <param name="Trunk">Prefill last row, then one row per decode step.</param>
    /// <param name="Mtp">One row per MTP draft step (empty without an MTP head).</param>
    public sealed record Result(List<float[]> Trunk, List<float[]> Mtp);

    /// <summary>
    /// Prefills <paramref name="promptLen"/> tokens at positions <c>0..promptLen-1</c> into <paramref name="kv"/>, decodes
    /// <paramref name="decodeSteps"/> teacher-forced tokens, then (when the model has an MTP head) drafts
    /// <paramref name="mtpSteps"/> steps.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="kv">A KV cache for at least <c>promptLen + decodeSteps</c> positions.</param>
    /// <param name="promptLen">Prompt tokens.</param>
    /// <param name="decodeSteps">Single-token decode steps after the prompt.</param>
    /// <param name="mtpSteps">MTP draft steps after the decode steps.</param>
    /// <returns>The logits.</returns>
    public static Result RunCached(IModel model, IKvCache kv, int promptLen, int decodeSteps, int mtpSteps)
    {
        using IMtpState? mtp = model.SupportsMtp ? model.CreateMtpState() : null;
        var trunk = new List<float[]>();
        int[] tokens = Enumerable.Range(0, promptLen).Select(PromptToken).ToArray();
        int[] positions = Enumerable.Range(0, promptLen).ToArray();
        using (ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv, adapter: null, mtp))
            trunk.Add(LastRow(logits));

        int pos = promptLen;
        for (int s = 0; s < decodeSteps; s++, pos++)
        {
            using ITensor logits = model.Forward([DecodeToken(s)], [pos], deviceId: -1, kv, adapter: null, mtp);
            trunk.Add(LastRow(logits));
        }

        var drafts = new List<float[]>();
        if (mtp is not null)
        {
            for (int s = 0; s < mtpSteps; s++)
            {
                using ITensor logits = model.ForwardMtp(mtp, DecodeToken(decodeSteps + s), pos + s);
                drafts.Add(LastRow(logits));
            }
        }
        return new Result(trunk, drafts);
    }

    /// <summary>
    /// Cacheless forward of <paramref name="length"/> tokens at positions <c>start..start+length-1</c>; returns every row.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="start">First position (past the native window for the far-position tests).</param>
    /// <param name="length">Tokens.</param>
    /// <returns>One logits row per token.</returns>
    public static List<float[]> RunUncached(IModel model, int start, int length)
    {
        int[] tokens = Enumerable.Range(0, length).Select(PromptToken).ToArray();
        int[] positions = Enumerable.Range(start, length).ToArray();
        using ITensor logits = model.Forward(tokens, positions, deviceId: -1);
        return AllRows(logits);
    }

    /// <summary>
    /// Asserts each row of <paramref name="actual"/> is within <c>abs + rel·|expected|</c> of <paramref name="expected"/>, and
    /// returns the largest |difference| seen.
    /// </summary>
    /// <param name="expected">Reference rows (the CPU oracle).</param>
    /// <param name="actual">Rows under test.</param>
    /// <param name="absTol">Absolute tolerance.</param>
    /// <param name="relTol">Relative tolerance.</param>
    /// <param name="label">Message label.</param>
    /// <returns>The largest absolute difference.</returns>
    public static float AssertClose(IReadOnlyList<float[]> expected, IReadOnlyList<float[]> actual, float absTol, float relTol, string label)
    {
        Assert.Equal(expected.Count, actual.Count);
        float worst = 0f;
        for (int r = 0; r < expected.Count; r++)
        {
            Assert.Equal(expected[r].Length, actual[r].Length);
            for (int c = 0; c < expected[r].Length; c++)
            {
                float diff = MathF.Abs(expected[r][c] - actual[r][c]);
                worst = MathF.Max(worst, diff);
                float bar = absTol + relTol * MathF.Abs(expected[r][c]);
                Assert.True(diff <= bar,
                    $"{label} row {r} col {c}: expected {expected[r][c]:F6} vs actual {actual[r][c]:F6} (|diff|={diff:E3} > {bar:E3})");
            }
        }
        return worst;
    }

    /// <summary>Largest |difference| between two equally shaped row lists.</summary>
    /// <param name="a">First rows.</param>
    /// <param name="b">Second rows.</param>
    /// <returns>The largest absolute difference.</returns>
    public static float MaxAbsDiff(IReadOnlyList<float[]> a, IReadOnlyList<float[]> b)
    {
        float worst = 0f;
        for (int r = 0; r < a.Count; r++)
            for (int c = 0; c < a[r].Length; c++)
                worst = MathF.Max(worst, MathF.Abs(a[r][c] - b[r][c]));
        return worst;
    }

    private static int PromptToken(int i) => (i * 7 + 3) % Vocab;

    private static int DecodeToken(int i) => (i * 5 + 1) % Vocab;

    private static unsafe float[] LastRow(ITensor logits)
    {
        int rows = logits.Shape[0];
        return new ReadOnlySpan<float>((void*)logits.DataPointer, rows * Vocab).Slice((rows - 1) * Vocab, Vocab).ToArray();
    }

    private static unsafe List<float[]> AllRows(ITensor logits)
    {
        int rows = logits.Shape[0];
        var all = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * Vocab);
        var list = new List<float[]>(rows);
        for (int r = 0; r < rows; r++)
            list.Add(all.Slice(r * Vocab, Vocab).ToArray());
        return list;
    }
}
