using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan MTP against the <b>real</b> <c>Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0</c> checkpoint —
/// issue #435, acceptance criteria 2 and 4.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why the synthetic fixture is not enough.</b> <see cref="VulkanQwen3HybridDenseMtpTests"/>
/// covers the MTP graph, but its fixture is all-F32 and carries no <c>prism.hadamard.*</c> fold.
/// The real checkpoint ships <b>no</b> <c>nextn.embed_tokens</c> and <b>no</b>
/// <c>nextn.shared_head_head</c>, so its MTP head falls back to the trunk's <c>token_embd.weight</c>
/// (Hadamard-<i>latent</i>) and <c>output.weight</c> (Hadamard-<i>folded</i>). Those two rotations
/// are the ones the CPU reference was missing entirely, and fixing them took CPU draft acceptance
/// from ~0% to ~35% on this checkpoint. <b>Nothing except this test exercises them on Vulkan.</b>
/// Its PQ2_0 trunk and Q8_0 MTP block are also the only place the MTP host dispatches a quantized
/// matmul at all.
/// </para>
/// <para>
/// <b>Gated, not skipped-by-default-forever.</b> Resolves the checkpoint from
/// <c>DOTLLM_BONSAI2_MTP_GGUF</c> or the HF hub cache, and skips cleanly when absent so CI never
/// needs the 7.66 GB file. It is deliberately NOT in <c>scripts/test-435-vulkan-mtp.sh</c>'s default
/// set: it wants roughly ten GPU-lock minutes, which belongs in a coordinated validation pass
/// rather than a routine run.
/// </para>
/// <para>
/// <b>Fixed token IDs, no tokenizer.</b> What matters is that both backends see the <i>same</i>
/// input; an English prompt would add a tokenizer dependency without strengthening the comparison.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection("VulkanKernels")]
public sealed class VulkanBonsai2MtpRealCheckpointTests
{
    private const int DraftSteps = 4;

    // An arbitrary but fixed in-vocabulary prefix. Values are irrelevant; identity across the two
    // backends is the whole point.
    private static readonly int[] PromptTokens = [1, 785, 6722, 315, 9625, 374, 12095, 11, 323, 279, 6722, 315, 9856];

    private readonly ITestOutputHelper _out;

    public VulkanBonsai2MtpRealCheckpointTests(ITestOutputHelper output) => _out = output;

    private static string? FindCheckpoint()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_MTP_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string repo = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--ProCreations--Ternary-Bonsai-2-27B-MTP", "snapshots");
        if (!Directory.Exists(repo)) return null;

        foreach (string snapshot in Directory.EnumerateDirectories(repo))
        {
            string[] hits = Directory.GetFiles(snapshot, "*MTP*.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    /// <summary>
    /// Acceptance criterion 2: the same draft tokens, from the same checkpoint, at the same
    /// positions, on CPU and Vulkan. Because this checkpoint has no head-local <c>nextn</c> head
    /// tensors, passing requires the Vulkan MTP path to apply the trunk's inverse-Hadamard on the
    /// embedding lookup and the trunk's forward-Hadamard before the LM head — exactly the two
    /// rotations that were missing on CPU.
    /// </summary>
    /// <remarks>
    /// Draft token IDs are asserted <b>exactly</b>. This is an argmax over a 248320-wide vocabulary
    /// from identical weights; the two backends differ only in GEMM reduction order, which does not
    /// move an argmax unless the top-2 gap is at the noise floor. If this ever fails, print the
    /// top-2 gap before concluding it is a real divergence — a genuine basis error misses by a mile,
    /// a tie does not.
    /// </remarks>
    [SkippableFact]
    public void ForwardMtp_DraftTokens_MatchCpuOracle_OnRealBonsai2MtpCheckpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int[] positions = new int[PromptTokens.Length];
        for (int i = 0; i < positions.Length; i++) positions[i] = i;

        var cpu = RunCpuDraft(path!, positions);
        var vk = RunVulkanDraft(path!, spvDir, positions);

        _out.WriteLine($"cpu    draft tokens: {string.Join(",", cpu.Tokens)}");
        _out.WriteLine($"vulkan draft tokens: {string.Join(",", vk.Tokens)}");
        // Printed unconditionally so a failure diagnoses itself without a second GPU-lock window:
        // a top-2 gap at the reduction-order noise floor means the two backends agree and the
        // argmax merely tipped, which is parity; a large gap means a real basis/weight error. The
        // difference decides whether a red result is a bug or a tie, and you cannot tell from the
        // token lists alone.
        _out.WriteLine($"cpu    top-2 gaps:   {string.Join(",", cpu.Top2Gaps.Select(g => g.ToString("E3")))}");
        _out.WriteLine($"vulkan top-2 gaps:   {string.Join(",", vk.Top2Gaps.Select(g => g.ToString("E3")))}");

        Assert.Equal(cpu.Tokens, vk.Tokens);
    }

    /// <summary>
    /// Acceptance criteria 3 and 4 on the real checkpoint: the unmodified
    /// <see cref="MtpSpeculativeDecoder"/> driving the Vulkan model must emit exactly what plain
    /// greedy decode of the same Vulkan model emits, and the round counters give the Vulkan
    /// acceptance rate to compare against the CPU figure.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This is the only test that reads a verify-batch logit row on real weights, and that is
    /// the point.</b> The draft-token parity test above chains <c>ForwardMtp</c> off captured
    /// <i>pre-final-norm</i> hidden rows, so it never touches the trunk's LM head at
    /// <c>seqLen &gt; 1</c>. That path — the all-row head introduced for MTP, its forward Hadamard
    /// recorded over <c>headRows</c> rows at once, and the PQ2_0 matmul dispatched at the
    /// verify-batch <c>n</c> rather than at <c>n = 1</c> — is what the decoder's accept/reject
    /// decisions actually read. Get a verify row wrong and speculative decode stops matching greedy
    /// decode, which is exactly what this asserts.
    /// </para>
    /// <para>
    /// <b>Why greedy equivalence is evidence here but not everywhere.</b> It says nothing about
    /// whether the <i>draft head</i> is any good — the trunk verifies every token, so a noise draft
    /// head still yields correct output (the prior CPU Hadamard bug produced byte-identical text at
    /// ~0% true acceptance). It says a great deal about the <i>verify</i> rows, because those rows
    /// are the comparison basis: a wrong row 0..k-2 accepts or corrects against the wrong
    /// distribution and the emitted sequence diverges. Acceptance rate is the observable for the
    /// draft head; greedy equivalence is the observable for the verify rows. Both are printed.
    /// </para>
    /// <para>
    /// The emitted-per-round histogram is printed and a setup guard requires at least one round to
    /// have emitted more than one token — otherwise every round rejected at draft position 0, the
    /// verify batch never ran, and the assertion would be vacuous with respect to the path above.
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void DraftAndVerify_MatchesPlainGreedy_AndReportsAcceptanceRate_OnRealBonsai2MtpCheckpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int NewTokens = 10;
        const int K = 4;

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        Assert.True(model.SupportsMtp, "Checkpoint must carry an MTP head.");

        int cacheLen = PromptTokens.Length + NewTokens + K + 4;
        int[] promptPositions = new int[PromptTokens.Length];
        for (int i = 0; i < promptPositions.Length; i++) promptPositions[i] = i;

        // Speculative run.
        model.ResetSequenceState();
        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
        var generated = new List<int>(PromptTokens);
        int drafted = 0, emitted = 0;
        var emittedPerRound = new List<int>();

        using (var kv = model.CreateKvCache(cacheLen))
        using (IMtpState mtpState = model.CreateMtpState()!)
        {
            using (ITensor _ = model.Forward(PromptTokens, promptPositions, deviceId: -1, kv, adapter: null, mtpState)) { }

            // DraftAndVerify's `position` is the last generated token's OWN KV slot.
            int position = PromptTokens.Length - 1;
            int guard = 0;
            int[] outputBuffer = new int[K + 1];
            while (generated.Count - PromptTokens.Length < NewTokens && guard++ < NewTokens * 4)
            {
                var result = decoder.DraftAndVerify(
                    model, kv, mtpState, pipeline, generated, constraint: null,
                    position, vocabSize: config.VocabSize, numCandidates: K, outputBuffer);

                Assert.True(result.AcceptedCount > 0, "Every round must emit at least the corrected/bonus token.");
                drafted += result.DraftedCount;
                emitted += result.AcceptedCount;
                emittedPerRound.Add(result.AcceptedCount);

                for (int i = 0; i < result.AcceptedCount && generated.Count - PromptTokens.Length < NewTokens; i++)
                    generated.Add(outputBuffer[i]);
                position += result.AcceptedCount;
            }
        }

        List<int> speculative = generated.Skip(PromptTokens.Length).Take(NewTokens).ToList();

        // Plain greedy run, same model instance, fresh recurrent and KV state.
        model.ResetSequenceState();
        var plain = new List<int>();
        using (var kv = model.CreateKvCache(cacheLen))
        {
            int next;
            using (ITensor logits = model.Forward(PromptTokens, promptPositions, deviceId: -1, kv))
                next = LastRowArgMax(logits, config.VocabSize);

            for (int step = 0; step < NewTokens; step++)
            {
                plain.Add(next);
                if (plain.Count == NewTokens) break;
                using ITensor stepLogits = model.Forward([next], [PromptTokens.Length + step], deviceId: -1, kv);
                next = LastRowArgMax(stepLogits, config.VocabSize);
            }
        }

        double trueAcceptance = drafted == 0 ? 0 : (emitted - emittedPerRound.Count) / (double)drafted;
        _out.WriteLine($"speculative:  [{string.Join(",", speculative)}]");
        _out.WriteLine($"plain greedy: [{string.Join(",", plain)}]");
        _out.WriteLine($"rounds={emittedPerRound.Count} drafted={drafted} emitted={emitted} " +
                       $"reported_rate={(drafted == 0 ? 0 : emitted / (double)drafted):F4} " +
                       $"true_draft_acceptance={trueAcceptance:F4}");
        _out.WriteLine($"emitted-per-round: [{string.Join(",", emittedPerRound)}]");

        // Setup guard: without a round that emitted more than the always-emitted correction token,
        // the verify batch never ran and the equality below says nothing about the verify rows.
        Assert.Contains(emittedPerRound, n => n > 1);
        Assert.Equal(plain, speculative);
    }

    private static unsafe int LastRowArgMax(ITensor logits, int vocabSize)
    {
        int rows = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocabSize)
            .Slice((rows - 1) * vocabSize, vocabSize);
        int best = 0;
        for (int i = 1; i < vocabSize; i++)
            if (span[i] > span[best]) best = i;
        return best;
    }

    private sealed record DraftRun(int[] Tokens, float[] Top2Gaps);

    private static DraftRun RunCpuDraft(string path, int[] positions)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (Qwen3HybridDenseTransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, config);
        Assert.True(model.SupportsMtp, "Checkpoint must carry an MTP head.");

        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim,
            PromptTokens.Length + DraftSteps + 2);
        using var mtpState = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(PromptTokens, positions, deviceId: -1, kv, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        return Draft(tok => model.ForwardMtp(mtpState, tok.token, tok.position), config.VocabSize, positions[^1]);
    }

    private static DraftRun RunVulkanDraft(string path, string spvDir, int[] positions)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3HybridDenseTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        Assert.True(model.SupportsMtp, "Checkpoint must carry an MTP head.");

        using var kv = model.CreateKvCache(PromptTokens.Length + DraftSteps + 2);
        using var mtpState = model.CreateMtpState()!;
        using (ITensor _ = model.Forward(PromptTokens, positions, deviceId: -1, kv, adapter: null, mtpState)) { }
        Assert.Equal(PromptTokens.Length, mtpState.CapturedRowCount);
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        return Draft(tok => model.ForwardMtp(mtpState, tok.token, tok.position), config.VocabSize, positions[^1]);
    }

    /// <summary>
    /// Chains <see cref="DraftSteps"/> MTP steps through their own argmax, as the decoder does, and
    /// records the winning-margin (top-1 minus top-2) at each step.
    /// </summary>
    private static DraftRun Draft(Func<(int token, int position), ITensor> forwardMtp, int vocabSize, int lastPosition)
    {
        var drafted = new int[DraftSteps];
        var gaps = new float[DraftSteps];
        int token = PromptTokens[^1];
        for (int i = 0; i < DraftSteps; i++)
        {
            using ITensor logits = forwardMtp((token, lastPosition + i));
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
                int best = 0;
                float second = float.NegativeInfinity;
                for (int c = 1; c < vocabSize; c++)
                {
                    if (span[c] > span[best]) { second = span[best]; best = c; }
                    else if (span[c] > second) second = span[c];
                }
                gaps[i] = span[best] - second;
                token = best;
            }
            drafted[i] = token;
        }
        return new DraftRun(drafted, gaps);
    }
}
