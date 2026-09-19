using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
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

        int[] cpuDraft = RunCpuDraft(path!, positions);
        int[] vkDraft = RunVulkanDraft(path!, spvDir, positions);

        _out.WriteLine($"cpu    draft tokens: {string.Join(",", cpuDraft)}");
        _out.WriteLine($"vulkan draft tokens: {string.Join(",", vkDraft)}");

        Assert.Equal(cpuDraft, vkDraft);
    }

    private static int[] RunCpuDraft(string path, int[] positions)
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

    private static int[] RunVulkanDraft(string path, string spvDir, int[] positions)
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

    /// <summary>Chains <see cref="DraftSteps"/> MTP steps through their own argmax, as the decoder does.</summary>
    private static int[] Draft(Func<(int token, int position), ITensor> forwardMtp, int vocabSize, int lastPosition)
    {
        var drafted = new int[DraftSteps];
        int token = PromptTokens[^1];
        for (int i = 0; i < DraftSteps; i++)
        {
            using ITensor logits = forwardMtp((token, lastPosition + i));
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)logits.DataPointer, vocabSize);
                int best = 0;
                for (int c = 1; c < vocabSize; c++)
                    if (span[c] > span[best]) best = c;
                token = best;
            }
            drafted[i] = token;
        }
        return drafted;
    }
}
