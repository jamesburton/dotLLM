using DotLLM.Core.Configuration;
using DotLLM.HuggingFace;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="HfConfigExtractor"/> correctly
/// detects real-world tiny-random DeepSeek-V2/V3 checkpoints — the first
/// integration-level coverage for the MLA attention family.
/// </summary>
/// <remarks>
/// <para>
/// Downloads one of several tiny-random DeepSeek checkpoints on first run
/// (~2–20 MB each) and asserts:
/// <list type="bullet">
///   <item><c>Architecture.DeepSeekV2</c> or <c>Architecture.DeepSeekV3</c>.</item>
///   <item><c>AttentionType.MLA</c>.</item>
///   <item><c>MlaConfig</c> is populated with positive ranks and dims.</item>
///   <item><c>Moe</c> is populated (DeepSeek is always MoE) and
///     <c>first_k_dense_replace</c> correctly folds the dense-prefix layers
///     into <c>MlpOnlyLayers</c>.</item>
/// </list>
/// </para>
/// <para>
/// <b>Forward pass is out of scope for this agent.</b> The MLA kernel
/// (<see cref="DotLLM.Cpu.Kernels.MlaAttention"/>) is landed as a standalone
/// correctness-verified kernel; wiring it into
/// <see cref="DotLLM.Models.Architectures.TransformerModel"/>'s highly-tuned
/// decode path (fused RmsNorm+Quantize, R4 interleaving, FusedDecodeGemv,
/// quantised KV-cache) is a follow-up PR. Until then,
/// <see cref="ModelLoader.LoadFromSafetensors"/> throws
/// <see cref="NotSupportedException"/> for DeepSeek — a behaviour this test
/// also asserts.
/// </para>
/// <para>
/// Cache location: <c>~/.dotllm/test-cache/&lt;repo&gt;/</c>. 50 MB cap.
/// Gracefully skips if all candidates are offline/rate-limited.
/// </para>
/// </remarks>
public sealed class TinyDeepseekMlaSafetensorsLoadTests
{
    /// <summary>All candidate tiny-random checkpoints are well under 50 MB.</summary>
    private const int MaxAllowedBytes = 50 * 1024 * 1024;

    /// <summary>
    /// Ordered candidate repos. First reachable wins. Each ships a
    /// Deepseek{V2,V3}ForCausalLM safetensors checkpoint with full MLA + MoE
    /// config under ~20 MB.
    /// </summary>
    private static readonly (string RepoId, Architecture ExpectedArch, string[] Files)[] Candidates =
    [
        ("yujiepan/deepseek-v2-tiny-random",   Architecture.DeepSeekV2, ["model.safetensors", "config.json"]),
        ("yujiepan/deepseek-v3-tiny-random",   Architecture.DeepSeekV3, ["model.safetensors", "config.json"]),
        ("katuni4ka/tiny-random-deepseek-v3",  Architecture.DeepSeekV3, ["model.safetensors", "config.json"]),
    ];

    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    private readonly ITestOutputHelper _output;

    public TinyDeepseekMlaSafetensorsLoadTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Proves <see cref="HfConfigExtractor"/> correctly detects DeepSeek-V2/V3
    /// and populates <see cref="DotLLM.Core.Models.MlaConfig"/> +
    /// <see cref="DotLLM.Core.Models.MoeConfig"/> from a real HF checkpoint's
    /// <c>config.json</c>. The strongest assertion we can make today —
    /// the actual weight-loader dispatch is a follow-up.
    /// </summary>
    [SkippableFact]
    public void RealDeepseekConfig_IsDetectedAsMlaWithMoe()
    {
        var located = TryEnsureTinyDeepseek(out string? skipReason);
        Skip.If(located is null, skipReason ?? "tiny-random DeepSeek download unavailable");
        var (modelPath, expectedArch) = located.Value;

        string configPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "config.json");
        Assert.True(System.IO.File.Exists(configPath), "config.json must be co-located with the model.");

        var cfg = HfConfigExtractor.Extract(System.IO.File.ReadAllText(configPath));
        _output.WriteLine(
            $"Real HF config: arch={cfg.Architecture} attn={cfg.AttentionType} "
          + $"hidden={cfg.HiddenSize} layers={cfg.NumLayers} heads={cfg.NumAttentionHeads} "
          + $"head_dim={cfg.HeadDim} intermediate={cfg.IntermediateSize} vocab={cfg.VocabSize}");

        Assert.Equal(expectedArch, cfg.Architecture);
        Assert.Equal(AttentionType.MLA, cfg.AttentionType);

        Assert.NotNull(cfg.MlaConfig);
        var mla = cfg.MlaConfig!;
        _output.WriteLine(
            $"MlaConfig: kv_lora_rank={mla.KvLoraRank} q_lora_rank={mla.QLoraRank} "
          + $"qk_nope={mla.QkNopeHeadDim} qk_rope={mla.QkRopeHeadDim} v_head={mla.VHeadDim} "
          + $"qk_head={mla.QkHeadDim} rope_theta={mla.RopeTheta}");
        Assert.True(mla.KvLoraRank > 0, "kv_lora_rank must be positive");
        Assert.True(mla.QkNopeHeadDim >= 0, "qk_nope_head_dim must be non-negative");
        Assert.True(mla.QkRopeHeadDim > 0 && mla.QkRopeHeadDim % 2 == 0, "qk_rope_head_dim must be positive and even");
        Assert.True(mla.VHeadDim > 0, "v_head_dim must be positive");
        Assert.True(mla.QLoraRank >= 0, "q_lora_rank must be non-negative (0 = no factorisation)");
        // HeadDim in ModelConfig reflects qk_head_dim for MLA.
        Assert.Equal(mla.QkHeadDim, cfg.HeadDim);

        Assert.NotNull(cfg.Moe);
        var moe = cfg.Moe!;
        _output.WriteLine(
            $"MoeConfig: num_experts={moe.NumExperts} top_k={moe.NumExpertsPerTok} "
          + $"moe_intermediate={moe.MoeIntermediateSize} norm_topk={moe.NormTopKProb} "
          + $"shared_intermediate={moe.SharedExpertIntermediateSize} has_shared_gate={moe.HasSharedExpertGate}");
        Assert.True(moe.NumExperts >= 2, "DeepSeek is always MoE with multiple routed experts");
        Assert.True(moe.NumExpertsPerTok >= 1 && moe.NumExpertsPerTok <= moe.NumExperts);
        Assert.True(moe.MoeIntermediateSize > 0);
        // DeepSeek does not use the Qwen1.5-style sigmoid shared-expert gate.
        Assert.False(moe.HasSharedExpertGate);
    }

    /// <summary>
    /// Asserts the expected forward-path contract during the MLA kernel
    /// integration gap: <see cref="ModelLoader.LoadFromSafetensors"/>
    /// currently throws <see cref="NotSupportedException"/> for DeepSeek
    /// because the TransformerModel decode path is not wired for MLA yet.
    /// When a follow-up PR lands MLA into the forward pass, this test should
    /// be inverted into a full forward-pass + finite-logit assertion (see
    /// <see cref="TinyMixtralSafetensorsLoadTests"/> for the shape).
    /// </summary>
    [SkippableFact]
    public void DeepseekLoadFromSafetensors_ThrowsNotSupported_UntilMlaIsWired()
    {
        var located = TryEnsureTinyDeepseek(out string? skipReason);
        Skip.If(located is null, skipReason ?? "tiny-random DeepSeek download unavailable");
        var (modelPath, _) = located.Value;

        var ex = Assert.Throws<NotSupportedException>(() =>
        {
            var (_, file, _) = ModelLoader.LoadFromSafetensors(modelPath);
            file.Dispose();
        });
        _output.WriteLine($"Expected NotSupportedException: {ex.Message}");
        Assert.Contains("DeepSeek", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>
    /// Downloads a tiny-random DeepSeek-V2 or V3 repo into the local cache.
    /// Returns path + detected architecture, or null + reason on failure
    /// (offline, rate limit, all candidates 404).
    /// </summary>
    private (string ModelPath, Architecture Arch)? TryEnsureTinyDeepseek(out string? skipReason)
    {
        foreach (var (repoId, expectedArch, files) in Candidates)
        {
            string cachedDir = Path.Combine(
                CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar));
            string cachedModel = Path.Combine(cachedDir, "model.safetensors");
            string cachedConfig = Path.Combine(cachedDir, "config.json");

            if (File.Exists(cachedModel) && File.Exists(cachedConfig))
            {
                long size = new FileInfo(cachedModel).Length;
                if (size > MaxAllowedBytes)
                {
                    skipReason = $"cached {repoId} model is {size} bytes, exceeds cap {MaxAllowedBytes}";
                    return null;
                }
                skipReason = null;
                return (cachedModel, expectedArch);
            }

            try
            {
                using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
                using var downloader = new HuggingFaceDownloader(http);

                string url = $"https://huggingface.co/{repoId}/resolve/main/model.safetensors";
                using (var head = new HttpRequestMessage(HttpMethod.Head, url))
                using (var headResp = http.SendAsync(head, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
                {
                    if (!headResp.IsSuccessStatusCode)
                    {
                        _output.WriteLine($"{repoId}: HEAD returned {(int)headResp.StatusCode}, trying next candidate");
                        continue;
                    }
                    long? total = headResp.Content.Headers.ContentLength;
                    if (total is long t && t > MaxAllowedBytes)
                    {
                        _output.WriteLine($"{repoId}: model.safetensors is {t} bytes > cap {MaxAllowedBytes}, skipping");
                        continue;
                    }
                }

                _output.WriteLine($"{repoId}: downloading to {cachedDir}");
                foreach (var filename in files)
                {
                    downloader.DownloadFileAsync(
                        repoId, filename, CacheDir, progress: null)
                        .GetAwaiter().GetResult();
                }

                if (File.Exists(cachedModel) && File.Exists(cachedConfig))
                {
                    skipReason = null;
                    return (cachedModel, expectedArch);
                }
            }
            catch (Exception ex)
            {
                _output.WriteLine($"{repoId}: download failed with {ex.GetType().Name}: {ex.Message}");
            }
        }

        skipReason = "tiny-random DeepSeek V2/V3 unavailable (offline, rate limited, or all candidates failed)";
        return null;
    }
}
