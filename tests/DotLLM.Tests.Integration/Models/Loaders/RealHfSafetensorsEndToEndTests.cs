using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors"/>
/// loads real HuggingFace checkpoints (not tiny-random) for architectures dotLLM
/// now claims to support: dense multi-shard transformers with fused qkv/gate-up
/// (Phi-3.5-mini) and Granite-3.x-family MoE with fused-per-expert
/// input_linear/output_linear tensors (Granite-3.0-3B-A800M-instruct). Each test
/// is gated on an env-var checkpoint path or a conventional
/// <c>C:/temp/dotllm-&lt;family&gt;/</c> directory; when neither resolves the test
/// skips gracefully so CI stays green.
/// </summary>
/// <remarks>
/// <para>
/// <b>Checkpoint sizes.</b> The real weights are not committed and not fetched
/// by CI. Expected footprints:
/// </para>
/// <list type="bullet">
///   <item><description><c>microsoft/Phi-3.5-mini-instruct</c> — ~7.6 GB, 2
///   safetensors shards, dense Llama-family with fused qkv_proj /
///   gate_up_proj.</description></item>
///   <item><description><c>ibm-granite/granite-3.0-3b-a800m-instruct</c> —
///   ~6.3 GB, 2 shards, 40 routed experts top-8, no shared expert,
///   fused per-layer input_linear [E, 2*I, H] / output_linear [E, H, I]
///   tensors.</description></item>
/// </list>
/// <para>
/// <b>To run locally.</b> Either place the checkpoint at the conventional path
/// or set the env var to the safetensors index JSON or its directory:
/// <code>
///   $env:DOTLLM_PHI35_CHECKPOINT_PATH = "C:/temp/dotllm-phi35-mini"
///   $env:DOTLLM_GRANITE3_CHECKPOINT_PATH = "C:/temp/dotllm-granite3-moe"
///   dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj `
///     --filter FullyQualifiedName~RealHfSafetensorsEndToEnd
/// </code>
/// </para>
/// </remarks>
public sealed class RealHfSafetensorsEndToEndTests
{
    private readonly ITestOutputHelper _output;

    public RealHfSafetensorsEndToEndTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ────────────────────────────────────────────────────────────────────
    // Phi-3.5-mini-instruct
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Phi35Mini_LoadsAndForwardsEndToEnd()
    {
        string? root = ResolveCheckpointRoot(
            envVar: "DOTLLM_PHI35_CHECKPOINT_PATH",
            conventional: "C:/temp/dotllm-phi35-mini");
        if (root is null)
        {
            _output.WriteLine(
                "[SKIP] Phi-3.5-mini checkpoint not found. Set DOTLLM_PHI35_CHECKPOINT_PATH "
                + "or place the snapshot at C:/temp/dotllm-phi35-mini/");
            return;
        }

        _output.WriteLine($"Root: {root}");

        var loadWatch = Stopwatch.StartNew();
        var (model, source, config) = ModelLoader.LoadFromSafetensors(root);
        loadWatch.Stop();

        try
        {
            _output.WriteLine(
                $"Load ({loadWatch.Elapsed.TotalMilliseconds:F1} ms): arch={config.Architecture} "
                + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers} "
                + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} "
                + $"head_dim={config.HeadDim} tied={config.TiedEmbeddings}");

            // Phi-3.5-mini-instruct: Phi3ForCausalLM, 32 layers, hidden=3072, 32 heads, vocab=32064
            Assert.True(
                config.Architecture == Architecture.Phi,
                $"Expected Phi architecture, got {config.Architecture}");
            Assert.Equal(32, config.NumLayers);
            Assert.Equal(3072, config.HiddenSize);
            Assert.Equal(32, config.NumAttentionHeads);

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            fwdWatch.Stop();

            _output.WriteLine(
                $"Forward ({fwdWatch.Elapsed.TotalSeconds:F2} s): shape=[{logits.Shape[0]}, {logits.Shape[1]}]");

            AssertFiniteLogits(logits, config.VocabSize);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Granite-3.0-3B-A800M-instruct
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Granite3Moe_LoadsAndForwardsEndToEnd()
    {
        string? root = ResolveCheckpointRoot(
            envVar: "DOTLLM_GRANITE3_CHECKPOINT_PATH",
            conventional: "C:/temp/dotllm-granite3-moe");
        if (root is null)
        {
            _output.WriteLine(
                "[SKIP] Granite-3 MoE checkpoint not found. Set DOTLLM_GRANITE3_CHECKPOINT_PATH "
                + "or place the snapshot at C:/temp/dotllm-granite3-moe/");
            return;
        }

        _output.WriteLine($"Root: {root}");

        var loadWatch = Stopwatch.StartNew();
        var (model, source, config) = ModelLoader.LoadFromSafetensors(root);
        loadWatch.Stop();

        try
        {
            _output.WriteLine(
                $"Load ({loadWatch.Elapsed.TotalMilliseconds:F1} ms): arch={config.Architecture} "
                + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers} "
                + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads}");
            if (config.Moe is not null)
            {
                _output.WriteLine(
                    $"MoE: num_experts={config.Moe.NumExperts} top_k={config.Moe.NumExpertsPerTok} "
                    + $"intermediate={config.Moe.MoeIntermediateSize} "
                    + $"shared_intermediate={config.Moe.SharedExpertIntermediateSize} "
                    + $"norm_topk_prob={config.Moe.NormTopKProb}");
            }

            // Granite-3.0-3B-A800M-instruct: GraniteMoeForCausalLM, 32 layers,
            // hidden=1536, 24 heads, 8 kv heads (GQA), 40 experts top-8,
            // intermediate_size=512, vocab=49155. No shared expert.
            Assert.Equal(Architecture.GraniteMoe, config.Architecture);
            Assert.Equal(32, config.NumLayers);
            Assert.NotNull(config.Moe);
            Assert.Equal(40, config.Moe!.NumExperts);
            Assert.Equal(8, config.Moe.NumExpertsPerTok);
            Assert.Null(config.Moe.SharedExpertIntermediateSize);

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            fwdWatch.Stop();

            _output.WriteLine(
                $"Forward ({fwdWatch.Elapsed.TotalSeconds:F2} s): shape=[{logits.Shape[0]}, {logits.Shape[1]}]");

            AssertFiniteLogits(logits, config.VocabSize);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Qwen2.5-0.5B (dense, byte-level BPE tokenizer, heavy GQA)
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Qwen25_0_5B_LoadsAndForwardsEndToEnd()
    {
        string? root = ResolveCheckpointRoot(
            envVar: "DOTLLM_QWEN25_CHECKPOINT_PATH",
            conventional: "C:/Users/james/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987");
        if (root is null)
        {
            _output.WriteLine(
                "[SKIP] Qwen2.5-0.5B checkpoint not found. Set DOTLLM_QWEN25_CHECKPOINT_PATH "
                + "or ensure the HF snapshot is present at the conventional path.");
            return;
        }

        _output.WriteLine($"Root: {root}");

        var loadWatch = Stopwatch.StartNew();
        var (model, source, config) = ModelLoader.LoadFromSafetensors(root);
        loadWatch.Stop();

        try
        {
            _output.WriteLine(
                $"Load ({loadWatch.Elapsed.TotalMilliseconds:F1} ms): arch={config.Architecture} "
                + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers} "
                + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads} "
                + $"head_dim={config.HeadDim} tied={config.TiedEmbeddings} "
                + $"sliding_window={config.SlidingWindowSize}");

            // Qwen2.5-0.5B: Qwen2ForCausalLM, 24 layers, hidden=896, 14 heads,
            // 2 kv heads (heavy GQA), vocab=151936, tied_embeddings=true,
            // sliding_window=32768, rope_theta=1e6.
            Assert.Equal(Architecture.Qwen, config.Architecture);
            Assert.Equal(24, config.NumLayers);
            Assert.Equal(896, config.HiddenSize);
            Assert.Equal(14, config.NumAttentionHeads);
            Assert.Equal(2, config.NumKvHeads);
            Assert.True(config.TiedEmbeddings);

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            fwdWatch.Stop();

            _output.WriteLine(
                $"Forward ({fwdWatch.Elapsed.TotalSeconds:F2} s): shape=[{logits.Shape[0]}, {logits.Shape[1]}]");

            AssertFiniteLogits(logits, config.VocabSize);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // TinyLlama-1.1B-Chat-v1.0 (small real Llama, cheap validation)
    // ────────────────────────────────────────────────────────────────────

    [Fact]
    public void TinyLlama_11B_LoadsAndForwardsEndToEnd()
    {
        string? root = ResolveCheckpointRoot(
            envVar: "DOTLLM_TINYLLAMA_CHECKPOINT_PATH",
            conventional: "C:/temp/dotllm-tinyllama");
        if (root is null)
        {
            _output.WriteLine(
                "[SKIP] TinyLlama-1.1B checkpoint not found. Set DOTLLM_TINYLLAMA_CHECKPOINT_PATH "
                + "or place the snapshot at C:/temp/dotllm-tinyllama/");
            return;
        }

        _output.WriteLine($"Root: {root}");

        var loadWatch = Stopwatch.StartNew();
        var (model, source, config) = ModelLoader.LoadFromSafetensors(root);
        loadWatch.Stop();

        try
        {
            _output.WriteLine(
                $"Load ({loadWatch.Elapsed.TotalMilliseconds:F1} ms): arch={config.Architecture} "
                + $"vocab={config.VocabSize} hidden={config.HiddenSize} layers={config.NumLayers} "
                + $"heads={config.NumAttentionHeads} kv_heads={config.NumKvHeads}");

            // TinyLlama-1.1B-Chat-v1.0: LlamaForCausalLM, 22 layers, hidden=2048,
            // 32 heads, 4 kv heads (GQA), vocab=32000.
            Assert.Equal(Architecture.Llama, config.Architecture);
            Assert.Equal(22, config.NumLayers);
            Assert.Equal(2048, config.HiddenSize);

            int[] tokenIds = [0, 1, 2];
            int[] positions = [0, 1, 2];

            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            fwdWatch.Stop();

            _output.WriteLine(
                $"Forward ({fwdWatch.Elapsed.TotalSeconds:F2} s): shape=[{logits.Shape[0]}, {logits.Shape[1]}]");

            AssertFiniteLogits(logits, config.VocabSize);
        }
        finally
        {
            model.Dispose();
            (source as IDisposable)?.Dispose();
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Helpers
    // ────────────────────────────────────────────────────────────────────

    private static string? ResolveCheckpointRoot(string envVar, string conventional)
    {
        string? env = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (ContainsSafetensorsCheckpoint(env)) return env;
        }
        if (ContainsSafetensorsCheckpoint(conventional)) return conventional;
        return null;
    }

    private static bool ContainsSafetensorsCheckpoint(string path)
    {
        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            return true;
        if (!Directory.Exists(path)) return false;
        // Skip if HF cache still has incomplete downloads (in-flight snapshot_download)
        string cacheDir = Path.Combine(path, ".cache", "huggingface", "download");
        if (Directory.Exists(cacheDir) && Directory.GetFiles(cacheDir, "*.incomplete").Length > 0)
            return false;
        if (File.Exists(Path.Combine(path, "model.safetensors.index.json")))
        {
            // Verify all shards referenced by the index actually exist
            try
            {
                string indexJson = File.ReadAllText(Path.Combine(path, "model.safetensors.index.json"));
                using var doc = System.Text.Json.JsonDocument.Parse(indexJson);
                if (doc.RootElement.TryGetProperty("weight_map", out var weightMap))
                {
                    var shards = new HashSet<string>(StringComparer.Ordinal);
                    foreach (var prop in weightMap.EnumerateObject())
                        shards.Add(prop.Value.GetString()!);
                    foreach (var shard in shards)
                        if (!File.Exists(Path.Combine(path, shard))) return false;
                    return true;
                }
            }
            catch { return false; }
        }
        if (File.Exists(Path.Combine(path, "model.safetensors"))) return true;
        if (Directory.GetFiles(path, "model-*-of-*.safetensors").Length > 0) return true;
        return false;
    }

    private unsafe void AssertFiniteLogits(ITensor logits, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        int total = seqLen * vocabSize;
        int finite = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        double sumSq = 0, sum = 0;
        var data = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
        for (int i = 0; i < total; i++)
        {
            float v = data[i];
            if (float.IsFinite(v))
            {
                finite++;
                if (v < min) min = v;
                if (v > max) max = v;
                sum += v;
                sumSq += (double)v * v;
            }
        }
        double mean = sum / total;
        double variance = sumSq / total - mean * mean;
        double stddev = Math.Sqrt(Math.Max(0, variance));
        _output.WriteLine(
            $"Logits: finite={finite}/{total} min={min:F3} max={max:F3} mean={mean:F4} stddev={stddev:F4}");
        Assert.Equal(total, finite);
        Assert.True(stddev > 0, "Logits have zero variance — degenerate output.");
    }
}
