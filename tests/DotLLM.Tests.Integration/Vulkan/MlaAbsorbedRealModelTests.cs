using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #913: real-model correctness of the absorbed-MLA Vulkan path (packed projections, latent KV cache, attention in the
/// latent space) on GLM-4.7-Flash Q4_K_M (<c>DOTLLM_GLM_GGUF</c> or the HF-hub copy). The CPU model is the oracle. The SAME file is
/// also loaded a second time with <c>DOTLLM_VK_MLA_LEGACY=1</c> (the switched-off control: F32-widened projections, expanded
/// K/V cache, scalar matmul) so the expanded path's own distance from the CPU is the yardstick the absorbed path must not exceed.
/// <para>
/// Four forward shapes exercise every kernel variant: pure decode (one row per forward: split-KV + merge), whole-prompt prefill,
/// 16-token chunked prefill (unsplit attention with history) and 3-token chunks (the small-chunk split path with causal masking
/// inside the chunk). Counters prove the absorbed kernels ran on the absorbed model and never on the control; the latent cache is
/// asserted to be much smaller than the expanded one. A model fed an expanded cache, or a cache-less decode, would be caught by
/// the decode-after-prefill comparison (the cache-less-MLA bug of #755 was invisible to PPL).
/// </para>
/// Take the GPU lock first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class MlaAbsorbedRealModelTests
{
    private readonly ITestOutputHelper _output;
    public MlaAbsorbedRealModelTests(ITestOutputHelper output) => _output = output;

    private static readonly string[] Prompts =
    [
        "The three primary colors are",
        "The capital of France is Paris, and the capital of Germany is",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) +",
        "In 1969, the Apollo 11 mission landed the first humans on the Moon. The commander of the mission was",
        // ~190 tokens: several 64-key tiles, causal ramp inside tiles, 16-token chunks with long history.
        string.Concat(Enumerable.Repeat("The history of the Roman Empire spans many centuries, during which the city grew from a small settlement on the banks of the Tiber into the capital of a vast realm. Its legions marched across Europe, and its roads linked provinces from Britain to Syria. ", 4)) + "In summary, the Roman Empire was",
        // ~500 tokens: split-KV decode with a deep cache.
        string.Concat(Enumerable.Repeat("Photosynthesis converts light energy into chemical energy stored in glucose. In the light-dependent reactions, chlorophyll absorbs photons and drives an electron transport chain that produces ATP and NADPH; in the Calvin cycle, those carriers fix carbon dioxide into sugars. ", 10)) + "To summarise, plants make food using",
    ];

    private static string? Find(string envVar, string org, string repo, string file)
        => DotLLM.Tests.Integration.Fixtures.TestFixtureResolver.ResolveFile(envVar, org, repo, file).Path;

    private sealed record Mode(string Name, int Chunk);   // Chunk: 1 = decode, 0 = whole prompt, n = n-token chunks

    private static readonly Mode[] Modes =
    [
        new("decode(1)", 1), new("prefill(all)", 0), new("chunk(16)", 16), new("chunk(3)", 3),
    ];

    [SkippableFact]
    public void Absorbed_IsLive_AndKlVsCpuIsNotWorseThanExpandedPath()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = Find("DOTLLM_GLM_GGUF", "unsloth", "GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf");
        Skip.If(path is null, "GLM-4.7-Flash Q4_K_M not found (set DOTLLM_GLM_GGUF).");
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int vocab = config.VocabSize;
        bool addBos = GgufAddBosResolver.Resolve(gguf.Metadata);
        int[][] ids = Prompts.Select(p => Encode(tokenizer, p, addBos)).ToArray();
        _output.WriteLine("prompt lengths: " + string.Join(", ", ids.Select(x => x.Length)));

        // CPU oracle: full-sequence logits + greedy continuation.
        const int Steps = 16;
        var cpuLogits = new float[ids.Length][][];
        var cpuGreedy = new int[ids.Length][];
        using (var cpu = DotLLM.Models.Architectures.TransformerModel.LoadFromGguf(gguf, config))
        {
            for (int p = 0; p < ids.Length; p++)
            {
                using (ITensor l = cpu.Forward(ids[p], Enumerable.Range(0, ids[p].Length).ToArray(), -1,
                    new SimpleKvCache(KvGeometry.FromConfig(config), ids[p].Length + 2)))
                    cpuLogits[p] = Rows(l, vocab);
                cpuGreedy[p] = Greedy(cpu, config, ids[p], Steps, size => new SimpleKvCache(KvGeometry.FromConfig(config), size)).Tokens;
            }
        }

        using var device = VulkanDevice.Create();

        // Control first: legacy expanded path (env var read at load time).
        VulkanTransformerModel legacy;
        Func<int, IKvCache> mkLegacy;
        string? prev = Environment.GetEnvironmentVariable("DOTLLM_VK_MLA_LEGACY");
        Environment.SetEnvironmentVariable("DOTLLM_VK_MLA_LEGACY", "1");
        try
        {
            var (m, mk) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
            legacy = (VulkanTransformerModel)m; mkLegacy = mk;
        }
        finally { Environment.SetEnvironmentVariable("DOTLLM_VK_MLA_LEGACY", prev); }
        using var legacyDisp = legacy as IDisposable;
        Assert.False(legacy.MlaAbsorbed, "control model must run the expanded MLA path");

        var (absModel, mkAbs) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spv);
        using var absDisp = absModel as IDisposable;
        var abs = (VulkanTransformerModel)absModel;
        Assert.True(abs.MlaAbsorbed, "default model must take the absorbed MLA path (Q8_0 attn_k_b/attn_v_b, kv_lora 512, rope 64)");

        // KV footprint: latent cache must be at least 10x smaller per token.
        using (var cA = (MlaVulkanKvCache)mkAbs(1024))
        using (var cL = (MlaVulkanKvCache)mkLegacy(1024))
        {
            Assert.True(cA.IsLatent && !cL.IsLatent);
            _output.WriteLine($"KV bytes @1024 tokens: absorbed {cA.AllocatedBytes / 1048576.0:F1} MiB, expanded {cL.AllocatedBytes / 1048576.0:F1} MiB ({(double)cL.AllocatedBytes / cA.AllocatedBytes:F1}x)");
            Assert.True(cA.AllocatedBytes * 10 < cL.AllocatedBytes);
        }

        var summary = new List<string>();
        double worstExcess = double.MinValue;
        foreach (var mode in Modes)
        {
            long d0 = MlaAbsorbedKernelBase.TotalDispatches;
            var (klAbs, agAbs, n) = Evaluate(abs, mkAbs, ids, cpuLogits, vocab, mode);
            long absDispatches = MlaAbsorbedKernelBase.TotalDispatches - d0;
            Assert.True(absDispatches > 0, $"{mode.Name}: the absorbed kernels never ran on the absorbed model (silent fall-back)");

            d0 = MlaAbsorbedKernelBase.TotalDispatches;
            var (klLeg, agLeg, _) = Evaluate(legacy, mkLegacy, ids, cpuLogits, vocab, mode);
            Assert.Equal(0, MlaAbsorbedKernelBase.TotalDispatches - d0);   // the control must be a real control

            string line = $"{mode.Name,-13} rows={n,4}  mean KL(cpu||vk)  absorbed {klAbs:F5}  expanded {klLeg:F5}   top-1 vs CPU  absorbed {agAbs}/{n}  expanded {agLeg}/{n}   absorbed dispatches {absDispatches}";
            _output.WriteLine(line);
            summary.Add(line);
            worstExcess = Math.Max(worstExcess, klAbs - Math.Max(klLeg * 2.0, klLeg + 0.01));
            Assert.True(klAbs <= Math.Max(klLeg * 2.0, klLeg + 0.01),
                $"{mode.Name}: absorbed KL {klAbs:F5} exceeds the expanded path's {klLeg:F5} by more than numerical noise");
            Assert.True(agAbs >= agLeg - Math.Max(2, n / 20), $"{mode.Name}: top-1 agreement dropped: absorbed {agAbs} vs expanded {agLeg} of {n}");
        }

        // Greedy text through the real engine path (prefill, then 1-row decode steps against the latent cache).
        int identAbs = 0, identLeg = 0;
        for (int p = 0; p < ids.Length; p++)
        {
            var gA = Greedy(abs, config, ids[p], Steps, mkAbs);
            var gL = Greedy(legacy, config, ids[p], Steps, mkLegacy);
            int dA = FirstDiff(gA.Tokens, cpuGreedy[p]), dL = FirstDiff(gL.Tokens, cpuGreedy[p]);
            if (dA < 0) identAbs++;
            if (dL < 0) identLeg++;
            _output.WriteLine($"greedy p{p}: first divergence vs CPU  absorbed {dA}  expanded {dL}"
                + (dA >= 0 ? $" (absorbed top1-top2 margin there {gA.Margins[dA]:F3})" : ""));
            _output.WriteLine($"     cpu      : '{tokenizer.Decode(cpuGreedy[p])}'");
            _output.WriteLine($"     absorbed : '{tokenizer.Decode(gA.Tokens)}'");
            _output.WriteLine($"     expanded : '{tokenizer.Decode(gL.Tokens)}'");
            // A divergence is acceptable only at a near-tie.
            if (dA >= 0) Assert.True(gA.Margins[dA] < 1.0f, $"prompt {p}: absorbed greedy diverged from CPU at step {dA} with a clear margin {gA.Margins[dA]:F3}");
        }
        _output.WriteLine($"greedy text identical to CPU over {Steps} tokens: absorbed {identAbs}/{ids.Length}, expanded {identLeg}/{ids.Length}");
        Assert.True(identAbs >= identLeg - 1, $"absorbed greedy identity {identAbs} fell below the expanded path's {identLeg}");
    }

    private (double Kl, int Agree, int Rows) Evaluate(VulkanTransformerModel vk, Func<int, IKvCache> mk, int[][] ids, float[][][] cpu, int vocab, Mode mode)
    {
        double klSum = 0; int agree = 0, rows = 0;
        for (int p = 0; p < ids.Length; p++)
        {
            int len = ids[p].Length;
            int chunk = mode.Chunk == 0 ? len : mode.Chunk;
            using var kv = mk(len + 2);
            for (int s = 0; s < len; s += chunk)
            {
                int n = Math.Min(chunk, len - s);
                using ITensor l = vk.Forward(ids[p].AsSpan(s, n), Enumerable.Range(s, n).ToArray(), -1, kv);
                float[] row = Rows(l, vocab)[^1];
                float[] want = cpu[p][s + n - 1];
                klSum += Kl(want, row);
                if (ArgMax(row) == ArgMax(want)) agree++;
                rows++;
            }
        }
        return (klSum / rows, agree, rows);
    }

    private static int[] Encode(DotLLM.Tokenizers.ITokenizer tok, string text, bool addBos)
    {
        int[] enc = tok.Encode(text);
        if (!addBos || tok.BosTokenId < 0) return enc;
        var ids = new int[enc.Length + 1];
        ids[0] = tok.BosTokenId;
        Array.Copy(enc, 0, ids, 1, enc.Length);
        return ids;
    }

    private static unsafe float[][] Rows(ITensor logits, int vocab)
    {
        int rows = logits.Shape[0];
        var res = new float[rows][];
        for (int r = 0; r < rows; r++)
            res[r] = new ReadOnlySpan<float>((float*)logits.DataPointer + (long)r * vocab, vocab).ToArray();
        return res;
    }

    private static (int[] Tokens, float[] Margins) Greedy(IModel m, ModelConfig config, int[] prompt, int steps, Func<int, IKvCache> mk)
    {
        int vocab = config.VocabSize;
        var toks = new List<int>(); var margins = new List<float>();
        using var kv = mk(prompt.Length + steps + 2);
        for (int s = 0; s < steps; s++)
        {
            int[] ids = s == 0 ? prompt : [toks[^1]];
            int[] pos = s == 0 ? Enumerable.Range(0, prompt.Length).ToArray() : [prompt.Length + s - 1];
            using ITensor l = m.Forward(ids, pos, -1, kv);
            float[] row = Rows(l, vocab)[^1];
            int best = ArgMax(row);
            float second = float.NegativeInfinity;
            for (int i = 0; i < row.Length; i++) if (i != best && row[i] > second) second = row[i];
            toks.Add(best); margins.Add(row[best] - second);
        }
        return (toks.ToArray(), margins.ToArray());
    }

    private static int FirstDiff(int[] a, int[] b)
    {
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (a[i] != b[i]) return i;
        return -1;
    }

    private static int ArgMax(float[] v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp) + mp, lzq = Math.Log(zq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lzp, lq = q[i] - lzq;
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }
}
