using System.Diagnostics;
using System.Reflection;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1686: the GQA-grouped split-KV decode pass against the per-query-head pass on a real file, from the SAME cache and recurrent
/// state at several depths, for every Vulkan host that opted in under #1686: the dense F16 host (<see cref="VulkanTransformerModel"/>:
/// Llama / Qwen / Mistral / Phi), the Qwen3.5/3.6 MoE hybrid (<see cref="VulkanQwen3MoeHybridTransformerModel"/>) and Nemotron-H
/// (<see cref="VulkanNemotronHTransformerModel"/>). Opt-in and slow: <c>DOTLLM_GQA_HOST_AB_GGUF=&lt;path&gt;</c>. Needs the Strix:GPU
/// lock (and Strix:MEMORY for a file over ~40 GB resident).
/// </summary>
/// <remarks>
/// <para>Per depth (default 1024, 4096, 8192, 16384; <c>DOTLLM_GQA_HOST_AB_DEPTHS</c>): a discarded warm run, then
/// <c>DOTLLM_GQA_HOST_AB_ROUNDS</c> (default 2) rounds of A B B A, each run <see cref="Steps"/> greedy decode steps from the identical
/// state (KV rolled back; GDN / SSM recurrent state restored from a snapshot). Reports ms/step min-max per arm, the first step's
/// KL(per-head || grouped), argmax, and the first greedy divergence. The grouped arm must have taken the grouped pass
/// (<see cref="VulkanSplitKvAttentionKernel.GqaDispatches"/>) and its first-step logits must differ from the per-head arm's in at
/// least one bit (substitution: identical logits would mean the toggle never reached the output).</para>
/// <para>Real wikitext ids come from the GGUF tokenizer over the LF corpus (<c>DOTLLM_WIKITEXT_LF</c>, default
/// <c>~/.dotllm/test-cache/corpora/wikitext-2-raw/wiki.test.lf.raw</c>).</para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGroupedSplitKvHostDepthTests
{
    private const int Steps = 32;
    private const int PrefillChunk = 512;

    private readonly ITestOutputHelper _out;
    public VulkanGroupedSplitKvHostDepthTests(ITestOutputHelper output) => _out = output;

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int b = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i;
        return b;
    }

    /// <summary>KL(softmax(p) || softmax(q)) in nats.</summary>
    private static double Kl(float[] p, float[] q)
    {
        static double[] LogSoftmax(float[] x)
        {
            double mx = x.Max(), s = 0;
            foreach (float f in x) s += Math.Exp(f - mx);
            double ls = Math.Log(s);
            return x.Select(f => f - mx - ls).ToArray();
        }
        var lp = LogSoftmax(p); var lq = LogSoftmax(q);
        double kl = 0;
        for (int i = 0; i < lp.Length; i++) kl += Math.Exp(lp[i]) * (lp[i] - lq[i]);
        return kl;
    }

    private static T Field<T>(object o, string name) where T : class
        => (T)(o.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)
               ?? throw new MissingFieldException(o.GetType().Name, name)).GetValue(o)!;

    /// <summary>Snapshot / restore of a host's recurrent state (a no-op for the dense host).</summary>
    private sealed class RecurrentSnapshot : IDisposable
    {
        private readonly VulkanDevice _device;
        private readonly List<(VulkanDevice.Buffer Live, VulkanDevice.Buffer Copy)> _buffers = new();
        private readonly VulkanGdnStateCache? _gdnLive;
        private VulkanGdnStateCache? _gdnCopy;

        public RecurrentSnapshot(VulkanDevice device, IModel model)
        {
            _device = device;
            switch (model)
            {
                case VulkanQwen3MoeHybridTransformerModel:
                    _gdnLive = Field<VulkanGdnStateCache>(model, "_gdnCache");
                    break;
                case VulkanNemotronHTransformerModel:
                    var ssm = Field<VulkanSsmStateCache>(model, "_ssmCache");
                    for (int i = 0; i < ssm.NumSsmLayers; i++)
                    {
                        foreach (var live in new[] { ssm.GetConvStateBuffer(i), ssm.GetSsmStateBuffer(i) })
                            _buffers.Add((live, device.Allocate(live.Size)));
                    }
                    break;
            }
        }

        public void Save()
        {
            if (_gdnLive is not null)
            {
                _gdnCopy ??= _gdnLive.CloneGeometry();
                _gdnLive.CopyTo(_gdnCopy);
            }
            foreach (var (live, copy) in _buffers) _device.CopyBufferSynchronous(live, copy, (ulong)live.Size);
        }

        public void Restore()
        {
            _gdnCopy?.CopyTo(_gdnLive!);
            foreach (var (live, copy) in _buffers) _device.CopyBufferSynchronous(copy, live, (ulong)live.Size);
        }

        public void Dispose()
        {
            _gdnCopy?.Dispose();
            foreach (var (_, copy) in _buffers) copy.Dispose();
        }
    }

    /// <summary>The host's split-KV decode kernel that serves the default KV storage of the file.</summary>
    private static VulkanSplitKvAttentionKernel? HostKernel(IModel model, IKvCache cache) => model switch
    {
        VulkanTransformerModel => Field<VulkanSplitKvAttentionKernel>(model,
            ((VulkanKvCache)cache).Storage == VulkanKvStorage.F16 ? "_splitKvAttentionKvF16" : "_splitKvAttention"),
        VulkanQwen3MoeHybridTransformerModel => Field<VulkanQwen3MoeHybridKernels>(model, "_kernels").SplitKvAttention,
        VulkanNemotronHTransformerModel => Field<VulkanSplitKvAttentionKernel>(model, "_splitKvAttention"),
        _ => throw new NotSupportedException($"no #1686 host: {model.GetType().Name}"),
    };

    [SkippableFact]
    public void GroupedSplitKv_MatchesPerHead_AtDepth_OnHost()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GQA_HOST_AB_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GQA_HOST_AB_GGUF=<path> to enable (loads the whole file).");
        Skip.IfNot(Environment.GetEnvironmentVariable(VulkanSplitKvAttentionKernel.GqaEnvVar) is null,
            $"{VulkanSplitKvAttentionKernel.GqaEnvVar} must be unset: the test checks the host's own default");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int[] depths = (Environment.GetEnvironmentVariable("DOTLLM_GQA_HOST_AB_DEPTHS") ?? "1024,4096,8192,16384")
            .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).OrderBy(d => d).ToArray();
        int rounds = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_GQA_HOST_AB_ROUNDS"), out int r) ? r : 2;
        int maxDepth = depths[^1];

        using var gguf = GgufFile.Open(path!);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        int numHeads = cfg.NumAttentionHeads, numKvHeads = cfg.NumKvHeads, headDim = cfg.HeadDim;
        _out.WriteLine($"{Path.GetFileName(path)}: {cfg.Architecture}, heads {numHeads} / kv {numKvHeads} / head_dim {headDim} / layers {cfg.NumLayers}");

        int[] ids = WikitextIds(gguf, maxDepth + 1);

        using var device = VulkanDevice.Create();
        var (model, cacheFactory) = VulkanModelLoader.CreateFromGguf(device, gguf, cfg, spvDir);
        using var _m = model;
        using IKvCache cache = cacheFactory(maxDepth + Steps + 16);
        _out.WriteLine($"host {model.GetType().Name}, cache {cache.GetType().Name}{(cache is VulkanKvCache vk ? " " + vk.Storage : "")}; " + device.MemorySnapshot());

        VulkanSplitKvAttentionKernel kernel = HostKernel(model, cache)
            ?? throw new InvalidOperationException("the host built no split-KV decode kernel");
        Assert.True(kernel.UsesGqa(numHeads, numKvHeads, headDim), $"the grouped pass cannot serve {numHeads} / {numKvHeads} / {headDim}");
        Assert.True(kernel.PreferGqa, $"{model.GetType().Name} must opt into the grouped split-KV pass by default (#1686)");
        _out.WriteLine($"kv storage {kernel.KvStorage}, grouped v2: {kernel.UsesGqa2(numHeads, numKvHeads, headDim)}, " +
                       $"sub-groups {VulkanSplitKvAttentionKernel.GqaSubGroups(numHeads, numKvHeads)}");

        using var snap = new RecurrentSnapshot(device, model);

        float[] Step(ReadOnlySpan<int> tok, int pos0)
        {
            int[] pos = Enumerable.Range(pos0, tok.Length).ToArray();
            using ITensor t = model.Forward(tok, pos, deviceId: -1, cache);
            return Copy(t);
        }

        int filled = 0;
        _out.WriteLine("depth\tarm\tms/step\ttok/s\tgqaDispatches");
        foreach (int depth in depths)
        {
            var swPre = Stopwatch.StartNew();
            // Every arm of the previous depth ended with a restore, so the live state is that depth's prefill state: extend it.
            while (filled < depth)
            {
                int n = Math.Min(PrefillChunk, depth - filled);
                Step(ids.AsSpan(filled, n), filled);
                filled += n;
            }
            _out.WriteLine($"prefilled to {filled} in {swPre.Elapsed.TotalSeconds:F1} s");
            Assert.Equal(depth, cache.CurrentLength);
            snap.Save();

            float[]? firstPerHead = null, firstGrouped = null;
            int[]? seqPerHead = null, seqGrouped = null;
            var msPerHead = new List<double>();
            var msGrouped = new List<double>();
            var arms = new List<bool> { true };   // warm (discarded)
            for (int i = 0; i < rounds; i++) arms.AddRange(i % 2 == 0 ? [false, true, true, false] : [true, false, false, true]);
            for (int a = 0; a < arms.Count; a++)
            {
                bool grouped = arms[a];
                kernel.PreferGqa = grouped;
                long before = kernel.GqaDispatches;
                var seq = new int[Steps];
                float[]? first = null;
                int tok = ids[depth];
                var sw = Stopwatch.StartNew();
                for (int s = 0; s < Steps; s++)
                {
                    float[] logits = Step([tok], depth + s);
                    first ??= logits;
                    tok = Argmax(logits);
                    seq[s] = tok;
                }
                double ms = sw.Elapsed.TotalMilliseconds / Steps;
                long dispatches = kernel.GqaDispatches - before;
                cache.Rollback(depth);
                snap.Restore();

                if (grouped) Assert.True(dispatches > 0, "the grouped arm did not take the grouped pass");
                else Assert.Equal(0, dispatches);
                if (a == 0) { _out.WriteLine($"{depth}\twarm\t{ms:F1}\t{1000 / ms:F2}\t{dispatches}"); continue; }

                if (grouped) { firstGrouped ??= first; seqGrouped ??= seq; msGrouped.Add(ms); }
                else { firstPerHead ??= first; seqPerHead ??= seq; msPerHead.Add(ms); }
                _out.WriteLine($"{depth}\t{(grouped ? "grouped" : "per-head")}\t{ms:F1}\t{1000 / ms:F2}\t{dispatches}");
            }

            double kl = Kl(firstPerHead!, firstGrouped!);
            int aP = Argmax(firstPerHead!), aG = Argmax(firstGrouped!);
            int div = FirstDivergence(seqPerHead!, seqGrouped!);
            bool bitIdentical = firstPerHead!.AsSpan().SequenceEqual(firstGrouped!);
            msPerHead.Sort(); msGrouped.Sort();
            double medP = msPerHead[msPerHead.Count / 2], medG = msGrouped[msGrouped.Count / 2];
            _out.WriteLine($"SUMMARY depth {depth}: tok/s per-head {1000 / medP:F2} [{1000 / msPerHead[^1]:F2}-{1000 / msPerHead[0]:F2}] " +
                           $"grouped {1000 / medG:F2} [{1000 / msGrouped[^1]:F2}-{1000 / msGrouped[0]:F2}] ({medP / medG:F2}x median); " +
                           $"first-step KL {kl:E2}, argmax {aP}/{aG}, bit-identical {bitIdentical}, " +
                           $"greedy {Steps}: {(div < 0 ? "identical" : $"first divergence at step {div}")}");
            Assert.False(bitIdentical, $"depth {depth}: the two passes gave bit-identical logits; the toggle did not reach the output");
            Assert.True(kl < 1e-3, $"depth {depth}: grouped pass diverges from the per-head pass at the first step: KL {kl}");
            Assert.Equal(aP, aG);
        }

        kernel.PreferGqa = true;
    }

    private static int FirstDivergence(int[] a, int[] b)
    {
        for (int i = 0; i < Math.Min(a.Length, b.Length); i++) if (a[i] != b[i]) return i;
        return -1;
    }

    private int[] WikitextIds(GgufFile gguf, int count)
    {
        string corpus = Environment.GetEnvironmentVariable("DOTLLM_WIKITEXT_LF")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache", "corpora", "wikitext-2-raw", "wiki.test.lf.raw");
        Skip.IfNot(File.Exists(corpus), $"wikitext LF corpus not found at {corpus} (DOTLLM_WIKITEXT_LF)");
        string text = File.ReadAllText(corpus);
        int chars = Math.Min(text.Length, count * 8 + 4096);
        var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(text[..chars]);
        _out.WriteLine($"tokenised {chars} chars -> {ids.Length} ids (need {count})");
        Assert.True(ids.Length >= count, $"corpus too short: {ids.Length} ids for {count}");
        return ids[..count];
    }
}
