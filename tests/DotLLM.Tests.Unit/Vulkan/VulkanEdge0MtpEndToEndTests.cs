using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.Samplers;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Edge0;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 MTP draft head on Vulkan (G6, #1990), against the CPU oracle and in speed. Opt-in stages in SEPARATE processes (the CPU and the GPU never hold the 105 GB mapping's
/// working set together): <c>DOTLLM_EDGE0_VK_E2E=mtpcpu</c> writes the CPU reference (<c>DOTLLM_EDGE0_MTP_REF</c>): per prompt the CPU head's first three draft-step logit rows
/// (CPU trunk prefill, CPU head with ordinary ggml Q2_0 banks relaid out from the planar blobs); <c>=mtpvk</c> attaches the head on Vulkan (planar blobs imported in place), compares the
/// same draft steps, then measures acceptance and the decode speed of plain vs speculative greedy decode, interleaved in one session.
/// </summary>
[Trait("Category", "GPU")]
public sealed class VulkanEdge0MtpEndToEndTests(ITestOutputHelper output)
{
    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n    return",
        "The mitochondrion is an organelle found in most eukaryotic cells. Its main job is to convert nutrients into adenosine triphosphate, the molecule that cells use as a source of chemical energy. In software, a similar story is told about caches:",
        "Water boils at a temperature of 100 degrees Celsius at sea level. At high altitudes the boiling point decreases because",
    ];

    private const int Chain = 3;

    private const int PrefillChunk = 1024;

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf",
        "native_experts.txt", "experts.bin", "ple-fp8.gguf", "mtp/dense.txt", "mtp/dense.bin", "mtp/experts.bin", "mtp/draft_vocab.bin");

    private static int[] Pos(int n, int start = 0) => Enumerable.Range(start, n).ToArray();

    private static unsafe float[] Row(ITensor t, int row)
    {
        int v = t.Shape[1];
        return new ReadOnlySpan<float>((void*)(t.DataPointer + (nint)((long)row * v * 4)), v).ToArray();
    }

    private static int Argmax(ReadOnlySpan<float> v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    private static (float Margin, int Second) Margin(ReadOnlySpan<float> v)
    {
        int a = Argmax(v), b = a == 0 ? 1 : 0;
        for (int i = 0; i < v.Length; i++) if (i != a && v[i] > v[b]) b = i;
        return (v[a] - v[b], b);
    }

    private static int[][] Encode(string skeleton)
    {
        using var g = GgufFile.Open(skeleton);
        var tok = GgufBpeTokenizerFactory.Load(g.Metadata);
        return Prompts.Select(p => tok.Encode(p).ToArray()).ToArray();
    }

    private sealed record Ref(int First, int[] ChainTokens, float[][] Logits);

    [SkippableFact]
    public void Stage_CpuDraftReference()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "mtpcpu", "DOTLLM_EDGE0_VK_E2E=mtpcpu (CPU minutes; hold the cpu resource lock)");
        string refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_REF") ?? throw new InvalidOperationException("set DOTLLM_EDGE0_MTP_REF");
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (mtp/)"));
        var prompts = Encode(fx.Path!);
        var (model, gguf, config) = ModelLoader.LoadFromGguf(fx.Path!, ThreadingConfig.Auto);
        using (gguf) using (model)
        {
            var cpu = (Qwen4ExpTransformerModel)model;
            var pack = Edge0MtpPack.Open(Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!)!, config.VocabSize);
            var view = pack.BuildHeadGguf(config, ggmlExpertBanks: true, trunk: gguf, viewOwnsPack: true);
            cpu.AttachMtpHead(view, ownsFile: true);
            using var w = new BinaryWriter(File.Create(refPath));
            w.Write(prompts.Length);
            foreach (var ids in prompts)
            {
                cpu.ResetSequenceState();
                using var kv = new SimpleKvCache(DotLLM.Core.Attention.KvGeometry.FromConfig(config), 64);
                using var st = cpu.CreateMtpState(64)!;
                using var pf = cpu.Forward(ids, Pos(ids.Length), -1, kv, null, st);
                int first = Argmax(Row(pf, ids.Length - 1));
                int tok = first;
                var chainTokens = new int[Chain]; var logits = new float[Chain][];
                for (int j = 0; j < Chain; j++)
                {
                    using var lg = cpu.ForwardMtp(st, tok, ids.Length + j);
                    logits[j] = Row(lg, 0);
                    chainTokens[j] = Argmax(logits[j]); tok = chainTokens[j];
                }
                w.Write(first); foreach (int t in chainTokens) w.Write(t);
                foreach (var l in logits) { w.Write(l.Length); foreach (float f in l) w.Write(f); }
                output.WriteLine($"cpu ref: {ids.Length} tokens, first {first}, chain {string.Join(' ', chainTokens)}");
            }
        }
    }

    private static List<Ref> LoadRef(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        int n = r.ReadInt32();
        var list = new List<Ref>();
        for (int i = 0; i < n; i++)
        {
            int first = r.ReadInt32(); var ct = new int[Chain]; for (int j = 0; j < Chain; j++) ct[j] = r.ReadInt32();
            var lg = new float[Chain][];
            for (int j = 0; j < Chain; j++) { int v = r.ReadInt32(); lg[j] = new float[v]; for (int k = 0; k < v; k++) lg[j][k] = r.ReadSingle(); }
            list.Add(new Ref(first, ct, lg));
        }
        return list;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), sp = 0, sq = 0;
        foreach (float v in p) sp += Math.Exp(v - mp);
        foreach (float v in q) sq += Math.Exp(v - mq);
        double lsp = Math.Log(sp) + mp, lsq = Math.Log(sq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++) { double lp = p[i] - lsp, lq = q[i] - lsq, pp = Math.Exp(lp); if (pp > 0) kl += pp * (lp - lq); }
        return kl;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double n = 0, d = 0;
        for (int i = 0; i < a.Length; i++) { double e = a[i] - b[i]; n += e * e; d += (double)a[i] * a[i]; }
        return Math.Sqrt(n / d);
    }

    [SkippableFact]
    public void Stage_VulkanHead_DraftLogitsAndAcceptanceAndSpeed()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "mtpvk", "DOTLLM_EDGE0_VK_E2E=mtpvk (loads ~8 GiB device + 50 GiB imported; hold the gpu lock)");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k (mtp/)"));
        string? refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_REF");
        var cpuRef = refPath is not null && File.Exists(refPath) ? LoadRef(refPath) : null;
        var prompts = Encode(fx.Path!);
        using var gguf = GgufFile.Open(fx.Path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var vk = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, config, spvDir);
        long liveBefore = device.LiveBytesTotal();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        vk.AttachEdge0MtpHead(Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!)!);
        output.WriteLine($"MTPATTACH {sw.Elapsed.TotalSeconds:F2} s; device live {liveBefore / (double)(1L << 30):F2} -> {device.LiveBytesTotal() / (double)(1L << 30):F2} GiB; head present {vk.HasMtpHead}");
        var pr = System.Diagnostics.Process.GetCurrentProcess();
        output.WriteLine($"MTPMEM peak working set {pr.PeakWorkingSet64 / (double)(1L << 30):F2} GiB, private {pr.PrivateMemorySize64 / (double)(1L << 30):F2} GiB");
        int V = config.VocabSize;
        var failures = new List<string>();

        bool onlyLong = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_LONG") == "only";
        // ── 1. the draft head's logits against the CPU oracle (same trunk prefill token, same chain) ──
        if (cpuRef is not null && !onlyLong)
            for (int p = 0; p < prompts.Length; p++)
            {
                var ids = prompts[p];
                vk.ResetSequenceState();
                using var kv = vk.CreateKvCache(64);
                using var st = vk.CreateMtpState(64)!;
                using (vk.Forward(ids, Pos(ids.Length), -1, kv, null, st)) { }
                int tok = cpuRef[p].First;
                for (int j = 0; j < Chain; j++)
                {
                    using var lg = vk.ForwardMtp(st, tok, ids.Length + j);
                    var row = Row(lg, 0);
                    double rel = RelL2(cpuRef[p].Logits[j], row), kl = Kl(cpuRef[p].Logits[j], row);
                    int a = Argmax(row);
                    output.WriteLine($"DRAFTLOGITS prompt {p} step {j + 1}: relL2 {rel:E3}, KL(cpu||vk) {kl:E3}, top-1 cpu {cpuRef[p].ChainTokens[j]} vk {a} {(a == cpuRef[p].ChainTokens[j] ? "same" : "DIFF")}");
                    if (j == 0 && kl > 0.1) failures.Add($"prompt {p}: first draft step KL {kl:E3}");
                    tok = cpuRef[p].ChainTokens[j];   // teacher-force the CPU chain so every step compares like with like
                }
            }

        // ── 2. acceptance (draft vs verified), K = 3 and 4 ──
        (List<int> Tokens, List<int> PerRound, double Seconds) Speculate(int[] prompt, int n, int k)
        {
            vk.ResetSequenceState();
            using var kv = vk.CreateKvCache(prompt.Length + n + 16);
            using var st = vk.CreateMtpState(prompt.Length + n + 16)!;
            var dec = new MtpSpeculativeDecoder(greedy: true);
            var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });
            var gen = new List<int>();
            var t0 = System.Diagnostics.Stopwatch.StartNew();
            for (int c = 0; c < prompt.Length; c += PrefillChunk)   // the Edge0 trunk forward takes at most 2048 rows (latent scratch)
            {
                int m = Math.Min(PrefillChunk, prompt.Length - c);
                using var pf = vk.Forward(prompt.AsSpan(c, m), Pos(m, c), -1, kv, null, st, true);
                if (c + m == prompt.Length) gen.Add(Argmax(Row(pf, pf.Shape[0] - 1)));
            }
            var buf = new int[k + 1];
            var per = new List<int>();
            while (gen.Count < n)
            {
                int pos = prompt.Length + gen.Count - 1;
                var r = dec.DraftAndVerify(vk, kv, st, pipeline, gen, null, pos, V, Math.Min(k, n - gen.Count), buf);
                per.Add(r.AcceptedCount);
                for (int i = 0; i < r.AcceptedCount && gen.Count < n; i++) gen.Add(buf[i]);
            }
            return (gen, per, t0.Elapsed.TotalSeconds);
        }
        (List<int> Tokens, double Seconds) Plain(int[] prompt, int n) { var (t, s, _) = PlainWithMargins(prompt, n, margins: false); return (t, s); }
        // margins[i] = top-1 minus top-2 logit of the plain decode at generated token i: where speculative output first differs, a near-tie says
        // multi-row rounding picked the other token; a large margin would point at state (KV / recurrent rollback) instead.
        (List<int> Tokens, double Seconds, List<(float Margin, int Second)> Margins) PlainWithMargins(int[] prompt, int n, bool margins)
        {
            vk.ResetSequenceState();
            using var kv = vk.CreateKvCache(prompt.Length + n + 16);
            var gen = new List<int>();
            var mg = new List<(float, int)>();
            void Take(float[] row) { gen.Add(Argmax(row)); if (margins) mg.Add(Margin(row)); }
            var t0 = System.Diagnostics.Stopwatch.StartNew();
            for (int c = 0; c < prompt.Length; c += PrefillChunk)
            {
                int m = Math.Min(PrefillChunk, prompt.Length - c);
                using var pf = vk.Forward(prompt.AsSpan(c, m), Pos(m, c), -1, kv, true);
                if (c + m == prompt.Length) Take(Row(pf, pf.Shape[0] - 1));
            }
            while (gen.Count < n)
                using (var lg = vk.Forward([gen[^1]], [prompt.Length + gen.Count - 1], -1, kv, true)) Take(Row(lg, 0));
            return (gen, t0.Elapsed.TotalSeconds, mg);
        }
        string Divergence(List<int> spec, List<int> plain, List<(float Margin, int Second)> mg, int n)
        {
            int d = Enumerable.Range(0, n).FirstOrDefault(i => spec[i] != plain[i], -1);
            if (d < 0) return "identical";
            return $"differs at {d}: plain {plain[d]} (margin {mg[d].Margin:F3} over its #2 {mg[d].Second}), spec {spec[d]}{(spec[d] == mg[d].Second ? " = plain's #2" : " NOT plain's #2")}";
        }

        const int N = 96;
        foreach (int k in onlyLong ? Array.Empty<int>() : new[] { 3, 4 })
        {
            int drafted = 0, accepted = 0, rounds = 0, tokens = 0, same = 0;
            for (int p = 0; p < prompts.Length; p++)
            {
                var (toks, per, _) = Speculate(prompts[p], N, k);
                var (plain, _, mg) = PlainWithMargins(prompts[p], N, margins: true);
                rounds += per.Count; tokens += toks.Count; drafted += per.Count * k; accepted += per.Sum() - per.Count;
                string div = Divergence(toks, plain, mg, N);
                if (div == "identical") same++;
                output.WriteLine($"ACCEPT K={k} prompt {p}: {per.Count} rounds, {per.Sum()} tokens ({(double)per.Sum() / per.Count:F2}/round), spec == plain: {div}");
            }
            output.WriteLine($"ACCEPTTOTAL K={k}: {accepted}/{drafted} drafts accepted ({100.0 * accepted / drafted:F1}%), {(double)tokens / rounds:F2} tokens per verify round, {same}/{prompts.Length} prompts identical to plain greedy");
        }

        // ── 2b. long context (DOTLLM_EDGE0_MTP_LONG=1; =only skips sections 1-2): a ~2.3K-token prompt puts the head's own attention past the dense window (~2051 cells), so its
        // sparse indexer (the index_qk_proj q/k split) is live, and 256 speculative tokens exercise many reject/rollback rounds against plain greedy ──
        if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_LONG") is "1" or "only")
        {
            // DOTLLM_EDGE0_MTP_LONG_TEXT = a UTF-8 text file (e.g. the wikitext corpus): its first ~2.3K tokens, so the continuation is not a copy of the prompt.
            // Without it the two paragraph prompts alternate, and the repeated text makes acceptance trivially high (a rollback/indexer smoke only).
            int[] longPrompt;
            string? textPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_LONG_TEXT");
            if (textPath is not null)
            {
                var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
                longPrompt = tok.Encode(File.ReadAllText(textPath)[..12000]).ToArray()[..2300];
            }
            else
            {
                var parts = new List<int>();
                for (int r = 0; parts.Count < 2300; r++) parts.AddRange(r == 0 ? prompts[2] : prompts[2 + (r % 2)]);   // the two paragraph prompts, alternating
                longPrompt = parts.ToArray();
            }
            const int NL = 256;
            var (toks, per, secs) = Speculate(longPrompt, NL, 4);
            var (plain, psecs, mg) = PlainWithMargins(longPrompt, NL, margins: true);
            int acc = per.Sum() - per.Count;
            output.WriteLine($"LONG prompt {longPrompt.Length} tokens, {NL} generated, K=4: {acc}/{per.Count * 4} drafts accepted ({100.0 * acc / (per.Count * 4):F1}%), {(double)per.Sum() / per.Count:F2} tokens/round; " +
                             $"{NL / secs:F1} vs plain {NL / psecs:F1} tok/s; spec == plain: {Divergence(toks, plain, mg, NL)}");
            if (acc == 0) failures.Add("long context: no draft accepted");
        }

        // ── 2c. draft-vocab subset vs full-vocab drafting (#2013), interleaved in one session. DOTLLM_EDGE0_MTP_DRAFTVOCAB=1 (SPEED=0 skips the plain speed run after it) ──
        if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_DRAFTVOCAB") == "1")
        {
            Assert.True(vk.HasDraftVocabSubset, "the Edge0 head attached no draft-vocab subset (LM head not Q8_0, or matmul_q8_0_mmvq_rows.spv missing)");
            int nIds = (int)(new FileInfo(Path.Combine(Edge0MtpPack.Find(Path.GetDirectoryName(fx.Path!)!)!, "draft_vocab.bin")).Length / 4);
            const int rowBytes = 2560 / 32 * 34;
            output.WriteLine($"DRAFTVOCAB LM-head bytes per draft step: full {(long)V * rowBytes / 1e6:F1} MB ({V} rows), subset {(long)nIds * rowBytes / 1e6:F1} MB ({nIds} rows) + ids {nIds * 4 / 1e3:F0} KB; logits download {V * 4 / 1e3:F0} KB -> {nIds * 4 / 1e3:F0} KB");
            Speculate(prompts[1], 16, 3);   // warm
            foreach (int k in new[] { 3, 4 })
            {
                var steps = new double[2]; var ms = new double[2]; var tps = new List<double>[] { new(), new() };
                int[] acc = new int[2], drafted = new int[2];
                int identicalRuns = 0, sameDecisions = 0, sameDecisionsSameText = 0, runs = 0;
                for (int rep = 0; rep < 3; rep++)
                    for (int p = 0; p < prompts.Length; p++)
                    {
                        var res = new (List<int> T, List<int> R, double S)[2];
                        for (int arm = 0; arm < 2; arm++)   // arm 0 = full vocabulary, arm 1 = subset; the order alternates per rep so neither owns the cold side
                        {
                            int a = (arm + rep) % 2;
                            VulkanQwen4ExpTransformerModel.Edge0DraftVocabEnabled = a == 1;
                            vk.ResetMtpTimings();
                            var (t, per, secs) = Speculate(prompts[p], N, k);
                            var tm = vk.MtpTimings;
                            ms[a] += tm.DraftMs; steps[a] += tm.DraftSteps;
                            acc[a] += per.Sum() - per.Count; drafted[a] += per.Count * k;
                            tps[a].Add(N / secs);
                            res[a] = (t, per, secs);
                        }
                        runs++;
                        if (res[0].T.SequenceEqual(res[1].T)) identicalRuns++;
                        if (res[0].R.SequenceEqual(res[1].R)) { sameDecisions++; if (res[0].T.SequenceEqual(res[1].T)) sameDecisionsSameText++; }
                    }
                VulkanQwen4ExpTransformerModel.Edge0DraftVocabEnabled = true;
                string F(List<double> v) => $"{v.Min():F1}-{v.Max():F1} (mean {v.Average():F1})";
                output.WriteLine($"DRAFTVOCAB K={k}: draft step {ms[0] / steps[0]:F2} ms full vs {ms[1] / steps[1]:F2} ms subset; acceptance {100.0 * acc[0] / drafted[0]:F1}% full vs {100.0 * acc[1] / drafted[1]:F1}% subset; " +
                                 $"decode tok/s incl. prefill full {F(tps[0])} vs subset {F(tps[1])}; identical text {identicalRuns}/{runs}, identical accept decisions {sameDecisions}/{runs} of which identical text {sameDecisionsSameText}");
                Assert.Equal(sameDecisions, sameDecisionsSameText);   // verify decides every token: the same decisions must give the same text
            }
        }

        // ── 3. speed, interleaved in one session (group skipping on); DOTLLM_EDGE0_MTP_SPEED=0 skips it ──
        if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_MTP_SPEED") == "0") { Assert.True(failures.Count == 0, string.Join("; ", failures)); return; }
        VulkanQwen3MoeHybridTransformerModel.Edge0SkipUnusedGroups = true;
        var plainTps = new List<double>(); var k3 = new List<double>(); var k4 = new List<double>();
        Plain(prompts[1], 16); Speculate(prompts[1], 16, 3);   // warm both paths
        for (int rep = 0; rep < 3; rep++)
            for (int p = 0; p < prompts.Length; p++)
            {
                var (_, ps) = Plain(prompts[p], N); plainTps.Add(N / ps);
                var (_, _, s3) = Speculate(prompts[p], N, 3); k3.Add(N / s3);
                var (_, _, s4) = Speculate(prompts[p], N, 4); k4.Add(N / s4);
            }
        string Fmt(List<double> v) => $"{v.Min():F1}-{v.Max():F1} (mean {v.Average():F1})";
        output.WriteLine($"MTPSPEED tokens/s incl. prefill, {N} tokens x {prompts.Length} prompts x 3 reps interleaved: plain {Fmt(plainTps)}; MTP K=3 {Fmt(k3)}; MTP K=4 {Fmt(k4)}; speedup mean K=3 {k3.Average() / plainTps.Average():F2}x K=4 {k4.Average() / plainTps.Average():F2}x");
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }
}
