using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0-145B on Vulkan against the CPU path (G3, #1981). Two opt-in stages that share a reference file, run in SEPARATE processes so the CPU and
/// the GPU never hold the 105 GB mapping's working set together and the CPU stage never overlaps a GPU run:
/// <list type="number">
/// <item><c>DOTLLM_EDGE0_VK_E2E=cpu</c> + <c>DOTLLM_EDGE0_REF</c>=path: runs the CPU model on the NLL passage and three short prompts, records per-row argmax,
/// the passage mean NLL, the last-row logits of each prompt and a 24-token greedy continuation.</item>
/// <item><c>DOTLLM_EDGE0_VK_E2E=vulkan</c> + the same path: runs the Vulkan model on the same inputs and compares: greedy text, NLL, KL(cpu || vulkan).</item>
/// </list>
/// </summary>
[Trait("Category", "GPU")]
public sealed class VulkanEdge0EndToEndTests(ITestOutputHelper output)
{
    private const string Passage =
        "The mitochondrion is an organelle found in most eukaryotic cells. Its main job is to convert nutrients into adenosine triphosphate, " +
        "the molecule that cells use as a source of chemical energy. In software, a similar story is told about caches: a small, fast memory " +
        "sits in front of a large, slow one, and a good replacement policy keeps the hot data close. The city of Lyon lies at the confluence " +
        "of the Rhone and the Saone rivers in the east of France.";

    private static readonly string[] Prompts =
    [
        "The capital of France is",
        "def reverse_string(s):\n    return",
        "Water boils at a temperature of",
    ];

    private const int GreedyTokens = 24;

    private static FixtureLocation Locate() => TestFixtureResolver.ResolveFile("DOTLLM_EDGE0_SKELETON", "Edge0", "Edge0-145B-q3k", "model-skeleton.gguf",
        "native_experts.txt", "experts.bin", "ple-fp8.gguf");

    private sealed record Run(int[] Ids, int[] RowArgmax, double MeanNll, float[] LastLogits, int[] Greedy, float[]? AllRows = null);

    private static unsafe int Argmax(ReadOnlySpan<float> row) { int b = 0; for (int i = 1; i < row.Length; i++) if (row[i] > row[b]) b = i; return b; }

    private static unsafe Run Execute(IModel model, ReadOnlySpan<int> ids, bool nll, Action<int>? allRows)
    {
        model.ResetSequenceState();
        allRows?.Invoke(ids.Length);
        var pos = new int[ids.Length];
        for (int i = 0; i < pos.Length; i++) pos[i] = i;
        using var logits = model.Forward(ids, pos, 0);
        int vocab = (int)logits.Shape[1];
        int rows = (int)logits.Shape[0];
        if (rows != ids.Length) throw new InvalidOperationException($"forward returned {rows} logit rows for {ids.Length} tokens (all-row logits unavailable)");
        var l = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab);
        var argmax = new int[rows];
        double total = 0;
        for (int t = 0; t < rows; t++)
        {
            var row = l.Slice(t * vocab, vocab);
            argmax[t] = Argmax(row);
            if (nll && t + 1 < rows)
            {
                float max = float.NegativeInfinity; foreach (float v in row) max = Math.Max(max, v);
                double sum = 0; foreach (float v in row) sum += Math.Exp(v - max);
                total -= row[ids[t + 1]] - max - Math.Log(sum);
            }
        }
        float[] last = l.Slice((rows - 1) * vocab, vocab).ToArray();
        var greedy = new int[GreedyTokens];
        int next = argmax[^1];
        for (int s = 0; s < GreedyTokens; s++)
        {
            greedy[s] = next;
            using var lg = model.Forward([next], [ids.Length + s], 0);
            next = Argmax(new ReadOnlySpan<float>((void*)lg.DataPointer, vocab));
        }
        return new Run(ids.ToArray(), argmax, nll ? total / (rows - 1) : double.NaN, last, greedy, nll ? l.ToArray() : null);
    }

    private static void Save(string path, List<Run> runs)
    {
        using var w = new BinaryWriter(File.Create(path));
        w.Write(runs.Count);
        foreach (var r in runs)
        {
            w.Write(r.Ids.Length); foreach (int i in r.Ids) w.Write(i);
            foreach (int a in r.RowArgmax) w.Write(a);
            w.Write(r.MeanNll);
            w.Write(r.LastLogits.Length); foreach (float f in r.LastLogits) w.Write(f);
            foreach (int g in r.Greedy) w.Write(g);
        }
    }

    private static List<Run> Load(string path)
    {
        using var rd = new BinaryReader(File.OpenRead(path));
        int n = rd.ReadInt32();
        var list = new List<Run>();
        for (int k = 0; k < n; k++)
        {
            int t = rd.ReadInt32(); var ids = new int[t]; for (int i = 0; i < t; i++) ids[i] = rd.ReadInt32();
            var am = new int[t]; for (int i = 0; i < t; i++) am[i] = rd.ReadInt32();
            double nll = rd.ReadDouble();
            int v = rd.ReadInt32(); var last = new float[v]; for (int i = 0; i < v; i++) last[i] = rd.ReadSingle();
            var g = new int[GreedyTokens]; for (int i = 0; i < g.Length; i++) g[i] = rd.ReadInt32();
            list.Add(new Run(ids, am, nll, last, g));
        }
        return list;
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), sp = 0, sq = 0;
        foreach (float v in p) sp += Math.Exp(v - mp);
        foreach (float v in q) sq += Math.Exp(v - mq);
        double lsp = Math.Log(sp) + mp, lsq = Math.Log(sq) + mq, kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - lsp, lq = q[i] - lsq;
            double pp = Math.Exp(lp);
            if (pp > 0) kl += pp * (lp - lq);
        }
        return kl;
    }

    private static string LastRowKlTop = "";
    private static (double Mean, double Max, int Rows)? RowKl(float[]? vkRows)
    {
        if (vkRows is null || Environment.GetEnvironmentVariable("DOTLLM_EDGE0_ROWS_REF") is not { Length: > 0 } path || !File.Exists(path)) return null;
        using var rd = new BinaryReader(File.OpenRead(path));
        int rows = rd.ReadInt32(), vocab = rd.ReadInt32();
        var cpuRows = new float[rows * vocab];
        for (int i = 0; i < cpuRows.Length; i++) cpuRows[i] = rd.ReadSingle();
        double sum = 0, max = 0; var per = new List<(int Row, double Kl)>();
        for (int t = 0; t < rows; t++)
        {
            double kl = Kl(cpuRows.AsSpan(t * vocab, vocab).ToArray(), vkRows.AsSpan(t * vocab, vocab).ToArray());
            sum += kl; max = Math.Max(max, kl); per.Add((t, kl));
        }
        LastRowKlTop = string.Join(" ", per.OrderByDescending(x => x.Kl).Take(6).Select(x => $"r{x.Row}:{x.Kl:E1}"));
        return (sum / rows, max, rows);
    }

    private static int[][] Inputs(string skeleton)
    {
        using var g = GgufFile.Open(skeleton);
        var tok = GgufBpeTokenizerFactory.Load(g.Metadata);
        var list = new List<int[]> { tok.Encode(Passage).ToArray() };
        foreach (string p in Prompts) list.Add(tok.Encode(p).ToArray());
        return list.ToArray();
    }

    [SkippableFact]
    public void Stage_CpuReference()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "cpu", "DOTLLM_EDGE0_VK_E2E=cpu (CPU minutes; hold the cpu resource lock)");
        string refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_REF") ?? throw new InvalidOperationException("set DOTLLM_EDGE0_REF");
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k"));
        var inputs = Inputs(fx.Path!);
        var (model, gguf, _) = ModelLoader.LoadFromGguf(fx.Path!, DotLLM.Core.Configuration.ThreadingConfig.Auto);
        using (gguf) using (model)
        {
            var runs = new List<Run>();
            for (int i = 0; i < inputs.Length; i++)
            {
                runs.Add(Execute(model, inputs[i], nll: i == 0, null));
                output.WriteLine($"cpu input {i}: {inputs[i].Length} tokens, nll {runs[^1].MeanNll:F4}, greedy {string.Join(' ', runs[^1].Greedy.Take(8))}");
            }
            Save(refPath, runs);
            if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_ROWS_REF") is { Length: > 0 } rowsPath && runs[0].AllRows is { } all)
            {
                // Teacher-forced reference (#1981): the passage's logits at EVERY row, so a format can be ranked by mean per-row KL over ~106 rows rather than by 24 knife-edge greedy tokens.
                using var rw = new BinaryWriter(File.Create(rowsPath));
                rw.Write(runs[0].Ids.Length); rw.Write(all.Length / runs[0].Ids.Length);
                foreach (float f in all) rw.Write(f);
            }
        }
    }

    [SkippableFact]
    public void Stage_VulkanAgainstCpuReference()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "vulkan", "DOTLLM_EDGE0_VK_E2E=vulkan (loads ~52 GiB onto the GPU; hold the gpu lock)");
        string refPath = Environment.GetEnvironmentVariable("DOTLLM_EDGE0_REF") ?? throw new InvalidOperationException("set DOTLLM_EDGE0_REF");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k"));
        var cpu = Load(refPath);
        var inputs = Inputs(fx.Path!);
        using var gguf = GgufFile.Open(fx.Path!);
        using var device = VulkanDevice.Create();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        using var vk = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, GgufModelConfigExtractor.Extract(gguf.Metadata), spvDir);
        output.WriteLine($"vulkan load: {sw.Elapsed.TotalSeconds:F1} s; device live {device.LiveBytesTotal() / (double)(1L << 30):F2} GiB");
        {
            var pr = System.Diagnostics.Process.GetCurrentProcess();
            output.WriteLine($"LOADMEM peak working set {pr.PeakWorkingSet64 / (double)(1L << 30):F2} GiB, peak commit {pr.PeakPagedMemorySize64 / (double)(1L << 30):F2} GiB, working set now {pr.WorkingSet64 / (double)(1L << 30):F2} GiB, private now {pr.PrivateMemorySize64 / (double)(1L << 30):F2} GiB; host-import tensors {VulkanWeightImportPolicy.ImportedTensorCount}");
        }

        using var tg = GgufFile.Open(fx.Path!);
        var tok = GgufBpeTokenizerFactory.Load(tg.Metadata);
        var failures = new List<string>();
        if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_SWEEP") is { Length: > 0 } sweepSpec)
        {
            // Parity + speed report (#1981): runs the four inputs (NLL, last-row KL, greedy-24, row argmax, mean per-row KL when DOTLLM_EDGE0_ROWS_REF exists) and the prefill / decode speed
            // once per ';'-separated label (the label is only printed; the kernels under test are whatever this build contains).
            foreach (string cfg in sweepSpec.Split(';'))
            {
                var kls = new List<string>(); var gr = new List<string>(); (double Mean, double Max, int Rows)? rowKl = null; double nllV = 0; int rowAg = 0, rowN = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var r = Execute(vk, inputs[i], nll: i == 0, n => vk.TrySetAllRowLogitsLimit(n));
                    var c = cpu[i];
                    kls.Add(Kl(c.LastLogits, r.LastLogits).ToString("E1"));
                    int ga = 0; for (int t = 0; t < GreedyTokens; t++) if (c.Greedy[t] == r.Greedy[t]) ga++;
                    gr.Add($"{ga}/{GreedyTokens}");
                    if (i == 0) { rowKl = RowKl(r.AllRows); nllV = r.MeanNll; for (int t = 0; t < c.RowArgmax.Length; t++) if (c.RowArgmax[t] == r.RowArgmax[t]) rowAg++; rowN = c.RowArgmax.Length; }
                }
                vk.TrySetAllRowLogitsLimit(1);
                var longIds2 = Enumerable.Range(0, 5).SelectMany(_ => inputs[0]).Take(512).ToArray();
                var pf2 = new List<double>(); var dc2 = new List<double>();
                for (int rep = 0; rep < 3; rep++)
                {
                    vk.ResetSequenceState();
                    var swp = System.Diagnostics.Stopwatch.StartNew();
                    using (vk.Forward(longIds2, Enumerable.Range(0, longIds2.Length).ToArray(), -1, null, true)) { }
                    pf2.Add(longIds2.Length / swp.Elapsed.TotalSeconds);
                }
                for (int rep = 0; rep < 3; rep++)
                {
                    vk.ResetSequenceState();
                    using (vk.Forward(inputs[1], Enumerable.Range(0, inputs[1].Length).ToArray(), -1, null, true)) { }
                    int pos = inputs[1].Length;
                    var swd = System.Diagnostics.Stopwatch.StartNew();
                    for (int sIdx = 0; sIdx < 48; sIdx++) { using var lg = vk.Forward([13], [pos++], -1, null, true); }
                    dc2.Add(48 / swd.Elapsed.TotalSeconds);
                }
                output.WriteLine($"SWEEP {cfg}: scratch {vk.Edge0ScratchBytesForTest / 1048576.0:F1} MiB; device live {device.LiveBytesTotal() / (double)(1L << 30):F2} GiB; NLL {nllV:F4} (cpu {cpu[0].MeanNll:F4}); ROWKL {(rowKl is { } rk ? $"mean {rk.Mean:E3} max {rk.Max:E3} over {rk.Rows} rows, top {LastRowKlTop}" : "n/a")}; KL {string.Join(" / ", kls)}; greedy {string.Join(" ", gr)}; row argmax {rowAg}/{rowN}; prefill {string.Join("/", pf2.Select(v => v.ToString("F0")))} tok/s; decode {string.Join("/", dc2.Select(v => v.ToString("F1")))} tok/s");
            }
            return;
        }
        for (int i = 0; i < inputs.Length; i++)
        {
            var r = Execute(vk, inputs[i], nll: i == 0, n => vk.TrySetAllRowLogitsLimit(n));
            var c = cpu[i];
            double kl = Kl(c.LastLogits, r.LastLogits);
            int rowAgree = 0; for (int t = 0; t < c.RowArgmax.Length; t++) if (c.RowArgmax[t] == r.RowArgmax[t]) rowAgree++;
            int greedyAgree = 0; for (int t = 0; t < GreedyTokens; t++) if (c.Greedy[t] == r.Greedy[t]) greedyAgree++;
            int firstDiv = Array.FindIndex(Enumerable.Range(0, GreedyTokens).ToArray(), t => c.Greedy[t] != r.Greedy[t]);
            output.WriteLine($"input {i}: {inputs[i].Length} tok; KL(cpu||vk) last row {kl:E3}; row argmax agree {rowAgree}/{c.RowArgmax.Length}; " +
                             $"greedy agree {greedyAgree}/{GreedyTokens} (first divergence {(firstDiv < 0 ? "none" : firstDiv.ToString())}); " +
                             (i == 0 ? $"NLL cpu {c.MeanNll:F4} vk {r.MeanNll:F4}" : ""));
            output.WriteLine($"  cpu text: {tok.Decode(c.Greedy).Replace((char)10, (char)126)}");
            output.WriteLine($"  vk  text: {tok.Decode(r.Greedy).Replace((char)10, (char)126)}");
            if (kl >= 0.05) failures.Add($"input {i}: KL {kl:E3}");
            if (i == 0)
            {
                // The passage is a knife-edge for ANY narrow latent format (greedy flips at its first token or at token 19 for every F16 / Q8 variant measured, and its last prompt row
                // is a near-tie between " In" and " The"), so it is gated teacher-forced over all of its rows instead (#1981, James's ruling 2026-11-25). Measured on the shipped F16 build
                // (Strix Halo, real Edge0-145B-q3k, CPU reference edge0_rows_tf.bin): NLL 1.6877 (cpu 1.6897), mean row KL 4.13e-3, max row KL 6.06e-2, row argmax 102/106.
                // The old F32 build measured NLL 1.7037, 4.30e-3, 4.82e-2, 102/106. Each gate has ~1.5x headroom on those.
                if (Math.Abs(c.MeanNll - r.MeanNll) >= 0.02) failures.Add($"passage NLL cpu {c.MeanNll:F4} vs vulkan {r.MeanNll:F4} (gate |diff| < 0.02)");
                var rk = RowKl(r.AllRows) ?? throw new InvalidOperationException("DOTLLM_EDGE0_ROWS_REF (the CPU teacher-forced all-row reference, written by Stage_CpuReference) is required for the passage gates");
                if (rk.Mean >= 6.0e-3) failures.Add($"passage mean row KL {rk.Mean:E3} (gate < 6.0e-3)");
                if (rk.Max >= 9.0e-2) failures.Add($"passage max row KL {rk.Max:E3} (gate < 9.0e-2)");
                if (rowAgree < 100) failures.Add($"passage row argmax {rowAgree}/{c.RowArgmax.Length} (gate >= 100)");
                output.WriteLine($"  passage teacher-forced: mean row KL {rk.Mean:E3}, max {rk.Max:E3}, row argmax {rowAgree}/{c.RowArgmax.Length}, top rows {LastRowKlTop}");
            }
            else
            {
                if (r.RowArgmax[^1] != c.RowArgmax[^1]) failures.Add($"input {i}: next-token argmax differs");
                if (greedyAgree != GreedyTokens) failures.Add($"input {i}: greedy {greedyAgree}/{GreedyTokens}");
            }
        }

        // Speed (warm): prefill of the passage repeated to ~512 tokens, then single-token decode. Same session, after the parity forwards warmed every pipeline.
        var longIds = Enumerable.Range(0, 5).SelectMany(_ => inputs[0]).Take(512).ToArray();
        vk.TrySetAllRowLogitsLimit(1);
        var pfs = new List<double>();
        for (int rep = 0; rep < 3; rep++)
        {
            vk.ResetSequenceState();
            var swp = System.Diagnostics.Stopwatch.StartNew();
            using (vk.Forward(longIds, Enumerable.Range(0, longIds.Length).ToArray(), -1, null, true)) { }
            pfs.Add(longIds.Length / swp.Elapsed.TotalSeconds);
        }
        var dec = new List<double>();
        for (int rep = 0; rep < 3; rep++)
        {
            vk.ResetSequenceState();
            using (vk.Forward(inputs[1], Enumerable.Range(0, inputs[1].Length).ToArray(), -1, null, true)) { }
            int pos = inputs[1].Length, next = 13;
            var swd = System.Diagnostics.Stopwatch.StartNew();
            const int N = 48;
            for (int sIdx = 0; sIdx < N; sIdx++) { using var lg = vk.Forward([next], [pos++], -1, null, true); }
            dec.Add(N / swd.Elapsed.TotalSeconds);
        }
        output.WriteLine($"SPEED prefill {longIds.Length} tok: {string.Join(" / ", pfs.Select(v => v.ToString("F0")))} tok/s; decode 48 tok: {string.Join(" / ", dec.Select(v => v.ToString("F1")))} tok/s; device live {device.LiveBytesTotal() / (double)(1L << 30):F2} GiB");
        {
            // Group skipping (decode D/U): identical numerics and the same-session speed A/B. Logits of 24 greedy decode steps are compared BITWISE skip-on vs skip-off.
            bool saved = VulkanQwen3MoeHybridTransformerModel.Edge0SkipUnusedGroups;
            float[][] RunDecode(bool skip)
            {
                VulkanQwen3MoeHybridTransformerModel.Edge0SkipUnusedGroups = skip;
                vk.ResetSequenceState();
                using (vk.Forward(inputs[1], Enumerable.Range(0, inputs[1].Length).ToArray(), -1, null, true)) { }
                int pos = inputs[1].Length, next = 13;
                var rows = new float[24][];
                for (int sIdx = 0; sIdx < 24; sIdx++)
                {
                    using var lg = vk.Forward([next], [pos++], -1, null, true);
                    int vocab = (int)lg.Shape[1];
                    var row = new float[vocab];
                    unsafe { new ReadOnlySpan<float>((void*)lg.DataPointer, vocab).CopyTo(row); }
                    rows[sIdx] = row; next = Argmax(row);
                }
                return rows;
            }
            long layersBefore = vk.Edge0SkipLayersForTest;
            var off = RunDecode(false);
            long afterOff = vk.Edge0SkipLayersForTest;
            var on = RunDecode(true);
            long afterOn = vk.Edge0SkipLayersForTest;
            int differing = 0;
            for (int t = 0; t < off.Length; t++) for (int i = 0; i < off[t].Length; i++) if (!off[t][i].Equals(on[t][i])) differing++;
            output.WriteLine($"SKIPCHECK 24 decode steps x vocab logits bitwise skip-off vs skip-on: {differing} differing values; skip path layers recorded off={afterOff - layersBefore} on={afterOn - afterOff}");
            if (differing != 0) failures.Add($"group skipping changed {differing} logits");
            if (afterOff != layersBefore || afterOn == afterOff) failures.Add("skip path did not run exactly when enabled");
            var tOff = new List<double>(); var tOn = new List<double>();
            for (int rep = 0; rep < 3; rep++)
                foreach (bool skip in new[] { false, true })
                {
                    VulkanQwen3MoeHybridTransformerModel.Edge0SkipUnusedGroups = skip;
                    vk.ResetSequenceState();
                    using (vk.Forward(inputs[1], Enumerable.Range(0, inputs[1].Length).ToArray(), -1, null, true)) { }
                    int pos = inputs[1].Length; const int N = 48;
                    var swd = System.Diagnostics.Stopwatch.StartNew();
                    for (int sIdx = 0; sIdx < N; sIdx++) { using var lg = vk.Forward([13], [pos++], -1, null, true); }
                    (skip ? tOn : tOff).Add(N / swd.Elapsed.TotalSeconds);
                }
            output.WriteLine($"SKIPSPEED decode interleaved A/B, tok/s: skip-off {string.Join(" / ", tOff.Select(v => v.ToString("F1")))}; skip-on {string.Join(" / ", tOn.Select(v => v.ToString("F1")))}");
            VulkanQwen3MoeHybridTransformerModel.Edge0SkipUnusedGroups = saved;
        }
        if (Environment.GetEnvironmentVariable("DOTLLM_EDGE0_STAGES") == "1")
        {
            // Stage profile of single-row decode (each stage ends in a submit + wait: totals overstate by the sync cost, the ranking is what matters).
            VulkanQwen4ExpTransformerModel.StageProfile = true;
            vk.ResetSequenceState();
            using (vk.Forward(inputs[1], Enumerable.Range(0, inputs[1].Length).ToArray(), -1, null, true)) { }
            vk.TakeStageTimes();
            int pos2 = inputs[1].Length;
            for (int sIdx = 0; sIdx < 16; sIdx++) { using var lg = vk.Forward([13], [pos2++], -1, null, true); }
            foreach (var kv in vk.TakeStageTimes().OrderByDescending(k => k.Value).Take(24))
                output.WriteLine($"STAGE {kv.Key}: {kv.Value / 16:F2} ms/token");
            VulkanQwen4ExpTransformerModel.StageProfile = false;
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    /// <summary>Per-layer divergence of the Vulkan residual stream from the CPU one (<c>DOTLLM_EDGE0_VK_E2E=trace</c>), on a short prompt (indexed MMVQ path) and a 40-token prefix (grouped coopmat path).</summary>
    [SkippableFact]
    public void Stage_LayerTrace()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_VK_E2E") == "trace", "DOTLLM_EDGE0_VK_E2E=trace (CPU then GPU; hold both locks)");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var fx = Locate();
        Skip.IfNot(fx.Found, fx.SkipMessage("Edge0-145B-q3k"));
        var all = Inputs(fx.Path!);
        int[][] cases = [all[3], all[0].AsSpan(0, 40).ToArray()];
        var cpuTraces = new List<Dictionary<string, float[]>>();
        {
            var (model, gguf, _) = ModelLoader.LoadFromGguf(fx.Path!, DotLLM.Core.Configuration.ThreadingConfig.Auto);
            using (gguf) using (model)
            {
                var cpu = (DotLLM.Models.Architectures.Qwen4ExpTransformerModel)model;
                foreach (var ids in cases)
                {
                    var d = new Dictionary<string, float[]>();
                    cpu.Trace = (n, data, r, c) => d[n] = data.ToArray();
                    model.ResetSequenceState();
                    using var _ = model.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), 0);
                    cpuTraces.Add(d);
                }
            }
        }
        using var g2 = GgufFile.Open(fx.Path!);
        using var device = VulkanDevice.Create();
        using var vk = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, g2, GgufModelConfigExtractor.Extract(g2.Metadata), spvDir);
        int defaultMin = vk.MoeGroupedMinTokens;
        for (int run = 0; run < cases.Length * 2; run++)
        {
            int k = run % cases.Length;
            bool flipped = run >= cases.Length;   // second pass: the OTHER expert path (grouped coopmat F16 operands vs indexed MMVQ int8 activations)
            vk.MoeGroupedMinTokens = flipped ? (cases[k].Length >= defaultMin ? 1_000_000 : 1) : defaultMin;
            var d = new Dictionary<string, float[]>();
            vk.Trace = (n, data, r, c) => d[n] = data;
            vk.ResetSequenceState();
            vk.TrySetAllRowLogitsLimit(cases[k].Length);
            using var _ = vk.Forward(cases[k], Enumerable.Range(0, cases[k].Length).ToArray(), -1);
            var sb = new System.Text.StringBuilder($"trace case {k} ({cases[k].Length} tokens, {(cases[k].Length >= vk.MoeGroupedMinTokens ? "grouped" : "indexed")} experts) rel-RMS(vk-cpu) per layer:");
            for (int il = 0; il < 48; il++)
            {
                var a = cpuTraces[k][$"blk.{il}.l_out"]; var b = d[$"blk.{il}.l_out"];
                double num = 0, den = 0;
                for (int i = 0; i < a.Length; i++) { double e = a[i] - b[i]; num += e * e; den += (double)a[i] * a[i]; }
                sb.Append($" {il}:{Math.Sqrt(num / den):E1}");
            }
            output.WriteLine(sb.ToString());
        }
    }
}
