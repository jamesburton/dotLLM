using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Real-file QSA measurement harness (issue #819). Skipped unless <c>DOTLLM_QWEN4EXP_REAL_GGUF</c> names shard 1 of the released file
/// (e.g. the Unsloth UD-Q4_K_XL). One model load serves every probe: needle-in-a-haystack prompts from
/// <c>DOTLLM_QWEN4EXP_NEEDLE_DIR</c> (<c>needle_*.txt</c>, answer code in <c>needle_*.answer</c> or the file name table below), chunked-prefill and
/// decode throughput at several context depths, and the device memory held. GPU box only: hold <c>scripts/gpu-lock.sh</c>.
/// </summary>
[Trait("Category", "GPU")]
[Trait("Category", "RealFile")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpRealQsaTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpRealQsaTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] Codes = ["plum-7391-harbor", "cedar-5520-lantern", "violet-8264-anchor"];

    private void Log(string s)
    {
        _out.WriteLine(s);
        string? path = Environment.GetEnvironmentVariable("DOTLLM_REAL_RESULTS");
        if (!string.IsNullOrEmpty(path)) File.AppendAllText(path, s + Environment.NewLine);
    }

    private static unsafe float[] LastRow(DotLLM.Core.Tensors.ITensor t)
    {
        using (t)
        {
            int v = t.Shape[1], rows = t.Shape[0];
            return new ReadOnlySpan<float>((void*)(t.DataPointer + (nint)((long)(rows - 1) * v * 4)), v).ToArray();
        }
    }

    private static int Argmax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    [SkippableFact]
    public void LoadAndOneForward()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        Log($"[smoke] load {sw.Elapsed.TotalSeconds:F1}s, capacity {model.ContextCapacity}, QSA enabled {VulkanQwen4ExpQsa.Enabled}, {device.MemorySnapshot()}");
        using var st = model.CreateState();
        var row = LastRow(model.Forward(Enumerable.Range(1000, 64).ToArray(), Enumerable.Range(0, 64).ToArray(), -1, st));
        Log($"[smoke] forward ok, argmax {Argmax(row)}, {device.MemorySnapshot()}");
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        for (int i = 0; i < p.Length; i++) { zp += Math.Exp(p[i] - mp); zq += Math.Exp(q[i] - mq); }
        double lzp = Math.Log(zp), lzq = Math.Log(zq), kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - mp - lzp;
            kl += Math.Exp(lp) * (lp - (q[i] - mq - lzq));
        }
        return kl;
    }

    /// <summary>
    /// Issue #437: same-process A/B of the K/V + indexer storage arms on identical token sequences. Every arm prefills the same text in 1024-row chunks;
    /// at each depth in <c>DOTLLM_REAL_KV_AB_DEPTHS</c> (default 4096,8192,16000) it records the chunk's last-row logits, then runs eight teacher-forced
    /// decode steps (timed, logits kept). The baseline arm (the first) is compared with every other arm by KL / top-1; eager vs lazy F32 must be bit-identical.
    /// Memory is the device snapshot at each depth. Text: <c>DOTLLM_REAL_KV_AB_TEXT</c>. Arms: <c>DOTLLM_REAL_KV_AB_ARMS</c> (default all).
    /// </summary>
    [SkippableFact]
    public void KvStorageAb_KlThroughputAndMemoryAtDepth()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        string? textPath = Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_AB_TEXT");
        Skip.If(string.IsNullOrEmpty(textPath) || !File.Exists(textPath), "DOTLLM_REAL_KV_AB_TEXT not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] depths = (Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_AB_DEPTHS") ?? "4096,8192,16000").Split(',').Select(int.Parse).ToArray();
        string[] armNames = (Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_AB_ARMS") ?? "f32-eager,f32-lazy,f16-lazy,f16-lazy-idx16").Split(',');
        const int Chunk = 1024, Steps = 16;

        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(File.ReadAllText(textPath!));
        int need = depths.Max() + Steps + 2;
        Skip.If(ids.Length < need, $"text has {ids.Length} tokens, needs {need}");
        Skip.If(depths.Max() + Steps + 1 > model.ContextCapacity, $"capacity {model.ContextCapacity} < depth {depths.Max() + Steps + 1}");
        Log($"[kvab] capacity {model.ContextCapacity}, text {ids.Length} tokens, depths {string.Join(",", depths)}, arms {string.Join(",", armNames)}");

        var oldKv = VulkanQwen4ExpTransformerModel.KvStorage; bool oldLazy = VulkanQwen4ExpTransformerModel.LazyKv, oldIdx = VulkanQwen4ExpTransformerModel.IndexerRawF16;
        var rows = new Dictionary<string, List<float[]>>();
        try
        {
            using (var warm = model.CreateState())
                LastRow(model.Forward(ids.AsSpan(0, 256), Enumerable.Range(0, 256).ToArray(), -1, warm));
            foreach (string arm in armNames)
            {
                VulkanQwen4ExpTransformerModel.KvStorage = arm.StartsWith("f16") ? VulkanKvStorage.F16 : VulkanKvStorage.F32;
                VulkanQwen4ExpTransformerModel.LazyKv = !arm.Contains("eager");
                VulkanQwen4ExpTransformerModel.IndexerRawF16 = arm.Contains("idx16");
                VulkanQwen4ExpQsa.SplitsOverride = arm.Contains("split1") ? 1 : 0;   // control: a pure summation-order change of the F32 path
                var kept = new List<float[]>(); rows[arm] = kept;
                using var st = model.CreateState();
                int at = 0;
                double preSec = 0;
                foreach (int depth in depths)
                {
                    var tp = Stopwatch.GetTimestamp();
                    int from = at;
                    float[]? last = null;
                    while (at < depth)
                    {
                        int n = Math.Min(Chunk, depth - at);
                        last = LastRow(model.Forward(ids.AsSpan(at, n), Enumerable.Range(at, n).ToArray(), -1, st));
                        at += n;
                    }
                    double sec = Stopwatch.GetElapsedTime(tp).TotalSeconds;
                    if (last is not null) kept.Add(last);
                    // One untimed decode step first: a lazy cache that was exactly full grows (and re-records) on it, which is a one-off stall, not a rate.
                    kept.Add(LastRow(model.Forward([ids[at]], [at], -1, st)));
                    at++;
                    var td = Stopwatch.GetTimestamp();
                    for (int i = 0; i < Steps; i++)
                    {
                        kept.Add(LastRow(model.Forward([ids[at]], [at], -1, st)));
                        at++;
                    }
                    double dms = Stopwatch.GetElapsedTime(td).TotalMilliseconds / Steps;
                    Log($"[kvab] {arm,-16} depth {depth,6}: prefill {(depth - from) / sec,6:F0} tok/s over {depth - from} rows, decode {1000 / dms:F1} tok/s ({dms:F1} ms/tok), {device.MemorySnapshot()}");
                    preSec += sec;
                }
                Log($"[kvab] {arm}: prefill total {preSec:F1}s");
            }
        }
        finally
        {
            VulkanQwen4ExpTransformerModel.KvStorage = oldKv; VulkanQwen4ExpTransformerModel.LazyKv = oldLazy; VulkanQwen4ExpTransformerModel.IndexerRawF16 = oldIdx;
            VulkanQwen4ExpQsa.SplitsOverride = 0;
        }
        string baseArm = armNames[0];
        foreach (string arm in armNames.Skip(1))
        {
            var a = rows[baseArm]; var b = rows[arm];
            double sum = 0, max = 0; int agree = 0; var perDepth = new List<string>();
            for (int i = 0; i < a.Count; i++)
            {
                double kl = Kl(a[i], b[i]); sum += kl; max = Math.Max(max, kl);
                if (Argmax(a[i]) == Argmax(b[i])) agree++;
                perDepth.Add(kl.ToString("E1"));
            }
            bool identical = a.Zip(b).All(p => p.First.AsSpan().SequenceEqual(p.Second));
            Log($"[kvab] KL({baseArm} || {arm}): mean {sum / a.Count:E2} max {max:E2} top1 {agree}/{a.Count} bit-identical {identical}; per row [{string.Join(" ", perDepth)}]");
            if (arm == "f32-lazy") Assert.True(identical, "lazy F32 KV must be bit-identical to eager F32 KV");
        }
    }

    /// <summary>
    /// Issue #437: throughput of the storage arms measured fairly. UMA bandwidth contention swings sequential A/B runs by 30-40 %, so all arms (live
    /// states) advance through the SAME text together: every 1024-row prefill chunk is timed per arm in rotating order, and at each depth the decode
    /// rate is timed in alternating rounds (<c>Rounds</c> x <c>Steps</c> teacher-forced steps per arm, medians reported). Env: <c>DOTLLM_REAL_KV_AB_TEXT</c>,
    /// <c>DOTLLM_REAL_KV_DEC_DEPTHS</c> (default 1024,4096,8192,16000), <c>DOTLLM_REAL_KV_DEC_ARMS</c> (default f32-lazy,f16-lazy).
    /// </summary>
    [SkippableFact]
    public void KvStorageDecodeAb_InterleavedArms()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        string? textPath = Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_AB_TEXT");
        Skip.If(string.IsNullOrEmpty(textPath) || !File.Exists(textPath), "DOTLLM_REAL_KV_AB_TEXT not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] depths = (Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_DEC_DEPTHS") ?? "1024,4096,8192,16000").Split(',').Select(int.Parse).ToArray();
        string[] armNames = (Environment.GetEnvironmentVariable("DOTLLM_REAL_KV_DEC_ARMS") ?? "f32-lazy,f16-lazy").Split(',');
        const int Chunk = 1024, Steps = 16, Rounds = 5;

        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(File.ReadAllText(textPath!));
        int need = depths.Max() + 1 + Steps * Rounds + 2;
        Skip.If(ids.Length < need, $"text has {ids.Length} tokens, needs {need}");
        Skip.If(need > model.ContextCapacity, $"capacity {model.ContextCapacity} < {need}");

        var oldKv = VulkanQwen4ExpTransformerModel.KvStorage; bool oldLazy = VulkanQwen4ExpTransformerModel.LazyKv, oldIdx = VulkanQwen4ExpTransformerModel.IndexerRawF16;
        var states = new List<VulkanQwen4ExpSequenceState>();
        try
        {
            foreach (string arm in armNames)
            {
                VulkanQwen4ExpTransformerModel.KvStorage = arm.StartsWith("f16") ? VulkanKvStorage.F16 : VulkanKvStorage.F32;
                VulkanQwen4ExpTransformerModel.LazyKv = !arm.Contains("eager");
                VulkanQwen4ExpTransformerModel.IndexerRawF16 = arm.Contains("idx16");
                states.Add(model.CreateState());
            }
            using (var warm = model.CreateState())
                LastRow(model.Forward(ids.AsSpan(0, 256), Enumerable.Range(0, 256).ToArray(), -1, warm));
            var pos = new int[armNames.Length];
            var prefillSec = new double[armNames.Length][];
            for (int a = 0; a < armNames.Length; a++) prefillSec[a] = new double[3];   // [dense chunks, sparse chunks, rows counted as sparse]
            int chunkNo = 0;
            foreach (int depth in depths)
            {
                while (pos[0] < depth)
                {
                    int n = Math.Min(Chunk, depth - pos[0]);
                    for (int k = 0; k < armNames.Length; k++)
                    {
                        int a = (k + chunkNo) % armNames.Length;
                        var t0 = Stopwatch.GetTimestamp();
                        LastRow(model.Forward(ids.AsSpan(pos[a], n), Enumerable.Range(pos[a], n).ToArray(), -1, states[a]));
                        double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                        bool sparse = pos[a] + n > model.DenseContextLimit;
                        prefillSec[a][sparse ? 1 : 0] += sec;
                        if (sparse) prefillSec[a][2] += n;
                        pos[a] += n;
                    }
                    chunkNo++;
                }
                var ms = armNames.Select(_ => new List<double>()).ToArray();
                for (int a = 0; a < armNames.Length; a++)   // one untimed step per state (growth stall, descriptor rebuild)
                { LastRow(model.Forward([ids[pos[a]]], [pos[a]], -1, states[a])); pos[a]++; }
                for (int r = 0; r < Rounds; r++)
                    for (int k = 0; k < armNames.Length; k++)
                    {
                        int a = (k + r) % armNames.Length;
                        var t0 = Stopwatch.GetTimestamp();
                        for (int i = 0; i < Steps; i++) { LastRow(model.Forward([ids[pos[a]]], [pos[a]], -1, states[a])); pos[a]++; }
                        ms[a].Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds / Steps);
                    }
                for (int a = 0; a < armNames.Length; a++)
                {
                    var s = ms[a].OrderBy(x => x).ToArray();
                    Log($"[kvdec] depth {depth,6} {armNames[a],-12}: decode median {s[s.Length / 2]:F1} ms/token ({1000 / s[s.Length / 2]:F1} tok/s) rounds [{string.Join(", ", ms[a].Select(x => x.ToString("F1")))}]");
                }
                // rotate so that every arm reaches the next depth together (the decode steps advanced all arms equally)
            }
            for (int a = 0; a < armNames.Length; a++)
                Log($"[kvdec] prefill {armNames[a],-12}: dense chunks {prefillSec[a][0]:F1}s, sparse chunks {prefillSec[a][1]:F1}s ({prefillSec[a][2] / Math.Max(prefillSec[a][1], 1e-9):F0} tok/s over {prefillSec[a][2]:F0} rows), live states {states[a].Length} tokens");
            Log($"[kvdec] growths {model.CacheGrowths}, {model.CacheGrowthSeconds:F2}s total; {device.MemorySnapshot()}");
        }
        finally
        {
            foreach (var s in states) s.Dispose();
            VulkanQwen4ExpTransformerModel.KvStorage = oldKv; VulkanQwen4ExpTransformerModel.LazyKv = oldLazy; VulkanQwen4ExpTransformerModel.IndexerRawF16 = oldIdx;
        }
    }

    [SkippableFact]
    public void NeedlesThroughputAndMemory()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? needleDir = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_NEEDLE_DIR");
        int chunk = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_REAL_CHUNK"), out int c) && c > 0 ? c : 1024;

        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Log($"[real] load {sw.Elapsed.TotalSeconds:F1}s, context capacity {model.ContextCapacity}, dense limit {model.DenseContextLimit}, {device.MemorySnapshot()}");

        int V = cfg.VocabSize;
        float[] Prefill(VulkanQwen4ExpSequenceState st, int[] ids, out double sec)
        {
            var t0 = Stopwatch.GetTimestamp();
            float[]? last = null;
            for (int a = 0; a < ids.Length; a += chunk)
            {
                int n = Math.Min(chunk, ids.Length - a);
                last = LastRow(model.Forward(ids.AsSpan(a, n), Enumerable.Range(a, n).ToArray(), -1, st));
            }
            sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            return last!;
        }

        // Warm-up (shader/pipeline first use, weight page-in) so the timed figures below are steady state.
        using (var warm = model.CreateState())
            Prefill(warm, Enumerable.Range(1000, 256).ToArray(), out _);

        // Same-session A/B of what the indexer costs at short context (decode at depth 1024, indexer off vs on, alternating).
        if (Environment.GetEnvironmentVariable("DOTLLM_REAL_SKIP_AB") != "1")   // two live states: skip at big capacities (each state holds the full K/V + indexer allocation)
        {
            var rngAb = new Random(7);
            int[] fill = Enumerable.Range(0, 1100).Select(_ => rngAb.Next(1000, 20000)).ToArray();
            VulkanQwen4ExpQsa.Enabled = false;
            using var off = model.CreateState();
            Prefill(off, fill.AsSpan(0, 1024).ToArray(), out _);
            VulkanQwen4ExpQsa.Enabled = true;
            using var on = model.CreateState();
            Prefill(on, fill.AsSpan(0, 1024).ToArray(), out _);
            double[] offMs = new double[3], onMs = new double[3];
            for (int round = 0; round < 3; round++)
            {
                foreach (bool useIndexer in new[] { false, true })
                {
                    VulkanQwen4ExpQsa.Enabled = useIndexer;
                    var st = useIndexer ? on : off;
                    int keep = st.Length;
                    var t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < 24; i++)
                        LastRow(model.Forward([fill[(keep + i) % fill.Length]], [st.Length], -1, st));
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / 24;
                    (useIndexer ? onMs : offMs)[round] = ms;
                }
            }
            VulkanQwen4ExpQsa.Enabled = true;
            Log($"[ab] decode @1024, ms/token: indexer OFF [{string.Join(", ", offMs.Select(x => x.ToString("F1")))}] ON [{string.Join(", ", onMs.Select(x => x.ToString("F1")))}]");
        }

        int passed = 0, total = 0;
        if (!string.IsNullOrEmpty(needleDir) && Directory.Exists(needleDir))
        {
            foreach (string file in Directory.GetFiles(needleDir, "needle_*.txt").OrderBy(x => x))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                int idx = int.Parse(name[^1..]);
                string code = Codes[idx];
                int[] ids = tokenizer.Encode(File.ReadAllText(file));
                if (ids.Length + 40 > model.ContextCapacity)
                {
                    Log($"[real] {name}: {ids.Length} tokens exceed the capacity {model.ContextCapacity} - skipped");
                    continue;
                }
                using var st = model.CreateState();
                var logits = Prefill(st, ids, out double psec);
                var gen = new List<int>();
                var d0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < 24; i++)
                {
                    int next = Argmax(logits);
                    gen.Add(next);
                    logits = LastRow(model.Forward([next], [st.Length], -1, st));
                }
                double dsec = Stopwatch.GetElapsedTime(d0).TotalSeconds;
                string text = tokenizer.Decode(gen.ToArray());
                bool ok = text.Contains(code, StringComparison.OrdinalIgnoreCase);
                total++; if (ok) passed++;
                Log($"[needle] {name}: {ids.Length} tokens, prefill {psec:F1}s ({ids.Length / psec:F0} tok/s), decode {24 / dsec:F1} tok/s, " +
                    $"answer \"{text.Replace("\n", " ")}\" expected \"{code}\" -> {(ok ? "PASS" : "FAIL")}");
            }
            Log($"[needle] {passed}/{total} passed");
        }

        // Decode throughput vs context depth (same state, growing): 1K, 2K (dense limit), 4K, 6K, 8K.
        using (var st = model.CreateState())
        {
            var rng = new Random(1);
            int[] filler = Enumerable.Range(0, model.ContextCapacity).Select(_ => rng.Next(1000, 20000)).ToArray();
            float[]? logits = null;
            int at = 0;
            foreach (int depth in new[] { 1024, 2048, 4096, 8000, 12000, 16000, 24000, 32000, 48000, 64000 })
            {
                if (depth + 40 > model.ContextCapacity) break;
                if (depth > at)
                {
                    var t0 = Stopwatch.GetTimestamp();
                    for (int a = at; a < depth; a += chunk)
                    {
                        int n = Math.Min(chunk, depth - a);
                        logits = LastRow(model.Forward(filler.AsSpan(a, n), Enumerable.Range(a, n).ToArray(), -1, st));
                        at = a + n;
                    }
                    double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                    Log($"[prefill] to depth {depth}: {sec:F1}s");
                }
                var d0 = Stopwatch.GetTimestamp();
                const int Steps = 24;
                for (int i = 0; i < Steps; i++)
                {
                    logits = LastRow(model.Forward([filler[at % filler.Length]], [st.Length], -1, st));
                    at++;
                }
                double dsec = Stopwatch.GetElapsedTime(d0).TotalSeconds;
                Log($"[decode] depth ~{depth}: {Steps / dsec:F1} tok/s ({dsec / Steps * 1000:F1} ms/token), state length {st.Length}");
            }
        }
        Log($"[real] done in {sw.Elapsed.TotalSeconds:F0}s, {device.MemorySnapshot()}");
        if (total > 0) Assert.True(passed == total, $"needles {passed}/{total}");
    }
}
