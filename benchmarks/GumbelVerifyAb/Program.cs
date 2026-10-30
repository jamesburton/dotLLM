// Diagnostics tool (not shipped) for issue #1382: device-side Gumbel-max over speculative verify rows (Vulkan).
//   GumbelVerifyAb parity <gguf> [positions=24] [rows=4] [seeds=1000]   batched-vs-single-row logits bits + Gumbel argmax flip rate (no timing)
//   GumbelVerifyAb ab     <gguf> [reps=3] [inner=5] [ctx=64] [rows=2,4,8] paired verify-round A/B: host (readback + CPU Gumbel) vs device
//   GumbelVerifyAb e2e    <gguf> [reps=3] [tokens=96]                      TextGenerator MTP, T=0.7, Gumbel: device vs host (text identity + tok/s)
// Timing modes need the GPU lock and the cpu-bench lock held by the caller.
using System.Diagnostics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

string mode = args[0];
string path = args[1];
var gguf = GgufFile.Open(path);
var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
var vdev = VulkanDevice.Create();
string spvDir = VulkanModelLoader.ResolveSpvDir();
var (model, kvf) = VulkanModelLoader.CreateFromGguf(vdev, gguf, cfg, spvDir);
var vm = (VulkanQwen3HybridDenseTransformerModel)model;
Console.WriteLine($"[gva] {Path.GetFileName(path)} model={model.GetType().Name} vocab={cfg.VocabSize} mtp={model.SupportsMtp} deviceGumbel={model.SupportsDeviceGumbelSampling} maxRows={model.DeviceGumbelMaxRows} snapshots={model.SupportsRecurrentRowSnapshots}");

float Temperature = float.Parse(Environment.GetEnvironmentVariable("GVA_T") ?? "0.7", System.Globalization.CultureInfo.InvariantCulture);
float invT = 1f / Temperature;
string Chat(string q) => "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
static double Med(List<double> l) { var s = l.OrderBy(x => x).ToList(); return s[s.Count / 2]; }

switch (mode)
{
    case "parity": Parity(); break;
    case "ab": Ab(); break;
    case "e2e": E2e(); break;
    default: throw new ArgumentException(mode);
}
(model as IDisposable)?.Dispose();
vdev.Dispose();
return;

void Parity()
{
    int positions = args.Length > 2 ? int.Parse(args[2]) : 24;
    int rows = args.Length > 3 ? int.Parse(args[3]) : 4;
    int seeds = args.Length > 4 ? int.Parse(args[4]) : 1000;
    int V = cfg.VocabSize;

    // Realistic text: a greedy continuation of a chat prompt.
    var gen = new TextGenerator(model, tokenizer, (c, n) => kvf(n), mtpEnabled: false);
    var promptIds = tokenizer.Encode(Chat("Write a long, detailed story about a lighthouse keeper, a storm, and an unexpected visitor. Do not stop early."));
    var resp = gen.Generate(Chat("Write a long, detailed story about a lighthouse keeper, a storm, and an unexpected visitor. Do not stop early."),
        new InferenceOptions { Temperature = 0, MaxTokens = positions * 6 + rows + 8 });
    var all = promptIds.Concat(resp.GeneratedTokenIds).ToArray();
    Console.WriteLine($"[gva] text tokens: prompt {promptIds.Length}, total {all.Length}");

    using var kernel = GumbelArgMaxKernel.Create(vdev, spvDir);
    using var bufSeq = vdev.AllocateHostReadback((long)rows * V * sizeof(float));
    using var bufBat = vdev.AllocateHostReadback((long)rows * V * sizeof(float));
    using var idsSeq = vdev.AllocateHostReadback(rows * sizeof(uint));
    using var idsBat = vdev.AllocateHostReadback(rows * sizeof(uint));
    var seq = new float[rows * V];
    var bat = new float[rows * V];
    var gotSeq = new int[rows];
    var gotBat = new int[rows];

    long elems = 0, diffElems = 0, rowsTotal = 0, rowsBitEqual = 0;
    double maxAbs = 0; long maxUlp = 0;
    long draws = 0, flips = 0, argmaxFlips = 0;
    var flipsByRow = new long[rows];
    var kv = kvf(all.Length + 8);
    for (int pi = 0; pi < positions; pi++)
    {
        int stride = Math.Max(1, (all.Length - promptIds.Length - rows - 4) / positions);
        int p = promptIds.Length + 2 + pi * stride;           // inside the generated text
        if (p + rows >= all.Length) break;
        model.ResetSequenceState();
        kv.Rollback(0);
        for (int s = 0; s < p; s += 32)
        {
            int n = Math.Min(32, p - s);
            using var w = model.Forward(all.AsSpan(s, n), Enumerable.Range(s, n).ToArray(), -1, kv);
        }
        object ckpt = model.CheckpointRecurrentState()!;
        // plain decode: one row per step
        for (int r = 0; r < rows; r++)
        {
            using var lg = model.Forward(all.AsSpan(p + r, 1), [p + r], -1, kv);
            unsafe { new ReadOnlySpan<float>((void*)lg.DataPointer, V).CopyTo(seq.AsSpan(r * V, V)); }
        }
        kv.Rollback(p); model.RestoreRecurrentState(ckpt);
        // verify: all rows in one batched forward
        using (var lg = model.ForwardWithRecurrentSnapshots(all.AsSpan(p, rows), Enumerable.Range(p, rows).ToArray(), -1, kv, null))
        {
            unsafe { new ReadOnlySpan<float>((void*)lg.DataPointer, rows * V).CopyTo(bat); }
        }
        kv.Rollback(p); model.RestoreRecurrentState(ckpt);

        for (int r = 0; r < rows; r++)
        {
            rowsTotal++;
            bool eq = true;
            for (int i = 0; i < V; i++)
            {
                elems++;
                int a = BitConverter.SingleToInt32Bits(seq[r * V + i]), b = BitConverter.SingleToInt32Bits(bat[r * V + i]);
                if (a != b)
                {
                    eq = false; diffElems++;
                    maxAbs = Math.Max(maxAbs, Math.Abs(seq[r * V + i] - bat[r * V + i]));
                    long ua = a < 0 ? int.MinValue - (long)a : a, ub = b < 0 ? int.MinValue - (long)b : b;
                    maxUlp = Math.Max(maxUlp, Math.Abs(ua - ub));
                }
            }
            if (eq) rowsBitEqual++;
            if (TensorPrimitives.IndexOfMax(seq.AsSpan(r * V, V)) != TensorPrimitives.IndexOfMax(bat.AsSpan(r * V, V))) argmaxFlips++;
        }
        vdev.Upload(seq, bufSeq);
        vdev.Upload(bat, bufBat);
        for (int s = 0; s < seeds; s++)
        {
            ulong seed = 0x9E3779B97F4A7C15UL * (ulong)(s + 1) + (ulong)pi;
            kernel.Launch(bufSeq, idsSeq, rows, V, V, seed, p, invT);
            kernel.Launch(bufBat, idsBat, rows, V, V, seed, p, invT);
            vdev.Download(idsSeq, MemoryMarshal.Cast<int, float>(gotSeq.AsSpan()));
            vdev.Download(idsBat, MemoryMarshal.Cast<int, float>(gotBat.AsSpan()));
            for (int r = 0; r < rows; r++) { draws++; if (gotSeq[r] != gotBat[r]) { flips++; flipsByRow[r]++; } }
        }
        Console.WriteLine($"[gva] pos {p}: rows bit-equal so far {rowsBitEqual}/{rowsTotal}, flips {flips}/{draws}");
    }
    Console.WriteLine($"RESULT parity: rows={rowsTotal} bitEqualRows={rowsBitEqual} differingElems={diffElems}/{elems} maxAbsDiff={maxAbs:E2} maxUlp={maxUlp} greedyArgmaxFlips={argmaxFlips}");
    Console.WriteLine($"RESULT gumbel: T={Temperature} draws={draws} flips={flips} rate={(double)flips / Math.Max(1, draws):E3} perRowFlips=[{string.Join(",", flipsByRow)}]");
}

void Ab()
{
    int reps = args.Length > 2 ? int.Parse(args[2]) : 3;
    int inner = args.Length > 3 ? int.Parse(args[3]) : 5;
    int ctx = args.Length > 4 ? int.Parse(args[4]) : 64;
    int[] rowsList = (args.Length > 5 ? args[5] : "2,4,8").Split(',').Select(int.Parse).ToArray();
    int V = cfg.VocabSize;
    var rng = new Random(1234);
    int[] Ids(int n) { var a = new int[n]; for (int i = 0; i < n; i++) a[i] = rng.Next(1000, 20000); return a; }
    var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = Temperature, Seed = 7, GumbelSampling = true });
    float invTp = pipeline.DeviceGumbelInvTemperature ?? throw new InvalidOperationException("pipeline not device-eligible");
    var kv = kvf(ctx + rowsList.Max() + 8);
    model.ResetSequenceState();
    var ctxIds = Ids(ctx);
    for (int p = 0; p < ctx; p += 32)
    {
        int n = Math.Min(32, ctx - p);
        using var w = model.Forward(ctxIds.AsSpan(p, n), Enumerable.Range(p, n).ToArray(), -1, kv);
    }
    object ckpt = model.CheckpointRecurrentState()!;
    var history = new List<int>();
    for (int i = 0; i < 40; i++) history.Add(1);
    var sink = new int[32];

    // arms: 0 = host (verify forward, rows read back, PrepareDistribution + Gumbel per row, as the decoder does)
    //       1 = device (ForwardGumbelSampled)   2 = readback only (verify forward, no sampling: the floor of arm 0)
    double Time(int arm, int rows)
    {
        var ids = Ids(rows);
        var pos = Enumerable.Range(ctx, rows).ToArray();
        long t0 = Stopwatch.GetTimestamp();
        if (arm == 1)
            model.ForwardGumbelSampled(ids, pos, -1, kv, null, true, pipeline.GumbelSeed, history.Count, invTp, sink);
        else
        {
            using var lg = model.ForwardWithRecurrentSnapshots(ids, pos, -1, kv, null);
            if (arm == 0)
                unsafe
                {
                    for (int r = 0; r < rows; r++)
                    {
                        var row = new Span<float>((void*)(lg.DataPointer + (long)r * V * sizeof(float)), V);
                        pipeline.PrepareDistribution(row, history);
                        sink[r] = pipeline.SamplePrepared(row, history.Count + r);
                    }
                }
        }
        double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
        kv.Rollback(ctx);
        model.RestoreRecurrentState(ckpt);
        return ms;
    }
    foreach (int r in rowsList) for (int a = 0; a < 3; a++) { Time(a, r); Time(a, r); }
    int before = vm.DeviceGumbelForwards;
    var lists = rowsList.ToDictionary(r => r, _ => new[] { new List<double>(), new List<double>(), new List<double>() });
    var repMed = rowsList.ToDictionary(r => r, _ => new[] { new List<double>(), new List<double>(), new List<double>() });
    var sw = Stopwatch.StartNew();
    for (int rep = 0; rep < reps; rep++)
    {
        foreach (int r in (rep % 2 == 0 ? rowsList : rowsList.Reverse().ToArray()))
        {
            var cur = new[] { new List<double>(), new List<double>(), new List<double>() };
            for (int i = 0; i < inner; i++)
            {
                int[] order = ((rep + i) & 1) == 0 ? [0, 1, 2] : [2, 1, 0];
                foreach (int a in order) cur[a].Add(Time(a, r));
            }
            for (int a = 0; a < 3; a++) { lists[r][a].AddRange(cur[a]); repMed[r][a].Add(Med(cur[a])); }
        }
        Console.WriteLine($"[gva] ab rep {rep + 1}/{reps} done {sw.Elapsed.TotalSeconds:F0} s");
    }
    Console.WriteLine($"[gva] device forwards counted: {vm.DeviceGumbelForwards - before} (expected {reps * inner * rowsList.Length})");
    Console.WriteLine("RESULT ab rows\tarm\tmin\tmed\tmax\trepMedians");
    string[] names = ["host(readback+cpu-gumbel)", "device", "readback-only"];
    foreach (int r in rowsList)
        for (int a = 0; a < 3; a++)
            Console.WriteLine($"RESULT ab {r}\t{names[a]}\t{lists[r][a].Min():F2}\t{Med(lists[r][a]):F2}\t{lists[r][a].Max():F2}\t{string.Join(" ", repMed[r][a].Select(x => x.ToString("F2")))}");
}

void E2e()
{
    int reps = args.Length > 2 ? int.Parse(args[2]) : 3;
    int tokens = args.Length > 3 ? int.Parse(args[3]) : 96;
    string[] questions = ["Describe how a refrigerator works.", "Explain why the sky is blue in two sentences.", "List three uses for a paperclip."];
    InferenceOptions Opts(int seed) => new() { Temperature = Temperature, Seed = seed, GumbelSampling = true, MaxTokens = tokens };

    (int[] Ids, double Ms, int Rounds) Run(string arm, string q, int seed)
    {
        Environment.SetEnvironmentVariable("DOTLLM_DEVICE_GUMBEL", arm == "host" ? "0" : "1");
        var gen = new TextGenerator(model, tokenizer, (c, n) => kvf(n), mtpEnabled: arm != "plain");
        int before = vm.DeviceGumbelForwards;
        var sw = Stopwatch.StartNew();
        var r = gen.Generate(Chat(q), Opts(seed));
        double ms = sw.Elapsed.TotalMilliseconds;
        int dev = vm.DeviceGumbelForwards - before;
        if (arm == "device" && dev == 0) Console.WriteLine("[gva] WARNING: device arm made no device draws");
        if (arm != "device" && dev != 0) Console.WriteLine("[gva] WARNING: non-device arm made device draws");
        return (r.GeneratedTokenIds, r.Timings.DecodeTimeMs, dev);
    }

    foreach (var q in questions) { Run("device", q, 1); Run("host", q, 1); }
    var tps = new Dictionary<string, List<double>> { ["plain"] = [], ["host"] = [], ["device"] = [] };
    int hostVsDevice = 0, hostVsDeviceTotal = 0, plainVsSpec = 0, plainVsSpecTotal = 0; long tokEq = 0, tokTot = 0;
    for (int rep = 0; rep < reps; rep++)
        for (int qi = 0; qi < questions.Length; qi++)
        {
            int seed = 100 + rep * 10 + qi;
            string[] order = ((rep + qi) & 1) == 0 ? ["plain", "host", "device"] : ["device", "host", "plain"];
            var res = new Dictionary<string, (int[] Ids, double Ms, int Dev)>();
            foreach (var arm in order) { var (ids, ms, dev) = Run(arm, questions[qi], seed); res[arm] = (ids, ms, dev); tps[arm].Add(ids.Length * 1000.0 / ms); }
            hostVsDeviceTotal++; if (res["host"].Ids.AsSpan().SequenceEqual(res["device"].Ids)) hostVsDevice++;
            plainVsSpecTotal++; if (res["plain"].Ids.AsSpan().SequenceEqual(res["device"].Ids)) plainVsSpec++;
            int common = 0; int n = Math.Min(res["plain"].Ids.Length, res["device"].Ids.Length);
            while (common < n && res["plain"].Ids[common] == res["device"].Ids[common]) common++;
            tokEq += common; tokTot += res["plain"].Ids.Length;
            Console.WriteLine($"[gva] rep {rep} q{qi} seed {seed}: host==device {res["host"].Ids.AsSpan().SequenceEqual(res["device"].Ids)}, plain==device {res["plain"].Ids.AsSpan().SequenceEqual(res["device"].Ids)} (common prefix {common}/{res["plain"].Ids.Length}), " +
                              $"tok/s plain {tps["plain"][^1]:F1} host {tps["host"][^1]:F1} device {tps["device"][^1]:F1}, deviceRounds {res["device"].Dev}");
        }
    Console.WriteLine($"RESULT e2e host==device {hostVsDevice}/{hostVsDeviceTotal}; plain==device {plainVsSpec}/{plainVsSpecTotal}; plain/device common-prefix tokens {tokEq}/{tokTot}");
    foreach (var (k, v) in tps) Console.WriteLine($"RESULT e2e tok/s {k}: min {v.Min():F1} med {Med(v):F1} max {v.Max():F1}");
}
