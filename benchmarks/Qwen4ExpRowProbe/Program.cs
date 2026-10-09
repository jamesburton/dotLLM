using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

// Model-resident driver for the qwen4exp small-row work (#876). The real-file load takes ~160 s, so this process loads ONCE and then
// executes job files dropped into <workdir>/jobs: <id>.job (one command per line) -> <id>.out, then an empty <id>.done marker.
//
// Commands:
//   fwd <rows> <reps> [ctx=16]     wall ms of a forward over <rows> tokens after <ctx> context tokens (fresh state each rep): min/median/mean
//   stage <rows> <reps> [ctx=16]   split-submit per-stage wall ms, averaged per forward (sync cost ~50us/stage is included)
//   dump <name> <rows> [tokens=48] prefill <tokens> fixed pseudo-random ids in chunks of <rows>; store each chunk's last-row logits in <workdir>/<name>.f32
//   kl <name> <rows> [tokens=48]   same run, compared with the stored <name>.f32 (mean/max KL(stored||now), max |dlogit|, top-1 agree)
//   env NAME=VALUE                 set a process env var (for switches read per call)
//   quit
const string Marker = "[probe]";
if (args[0] == "kbench") { KBench.Run(args); return; }
string gguf = args[0];
string work = args[1];
Directory.CreateDirectory(Path.Combine(work, "jobs"));
string lockName = args.Length > 2 ? args[2] : "";

void Log(string s) { Console.WriteLine($"{Marker} {DateTime.Now:HH:mm:ss} {s}"); Console.Out.Flush(); }

var sw0 = Stopwatch.StartNew();
using var file = GgufFile.Open(gguf);
var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
using var device = VulkanDevice.Create();
string spv = Path.Combine(AppContext.BaseDirectory, "spv");
using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, file, cfg, spv);
Log($"loaded in {sw0.Elapsed.TotalSeconds:F0} s");
int vocab = cfg.VocabSize;
var tok = GgufTokenizerFactory.Load(file.Metadata);
string[] prompts =
{
    "The capital of France is Paris, and the capital of Germany is",
    "Mixture-of-experts models route each token to a small subset of expert networks, which lets the total parameter count grow much faster than the compute per token. Explain in detail how the router is trained and why load balancing matters for throughput.",
    "def is_palindrome(s: str) -> bool:" + (char)10 + "    cleaned = " + "''" + ".join(c.lower() for c in s if c.isalnum())" + (char)10 + "    return cleaned == cleaned[::-1]" + (char)10 + (char)10 + "# Write unit tests for the function above",
    "The French Revolution began in 1789 and transformed the political landscape of Europe. Key events included the storming of the Bastille, the Declaration of the Rights of Man, the Reign of Terror, and the rise of Napoleon Bonaparte, who eventually crowned himself emperor in 1804. Summarize the main causes.",
};

bool realText = true;
int[] longText = null!;
int[] Ids(int n, int seed)
{
    if (realText)
    {
        longText ??= Enumerable.Range(0, 8).SelectMany(_ => prompts.SelectMany(p => tok.Encode(p))).ToArray();
        // seed 11 = the shared context; other seeds pick distinct windows after it
        int start = seed == 11 ? 0 : 16 + ((seed * 37) % 200);
        return Enumerable.Range(0, n).Select(i => longText[(start + i) % longText.Length]).ToArray();
    }
    var r = new Random(seed);
    var a = new int[n];
    for (int i = 0; i < n; i++) a[i] = r.Next(1000, 60000);
    return a;
}

unsafe float[] Run(int rows, int tokens, out List<double> perChunkMs)
{
    perChunkMs = new();
    var ids = Ids(tokens, 7);
    using var st = model.CreateState();
    var outv = new List<float>();
    for (int p = 0; p < tokens; p += rows)
    {
        int n = Math.Min(rows, tokens - p);
        var pos = Enumerable.Range(p, n).ToArray();
        var t0 = Stopwatch.GetTimestamp();
        using var lg = model.Forward(ids.AsSpan(p, n), pos, -1, st);
        perChunkMs.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
        outv.AddRange(new ReadOnlySpan<float>((void*)lg.DataPointer, vocab).ToArray());
    }
    return outv.ToArray();
}

double TimeFwd(int rows, int ctx, int seed)
{
    var ctxIds = Ids(ctx, 11);
    var ids = Ids(rows, seed);
    using var st = model.CreateState();
    if (ctx > 0)
    {
        using var w = model.Forward(ctxIds, Enumerable.Range(0, ctx).ToArray(), -1, st);
    }
    var pos = Enumerable.Range(ctx, rows).ToArray();
    var t0 = Stopwatch.GetTimestamp();
    using var lg = model.Forward(ids, pos, -1, st);
    return Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
}

void Exec(string line, StringBuilder o)
{
    var a = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    if (a.Length == 0) return;
    switch (a[0])
    {
        case "smallrow":
            VulkanQwen4ExpTransformerModel.SmallRowGemv = a[1] == "1";
            o.AppendLine("smallrow " + a[1]);
            break;
        case "ratio":
        {
            int rows = int.Parse(a[1]), reps = int.Parse(a[2]);
            TimeFwd(1, 16, 1); TimeFwd(rows, 16, 1);
            var t1 = new List<double>(); var tn = new List<double>();
            for (int i = 0; i < reps; i++) { t1.Add(TimeFwd(1, 16, 2 + i)); tn.Add(TimeFwd(rows, 16, 2 + i)); }
            t1.Sort(); tn.Sort();
            o.AppendLine($"ratio rows={rows}: 1-row med {t1[t1.Count / 2]:F1} min {t1[0]:F1} | {rows}-row med {tn[tn.Count / 2]:F1} min {tn[0]:F1} | med ratio {tn[tn.Count / 2] / t1[t1.Count / 2]:F2} min ratio {tn[0] / t1[0]:F2}");
            break;
        }
        case "grouped":
            model.MoeGroupedMinTokens = int.Parse(a[1]);
            o.AppendLine("grouped " + a[1]);
            break;
        case "text":
            realText = a[1] == "real";
            o.AppendLine("text " + a[1]);
            break;
        case "alt1":
        {
            // 1-row decode A/B, interleaved: multi-row MoE MMVQ off (min rows 2 = default) vs on at 1 row
            int reps = int.Parse(a[1]);
            var d = new List<double>(); var e = new List<double>();
            int saved = VulkanQwen4ExpTransformerModel.MoeMultiRowMinRows;
            TimeFwd(1, 16, 1);
            for (int i = 0; i < reps; i++)
            {
                VulkanQwen4ExpTransformerModel.MoeMultiRowMinRows = 2; d.Add(TimeFwd(1, 16, 2 + i));
                VulkanQwen4ExpTransformerModel.MoeMultiRowMinRows = 1; e.Add(TimeFwd(1, 16, 2 + i));
            }
            VulkanQwen4ExpTransformerModel.MoeMultiRowMinRows = saved;
            d.Sort(); e.Sort();
            o.AppendLine($"alt1: default 1-row med {d[d.Count / 2]:F1} min {d[0]:F1} | MR-at-1-row med {e[e.Count / 2]:F1} min {e[0]:F1}");
            break;
        }
        case "moemr":
            VulkanQwen4ExpTransformerModel.MoeMultiRowMinRows = int.Parse(a[1]);
            o.AppendLine("moemr " + a[1]);
            break;
        case "split":
            VulkanQwen4ExpTransformerModel.SplitAbove = int.Parse(a[1]);
            o.AppendLine("split " + a[1]);
            break;
        case "env":
        {
            var kv = a[1].Split('=', 2);
            Environment.SetEnvironmentVariable(kv[0], kv[1].Length == 0 ? null : kv[1]);
            o.AppendLine($"env {a[1]}");
            break;
        }
        case "fwd":
        {
            int rows = int.Parse(a[1]), reps = int.Parse(a[2]), ctx = a.Length > 3 ? int.Parse(a[3]) : 16;
            TimeFwd(rows, ctx, 1);   // warm
            var t = new List<double>();
            for (int i = 0; i < reps; i++) t.Add(TimeFwd(rows, ctx, 2 + i));
            t.Sort();
            o.AppendLine($"fwd rows={rows} ctx={ctx} reps={reps}: min {t[0]:F1} med {t[t.Count / 2]:F1} mean {t.Average():F1} max {t[^1]:F1} ms");
            break;
        }
        case "stage":
        {
            int rows = int.Parse(a[1]), reps = int.Parse(a[2]), ctx = a.Length > 3 ? int.Parse(a[3]) : 16;
            TimeFwd(rows, ctx, 1);
            VulkanQwen4ExpTransformerModel.StageProfile = true;
            model.TakeStageTimes();
            double tot = 0;
            var sums = new Dictionary<string, double>();
            for (int i = 0; i < reps; i++)
            {
                var ctxIds = Ids(ctx, 11);
                using var st = model.CreateState();
                if (ctx > 0) { using var w = model.Forward(ctxIds, Enumerable.Range(0, ctx).ToArray(), -1, st); }
                model.TakeStageTimes();
                var ids = Ids(rows, 2 + i);
                var t0 = Stopwatch.GetTimestamp();
                using var lg = model.Forward(ids, Enumerable.Range(ctx, rows).ToArray(), -1, st);
                tot += Stopwatch.GetElapsedTime(t0).TotalMilliseconds;
                foreach (var kv in model.TakeStageTimes()) sums[kv.Key] = sums.GetValueOrDefault(kv.Key) + kv.Value;
            }
            VulkanQwen4ExpTransformerModel.StageProfile = false;
            o.AppendLine($"stage rows={rows} ctx={ctx} reps={reps}: forward {tot / reps:F1} ms, stages sum {sums.Values.Sum() / reps:F1} ms");
            foreach (var kv in sums.OrderByDescending(k => k.Value))
                o.AppendLine($"   {kv.Key,-22} {kv.Value / reps,8:F2} ms");
            break;
        }
        case "rdump":
        case "rkl":
        {
            // Real prompts: prefill each in chunks of <rows>, keep the final-position logits (4 vectors).
            string name = a[1]; int rows = int.Parse(a[2]);
            var vecs = new List<float>(); var ms = new List<double>(); var counts = new List<int>();
            foreach (var pr in prompts)
            {
                var ids = tok.Encode(pr); counts.Add(ids.Length);
                using var st = model.CreateState();
                float[] last = Array.Empty<float>();
                for (int p = 0; p < ids.Length; p += rows)
                {
                    int n = Math.Min(rows, ids.Length - p);
                    var t0 = Stopwatch.GetTimestamp();
                    using var lg = model.Forward(ids.AsSpan(p, n), Enumerable.Range(p, n).ToArray(), -1, st);
                    ms.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds);
                    unsafe { last = new ReadOnlySpan<float>((void*)lg.DataPointer, vocab).ToArray(); }
                }
                vecs.AddRange(last);
            }
            var now = vecs.ToArray();
            string path = Path.Combine(work, name + ".f32");
            if (a[0] == "rdump")
            {
                File.WriteAllBytes(path, MemoryMarshal.AsBytes(now.AsSpan()).ToArray());
                o.AppendLine($"rdump {name} rows={rows} prompts tokens [{string.Join(",", counts)}] -> {path}");
            }
            else
            {
                var old = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path).AsSpan()).ToArray();
                for (int v = 0; v < prompts.Length; v++)
                {
                    var A = old.AsSpan(v * vocab, vocab); var B = now.AsSpan(v * vocab, vocab);
                    double ma = double.MinValue, mb = double.MinValue, za = 0, zb = 0, md = 0, kl = 0; int ia = 0, ib = 0;
                    for (int i = 0; i < vocab; i++) { ma = Math.Max(ma, A[i]); mb = Math.Max(mb, B[i]); }
                    for (int i = 0; i < vocab; i++) { za += Math.Exp(A[i] - ma); zb += Math.Exp(B[i] - mb); }
                    double la = Math.Log(za) + ma, lb = Math.Log(zb) + mb;
                    for (int i = 0; i < vocab; i++)
                    {
                        kl += Math.Exp(A[i] - la) * ((A[i] - la) - (B[i] - lb)); md = Math.Max(md, Math.Abs(A[i] - B[i]));
                        if (A[i] > A[ia]) ia = i; if (B[i] > B[ib]) ib = i;
                    }
                    o.AppendLine($"rkl {name} rows={rows} prompt{v} ({counts[v]} tok): KL {kl:E3} max|dlogit| {md:E3} top1 {(ia == ib ? "same" : "DIFF")}");
                }
            }
            break;
        }
        case "dump":
        case "kl":
        {
            string name = a[1]; int rows = int.Parse(a[2]); int tokens = a.Length > 3 ? int.Parse(a[3]) : 48;
            Run(rows, tokens, out _);   // warm
            var now = Run(rows, tokens, out var ms);
            string path = Path.Combine(work, name + ".f32");
            double med = ms.OrderBy(x => x).ElementAt(ms.Count / 2);
            if (a[0] == "dump")
            {
                File.WriteAllBytes(path, MemoryMarshal.AsBytes(now.AsSpan()).ToArray());
                o.AppendLine($"dump {name} rows={rows} tokens={tokens}: {now.Length / vocab} vectors, {new FileInfo(path).Length} bytes, chunk ms med {med:F1}");
            }
            else
            {
                var raw = File.ReadAllBytes(path);
                var old = MemoryMarshal.Cast<byte, float>(raw.AsSpan()).ToArray();
                if (old.Length != now.Length) { o.AppendLine($"kl {name}: SIZE MISMATCH {old.Length} vs {now.Length}"); break; }
                int nv = now.Length / vocab; double sumKl = 0, maxKl = 0, maxD = 0; int agree = 0;
                for (int v = 0; v < nv; v++)
                {
                    var A = old.AsSpan(v * vocab, vocab); var B = now.AsSpan(v * vocab, vocab);
                    double ma = double.MinValue, mb = double.MinValue, za = 0, zb = 0;
                    for (int i = 0; i < vocab; i++) { if (A[i] > ma) ma = A[i]; if (B[i] > mb) mb = B[i]; }
                    for (int i = 0; i < vocab; i++) { za += Math.Exp(A[i] - ma); zb += Math.Exp(B[i] - mb); }
                    double la = Math.Log(za) + ma, lb = Math.Log(zb) + mb, kl = 0; int ia = 0, ib = 0;
                    for (int i = 0; i < vocab; i++)
                    {
                        double pa = Math.Exp(A[i] - la);
                        kl += pa * ((A[i] - la) - (B[i] - lb));
                        maxD = Math.Max(maxD, Math.Abs(A[i] - B[i]));
                        if (A[i] > A[ia]) ia = i;
                        if (B[i] > B[ib]) ib = i;
                    }
                    sumKl += kl; maxKl = Math.Max(maxKl, kl); if (ia == ib) agree++;
                }
                o.AppendLine($"kl {name} rows={rows} tokens={tokens}: vectors {nv}  meanKL {sumKl / nv:E3}  maxKL {maxKl:E3}  max|dlogit| {maxD:E3}  top1 {agree}/{nv}  chunk ms med {med:F1}");
            }
            break;
        }
        default: o.AppendLine("unknown command " + line); break;
    }
}

var lastRefresh = Stopwatch.StartNew();
string jobs = Path.Combine(work, "jobs");
Log("ready");
File.WriteAllText(Path.Combine(work, "ready"), "1");
while (true)
{
    var job = Directory.GetFiles(jobs, "*.job").OrderBy(f => f).FirstOrDefault();
    if (job is null)
    {
        if (lockName.Length > 0 && lastRefresh.Elapsed.TotalMinutes > 10)
        {
            try
            {
                var psi = new ProcessStartInfo("bash", "/c/Development/dotLLM/scripts/gpu-lock.sh refresh " + lockName)
                    { UseShellExecute = false, CreateNoWindow = true };
                psi.Environment["DOTLLM_GPU_LOCK_DIR"] = "/c/Development/dotLLM/.gpu-lock";
                Process.Start(psi)?.WaitForExit();
            }
            catch { }
            lastRefresh.Restart();
        }
        Thread.Sleep(500);
        continue;
    }
    string id = Path.GetFileNameWithoutExtension(job);
    var sb = new StringBuilder();
    bool quit = false;
    foreach (var line in File.ReadAllLines(job))
    {
        if (line.Trim() == "quit") { quit = true; break; }
        try { Exec(line.Trim(), sb); } catch (Exception e) { sb.AppendLine($"ERROR in '{line}': {e}"); }
        File.WriteAllText(Path.Combine(jobs, id + ".out"), sb.ToString());
    }
    File.Delete(job);
    File.WriteAllText(Path.Combine(jobs, id + ".out"), sb.ToString());
    File.WriteAllText(Path.Combine(jobs, id + ".done"), "");
    Log($"job {id} done");
    if (quit) break;
}
