// Diagnostics tool (not shipped) for issue #1220: per-dispatch time and achieved GB/s of the Q4_K multi-column GEMV
// (MatMulQuantGemvMultiKernel, the dense qwen35 verify FFN kernel) at n = 1..8 columns for the Qwen3.8-27B FFN shapes, per NR.
//   Q4KVerifyProbe [reps=5] [nrList=1,2,3,4] [maxColumns=8..16] [format=q4k|q5k|q6k|q8_0]
// (#1515: q5k has no NR, so its nrList is ignored and one arm is run; q6k = the Q6_K rows kernel; q8_0 = MatMulExactGemvMultiKernel.)
// Weights rotate through several distinct buffers (> the 32 MB infinity cache) so DRAM is measured. Dispatches are back-to-back in one
// command buffer with a compute barrier between them (as the model records them). Arms are interleaved per rep (order alternates).
using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

int reps = args.Length > 0 ? int.Parse(args[0]) : 5;
int[] nrs = (args.Length > 1 ? args[1] : "1,2,3,4").Split(',').Select(int.Parse).ToArray();
const int Copies = 4, PerSubmit = 24;
int NMax = args.Length > 2 ? int.Parse(args[2]) : 8;
string fmt = args.Length > 3 ? args[3] : "q4k";
// (block bytes, elements per block, fp16 scale byte offsets that must stay finite)
var (blockBytes, groupSize, scaleOffsets) = fmt switch
{
    "q4k" => (144, 256, new[] { 0, 2 }),
    "q5k" => (176, 256, new[] { 0, 2 }),
    "q6k" => (210, 256, new[] { 208 }),
    "q8_0" => (34, 32, new[] { 0 }),
    _ => throw new ArgumentException($"format {fmt}"),
};

(string tag, int m, int k)[] shapes = [("gate/up", 17408, 5120), ("down", 5120, 17408)];

using var dev = VulkanDevice.Create();
string spv = VulkanModelLoader.ResolveSpvDir();
Console.WriteLine($"device {dev.DeviceName} format {fmt}");
var record = new Dictionary<int, Action<nint, VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer, int, int, int>>();
foreach (int nr in nrs)
{
    if (fmt == "q8_0")
    {
        var q8 = MatMulExactGemvMultiKernel.TryCreate(dev, spv, ExactGemvFormat.Q8_0, 2, nr)!;
        record[nr] = (cb, w, xb, yb, m, k, n) => q8.Record(cb, w, xb, yb, m, k, n);
    }
    else
    {
        var fmtId = fmt switch { "q4k" => QuantGemvMultiFormat.Q4_K, "q5k" => QuantGemvMultiFormat.Q5_K, _ => QuantGemvMultiFormat.Q6_K_Rows };
        var kq = fmt == "q5k" ? MatMulQuantGemvMultiKernel.TryCreate(dev, spv, fmtId, 1, nr, nr)! : MatMulQuantGemvMultiKernel.TryCreate(dev, spv, fmtId, 2, nr, Math.Min(nr, 4))!;
        record[nr] = (cb, w, xb, yb, m, k, n) => kq.Record(cb, w, xb, yb, m, k, Math.Max(n, 2));   // q5k has no 1-column pipeline: n=1 row repeats n=2
    }
}

if (args.Length > 4 && args[4] == "hash")
{
    // Bit-identity probe: output hash per column count and NR on fixed random data. Run it against the old and the new .spv (swap the file under
    // bin/.../spv) and diff the text. Odd M and odd block counts exercise the row tail and the pipeline tail.
    foreach (var (hm, hk) in new[] { (1003, 5120), (37, 768), (21, 1280) })
    {
        long hRow = (long)(hk / groupSize) * blockBytes;
        var hr = new Random(11);
        var hw = new byte[hm * hRow]; hr.NextBytes(hw);
        for (long b = 0; b < (long)hm * (hk / groupSize); b++) foreach (int so in scaleOffsets) hw[b * blockBytes + so + 1] = 0x2c;
        var hxs = new float[16 * hk]; for (int i = 0; i < hxs.Length; i++) hxs[i] = (float)(hr.NextDouble() * 2 - 1);
        using var hbw = dev.Allocate(hw.Length); using var hbx = dev.Allocate(hxs.Length * 4L); using var hby = dev.Allocate(16L * hm * 4);
        dev.Upload(hw, hbw); dev.Upload(hxs, hbx);
        for (int n = 1; n <= NMax; n++)
        {
            if (fmt == "q5k" && n == 1) continue;
            Console.Write($"hash m={hm} k={hk} n={n,-2}");
            foreach (int nr in nrs)
            {
                using (var ctx = dev.CreateSubmitContext()) { ctx.Begin(); record[nr](ctx.CommandBuffer, hbw, hbx, hby, hm, hk, n); ctx.SubmitAndWait(); }
                var outp = new float[n * hm]; dev.Download(hby, outp);
                ulong h = 1469598103934665603UL;
                foreach (float f in outp) { h ^= BitConverter.SingleToUInt32Bits(f); h *= 1099511628211UL; }
                Console.Write($" nr{nr}={h:x16}");
            }
            Console.WriteLine();
        }
    }
    return;
}

foreach (var (tag, m, k) in shapes)
{
    long rowBytes = (long)(k / groupSize) * blockBytes, wBytes = m * rowBytes;
    var ws = new VulkanDevice.Buffer[Copies];
    var rnd = new Random(7);
    var host = new byte[wBytes];
    for (int c = 0; c < Copies; c++)
    {
        rnd.NextBytes(host);
        for (long b = 0; b < (long)m * (k / groupSize); b++) foreach (int so in scaleOffsets) host[b * blockBytes + so + 1] = 0x2c;
        ws[c] = dev.Allocate(wBytes);
        dev.Upload(host, ws[c]);
    }
    var x = dev.Allocate(16L * k * 4); var y = dev.Allocate(16L * m * 4);
    var xs = new float[16 * k]; for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rnd.NextDouble() * 0.2 - 0.1);
    dev.Upload(xs, x);
    Console.WriteLine($"\n== {tag} m={m} k={k} weights {wBytes / 1e6:F1} MB ==  (median us/dispatch [GB/s of weights] min-max)");
    Console.Write("n   "); foreach (int nr in nrs) Console.Write($"{"NR=" + nr,26}"); Console.WriteLine();
    var res = new Dictionary<(int n, int nr), List<double>>();
    for (int rep = 0; rep < reps + 1; rep++)
        for (int ni = 0; ni < NMax; ni++)
        {
            int n = rep % 2 == 0 ? ni + 1 : NMax - ni;
            var order = rep % 2 == 0 ? nrs : nrs.Reverse().ToArray();
            foreach (int nr in order)
            {
                using var ctx = dev.CreateSubmitContext();
                ctx.Begin();
                for (int i = 0; i < PerSubmit; i++)
                {
                    record[nr](ctx.CommandBuffer, ws[i % Copies], x, y, m, k, n);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                }
                var sw = Stopwatch.StartNew();
                ctx.SubmitAndWait();
                double us = sw.Elapsed.TotalMicroseconds / PerSubmit;
                if (rep == 0) continue;
                if (!res.TryGetValue((n, nr), out var l)) res[(n, nr)] = l = [];
                l.Add(us);
            }
        }
    for (int n = 1; n <= NMax; n++)
    {
        Console.Write($"{n,-4}");
        foreach (int nr in nrs)
        {
            var l = res[(n, nr)]; l.Sort();
            double med = l[l.Count / 2];
            Console.Write($"{med,7:F0} [{wBytes / med / 1e3,4:F0}] {l[0]:F0}-{l[^1]:F0}".PadRight(26));
        }
        Console.WriteLine();
    }
    foreach (var w in ws) w.Dispose();
    x.Dispose(); y.Dispose();
}
