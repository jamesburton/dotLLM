// Diagnostics tool (not shipped) for issue #1220: per-dispatch time and achieved GB/s of the Q4_K multi-column GEMV
// (MatMulQuantGemvMultiKernel, the dense qwen35 verify FFN kernel) at n = 1..8 columns for the Qwen3.8-27B FFN shapes, per NR.
//   Q4KVerifyProbe [reps=5] [nrList=1,2,3,4]
// Weights rotate through several distinct buffers (> the 32 MB infinity cache) so DRAM is measured. Dispatches are back-to-back in one
// command buffer with a compute barrier between them (as the model records them). Arms are interleaved per rep (order alternates).
using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

int reps = args.Length > 0 ? int.Parse(args[0]) : 5;
int[] nrs = (args.Length > 1 ? args[1] : "1,2,3,4").Split(',').Select(int.Parse).ToArray();
const int Copies = 4, PerSubmit = 24;
(string tag, int m, int k)[] shapes = [("gate/up", 17408, 5120), ("down", 5120, 17408)];

using var dev = VulkanDevice.Create();
string spv = VulkanModelLoader.ResolveSpvDir();
Console.WriteLine($"device {dev.DeviceName}");
var kernels = new Dictionary<int, MatMulQuantGemvMultiKernel>();
foreach (int nr in nrs)
    kernels[nr] = MatMulQuantGemvMultiKernel.TryCreate(dev, spv, QuantGemvMultiFormat.Q4_K, 2, nr, nr)!;

foreach (var (tag, m, k) in shapes)
{
    long rowBytes = (long)(k / 256) * 144, wBytes = m * rowBytes;
    var ws = new VulkanDevice.Buffer[Copies];
    var rnd = new Random(7);
    var host = new byte[wBytes];
    for (int c = 0; c < Copies; c++)
    {
        rnd.NextBytes(host);
        for (long b = 0; b < (long)m * (k / 256); b++) { host[b * 144 + 1] = 0x2c; host[b * 144 + 3] = 0x2c; }
        ws[c] = dev.Allocate(wBytes);
        dev.Upload(host, ws[c]);
    }
    var x = dev.Allocate(8L * k * 4); var y = dev.Allocate(8L * m * 4);
    var xs = new float[8 * k]; for (int i = 0; i < xs.Length; i++) xs[i] = (float)(rnd.NextDouble() * 0.2 - 0.1);
    dev.Upload(xs, x);
    Console.WriteLine($"\n== {tag} m={m} k={k} weights {wBytes / 1e6:F1} MB ==  (median us/dispatch [GB/s of weights] min-max)");
    Console.Write("n   "); foreach (int nr in nrs) Console.Write($"{"NR=" + nr,26}"); Console.WriteLine();
    var res = new Dictionary<(int n, int nr), List<double>>();
    for (int rep = 0; rep < reps + 1; rep++)
        for (int ni = 0; ni < 8; ni++)
        {
            int n = rep % 2 == 0 ? ni + 1 : 8 - ni;
            var order = rep % 2 == 0 ? nrs : nrs.Reverse().ToArray();
            foreach (int nr in order)
            {
                using var ctx = dev.CreateSubmitContext();
                ctx.Begin();
                for (int i = 0; i < PerSubmit; i++)
                {
                    kernels[nr].Record(ctx.CommandBuffer, ws[i % Copies], x, y, m, k, n);
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
    for (int n = 1; n <= 8; n++)
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
