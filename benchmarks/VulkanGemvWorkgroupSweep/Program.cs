using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

// Q8_0 GEMV workgroup-size sweep on Strix Halo (RDNA3.5, subgroupSize=64).
// Compares 4 SPV variants of matmul_q8_0:
//
//   wg64    : local_size_x =  64 (1 wavefront), shared-mem reduce
//   wg128   : local_size_x = 128 (2 wavefronts), shared-mem reduce  ← current production
//   wg256   : local_size_x = 256 (4 wavefronts), shared-mem reduce
//   sg      : local_size_x = 128, subgroupAdd reduce (2 barriers vs 8)
//
// Implementation notes:
// - MatMulQ8_0Kernel.Create() loads "matmul_q8_0.spv" by filename. Rather than
//   fork the kernel we copy each variant on top of that filename per-run,
//   recreate the pipeline, and time it. Restored to the WG=128 reference at exit.
// - Submission cost (vkQueueSubmit + vkQueueWaitIdle) dominates for short kernels,
//   so we batch BatchSize Record() calls per submit and divide by BatchSize.
// - Strix Halo iGPU clocks vary run-to-run; we report the median of NPasses
//   passes per variant. Single-run numbers are not load-bearing.

const int batchSize = 32;
const int warmupPasses = 3;
const int passes = 7;
const int batchesPerPass = 8;  // 32 * 8 = 256 timed launches per pass

// Llama-3.2-1B-class GEMV shapes that hit the decode hot path. K kept multiple
// of 32 (Q8_0 block size). Per-shape decode-step dispatch counts in parens.
(string Tag, int M, int K, int CountPerStep)[] shapes =
[
    ("attn_q",  2048, 2048, 1),
    ("attn_kv",  512, 2048, 2),
    ("attn_o",  2048, 2048, 1),
    ("ffn_gu",  8192, 2048, 2),
    ("ffn_dn",  2048, 8192, 1),
    ("lm_head", 128256, 2048, 1)
];

string repoRoot = FindRepoRoot();
string spvDir = Path.Combine(repoRoot, "native", "vulkan", "spv");

string canonicalSpv = Path.Combine(spvDir, "matmul_q8_0.spv");
string backupSpv = canonicalSpv + ".bench-backup";

string[] variants = ["wg64", "wg128", "wg256", "sg"];

if (!File.Exists(backupSpv))
    File.Copy(canonicalSpv, backupSpv);

try
{
    using var device = VulkanDevice.Create();
    Console.WriteLine($"Device: {device.DeviceName}");
    Console.WriteLine($"Subgroup size: {device.SubgroupSize}, arithmetic={device.HasSubgroupArithmetic}");
    Console.WriteLine($"Variants: {string.Join(", ", variants)}");
    Console.WriteLine($"Schedule: {warmupPasses} warmup + {passes} timed passes,"
        + $" {batchesPerPass} batches × {batchSize} launches per pass per variant");
    Console.WriteLine();

    Console.Write($"{"shape",-10} {"M",6} {"K",6}  ");
    foreach (var v in variants)
        Console.Write($"{v + " us(med)",14}");
    Console.Write("  best vs wg128");
    Console.WriteLine();
    Console.WriteLine(new string('-', 36 + 14 * variants.Length + 18));

    var totalsMedian = new Dictionary<string, double>();
    foreach (var v in variants) totalsMedian[v] = 0;

    foreach (var (tag, m, k, _) in shapes)
    {
        long blocksPerRow = k / 32;
        long rowBytes = blocksPerRow * 34;
        long weightBytes = (long)m * rowBytes;

        using var hostWeights = device.Allocate(weightBytes);
        using var hostX = device.Allocate((long)k * sizeof(float));
        using var hostY = device.Allocate((long)m * sizeof(float));

        var rnd = new Random(42);
        var weightBuf = new byte[weightBytes];
        for (int i = 0; i < weightBytes; i++) weightBuf[i] = (byte)rnd.Next(256);
        var xBuf = new float[k];
        for (int i = 0; i < k; i++) xBuf[i] = (float)(rnd.NextDouble() * 0.1 - 0.05);
        device.Upload(weightBuf, hostWeights);
        device.Upload(xBuf, hostX);

        Console.Write($"{tag,-10} {m,6} {k,6}  ");

        var resultsMedian = new Dictionary<string, double>();
        foreach (var v in variants)
        {
            string variantPath = Path.Combine(spvDir, $"matmul_q8_0_{v}.spv");
            if (v == "wg128") variantPath = backupSpv;
            File.Copy(variantPath, canonicalSpv, overwrite: true);

            using var kernel = MatMulQ8_0Kernel.Create(device, spvDir);

            // Warmup
            for (int p = 0; p < warmupPasses; p++)
                MeasurePass(device, kernel, hostWeights, hostX, hostY, m, k, batchSize, batchesPerPass);

            var passUs = new double[passes];
            for (int p = 0; p < passes; p++)
                passUs[p] = MeasurePass(device, kernel, hostWeights, hostX, hostY, m, k, batchSize, batchesPerPass);

            Array.Sort(passUs);
            double median = passUs[passes / 2];
            resultsMedian[v] = median;
            Console.Write($"{median,14:F1}");
        }

        double baseline = resultsMedian["wg128"];
        var best = resultsMedian.OrderBy(kv => kv.Value).First();
        double speedup = baseline / best.Value;
        Console.Write($"  {best.Key} ({speedup:F2}x)");
        Console.WriteLine();

        int countPerStep = shapes.First(s => s.Tag == tag).CountPerStep;
        foreach (var (v, us) in resultsMedian)
            totalsMedian[v] += us * countPerStep;
    }

    Console.WriteLine();
    Console.WriteLine("Per-decode-step Q8_0 GEMV cost (sum of weighted shapes, median pass):");
    Console.Write($"{"variant",-10} ");
    foreach (var v in variants) Console.Write($"{v,14}");
    Console.WriteLine();
    Console.Write($"{"us/step",-10} ");
    foreach (var v in variants) Console.Write($"{totalsMedian[v],14:F1}");
    Console.WriteLine();
    Console.Write($"{"vs wg128",-10} ");
    foreach (var v in variants) Console.Write($"{totalsMedian["wg128"] / totalsMedian[v],14:F2}x");
    Console.WriteLine();
}
finally
{
    if (File.Exists(backupSpv))
    {
        File.Copy(backupSpv, canonicalSpv, overwrite: true);
        File.Delete(backupSpv);
    }
}

// One pass = `batches` × `batchSize` recorded launches per submit, then SubmitAndWait.
// Returns mean us per launch over the pass.
//
// Back-to-back launches with the same buffer write set are independent from
// the GPU's perspective without an explicit barrier — measures the kernel's
// raw throughput without serialisation overhead. Production decode inserts
// a ComputeToComputeBarrier between dispatches; we measure the underlying
// kernel cost so the relative ranking carries across that overhead.
static double MeasurePass(
    VulkanDevice device, MatMulQ8_0Kernel kernel,
    VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
    int m, int k, int batchSize, int batches)
{
    using var ctx = device.CreateSubmitContext();
    var sw = Stopwatch.StartNew();
    for (int b = 0; b < batches; b++)
    {
        ctx.Begin();
        for (int i = 0; i < batchSize; i++)
            kernel.Record(ctx.CommandBuffer, weights, x, y, m, k);
        ctx.SubmitAndWait();
    }
    sw.Stop();
    long launches = (long)batches * batchSize;
    return sw.Elapsed.TotalMicroseconds / launches;
}

static string FindRepoRoot()
{
    string? dir = AppContext.BaseDirectory;
    while (dir is not null && !File.Exists(Path.Combine(dir, "CLAUDE.md")))
        dir = Path.GetDirectoryName(dir);
    return dir ?? throw new InvalidOperationException("Could not locate repo root (CLAUDE.md)");
}
