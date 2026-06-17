using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using DotLLM.Vulkan.Interop;

// ── Per-SHAPE Q8_0 decode GEMV variant sweep ─────────────────────────────────
//
// Campaign V1 found the optimal Q8_0 decode-GEMV workgroup variant is vendor-
// dependent; the follow-up question (this bench) is whether it is *also*
// per-SHAPE — and whether a per-shape dispatch beats the per-vendor single
// default once you weight by how often each shape actually fires in a decode
// step.
//
// Variants timed (all held LIVE simultaneously — no .spv file swapping):
//   wg64    : local_size_x =  64, shared-mem reduce
//   wg128   : local_size_x = 128, shared-mem reduce   (canonical matmul_q8_0.spv)
//   wg256   : local_size_x = 256, shared-mem reduce
//   sg      : local_size_x = 128, subgroupAdd reduce
//   dp4a-pq : INT8 dotPacked4x8 with a shared activation pre-quant pass
//             (only on devices with VK_KHR_shader_integer_dot_product)
//
// Methodology (campaign discipline):
//   - All variants instantiated up front via DOTLLM_VULKAN_Q8_GEMV_VARIANT +
//     Create(), giving one live pipeline per variant. The on-disk .spv is never
//     mutated, so variants can be interleaved within a round.
//   - Batched-fence timing: BatchSize Record()s per submit, divided by BatchSize,
//     to amortise submit/wait overhead. min-of-Batches per (round, shape, variant)
//     discards throttle spikes within the round.
//   - INTERLEAVED: within each round every variant is timed back-to-back on the
//     same shape, so clock drift hits all variants equally.
//   - PAIRED per-round ratios: for each round we form variant_us / baseline_us
//     within that round, then report the MEDIAN of those per-round ratios across
//     rounds. We never divide a min taken in one place by a min taken elsewhere.
//   - Decode weighting: each per-layer projection fires CountPerStep × NumLayers
//     times per decode step; lm_head fires once. The weighted-sum table tells us
//     the real decode-path win, not a per-shape headline.
//
// Device select: default = NVIDIA RTX 3060; set DOTLLM_VULKAN_DEVICE_VENDOR=0x8086
// to target the Intel Arc iGPU.

// Record()s per submit are bounded by total dispatched rows so a single
// barrier-free submit cannot trip NVIDIA's ~2s TDR watchdog on large-M shapes
// (ffn 8192, lm_head 49152/128256). batchSize depends ONLY on M, so it is
// identical across variants for a given shape → paired per-round ratios stay
// clean. Large-M shapes clamp toward 1, where submit/wait overhead grows; we
// raise batches-per-round there to keep min-of-N meaningful.
// Budget chosen so even large-M shapes get several Record()s per submit — small
// batches leave submit/quant overhead dominating and compress all variants to
// ~1.00x noise. TDR is not the constraint (the fault we chased was stale
// descriptors, since fixed); the parity test runs a single M=128256 dispatch
// fine, and batching a handful is well within budget.
const int rowsPerSubmitBudget = 1 << 20; // 1,048,576 rows/submit
const int maxBatchSize = 256;            // cap so tiny shapes don't over-batch
const int batchesPerRound = 6;  // min-of-6 batches within a round (per shape/variant)
const int warmupRounds = 4;
const int maxRounds = 21;       // odd → clean median; small shapes get the full count

// This box's NVIDIA driver imposes a ~3 ms FLOOR per compute dispatch at
// M >= ~2048 (a 2048×2048 Q8 GEMV reads ~4 MB ≈ 11 µs at bandwidth, yet
// measures ~3 ms — pure per-dispatch launch latency). Those large-M shapes are
// therefore latency-bound, identical across every variant, and a full
// maxRounds×batchesPerRound schedule would take many minutes for zero extra
// signal. We bound total timed launches per (shape,variant) by a work budget so
// large shapes get just enough samples to confirm convergence while small/
// medium shapes (where real variant differences live) keep the full schedule.
const long launchWorkBudget = 30_000_000L; // ~M-rows summed over all timed launches

static int BatchSizeForM(int m) => Math.Clamp(rowsPerSubmitBudget / m, 1, maxBatchSize);

// Timed rounds for a shape: full maxRounds for cheap shapes; fewer (>=3, odd) for
// expensive large-M shapes so the run stays tractable. batchesPerRound and
// batchSize are unchanged so per-round ratios remain comparable.
static int RoundsForM(int m, int batchSize)
{
    long perRoundWork = (long)batchesPerRound * batchSize * m;
    int r = (int)Math.Clamp(launchWorkBudget / Math.Max(1, perRoundWork), 3, maxRounds);
    if ((r & 1) == 0) r++; // keep odd for a clean median
    return r;
}

string repoRoot = FindRepoRoot();
string spvDir = Path.Combine(repoRoot, "native", "vulkan", "spv");

// ── Model decode GEMV shapes (M = output dim, K = input/contraction dim) ──────
// CountPerStep = dispatches of this projection per decode step within ONE layer.
// PerLayer shapes are weighted ×NumLayers; lm_head is ×1.
//
// SmolLM2-135M: hidden=576, inter=1536, heads=9·64, kv=3·64=192, layers=30,
//   vocab=49152 (tied → lm_head = 49152×576).
// Llama-3.2-1B: hidden=2048, inter=8192, heads=32·64, kv=8·64=512, layers=16,
//   vocab=128256 (tied → lm_head = 128256×2048).
var models = new (string Name, int Layers, (string Tag, int M, int K, int CountPerStep, bool PerLayer)[] Shapes)[]
{
    ("SmolLM2-135M", 30, new (string, int, int, int, bool)[]
    {
        ("attn_q",     576,   576, 1, true),
        ("attn_kv",    192,   576, 2, true),
        ("attn_o",     576,   576, 1, true),
        ("ffn_gateup", 1536,  576, 2, true),
        ("ffn_down",   576,  1536, 1, true),
        ("lm_head",   49152,  576, 1, false),
    }),
    ("Llama-3.2-1B", 16, new (string, int, int, int, bool)[]
    {
        ("attn_q",     2048, 2048, 1, true),
        ("attn_kv",     512, 2048, 2, true),
        ("attn_o",     2048, 2048, 1, true),
        ("ffn_gateup", 8192, 2048, 2, true),
        ("ffn_down",   2048, 8192, 1, true),
        ("lm_head",  128256, 2048, 1, false),
    }),
};

string[] wgVariants = ["wg64", "wg128", "wg256", "sg"];

using var device = VulkanDevice.Create();
Console.WriteLine($"Device      : {device.DeviceName} (vendor 0x{device.VendorId:X4})");
Console.WriteLine($"Subgroup    : size={device.SubgroupSize}, arithmetic={device.HasSubgroupArithmetic}");
Console.WriteLine($"IntDotProd  : {device.HasIntegerDotProduct}");

// Current per-vendor default (the baseline we must beat). Mirrors
// MatMulQ8_0Kernel.Create + the forward's DP4a default-on for Intel.
const uint VendorIntel = 0x8086, VendorNvidia = 0x10DE;
bool dp4aSupported = device.HasIntegerDotProduct;
string vendorDefault = device.VendorId switch
{
    VendorIntel when dp4aSupported => "dp4a-pq",      // Intel: DP4a default-on
    VendorIntel when device.HasSubgroupArithmetic => "sg",
    VendorNvidia => "wg64",
    _ => "wg128",
};
Console.WriteLine($"Per-vendor default (baseline): {vendorDefault}");
Console.WriteLine($"Schedule    : {warmupRounds} warmup + up to {maxRounds} timed rounds "
    + "(adaptive: fewer for latency-bound large-M shapes), "
    + $"{batchesPerRound} batches × (M-bounded batchSize, budget {rowsPerSubmitBudget} rows/submit) "
    + "per (round,shape,variant), min-of-batch, median-of-round-ratios");
Console.WriteLine();

// Instantiate every workgroup variant LIVE (one pipeline each).
var kernels = new Dictionary<string, MatMulQ8_0Kernel>();
foreach (var v in wgVariants)
{
    Environment.SetEnvironmentVariable("DOTLLM_VULKAN_Q8_GEMV_VARIANT", v);
    kernels[v] = MatMulQ8_0Kernel.Create(device, spvDir);
}
Environment.SetEnvironmentVariable("DOTLLM_VULKAN_Q8_GEMV_VARIANT", null);

// Test A toggle: DOTLLM_BENCH_NO_DP4A=1 drops the dp4a-pq path to isolate
// whether an async fault originates there.
bool benchNoDp4a = Environment.GetEnvironmentVariable("DOTLLM_BENCH_NO_DP4A") == "1";
MatMulQ8_0Dp4aPqKernel? dp4a = null;
if (dp4aSupported && !benchNoDp4a)
    dp4a = MatMulQ8_0Dp4aPqKernel.Create(device, spvDir, maxDescriptorSets: 4096);

// All variant tags we will report (dp4a-pq only where supported).
var variants = new List<string>(wgVariants);
if (dp4a is not null) variants.Add("dp4a-pq");

// Pre-allocate EVERY shape's device buffers up front and keep them alive for the
// whole run. The live kernels' DescriptorSetCache is keyed by buffer HANDLE; if
// we freed buffers per shape (using-scoped), Vulkan would recycle those handles
// and the cache would hand back descriptor sets bound to destroyed buffers →
// the GPU touches freed memory → asynchronous VK_ERROR_DEVICE_LOST several
// shapes later. Holding all buffers live for the run keeps every cached
// descriptor valid. Total footprint ≈ 350 MB (Llama lm_head weights dominate),
// trivial on the 3060/Arc.
var shapeBufs = new Dictionary<(string, string),
    (VulkanDevice.Buffer W, VulkanDevice.Buffer X, VulkanDevice.Buffer Y,
     VulkanDevice.Buffer? Xq, VulkanDevice.Buffer? Dx)>();
var allBuffers = new List<IDisposable>();

try
{
    foreach (var (modelName, _, shapes) in models)
    foreach (var (tag, m, k, _, _) in shapes)
    {
        long blocksPerRow = k / 32;
        long rowBytes = blocksPerRow * 34;
        long weightBytes = (long)m * rowBytes;
        long weightBufBytes = (weightBytes + 3) & ~3L;

        // Weights are DEVICE-LOCAL, exactly as production loads them. On a
        // DISCRETE GPU (RTX 3060) a host-visible weight buffer is read over PCIe
        // at host bandwidth — a 2 MB+ GEMV then measures ~3 ms/dispatch (pure
        // PCIe latency), dwarfing the kernel and burying every variant in noise.
        // Device-local puts weights in VRAM in the driver's tiled layout, which
        // is what the decode path actually runs against. Activations / output /
        // scratch stay host-visible (that matches production scratch buffers).
        var bufW = device.AllocateDeviceLocal(weightBufBytes);
        var bufX = device.Allocate((long)k * sizeof(float));
        var bufY = device.Allocate((long)m * sizeof(float));
        var bufXq = dp4a is not null ? device.Allocate(MatMulQ8_0Dp4aPqKernel.XqScratchBytes(k)) : null;
        var bufDx = dp4a is not null ? device.Allocate(MatMulQ8_0Dp4aPqKernel.DxScratchBytes(k)) : null;
        allBuffers.Add(bufW); allBuffers.Add(bufX); allBuffers.Add(bufY);
        if (bufXq is not null) allBuffers.Add(bufXq);
        if (bufDx is not null) allBuffers.Add(bufDx);

        var rnd = new Random(42 + m * 7 + k);
        var wb = new byte[weightBytes];
        rnd.NextBytes(wb);
        var xb = new float[k];
        for (int i = 0; i < k; i++) xb[i] = (float)(rnd.NextDouble() * 0.1 - 0.05);
        using (var staging = device.Allocate(weightBufBytes))
            device.UploadToDeviceLocal(new ReadOnlySpan<byte>(wb), staging, bufW);
        device.Upload(xb, bufX);

        shapeBufs[(modelName, tag)] = (bufW, bufX, bufY, bufXq, bufDx);
    }

    foreach (var (modelName, layers, shapes) in models)
    {
        Console.WriteLine($"════ {modelName} ({layers} layers) ════");
        Console.Write($"{"shape",-11}{"M",7}{"K",7}  ");
        foreach (var v in variants) Console.Write($"{v + " us",13}");
        Console.Write($"   best  vs[{vendorDefault}]");
        Console.WriteLine();
        Console.WriteLine(new string('-', 27 + 13 * variants.Count + 24));

        // Weighted decode-step accumulators (median us per shape × weight).
        var weightedSum = new Dictionary<string, double>();
        foreach (var v in variants) weightedSum[v] = 0;
        double oracleWeighted = 0; // sum of per-shape best medUs × weight (per-shape upper bound)

        foreach (var (tag, m, k, countPerStep, perLayer) in shapes)
        {
            var (bufW, bufX, bufY, bufXq, bufDx) = shapeBufs[(modelName, tag)];

            // Per-round ratios vs baseline, plus raw us for the headline median.
            var roundRatios = new Dictionary<string, List<double>>();
            var roundUs = new Dictionary<string, List<double>>();
            foreach (var v in variants) { roundRatios[v] = new(); roundUs[v] = new(); }

            int batchSize = BatchSizeForM(m);
            int rounds = RoundsForM(m, batchSize);
            for (int r = 0; r < warmupRounds + rounds; r++)
            {
                // One round: time every variant interleaved on this shape.
                var us = new Dictionary<string, double>();
                foreach (var v in variants)
                {
                    // One SubmitContext reused across all batches for this variant —
                    // avoids per-batch command-buffer + fence churn (tens of
                    // thousands of allocations over a run), which NVIDIA's driver
                    // can choke on and surface as an async DEVICE_LOST.
                    using var ctx = device.CreateSubmitContext();
                    double best = double.MaxValue;
                    for (int b = 0; b < batchesPerRound; b++)
                    {
                        double t = v == "dp4a-pq"
                            ? MeasureDp4aBatch(ctx, dp4a!, bufW, bufX, bufXq!, bufDx!, bufY, m, k, batchSize)
                            : MeasureWgBatch(ctx, kernels[v], bufW, bufX, bufY, m, k, batchSize);
                        if (t < best) best = t;
                    }
                    us[v] = best;
                }
                if (r < warmupRounds) continue;

                double baseUs = us[vendorDefault];
                foreach (var v in variants)
                {
                    roundUs[v].Add(us[v]);
                    roundRatios[v].Add(baseUs / us[v]); // >1 means v is faster than baseline
                }
            }

            // Median us (for display) and median of per-round ratios (the verdict metric).
            var medUs = new Dictionary<string, double>();
            var medRatio = new Dictionary<string, double>();
            foreach (var v in variants)
            {
                medUs[v] = Median(roundUs[v]);
                medRatio[v] = Median(roundRatios[v]);
            }

            Console.Write($"{tag,-11}{m,7}{k,7}  ");
            foreach (var v in variants) Console.Write($"{medUs[v],13:F1}");
            var bestVar = medUs.OrderBy(kv => kv.Value).First();
            Console.Write($"   {bestVar.Key,-8} {medRatio[bestVar.Key]:F2}x  (bs={batchSize},r={rounds})");
            Console.WriteLine();

            long weight = perLayer ? (long)countPerStep * layers : countPerStep;
            foreach (var v in variants) weightedSum[v] += medUs[v] * weight;
            oracleWeighted += bestVar.Value * weight; // per-shape best for this shape
        }

        // Decode-weighted totals: baseline (per-vendor default) vs each variant,
        // and vs an oracle per-shape pick (lower bound on per-shape benefit).
        Console.WriteLine();
        // "per-shape" column = the oracle dispatch that always picks the best
        // variant for each shape — the upper bound on what per-shape can deliver.
        Console.WriteLine($"Decode-weighted Q8_0 GEMV cost per step (us, ×layers for per-layer shapes):");
        Console.Write($"{"variant",-12}");
        foreach (var v in variants) Console.Write($"{v,13}");
        Console.WriteLine($"{"per-shape",13}");

        Console.Write($"{"us/step",-12}");
        foreach (var v in variants) Console.Write($"{weightedSum[v],13:F1}");
        Console.WriteLine($"{oracleWeighted,13:F1}");
        Console.Write($"{"vs default",-12}");
        foreach (var v in variants) Console.Write($"{weightedSum[vendorDefault] / weightedSum[v],12:F3}x");
        Console.WriteLine($"{weightedSum[vendorDefault] / oracleWeighted,12:F3}x");
        Console.WriteLine();
        Console.WriteLine();
    }
}
finally
{
    foreach (var kv in kernels) kv.Value.Dispose();
    dp4a?.Dispose();
    foreach (var b in allBuffers) b.Dispose();
}

// One batch = `batchSize` barrier-free Record()s in a single submit (reused
// SubmitContext), returns mean us/launch. Batching amortises submit+wait
// overhead — without it (one submit per launch) overhead swamps the kernel and
// every variant reads ~1.00x. Back-to-back same-output writes are independent
// from the GPU's view without a barrier (we measure raw kernel throughput; the
// relative ranking carries across the production barrier). batchSize is
// M-bounded so a single submit's total work stays modest.
static double MeasureWgBatch(
    VulkanDevice.SubmitContext ctx, MatMulQ8_0Kernel kernel,
    VulkanDevice.Buffer weights, VulkanDevice.Buffer x, VulkanDevice.Buffer y,
    int m, int k, int batchSize)
{
    ctx.Begin();
    for (int i = 0; i < batchSize; i++)
        kernel.Record(ctx.CommandBuffer, weights, x, y, m, k);
    var sw = Stopwatch.StartNew();
    ctx.SubmitAndWait();
    sw.Stop();
    return sw.Elapsed.TotalMicroseconds / batchSize;
}

// DP4a-pq batch: quantize the activation ONCE into shared xq/dx scratch, then
// batch read-only RecordGemvPrequant calls — read-only shared scratch ⇒ no WAR
// hazard between iterations ⇒ no per-iteration barrier needed, and it matches
// how production amortises the activation quant across same-input projections.
static double MeasureDp4aBatch(
    VulkanDevice.SubmitContext ctx, MatMulQ8_0Dp4aPqKernel kernel,
    VulkanDevice.Buffer weights, VulkanDevice.Buffer x,
    VulkanDevice.Buffer xq, VulkanDevice.Buffer dx, VulkanDevice.Buffer y,
    int m, int k, int batchSize)
{
    ctx.Begin();
    kernel.RecordQuantizeActivation(ctx.CommandBuffer, x, xq, dx, k);
    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer); // xq/dx visible to GEMVs
    for (int i = 0; i < batchSize; i++)
        kernel.RecordGemvPrequant(ctx.CommandBuffer, weights, xq, dx, y, m, k);
    var sw = Stopwatch.StartNew();
    ctx.SubmitAndWait();
    sw.Stop();
    return sw.Elapsed.TotalMicroseconds / batchSize;
}

static double Median(List<double> xs)
{
    if (xs.Count == 0) return double.NaN;
    var a = xs.ToArray();
    Array.Sort(a);
    int n = a.Length;
    return (n & 1) == 1 ? a[n / 2] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
}

static string FindRepoRoot()
{
    string? dir = AppContext.BaseDirectory;
    while (dir is not null && !File.Exists(Path.Combine(dir, "CLAUDE.md")))
        dir = Path.GetDirectoryName(dir);
    return dir ?? throw new InvalidOperationException("Could not locate repo root (CLAUDE.md)");
}
