using BenchmarkDotNet.Attributes;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Benchmarks;

/// <summary>
/// Attention-kernel microbench: compares <see cref="VulkanFlashAttentionF32Kernel"/>
/// against <see cref="AttentionF32Kernel"/> at prefill-shaped (seqQ &gt; 1)
/// dispatches. Reports wall-clock per Launch call; divide seqQ * numHeads by the
/// reported median to get the relative dispatch cost per (token, head).
/// </summary>
/// <remarks>
/// <para>
/// Shapes chosen to bracket the headline llama.cpp Vulkan FA benchmark:
///   - <c>pp512</c>: 512 query tokens × 512 KV tokens.
///   - <c>pp2048</c>: 2048 × 2048 (long-context prefill).
///   - <c>pp4096</c>: 4096 × 4096 (stretches the BR=16 amortisation factor).
/// Head config tracks Llama-3.2-1B (32 heads / 8 KV heads, head_dim 64). The
/// Q8_0 weight path doesn't matter here — this is a synthetic attention-only
/// kernel comparison.
/// </para>
/// <para>
/// To reproduce on Strix Halo: <c>dotnet run -c Release --project
/// benchmarks/DotLLM.Benchmarks -- --filter *VulkanFlashAttention* --runtimes net10.0</c>.
/// To compare against llama.cpp's Vulkan FA path, run llama-bench with
/// <c>-fa 1 -p 512,2048,4096 -t 1 -ngl 99</c> on the same prompt-length grid;
/// the dotLLM-vs-llama.cpp wall-clock ratio is the GAIA/lemonade-research
/// acceptance metric.
/// </para>
/// </remarks>
[SimpleJob(warmupCount: 2, iterationCount: 5)]
public class VulkanFlashAttentionBenchmarks
{
    /// <summary>Sequence length to benchmark (pp512 / pp2048 / pp4096).</summary>
    [Params(512, 2048, 4096)]
    public int SeqLen { get; set; }

    /// <summary>Number of query heads (Llama-3.2-1B has 32).</summary>
    [Params(32)]
    public int NumHeads { get; set; }

    /// <summary>Number of KV heads (GQA-4 with 32 / 8 query heads).</summary>
    [Params(8)]
    public int NumKvHeads { get; set; }

    /// <summary>Per-head dimension (Llama-3.2-1B has 64).</summary>
    [Params(64)]
    public int HeadDim { get; set; }

    private VulkanDevice _device = null!;
    private AttentionF32Kernel _naive = null!;
    private VulkanFlashAttentionF32Kernel _fa = null!;
    private VulkanDevice.Buffer _q = null!;
    private VulkanDevice.Buffer _k = null!;
    private VulkanDevice.Buffer _v = null!;
    private VulkanDevice.Buffer _outputNaive = null!;
    private VulkanDevice.Buffer _outputFa = null!;

    [GlobalSetup]
    public void Setup()
    {
        string spvDir = ResolveSpvDir();
        _device = VulkanDevice.Create();
        _naive = AttentionF32Kernel.Create(_device, spvDir);
        _fa = VulkanFlashAttentionF32Kernel.Create(_device, spvDir);

        int seqQ = SeqLen;
        int seqKv = SeqLen;
        long qElems  = (long)seqQ * NumHeads * HeadDim;
        long kvElems = (long)seqKv * NumKvHeads * HeadDim;

        _q = _device.Allocate(qElems * sizeof(float));
        _k = _device.Allocate(kvElems * sizeof(float));
        _v = _device.Allocate(kvElems * sizeof(float));
        _outputNaive = _device.Allocate(qElems * sizeof(float));
        _outputFa    = _device.Allocate(qElems * sizeof(float));

        var rng = new Random(0xF1A54);
        float[] tmp = new float[Math.Max(qElems, kvElems)];
        FillRandom(rng, tmp, (int)qElems);
        _device.Upload(tmp.AsSpan(0, (int)qElems), _q);
        FillRandom(rng, tmp, (int)kvElems);
        _device.Upload(tmp.AsSpan(0, (int)kvElems), _k);
        FillRandom(rng, tmp, (int)kvElems);
        _device.Upload(tmp.AsSpan(0, (int)kvElems), _v);
    }

    private static void FillRandom(Random rng, float[] buf, int count)
    {
        for (int i = 0; i < count; i++)
            buf[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
    }

    [Benchmark(Description = "Naive (per-token) attention")]
    public void Naive()
    {
        _naive.Launch(_q, _k, _v, _outputNaive,
            seqQ: SeqLen, seqKv: SeqLen,
            numHeads: NumHeads, numKvHeads: NumKvHeads, headDim: HeadDim,
            positionOffset: 0, slidingWindow: 0);
    }

    [Benchmark(Description = "Flash-Attention", Baseline = true)]
    public void FlashAttention()
    {
        _fa.Launch(_q, _k, _v, _outputFa,
            seqQ: SeqLen, seqKv: SeqLen,
            numHeads: NumHeads, numKvHeads: NumKvHeads, headDim: HeadDim,
            positionOffset: 0, slidingWindow: 0);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _outputFa?.Dispose();
        _outputNaive?.Dispose();
        _v?.Dispose();
        _k?.Dispose();
        _q?.Dispose();
        _fa?.Dispose();
        _naive?.Dispose();
        _device?.Dispose();
    }

    private static string ResolveSpvDir()
    {
        // Same heuristic the Vulkan kernel tests use: look next to the running
        // assembly for the spv/ folder.
        string baseDir = AppContext.BaseDirectory;
        string candidate = Path.Combine(baseDir, "spv");
        if (Directory.Exists(candidate)) return candidate;
        // Repo-root fallback for `dotnet run` from a non-bin cwd.
        string root = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "native", "vulkan", "spv"));
        if (Directory.Exists(root)) return root;
        throw new DirectoryNotFoundException(
            $"Vulkan SPV directory not found near {baseDir}. Rebuild the Vulkan project to copy native/vulkan/spv/.");
    }
}
