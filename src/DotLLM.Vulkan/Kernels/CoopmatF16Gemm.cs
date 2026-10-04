using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// F16-activation companion of a blocked 128x128x4 coopmat GEMM (Q4_K / Q5_K / Q6_K / Q8_0): same weights and output, but the
/// activation matrix B is F16 (packed halves, <c>n * k * 2</c> bytes) instead of F32. The GEMM already stages B into LDS as F16,
/// so for activations already rounded to F16 this kernel is bit-identical to the F32-B one (tested), and a producer that rounds once
/// (see <c>SwiGluF32Kernel.RecordF16Out</c>) halves the activation traffic. End to end it is equivalent, not bit-identical: the
/// separately compiled SwiGLU shader can differ from the F32 one by an F32 ULP before the F16 rounding (Tev1-4B NLL 2.503930 vs 2.503940). That matters when <c>k</c> is large: at ffn-down (k = 9216) the F32 activations plus the weights
/// overflow the 32 MB Infinity Cache and the GEMM falls to ~13.5 TFLOPS; with F16 it holds ~23.5 (n = 512). Small-k shapes are
/// neutral, so callers gate on k. Underfilled grids still split K (<see cref="CoopmatSplitK"/>, its own <c>_bh_splitk</c> twin).
/// Opt out with <c>DOTLLM_VK_F16_ACT=0</c>.
/// </summary>
internal sealed class CoopmatF16Gemm : IDisposable
{
    /// <summary>Env var that disables the F16-activation path (<c>0</c>) for A/B runs.</summary>
    public const string EnvVar = "DOTLLM_VK_F16_ACT";

    /// <summary>Smallest contraction width worth the F16 path: below this the F32 activations stay cache-resident (measured neutral).</summary>
    public const int MinK = 6144;

    private const int Tile = 128;
    private const int PushBytes = 5 * sizeof(uint);   // M, K, N, blocksPerRow, rowUints

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _pool;
    private readonly DescriptorSetCache _cache;
    private readonly CoopmatSplitK? _splitK;
    private readonly int _blockBytes, _groupSize;
    private bool _disposed;

    private CoopmatF16Gemm(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, CoopmatSplitK? splitK,
                           int blockBytes, int groupSize)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _splitK = splitK;
        _blockBytes = blockBytes;
        _groupSize = groupSize;
        _pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: 3);
        _cache = new DescriptorSetCache(device, _pool, pipeline, buffersPerSet: 3);
    }

    /// <summary>
    /// Creates the F16-activation GEMM, or null when disabled by env, the SPIR-V is missing, or the device cannot run the blocked
    /// template (cooperative matrix + native wave64, same gate as the base kernel).
    /// </summary>
    /// <param name="device">Device.</param>
    /// <param name="spvDir">SPIR-V directory.</param>
    /// <param name="plainSpv">The <c>*_bh.spv</c> file.</param>
    /// <param name="splitSpv">The <c>*_bh_splitk.spv</c> file.</param>
    /// <param name="blockBytes">Bytes per quant block.</param>
    /// <param name="groupSize">Elements per quant block (256 for K-quants, 32 for Q8_0).</param>
    public static CoopmatF16Gemm? TryCreate(VulkanDevice device, string spvDir, string plainSpv, string splitSpv,
                                            int blockBytes, int groupSize)
    {
        if (Environment.GetEnvironmentVariable(EnvVar) == "0") return null;
        if (!device.HasCooperativeMatrix || device.SubgroupSize != 64) return null;
        string path = Path.Combine(spvDir, plainSpv);
        if (!File.Exists(path)) return null;

        var module = VulkanModule.LoadFromFile(device, path);
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[3];
            for (int i = 0; i < 3; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            var pipeline = module.CreateComputePipeline("main", bindings, PushBytes, requiredSubgroupSize: 0);
            var split = CoopmatSplitK.TryCreate(device, spvDir, splitSpv, blockBytes, groupSize);
            return new CoopmatF16Gemm(device, module, pipeline, split, blockBytes, groupSize);
        }
        catch
        {
            module.Dispose();
            throw;
        }
    }

    internal void InvalidateDescriptorCache()
    {
        _cache.Reset();
        _splitK?.InvalidateDescriptorCache();
    }

    /// <summary>
    /// Records <c>C[n, m] = B_f16[n, k] @ W[m, k]^T</c>. <paramref name="inputB"/> holds F16 activations (<c>n * k * 2</c> bytes).
    /// </summary>
    public unsafe void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer inputB, VulkanDevice.Buffer outputC,
                              int m, int k, int n)
    {
        if (m <= 0 || k <= 0 || n <= 0) throw new ArgumentOutOfRangeException(nameof(m), "m, k, n must be positive.");
        if ((k % _groupSize) != 0) throw new ArgumentException($"k must be a multiple of {_groupSize}, got {k}", nameof(k));
        if ((k % 4) != 0) throw new ArgumentException("k must be a multiple of 4 (vec4 activation loads).", nameof(k));

        int blocksPerRow = k / _groupSize;
        long rowBytes = (long)blocksPerRow * _blockBytes;
        int rowUints = (int)((rowBytes + 3) / 4);
        if (weights.Size < (long)m * rowBytes) throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (inputB.Size < (long)n * k * sizeof(ushort)) throw new ArgumentException("F16 input buffer too small.", nameof(inputB));
        if (outputC.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(outputC));

        if (_splitK is not null)
        {
            int chunks = _groupSize == 32 ? blocksPerRow : blocksPerRow * 8;
            int splits = CoopmatSplitK.ChooseSplits(m, n, chunks);
            if (splits > 1)
            {
                _splitK.Record(cmdBuf, weights, inputB, outputC, m, k, n, splits);
                return;
            }
        }

        Span<nint> buffers = stackalloc nint[3] { weights.Handle, inputB.Handle, outputC.Handle };
        nint set = _cache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, set, 0, 0);
        Span<uint> pc = stackalloc uint[5] { (uint)m, (uint)k, (uint)n, (uint)blocksPerRow, (uint)rowUints };
        fixed (uint* p = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushBytes, (nint)p);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)((m + Tile - 1) / Tile), (uint)((n + Tile - 1) / Tile), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _splitK?.Dispose();
        if (_pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _pool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
