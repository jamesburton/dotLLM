using DotLLM.Core.Configuration;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Batched quantized GEMM on int8 cooperative matrices (issue #1226): <c>C[N, M] = B[N, K] @ W[M, K]^T</c> with <c>W</c> read packed in
/// place (Q8_0, Q5_0, Q4_K or Q4_1, #1235) and <c>B</c> pre-quantized to Q8_1 by <see cref="QuantizeQ8_1RowsKernel"/>. A port of llama.cpp's
/// <c>mul_mmq_cm1</c>, the prefill matmul it runs on RDNA3/3.5; see <c>mmq_cm1_main.glsl</c>. The grouped form serves MoE expert
/// banks over expert-sorted rows (the <c>mul_mat_id</c> twin).
/// </summary>
/// <remarks>
/// The shader reads coopmat accumulator elements by index and maps them to rows with the RDNA3 wave64 WMMA layout, so the kernel is
/// only created on AMD devices that report a Sint8 x Sint8 -&gt; Sint32 16x16x16 subgroup shape and can pin the pipeline to wave64, and
/// only after a small exact-integer GEMM on the device confirms that layout (<see cref="LayoutValidated"/>).
/// Activations are rounded to int8 per 32-block (the dp4a MMQ / MMVQ contract), so results differ from the F16-staged coopmat GEMM
/// at the Q8_1 rounding level.
/// </remarks>
public sealed class MatMulMmqCm1Kernel : IDisposable
{
    /// <summary>K granularity: one 32-element block (a partial last staging step is zero-filled).</summary>
    public const int KGranularity = 32;

    /// <summary>Token rows per workgroup tile (the tile-list row count for the grouped form).</summary>
    public const int RowTile = 64;

    private const int Tile = 64;   // token rows per tile (and weight rows for the 64-row form)
    private const int PushConstantBytes = 6 * sizeof(uint); // M, K, N, rowStride, numExperts, tileList
    private const uint AmdVendorId = 0x1002;

    private readonly VulkanDevice _device;
    private readonly VulkanModule _module;
    private readonly ComputePipeline _pipeline;
    private readonly nint _descriptorPool;
    private readonly DescriptorSetCache _descriptorCache;
    private readonly QuantizationType _quant;
    private readonly bool _grouped;
    private bool _disposed;

    private MatMulMmqCm1Kernel(VulkanDevice device, VulkanModule module, ComputePipeline pipeline, nint pool, QuantizationType quant, bool grouped)
    {
        _device = device;
        _module = module;
        _pipeline = pipeline;
        _descriptorPool = pool;
        _quant = quant;
        _grouped = grouped;
        _descriptorCache = new DescriptorSetCache(device, pool, pipeline, buffersPerSet: grouped ? 5 : 4);
    }

    /// <summary><c>DOTLLM_VK_MMQ_CM1=0</c> disables the int8-coopmat GEMMs (A/B against the F16 coopmat GEMMs).</summary>
    public static bool Enabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_VK_MMQ_CM1") != "0";

    /// <summary>The weight formats this kernel reads.</summary>
    /// <param name="qt">Weight format.</param>
    /// <returns><see langword="true"/> for Q8_0, Q5_0, Q4_K and Q4_1.</returns>
    public static bool IsSupportedQuant(QuantizationType qt)
        => qt is QuantizationType.Q8_0 or QuantizationType.Q5_0 or QuantizationType.Q4_K or QuantizationType.Q4_1;

    private static string SpvName(QuantizationType qt, bool grouped)
    {
        string q = qt switch { QuantizationType.Q8_0 => "q8_0", QuantizationType.Q5_0 => "q5_0", QuantizationType.Q4_1 => "q4_1", _ => "q4_k" };
        return grouped ? $"moe_grouped_{q}_mmq_cm1.spv" : $"matmul_{q}_mmq_cm1.spv";
    }

    /// <summary>
    /// Whether <paramref name="device"/> can run the kernel: AMD (the accumulator layout), a Sint8 16x16x16 subgroup coopmat shape,
    /// a pinnable wave32 compute pipeline, and the SPIR-V present.
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <param name="quant">Weight format.</param>
    /// <param name="grouped">The MoE expert-grouped form.</param>
    /// <returns><see langword="true"/> when <see cref="TryCreate"/> would succeed.</returns>
    public static bool IsSupportedOn(VulkanDevice device, string spvDir, QuantizationType quant, bool grouped = false)
        => Enabled && IsSupportedQuant(quant)
            && device.VendorId == AmdVendorId
            && device.HasCooperativeMatrix
            && device.SupportsRequiredSubgroupSize(64, (uint)VkShaderStageFlags.Compute)
            && device.SupportedCooperativeMatrixProperties.Any(p =>
                p.MSize == 16 && p.NSize == 16 && p.KSize == 16 && p.Scope == VkScopeKhr.Subgroup
                && p.AType == VkComponentTypeKhr.Sint8 && p.BType == VkComponentTypeKhr.Sint8
                && p.CType == VkComponentTypeKhr.Sint32 && p.ResultType == VkComponentTypeKhr.Sint32)
            && File.Exists(Path.Combine(spvDir, SpvName(quant, grouped)));

    /// <summary>Creates the kernel, or returns <see langword="null"/> when <see cref="IsSupportedOn"/> is false.</summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <param name="quant">Weight format (Q8_0, Q5_0, Q4_K or Q4_1).</param>
    /// <param name="grouped">The MoE expert-grouped form.</param>
    /// <returns>The kernel, or <see langword="null"/>.</returns>
    public static MatMulMmqCm1Kernel? TryCreate(VulkanDevice device, string spvDir, QuantizationType quant, bool grouped = false)
    {
        if (!IsSupportedOn(device, spvDir, quant, grouped) || !LayoutValidated(device, spvDir)) return null;
        return CreateCore(device, spvDir, quant, grouped);
    }

    private static MatMulMmqCm1Kernel CreateCore(VulkanDevice device, string spvDir, QuantizationType quant, bool grouped)
    {
        var module = VulkanModule.LoadFromFile(device, Path.Combine(spvDir, SpvName(quant, grouped)));
        int bufferCount = grouped ? 5 : 4;
        ComputePipeline pipeline;
        try
        {
            Span<VkDescriptorBinding> bindings = stackalloc VkDescriptorBinding[bufferCount];
            for (int i = 0; i < bindings.Length; i++) bindings[i] = new VkDescriptorBinding((uint)i);
            pipeline = module.CreateComputePipeline("main", bindings, PushConstantBytes, requiredSubgroupSize: 64);
        }
        catch
        {
            module.Dispose();
            throw;
        }
        nint pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)bufferCount);
        return new MatMulMmqCm1Kernel(device, module, pipeline, pool, quant, grouped);
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<VulkanDevice, object> LayoutChecks = new();

    /// <summary>
    /// Runs a 64x64x128 Q8_0 GEMM of small exact integers (d = 1) once per device and compares it bit for bit: the shader maps
    /// accumulator elements to rows by a fixed hardware layout, which a driver that executes the pipeline at another wave width would
    /// silently violate (rows of the int8 product paired with another row's scale).
    /// </summary>
    /// <param name="device">The device.</param>
    /// <param name="spvDir">Directory holding the compiled shaders.</param>
    /// <returns><see langword="true"/> when the layout holds on <paramref name="device"/>.</returns>
    internal static bool LayoutValidated(VulkanDevice device, string spvDir)
    {
        lock (LayoutChecks)
        {
            if (LayoutChecks.TryGetValue(device, out var cached)) return (bool)cached;
            bool ok;
            try
            {
                using var k = CreateCore(device, spvDir, QuantizationType.Q8_0, grouped: false);
                ok = k.SelfTest();
            }
            catch (Exception)
            {
                ok = false;
            }
            if (!ok) Console.Error.WriteLine("[vulkan] int8-coopmat MMQ disabled: accumulator layout check failed on this device (#1226).");
            LayoutChecks.Add(device, ok);
            return ok;
        }
    }

    private bool SelfTest()
    {
        const int M = 64, N = 64, K = 128, Blocks = K / 32;
        var rng = new Random(1226);
        var a = new sbyte[M * K];
        var b = new sbyte[N * K];
        for (int i = 0; i < a.Length; i++) a[i] = (sbyte)rng.Next(-8, 9);
        for (int i = 0; i < b.Length; i++) b[i] = (sbyte)rng.Next(-8, 9);
        var wq = new byte[M * Blocks * 34];
        ushort one = BitConverter.HalfToUInt16Bits((Half)1.0f);
        for (int m = 0; m < M; m++)
            for (int blk = 0; blk < Blocks; blk++)
            {
                int o = (m * Blocks + blk) * 34;
                wq[o] = (byte)one;
                wq[o + 1] = (byte)(one >> 8);
                for (int j = 0; j < 32; j++) wq[o + 2 + j] = (byte)a[m * K + blk * 32 + j];
            }
        var xds = new float[N * Blocks * 2];
        for (int i = 0; i < N * Blocks; i++) xds[2 * i] = 1f;   // the Q8_0 path never reads the sum
        using var bufW = _device.Allocate(wq.Length);
        using var bufXq = _device.Allocate(b.Length);
        using var bufXds = _device.Allocate(xds.Length * sizeof(float));
        using var bufC = _device.Allocate((long)N * M * sizeof(float));
        _device.Upload(wq, bufW);
        _device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(b.AsSpan()), bufXq);
        _device.Upload(xds, bufXds);
        Launch(bufW, bufXq, bufXds, bufC, M, K, N);
        var c = new float[N * M];
        _device.Download(bufC, c);
        for (int n = 0; n < N; n++)
            for (int m = 0; m < M; m++)
            {
                int acc = 0;
                for (int k = 0; k < K; k++) acc += a[m * K + k] * b[n * K + k];
                if (c[n * M + m] != acc) return false;
            }
        return true;
    }

    /// <summary>Whether a weight of this kernel's format with contraction length <paramref name="k"/> can be dispatched.</summary>
    /// <param name="k">Contraction length.</param>
    /// <returns><see langword="true"/> for K in whole 32-element blocks (whole 256-element super-blocks for Q4_K).</returns>
    public bool SupportsK(int k) => k > 0 && k % KGranularity == 0 && (_quant != QuantizationType.Q4_K || k % 256 == 0);

    /// <summary>Weight-row tiles of an <paramref name="m"/>-row weight (the x extent of the grid).</summary>
    /// <param name="m">Weight rows.</param>
    /// <returns><c>ceil(m / 64)</c>.</returns>
    public int MTiles(int m) => (m + Tile - 1) / Tile;

    internal void InvalidateDescriptorCache() => _descriptorCache.Reset();

    /// <summary>Synchronous dense launch (tests).</summary>
    /// <param name="weights">Packed weight rows.</param>
    /// <param name="xq">Q8_1 int8 activations.</param>
    /// <param name="xds">Q8_1 per-block (d, d*sum).</param>
    /// <param name="output">F32 <c>[n, m]</c> output.</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Contraction length.</param>
    /// <param name="n">Token rows.</param>
    public void Launch(VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer output, int m, int k, int n)
    {
        using var ctx = _device.CreateSubmitContext();
        ctx.Begin();
        Record(ctx.CommandBuffer, weights, xq, xds, output, m, k, n);
        ctx.SubmitAndWait();
    }

    /// <summary>Records the dense GEMM into <paramref name="cmdBuf"/>.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="weights">Packed weight rows.</param>
    /// <param name="xq">Q8_1 int8 activations (<see cref="QuantizeQ8_1RowsKernel"/> layout).</param>
    /// <param name="xds">Q8_1 per-block (d, d*sum).</param>
    /// <param name="output">F32 <c>[n, m]</c> output.</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="k">Contraction length; see <see cref="SupportsK"/>.</param>
    /// <param name="n">Token rows.</param>
    /// <exception cref="ArgumentException">A dimension or buffer size is invalid.</exception>
    public void Record(nint cmdBuf, VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer output, int m, int k, int n)
    {
        if (_grouped) throw new InvalidOperationException("This is the grouped kernel; use RecordGrouped.");
        Validate(weights, xq, xds, output, m, k, n, numExperts: 1);
        Bind(cmdBuf, [weights.Handle, xq.Handle, xds.Handle, output.Handle], m, k, n, numExperts: 0, tileList: 0);
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)MTiles(m), (uint)((n + Tile - 1) / Tile), 1);
    }

    /// <summary>
    /// Records the expert-grouped GEMM: <c>output[r, m] = xq[r, :] . bank[expert(r), m, :]</c> over the <paramref name="rows"/> expert-sorted
    /// rows, launching only the (expert, 64-row tile) pairs <see cref="MoeBuildTileListKernel"/> wrote after the offsets
    /// (<paramref name="dispatchArgs"/> at <paramref name="dispatchArgsOffset"/>; tile rows <see cref="RowTile"/>).
    /// </summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="bank">Packed expert bank <c>[numExperts, m, k]</c>.</param>
    /// <param name="xq">Expert-sorted Q8_1 int8 activations.</param>
    /// <param name="xds">Their per-block (d, d*sum).</param>
    /// <param name="offsetsAndTiles">Group offsets followed by the tile list.</param>
    /// <param name="output">F32 <c>[rows, m]</c> expert-sorted output.</param>
    /// <param name="dispatchArgs">Indirect dispatch arguments.</param>
    /// <param name="dispatchArgsOffset">Byte offset of this projection's argument triple.</param>
    /// <param name="m">Weight rows per expert.</param>
    /// <param name="k">Contraction length.</param>
    /// <param name="rows">Expert-sorted rows.</param>
    /// <param name="numExperts">Experts in the bank.</param>
    /// <exception cref="ArgumentException">A dimension or buffer size is invalid.</exception>
    public void RecordGroupedIndirect(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsetsAndTiles, VulkanDevice.Buffer output, VulkanDevice.Buffer dispatchArgs, int dispatchArgsOffset,
        int m, int k, int rows, int numExperts)
    {
        if (!_grouped) throw new InvalidOperationException("This is the dense kernel; use Record.");
        Validate(bank, xq, xds, output, m, k, rows, numExperts);
        if (offsetsAndTiles.Size < MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows) * sizeof(uint))
            throw new ArgumentException("offsets/tile-list buffer too small.", nameof(offsetsAndTiles));
        Bind(cmdBuf, [bank.Handle, xq.Handle, xds.Handle, output.Handle, offsetsAndTiles.Handle], m, k, rows, numExperts,
            tileList: (uint)(numExperts + 1));
        VulkanApi.vkCmdDispatchIndirect(cmdBuf, dispatchArgs.Handle, (ulong)dispatchArgsOffset);
    }

    /// <summary>Grouped GEMM on the dense 3-D grid (x = M tile, y = row tile, z = expert); tests and the no-tile-list fallback.</summary>
    /// <param name="cmdBuf">Command buffer in the recording state.</param>
    /// <param name="bank">Packed expert bank <c>[numExperts, m, k]</c>.</param>
    /// <param name="xq">Expert-sorted Q8_1 int8 activations.</param>
    /// <param name="xds">Their per-block (d, d*sum).</param>
    /// <param name="offsets">Group offsets <c>[numExperts + 1]</c>.</param>
    /// <param name="output">F32 <c>[rows, m]</c> expert-sorted output.</param>
    /// <param name="m">Weight rows per expert.</param>
    /// <param name="k">Contraction length.</param>
    /// <param name="rows">Expert-sorted rows.</param>
    /// <param name="numExperts">Experts in the bank.</param>
    /// <param name="maxRowsPerExpert">Bound on any expert's rows (sizes the row-tile grid); 0 = <paramref name="rows"/>.</param>
    public void RecordGrouped(nint cmdBuf, VulkanDevice.Buffer bank, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer offsets, VulkanDevice.Buffer output, int m, int k, int rows, int numExperts, int maxRowsPerExpert = 0)
    {
        if (!_grouped) throw new InvalidOperationException("This is the dense kernel; use Record.");
        Validate(bank, xq, xds, output, m, k, rows, numExperts);
        Bind(cmdBuf, [bank.Handle, xq.Handle, xds.Handle, output.Handle, offsets.Handle], m, k, rows, numExperts, tileList: 0);
        int rowTiles = maxRowsPerExpert > 0 ? Math.Min(maxRowsPerExpert, rows) : rows;
        VulkanApi.vkCmdDispatch(cmdBuf, (uint)MTiles(m), (uint)((rowTiles + Tile - 1) / Tile), (uint)numExperts);
    }

    private void Validate(VulkanDevice.Buffer weights, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer output,
        int m, int k, int n, int numExperts)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        if (numExperts <= 0) throw new ArgumentOutOfRangeException(nameof(numExperts));
        if (!SupportsK(k)) throw new ArgumentException($"k={k} is not supported by the {_quant} int8-coopmat GEMM.", nameof(k));
        if (weights.Size < numExperts * (long)m * DotLLM.Cpu.Kernels.Dequantize.RowByteSize(k, _quant))
            throw new ArgumentException("Weights buffer too small.", nameof(weights));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(n, k)) throw new ArgumentException("Packed activation buffer too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(n, k)) throw new ArgumentException("Activation scale buffer too small.", nameof(xds));
        if (output.Size < (long)n * m * sizeof(float)) throw new ArgumentException("Output buffer too small.", nameof(output));
    }

    private unsafe void Bind(nint cmdBuf, ReadOnlySpan<nint> buffers, int m, int k, int n, int numExperts, uint tileList)
    {
        nint descriptorSet = _descriptorCache.GetOrCreate(buffers);
        VulkanApi.vkCmdBindPipeline(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Pipeline);
        VulkanApi.vkCmdBindDescriptorSets(cmdBuf, VkPipelineBindPoint.Compute, _pipeline.Layout, 0, 1, descriptorSet, 0, 0);
        // Row stride in the shader's read unit: uint16 for Q8_0 / Q5_0 (34 / 22-byte blocks), uint for Q4_K / Q4_1 (144 / 20-byte blocks).
        long rowBytes = DotLLM.Cpu.Kernels.Dequantize.RowByteSize(k, _quant);
        uint rowStride = (uint)(_quant is QuantizationType.Q4_K or QuantizationType.Q4_1 ? rowBytes / 4 : rowBytes / 2);
        Span<uint> pc = stackalloc uint[6] { (uint)m, (uint)k, (uint)n, rowStride, (uint)numExperts, tileList };
        fixed (uint* pcPtr = pc)
            VulkanApi.vkCmdPushConstants(cmdBuf, _pipeline.Layout, VkShaderStageFlags.Compute, 0, PushConstantBytes, (nint)pcPtr);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_descriptorPool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, _descriptorPool, 0);
        _pipeline.Dispose();
        _module.Dispose();
    }
}
