using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The glue kernels of the Edge0 latent MoE (G3, #1981): the per-(token, expert) gather of the group latent row with the column scale
/// (<c>edge0_latent_gather</c>, reads the F16 D output), the per-group routing-weighted sum of the expert outputs written as Q8_1 blocks
/// (<c>edge0_group_combine</c>, or F16 for the GEMM path) and the ordered accumulation of the per-group F16 U outputs (<c>edge0_u_acc</c>). D and U themselves run on Q8_0
/// kernels with F16 output (single-row MMVQ, small-row MMVQ, blocked coopmat GEMM; no F32 scratch exists); the experts through the Q3_K / IQ4_NL
/// resident-bank kernels.
/// </summary>
public sealed class Edge0LatentKernels : IDisposable
{
    private sealed class Pipe : IDisposable
    {
        public readonly VulkanModule Module;
        public readonly ComputePipeline Pipeline;
        public readonly nint Pool;
        public readonly DescriptorSetCache Cache;
        public readonly int Buffers, PushBytes;
        private readonly VulkanDevice _device;

        public Pipe(VulkanDevice device, string spvDir, string name, int buffers, int pushBytes, bool wave32 = false)
        {
            _device = device; Buffers = buffers; PushBytes = pushBytes;
            string path = Path.Combine(spvDir, name + ".spv");
            if (!File.Exists(path)) throw new FileNotFoundException($"Vulkan SPIR-V not found: {path}.");
            Module = VulkanModule.LoadFromFile(device, path);
            try
            {
                Span<VkDescriptorBinding> b = stackalloc VkDescriptorBinding[buffers];
                for (int i = 0; i < buffers; i++) b[i] = new VkDescriptorBinding((uint)i);
                Pipeline = Module.CreateComputePipeline("main", b, (uint)pushBytes, requiredSubgroupSize: wave32 ? Wave32SubgroupControl.RequiredSubgroupSizeFor(device) : 0);
            }
            catch { Module.Dispose(); throw; }
            Pool = KernelSupport.CreateDescriptorPool(device, buffersPerSet: (uint)buffers);
            Cache = new DescriptorSetCache(device, Pool, Pipeline, buffersPerSet: buffers);
        }

        public unsafe void Dispatch(nint cmd, ReadOnlySpan<nint> buffers, ReadOnlySpan<uint> pc, uint gx, uint gy)
        {
            nint set = Cache.GetOrCreate(buffers);
            VulkanApi.vkCmdBindPipeline(cmd, VkPipelineBindPoint.Compute, Pipeline.Pipeline);
            VulkanApi.vkCmdBindDescriptorSets(cmd, VkPipelineBindPoint.Compute, Pipeline.Layout, 0, 1, set, 0, 0);
            fixed (uint* p = pc)
                VulkanApi.vkCmdPushConstants(cmd, Pipeline.Layout, VkShaderStageFlags.Compute, 0, (uint)PushBytes, (nint)p);
            VulkanApi.vkCmdDispatch(cmd, gx, gy, 1);
        }

        public void Dispose()
        {
            if (Pool != 0) VulkanApi.vkDestroyDescriptorPool(_device.Handle, Pool, 0);
            Pipeline.Dispose();
            Module.Dispose();
        }
    }

    private readonly Pipe _gather, _combine, _acc;
    private Pipe? _decodeD, _decodeU, _combineAll, _combineF16, _gemmD, _gemmU;
    private MatMulQ8_0MmvqMultiKernel? _multiC16;
    private Edge0WideGemvKernel? _wideD, _wideU;

    /// <summary>True when the single-row kernels that skip unused D/U groups exist (integer dot product, wave32, SPIR-V present).</summary>
    public bool HasDecodeKernels => _decodeD is not null && _decodeU is not null && _combineAll is not null;

    /// <summary>Record-strided indexed MMVQ for the Q3_K gate / up projections (null without integer dot product or its SPIR-V).</summary>
    internal MoeIndexedStridedMmvqKernel? IndexedQ3K { get; private set; }
    /// <summary>Record-strided indexed MMVQ for the IQ4_NL down projection.</summary>
    internal MoeIndexedStridedMmvqKernel? IndexedIq4Nl { get; private set; }
    /// <summary>Record-strided grouped coopmat for the Q3_K gate / up projections (null without cooperative matrices / wave64).</summary>
    internal MoeGroupedStridedCoopmatKernel? GroupedQ3K { get; private set; }
    /// <summary>Record-strided grouped coopmat for the IQ4_NL down projection.</summary>
    internal MoeGroupedStridedCoopmatKernel? GroupedIq4Nl { get; private set; }

    /// <summary>True when D / U can run for any row count with F16 output: the single-row kernels and both blocked coopmat GEMMs (wave64) exist.</summary>
    public bool HasNarrowPath => HasDecodeKernels && _gemmD is not null && _gemmU is not null;

    /// <summary>F16-output small-row (2..8) MMVQ; null when unavailable (the GEMMs then serve those rows).</summary>
    internal MatMulQ8_0MmvqMultiKernel? MultiC16 => _multiC16;

    /// <summary>F16-output wide GEMV for D at 9..16 rows (F32 in); null when unavailable (the GEMM then serves those rows).</summary>
    internal Edge0WideGemvKernel? WideD => _wideD;
    /// <summary>F16-in / F16-out wide GEMV for U at 9..16 rows; null when unavailable.</summary>
    internal Edge0WideGemvKernel? WideU => _wideU;

    private Edge0LatentKernels(Pipe gather, Pipe combine, Pipe acc) { _gather = gather; _combine = combine; _acc = acc; }

    /// <summary>Loads the three SPIR-V modules from <paramref name="spvDir"/>.</summary>
    public static Edge0LatentKernels Create(VulkanDevice device, string spvDir)
    {
        Pipe? g = null, c = null, s = null;
        try
        {
            g = new Pipe(device, spvDir, "edge0_latent_gather", 5, 5 * sizeof(uint));
            c = new Pipe(device, spvDir, "edge0_group_combine", 6, 5 * sizeof(uint));
            s = new Pipe(device, spvDir, "edge0_u_acc", 2, 2 * sizeof(uint));
            var k = new Edge0LatentKernels(g, c, s);
            if (device.HasIntegerDotProduct && (Wave32SubgroupControl.RequiredSubgroupSizeFor(device) != 0 || device.SubgroupSize >= 32)
                && File.Exists(Path.Combine(spvDir, "edge0_d_mmvq.spv")) && File.Exists(Path.Combine(spvDir, "edge0_u_mmvq.spv")) && File.Exists(Path.Combine(spvDir, "edge0_group_combine_all.spv")))
            {
                k._decodeD = new Pipe(device, spvDir, "edge0_d_mmvq", 6, 7 * sizeof(uint), wave32: true);
                k._decodeU = new Pipe(device, spvDir, "edge0_u_mmvq", 13, 6 * sizeof(uint), wave32: true);
                k._combineAll = new Pipe(device, spvDir, "edge0_group_combine_all", 6, 3 * sizeof(uint));
            }
            // The blocked coopmat GEMMs (F32-in / F16-out for D, Q8_1-in / F16-out for U) need wave64 + cooperative matrices, like the shipped blocked Q8_0 GEMM.
            if (device.HasCooperativeMatrix && device.SubgroupSize == 64
                && File.Exists(Path.Combine(spvDir, "matmul_q8_0_gemm_coopmat_128x128x4_c16.spv")) && File.Exists(Path.Combine(spvDir, "matmul_q8_0_gemm_coopmat_128x128x4_hh.spv"))
                && File.Exists(Path.Combine(spvDir, "edge0_group_combine_f16.spv")))
            {
                k._gemmD = new Pipe(device, spvDir, "matmul_q8_0_gemm_coopmat_128x128x4_c16", 3, 5 * sizeof(uint));
                k._gemmU = new Pipe(device, spvDir, "matmul_q8_0_gemm_coopmat_128x128x4_hh", 3, 5 * sizeof(uint));   // F16 in / F16 out (the shipped encoder GEMM)
                k._combineF16 = new Pipe(device, spvDir, "edge0_group_combine_f16", 5, 5 * sizeof(uint));
            }
            k._multiC16 = MatMulQ8_0MmvqMultiKernel.TryCreate(device, spvDir, f16Output: true);
            k._wideD = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: false);
            k._wideU = Edge0WideGemvKernel.TryCreate(device, spvDir, f16Input: true);
            k.IndexedQ3K = MoeIndexedStridedMmvqKernel.TryCreate(device, spvDir, MoeIqQuant.Q3_K);
            k.IndexedIq4Nl = MoeIndexedStridedMmvqKernel.TryCreate(device, spvDir, MoeIqQuant.IQ4_NL);
            if (MoeGroupedStridedCoopmatKernel.IsSupportedOn(device, spvDir, MoeIqQuant.Q3_K) && MoeGroupedStridedCoopmatKernel.IsSupportedOn(device, spvDir, MoeIqQuant.IQ4_NL))
            {
                k.GroupedQ3K = MoeGroupedStridedCoopmatKernel.Create(device, spvDir, MoeIqQuant.Q3_K);
                k.GroupedIq4Nl = MoeGroupedStridedCoopmatKernel.Create(device, spvDir, MoeIqQuant.IQ4_NL);
            }
            return k;
        }
        catch { g?.Dispose(); c?.Dispose(); s?.Dispose(); throw; }
    }

    /// <summary>Drops cached descriptor sets (scratch buffers were re-allocated).</summary>
    internal void InvalidateDescriptorCache()
    {
        _gather.Cache.Reset(); _combine.Cache.Reset(); _acc.Cache.Reset(); _decodeD?.Cache.Reset(); _decodeU?.Cache.Reset(); _combineAll?.Cache.Reset(); _combineF16?.Cache.Reset(); _gemmD?.Cache.Reset(); _gemmU?.Cache.Reset(); _multiC16?.InvalidateDescriptorCache(); _wideD?.InvalidateDescriptorCache(); _wideU?.InvalidateDescriptorCache();
        IndexedQ3K?.InvalidateDescriptorCache(); IndexedIq4Nl?.InvalidateDescriptorCache(); GroupedQ3K?.InvalidateDescriptorCache(); GroupedIq4Nl?.InvalidateDescriptorCache();
    }

    /// <summary><c>out[v, c] = float(xl[v / k, labels[indices[v]], c]) * inv[c]</c> for <paramref name="rows"/> = tokens * k rows; <paramref name="xl"/> is F16.</summary>
    public void RecordGather(nint cmd, VulkanDevice.Buffer xl, VulkanDevice.Buffer indices, VulkanDevice.Buffer labels, VulkanDevice.Buffer inv,
        VulkanDevice.Buffer output, int latent, int groups, int topK, int rows, int numExperts)
    {
        if (xl.Size < (long)(rows / topK) * groups * latent * 2) throw new ArgumentException("xl (F16) too small.", nameof(xl));
        if (output.Size < (long)rows * latent * 4) throw new ArgumentException("output too small.", nameof(output));
        if (inv.Size < (long)latent * 4) throw new ArgumentException("inv too small.", nameof(inv));
        _gather.Dispatch(cmd, [xl.Handle, indices.Handle, labels.Handle, inv.Handle, output.Handle],
            [(uint)latent, (uint)groups, (uint)topK, (uint)rows, (uint)numExperts], (uint)((latent + 255) / 256), (uint)rows);
    }

    /// <summary>
    /// <c>z[t, c] = sum over the slots of token t whose expert is in group <paramref name="group"/> of w * down</c>, rounded to F16 and written as Q8_1 blocks (for the small-row MMVQ): row t of
    /// (<paramref name="xq"/> <c>[T, L/4]</c>, <paramref name="xds"/> <c>[T, L/32]</c>). <paramref name="latent"/> must be a multiple of 32.
    /// </summary>
    public void RecordGroupCombine(nint cmd, VulkanDevice.Buffer down, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights, VulkanDevice.Buffer labels,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int latent, int topK, int group, int tokens, int numExperts)
    {
        if ((latent & 31) != 0) throw new ArgumentException("latent must be a multiple of 32.", nameof(latent));
        if (down.Size < (long)tokens * topK * latent * 4) throw new ArgumentException("down too small.", nameof(down));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(tokens, latent)) throw new ArgumentException("xq too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(tokens, latent)) throw new ArgumentException("xds too small.", nameof(xds));
        _combine.Dispatch(cmd, [down.Handle, indices.Handle, weights.Handle, labels.Handle, xq.Handle, xds.Handle],
            [(uint)latent, (uint)topK, (uint)group, (uint)tokens, (uint)numExperts], (uint)(latent / 32), (uint)tokens);
    }

    /// <summary>
    /// Single-row D: the Q8_0 matrix times the Q8_1 row already in <paramref name="xq"/> / <paramref name="xds"/>, F16 output. With <paramref name="skipUnused"/> only the groups some
    /// expert of <paramref name="indices"/> belongs to are computed (<c>edge0_d_mmvq</c>; the rows of an unused group are left untouched and never read).
    /// </summary>
    public void RecordDecodeD(nint cmd, VulkanDevice.Buffer d, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer labels, int groups, int latent, int hidden, int topK, int numExperts, bool skipUnused)
        => _decodeD!.Dispatch(cmd, [d.Handle, xq.Handle, xds.Handle, y.Handle, indices.Handle, labels.Handle],
            [(uint)(groups * latent), (uint)hidden, (uint)(hidden / 32), (uint)latent, (uint)topK, (uint)numExperts, skipUnused ? 0u : 1u], (uint)(groups * latent), 1);

    /// <summary><see cref="RecordGroupCombine"/> for all eight groups of ONE token in a single dispatch: row g of (<paramref name="xq"/>, <paramref name="xds"/>) is z_g.</summary>
    public void RecordCombineAll(nint cmd, VulkanDevice.Buffer down, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights, VulkanDevice.Buffer labels,
        VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, int latent, int topK, int numExperts)
    {
        if ((latent & 31) != 0) throw new ArgumentException("latent must be a multiple of 32.", nameof(latent));
        if (xq.Size < QuantizeQ8_1RowsKernel.PackedBytes(8, latent)) throw new ArgumentException("xq too small.", nameof(xq));
        if (xds.Size < QuantizeQ8_1RowsKernel.ScaleBytes(8, latent)) throw new ArgumentException("xds too small.", nameof(xds));
        _combineAll!.Dispatch(cmd, [down.Handle, indices.Handle, weights.Handle, labels.Handle, xq.Handle, xds.Handle], [(uint)latent, (uint)topK, (uint)numExperts], (uint)(latent / 32), 1);
    }

    /// <summary>
    /// Single-row U + group sum in one dispatch: <c>out[r] = sum over the used groups g (ascending) of U_g[r] . z_g</c> with z of the eight groups quantised as eight rows of
    /// <paramref name="latent"/> in <paramref name="xq"/> / <paramref name="xds"/> (<c>edge0_u_mmvq</c>); <paramref name="skipUnused"/> false visits all eight groups.
    /// </summary>
    public void RecordDecodeU(nint cmd, ReadOnlySpan<VulkanDevice.Buffer> u, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer labels, int hidden, int latent, int topK, int numExperts, bool skipUnused)
    {
        if (u.Length != 8) throw new ArgumentException("Exactly eight group matrices.", nameof(u));
        Span<nint> b = stackalloc nint[13];
        for (int i = 0; i < 8; i++) b[i] = u[i].Handle;
        b[8] = xq.Handle; b[9] = xds.Handle; b[10] = y.Handle; b[11] = indices.Handle; b[12] = labels.Handle;
        _decodeU!.Dispatch(cmd, b, [(uint)hidden, (uint)latent, (uint)(latent / 32), (uint)topK, (uint)numExperts, skipUnused ? 0u : 1u], (uint)hidden, 1);
    }

    private static uint[] GemmPush(int m, int k, int n)
    {
        uint blocksPerRow = (uint)(k / 32);
        return [(uint)m, (uint)k, (uint)n, blocksPerRow, (blocksPerRow * 34u + 3u) / 4u];
    }

    /// <summary>D for any row count: <c>C[n, m] (F16) = B[n, k] (F32) @ W_q8_0[m, k]^T</c> on the blocked 128x128 coopmat GEMM, F32 accumulate, one F16 rounding at the store.</summary>
    public void RecordGemmD(nint cmd, VulkanDevice.Buffer weights, VulkanDevice.Buffer b, VulkanDevice.Buffer c, int m, int k, int n)
    {
        var p = _gemmD ?? throw new NotSupportedException("The F16-output D GEMM needs cooperative matrices and wave64.");
        if (k % 32 != 0 || ((k / 32) & 1) != 0) throw new ArgumentException($"k={k} is not legal (multiple of 64).", nameof(k));
        if (weights.Size < (long)m * (k / 32) * 34) throw new ArgumentException("weights too small.", nameof(weights));
        if (b.Size < (long)n * k * 4) throw new ArgumentException("b too small.", nameof(b));
        if (c.Size < (long)n * m * 2) throw new ArgumentException("c (F16) too small.", nameof(c));
        p.Dispatch(cmd, [weights.Handle, b.Handle, c.Handle], GemmPush(m, k, n), (uint)((m + 127) / 128), (uint)((n + 127) / 128));
    }

    /// <summary>
    /// <c>z[t, c]</c> as F16 (F32 accumulate in slot order, one rounding at the store) for ONE group: the input of <see cref="RecordGemmU"/> for more than 8 rows. Same arithmetic as
    /// <see cref="RecordGroupCombine"/> before its Q8_1 packing.
    /// </summary>
    public void RecordGroupCombineF16(nint cmd, VulkanDevice.Buffer down, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights, VulkanDevice.Buffer labels,
        VulkanDevice.Buffer z, int latent, int topK, int group, int tokens, int numExperts)
    {
        var p = _combineF16 ?? throw new NotSupportedException("The F16 group combine is unavailable.");
        if (down.Size < (long)tokens * topK * latent * 4) throw new ArgumentException("down too small.", nameof(down));
        if (z.Size < (long)tokens * latent * 2) throw new ArgumentException("z (F16) too small.", nameof(z));
        p.Dispatch(cmd, [down.Handle, indices.Handle, weights.Handle, labels.Handle, z.Handle],
            [(uint)latent, (uint)topK, (uint)group, (uint)tokens, (uint)numExperts], (uint)((latent + 255) / 256), (uint)tokens);
    }

    /// <summary>U for any row count: <c>C[n, m] (F16) = z[n, k] (F16) @ W_q8_0[m, k]^T</c> on the blocked 128x128 coopmat GEMM (F16 in, F32 accumulate, one F16 rounding at the store).</summary>
    public void RecordGemmU(nint cmd, VulkanDevice.Buffer weights, VulkanDevice.Buffer z, VulkanDevice.Buffer c, int m, int k, int n)
    {
        var p = _gemmU ?? throw new NotSupportedException("The F16 U GEMM needs cooperative matrices and wave64.");
        if (k % 32 != 0 || ((k / 32) & 1) != 0) throw new ArgumentException($"k={k} is not legal (multiple of 64).", nameof(k));
        if (weights.Size < (long)m * (k / 32) * 34) throw new ArgumentException("weights too small.", nameof(weights));
        if (z.Size < (long)n * k * 2) throw new ArgumentException("z (F16) too small.", nameof(z));
        if (c.Size < (long)n * m * 2) throw new ArgumentException("c (F16) too small.", nameof(c));
        p.Dispatch(cmd, [weights.Handle, z.Handle, c.Handle], GemmPush(m, k, n), (uint)((m + 127) / 128), (uint)((n + 127) / 128));
    }

    /// <summary><c>output = first ? float(u) : output + float(u)</c> over <paramref name="n"/> floats (ascending group order gives the F32 sum ((u0 + u1) + ...)).</summary>
    public void RecordUAcc(nint cmd, VulkanDevice.Buffer u, VulkanDevice.Buffer output, int n, bool first)
    {
        if (u.Size < (long)n * 2) throw new ArgumentException("u (F16) too small.", nameof(u));
        if (output.Size < (long)n * 4) throw new ArgumentException("output too small.", nameof(output));
        _acc.Dispatch(cmd, [u.Handle, output.Handle], [(uint)n, first ? 1u : 0u], (uint)((n + 255) / 256), 1);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _gather.Dispose(); _combine.Dispose(); _acc.Dispose(); _decodeD?.Dispose(); _decodeU?.Dispose(); _combineAll?.Dispose(); _combineF16?.Dispose(); _gemmD?.Dispose(); _gemmU?.Dispose(); _multiC16?.Dispose(); _wideD?.Dispose(); _wideU?.Dispose();
        IndexedQ3K?.Dispose(); IndexedIq4Nl?.Dispose(); GroupedQ3K?.Dispose(); GroupedIq4Nl?.Dispose();
    }
}
