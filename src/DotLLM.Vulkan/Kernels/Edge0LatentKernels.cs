using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// The three small glue kernels of the Edge0 latent MoE (G3, #1981): the per-(token, expert) gather of the group latent row with the column scale
/// (<c>edge0_latent_gather</c>), the per-group routing-weighted sum of the expert outputs (<c>edge0_group_combine</c>) and the sum of the eight
/// per-group U outputs (<c>edge0_u_sum</c>). D and U themselves run through the ordinary Q8_0 / Q4_0 matmuls; the experts through the Q3_K / IQ4_NL
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

    private readonly Pipe _gather, _combine, _sum;
    private Pipe? _decodeD, _decodeU, _combineAll, _roundTrip;

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

    private Edge0LatentKernels(Pipe gather, Pipe combine, Pipe sum) { _gather = gather; _combine = combine; _sum = sum; }

    /// <summary>Loads the three SPIR-V modules from <paramref name="spvDir"/>.</summary>
    public static Edge0LatentKernels Create(VulkanDevice device, string spvDir)
    {
        Pipe? g = null, c = null, s = null;
        try
        {
            g = new Pipe(device, spvDir, "edge0_latent_gather", 5, 5 * sizeof(uint));
            c = new Pipe(device, spvDir, "edge0_group_combine", 5, 5 * sizeof(uint));
            s = new Pipe(device, spvDir, "edge0_u_sum", 9, sizeof(uint));
            var k = new Edge0LatentKernels(g, c, s);
            if (device.HasIntegerDotProduct && (Wave32SubgroupControl.RequiredSubgroupSizeFor(device) != 0 || device.SubgroupSize >= 32)
                && File.Exists(Path.Combine(spvDir, "edge0_d_mmvq.spv")) && File.Exists(Path.Combine(spvDir, "edge0_u_mmvq.spv")) && File.Exists(Path.Combine(spvDir, "edge0_group_combine_all.spv")))
            {
                k._decodeD = new Pipe(device, spvDir, "edge0_d_mmvq", 6, 6 * sizeof(uint), wave32: true);
                k._decodeU = new Pipe(device, spvDir, "edge0_u_mmvq", 13, 5 * sizeof(uint), wave32: true);
                k._combineAll = new Pipe(device, spvDir, "edge0_group_combine_all", 5, 3 * sizeof(uint));
            }
            if (File.Exists(Path.Combine(spvDir, "edge0_roundtrip.spv"))) k._roundTrip = new Pipe(device, spvDir, "edge0_roundtrip", 1, 2 * sizeof(uint));
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
        _gather.Cache.Reset(); _combine.Cache.Reset(); _sum.Cache.Reset(); _decodeD?.Cache.Reset(); _decodeU?.Cache.Reset(); _combineAll?.Cache.Reset(); _roundTrip?.Cache.Reset();
        IndexedQ3K?.InvalidateDescriptorCache(); IndexedIq4Nl?.InvalidateDescriptorCache(); GroupedQ3K?.InvalidateDescriptorCache(); GroupedIq4Nl?.InvalidateDescriptorCache();
    }

    /// <summary><c>out[v, c] = xl[v / k, labels[indices[v]], c] * inv[c]</c> for <paramref name="rows"/> = tokens * k rows.</summary>
    public void RecordGather(nint cmd, VulkanDevice.Buffer xl, VulkanDevice.Buffer indices, VulkanDevice.Buffer labels, VulkanDevice.Buffer inv,
        VulkanDevice.Buffer output, int latent, int groups, int topK, int rows, int numExperts)
    {
        if (xl.Size < (long)(rows / topK) * groups * latent * 4) throw new ArgumentException("xl too small.", nameof(xl));
        if (output.Size < (long)rows * latent * 4) throw new ArgumentException("output too small.", nameof(output));
        if (inv.Size < (long)latent * 4) throw new ArgumentException("inv too small.", nameof(inv));
        _gather.Dispatch(cmd, [xl.Handle, indices.Handle, labels.Handle, inv.Handle, output.Handle],
            [(uint)latent, (uint)groups, (uint)topK, (uint)rows, (uint)numExperts], (uint)((latent + 255) / 256), (uint)rows);
    }

    /// <summary><c>z[t, c] = sum over the slots of token t whose expert is in group <paramref name="group"/> of w * down</c>.</summary>
    public void RecordGroupCombine(nint cmd, VulkanDevice.Buffer down, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights, VulkanDevice.Buffer labels,
        VulkanDevice.Buffer z, int latent, int topK, int group, int tokens, int numExperts)
    {
        if (down.Size < (long)tokens * topK * latent * 4) throw new ArgumentException("down too small.", nameof(down));
        if (z.Size < (long)tokens * latent * 4) throw new ArgumentException("z too small.", nameof(z));
        _combine.Dispatch(cmd, [down.Handle, indices.Handle, weights.Handle, labels.Handle, z.Handle],
            [(uint)latent, (uint)topK, (uint)group, (uint)tokens, (uint)numExperts], (uint)((latent + 255) / 256), (uint)tokens);
    }

    /// <summary>
    /// Single-row D: the Q8_0 matrix times the Q8_1 row already in <paramref name="xq"/> / <paramref name="xds"/>, computing only the groups some expert of <paramref name="indices"/> belongs to
    /// (<c>edge0_d_mmvq</c>; the rows of an unused group are left untouched and never read).
    /// </summary>
    public void RecordDecodeD(nint cmd, VulkanDevice.Buffer d, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer labels, int groups, int latent, int hidden, int topK, int numExperts)
        => _decodeD!.Dispatch(cmd, [d.Handle, xq.Handle, xds.Handle, y.Handle, indices.Handle, labels.Handle],
            [(uint)(groups * latent), (uint)hidden, (uint)(hidden / 32), (uint)latent, (uint)topK, (uint)numExperts], (uint)(groups * latent), 1);

    /// <summary><see cref="RecordGroupCombine"/> for all eight groups of ONE token in a single dispatch: <c>z[g * latent + c]</c>.</summary>
    public void RecordCombineAll(nint cmd, VulkanDevice.Buffer down, VulkanDevice.Buffer indices, VulkanDevice.Buffer weights, VulkanDevice.Buffer labels,
        VulkanDevice.Buffer z, int latent, int topK, int numExperts)
        => _combineAll!.Dispatch(cmd, [down.Handle, indices.Handle, weights.Handle, labels.Handle, z.Handle], [(uint)latent, (uint)topK, (uint)numExperts], (uint)((latent + 255) / 256), 1);

    /// <summary>
    /// Single-row U + group sum in one dispatch: <c>out[r] = sum over the used groups g (ascending) of U_g[r] . z_g</c> with z of the eight groups quantised as eight rows of
    /// <paramref name="latent"/> in <paramref name="xq"/> / <paramref name="xds"/> (<c>edge0_u_mmvq</c>).
    /// </summary>
    public void RecordDecodeU(nint cmd, ReadOnlySpan<VulkanDevice.Buffer> u, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds, VulkanDevice.Buffer y, VulkanDevice.Buffer indices,
        VulkanDevice.Buffer labels, int hidden, int latent, int topK, int numExperts)
    {
        if (u.Length != 8) throw new ArgumentException("Exactly eight group matrices.", nameof(u));
        Span<nint> b = stackalloc nint[13];
        for (int i = 0; i < 8; i++) b[i] = u[i].Handle;
        b[8] = xq.Handle; b[9] = xds.Handle; b[10] = y.Handle; b[11] = indices.Handle; b[12] = labels.Handle;
        _decodeU!.Dispatch(cmd, b, [(uint)hidden, (uint)latent, (uint)(latent / 32), (uint)topK, (uint)numExperts], (uint)hidden, 1);
    }

    /// <summary><c>output = u[0] + ... + u[7]</c> over <paramref name="n"/> floats.</summary>
    public void RecordSum(nint cmd, ReadOnlySpan<VulkanDevice.Buffer> u, VulkanDevice.Buffer output, int n)
    {
        if (u.Length != 8) throw new ArgumentException("Exactly eight group outputs are summed.", nameof(u));
        Span<nint> b = stackalloc nint[9];
        for (int i = 0; i < 8; i++) { if (u[i].Size < (long)n * 4) throw new ArgumentException("u too small.", nameof(u)); b[i] = u[i].Handle; }
        b[8] = output.Handle;
        _sum.Dispatch(cmd, b, [(uint)n], (uint)((n + 255) / 256), 1);
    }

    /// <summary>Experiment: rounds the first <paramref name="n"/> floats of <paramref name="x"/> in place through a candidate storage format (1 = F16, 2 = BF16, 3 = Q8_0).</summary>
    public void RecordRoundTrip(nint cmd, VulkanDevice.Buffer x, int n, int mode)
        => (_roundTrip ?? throw new InvalidOperationException("edge0_roundtrip.spv missing")).Dispatch(cmd, [x.Handle], [(uint)n, (uint)mode], (uint)((n + 31) / 32), 1);

    /// <inheritdoc/>
    public void Dispose()
    {
        _gather.Dispose(); _combine.Dispose(); _sum.Dispose(); _decodeD?.Dispose(); _decodeU?.Dispose(); _combineAll?.Dispose(); _roundTrip?.Dispose();
        IndexedQ3K?.Dispose(); IndexedIq4Nl?.Dispose(); GroupedQ3K?.Dispose(); GroupedIq4Nl?.Dispose();
    }
}
