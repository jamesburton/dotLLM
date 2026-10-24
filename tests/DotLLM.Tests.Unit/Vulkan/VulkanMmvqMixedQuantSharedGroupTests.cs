using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-identical parity for the extended shared-activation-quant group (issue
/// #150): a same-input decode group whose members use DIFFERENT weight quants
/// (Q4_K / Q6_K / IQ4_XS — the 8B Q4_K_M and 3B IQ4_XS projection mixes) must
/// produce exactly the same per-projection outputs whether the Q8_1 activation
/// is quantized ONCE and shared across all the group's MMVQ GEMVs (SHARE) or
/// re-quantized before each GEMV (NO_SHARE, the per-projection form the group
/// previously fell back to for any non-Q8_0 member).
/// </summary>
/// <remarks>
/// <para>
/// The Q8_1 activation format is weight-quant-independent: every MMVQ decode
/// kernel consumes the same packed int8 <c>xq</c> + per-32-block (scale, sum)
/// <c>xds</c> pair, and the quantize is deterministic — so shared vs unshared
/// must agree bit-for-bit per projection. This discriminates a sharing
/// implementation that clobbers the shared scratch between different kernel
/// types, mis-orders the barrier, or routes a member to the wrong kernel.
/// Distinct output dims per member (and members deliberately ordered
/// differently across cases) catch dispatch-routing mix-ups that same-shape
/// groups would mask.
/// </para>
/// <para>
/// Skipped without <c>VK_KHR_shader_integer_dot_product</c> — the MMVQ path
/// (and therefore sharing) is unavailable there.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMmvqMixedQuantSharedGroupTests
{
    private enum Wq { Q4K, Q6K, Iq4Xs }

    public static IEnumerable<object[]> Cases()
    {
        // 8B Q4_K_M attn mix: q (Q4_K), k (Q4_K), v (Q6_K) over hidden=1024.
        yield return new object[] { 1024, new[] { (int)Wq.Q4K, (int)Wq.Q4K, (int)Wq.Q6K }, new[] { 1024, 256, 256 } };
        // gate/up mix with the Q6_K member FIRST (routing-order discriminator).
        yield return new object[] { 512, new[] { (int)Wq.Q6K, (int)Wq.Q4K }, new[] { 1536, 1536 } };
        // 3B IQ4_XS mix incl. a Q8_0-free all-IQ group + K-quant member.
        yield return new object[] { 768, new[] { (int)Wq.Iq4Xs, (int)Wq.Q6K, (int)Wq.Iq4Xs }, new[] { 320, 64, 192 } };
        // minimum super-block K, three distinct quants, distinct small dims.
        yield return new object[] { 256, new[] { (int)Wq.Q4K, (int)Wq.Iq4Xs, (int)Wq.Q6K }, new[] { 96, 32, 64 } };
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void MixedQuantSharedGroup_BitIdenticalToPerProjection(int k, int[] quants, int[] outDims)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvqQ4K = MatMulQ4KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q4_k_mmvq.spv missing or unsupported.");
        using var mmvqQ6K = MatMulQ6KMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q6_k_mmvq.spv missing or unsupported.");
        using var mmvqIq4Xs = MatMulIq4XsMmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_iq4_xs_mmvq.spv missing or unsupported.");

        var rng = new Random(0x150 + k * 31 + outDims.Length);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        device.Upload(x, bufX);

        var bufW = new VulkanDevice.Buffer[outDims.Length];
        var sharedOut = new VulkanDevice.Buffer[outDims.Length];
        var perProjOut = new VulkanDevice.Buffer[outDims.Length];
        try
        {
            for (int p = 0; p < outDims.Length; p++)
            {
                int m = outDims[p];
                byte[] wq = (Wq)quants[p] switch
                {
                    Wq.Q4K => Q4KFixture.QuantizeRows(RandomFloats(rng, m * k, range: 0.1f), m, k),
                    Wq.Q6K => Q6KFixture.QuantizeRows(RandomFloats(rng, m * k, range: 0.1f), m, k),
                    Wq.Iq4Xs => Iq4Fixture.QuantizeRowsIq4Xs(RandomFloats(rng, m * k, range: 0.1f), m, k),
                    _ => throw new InvalidOperationException(),
                };
                long wbytes = ((long)wq.Length + 3) & ~3L;
                bufW[p] = device.Allocate(wbytes);
                sharedOut[p] = device.Allocate((long)m * sizeof(float));
                perProjOut[p] = device.Allocate((long)m * sizeof(float));
                device.Upload(new ReadOnlySpan<byte>(wq), bufW[p]);
            }

            // SHARE: ONE quantize, then each member's own MMVQ kernel against the
            // shared xq/xds — exactly the extended RecordSharedInputMmvqGroup form.
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                for (int p = 0; p < outDims.Length; p++)
                    RecordGemv((Wq)quants[p], ctx.CommandBuffer, bufW[p], bufXq, bufXds,
                        sharedOut[p], outDims[p], k);
                ctx.SubmitAndWait();
            }

            // NO_SHARE: re-quantize before each GEMV (the old per-projection form).
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                for (int p = 0; p < outDims.Length; p++)
                {
                    if (p > 0) KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
                    KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    RecordGemv((Wq)quants[p], ctx.CommandBuffer, bufW[p], bufXq, bufXds,
                        perProjOut[p], outDims[p], k);
                }
                ctx.SubmitAndWait();
            }

            for (int p = 0; p < outDims.Length; p++)
            {
                int m = outDims[p];
                float[] s = new float[m];
                float[] np = new float[m];
                device.Download(sharedOut[p], s);
                device.Download(perProjOut[p], np);
                for (int i = 0; i < m; i++)
                    Assert.True(s[i].Equals(np[i]),
                        $"Shared vs per-projection mismatch at proj {p} ({(Wq)quants[p]}), idx {i} " +
                        $"(k={k}, m={m}): shared={s[i]:G9}, perProj={np[i]:G9}.");
            }
        }
        finally
        {
            foreach (var b in bufW) b?.Dispose();
            foreach (var b in sharedOut) b?.Dispose();
            foreach (var b in perProjOut) b?.Dispose();
        }

        void RecordGemv(Wq wq, nint cmdBuf, VulkanDevice.Buffer w, VulkanDevice.Buffer xq,
            VulkanDevice.Buffer xds, VulkanDevice.Buffer y, int m, int kk)
        {
            switch (wq)
            {
                case Wq.Q4K: mmvqQ4K.Record(cmdBuf, w, xq, xds, y, m, kk); break;
                case Wq.Q6K: mmvqQ6K.Record(cmdBuf, w, xq, xds, y, m, kk); break;
                case Wq.Iq4Xs: mmvqIq4Xs.Record(cmdBuf, w, xq, xds, y, m, kk); break;
                default: throw new InvalidOperationException();
            }
        }
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }
}
