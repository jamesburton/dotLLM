using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Three-way parity test for the Vulkan FP32 attention kernel. Runs the same
/// inputs through shared-memory, subgroup, and cooperative-matrix pipelines
/// (toggled via the DOTLLM_VULKAN_FORCE_* env vars) and asserts cross-path
/// numerical agreement within the per-variant tolerance.
/// </summary>
/// <remarks>
/// <para>
/// Rationale: the three paths share the same flash-attention online-softmax
/// math but differ in reduction ordering (shared-mem vs subgroup) and input
/// precision (coopmat uses f16 A/B, f32 C). F32 non-associativity + f16
/// quantization make bitwise parity infeasible, but all three must stay
/// inside the tolerance envelope the kernel binding ships against.
/// </para>
/// <para>
/// Tolerance tiering mirrors <see cref="VulkanAttentionF32KernelTests"/>:
/// the shared↔subgroup comparison uses the strict 1e-4 / 1e-3 envelope
/// (same reduction family, only ordering differs); the coopmat comparison
/// uses 5e-4 / 5e-3 (f16 inputs add ~3× abs noise).
/// </para>
/// <para>
/// Env-var toggling requires the tests to run sequentially — see
/// <c>TestCollections.cs</c> for the <c>VulkanKernels</c> collection with
/// <c>DisableParallelization</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanAttentionCoopmatParityTests
{
    // Same strict tolerance as the scalar-parity tests — applied to the
    // shared↔subgroup comparison.
    private const float AbsTolStrict = 1e-4f;
    private const float RelTolStrict = 1e-3f;

    // Coopmat adds f16 matmul noise on top of the F32 non-associativity
    // the two other variants share.
    private const float AbsTolCoopmat = 5e-4f;
    private const float RelTolCoopmat = 5e-3f;

    [SkippableFact]
    public void CrossPath_SingleHeadDecode()
        => RunParity(seqQ: 1, seqKv: 8,   numHeads: 1, numKvHeads: 1, headDim: 64,  positionOffset: 7);

    [SkippableFact]
    public void CrossPath_SmolLmDecode()
        => RunParity(seqQ: 1, seqKv: 128, numHeads: 9, numKvHeads: 3, headDim: 64,  positionOffset: 127);

    [SkippableFact]
    public void CrossPath_SmolLmPrefill()
        // 64 queries vs 64 keys, SmolLM head config. The shape the coopmat
        // variant is expected to win on.
        => RunParity(seqQ: 64, seqKv: 64, numHeads: 9, numKvHeads: 3, headDim: 64,  positionOffset: 0);

    [SkippableFact]
    public void CrossPath_LlamaHeadDim128Decode()
        => RunParity(seqQ: 1, seqKv: 128, numHeads: 32, numKvHeads: 8, headDim: 128, positionOffset: 127);

    [SkippableFact]
    public void CrossPath_MultiTileOnlineSoftmax()
        // seqKv > TILE_KV (256) + > Bc (16) — exercises the coopmat KV-tile
        // loop's online-softmax running max/sum rescale path multiple times.
        => RunParity(seqQ: 1, seqKv: 400, numHeads: 4, numKvHeads: 2, headDim: 64,  positionOffset: 399);

    // ─────────────────────────────────────────────────────────────

    private static void RunParity(int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        // Allocate once, upload once, re-run each dispatch variant against
        // the same device buffers. Saves repeated vkAllocateMemory churn and
        // keeps the comparison apples-to-apples (identical random inputs).
        var rng = new Random(0xC00F + seqQ * 41 + seqKv * 17 + numHeads * 7 + headDim);
        float[] qh = RandomFloats(rng, seqQ * numHeads   * headDim);
        float[] kh = RandomFloats(rng, seqKv * numKvHeads * headDim);
        float[] vh = RandomFloats(rng, seqKv * numKvHeads * headDim);

        using var device = VulkanDevice.Create();
        using var bufQ   = device.Allocate((long)qh.Length * sizeof(float));
        using var bufK   = device.Allocate((long)kh.Length * sizeof(float));
        using var bufV   = device.Allocate((long)vh.Length * sizeof(float));
        using var bufOut = device.Allocate((long)seqQ * numHeads * headDim * sizeof(float));
        device.Upload(qh.AsSpan(), bufQ);
        device.Upload(kh.AsSpan(), bufK);
        device.Upload(vh.AsSpan(), bufV);

        // Scalar reference — the strict oracle every path is compared against.
        float[] expected = new float[seqQ * numHeads * headDim];
        Attention.ExecuteScalar(qh, kh, vh, expected,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);

        // Force each path in turn via the env overrides. The kernel caches
        // its dispatch mode at Create() time, so re-create per variant.
        float[]? sharedResult   = RunVariant(device, bufQ, bufK, bufV, bufOut, spvDir,
            forceSharedReduce: true, useCoopmat: false,
            expectedMode: AttentionF32Kernel.DispatchMode.SharedMem,
            seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);
        Skip.If(sharedResult is null, "shared-mem variant not available (this should never happen).");

        float[]? subgroupResult = null;
        if (device.HasSubgroupArithmetic)
        {
            subgroupResult = RunVariant(device, bufQ, bufK, bufV, bufOut, spvDir,
                forceSharedReduce: false, useCoopmat: false,
                expectedMode: AttentionF32Kernel.DispatchMode.Subgroup,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);
        }

        float[]? coopmatResult = null;
        if (device.HasCooperativeMatrix)
        {
            coopmatResult = RunVariant(device, bufQ, bufK, bufV, bufOut, spvDir,
                forceSharedReduce: false, useCoopmat: true,
                expectedMode: AttentionF32Kernel.DispatchMode.Coopmat,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);
        }

        // 1. Every active variant vs the scalar oracle — strict on shared/
        //    subgroup, looser on coopmat.
        AssertClose("shared-mem vs scalar",    expected, sharedResult!,   AbsTolStrict,  RelTolStrict);
        if (subgroupResult is not null)
            AssertClose("subgroup vs scalar",  expected, subgroupResult,  AbsTolStrict,  RelTolStrict);
        if (coopmatResult is not null)
            AssertClose("coopmat vs scalar",   expected, coopmatResult,   AbsTolCoopmat, RelTolCoopmat);

        // 2. Cross-path agreement — makes the three-way A/B explicit so a
        //    regression in one variant does not slip past a lenient scalar
        //    tolerance. Strict envelope between shared and subgroup; looser
        //    between either and coopmat.
        if (subgroupResult is not null)
            AssertClose("shared-mem vs subgroup", sharedResult!, subgroupResult,
                AbsTolStrict, RelTolStrict);
        if (coopmatResult is not null)
        {
            AssertClose("shared-mem vs coopmat", sharedResult!, coopmatResult,
                AbsTolCoopmat, RelTolCoopmat);
            if (subgroupResult is not null)
                AssertClose("subgroup vs coopmat", subgroupResult, coopmatResult,
                    AbsTolCoopmat, RelTolCoopmat);
        }
    }

    private static float[] RunVariant(
        VulkanDevice device,
        VulkanDevice.Buffer bufQ, VulkanDevice.Buffer bufK, VulkanDevice.Buffer bufV, VulkanDevice.Buffer bufOut,
        string spvDir,
        bool forceSharedReduce, bool useCoopmat,
        AttentionF32Kernel.DispatchMode expectedMode,
        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset)
    {
        string? prevShared  = Environment.GetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar);
        string? prevCoopmat = Environment.GetEnvironmentVariable(AttentionF32Kernel.UseCoopmatEnvVar);
        Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar,   forceSharedReduce ? "1" : null);
        Environment.SetEnvironmentVariable(AttentionF32Kernel.UseCoopmatEnvVar,         useCoopmat         ? "1" : null);

        try
        {
            using var kernel = AttentionF32Kernel.Create(device, spvDir);
            Assert.Equal(expectedMode, kernel.Mode);

            kernel.Launch(bufQ, bufK, bufV, bufOut,
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset);
            float[] outHost = new float[seqQ * numHeads * headDim];
            device.Download(bufOut, outHost);
            return outHost;
        }
        finally
        {
            Environment.SetEnvironmentVariable(RmsNormF32Kernel.ForceSharedReduceEnvVar, prevShared);
            Environment.SetEnvironmentVariable(AttentionF32Kernel.UseCoopmatEnvVar,       prevCoopmat);
        }
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    private static void AssertClose(string label, float[] expected, float[] actual, float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel)  maxRel = rel;
            if (diff > absTol && rel > relTol) errors++;
        }
        Assert.True(errors == 0,
            $"[{label}] attention parity drift exceeded tolerance: " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, " +
            $"absTol={absTol:G9}, relTol={relTol:G9}");
    }
}
