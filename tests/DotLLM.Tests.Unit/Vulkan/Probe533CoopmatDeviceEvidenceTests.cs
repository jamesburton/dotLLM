using System.Text;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// PROBE for issue #533 — device evidence for the cross-vendor discriminator.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Probe533VulkanRowCountInvarianceTests"/> reports a parity structure
/// in <c>attention_flash_f32_coopmat.comp</c>. Taking that reading to a second
/// vendor only means something if the coopmat kernel <em>actually ran there</em>:
/// <c>VulkanFlashAttentionCoopmatKernel.TryCreate</c> returns <c>null</c> on a
/// device without a usable 16x16x16 tile and the caller silently falls back to
/// the scalar kernel, which would print "clean" and mean nothing.
/// </para>
/// <para>
/// This prints, and does not assert:
/// (a) the device identity + coopmat capability dump, and
/// (b) an <b>output-perturbation proof</b> — the coopmat kernel rounds Q/K/V to
/// f16 for the matrix multiplies while the scalar FA kernel keeps them F32, so
/// on identical inputs the two must differ by an f16-rounding-class delta.
/// <c>0 differing</c> would mean the scalar kernel ran under both labels.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe533CoopmatDeviceEvidenceTests
{
    private readonly ITestOutputHelper _out;
    public Probe533CoopmatDeviceEvidenceTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void Coopmat_DeviceEvidenceAndPerturbationProof()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();

        var sb = new StringBuilder();
        sb.AppendLine("=== #533 coopmat device evidence ===");
        sb.AppendLine($"  DeviceName           : {device.DeviceName}");
        sb.AppendLine($"  VendorId             : 0x{device.VendorId:X4} " +
                      $"({(device.VendorId == 0x10DE ? "NVIDIA" : device.VendorId == 0x1002 ? "AMD" : device.VendorId == 0x8086 ? "Intel" : "other")})");
        sb.AppendLine($"  DeviceType           : {device.DeviceType} (1=integrated, 2=discrete, 4=CPU)");
        sb.AppendLine($"  SubgroupSize         : {device.SubgroupSize}");
        sb.AppendLine($"  HasSubgroupArithmetic: {device.HasSubgroupArithmetic}");
        sb.AppendLine($"  HasCooperativeMatrix : {device.HasCooperativeMatrix}");
        sb.AppendLine($"  spvDir               : {spvDir}");
        sb.AppendLine($"  coopmat shapes ({device.SupportedCooperativeMatrixProperties.Count}):");
        foreach (var s in device.SupportedCooperativeMatrixProperties)
            sb.AppendLine($"    MxNxK={s.MSize}x{s.NSize}x{s.KSize} A={s.AType} B={s.BType} C={s.CType} R={s.ResultType} scope={s.Scope}");

        // ── Perturbation proof ────────────────────────────────────
        const int numHeads = 32, numKvHeads = 8, headDim = 64, n = 8;
        int qRow = numHeads * headDim, kvRow = numKvHeads * headDim;

        var rng = new Random(5336);
        float[] q = RandomFloats(rng, n * qRow);
        float[] kk = RandomFloats(rng, n * kvRow);
        float[] vv = RandomFloats(rng, n * kvRow);

        using var bufQ = device.Allocate((long)n * qRow * sizeof(float));
        using var bufK = device.Allocate((long)n * kvRow * sizeof(float));
        using var bufV = device.Allocate((long)n * kvRow * sizeof(float));
        using var bufO = device.Allocate((long)n * qRow * sizeof(float));
        device.Upload(q, bufQ);
        device.Upload(kk, bufK);
        device.Upload(vv, bufV);

        float[] scalarOut;
        var flash = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        if (flash is null)
        {
            sb.AppendLine("  scalar FA: UNAVAILABLE — no perturbation baseline.");
            _out.WriteLine(sb.ToString());
            return;
        }
        using (flash)
        {
            flash.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
            scalarOut = new float[(long)n * qRow];
            device.Download(bufO, scalarOut);
        }

        var coop = VulkanFlashAttentionCoopmatKernel.TryCreate(device, spvDir);
        sb.AppendLine($"  VulkanFlashAttentionCoopmatKernel.TryCreate: {(coop is null ? "NULL (coopmat FA unavailable -> production falls back to scalar)" : "non-null")}");
        if (coop is null)
        {
            _out.WriteLine(sb.ToString());
            return;
        }

        float[] coopOut;
        using (coop)
        {
            coop.Launch(bufQ, bufK, bufV, bufO, n, n, numHeads, numKvHeads, headDim);
            coopOut = new float[(long)n * qRow];
            device.Download(bufO, coopOut);
        }

        long differing = 0;
        float maxAbs = 0, maxRel = 0, maxMag = 0;
        for (int i = 0; i < scalarOut.Length; i++)
        {
            maxMag = MathF.Max(maxMag, MathF.Abs(scalarOut[i]));
            if (BitConverter.SingleToInt32Bits(scalarOut[i]) != BitConverter.SingleToInt32Bits(coopOut[i]))
            {
                differing++;
                float d = MathF.Abs(scalarOut[i] - coopOut[i]);
                maxAbs = MathF.Max(maxAbs, d);
                float den = MathF.Abs(scalarOut[i]);
                if (den > 1e-6f) maxRel = MathF.Max(maxRel, d / den);
            }
        }

        sb.AppendLine("  PERTURBATION PROOF (coopmat vs scalar FA, identical inputs, n=8 square):");
        sb.AppendLine($"    differing = {differing}/{scalarOut.Length}");
        sb.AppendLine($"    maxAbs    = {maxAbs:E3}   maxRel(elementwise, den>1e-6) = {maxRel:E3}   refMaxMag = {maxMag:E3}");
        // Scale-relative is the meaningful metric: elementwise maxRel explodes on
        // near-zero denominators and says nothing. f16 has ~1e-3 relative precision,
        // so maxAbs/refMaxMag in [1e-5, 1e-2] is the f16-rounding class.
        float scaleRel = maxMag > 0 ? maxAbs / maxMag : 0f;
        sb.AppendLine($"    maxAbs / refMaxMag = {scaleRel:E3}   (f32 ULP ~1E-07, f16 rounding ~1E-03)");
        sb.AppendLine(differing == 0
            ? "    VERDICT: IDENTICAL -> the coopmat kernel did NOT run a distinct path. Parity numbers are UNINTERPRETABLE."
            : scaleRel < 1e-5f
                ? "    VERDICT: f32-ULP-class only -> suspicious; coopmat should show f16 rounding. Check the path."
                : scaleRel > 1e-1f
                    ? "    VERDICT: O(1) divergence -> coopmat output is GROSSLY wrong on this device (not f16 rounding)."
                    : "    VERDICT: f16-rounding-class divergence -> the coopmat kernel DID execute its own path.");

        _out.WriteLine(sb.ToString());
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
