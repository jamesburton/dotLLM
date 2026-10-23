using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #437 follow-up: how much of the RoPE rotation is lost to the fp32 angle <c>pos * theta^(-2i/d)</c> at very long positions, on the GPU shader
/// versus the CPU float table versus a double-precision truth. The shader mirrors the CPU table's formula, so it can only diverge from the CPU by
/// the GPU's <c>pow</c>/<c>sin</c>/<c>cos</c> accuracy; HF computes the same fp32 product, so loss shared with the CPU is the convention, not a bug.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanRopeAngleAccuracyTests
{
    private readonly ITestOutputHelper _out;
    public VulkanRopeAngleAccuracyTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(1.0e7f, 128)]
    [InlineData(1.0e6f, 256)]
    [InlineData(10000f, 128)]
    public void ShaderAngleError_VsDoubleTruth_AtLongPositions(float theta, int ropeDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        int[] bases = [1024, 8192, 32768, 65536, 131072, 262143];
        var positions = new List<int>();
        var rng = new Random(9);
        foreach (int b in bases) for (int i = 0; i < 24; i++) positions.Add(Math.Max(0, b - rng.Next(0, 512)));
        int T = positions.Count, half = ropeDim / 2;
        // One head, NeoX pairs (i, i + half): q = (1, 0) rotates to (cos, sin).
        var q = new float[T * ropeDim];
        for (int t = 0; t < T; t++) for (int i = 0; i < half; i++) q[t * ropeDim + i] = 1f;
        var k = (float[])q.Clone();
        using var device = VulkanDevice.Create();
        using var kernel = RopeF32Kernel.Create(device, spvDir);
        using var bQ = device.Allocate(q.Length * 4L);
        using var bK = device.Allocate(k.Length * 4L);
        using var bP = device.Allocate(T * 4L);
        device.Upload(q.AsSpan(), bQ); device.Upload(k.AsSpan(), bK);
        device.Upload(MemoryMarshal.AsBytes(positions.ToArray().AsSpan()), bP);
        kernel.Launch(bQ, bK, bP, T, 1, 1, ropeDim, ropeDim, theta, RopeF32Kernel.Variant.NeoX);
        var gpu = new float[q.Length];
        device.Download(bQ, gpu);

        var cpuFreq = new float[half];
        for (int i = 0; i < half; i++) cpuFreq[i] = 1.0f / MathF.Pow(theta, 2.0f * i / ropeDim);

        _out.WriteLine($"theta {theta:E1}, rotary dims {ropeDim}: max |error| of (cos, sin) vs double truth, per base position");
        _out.WriteLine("  base      GPU shader   CPU float table");
        foreach (int b in bases)
        {
            double gMax = 0, cMax = 0;
            for (int t = 0; t < T; t++)
            {
                if (positions[t] > b || positions[t] < b - 512) continue;
                for (int i = 0; i < half; i++)
                {
                    double truth = positions[t] * Math.Pow(theta, -2.0 * i / ropeDim);
                    double tc = Math.Cos(truth), ts = Math.Sin(truth);
                    float angle = positions[t] * cpuFreq[i];
                    gMax = Math.Max(gMax, Math.Max(Math.Abs(gpu[t * ropeDim + i] - tc), Math.Abs(gpu[t * ropeDim + half + i] - ts)));
                    cMax = Math.Max(cMax, Math.Max(Math.Abs(MathF.Cos(angle) - tc), Math.Abs(MathF.Sin(angle) - ts)));
                }
            }
            _out.WriteLine($"  {b,7}   {gMax:E2}     {cMax:E2}");
            // Before the range reduction (#898) the shader was 1.5-2.7x the CPU table's error from 8K positions on (the GPU's sin/cos reduce in fp32);
            // now the only loss left is the fp32 angle product both share (and HF / llama.cpp too).
            Assert.True(gMax <= 1.3 * cMax + 1e-4, $"base {b}: GPU error {gMax:E2} vs CPU {cMax:E2}");
        }
    }
}
