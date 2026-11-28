using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Op-level gate for the F16 activation store of the Inkling 2 Vulkan kernels (#2075, gate 1): <c>fToAct</c> / <c>actToF</c> of <c>olala_common.glsl</c> (<c>ACT_F16</c>) against .NET
/// <see cref="Half"/> over round-to-nearest-even ties, the subnormal range, the overflow-to-infinity boundary (65,520), infinities and NaN, plus a random sweep over the whole F32 bit space.
/// A mutated shader (truncation instead of RNE, loaded through <c>DOTLLM_INKLING2_SPV_OVERRIDE</c>) must fail it.
/// </summary>
public sealed class Inkling2F16RoundingTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly VulkanDevice? _dev;
    private readonly string _spv = Inkling2VulkanTests.SpvDir();
    private readonly string? _ovr = Environment.GetEnvironmentVariable("DOTLLM_INKLING2_SPV_OVERRIDE");

    public Inkling2F16RoundingTests(ITestOutputHelper output)
    {
        _output = output;
        if (Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") != "1" && VulkanDevice.IsAvailable()) _dev = VulkanDevice.Create();
    }

    public void Dispose() => _dev?.Dispose();

    /// <summary>The probe inputs: every adjacent-pair midpoint of the half grid (exact ties) and its F32 neighbours, the boundary values, NaNs / infinities / zeros, F32 denormals and a random bit sweep.</summary>
    internal static uint[] Inputs()
    {
        var l = new List<uint>();
        static uint B(float f) => BitConverter.SingleToUInt32Bits(f);
        // every half value, its midpoint to the next half (a tie, exactly representable in F32) and the F32 values one ULP either side of the tie
        for (int h = 0; h < 0x7C00; h++)
        {
            float a = (float)BitConverter.UInt16BitsToHalf((ushort)h), b = (float)BitConverter.UInt16BitsToHalf((ushort)(h + 1));
            // the midpoint of the last finite half and the (non-existent) next one is 65520, the overflow tie
            float mid = h == 0x7BFF ? 65520f : (float)(((double)a + (double)b) / 2);
            uint m = B(mid);
            foreach (uint bits in new[] { B(a), m - 1, m, m + 1 })
            {
                l.Add(bits); l.Add(bits | 0x80000000u);
            }
        }
        foreach (float f in new[] { 65504f, 65519.996f, 65520f, 65520.004f, 65535f, 65536f, 1e9f, 3.4028235e38f, 5.9604645e-8f, 2.9802322e-8f, 2.9802325e-8f, 2.98023e-8f, 6.1035156e-5f, 6.103515e-5f, 6.1035163e-5f, 1f, 1.0004883f, 1.00048828125f, 1.0009766f })
        {
            l.Add(B(f)); l.Add(B(-f));
        }
        l.AddRange([0u, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x7FFFFFFFu, 0xFF800001u, 0xFFFFFFFFu]);
        for (uint d = 1; d < 64; d++) { l.Add(d); l.Add(0x80000000u | d); }          // F32 denormals (all flush to a signed zero in F16)
        for (uint e = 0; e < 256; e++) { l.Add((e << 23) | 0x1000u); l.Add((e << 23) | 0x7FFFFFu); }   // every exponent, ties-ish and full mantissas
        var rnd = new Random(20260528);
        for (int i = 0; i < 1_500_000; i++) l.Add((uint)rnd.NextInt64(0, 1L << 32));
        // a dense sweep of the half range: random mantissas under every half exponent, where the rounding decision is made
        for (int i = 0; i < 1_000_000; i++) l.Add(((uint)rnd.Next(0x31, 0x8F) << 23) | (uint)rnd.Next(0, 1 << 23) | ((rnd.Next(2) == 0) ? 0u : 0x80000000u));
        return [.. l];
    }

    private static bool IsNaN(ushort h) => (h & 0x7C00) == 0x7C00 && (h & 0x03FF) != 0;

    /// <summary>Runs the probe kernel on <paramref name="bits"/>; returns the stored 16-bit words and the decoded floats.</summary>
    private (ushort[] Stored, float[] Decoded) Run(string kernel, uint[] bits)
    {
        var dev = _dev!;
        string dir = !string.IsNullOrEmpty(_ovr) && File.Exists(Path.Combine(_ovr, kernel + ".spv")) ? _ovr : _spv;
        using var k = new OlalaKernel(dev, dir, kernel, 3, 1);
        int n = bits.Length;
        var inB = dev.AllocateDeviceLocal(n * 4L); dev.Upload(MemoryMarshal.AsBytes(bits.AsSpan()), inB);
        var outB = dev.AllocateDeviceLocal(((n + 1L) / 2) * 4); var rtB = dev.AllocateDeviceLocal(n * 4L);
        using (var sub = dev.CreateSubmitContext())
        {
            sub.Begin();
            KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
            k.Record(sub.CommandBuffer, [inB, outB, rtB], [(uint)n], (uint)((n + 255) / 256), 1, 1);
            sub.SubmitAndWait();
        }
        var raw = new float[(n + 1) / 2]; dev.Download(outB, raw);
        var dec = new float[n]; dev.Download(rtB, dec);
        inB.Dispose(); outB.Dispose(); rtB.Dispose();
        return (MemoryMarshal.Cast<float, ushort>(raw.AsSpan())[..n].ToArray(), dec);
    }

    [SkippableFact]
    public void F16Store_MatchesNetHalf_RneTiesSubnormalsOverflowNaN()
    {
        Skip.If(_dev is null, "No Vulkan device");
        var bits = Inputs();
        var (stored, decoded) = Run("inkling2_actround_f16act", bits);
        int badStore = 0, badDecode = 0, ties = 0, nan = 0, sub = 0, over = 0; string first = "";
        for (int i = 0; i < bits.Length; i++)
        {
            float f = BitConverter.UInt32BitsToSingle(bits[i]);
            ushort want = BitConverter.HalfToUInt16Bits((Half)f);
            float wantDec = (float)BitConverter.UInt16BitsToHalf(want);
            bool isNan = float.IsNaN(f);
            if (isNan) nan++;
            else if (MathF.Abs(f) >= 65520f) over++;
            else if (MathF.Abs(f) < 6.103515625e-5f && f != 0f) sub++;
            bool ok = isNan ? IsNaN(stored[i]) : stored[i] == want;
            if (!ok) { badStore++; if (first.Length == 0) first = $"input 0x{bits[i]:X8} ({f:R}): got 0x{stored[i]:X4}, .NET Half 0x{want:X4}"; }
            bool okDec = isNan ? float.IsNaN(decoded[i]) : BitConverter.SingleToUInt32Bits(decoded[i]) == BitConverter.SingleToUInt32Bits(wantDec);
            if (!okDec) badDecode++;
            if (!isNan && Math.Abs(f) < 65520f && Math.Abs(f) > 0)
            {
                double a = Math.Abs((double)f); var hv = (Half)(float)a; double lo = (double)hv, hi = (double)Half.BitIncrement(hv);
                if (a == (lo + hi) / 2 || a == (lo + (double)Half.BitDecrement(hv)) / 2) ties++;
            }
        }
        _output.WriteLine($"{bits.Length} inputs ({ties} exact RNE ties, {sub} subnormal-range, {over} overflow-range, {nan} NaN): {badStore} store mismatches, {badDecode} decode mismatches. {first}");
        Assert.True(badStore == 0, $"{badStore} F16 stores differ from .NET Half; first: {first}");
        Assert.Equal(0, badDecode);
        // the fixture must contain every class it claims to cover (checked after the comparison so a mutant dies on the mismatch, not on the census)
        Assert.True(ties > 60_000, $"the fixture holds only {ties} exact ties");   // 31,743 ties per sign between adjacent finite halves, plus the 65,520 overflow tie
        Assert.True(sub > 10_000 && over > 100 && nan > 10, "the fixture misses a class (subnormal / overflow / NaN)");
    }

    /// <summary>
    /// The final softcap runs on the logits BEFORE the host's non-finite guard (<c>CheckLogitsFinite</c>, the runtime gate of the F16 default), so it must pass NaN and infinities through:
    /// <c>tanh(clamp(x / cap, -20, 20))</c> turned +-inf into +-cap and (AMD min/max drop a NaN operand) NaN into a finite value, which made the guard blind to a real F16 overflow.
    /// Mutant: the previous shader (load it through <c>DOTLLM_INKLING2_SPV_OVERRIDE</c>) fails this test. Finite logits keep the softcap.
    /// </summary>
    [SkippableFact]
    public void Softcap_PassesNonFiniteThrough_ForTheF16Guard()
    {
        Skip.If(_dev is null, "No Vulkan device");
        var dev = _dev!;
        float[] x = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, -float.NaN, 0f, 1f, -1f, 29.5f, 100f, -1e30f, 3.4028235e38f];
        const float cap = 30f;
        string dir = !string.IsNullOrEmpty(_ovr) && File.Exists(Path.Combine(_ovr, "inkling2_softcap.spv")) ? _ovr : _spv;
        using var k = new OlalaKernel(dev, dir, "inkling2_softcap", 1, 2);
        var buf = dev.AllocateDeviceLocal(x.Length * 4L);
        try
        {
            dev.Upload(MemoryMarshal.AsBytes(x.AsSpan()), buf);
            using (var sub = dev.CreateSubmitContext())
            {
                sub.Begin();
                KernelSupport.HostToComputeBarrier(sub.CommandBuffer);
                k.Record(sub.CommandBuffer, [buf], [(uint)x.Length, BitConverter.SingleToUInt32Bits(cap)], 1, 1, 1);
                sub.SubmitAndWait();
            }
            var y = new float[x.Length]; dev.Download(buf, y);
            _output.WriteLine(string.Join(", ", x.Zip(y, (a, b) => $"{a} -> {b}")));
            for (int i = 0; i < x.Length; i++)
            {
                if (float.IsNaN(x[i])) Assert.True(float.IsNaN(y[i]), $"softcap({x[i]}) = {y[i]}: a NaN logit must reach the host guard");
                else if (float.IsInfinity(x[i])) Assert.Equal(x[i], y[i]);
                else
                {
                    float want = cap * MathF.Tanh(Math.Clamp(x[i] / cap, -20f, 20f));
                    Assert.True(MathF.Abs(y[i] - want) <= 1e-5f * cap, $"softcap({x[i]}) = {y[i]}, want {want}");
                }
            }
        }
        finally { buf.Dispose(); }
    }

    /// <summary>The BF16 twin (the shipped default) against its definition, so the probe harness itself is shown to be sensitive in both directions.</summary>
    [SkippableFact]
    public void Bf16Store_MatchesRneDefinition()
    {
        Skip.If(_dev is null, "No Vulkan device");
        var bits = Inputs();
        var (stored, _) = Run("inkling2_actround", bits);
        int bad = 0;
        for (int i = 0; i < bits.Length; i++)
        {
            uint b = bits[i];
            bool isNan = (b & 0x7FFFFFFFu) > 0x7F800000u;
            ushort want = isNan ? (ushort)((b >> 16) | 0x40) : (ushort)((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
            if (stored[i] != want) bad++;
        }
        Assert.Equal(0, bad);
    }
}
