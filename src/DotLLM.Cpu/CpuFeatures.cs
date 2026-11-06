using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu;

/// <summary>
/// The CPU instruction-set tiers dotLLM's kernels dispatch on. Ordered: a higher tier includes every lower one.
/// </summary>
public enum CpuIsaTier
{
    /// <summary>No SIMD intrinsics at all: plain C# loops (what runs on an architecture with no kernel, and the test oracle).</summary>
    Scalar = 0,

    /// <summary>128-bit: SSE2/SSE3/SSSE3/SSE4.1 intrinsics and the portable <c>Vector128</c> API (Westmere-class and ARM64 NEON).</summary>
    Vector128 = 1,

    /// <summary>256-bit: AVX, AVX2, FMA and (when present) AVX-VNNI, plus the portable <c>Vector256</c> API.</summary>
    Avx2 = 2,

    /// <summary>512-bit: AVX-512 F/BW on top of everything in <see cref="Avx2"/>.</summary>
    Avx512 = 3,
}

/// <summary>
/// The one place kernels ask "may I use this instruction set?" (#1620). Every flag is
/// <c>(requested tier cap allows it) &amp;&amp; (the hardware and runtime report it)</c>, so it is exactly as correct as
/// <c>X.IsSupported</c> alone on a default run and additionally honours <c>DOTLLM_CPU_ISA</c>.
///
/// <para><b>Convention.</b> A kernel that uses an intrinsic class must be reachable only through a dispatcher that tests the
/// matching flag here, and the dispatcher must end in a scalar or <c>Vector128</c> fallback. Never test
/// <c>Avx2.IsSupported</c> and then use <c>Fma</c>/<c>AvxVnni</c>/<c>Avx512*</c>: use the flag that names the whole feature SET
/// the kernel needs (<see cref="Avx2Fma"/>, <see cref="Avx2FmaVnni"/>, <see cref="Avx512Bw"/>). Do not throw
/// <see cref="PlatformNotSupportedException"/> from a public entry point; fall back.</para>
///
/// <para><b>Override.</b> <c>DOTLLM_CPU_ISA=scalar|v128|avx2|avx512</c> (case-insensitive; empty, <c>auto</c> or <c>best</c> =
/// no cap) is read once at startup and caps the highest tier: <c>v128</c> turns every AVX/FMA/VNNI/AVX-512 flag off,
/// <c>avx2</c> turns the AVX-512 flags off, <c>scalar</c> turns every flag off. The cap never raises what the CPU lacks.
/// <c>DOTNET_EnableAVX=0</c>, <c>DOTNET_EnableAVX2=0</c>, <c>DOTNET_EnableAVX512F=0</c>, <c>DOTNET_EnableAVXVNNI=0</c> etc. keep working
/// because they change what <c>IsSupported</c> reports and every flag here is derived from it.</para>
///
/// <para><b>Cost.</b> Plain <c>static readonly bool</c> fields in a <c>beforefieldinit</c> class (no static constructor), so the JIT folds
/// them to constants exactly as it does <c>X.IsSupported</c> once the class is initialised (tier-1 / OSR / full-opt methods alike).
/// Do not turn them into properties computed per call.</para>
/// </summary>
public static class CpuFeatures
{
    /// <summary>Name of the tier-cap environment variable.</summary>
    public const string EnvVar = "DOTLLM_CPU_ISA";

    /// <summary>The cap requested through <see cref="EnvVar"/> (<see cref="CpuIsaTier.Avx512"/> = none).</summary>
    public static readonly CpuIsaTier Cap = ParseCap(Environment.GetEnvironmentVariable(EnvVar), out _);

    /// <summary>The unrecognised <see cref="EnvVar"/> value, or null when it was absent or valid (an invalid value means no cap).</summary>
    public static readonly string? InvalidCapValue = InvalidOf(Environment.GetEnvironmentVariable(EnvVar));

    /// <summary>The highest tier the hardware/runtime supports, ignoring <see cref="Cap"/>.</summary>
    public static readonly CpuIsaTier HardwareTier = ComputeHardwareTier();

    /// <summary>The tier actually in force: <c>min(<see cref="HardwareTier"/>, <see cref="Cap"/>)</c>.</summary>
    public static readonly CpuIsaTier Tier = HardwareTier < Cap ? HardwareTier : Cap;

    // ── 128-bit ────────────────────────────────────────────────────────────

    /// <summary>SSE2 (always present on x64, absent on ARM64).</summary>
    public static readonly bool Sse2 = Cap >= CpuIsaTier.Vector128 && System.Runtime.Intrinsics.X86.Sse2.IsSupported;

    /// <summary>SSE3.</summary>
    public static readonly bool Sse3 = Cap >= CpuIsaTier.Vector128 && System.Runtime.Intrinsics.X86.Sse3.IsSupported;

    /// <summary>SSSE3 (the SSE tier's <c>pshufb</c>/<c>pmaddubsw</c> kernels). Implies SSE2 and SSE3.</summary>
    public static readonly bool Ssse3 = Sse2 && Sse3 && System.Runtime.Intrinsics.X86.Ssse3.IsSupported;

    /// <summary>SSE4.1. Implies <see cref="Ssse3"/>.</summary>
    public static readonly bool Sse41 = Ssse3 && System.Runtime.Intrinsics.X86.Sse41.IsSupported;

    /// <summary>Portable <c>Vector128</c> is hardware accelerated and the cap allows 128-bit code.</summary>
    public static readonly bool Vector128Accelerated = Cap >= CpuIsaTier.Vector128 && Vector128.IsHardwareAccelerated;

    /// <summary>
    /// Whether the RUNTIME accelerates <c>Vector128</c>, ignoring <see cref="Cap"/>. Only for predicting what BCL code such as
    /// <c>TensorPrimitives</c> will do (the cap does not change the BCL); never for choosing one of dotLLM's own kernels.
    /// <c>PortableMath.Exp</c> uses it to know whether <c>TensorPrimitives.Exp</c> vectorizes or falls back to the C runtime's <c>expf</c>.
    /// </summary>
    public static readonly bool RuntimeVector128Accelerated = Vector128.IsHardwareAccelerated;

    // ── 256-bit ────────────────────────────────────────────────────────────

    /// <summary>AVX and AVX2 (the integer 256-bit tier). Does NOT imply <see cref="Fma"/>.</summary>
    public static readonly bool Avx2 = Cap >= CpuIsaTier.Avx2 && System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Avx2.IsSupported;

    /// <summary>FMA3 on 256-bit vectors. Use <see cref="Avx2Fma"/> when the kernel needs both.</summary>
    public static readonly bool Fma = Cap >= CpuIsaTier.Avx2 && System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Fma.IsSupported;

    /// <summary>AVX2 and FMA together: the feature SET of every AVX2 float kernel that uses <c>Fma.MultiplyAdd</c>.</summary>
    public static readonly bool Avx2Fma = Avx2 && Fma;

    /// <summary>AVX-VNNI (VEX-encoded <c>vpdpbusd</c> on 256-bit) with AVX2.</summary>
    public static readonly bool AvxVnni = Avx2 && System.Runtime.Intrinsics.X86.AvxVnni.IsSupported;

    /// <summary>AVX2 + FMA + AVX-VNNI: the feature set of the VNNI Q8_0 kernels that also use <c>Fma.MultiplyAdd</c>.</summary>
    public static readonly bool Avx2FmaVnni = Avx2Fma && AvxVnni;

    /// <summary>Portable <c>Vector256</c> is hardware accelerated and the cap allows 256-bit code.</summary>
    public static readonly bool Vector256Accelerated = Cap >= CpuIsaTier.Avx2 && Vector256.IsHardwareAccelerated;

    // ── 512-bit ────────────────────────────────────────────────────────────

    /// <summary>AVX-512 F. Implies <see cref="Avx2"/> (a 512-bit kernel freely mixes in 256-bit integer ops).</summary>
    public static readonly bool Avx512F = Avx2 && Cap >= CpuIsaTier.Avx512 && System.Runtime.Intrinsics.X86.Avx512F.IsSupported;

    /// <summary>AVX-512 F + BW (byte/word integer ops). Implies <see cref="Avx512F"/>.</summary>
    public static readonly bool Avx512Bw = Avx512F && System.Runtime.Intrinsics.X86.Avx512BW.IsSupported;

    /// <summary>Portable <c>Vector512</c> is hardware accelerated and the cap allows 512-bit code.</summary>
    public static readonly bool Vector512Accelerated = Cap >= CpuIsaTier.Avx512 && Vector512.IsHardwareAccelerated;

    private static string? InvalidOf(string? value) { ParseCap(value, out string? invalid); return invalid; }

    private static CpuIsaTier ComputeHardwareTier()
    {
        if (System.Runtime.Intrinsics.X86.Avx.IsSupported && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
            return System.Runtime.Intrinsics.X86.Avx512F.IsSupported && System.Runtime.Intrinsics.X86.Avx512BW.IsSupported
                ? CpuIsaTier.Avx512 : CpuIsaTier.Avx2;
        if (System.Runtime.Intrinsics.X86.Sse2.IsSupported && System.Runtime.Intrinsics.X86.Ssse3.IsSupported
            || Vector128.IsHardwareAccelerated)
            return CpuIsaTier.Vector128;
        return CpuIsaTier.Scalar;
    }

    /// <summary>
    /// Parses a <see cref="EnvVar"/> value. Null/empty/<c>auto</c>/<c>best</c> mean no cap; an unrecognised value also means no
    /// cap and is returned in <paramref name="invalid"/> so a typo cannot silently pin a slower tier.
    /// </summary>
    public static CpuIsaTier ParseCap(string? value, out string? invalid)
    {
        invalid = null;
        string v = (value ?? "").Trim().ToLowerInvariant();
        switch (v)
        {
            case "":
            case "auto":
            case "best":
            case "avx512":
            case "avx-512":
            case "avx512f":
                return CpuIsaTier.Avx512;
            case "avx2":
            case "avx":
                return CpuIsaTier.Avx2;
            case "v128":
            case "vector128":
            case "sse":
            case "ssse3":
            case "sse4":
                return CpuIsaTier.Vector128;
            case "scalar":
            case "none":
                return CpuIsaTier.Scalar;
            default:
                invalid = value;
                return CpuIsaTier.Avx512;
        }
    }

    /// <summary>One-line description for logs and test diagnostics: tier in force, hardware tier, cap and the flags.</summary>
    public static string Describe()
        => $"tier={Tier} hw={HardwareTier} cap={Cap}{(InvalidCapValue is null ? "" : $" (ignored invalid {EnvVar}='{InvalidCapValue}')")} " +
           $"Ssse3={Ssse3} Sse41={Sse41} Avx2={Avx2} Fma={Fma} AvxVnni={AvxVnni} Avx512F={Avx512F} Avx512Bw={Avx512Bw}";
}
