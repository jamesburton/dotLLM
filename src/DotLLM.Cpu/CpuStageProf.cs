using System.Diagnostics;

namespace DotLLM.Cpu;

/// <summary>
/// Env-gated (<c>DOTLLM_CPU_PROF=1</c>) stopwatch buckets for the generic CPU layer path. Diagnostic only; dumps at process exit.
/// Disabled cost = one static readonly bool check per probe.
/// </summary>
public static class CpuStageProf
{
    /// <summary>True when <c>DOTLLM_CPU_PROF=1</c>.</summary>
    public static readonly bool On = Environment.GetEnvironmentVariable("DOTLLM_CPU_PROF") == "1";
    private static readonly long[] Ticks = new long[32];
    private static readonly long[] Calls = new long[32];
    private static readonly string[] Names =
    {
        "qkv(norm+proj)", "rope+attention", "o_proj+resid", "moe_norm", "moe_route", "moe_experts(total)", "moe_final_resid",
        "dense_ffn", "lm_head", "exp:gather+quant", "exp:gate+up gemm", "exp:swiglu", "exp:down gemm", "exp:scatter", "moe_shared+acc", "embed",
    };

    static CpuStageProf() { if (On) AppDomain.CurrentDomain.ProcessExit += (_, _) => Dump(); }

    /// <summary>Timestamp (0 when disabled).</summary>
    public static long T()
    {
        if (!On) return 0;
        if (_faultsAtFirst < 0) _faultsAtFirst = DotLLM.Cpu.Kernels.ProcessFaults.Current();
        return Stopwatch.GetTimestamp();
    }

    private static long _faultsAtFirst = -1;

    /// <summary>Adds elapsed ticks since <paramref name="t0"/> to bucket <paramref name="id"/>.</summary>
    public static void E(int id, long t0)
    {
        if (!On) return;
        Interlocked.Add(ref Ticks[id], Stopwatch.GetTimestamp() - t0);
        Interlocked.Increment(ref Calls[id]);
    }

    /// <summary>Clears all buckets.</summary>
    public static void Reset() { Array.Clear(Ticks); Array.Clear(Calls); }

    /// <summary>Writes the table to stderr.</summary>
    public static void Dump()
    {
        double f = 1000.0 / Stopwatch.Frequency;
        Console.Error.WriteLine($"[CpuProf] process page faults during generation {DotLLM.Cpu.Kernels.ProcessFaults.Current() - _faultsAtFirst}");
        Console.Error.WriteLine("[CpuProf] bucket                 total_ms     calls   us/call");
        for (int i = 0; i < Names.Length; i++)
            if (Calls[i] > 0)
                Console.Error.WriteLine($"[CpuProf] {Names[i],-22} {Ticks[i] * f,10:F1} {Calls[i],9} {Ticks[i] * f * 1000 / Calls[i],9:F1}");
    }
}
