using System;
using System.Threading;
namespace DotLLM.Models.Architectures;

/// <summary>Env-gated (DOTLLM_G4_PROF=1) stopwatch buckets for the Gemma-4 CPU layer path.</summary>
internal static class G4Prof
{
    internal static readonly bool On = Environment.GetEnvironmentVariable("DOTLLM_G4_PROF") == "1";
    private static readonly long[] Ticks = new long[32];
    private static readonly string[] Names = { "attn_norm","qkv_gemm","qk_norm_rope","attention","o_gemm","post_attn_norm_add","ffn_norm","gate_up_gemm","geglu","down_gemm","post_ffn_norm_add","ple_inject","ple_compute","lm_head","embed","out_scale" };
    static G4Prof() { if (On) AppDomain.CurrentDomain.ProcessExit += (_, _) => Dump(); }
    internal static long T() => On ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
    internal static void E(int id, long t0) { if (On) Interlocked.Add(ref Ticks[id], System.Diagnostics.Stopwatch.GetTimestamp() - t0); }
    internal static void Reset() { Array.Clear(Ticks); }
    internal static void Dump()
    {
        double f = 1000.0 / System.Diagnostics.Stopwatch.Frequency; double tot = 0;
        for (int i = 0; i < Names.Length; i++) tot += Ticks[i] * f;
        Console.Error.WriteLine($"[G4Prof] total {tot:F0} ms");
        for (int i = 0; i < Names.Length; i++) if (Ticks[i] > 0) Console.Error.WriteLine($"[G4Prof] {Names[i],-20} {Ticks[i]*f,9:F1} ms {Ticks[i]*f*100/tot,5:F1}%");
    }
}
