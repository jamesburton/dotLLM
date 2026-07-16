using System.Diagnostics;
using System.Security.Cryptography;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Model-load microbench for the Vulkan weight-upload path (issue #147).
/// Loads one GGUF end-to-end via <see cref="VulkanTransformerModel.LoadFromGguf(GgufFile, DotLLM.Core.Models.ModelConfig, string?)"/>,
/// then (optionally) runs one fixed prefill forward, reporting:
///
///   • wall time — GGUF open, CPU weight load + device upload, first forward;
///   • peak host commit (<see cref="Process.PeakPagedMemorySize64"/>) and peak
///     working set — the issue-#146/#147 pressure metrics (on Windows/WDDM a
///     mapped host-visible Vulkan allocation is commit-charged in full);
///   • upload-path counters (zero-copy vs staging matrices/bytes);
///   • a SHA-256 over the first-forward logits row — the load-parity gate
///     (identical bytes ⇒ identical load result vs a baseline build).
///
/// Run one model per process invocation so the process-lifetime peaks are
/// attributable to this load:
///
///   dotnet run --project benchmarks/DotLLM.Benchmarks -c Release -- profile-vulkan-load --gguf path/to/model.gguf [--no-forward]
/// </summary>
internal static class VulkanLoadProfile
{
    public static int Run(string[] args)
    {
        string? ggufPath = null;
        bool forward = true;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--gguf" && i + 1 < args.Length) ggufPath = args[++i];
            else if (args[i] == "--no-forward") forward = false;
        }
        if (ggufPath is null || !File.Exists(ggufPath))
        {
            Console.Error.WriteLine("Usage: profile-vulkan-load --gguf <path> [--no-forward]");
            return 2;
        }
        if (!VulkanDevice.IsAvailable())
        {
            Console.Error.WriteLine("Vulkan device not available.");
            return 2;
        }

        Console.WriteLine($"GGUF: {ggufPath} ({new FileInfo(ggufPath).Length / (1024.0 * 1024):F1} MiB)");
        ReportPeaks("start");

        var swTotal = Stopwatch.StartNew();
        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        sw.Stop();
        Console.WriteLine($"gguf-open+config : {sw.Elapsed.TotalMilliseconds,9:F1} ms");

        sw.Restart();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config);
        sw.Stop();
        Console.WriteLine($"model-load       : {sw.Elapsed.TotalMilliseconds,9:F1} ms  (CPU weight load + Vulkan upload)");
        Console.WriteLine($"upload-counters  : zeroCopy={VulkanWeights.LastUploadZeroCopyMatrices} matrices " +
                          $"({VulkanWeights.LastUploadZeroCopyBytes / (1024.0 * 1024):F1} MiB), " +
                          $"staging={VulkanWeights.LastUploadStagingMatrices} matrices, " +
                          $"lastFallbackReason='{VulkanWeights.LastUploadFallbackReason}'");
        ReportPeaks("after-load");

        if (forward)
        {
            int vocab = config.VocabSize;
            int[] tokens = new int[8];
            int[] positions = new int[8];
            for (int i = 0; i < tokens.Length; i++)
            {
                tokens[i] = (i * 977 + 11) % Math.Min(vocab, 32000);
                positions[i] = i;
            }
            try
            {
                sw.Restart();
                using var kv = model.CreateKvCache(64);
                using ITensor logits = model.Forward(tokens, positions, deviceId: -1, kv);
                sw.Stop();
                Console.WriteLine($"first-forward    : {sw.Elapsed.TotalMilliseconds,9:F1} ms");
                unsafe
                {
                    int n = (int)logits.ElementCount;
                    var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
                    var row = span[^vocab..];
                    byte[] hash = SHA256.HashData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(row));
                    int argmax = 0;
                    for (int i = 1; i < row.Length; i++)
                        if (row[i] > row[argmax]) argmax = i;
                    Console.WriteLine($"logits-parity    : sha256={Convert.ToHexString(hash)[..16]} argmax={argmax} " +
                                      $"l0={row[0]:G9} l1={row[1]:G9} lLast={row[^1]:G9}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"first-forward    : FAILED ({ex.GetType().Name}: {ex.Message})");
            }
        }

        swTotal.Stop();
        Console.WriteLine($"total            : {swTotal.Elapsed.TotalMilliseconds,9:F1} ms");
        ReportPeaks("end");
        return 0;
    }

    private static void ReportPeaks(string label)
    {
        var p = Process.GetCurrentProcess();
        p.Refresh();
        Console.WriteLine($"peaks[{label,-10}] : commit(peak)={p.PeakPagedMemorySize64 / (1024.0 * 1024),9:F1} MiB " +
                          $"ws(peak)={p.PeakWorkingSet64 / (1024.0 * 1024),9:F1} MiB " +
                          $"commit(now)={p.PagedMemorySize64 / (1024.0 * 1024),9:F1} MiB");
    }
}
