using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.HuggingFace;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Benchmarks.Profile;

/// <summary>
/// Cold-load microbench for the VK_EXT_external_memory_host zero-copy
/// weight path. Loads a GGUF model twice into a Vulkan device — once via
/// staging copies (the legacy path), once via mmap'd host-import — and
/// reports wall time + process RSS delta for each.
///
/// On a unified-memory APU (Strix Halo, Apple Silicon, Intel iGPU) the
/// zero-copy path should:
///   • finish in time dominated by disk I/O (no host→device staging copy);
///   • show process RSS counted once for the model weights, not twice;
///   • produce identical kernel output (validated separately by
///     VulkanHostImportParityTests).
/// </summary>
internal static class VulkanHostImportProfile
{
    private const string DefaultRepoId = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
    private const string DefaultFilename = "tinyllama-1.1b-chat-v1.0.Q8_0.gguf";

    public static int Run(string[] args)
    {
        if (!VulkanDevice.IsAvailable())
        {
            Console.Error.WriteLine("Vulkan device not available — install loader/driver and retry.");
            return 2;
        }

        string? ggufPath = null;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--gguf" && i + 1 < args.Length) ggufPath = args[++i];
        }

        if (ggufPath is null)
        {
            string cacheDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".cache", "dotllm-bench");
            using var dl = new HuggingFaceDownloader();
            ggufPath = dl.DownloadFileAsync(DefaultRepoId, DefaultFilename, cacheDir).GetAwaiter().GetResult();
        }

        Console.WriteLine($"GGUF: {ggufPath}");
        long ggufBytes = new FileInfo(ggufPath).Length;
        Console.WriteLine($"GGUF size: {ggufBytes / 1024.0 / 1024.0:F1} MiB");

        using var device = VulkanDevice.Create();
        Console.WriteLine($"Device: {device.DeviceName} (vendor=0x{device.VendorId:X4})");
        Console.WriteLine($"HasExternalMemoryHost: {device.HasExternalMemoryHost}");
        if (device.HasExternalMemoryHost)
            Console.WriteLine($"minImportedHostPointerAlignment: {device.MinImportedHostPointerAlignment} bytes");

        if (!device.HasExternalMemoryHost)
        {
            Console.Error.WriteLine();
            Console.Error.WriteLine("Device does not advertise VK_EXT_external_memory_host —");
            Console.Error.WriteLine("the staging baseline is the only path available on this hardware.");
            Console.Error.WriteLine();
        }

        // ─────────────────────────────────────────────────────────────
        // Run A: staging path (DOTLLM_VULKAN_DISABLE_HOST_IMPORT=1).
        // ─────────────────────────────────────────────────────────────
        Console.WriteLine();
        Console.WriteLine("─── Run A: staging upload baseline ───");
        Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", "1");
        var stagingResult = LoadAndReport(ggufPath, device, label: "staging");

        // ─────────────────────────────────────────────────────────────
        // Run B: zero-copy host-import path.
        // ─────────────────────────────────────────────────────────────
        Console.WriteLine();
        Console.WriteLine("─── Run B: host-import zero-copy ───");
        Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DISABLE_HOST_IMPORT", null);
        var importResult = LoadAndReport(ggufPath, device, label: "host-import");

        // ─────────────────────────────────────────────────────────────
        // Summary.
        // ─────────────────────────────────────────────────────────────
        Console.WriteLine();
        Console.WriteLine("─── Summary ───");
        Console.WriteLine($"  Staging     : {stagingResult.WallMs,8:F1} ms wall   {stagingResult.RssDeltaMib,7:F1} MiB RSS-delta   {stagingResult.AllocatedMib,7:F1} MiB allocated");
        Console.WriteLine($"  Host-import : {importResult.WallMs,8:F1} ms wall   {importResult.RssDeltaMib,7:F1} MiB RSS-delta   {importResult.AllocatedMib,7:F1} MiB allocated");
        Console.WriteLine($"  Speed-up    : {stagingResult.WallMs / Math.Max(importResult.WallMs, 1):F2}×");
        Console.WriteLine($"  RSS saved   : {stagingResult.RssDeltaMib - importResult.RssDeltaMib:F1} MiB");
        Console.WriteLine($"  Import path matrices: {importResult.ZeroCopyMatrices} (staging on same run: {importResult.StagingMatrices})");
        Console.WriteLine($"  Import bytes        : {importResult.ZeroCopyMib:F1} MiB");

        return 0;
    }

    private static LoadResult LoadAndReport(string ggufPath, VulkanDevice device, string label)
    {
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var weights = TransformerWeights.LoadFromGguf(gguf, config, skipF32MoeDequant: false);

        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long rssBefore = proc.WorkingSet64;

        // Ensure both runs measure cold-cache equivalent work by running a
        // single warmup iteration to JIT the upload code; the JIT cost is
        // small but visible at sub-100ms loads. We do this by allocating
        // and disposing a 1-block weight pre-emptively. (Skipped — the
        // upload path is already JIT'd from the parity tests run earlier.)

        var sw = Stopwatch.StartNew();
        using var vw = VulkanWeights.Upload(device, weights, config.NumLayers, dequantToFp32: false);
        sw.Stop();

        proc.Refresh();
        long rssAfter = proc.WorkingSet64;

        long zeroCopyMatrices = VulkanWeights.LastUploadZeroCopyMatrices;
        long stagingMatrices = VulkanWeights.LastUploadStagingMatrices;
        long zeroCopyBytes = VulkanWeights.LastUploadZeroCopyBytes;

        double wallMs = sw.Elapsed.TotalMilliseconds;
        double rssDeltaMib = (rssAfter - rssBefore) / (1024.0 * 1024.0);
        double allocatedMib = vw.AllocatedBytes / (1024.0 * 1024.0);
        double zeroCopyMib = zeroCopyBytes / (1024.0 * 1024.0);

        Console.WriteLine($"  [{label}] wall={wallMs:F1} ms  AllocatedBytes={allocatedMib:F1} MiB");
        Console.WriteLine($"  [{label}] RSS before={rssBefore / 1024.0 / 1024.0:F1} MiB  after={rssAfter / 1024.0 / 1024.0:F1} MiB  delta={rssDeltaMib:F1} MiB");
        Console.WriteLine($"  [{label}] Matrices: zero-copy={zeroCopyMatrices}  staging={stagingMatrices}");
        Console.WriteLine($"  [{label}] Zero-copy bytes: {zeroCopyMib:F1} MiB");
        if (!string.IsNullOrEmpty(VulkanWeights.LastUploadFallbackReason))
        {
            Console.WriteLine($"  [{label}] Fallback reason (last): {VulkanWeights.LastUploadFallbackReason}");
            if (VulkanWeights.LastUploadFallbackReason == "import_rejected"
                && !string.IsNullOrEmpty(HostVisibleBuffer.LastImportFailureStage))
            {
                Console.WriteLine($"  [{label}] Import failure detail: stage={HostVisibleBuffer.LastImportFailureStage} VkResult={HostVisibleBuffer.LastImportFailureCode}");
            }
        }

        return new LoadResult(wallMs, rssDeltaMib, allocatedMib, zeroCopyMatrices, stagingMatrices, zeroCopyMib);
    }

    private readonly record struct LoadResult(
        double WallMs, double RssDeltaMib, double AllocatedMib,
        long ZeroCopyMatrices, long StagingMatrices, double ZeroCopyMib);
}
