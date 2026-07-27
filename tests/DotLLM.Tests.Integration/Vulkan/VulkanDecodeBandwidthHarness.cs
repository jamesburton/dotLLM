using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Decode-bandwidth research harness for the Vulkan backend. Loads a real GGUF, prefills to a
/// controlled context length, then times steady-state single-token decode and derives the implied
/// effective DRAM bandwidth from the bytes streamed per token (weights + KV). Supports an A/B between
/// the default F32 <see cref="VulkanKvCache"/> and the 4-bit <see cref="VulkanTurboQuantKvCache"/> so
/// the long-context KV-read lever can be measured directly.
///
/// Gated by DOTLLM_VULKAN_PERF=1. Reports min/median of N to defuse UMA memory-bandwidth contention.
///
/// Env:
///   DOTLLM_VULKAN_PERF=1                 — required to run
///   DOTLLM_VULKAN_PERF_MODEL=&lt;path&gt;      — GGUF path (required)
///   DOTLLM_VULKAN_CONTEXT=128            — prefill context length (KV occupancy during decode)
///   DOTLLM_VULKAN_KV=default|turbo       — which KV cache to use (default)
///   DOTLLM_VULKAN_TQ_BITS=4              — TurboQuant MSE bits when KV=turbo
///   DOTLLM_VULKAN_DECODE_STEPS=64        — timed steady-state decode steps
///   DOTLLM_VULKAN_WARMUP=8               — warm-up decode steps (reported separately)
/// </summary>
[Trait("Category", "GPU")]
public sealed class VulkanDecodeBandwidthHarness
{
    private const ulong Seed = 0xC0FFEE_4B2CUL;
    private const ulong VSeedXor = 0xD1B54A32D192ED03UL;

    private readonly ITestOutputHelper _output;
    public VulkanDecodeBandwidthHarness(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void MeasureDecodeBandwidth()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF") == "1", "DOTLLM_VULKAN_PERF=1 not set.");
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader or physical device available.");

        string? modelPath = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF_MODEL");
        Skip.If(string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath), $"DOTLLM_VULKAN_PERF_MODEL not found: {modelPath}");

        string spvDir = ResolveSpvDir();
        int context = ParseEnvInt("DOTLLM_VULKAN_CONTEXT", 128);
        int warmupSteps = ParseEnvInt("DOTLLM_VULKAN_WARMUP", 8);
        int decodeSteps = ParseEnvInt("DOTLLM_VULKAN_DECODE_STEPS", 64);
        int tqBits = ParseEnvInt("DOTLLM_VULKAN_TQ_BITS", 4);
        string kvMode = (Environment.GetEnvironmentVariable("DOTLLM_VULKAN_KV") ?? "default").ToLowerInvariant();

        using var gguf = GgufFile.Open(modelPath!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        long fileLen = new FileInfo(modelPath!).Length;
        long dataBytes = fileLen - gguf.DataSectionOffset;
        (long totalTensorBytes, long embedBytes) = ComputeTensorBytes(gguf, fileLen);

        // Per-token weight read: all blocks + LM head read fully. A separate token_embd (untied) is
        // only gathered (1 row) at decode, so subtract it from the streamed-bytes estimate; for tied
        // embeddings token_embd IS the LM head and stays counted.
        long weightReadPerToken = totalTensorBytes - embedBytes;

        _output.WriteLine($"model={Path.GetFileName(modelPath)}");
        _output.WriteLine($"layers={config.NumLayers} kvHeads={config.NumKvHeads} headDim={config.HeadDim} " +
                          $"hidden={config.HiddenSize} vocab={config.VocabSize}");
        _output.WriteLine($"file_bytes={fileLen:N0} data_bytes={dataBytes:N0} tensor_bytes={totalTensorBytes:N0} " +
                          $"untied_embed_bytes={embedBytes:N0} weight_read_per_token={weightReadPerToken:N0}");
        _output.WriteLine($"kv_mode={kvMode} context={context} warmup={warmupSteps} decode_steps={decodeSteps}");

        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

        int headDim = config.HeadDim;
        bool wantTurbo = kvMode is "turbo" or "ab";
        if (wantTurbo)
            Skip.If(headDim > DotLLM.Vulkan.Kernels.TurboQuantDequantF32Kernel.MaxHeadDim || (headDim & (headDim - 1)) != 0,
                $"headDim {headDim} unsupported by TurboQuant kernels.");

        // Build a prompt of exactly `context` tokens by tiling an encoded base sentence.
        int[] baseTok = tokenizer.Encode("The history of science is a long and winding road that ");
        Assert.NotEmpty(baseTok);
        int[] prompt = new int[context];
        for (int i = 0; i < context; i++) prompt[i] = baseTok[i % baseTok.Length];
        int[] positions = new int[context];
        for (int i = 0; i < context; i++) positions[i] = i;

        int maxSeq = context + warmupSteps + decodeSteps + 8;
        string fileName = Path.GetFileName(modelPath)!;

        // kv=ab runs default then turbo BACK TO BACK in the SAME process so the comparison is immune
        // to the ~40% UMA contention swing between fresh loads.
        if (kvMode == "ab")
        {
            using (var def = model.CreateKvCache(maxSeq))
                RunDecode(model, def, "default", false, headDim, tqBits, context, config, weightReadPerToken,
                          prompt, positions, warmupSteps, decodeSteps, fileName);

            var ck = new TurboQuantCodec(headDim, tqBits, Seed, useQjl: false);
            var cv = new TurboQuantCodec(headDim, tqBits, Seed ^ VSeedXor, useQjl: false);
            using (var tq = model.CreateTurboQuantKvCache(spvDir, maxSeq, ck.MseBits,
                       ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD))
                RunDecode(model, tq, $"turbo{tqBits}", true, headDim, tqBits, context, config, weightReadPerToken,
                          prompt, positions, warmupSteps, decodeSteps, fileName);
            return;
        }

        bool turbo = kvMode == "turbo";
        IKvCache cache;
        if (turbo)
        {
            var codecK = new TurboQuantCodec(headDim, tqBits, Seed, useQjl: false);
            var codecV = new TurboQuantCodec(headDim, tqBits, Seed ^ VSeedXor, useQjl: false);
            cache = model.CreateTurboQuantKvCache(spvDir, maxSeq, codecK.MseBits,
                codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD);
        }
        else
        {
            cache = model.CreateKvCache(maxSeq);
        }
        using (cache)
            RunDecode(model, cache, turbo ? $"turbo{tqBits}" : "default", turbo, headDim, tqBits, context, config,
                      weightReadPerToken, prompt, positions, warmupSteps, decodeSteps, fileName);
    }

    private void RunDecode(VulkanTransformerModel model, IKvCache cache, string label, bool turbo,
                           int headDim, int tqBits, int context, DotLLM.Core.Models.ModelConfig config,
                           long weightReadPerToken, int[] prompt, int[] positions, int warmupSteps, int decodeSteps,
                           string fileName)
    {
        long kvBytesPerToken = turbo
            ? (long)2 * config.NumLayers * config.NumKvHeads * context * ((headDim * tqBits + 7) / 8 + sizeof(float))
            : (long)2 * config.NumLayers * config.NumKvHeads * headDim * context * sizeof(float);

        // Chunked prefill: a single 4K-token MMQ dispatch trips the gfx1151 driver watchdog
        // (VK_ERROR_DEVICE_LOST). Chunks of 256 keep each dispatch small while filling the KV cache
        // to the full context. Only the LAST chunk's logits matter (decode continues from there).
        const int PrefillChunk = 256;
        var prefillSw = Stopwatch.StartNew();
        int nextToken = 0;
        for (int off = 0; off < context; off += PrefillChunk)
        {
            int len = Math.Min(PrefillChunk, context - off);
            var chunkTok = prompt.AsSpan(off, len);
            var chunkPos = positions.AsSpan(off, len);
            using var logits = model.Forward(chunkTok, chunkPos, deviceId: -1, cache);
            if (off + len >= context) nextToken = Argmax(logits);
        }
        prefillSw.Stop();

        int nextPos = context;
        for (int i = 0; i < warmupSteps; i++)
        {
            int[] s = { nextToken }; int[] p = { nextPos };
            using var l = model.Forward(s, p, -1, cache);
            nextToken = Argmax(l); nextPos++;
        }

        var times = new double[decodeSteps];
        for (int i = 0; i < decodeSteps; i++)
        {
            int[] s = { nextToken }; int[] p = { nextPos };
            var sw = Stopwatch.StartNew();
            using (var l = model.Forward(s, p, -1, cache))
            {
                sw.Stop();
                nextToken = Argmax(l);
            }
            nextPos++;
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        double min = times[0];
        double median = times[decodeSteps / 2];
        double avg = times.Average();
        double tokPerSecMedian = 1000.0 / median;
        long bytesPerToken = weightReadPerToken + kvBytesPerToken;
        double bwMedian = bytesPerToken / (median / 1000.0) / 1e9;
        double peak = 256.0;

        _output.WriteLine($"--- {label} ctx={context} ---");
        _output.WriteLine($"prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F2} " +
                          $"decode_min_ms={min:F2} decode_median_ms={median:F2} decode_avg_ms={avg:F2}");
        _output.WriteLine($"tok_per_sec_median={tokPerSecMedian:F2} best={1000.0 / min:F2}");
        _output.WriteLine($"kv_bytes_per_token={kvBytesPerToken:N0} total_bytes_per_token={bytesPerToken:N0}");
        _output.WriteLine($"eff_bw_median_GBps={bwMedian:F1} ({bwMedian / peak * 100:F1}% of {peak:F0})");
        _output.WriteLine($"RESULT\t{fileName}\tkv={label}\tctx={context}\tmedian_ms={median:F2}\t" +
                          $"tok_s={tokPerSecMedian:F2}\tbw_GBps={bwMedian:F1}\tkvB={kvBytesPerToken}\twB={weightReadPerToken}");
    }

    // Sum exact tensor byte sizes by sorting data offsets and diffing; also return the size of the
    // token_embd tensor (used to subtract the gather-only embedding when it is untied).
    private static (long total, long embed) ComputeTensorBytes(GgufFile gguf, long fileLen)
    {
        var infos = gguf.Tensors.OrderBy(t => t.DataOffset).ToArray();
        long dataLen = fileLen - gguf.DataSectionOffset;
        long total = 0, embed = 0;
        bool hasOutput = gguf.Tensors.Any(t => t.Name is "output.weight");
        for (int i = 0; i < infos.Length; i++)
        {
            ulong end = i + 1 < infos.Length ? infos[i + 1].DataOffset : (ulong)dataLen;
            long size = (long)(end - infos[i].DataOffset);
            total += size;
            // Only treat token_embd as gather-only (untied) when a separate output.weight exists.
            if (hasOutput && infos[i].Name is "token_embd.weight")
                embed = size;
        }
        return (total, embed);
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0; float best = span[0];
        for (int i = 1; i < n; i++) if (span[i] > best) { best = span[i]; idx = i; }
        return idx;
    }

    private static int ParseEnvInt(string key, int fallback)
    {
        string? v = Environment.GetEnvironmentVariable(key);
        return int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException("SPIR-V blobs not found. Run native/vulkan/build.ps1.");
    }
}
