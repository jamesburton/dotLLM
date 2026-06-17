using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

namespace DotLLM.Tests.Integration.Backends;

/// <summary>
/// Backend a cross-backend timing run targets. <see cref="Cpu"/> always works;
/// <see cref="Vulkan"/> / <see cref="Cuda"/> are skipped cleanly when no physical
/// device is present (see <see cref="CrossBackendTimingHarness.IsAvailable"/>).
/// </summary>
public enum TimingBackend
{
    /// <summary>CPU reference path (<see cref="TransformerModel"/>). Always available.</summary>
    Cpu,
    /// <summary>Vulkan compute path (<see cref="VulkanTransformerModel"/>). Needs a Vulkan device.</summary>
    Vulkan,
    /// <summary>CUDA path (<see cref="CudaTransformerModel"/>). Needs an NVIDIA device + driver.</summary>
    Cuda,
}

/// <summary>
/// Reusable, backend-neutral per-phase timing harness. Loads ANY GGUF supported by the
/// chosen backend, runs <c>warmup + N prefill forwards + M decode forwards</c>, and emits
/// one <see cref="PhaseTiming"/> per phase in the SAME CSV shape as the CPU
/// <see cref="SyntheticGemma4Harness"/> (<c>phase,name,ms,tokens_per_sec</c>) so CPU,
/// Vulkan and CUDA results compose into one table.
/// </summary>
/// <remarks>
/// <para><b>Backend selection is graceful.</b> Call <see cref="IsAvailable"/> to test a
/// backend before <see cref="Run"/>; <see cref="TryRun"/> returns <c>false</c> with a
/// <paramref name="skipReason"/> instead of throwing when the device is missing, so a
/// caller sweeping all three backends logs a clean skip for the unavailable ones.</para>
/// <para><b>Why <see cref="IModel"/>.</b> The CPU <see cref="TransformerModel"/>, the
/// <see cref="VulkanTransformerModel"/> and the <see cref="CudaTransformerModel"/> all
/// implement <see cref="IModel"/> and an <see cref="IKvCache"/>; the harness times the
/// same uniform <c>Forward(tokens, positions, deviceId, cache)</c> contract on each, so the
/// numbers are directly comparable for any GGUF every backend can load.</para>
/// <para><b>Scope.</b> This is COARSE per-forward wall-clock timing (a <see cref="Stopwatch"/>
/// around the whole forward), matching the CPU harness rationale — sufficient to compare
/// backends on the same artifact. Fine per-stage timing is left to the BenchmarkDotNet project.</para>
/// </remarks>
public static class CrossBackendTimingHarness
{
    /// <summary>Knobs for one timing run. Defaults mirror the CPU <see cref="SyntheticGemma4Harness"/>.</summary>
    public readonly record struct Options(
        int PrefillForwards = 3,
        int PrefillTokens = 16,
        int DecodeForwards = 16,
        int WarmupForwards = 2,
        int MaxSeqLen = 256,
        bool UseKvCache = true);

    /// <summary>True when <paramref name="backend"/> has a usable device on this host (never throws).</summary>
    public static bool IsAvailable(TimingBackend backend) => backend switch
    {
        TimingBackend.Cpu => true,
        TimingBackend.Vulkan => SafeProbe(static () => VulkanDevice.IsAvailable()),
        TimingBackend.Cuda => SafeProbe(static () => CudaDevice.IsAvailable()),
        _ => false,
    };

    /// <summary>
    /// Loads <paramref name="ggufPath"/> on <paramref name="backend"/>, runs the warmup +
    /// prefill + decode sequence and returns the per-phase timing rows. Throws if the backend
    /// is unavailable or the model cannot be loaded/run on it — use <see cref="TryRun"/> for
    /// the graceful-skip variant.
    /// </summary>
    /// <param name="backend">Which backend to time.</param>
    /// <param name="ggufPath">Path to a GGUF the backend can load.</param>
    /// <param name="label">Phase-row label (e.g. "smollm-vk", "gemma4-tiny").</param>
    /// <param name="options">Run knobs.</param>
    /// <param name="spvOrPtxDir">Vulkan SPIR-V dir / CUDA PTX dir. Null ⇒ backend auto-detect.</param>
    public static IReadOnlyList<PhaseTiming> Run(
        TimingBackend backend, string ggufPath, string label,
        Options options = default, string? spvOrPtxDir = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(ggufPath);
        // record-struct default() leaves all ints at 0; substitute the documented defaults.
        if (options == default) options = new Options();

        var rows = new List<PhaseTiming>();

        // 1) Load (timed). Each backend opens its OWN GgufFile handle; the harness owns it.
        var sw = Stopwatch.StartNew();
        var (model, disposables) = LoadModel(backend, ggufPath, spvOrPtxDir);
        sw.Stop();
        rows.Add(new PhaseTiming("load", label, sw.Elapsed.TotalMilliseconds, 0));

        try
        {
            int vocab = model.Config.VocabSize;
            IKvCache? cache = options.UseKvCache ? CreateCache(backend, model, options.MaxSeqLen) : null;
            try
            {
                // 2) Warmup forwards (JIT / driver shader-compile / first-touch faults).
                for (int w = 0; w < options.WarmupForwards; w++)
                {
                    int[] ids = MakeIds(options.PrefillTokens, vocab, salt: 1000 + w);
                    int[] pos = Positions(0, options.PrefillTokens);
                    sw.Restart();
                    using (ITensor _ = model.Forward(ids, pos, -1, cache)) { }
                    sw.Stop();
                    rows.Add(new PhaseTiming("warmup", $"{label}#{w}", sw.Elapsed.TotalMilliseconds, 0));
                    ResetCache(ref cache, backend, model, options);
                }

                // 3) N prefill forwards (tokens/sec = prefillTokens / forward-time).
                double prefillTotalMs = 0;
                for (int it = 0; it < options.PrefillForwards; it++)
                {
                    int[] ids = MakeIds(options.PrefillTokens, vocab, salt: it + 1);
                    int[] pos = Positions(0, options.PrefillTokens);
                    sw.Restart();
                    using (ITensor _ = model.Forward(ids, pos, -1, cache)) { }
                    sw.Stop();
                    double ms = sw.Elapsed.TotalMilliseconds;
                    prefillTotalMs += ms;
                    double tps = ms > 0 ? options.PrefillTokens / (ms / 1000.0) : 0;
                    rows.Add(new PhaseTiming("prefill", $"{label}#{it}", ms, tps));
                    ResetCache(ref cache, backend, model, options);
                }
                if (options.PrefillForwards > 0)
                {
                    double avgMs = prefillTotalMs / options.PrefillForwards;
                    double avgTps = avgMs > 0 ? options.PrefillTokens / (avgMs / 1000.0) : 0;
                    rows.Add(new PhaseTiming("prefill_avg", label, avgMs, avgTps));
                }

                // 4) M decode forwards: single-token steps after a fresh prefill (uses the
                // KV-cache when enabled; otherwise a 1-token uncached forward). tokens/sec
                // = 1 / decode-time. With no cache, decode degenerates to a 1-token prefill
                // (still a valid latency datapoint, logged with the same shape).
                ResetCache(ref cache, backend, model, options);
                int decodePos = 0;
                if (cache is not null)
                {
                    // Prime the cache with a short prefill so decode steps attend a real prefix.
                    int[] prime = MakeIds(options.PrefillTokens, vocab, salt: 7);
                    int[] primePos = Positions(0, options.PrefillTokens);
                    using (ITensor _ = model.Forward(prime, primePos, -1, cache)) { }
                    decodePos = options.PrefillTokens;
                }

                double decodeTotalMs = 0;
                for (int d = 0; d < options.DecodeForwards; d++)
                {
                    int tok = MakeIds(1, vocab, salt: 5000 + d)[0];
                    int[] one = { tok };
                    int[] onePos = { cache is not null ? decodePos : 0 };
                    sw.Restart();
                    using (ITensor _ = model.Forward(one, onePos, -1, cache)) { }
                    sw.Stop();
                    double ms = sw.Elapsed.TotalMilliseconds;
                    decodeTotalMs += ms;
                    double tps = ms > 0 ? 1.0 / (ms / 1000.0) : 0;
                    rows.Add(new PhaseTiming("decode", $"{label}#{d}", ms, tps));
                    decodePos++;
                }
                if (options.DecodeForwards > 0)
                {
                    double avgMs = decodeTotalMs / options.DecodeForwards;
                    double avgTps = avgMs > 0 ? 1.0 / (avgMs / 1000.0) : 0;
                    rows.Add(new PhaseTiming("decode_avg", label, avgMs, avgTps));
                }
            }
            finally
            {
                cache?.Dispose();
            }
        }
        finally
        {
            model.Dispose();
            for (int i = disposables.Count - 1; i >= 0; i--) disposables[i].Dispose();
        }

        return rows;
    }

    /// <summary>
    /// Graceful variant of <see cref="Run"/>: returns <c>false</c> (with <paramref name="skipReason"/>)
    /// when the backend has no device or the model cannot be loaded on it, instead of throwing.
    /// A caller sweeping CPU/Vulkan/CUDA logs a clean skip for the unavailable backends.
    /// </summary>
    public static bool TryRun(
        TimingBackend backend, string ggufPath, string label,
        out IReadOnlyList<PhaseTiming> rows, out string? skipReason,
        Options options = default, string? spvOrPtxDir = null)
    {
        rows = Array.Empty<PhaseTiming>();
        skipReason = null;

        if (!IsAvailable(backend))
        {
            skipReason = $"{backend} backend has no available device on this host.";
            return false;
        }

        try
        {
            rows = Run(backend, ggufPath, label, options, spvOrPtxDir);
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or NotSupportedException or InvalidOperationException)
        {
            skipReason = $"{backend} run failed: {ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    /// <summary>Renders a row set as a CSV document (header + one line per row).</summary>
    public static string ToCsv(IEnumerable<PhaseTiming> rows)
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine(PhaseTiming.CsvHeader);
        foreach (var r in rows) sb.AppendLine(r.ToCsv());
        return sb.ToString();
    }

    // ── Backend model construction ───────────────────────────────────────────

    private static (IModel Model, List<IDisposable> Disposables) LoadModel(
        TimingBackend backend, string ggufPath, string? spvOrPtxDir)
    {
        var disposables = new List<IDisposable>();
        switch (backend)
        {
            case TimingBackend.Cpu:
            {
                var (model, gguf, _) = ModelLoader.LoadFromGguf(ggufPath, ThreadingConfig.SingleThreaded);
                disposables.Add(gguf);
                return (model, disposables);
            }
            case TimingBackend.Vulkan:
            {
                var gguf = GgufFile.Open(ggufPath);
                disposables.Add(gguf);
                var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
                var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvOrPtxDir);
                return (model, disposables);
            }
            case TimingBackend.Cuda:
            {
                var (model, gguf, _) = CudaModelLoader.LoadFromGguf(ggufPath, deviceId: 0, spvOrPtxDir);
                disposables.Add(gguf);
                return (model, disposables);
            }
            default:
                throw new NotSupportedException($"Unknown backend {backend}.");
        }
    }

    private static IKvCache? CreateCache(TimingBackend backend, IModel model, int maxSeqLen) => backend switch
    {
        TimingBackend.Vulkan => ((VulkanTransformerModel)model).CreateKvCache(maxSeqLen),
        TimingBackend.Cuda => ((CudaTransformerModel)model).CreateKvCache(maxSeqLen),
        // CPU model has no CreateKvCache; build the standard contiguous SimpleKvCache
        // (numLayers, numKvHeads, headDim, maxSeqLen) — valid for dense / uniform-GQA
        // supported architectures. Per-layer-head-dim archs (gemma4) are not run on the
        // CPU path of THIS harness (the dedicated SyntheticGemma4Harness covers CPU gemma4).
        TimingBackend.Cpu => new SimpleKvCache(
            model.Config.NumLayers, model.Config.NumKvHeads, model.Config.HeadDim, maxSeqLen),
        _ => null,
    };

    // The KV-cache is sequence-scoped: between independent prefill iterations it must be
    // reset so positions start at 0. Cheapest portable reset is dispose + recreate.
    private static void ResetCache(ref IKvCache? cache, TimingBackend backend, IModel model, Options options)
    {
        if (cache is null) return;
        cache.Dispose();
        cache = CreateCache(backend, model, options.MaxSeqLen);
    }

    // ── Deterministic token generation (BOS-first, in-range, no special tokens) ──

    private static int[] Positions(int start, int n)
    {
        var p = new int[n];
        for (int i = 0; i < n; i++) p[i] = start + i;
        return p;
    }

    private static int[] MakeIds(int n, int vocab, int salt)
    {
        var ids = new int[n];
        uint state = 0x1234567u + (uint)salt * 2654435761u;
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            // Keep ids in [5, vocab) to avoid colliding with low special-token ids
            // (BOS/EOS/UNK/MASK = 0..4 in the synthetic fixtures).
            int range = Math.Max(1, vocab - 5);
            ids[i] = 5 + (int)(state % (uint)range);
        }
        return ids;
    }

    private static bool SafeProbe(Func<bool> probe)
    {
        try { return probe(); }
        catch { return false; }
    }
}
