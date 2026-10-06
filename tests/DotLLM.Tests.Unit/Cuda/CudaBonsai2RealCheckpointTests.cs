using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Threading;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA against the <b>real</b> Hadamard-folded Bonsai 2 27B MTP checkpoint — issue #479's last
/// acceptance criterion: it loads packed on a 12 GB card and matches CPU greedy tokens.
/// </summary>
/// <remarks>
/// <para>
/// <b>Gated.</b> Resolves the checkpoint from <c>DOTLLM_BONSAI2_MTP_GGUF</c> or the HF hub cache
/// (same lookup as <c>VulkanBonsai2MtpRealCheckpointTests</c>) and skips cleanly when absent.
/// </para>
/// <para>
/// <b>The CPU oracle is expensive, so it can be supplied instead of recomputed.</b> A 27B CPU
/// greedy run takes minutes. Set <c>DOTLLM_BONSAI2_CPU_GREEDY_TOKENS</c> to a comma-separated list
/// (the <c>cpu greedy:</c> line this test prints) to compare against that instead, or fill
/// <see cref="ExpectedCpuGreedyTokens"/> once it has been captured on real hardware. With neither,
/// the test runs the CPU model itself — it never passes vacuously.
/// </para>
/// <para>
/// Token IDs are asserted exactly. Both backends run the same weights and differ only in reduction
/// order, which does not move an argmax unless the top-2 gap is at the noise floor; the gaps are
/// printed so a red result diagnoses itself (a real basis error misses by a mile, a tie does not).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaBonsai2RealCheckpointTests
{
    private const int NewTokens = 16;
    private const int DraftSteps = 4;

    private static readonly int[] PromptTokens =
        [7734, 264, 2716, 10597, 15673, 314, 1204, 264, 4779, 42209, 311, 4623, 26642, 9714, 13];

    /// <summary>
    /// CPU greedy continuation of <see cref="PromptTokens"/> on the reference checkpoint. Empty until
    /// captured on real hardware; while empty (and without the env override) the CPU runs live.
    /// </summary>
    private static readonly int[] ExpectedCpuGreedyTokens = [];

    private readonly ITestOutputHelper _out;

    public CudaBonsai2RealCheckpointTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void GreedyDecode_MatchesCpu_OnRealBonsai2Checkpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();

        Run cuda;
        try
        {
            // The F16-activation path, whatever DOTLLM_CUDA_PQ2_0_DP4A says (the dp4a run is its own test).
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            cuda = RunCudaGreedy(path!, ptxDir);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = null;
        }
        _out.WriteLine($"cuda greedy:  {string.Join(",", cuda.Tokens)}");
        _out.WriteLine($"cuda top-2 gaps: {string.Join(",", cuda.Gaps.Select(g => g.ToString("E3")))}");

        int[] expected = ResolveCpuOracle(path!);
        Assert.Equal(expected, cuda.Tokens);
    }

    /// <summary>
    /// Issue #485: the same greedy run with the int8-activation dp4a PQ2_0 GEMV switched on (what
    /// <c>DOTLLM_CUDA_PQ2_0_DP4A=1</c> does), against the same CPU oracle. The CPU PQ2_0 GEMV on
    /// SSSE3/AVX2 hardware is itself the W2A8 tier with the identical activation quantization, so the
    /// dp4a path should track it at least as closely as the F16-activation path does. The 15-token
    /// prefill stays on dequant + cuBLAS; every decode step's PQ2_0 projections run dp4a.
    /// </summary>
    [SkippableFact]
    public void GreedyDecode_Dp4a_MatchesCpu_OnRealBonsai2Checkpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "pq2_0_gemv_dp4a.ptx")),
            "pq2_0_gemv_dp4a.ptx not generated (run native/build_ptx.bat on a CUDA box)");

        Run cuda;
        try
        {
            CudaSmallSGemvDispatch.Dp4aOverride = true;
            cuda = RunCudaGreedy(path!, ptxDir, requireDp4a: true);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = null;
        }
        _out.WriteLine($"cuda dp4a greedy:  {string.Join(",", cuda.Tokens)}");
        _out.WriteLine($"cuda dp4a top-2 gaps: {string.Join(",", cuda.Gaps.Select(g => g.ToString("E3")))}");

        int[] expected = ResolveCpuOracle(path!);
        Assert.Equal(expected, cuda.Tokens);
    }

    /// <summary>
    /// MTP draft tokens from the real checkpoint's head, which ships no head-local embedding or
    /// lm_head: passing requires the CUDA trunk-embedding inverse fold and the folded trunk-lm_head
    /// fallback. Runs the CPU draft live (prefill + 4 head steps — far cheaper than greedy decode).
    /// </summary>
    [SkippableFact]
    public void MtpDraftTokens_MatchCpu_OnRealBonsai2Checkpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();

        // Both chains start from the SAME token (the CPU prefill's argmax, handed to CUDA too), so a
        // near-tie in the trunk's last-row argmax cannot desynchronise the two draft chains.
        Run cpu;
        int first;
        using (var gguf = GgufFile.Open(path!))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.NotNull(config.HadamardFold);
            using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
            Assert.True(model.SupportsMtp);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim,
                PromptTokens.Length + DraftSteps + 2);
            using var mtp = model.CreateMtpState()!;
            using (ITensor prefill = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv, adapter: null, mtp))
                first = ArgMaxWithGap(prefill, config.VocabSize, lastRow: true).Token;
            cpu = Draft((t, p) => model.ForwardMtp(mtp, t, p), first, config.VocabSize);
        }

        Run gpu;
        using (var gguf = GgufFile.Open(path!))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            Assert.True(model.SupportsMtp);
            using var kv = model.CreateKvCache(PromptTokens.Length + DraftSteps + 2);
            using var mtp = model.CreateMtpState()!;
            using (ITensor prefill = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv, adapter: null, mtp))
            {
                int cudaFirst = ArgMaxWithGap(prefill, config.VocabSize, lastRow: true).Token;
                _out.WriteLine($"prefill argmax: cpu={first} cuda={cudaFirst}");
            }
            gpu = Draft((t, p) => model.ForwardMtp(mtp, t, p), first, config.VocabSize);
        }

        _out.WriteLine($"cpu  draft: {string.Join(",", cpu.Tokens)}  gaps {string.Join(",", cpu.Gaps.Select(g => g.ToString("E3")))}");
        _out.WriteLine($"cuda draft: {string.Join(",", gpu.Tokens)}  gaps {string.Join(",", gpu.Gaps.Select(g => g.ToString("E3")))}");
        Assert.Equal(cpu.Tokens, gpu.Tokens);
    }

    /// <summary>
    /// Issue #486: the device-argmax draft (<c>ForwardMtpArgMax</c>, what the decoder uses for an
    /// unconstrained greedy draft) must draft the same tokens on the real checkpoint as the host
    /// argmax of the full logits — including when two logits tie. One prefill; the head is rewound
    /// to the post-prefill state between the two chains.
    /// </summary>
    [SkippableFact]
    public void MtpDraftTokens_DeviceArgMax_MatchesFullLogits_OnRealBonsai2Checkpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Skip.IfNot(model.SupportsMtpArgMax, "argmax_f32.ptx not generated (nvcc -ptx -arch=compute_75 on a CUDA box)");
        int p = PromptTokens.Length;
        using var kv = model.CreateKvCache(p + DraftSteps + 2);
        using var mtp = model.CreateMtpState()!;
        int first;
        using (ITensor prefill = model.Forward(PromptTokens, Positions(p), -1, kv, adapter: null, mtp))
            first = ArgMaxWithGap(prefill, config.VocabSize, lastRow: true).Token;

        Run full = Draft((t, pos) => model.ForwardMtp(mtp, t, pos), first, config.VocabSize);

        mtp.Rollback(p);
        mtp.SeedFromCapturedRow(p - 1);   // the prefill absorb's own seed
        var device = new int[DraftSteps];
        int token = first;
        for (int i = 0; i < DraftSteps; i++)
            device[i] = token = model.ForwardMtpArgMax(mtp, token, p + i);

        _out.WriteLine($"full-logits draft:   {string.Join(",", full.Tokens)}  gaps {string.Join(",", full.Gaps.Select(g => g.ToString("E3")))}");
        _out.WriteLine($"device-argmax draft: {string.Join(",", device)}");
        Assert.Equal(full.Tokens, device);
    }

    /// <summary>
    /// Issue #482 perf probe (no perf assertion): wall-clock ms per MTP draft step on the real
    /// checkpoint, plus — with <c>DOTLLM_HYBRID_PROFILE=1</c> — the <c>mtp-*</c> category breakdown
    /// of <c>ForwardMtpCore</c> and the batched verify absorb. Each round drafts
    /// <see cref="DraftSteps"/> tokens from the same position (the head rolls its own KV back) and
    /// then runs one S = DraftSteps+1 verify forward with the MTP state, like the decoder does.
    /// </summary>
    [SkippableFact]
    public void MtpDraftCost_Profile_OnRealBonsai2Checkpoint()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();
        const int warmupRounds = 2;
        const int rounds = 12;

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsMtp);
        int p = PromptTokens.Length;
        using var kv = model.CreateKvCache(p + DraftSteps + 2);
        using var mtp = model.CreateMtpState()!;
        int first;
        using (ITensor prefill = model.Forward(PromptTokens, Positions(p), -1, kv, adapter: null, mtp))
            first = ArgMaxWithGap(prefill, config.VocabSize, lastRow: true).Token;

        var verifyTokens = new int[DraftSteps + 1];
        var verifyPositions = new int[DraftSteps + 1];
        long draftTicks = 0, verifyTicks = 0;
        int drafts = 0;
        for (int r = 0; r < warmupRounds + rounds; r++)
        {
            if (r == warmupRounds)
            {
                CudaQwen3HybridDenseTransformerModel.ProfileReportAndReset(); // drop prefill + warm-up
                draftTicks = verifyTicks = 0;
                drafts = 0;
            }

            int token = first;
            verifyTokens[0] = first;
            for (int i = 0; i < DraftSteps; i++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                using ITensor logits = model.ForwardMtp(mtp, token, p + i);
                draftTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
                drafts++;
                token = ArgMaxWithGap(logits, config.VocabSize, lastRow: true).Token;
                Assert.InRange(token, 0, config.VocabSize - 1);
                verifyTokens[i + 1] = token;
            }

            for (int i = 0; i <= DraftSteps; i++) verifyPositions[i] = p + i;
            // Deliberate for a perf probe: kv.Rollback below does not restore the GDN recurrent
            // state, so later rounds verify from an advanced state — same cost, values not meaningful.
            mtp.Rollback(p);
            long v0 = System.Diagnostics.Stopwatch.GetTimestamp();
            using (ITensor _ = model.Forward(verifyTokens, verifyPositions, -1, kv, adapter: null, mtp)) { }
            verifyTicks += System.Diagnostics.Stopwatch.GetTimestamp() - v0;
            kv.Rollback(p);
            mtp.Rollback(p);
            mtp.SeedFromCapturedRow(0);
        }

        double tickMs = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _out.WriteLine($"MTP draft: {draftTicks * tickMs / drafts:F3} ms/step over {drafts} steps " +
                       $"(Q8_0 GEMV: {(model.MtpUsesRbQ8Gemv ? "register-blocked" : model.MtpUsesStagedQ8Gemv ? "staged" : "original — PTX absent or disabled")}; " +
                       $"profiler {(Environment.GetEnvironmentVariable("DOTLLM_HYBRID_PROFILE") == "1" ? "ON — timings perturbed by per-mark syncs" : "off")})");
        _out.WriteLine($"verify S={DraftSteps + 1} (incl. batched absorb): {verifyTicks * tickMs / rounds:F3} ms/round");
        // Mirror the profiler's totals into the test output (ProfileReportAndReset writes stderr).
        foreach (var kvp in CudaQwen3HybridDenseTransformerModel.ProfileTotalsMs.OrderBy(e => e.Key, StringComparer.Ordinal))
        {
            int n = CudaQwen3HybridDenseTransformerModel.ProfileCounts.GetValueOrDefault(kvp.Key, 1);
            _out.WriteLine($"  {kvp.Key,-28} total={kvp.Value,9:F2}ms  calls={n,5}  avg={kvp.Value / n,8:F4}ms");
        }
        CudaQwen3HybridDenseTransformerModel.ProfileReportAndReset(); // no-op when the env var is unset

        // Issue #486: the same draft steps through the device argmax (what the decoder takes for an
        // unconstrained greedy draft) — no 248k-float logits D2H per step.
        if (!model.SupportsMtpArgMax)
        {
            _out.WriteLine("MTP draft (device argmax): skipped — argmax_f32.ptx absent");
            return;
        }
        long argTicks = 0;
        int argDrafts = 0;
        for (int r = 0; r < warmupRounds + rounds; r++)
        {
            if (r == warmupRounds)
            {
                CudaQwen3HybridDenseTransformerModel.ProfileReportAndReset();
                argTicks = 0;
                argDrafts = 0;
            }
            mtp.Rollback(p);
            mtp.SeedFromCapturedRow(0);
            int token = first;
            for (int i = 0; i < DraftSteps; i++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                token = model.ForwardMtpArgMax(mtp, token, p + i);
                argTicks += System.Diagnostics.Stopwatch.GetTimestamp() - t0;
                argDrafts++;
                Assert.InRange(token, 0, config.VocabSize - 1);
            }
        }
        _out.WriteLine($"MTP draft (device argmax): {argTicks * tickMs / argDrafts:F3} ms/step over {argDrafts} steps");
        foreach (var kvp in CudaQwen3HybridDenseTransformerModel.ProfileTotalsMs.OrderBy(e => e.Key, StringComparer.Ordinal))
        {
            int n = CudaQwen3HybridDenseTransformerModel.ProfileCounts.GetValueOrDefault(kvp.Key, 1);
            _out.WriteLine($"  {kvp.Key,-28} total={kvp.Value,9:F2}ms  calls={n,5}  avg={kvp.Value / n,8:F4}ms");
        }
        CudaQwen3HybridDenseTransformerModel.ProfileReportAndReset();
    }

    // ── runs ─────────────────────────────────────────────────────────────────

    private sealed record Run(int[] Tokens, float[] Gaps);

    private static Run RunCudaGreedy(string path, string ptxDir, bool requireDp4a = false)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.NotNull(config.HadamardFold); // the whole point: a folded checkpoint now loads on CUDA
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        if (requireDp4a)
            Assert.True(model.PQ2_0Dp4aAvailable, "pq2_0_gemv_dp4a.ptx present but the dp4a module did not load (stale PTX?)");
        using var kv = model.CreateKvCache(PromptTokens.Length + NewTokens + 1);
        return Greedy((toks, pos) => model.Forward(toks, pos, -1, kv), config.VocabSize);
    }

    private int[] ResolveCpuOracle(string path)
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_CPU_GREEDY_TOKENS");
        if (!string.IsNullOrWhiteSpace(env))
        {
            _out.WriteLine("cpu oracle: DOTLLM_BONSAI2_CPU_GREEDY_TOKENS");
            return env.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                      .Select(int.Parse).ToArray();
        }
        if (ExpectedCpuGreedyTokens.Length > 0)
        {
            _out.WriteLine("cpu oracle: hard-coded ExpectedCpuGreedyTokens");
            return ExpectedCpuGreedyTokens;
        }

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.Auto);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim,
            PromptTokens.Length + NewTokens + 1);
        var cpu = Greedy((toks, pos) => model.Forward(toks, pos, -1, kv), config.VocabSize);
        _out.WriteLine($"cpu greedy:   {string.Join(",", cpu.Tokens)}");
        _out.WriteLine($"cpu top-2 gaps:  {string.Join(",", cpu.Gaps.Select(g => g.ToString("E3")))}");
        return cpu.Tokens;
    }

    private static Run Greedy(Func<int[], int[], ITensor> forward, int vocab)
    {
        var tokens = new int[NewTokens];
        var gaps = new float[NewTokens];
        (int next, float gap) = (0, 0f);
        using (ITensor prefill = forward(PromptTokens, Positions(PromptTokens.Length)))
            (next, gap) = ArgMaxWithGap(prefill, vocab, lastRow: true);

        for (int i = 0; i < NewTokens; i++)
        {
            tokens[i] = next;
            gaps[i] = gap;
            if (i == NewTokens - 1) break;
            using ITensor step = forward([next], [PromptTokens.Length + i]);
            (next, gap) = ArgMaxWithGap(step, vocab, lastRow: true);
        }
        return new Run(tokens, gaps);
    }

    private static Run Draft(Func<int, int, ITensor> forwardMtp, int firstToken, int vocab)
    {
        var tokens = new int[DraftSteps];
        var gaps = new float[DraftSteps];
        int token = firstToken;
        for (int i = 0; i < DraftSteps; i++)
        {
            using ITensor logits = forwardMtp(token, PromptTokens.Length + i);
            (token, gaps[i]) = ArgMaxWithGap(logits, vocab, lastRow: true);
            tokens[i] = token;
        }
        return new Run(tokens, gaps);
    }

    private static unsafe (int Token, float Gap) ArgMaxWithGap(ITensor logits, int vocab, bool lastRow)
    {
        int rows = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab)
            .Slice(lastRow ? (rows - 1) * vocab : 0, vocab);
        int best = 0;
        float second = float.NegativeInfinity;
        for (int c = 1; c < vocab; c++)
        {
            if (span[c] > span[best]) { second = span[best]; best = c; }
            else if (span[c] > second) second = span[c];
        }
        return (best, span[best] - second);
    }

    private static int[] Positions(int n) => Enumerable.Range(0, n).ToArray();

    private static string? FindCheckpoint()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_BONSAI2_MTP_GGUF");
        if (!string.IsNullOrEmpty(env) && File.Exists(env)) return env;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string repo = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--ProCreations--Ternary-Bonsai-2-27B-MTP", "snapshots");
        if (!Directory.Exists(repo)) return null;

        foreach (string snapshot in Directory.EnumerateDirectories(repo))
        {
            string[] hits = Directory.GetFiles(snapshot, "*MTP*.gguf");
            if (hits.Length > 0) return hits[0];
        }
        return null;
    }

    private static string SkipUnlessCudaWithFwht()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");

        string? ptxDir = null;
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) { ptxDir = full; break; }
        }
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "hadamard_fwht.ptx")),
            "hadamard_fwht.ptx not generated (run native/build.ps1 on a CUDA box)");
        return ptxDir!;
    }
}
