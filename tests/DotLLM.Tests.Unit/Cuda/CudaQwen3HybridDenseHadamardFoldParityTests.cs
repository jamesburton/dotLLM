using System.Collections.Frozen;
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
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Model-level CUDA↔CPU parity for the PrismML Hadamard weight fold on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/> (issue #479).
/// </summary>
/// <remarks>
/// <para>
/// Both backends load the same synthetic <c>qwen35</c> fixture with the same injected fold
/// declaration (<see cref="SyntheticHadamardFold"/>): block 8, explicit signs, <c>gdn_v_grouped</c>,
/// and GDN geometry NKHead=2 / NVHead=4 so the <c>ssm_out</c> permute is a real reordering. The MTP
/// head ships no <c>nextn.embed_tokens</c> / <c>nextn.shared_head_head</c>, so it exercises the
/// trunk-embedding inverse fold and the folded trunk-lm_head fallback — the same shape as the real
/// Bonsai 2 MTP checkpoint.
/// </para>
/// <para>
/// <b>What a mutant looks like.</b> <c>Qwen3HybridDenseSyntheticHadamardFoldTests</c> pins, on the
/// CPU, that every transform component moves the logits by more than ~8× this test's tolerance
/// (fold ≈ 0.17, signs ≈ 0.13, permute ≈ 0.024, MTP ≈ 0.53 max |Δ|). So any of these CUDA mutants
/// fails here: dropping the rotation at one site (e.g. feeding <c>normOut</c> instead of the rotated
/// scratch to <c>attn_gate</c>, or the rotated scratch to the unfolded <c>ssm_alpha</c>), passing
/// <c>permuteGdnValueHeads: false</c> at <c>ssm_out</c>, skipping the host inverse fold on the
/// embedding rows or applying it in forward order (signs before rotation), or omitting the lm_head /
/// MTP-head rotation.
/// </para>
/// <para>
/// Tolerances match <c>HybridQwen3HybridDenseTransformerModelSplitParityTests</c>: the fixture is
/// all-F32, so the only CPU↔CUDA difference is GEMM reduction order (the FWHT itself is bit-exact).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQwen3HybridDenseHadamardFoldParityTests : IDisposable
{
    private const float AbsTol = 1.5e-3f;
    private const float RelTol = 5e-3f;

    private static readonly int[] Prompt = [1, 3, 5, 7, 9];

    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseHadamardFoldParityTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-qwen35-fold-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <summary>
    /// Prefill (every row), then three single-token decode steps through the KV cache: the
    /// multi-row and single-row paths of every rotation site, the GDN recurrence carrying the
    /// permuted <c>ssm_out</c> input forward in time, and the host-side embedding inverse fold.
    /// </summary>
    [SkippableFact]
    public void Forward_PrefillAndDecode_WithFold_MatchesCpu()
    {
        string ptxDir = SkipUnlessCudaWithFwht();
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold.gguf"));
        int[] decodeTokens = [4, 6, 8];

        List<float[]> cpu = RunCpu(path, decodeTokens);
        List<float[]> gpu = RunCuda(path, ptxDir, decodeTokens);

        Assert.Equal(cpu.Count, gpu.Count);
        for (int i = 0; i < cpu.Count; i++)
            AssertLogitsMatch(cpu[i], gpu[i], i == 0 ? "prefill (all rows)" : $"decode step {i}");
    }

    /// <summary>
    /// <c>lastTokenLogitsOnly</c> rotates only the last row, from an offset into the hidden state —
    /// the lm_head rotation must use that row, not row 0.
    /// </summary>
    [SkippableFact]
    public void Forward_LastTokenLogitsOnly_WithFold_MatchesCpuLastRow()
    {
        string ptxDir = SkipUnlessCudaWithFwht();
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold-last.gguf"));

        float[] cpuAll = RunCpu(path, [])[0];
        int vocab = cpuAll.Length / Prompt.Length;
        float[] cpuLast = cpuAll.AsSpan((Prompt.Length - 1) * vocab, vocab).ToArray();

        using var gguf = GgufFile.Open(path);
        var config = WithFold(GgufModelConfigExtractor.Extract(gguf.Metadata));
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);
        using ITensor logits = model.Forward(Prompt, Positions(Prompt.Length), deviceId: -1, kv, lastTokenLogitsOnly: true);
        Assert.Equal(1, logits.Shape[0]);

        AssertLogitsMatch(cpuLast, ToArray(logits, vocab), "lastTokenLogitsOnly");
    }

    /// <summary>
    /// The MTP head's two fold sites: the trunk <c>token_embd</c> fallback (inverse fold on the host
    /// row) and the trunk <c>output.weight</c> fallback (forward rotation of <c>NormedHead</c>).
    /// Two chained draft steps, so the second one also checks that the pending hidden state handed
    /// to the next step stays UNROTATED.
    /// </summary>
    [SkippableFact]
    public void ForwardMtp_WithFold_TrunkFallbacks_MatchCpu()
    {
        string ptxDir = SkipUnlessCudaWithFwht();
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold-mtp.gguf"));

        float[][] cpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = WithFold(GgufModelConfigExtractor.Extract(gguf.Metadata));
            using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
            using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);
            using var mtp = model.CreateMtpState()!;
            using (ITensor _ = model.Forward(Prompt, Positions(Prompt.Length), deviceId: -1, kv, adapter: null, mtp)) { }
            cpu = Draft((tok, pos) => model.ForwardMtp(mtp, tok, pos), config.VocabSize);
        }

        float[][] gpu;
        using (var gguf = GgufFile.Open(path))
        {
            var config = WithFold(GgufModelConfigExtractor.Extract(gguf.Metadata));
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            Assert.True(model.SupportsMtp);
            using var kv = model.CreateKvCache(config.MaxSequenceLength);
            using var mtp = model.CreateMtpState()!;
            using (ITensor _ = model.Forward(Prompt, Positions(Prompt.Length), deviceId: -1, kv, adapter: null, mtp)) { }
            gpu = Draft((tok, pos) => model.ForwardMtp(mtp, tok, pos), config.VocabSize);
        }

        for (int i = 0; i < cpu.Length; i++)
            AssertLogitsMatch(cpu[i], gpu[i], $"MTP draft step {i}");
    }

    /// <summary>
    /// The refusal survives for declarations the CUDA sites do not cover: a checkpoint that folds a
    /// weight we never rotate (here the unfolded-in-Bonsai-2 <c>ssm_alpha</c>) must not load.
    /// </summary>
    [SkippableFact]
    public void LoadFromGguf_FoldSetMismatch_Refuses()
    {
        string ptxDir = SkipUnlessCudaWithFwht();
        string path = SyntheticHadamardFold.WriteFixture(Path.Combine(_scratch, "fold-bad.gguf"));

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var fold = SyntheticHadamardFold.For(config);
        var badFold = fold with
        {
            FoldedWeights = fold.FoldedWeights.Append("blk.0.ssm_alpha.weight").ToFrozenSet(StringComparer.Ordinal),
        };

        var ex = Assert.Throws<NotSupportedException>(() =>
            CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config with { HadamardFold = badFold }, deviceId: 0, ptxDir));
        Assert.Contains("ssm_alpha", ex.Message, StringComparison.Ordinal);
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    private static ModelConfig WithFold(ModelConfig config) =>
        config with { HadamardFold = SyntheticHadamardFold.For(config) };

    private static int[] Positions(int n) => Enumerable.Range(0, n).ToArray();

    private static List<float[]> RunCpu(string path, int[] decodeTokens)
    {
        using var gguf = GgufFile.Open(path);
        var config = WithFold(GgufModelConfigExtractor.Extract(gguf.Metadata));
        using var model = Qwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        using var kv = new SimpleKvCache(model.AttentionLayerCount, config.NumKvHeads, config.HeadDim, config.MaxSequenceLength);

        var results = new List<float[]>();
        using (ITensor prefill = model.Forward(Prompt, Positions(Prompt.Length), deviceId: -1, kv))
            results.Add(ToArray(prefill, Prompt.Length * config.VocabSize));
        for (int i = 0; i < decodeTokens.Length; i++)
        {
            using ITensor step = model.Forward([decodeTokens[i]], [Prompt.Length + i], deviceId: -1, kv);
            results.Add(ToArray(step, config.VocabSize));
        }
        return results;
    }

    private static List<float[]> RunCuda(string path, string ptxDir, int[] decodeTokens)
    {
        using var gguf = GgufFile.Open(path);
        var config = WithFold(GgufModelConfigExtractor.Extract(gguf.Metadata));
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kv = model.CreateKvCache(config.MaxSequenceLength);

        var results = new List<float[]>();
        using (ITensor prefill = model.Forward(Prompt, Positions(Prompt.Length), deviceId: -1, kv))
            results.Add(ToArray(prefill, Prompt.Length * config.VocabSize));
        for (int i = 0; i < decodeTokens.Length; i++)
        {
            using ITensor step = model.Forward([decodeTokens[i]], [Prompt.Length + i], deviceId: -1, kv);
            results.Add(ToArray(step, config.VocabSize));
        }
        return results;
    }

    /// <summary>
    /// Two chained draft steps starting at the position after the prompt. Both backends take the
    /// SAME token sequence (the CPU-independent constant, then 2), so a divergent argmax on one side
    /// cannot desynchronise the comparison.
    /// </summary>
    private static float[][] Draft(Func<int, int, ITensor> forwardMtp, int vocab)
    {
        int[] tokens = [4, 2];
        var outp = new float[tokens.Length][];
        for (int i = 0; i < tokens.Length; i++)
        {
            using ITensor logits = forwardMtp(tokens[i], Prompt.Length + i);
            outp[i] = ToArray(logits, vocab);
        }
        return outp;
    }

    private static float[] ToArray(ITensor t, int count) =>
        new ReadOnlySpan<float>((void*)t.DataPointer, count).ToArray();

    private void AssertLogitsMatch(float[] cpu, float[] gpu, string what)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        float worst = 0;
        int worstIdx = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(float.IsFinite(gpu[i]), $"{what}: non-finite CUDA logit at {i}");
            float diff = MathF.Abs(cpu[i] - gpu[i]);
            if (diff > worst) { worst = diff; worstIdx = i; }
        }
        _out.WriteLine($"{what}: max |cpu-cuda| = {worst:E3} at {worstIdx} (cpu={cpu[worstIdx]}, cuda={gpu[worstIdx]})");

        for (int i = 0; i < cpu.Length; i++)
        {
            float tol = AbsTol + RelTol * MathF.Abs(cpu[i]);
            Assert.True(MathF.Abs(cpu[i] - gpu[i]) <= tol,
                $"{what}: logit[{i}] cpu={cpu[i]} cuda={gpu[i]} |diff|={MathF.Abs(cpu[i] - gpu[i]):E3} > tol={tol:E3}");
        }
    }

    private static string SkipUnlessCudaWithFwht()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "hadamard_fwht.ptx")),
            "hadamard_fwht.ptx not generated (run native/build.ps1 on a CUDA box)");
        return ptxDir!;
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}
