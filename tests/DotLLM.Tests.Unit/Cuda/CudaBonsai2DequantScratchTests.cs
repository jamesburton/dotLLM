using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #495: the F16 weight-dequant scratch is allocated on first actual use at the size that use
/// needs, not at model load sized to the widest tile in the model.
/// </summary>
/// <remarks>
/// <para>
/// <b>Asserts on the model's own accounting, not on nvidia-smi.</b>
/// <see cref="CudaQwen3HybridDenseTransformerModel.DequantScratchF16WeightBytes"/> reports exactly
/// what <c>EnsureDequantScratchF16Weight</c> has handed to <c>cuMemAlloc</c>, so the assertions are
/// deterministic and independent of driver bookkeeping, WDDM paging, fragmentation, or whatever else
/// happens to be resident on the card. A process- or driver-level reading could not distinguish this
/// buffer from the ~7.6 GB of weights next to it.
/// </para>
/// <para>
/// <b>The "before" number is arithmetic, not a measurement.</b> The old policy was
/// <c>maxTileFloats * sizeof(ushort)</c> with <c>maxTileFloats</c> including the lm_head tile, i.e.
/// exactly <c>vocab * hidden * 2</c> bytes, unconditionally, at load. On Bonsai 2 27B that is
/// 248320 x 5120 x 2 = 2.54 GiB. Each test prints it next to what the new policy actually allocated.
/// </para>
/// <para>
/// <b>Gated</b> on <c>DOTLLM_BONSAI2_MTP_GGUF</c> / the HF hub cache and on a CUDA device with the
/// FWHT PTX, exactly like <see cref="CudaBonsai2RealCheckpointTests"/>; skips cleanly otherwise.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaBonsai2DequantScratchTests
{
    private static readonly int[] PromptTokens =
        [7734, 264, 2716, 10597, 15673, 314, 1204, 264, 4779, 42209, 311, 4623, 26642, 9714, 13];

    private readonly ITestOutputHelper _out;

    public CudaBonsai2DequantScratchTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// The whole point of #495: nothing is allocated at load, a generation-shaped prefill
    /// (last-token logits only — what every serving/bench path asks for) never grows the buffer past
    /// the widest projection it actually dequantizes, and the lm_head tile only appears if a caller
    /// really does project every row.
    /// </summary>
    [SkippableFact]
    public void DequantScratch_IsZeroAtLoad_AndGrowsOnlyToWhatTheActivePathNeeds()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null, "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF).");
        string ptxDir = SkipUnlessCudaWithFwht();

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        long lmHeadBytes = (long)config.VocabSize * config.HiddenSize * sizeof(ushort);
        long ffnBytes = (long)config.IntermediateSize * config.HiddenSize * sizeof(ushort);
        _out.WriteLine($"old policy (load-time, unconditional): {Mib(lmHeadBytes)}  (lm_head {config.VocabSize} x {config.HiddenSize})");
        _out.WriteLine($"widest non-lm_head tile:               {Mib(ffnBytes)}  (ffn {config.IntermediateSize} x {config.HiddenSize})");

        bool? dp4a = CudaSmallSGemvDispatch.Dp4aOverride;
        bool? mmq = CudaSmallSGemvDispatch.MmqOverride;
        try
        {
            // Both packed prefill switches OFF, so the dequant+cuBLAS fallback is the path under test.
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            CudaSmallSGemvDispatch.MmqOverride = false;

            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

            Assert.Equal(0L, model.DequantScratchF16WeightBytes);
            _out.WriteLine($"after load:                            0 B  (was {Mib(lmHeadBytes)})");

            // (a) Generation-shaped prefill: the lm_head runs at one row, so it takes the F32-native
            //     PQ2_0 GEMV and never touches the scratch. Only the per-layer projections fall back.
            using (var kv = model.CreateKvCache(PromptTokens.Length + 2))
            using (ITensor _ = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv,
                                             lastTokenLogitsOnly: true))
            {
            }

            long afterGenPrefill = model.DequantScratchF16WeightBytes;
            _out.WriteLine($"after last-token-only prefill:         {Mib(afterGenPrefill)}");
            Assert.True(afterGenPrefill > 0,
                "the dequant+cuBLAS fallback ran, so the buffer must have appeared on demand");
            Assert.True(afterGenPrefill <= ffnBytes,
                $"expected at most the widest non-lm_head tile ({Mib(ffnBytes)}), got {Mib(afterGenPrefill)}");
            Assert.True(afterGenPrefill < lmHeadBytes,
                $"expected less than the old load-time size ({Mib(lmHeadBytes)}), got {Mib(afterGenPrefill)}");

            // (b) All-rows prefill (what perplexity's whole-window forward does): the lm_head really
            //     is dequantized, so the buffer MUST grow to it. This is the correctness half of the
            //     change — lazy must not mean absent.
            using (var kv = model.CreateKvCache(PromptTokens.Length + 2))
            using (ITensor _ = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv,
                                             lastTokenLogitsOnly: false))
            {
            }

            long afterFullPrefill = model.DequantScratchF16WeightBytes;
            _out.WriteLine($"after all-rows prefill:                {Mib(afterFullPrefill)}");
            Assert.Equal(lmHeadBytes, afterFullPrefill);

            // (c) Grow-only: a narrower projection after a wide one must not reallocate downwards.
            using (var kv = model.CreateKvCache(PromptTokens.Length + 2))
            using (ITensor _ = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv,
                                             lastTokenLogitsOnly: true))
            {
            }

            Assert.Equal(lmHeadBytes, model.DequantScratchF16WeightBytes);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = dp4a;
            CudaSmallSGemvDispatch.MmqOverride = mmq;
        }
    }

    /// <summary>
    /// With the packed PQ2_0 paths covering every projection (#485 at 1..8 rows, #490 above), a
    /// folded PQ2_0 checkpoint never dequantizes a weight tile at all — so the buffer that used to
    /// cost 2.54 GiB at load is never allocated, for the whole life of the model.
    /// </summary>
    [SkippableFact]
    public void DequantScratch_StaysZero_WhenThePackedPrefillCoversEveryProjection()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null, "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF).");
        string ptxDir = SkipUnlessCudaWithFwht();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "pq2_0_mmq_dp4a.ptx"))
                   && File.Exists(Path.Combine(ptxDir, "pq2_0_gemv_dp4a.ptx")),
            "pq2_0_mmq_dp4a.ptx / pq2_0_gemv_dp4a.ptx not generated (run native/build_ptx.bat on a CUDA box)");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        long lmHeadBytes = (long)config.VocabSize * config.HiddenSize * sizeof(ushort);

        bool? dp4a = CudaSmallSGemvDispatch.Dp4aOverride;
        bool? mmq = CudaSmallSGemvDispatch.MmqOverride;
        try
        {
            CudaSmallSGemvDispatch.Dp4aOverride = true;
            CudaSmallSGemvDispatch.MmqOverride = true;

            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
            Assert.Equal(0L, model.DequantScratchF16WeightBytes);

            using (var kv = model.CreateKvCache(PromptTokens.Length + 2))
            using (ITensor _ = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv,
                                             lastTokenLogitsOnly: false))
            {
            }

            _out.WriteLine($"all-rows prefill with the packed PQ2_0 GEMM: {model.DequantScratchF16WeightBytes} B "
                           + $"(old policy: {Mib(lmHeadBytes)} at load)");
            Assert.Equal(0L, model.DequantScratchF16WeightBytes);
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = dp4a;
            CudaSmallSGemvDispatch.MmqOverride = mmq;
        }
    }

    /// <summary>
    /// #495 is allocation policy, not math. The same all-rows prefill must produce bit-identical
    /// logits whether the scratch is allocated lazily part-way through the forward pass (a fresh
    /// model, the new behaviour) or already sized to the lm_head tile before the pass starts (a
    /// model that has run one prefill — which is what the old load-time allocation always gave).
    /// </summary>
    [SkippableFact]
    public void Logits_AreBitIdentical_WhetherTheScratchWasPreSizedOrGrownMidPass()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null, "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF).");
        string ptxDir = SkipUnlessCudaWithFwht();

        bool? dp4a = CudaSmallSGemvDispatch.Dp4aOverride;
        bool? mmq = CudaSmallSGemvDispatch.MmqOverride;
        try
        {
            CudaSmallSGemvDispatch.Dp4aOverride = false;
            CudaSmallSGemvDispatch.MmqOverride = false;

            float[] lazy = PrefillLogits(path!, ptxDir, preSize: false, out long lazyBytesBefore);
            float[] preSized = PrefillLogits(path!, ptxDir, preSize: true, out long preSizedBytesBefore);

            _out.WriteLine($"scratch held entering the measured pass: lazy={lazyBytesBefore} B, "
                           + $"pre-sized={Mib(preSizedBytesBefore)}");
            Assert.Equal(0L, lazyBytesBefore);
            Assert.True(preSizedBytesBefore > 0, "the pre-sizing pass should have allocated the buffer");

            Assert.Equal(lazy.Length, preSized.Length);
            for (int i = 0; i < lazy.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(lazy[i]) != BitConverter.SingleToInt32Bits(preSized[i]))
                    Assert.Fail($"logit[{i}] differs: lazy={lazy[i]:R} pre-sized={preSized[i]:R}");
            }
        }
        finally
        {
            CudaSmallSGemvDispatch.Dp4aOverride = dp4a;
            CudaSmallSGemvDispatch.MmqOverride = mmq;
        }
    }

    /// <summary>
    /// Runs the measured all-rows prefill on a fresh model and copies its logits out. With
    /// <paramref name="preSize"/> the model first runs an identical prefill on its own KV cache, so
    /// the dequant scratch is already at its final size when the measured pass starts.
    /// </summary>
    private static unsafe float[] PrefillLogits(string path, string ptxDir, bool preSize, out long bytesBefore)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        if (preSize)
        {
            using var warm = model.CreateKvCache(PromptTokens.Length + 2);
            using ITensor _ = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, warm,
                                            lastTokenLogitsOnly: false);
        }

        bytesBefore = model.DequantScratchF16WeightBytes;

        using var kv = model.CreateKvCache(PromptTokens.Length + 2);
        using ITensor logits = model.Forward(PromptTokens, Positions(PromptTokens.Length), -1, kv,
                                             lastTokenLogitsOnly: false);
        int n = logits.Shape[0] * config.VocabSize;
        var copy = new float[n];
        new ReadOnlySpan<float>((void*)logits.DataPointer, n).CopyTo(copy);
        return copy;
    }

    private static string Mib(long bytes) => $"{bytes / (1024.0 * 1024.0):F1} MiB";

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
