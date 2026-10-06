using System.Runtime.InteropServices;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #493 on the MTP trunk: <c>lastTokenLogitsOnly</c> now composes with an MTP capture on
/// <see cref="CudaQwen3HybridDenseTransformerModel"/>, because the final RMSNorm's row count is
/// decoupled from the LM head's. The head consumes llama.cpp's <c>h_nextn</c> — the post-<c>output_norm</c>
/// hidden state for <b>every</b> position — while the LM head needs only the last one, so an MTP
/// prefill can skip the <c>[S, vocab]</c> projection and still absorb correctly.
/// </summary>
/// <remarks>
/// <para><b>Why this test and not a generation-parity test.</b> Mutate <c>normRows</c> back to
/// <c>logitsRows</c> and the capture receives UNNORMALISED rows for positions 0..S-2 and a
/// normalised row S-1. <c>SeedFromCapturedRow(S-1)</c> is still right, so the first draft is fine;
/// only the head's KV for the prefix is wrong, which lowers the acceptance rate. Speculative
/// decoding guarantees the output tokens regardless, so every greedy-parity test still passes. The
/// captured rows are the only oracle that sees it.</para>
/// <para>Unlike the logits comparison, the captured rows must be <b>bit-identical</b>: both arms run
/// the same RmsNorm kernel over the same S rows of the same buffer and D2H the same bytes. Only the
/// LM head downstream differs. A tolerance here would hide the mutant.</para>
/// <para>Fresh model instances per arm, sequentially: the GatedDeltaNet state is owned by the model,
/// not by the <c>IKvCache</c>, and Bonsai 2's weights do not fit twice on a 12 GB card.</para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "RealModel")]
[Collection(CudaCollection.Name)]
public sealed class CudaBonsai2MtpLastRowLogitsTests
{
    private static readonly int[] PromptTokens =
        [7734, 264, 2716, 10597, 15673, 314, 1204, 264, 4779, 42209, 311, 4623, 26642, 9714, 13];

    private readonly ITestOutputHelper _out;
    public CudaBonsai2MtpLastRowLogitsTests(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void MtpPrefill_LastRowLogitsOnly_CapturesTheSameRowsAsTheAllRowCall()
    {
        string? path = FindCheckpoint();
        Skip.If(path is null,
            "Bonsai 2 MTP checkpoint not found (set DOTLLM_BONSAI2_MTP_GGUF or populate the HF hub cache).");
        string ptxDir = SkipUnlessCudaWithFwht();

        int s = PromptTokens.Length;
        int vocab;

        _out.WriteLine("Arm A: lastTokenLogitsOnly=false (all rows)...");
        (float[] capturedAll, float[] lastRowAll, vocab) = RunPrefill(path!, ptxDir, lastRowOnly: false, expectRows: s);

        _out.WriteLine("Arm B: lastTokenLogitsOnly=true (one row)...");
        (float[] capturedHinted, float[] lastRowHinted, _) = RunPrefill(path!, ptxDir, lastRowOnly: true, expectRows: 1);

        // The capture is a side effect on the MTP state and must not change at all.
        Assert.Equal(capturedAll.Length, capturedHinted.Length);
        Assert.Equal(0, capturedAll.Length % s);     // S whole rows, not one
        int hidden = capturedAll.Length / s;
        for (int i = 0; i < capturedAll.Length; i++)
        {
            Assert.True(capturedAll[i].Equals(capturedHinted[i]),
                $"captured h_nextn row element {i} (row {i / hidden} of {s}): "
                + $"all-rows={capturedAll[i]}, lastRowOnly={capturedHinted[i]} — the MTP capture must be "
                + "bit-identical, so the final norm must still cover every row when only the last "
                + "row's logits are wanted (issue #493).");
        }

        // The logits themselves take different LM-head routes (GEMV vs dequant + cuBLAS F16), the
        // same benign drift CudaQwen3HybridDenseLastTokenLogitsOnlyTest documents.
        const float AbsTol = 1e-4f;
        const float RelTol = 1e-3f;
        int compare = Math.Min(64, vocab);
        for (int i = 0; i < compare; i++)
        {
            float a = lastRowAll[i];
            float diff = MathF.Abs(a - lastRowHinted[i]);
            Assert.True(diff <= AbsTol + RelTol * MathF.Abs(a),
                $"logits[{i}]: all-rows last row={a}, lastRowOnly={lastRowHinted[i]}, diff={diff}.");
        }

        _out.WriteLine($"{capturedAll.Length} captured floats bit-identical; {compare} logits within tolerance.");
    }

    private static (float[] Captured, float[] LastRow, int Vocab) RunPrefill(
        string path, string ptxDir, bool lastRowOnly, int expectRows)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsMtp);

        using var kv = model.CreateKvCache(PromptTokens.Length + 8);
        using var mtp = (CudaMtpState)model.CreateMtpState()!;
        int[] positions = [.. Enumerable.Range(0, PromptTokens.Length)];

        using ITensor logits = model.Forward(PromptTokens, positions, deviceId: -1, kv,
                                             adapter: null, mtp, lastRowOnly);
        Assert.Equal(expectRows, logits.Shape[0]);
        Assert.Equal(PromptTokens.Length, mtp.CapturedRowCount);

        float[] captured = mtp.CapturedHiddenRows.ToArray();
        return (captured, ExtractRow(logits, logits.Shape[0] - 1, Math.Min(64, config.VocabSize)), config.VocabSize);
    }

    private static unsafe float[] ExtractRow(ITensor logits, int row, int sliceLen)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        float* basePtr = (float*)logits.DataPointer + (long)row * vocab;
        var slice = new float[sliceLen];
        for (int i = 0; i < sliceLen; i++) slice[i] = basePtr[i];
        return slice;
    }

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

        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "fwht.ptx")),
            "fwht.ptx not generated (run native/build_ptx.bat on a CUDA box)");
        return ptxDir!;
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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
