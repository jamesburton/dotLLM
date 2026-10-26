using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// The correctness gate for PTQ1_0 (issue #436): a <b>bit-exact</b> comparison of
/// <see cref="Dequantize.ToFloat32"/> against the PrismML fork's own
/// <c>dequantize_row_ptq1_0</c>, fed with real block bytes lifted out of
/// <c>Ternary-Bonsai-2-27B-PTQ1_0.gguf</c>.
///
/// <para>Perplexity is deliberately <b>not</b> the gate here. A transposed Q3_K dequant shipped
/// undetected across every backend for months because the only check applied to it was coarse
/// enough to absorb it. A sign error, a wrong trit position or a misplaced group scale all move
/// PPL by less than its own error bar; none of them survive a raw-bit comparison.</para>
///
/// <para>Gated on two local fixtures, and skipped (not failed) when either is absent:</para>
/// <list type="bullet">
/// <item>the fork's built <c>ggml-base.dll</c>, which exports the reference routine —
/// <c>DOTLLM_PRISM_GGML_DIR</c>, else <c>C:\Development\prism-llama.cpp\build\bin\Release</c>;</item>
/// <item>the GGUF itself — <c>DOTLLM_BONSAI_PTQ1_0_GGUF</c>, else the HF hub cache, else
/// <c>~/.dotllm/{models,test-cache}/PrismML/Ternary-Bonsai-2-27B-GGUF/</c>.</item>
/// </list>
/// </summary>
public sealed class PTQ1_0ForkOracleTests
{
    private const string GgmlDirEnvVar = "DOTLLM_PRISM_GGML_DIR";
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PTQ1_0_GGUF";
    private const string FileName = "Ternary-Bonsai-2-27B-PTQ1_0.gguf";
    private const string DefaultGgmlDir = @"C:\Development\prism-llama.cpp\build\bin\Release";

    /// <summary>
    /// <c>void dequantize_row_ptq1_0(const block_ptq1_0 * x, float * y, int64_t k)</c> as exported
    /// from the fork's <c>ggml-base</c>. Bound by delegate rather than <c>[LibraryImport]</c> so the
    /// DLL is located at runtime from a configurable directory instead of the probing path.
    /// </summary>
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void DequantizeRowPtq1_0(byte* x, float* y, long k);

    /// <summary>
    /// The gate. Takes the first 80 blocks (10 240 weights, two full K=5120 rows) of a real
    /// PTQ1_0 projection and requires every one of the 10 240 decoded floats to match the fork's
    /// output in <b>all 32 bits</b>.
    /// </summary>
    [SkippableFact]
    public unsafe void RealTensorBlocks_DequantizeBitIdenticallyToTheForkReference()
    {
        string? ggmlDir = ResolveGgmlDir();
        Skip.If(ggmlDir is null,
            $"PrismML ggml-base.dll not found. Set {GgmlDirEnvVar} to the directory holding the fork's built DLLs.");

        string? modelPath = ResolveFixturePath();
        Skip.If(modelPath is null,
            $"Bonsai 2 PTQ1_0 GGUF fixture not found. Set {ModelPathEnvVar} to the path of {FileName}.");

        const string TensorName = "blk.0.attn_qkv.weight";
        const int BlockCount = 80;
        const int BlockBytes = QuantFormat.PTQ1_0BlockBytes;
        const int GroupSize = QuantFormat.TernaryGroupSize;
        const int ElementCount = BlockCount * GroupSize;

        using var gguf = GgufFile.Open(modelPath!);
        var tensor = gguf.Tensors.Single(t => t.Name == TensorName);
        Assert.Equal(QuantizationType.PTQ1_0, tensor.QuantizationType);

        // Copy the packed bytes out of the mapping so the reference call and ours see identical input.
        var packed = new byte[BlockCount * BlockBytes];
        new ReadOnlySpan<byte>((void*)(gguf.DataBasePointer + (nint)tensor.DataOffset), packed.Length)
            .CopyTo(packed);

        var ours = new float[ElementCount];
        fixed (byte* packedPtr = packed)
        {
            Dequantize.ToFloat32((nint)packedPtr, ElementCount, QuantizationType.PTQ1_0, ours);
        }

        var reference = new float[ElementCount];
        nint handle = NativeLibrary.Load(Path.Combine(ggmlDir!, "ggml-base.dll"));
        try
        {
            nint symbol = NativeLibrary.GetExport(handle, "dequantize_row_ptq1_0");
            var fn = Marshal.GetDelegateForFunctionPointer<DequantizeRowPtq1_0>(symbol);
            fixed (byte* packedPtr = packed)
            fixed (float* outPtr = reference)
            {
                fn(packedPtr, outPtr, ElementCount);
            }
        }
        finally
        {
            NativeLibrary.Free(handle);
        }

        // Bit-exact, not approximate: compare the raw IEEE-754 patterns.
        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < ElementCount; i++)
        {
            if (BitConverter.SingleToUInt32Bits(ours[i]) != BitConverter.SingleToUInt32Bits(reference[i]))
            {
                if (firstMismatch < 0) firstMismatch = i;
                mismatches++;
            }
        }

        Assert.True(mismatches == 0,
            $"'{TensorName}': {mismatches}/{ElementCount} decoded weights differ from the fork's "
            + $"dequantize_row_ptq1_0. First at element {firstMismatch}: ours="
            + (firstMismatch >= 0 ? ours[firstMismatch].ToString("R") : "-") + ", fork="
            + (firstMismatch >= 0 ? reference[firstMismatch].ToString("R") : "-") + ".");

        // A guard against the degenerate pass: an all-zero decode would match trivially.
        Assert.Contains(reference, v => v != 0f);
    }

    /// <summary>
    /// Cross-pack check: the SAME checkpoint is published as both PQ2_0 and PTQ1_0, and both are
    /// ternary at group 128 — PTQ1_0's whole claim is that it is lossless for exactly such a
    /// checkpoint. So the two packs must decode to the same weights, and a PTQ1_0 decode that is
    /// self-consistently wrong in some way the fork's own routine happens to share would still
    /// show up here as decorrelation against an independently-implemented decoder.
    ///
    /// <para>Three tensors, chosen to be structurally different rather than convenient:
    /// <c>attn_qkv</c> (5120-wide), <c>ssm_out</c> (6144-wide — the <c>gdn_v_grouped</c> fold
    /// site), and <c>ffn_down</c> (17408-wide, many groups per row).</para>
    /// </summary>
    [SkippableFact]
    public unsafe void DecodesToTheSameWeightsAsThePQ2_0PackOfTheSameCheckpoint()
    {
        string? ptq1Path = ResolveFixturePath();
        Skip.If(ptq1Path is null, $"Set {ModelPathEnvVar} to the path of {FileName}.");

        string? pq2Path = ResolvePq2_0FixturePath();
        Skip.If(pq2Path is null,
            $"PQ2_0 sibling pack not found. Set {Pq2ModelPathEnvVar} to the path of {Pq2FileName}.");

        using var ptq1 = GgufFile.Open(ptq1Path!);
        using var pq2 = GgufFile.Open(pq2Path!);

        string[] names = ["blk.0.attn_qkv.weight", "blk.0.ssm_out.weight", "blk.0.ffn_down.weight"];
        foreach (string name in names)
        {
            var a = ptq1.Tensors.Single(t => t.Name == name);
            var b = pq2.Tensors.Single(t => t.Name == name);
            Assert.Equal(QuantizationType.PTQ1_0, a.QuantizationType);
            Assert.Equal(QuantizationType.PQ2_0, b.QuantizationType);
            Assert.Equal(a.Shape.ElementCount, b.Shape.ElementCount);

            // A few rows is plenty and keeps the test off the multi-GB end of these files.
            int rowLen = (int)a.Shape.Dimensions[0];
            const int Rows = 4;
            int n = rowLen * Rows;

            var fromPtq1 = new float[n];
            var fromPq2 = new float[n];
            Dequantize.ToFloat32(ptq1.DataBasePointer + (nint)a.DataOffset, n, QuantizationType.PTQ1_0, fromPtq1);
            Dequantize.ToFloat32(pq2.DataBasePointer + (nint)b.DataOffset, n, QuantizationType.PQ2_0, fromPq2);

            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
            int signMismatches = 0;
            for (int i = 0; i < n; i++)
            {
                double x = fromPtq1[i], y = fromPq2[i];
                sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
                if (Math.Sign(x) != Math.Sign(y)) signMismatches++;
            }

            double cov = sxy / n - (sx / n) * (sy / n);
            double corr = cov / (Math.Sqrt(sxx / n - (sx / n) * (sx / n)) * Math.Sqrt(syy / n - (sy / n) * (sy / n)));

            Assert.True(corr > 0.999,
                $"'{name}': PTQ1_0 vs PQ2_0 correlation {corr:F6} over {n} weights — the two packs of the "
                + "same checkpoint should agree. A low value means one of the two decoders is wrong.");
            Assert.True(signMismatches == 0,
                $"'{name}': {signMismatches}/{n} weights differ in SIGN between the PTQ1_0 and PQ2_0 packs.");
        }
    }

    private const string Pq2ModelPathEnvVar = "DOTLLM_BONSAI_2_PQ2_0_GGUF";
    private const string Pq2FileName = "Ternary-Bonsai-2-27B-PQ2_0.gguf";

    private static string? ResolvePq2_0FixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(Pq2ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string hub = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--prism-ml--Ternary-Bonsai-2-27B-gguf", "snapshots");
        if (Directory.Exists(hub))
            foreach (string snapshot in Directory.EnumerateDirectories(hub))
            {
                string candidate = Path.Combine(snapshot, Pq2FileName);
                if (File.Exists(candidate))
                    return candidate;
            }

        return null;
    }

    private static string? ResolveGgmlDir()
    {
        string? envDir = Environment.GetEnvironmentVariable(GgmlDirEnvVar);
        if (!string.IsNullOrWhiteSpace(envDir) && File.Exists(Path.Combine(envDir, "ggml-base.dll")))
            return envDir;

        return File.Exists(Path.Combine(DefaultGgmlDir, "ggml-base.dll")) ? DefaultGgmlDir : null;
    }

    internal static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-2-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-2-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        // The HF hub cache, where `hf download` puts it: blobs are reached through snapshots/*/.
        string hub = Path.Combine(home, ".cache", "huggingface", "hub",
            "models--prism-ml--Ternary-Bonsai-2-27B-gguf", "snapshots");
        if (Directory.Exists(hub))
            foreach (string snapshot in Directory.EnumerateDirectories(hub))
            {
                string candidate = Path.Combine(snapshot, FileName);
                if (File.Exists(candidate))
                    return candidate;
            }

        return null;
    }
}
