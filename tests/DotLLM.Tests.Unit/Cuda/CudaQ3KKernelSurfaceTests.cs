using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Pins the CUDA backend's Q3_K kernel surface: <b>decode-to-FP16 only, no packed
/// matmul</b>. This is a tripwire, not a correctness test — it fails the moment
/// someone adds a Q3_K GEMV/MMQ/MMVQ or embedding-lookup kernel, so that the
/// parity test such a kernel needs is written at the same time rather than years later.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists. Q3_K shipped with a transposed bit layout in every backend (#311).
/// Fixing it needed more than the dequant path: on Vulkan the <i>packed matmul</i> shaders
/// decode weight bytes inline and had to be rewritten separately — "a sub-block is now 16
/// consecutive bytes at a fixed bit-pair, not one funnelled word". Whether CUDA carries the
/// same class of surface is not something a reader can tell by inspection, and getting it
/// wrong in either direction is expensive: assume a kernel exists and you hunt a phantom
/// (a Q3_K matmul hunt was opened on exactly that assumption and found nothing to fix in
/// any backend); assume it does not and a real one ships untested.
/// </para>
/// <para>
/// It does not exist today. CUDA never grew a packed Q3_K matmul: the whole committed
/// PTX tree exposes exactly one Q3_K entry point, <c>dequant_q3_k_f16</c>, and
/// <see cref="CudaKernels.HasQuantizedGemv"/> excludes Q3_K, so
/// <c>CudaWeights.SkipFp16</c> is false for it and every Q3_K tensor is decoded once at
/// load into a persistent FP16 copy that type-agnostic cuBLAS GEMM/GEMV then consumes.
/// The entire Q3_K-specific surface on CUDA is therefore the one dequant kernel, which
/// <c>RealGgufQ3KCudaDequantParityTests</c> covers on real llama-quantize bytes.
/// </para>
/// <para>
/// If this test fails, do not relax it. A new Q3_K kernel that decodes weight bytes
/// inline needs its own real-GGUF parity test before it can be trusted — copy the shape
/// of <c>RealGgufQ3KCudaDequantParityTests</c> (real quantiser output, CPU oracle
/// reference, never a fixture we encoded ourselves), then widen the expected set here.
/// </para>
/// </remarks>
public sealed class CudaQ3KKernelSurfaceTests
{
    /// <summary>
    /// The complete set of Q3_K entry points the committed PTX is expected to expose.
    /// </summary>
    private static readonly string[] ExpectedQ3KEntryPoints = ["dequant_q3_k_f16"];

    [Fact]
    public void CommittedPtx_ExposesOnlyTheQ3KDequantKernel()
    {
        string ptxDir = FindPtxDir();
        string[] files = Directory.GetFiles(ptxDir, "*.ptx");
        Assert.NotEmpty(files);

        List<string> found = [];
        foreach (string file in files)
        {
            foreach (string raw in File.ReadLines(file))
            {
                string line = raw.Trim();
                if (!line.StartsWith(".visible .entry", StringComparison.Ordinal)) continue;

                string name = ExtractEntryName(line);
                if (name.Contains("q3_k", StringComparison.OrdinalIgnoreCase))
                    found.Add($"{name} ({Path.GetFileName(file)})");
            }
        }

        string[] unexpected = found
            .Where(f => !ExpectedQ3KEntryPoints.Any(e => f.StartsWith(e + " ", StringComparison.Ordinal)))
            .ToArray();

        Assert.True(
            unexpected.Length == 0,
            "The committed PTX exposes Q3_K kernels beyond the dequant path:" + Environment.NewLine
            + string.Join(Environment.NewLine, unexpected) + Environment.NewLine
            + "A kernel that decodes Q3_K weight bytes inline is exactly the surface that stayed "
            + "broken on Vulkan after the #311 dequant fix. Write a real-GGUF parity test for it "
            + "(see RealGgufQ3KCudaDequantParityTests for the pattern: real llama-quantize bytes, "
            + "CPU oracle reference, no self-authored quantiser anywhere in the loop), then add the "
            + "entry point to ExpectedQ3KEntryPoints.");

        // The dequant kernel itself must not vanish — otherwise this test would pass
        // vacuously on a tree with no Q3_K support at all.
        Assert.Contains(found, f => f.StartsWith("dequant_q3_k_f16 ", StringComparison.Ordinal));
    }

    /// <summary>
    /// The C# side of the same statement. <see cref="CudaKernels.HasQuantizedGemv"/> is
    /// what makes <c>CudaWeights.SkipFp16</c> false for Q3_K, which is what forces every
    /// Q3_K weight through the verified dequant kernel into a persistent FP16 copy. If
    /// this flips, Q3_K weights start staying quantized on device and a packed matmul
    /// path becomes live.
    /// </summary>
    [Fact]
    public void QuantizedGemv_DoesNotClaimQ3KSupport()
    {
        Assert.False(
            CudaKernels.HasQuantizedGemv(QuantizationType.Q3_K),
            "CudaKernels.HasQuantizedGemv now claims Q3_K. That makes CudaWeights.SkipFp16 true for "
            + "Q3_K, so weights stay quantized on device and a packed matmul kernel decodes them "
            + "inline — the surface that stayed broken on Vulkan after #311. Add a real-GGUF matmul "
            + "parity test before enabling this.");
    }

    private static string ExtractEntryName(string entryLine)
    {
        // ".visible .entry name(" or ".visible .entry name" — take the token after
        // ".entry", trimmed at the parameter-list parenthesis.
        int start = entryLine.IndexOf(".entry", StringComparison.Ordinal) + ".entry".Length;
        string rest = entryLine[start..].Trim();
        int paren = rest.IndexOf('(', StringComparison.Ordinal);
        if (paren >= 0) rest = rest[..paren];
        return rest.Trim();
    }

    private static string FindPtxDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate native/ptx from the test output directory.");
    }
}
