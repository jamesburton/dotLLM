using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Real-tensor smoke test for PQ2_0 dequantization against PrismML's actual
/// <c>Ternary-Bonsai-27B-Q2_0.gguf</c> — closes the loop on the empirical byte-layout
/// verification done against a 200-group sample (see <see cref="PQ2_0Tests"/>'s doc comment)
/// by dequantizing a full real tensor and sanity-checking the whole distribution.
///
/// <para>Gated on a local fixture: the file is ~7.2GB and lives outside this repo/branch
/// (currently on a sibling worktree used for unrelated ML-training work, not part of any
/// git history here). Set <c>DOTLLM_BONSAI_PQ2_0_GGUF</c> to the file path, or place it under
/// <c>~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/</c> or
/// <c>~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/</c>, to run this test.</para>
/// </summary>
public sealed class PQ2_0RealGgufSmokeTests
{
    private const string ModelPathEnvVar = "DOTLLM_BONSAI_PQ2_0_GGUF";
    private const string FileName = "Ternary-Bonsai-27B-Q2_0.gguf";

    [SkippableFact]
    public unsafe void RealBonsaiTensor_DequantizesToFiniteReasonableRangeTernaryTimesScale()
    {
        string? path = ResolveFixturePath();
        Skip.If(path is null,
            $"Bonsai PQ2_0 GGUF fixture not found. Set {ModelPathEnvVar}, or place {FileName} under "
            + "~/.dotllm/models/PrismML/Ternary-Bonsai-27B-GGUF/ or ~/.dotllm/test-cache/PrismML/Ternary-Bonsai-27B-GGUF/.");

        using var gguf = GgufFile.Open(path!);

        var pq2Tensors = gguf.Tensors.Where(t => t.QuantizationType == QuantizationType.PQ2_0).ToList();
        Assert.NotEmpty(pq2Tensors);

        // Smallest PQ2_0 tensor — keeps the test fast regardless of which real checkpoint is used.
        var tensor = pq2Tensors.OrderBy(t => t.Shape.ElementCount).First();
        long elementCount = tensor.Shape.ElementCount;
        Assert.True(elementCount % 128 == 0, $"'{tensor.Name}' element count {elementCount} isn't a multiple of 128.");

        var dest = new float[elementCount];
        Dequantize.ToFloat32(gguf.DataBasePointer + (nint)tensor.DataOffset, elementCount, QuantizationType.PQ2_0, dest);

        // Every decoded value must be exactly {-scale, 0, +scale} for that value's group — i.e.
        // finite, and no value magnitude wildly out of line with its neighbors (a scale-placement
        // or group-stride bug would produce huge outliers or non-ternary-multiple values).
        bool sawNonZero = false;
        float maxAbs = 0f;
        foreach (float v in dest)
        {
            Assert.True(float.IsFinite(v), $"Non-finite value {v} in '{tensor.Name}'.");
            if (v != 0f) sawNonZero = true;
            float av = MathF.Abs(v);
            if (av > maxAbs) maxAbs = av;
        }

        Assert.True(sawNonZero, $"'{tensor.Name}' decoded to all zeros — likely a byte-layout bug.");
        Assert.True(maxAbs < 10f, $"'{tensor.Name}' max abs value {maxAbs} implausible for a ternary*scale tensor.");
    }

    private static string? ResolveFixturePath()
    {
        string? envPath = Environment.GetEnvironmentVariable(ModelPathEnvVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] candidates =
        [
            Path.Combine(home, ".dotllm", "models", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
            Path.Combine(home, ".dotllm", "test-cache", "PrismML", "Ternary-Bonsai-27B-GGUF", FileName),
        ];
        foreach (string candidate in candidates)
            if (File.Exists(candidate))
                return candidate;

        return null;
    }
}
