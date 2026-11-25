using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA Olala (#1984): the device-state layout, checked without a GPU. James's 2026-11-25 rulings: the Mamba SSM matrix, rotary phase,
/// trapezoid carry and token-shift carry are F32; stored activations are BF16 (never FP16: they reach ~1e9..1e14). The allocator in
/// <c>CudaOlalaTransformerModel.Upload</c> sizes every buffer from <see cref="CudaOlalaStateLayout"/>, which reads the widths from
/// <see cref="OlalaStatePolicy"/>, so narrowing one buffer shows up here.
/// </summary>
public sealed class CudaOlalaStateLayoutTests
{
    private static (OlalaConfig O, ModelConfig C) Real7A1B()
    {
        var o = new OlalaConfig
        {
            LayerTypes = "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM", SsmInner = 3072, StateSize = 128, GroupCount = 1, MambaHeadDim = 64, MimoRank = 4,
            NumRopeAngles = 32, NumSignalHeads = 36, NumNoiseHeads = 12, TpaRank = 4, GateBias = 1.15f, ExpertCount = 256, ExpertUsedCount = 6,
            ExpertFfLength = 768, SharedExpertFfLength = 1536, ExpertWeightsScale = 2.45f, MoeLatentSize = 384,
        };
        var c = new ModelConfig
        {
            Architecture = Architecture.Olala, VocabSize = 151936, HiddenSize = 1536, IntermediateSize = 0, NumLayers = 36,
            NumAttentionHeads = 48, NumKvHeads = 12, HeadDim = 128, MaxSequenceLength = 100000, Olala = o,
        };
        return (o, c);
    }

    [Fact]
    public void RecurrentState_IsF32_PerBufferByteCounts()
    {
        var (o, c) = Real7A1B();
        var ssm = CudaOlalaStateLayout.Ssm(o);
        // docs/OLALA.md section 4 (per M layer): 1,572,864 + 6,144 + 98,304 + 12,288 bytes, i.e. 4 bytes per element for every buffer.
        Assert.Equal(48L * 64 * 128 * 4, ssm.SsmBytes);
        Assert.Equal(48L * 32 * 4, ssm.AngleBytes);
        Assert.Equal(4L * 48 * 128 * 4, ssm.TrapKBytes);
        Assert.Equal(48L * 64 * 4, ssm.TrapVBytes);
        Assert.Equal(1_572_864L + 6_144 + 98_304 + 12_288, ssm.Total);
        var attn = CudaOlalaStateLayout.Attn(o, c);
        Assert.Equal(12L * 128 * 4, attn.LastKBytes);   // token-shift carry: F32 (user ruling 2026-11-25), HF keeps bf16
        Assert.Equal(12L * 128 * 4, attn.LastVBytes);
        Assert.Equal(2L * 12 * 128 * 2, attn.HistoryBytesPerToken);   // K + V history: BF16
        Assert.Equal(sizeof(float), OlalaStatePolicy.ElementBytes(OlalaStatePolicy.Ssm));
    }

    [Fact]
    public void Activations_AreBf16_NotFp16_NotF32()
    {
        Assert.Equal(2, CudaOlalaStateLayout.ActivationBytes);
        // The kernel's act_t is `unsigned short` holding BF16 bits (native/kernels/olala.cu); the C# side must not diverge from it.
        string cu = File.ReadAllText(FindKernelSource());
        Assert.Contains("typedef unsigned short act_t;   // BF16 bits", cu);
        Assert.DoesNotContain("__half* act", cu);
    }

    [Fact]
    public void TotalBytes_7A1B_MatchesDocs()
    {
        var (o, c) = Real7A1B();
        long fixedPart = (1_572_864L + 6_144 + 98_304 + 12_288) * 29 + 12_288L * 7;
        Assert.Equal(fixedPart, CudaOlalaStateLayout.TotalBytes(o, c, 0));
        // 7 V layers x 6 KiB of BF16 K+V per token.
        Assert.Equal(fixedPart + 7L * 6144 * 1000, CudaOlalaStateLayout.TotalBytes(o, c, 1000));
    }

    [Fact]
    public void Geometry_Limits_RejectWhatTheKernelsCannotDo()
    {
        var (o, c) = Real7A1B();
        CudaOlalaTransformerModel.ValidateGeometry(c, o);
        Assert.Throws<NotSupportedException>(() => CudaOlalaTransformerModel.ValidateGeometry(c, o with { StateSize = 256 }));
        Assert.Throws<NotSupportedException>(() => CudaOlalaTransformerModel.ValidateGeometry(c, o with { MimoRank = 8 }));
        Assert.Throws<NotSupportedException>(() => CudaOlalaTransformerModel.ValidateGeometry(c with { HeadDim = 256 }, o));
    }

    [Fact]
    public void GenericCudaTransformerModel_StillRefusesOlala_AndPointsAtTheDedicatedLoader()
    {
        // Olala must reach CudaOlalaTransformerModel through CudaModelLoader.CreateFromGguf; the generic forward would silently run another model.
        var (_, c) = Real7A1B();
        var ex = Assert.Throws<NotSupportedException>(() => CudaTransformerModel.RejectUnsupportedArchitecture(c));
        Assert.Contains("CudaOlalaTransformerModel", ex.Message, StringComparison.Ordinal);
        Assert.Contains("CreateFromGguf", ex.Message, StringComparison.Ordinal);
    }

    private static string FindKernelSource([System.Runtime.CompilerServices.CallerFilePath] string? here = null)
    {
        for (var dir = new DirectoryInfo(Path.GetDirectoryName(here!)!); dir is not null; dir = dir.Parent)
        {
            string p = Path.Combine(dir.FullName, "native", "kernels", "olala.cu");
            if (File.Exists(p)) return p;
        }
        throw new FileNotFoundException("native/kernels/olala.cu");
    }
}
