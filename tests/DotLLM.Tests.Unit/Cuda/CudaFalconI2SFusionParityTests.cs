using System.Runtime.InteropServices;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #221: empirically confirms (not just by reading the gating conditions in isolation)
/// what actually engages for Falcon-E-3B-Instruct and Falcon3-3B-Base-1.58bit — Llama-arch bodies
/// with I2_S weights (fixed by #207) — on load and at decode time, and that it matches the
/// genuine BitNet-arch reference.
///
/// There are TWO independent mechanisms in play, and this test distinguishes them:
///
///  1. <c>CudaWeights.TryUploadPackedThree</c>/<c>TryUploadPackedTwo</c> (load-time VRAM-packing:
///     concatenate Q/K/V or Gate/Up into one device buffer, consumed later by the GENERIC
///     quantized-GEMV decode path <c>LaunchQuantizedGemv</c>/<c>LaunchQuantizedGemvMmq</c>). This
///     gates on <c>CudaKernels.HasLoadedQuantizedGemv(qt)</c>, whose switch (<c>HasMmq</c> /
///     <c>HasQuantizedGemvKernel</c>) has no <c>I2_S</c> case — I2_S is served by its own
///     dedicated kernel family, never by the generic quantized-GEMV dispatch. So this predicate
///     is <c>false</c> for I2_S UNCONDITIONALLY, for every I2_S model including real BitNet, not
///     just Falcon. <c>lw.QkvPacked</c>/<c>lw.GateUpPacked</c> are therefore always 0 for I2_S —
///     by design, not a Falcon-specific gap (verified below against the BitNet reference too).
///
///  2. <c>CanFuseI2SDecode</c> (the actually performance-relevant decode-time fusion: ONE kernel
///     launch — <c>LaunchI2_SGemv3F16In</c>/<c>LaunchI2_SGemv2F16In</c> — computing all of
///     Q+K+V or Gate+Up from the THREE/TWO separate per-tensor quantized pointers, no shared
///     buffer required). This is entirely independent of (1) — it never reads
///     <c>lw.QkvPacked</c>/<c>lw.GateUpPacked</c> — and gates purely on quant type + input-dim
///     match + 128-alignment, nothing architecture-specific. This is the one the issue is really
///     about, and this test confirms it evaluates true for every layer of both Falcon models.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public class CudaFalconI2SFusionParityTests
{
    private const string FalconEPathEnvVar = "DOTLLM_FALCON_E_3B_GGUF";
    private const string FalconEDefaultPath = @"E:\Development\bitnet-tests\models\Falcon-E-3B-Instruct\ggml-model-i2_s.gguf";

    private const string Falcon3PathEnvVar = "DOTLLM_FALCON3_3B_GGUF";
    private const string Falcon3DefaultPath = @"E:\Development\bitnet-tests\models\Falcon3-3B-Base-1.58bit\ggml-model-i2_s.gguf";

    private const string BitNetPathEnvVar = "DOTLLM_BITNET_2B4T_GGUF";
    private const string BitNetDefaultPath =
        @"E:\.cache\huggingface\hub\models--microsoft--bitnet-b1.58-2B-4T-gguf\snapshots\a1f2f1c765812aa8af3f6eda4a313707064bba15\ggml-model-i2_s.gguf";

    private readonly ITestOutputHelper _out;
    public CudaFalconI2SFusionParityTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [SkippableFact]
    public void FalconE3BInstruct_I2S_DecodeFusionMatchesBitNetDesign()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath(FalconEPathEnvVar, FalconEDefaultPath);
        Skip.If(path is null, $"Falcon-E-3B-Instruct I2_S GGUF fixture not found. Set {FalconEPathEnvVar}.");
        AssertFusionParity(path!, Architecture.Llama);
    }

    [SkippableFact]
    public void Falcon3_3BBase_I2S_DecodeFusionMatchesBitNetDesign()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath(Falcon3PathEnvVar, Falcon3DefaultPath);
        Skip.If(path is null, $"Falcon3-3B-Base-1.58bit I2_S GGUF fixture not found. Set {Falcon3PathEnvVar}.");
        AssertFusionParity(path!, Architecture.Llama);
    }

    [SkippableFact]
    public void BitNetB158_2B4T_Reference_QkvPackingAlsoNeverEngages()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? path = ResolveFixturePath(BitNetPathEnvVar, BitNetDefaultPath);
        Skip.If(path is null, $"BitNet b1.58 2B4T I2_S GGUF fixture not found. Set {BitNetPathEnvVar}.");
        // Reference point for finding (1) in the class remarks: confirms lw.QkvPacked/GateUpPacked
        // staying at 0 is universal I2_S behavior (HasLoadedQuantizedGemv(I2_S) is unconditionally
        // false), not something specific to the Llama-arch Falcon loader path.
        AssertFusionParity(path!, Architecture.BitNet);
    }

    private void AssertFusionParity(string path, Architecture expectedArch)
    {
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(expectedArch, config.Architecture);

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        int layerCount = model.DiagLayerCount;
        Assert.True(layerCount > 0, "Model reports zero layers.");

        int qkvPackedLayers = 0, gateUpPackedLayers = 0;
        int qkvFusableLayers = 0, gateUpFusableLayers = 0;
        for (int layer = 0; layer < layerCount; layer++)
        {
            if (model.DiagIsQkvPacked(layer)) qkvPackedLayers++;
            if (model.DiagIsGateUpPacked(layer)) gateUpPackedLayers++;
            if (model.DiagCanFuseQkvDecode(layer)) qkvFusableLayers++;
            if (model.DiagCanFuseGateUpDecode(layer)) gateUpFusableLayers++;
        }
        _out.WriteLine($"{Path.GetFileName(path)} ({expectedArch}): hidden={config.HiddenSize}, " +
                       $"intermediate={config.IntermediateSize}, layers={layerCount}, " +
                       $"QkvPacked={qkvPackedLayers}/{layerCount}, GateUpPacked={gateUpPackedLayers}/{layerCount}, " +
                       $"QkvDecodeFusable={qkvFusableLayers}/{layerCount}, GateUpDecodeFusable={gateUpFusableLayers}/{layerCount}");

        // (1) VRAM buffer-packing: never engages for I2_S, by design (HasLoadedQuantizedGemv has
        // no I2_S case — the packed buffer is only consumed by the generic quantized-GEMV path,
        // which I2_S never uses). True for BitNet as much as for Falcon — not a parity gap.
        Assert.Equal(0, qkvPackedLayers);
        Assert.Equal(0, gateUpPackedLayers);

        // (2) The actually performance-relevant decode-time multi-tensor fusion: purely
        // quant-type/dim gated, nothing architecture-specific, so it must engage for every layer
        // of a uniform-I2_S model regardless of GGUF architecture tag.
        Assert.Equal(layerCount, qkvFusableLayers);
        Assert.Equal(layerCount, gateUpFusableLayers);
    }

    private static string? ResolveFixturePath(string envVar, string defaultPath)
    {
        string? envPath = Environment.GetEnvironmentVariable(envVar);
        if (!string.IsNullOrWhiteSpace(envPath) && File.Exists(envPath))
            return envPath;
        return File.Exists(defaultPath) ? defaultPath : null;
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
