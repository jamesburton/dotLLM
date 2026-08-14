using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Real-weight CPU-vs-CUDA forward parity for <see cref="CudaMamba3TransformerModel"/>
/// on <c>ib-ssm/mamba3-370M-10BT</c> (issue #346's acceptance criterion: "Real-model
/// CPU vs CUDA parity test (prefill + decode)"). Combines
/// <c>IbSsmMamba3RealWeightsLoadTests</c>'s checkpoint-resolution + state-threaded
/// prefill/decode methodology with <c>IbSsmMamba3VulkanGenerationTests</c>'s
/// cross-backend comparison structure — both read in full while planning this test.
/// </summary>
/// <remarks>
/// <b>Gating.</b> Same as the CPU/Vulkan siblings: <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c>
/// env var, then <c>C:/temp/dotllm-ibssm/model.safetensors</c>, then
/// <c>%USERPROFILE%/dotllm-ibssm-370m/model.safetensors</c>. Skips gracefully if none
/// resolve. Additionally requires a CUDA device.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class IbSsmMamba3CudaParityTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    // Calibrated from IbSsmMamba3VulkanGenerationTests.LogitsAbsTol (also 3.0f) for the
    // same real 370M checkpoint's O(10)-magnitude logits — NOT the synthetic fixture's
    // 1e-6 scale (CudaMamba3ParitySyntheticTests), which is ~300x smaller and would be
    // meaningless here. See that class's remarks for the scale-mismatch reasoning.
    private const float LogitsAbsTol = 3.0f; // matches IbSsmMamba3VulkanGenerationTests

    private readonly ITestOutputHelper _output;
    public IbSsmMamba3CudaParityTests(ITestOutputHelper output) => _output = output;

    private static string? ResolveCheckpointPath()
    {
        string? env = Environment.GetEnvironmentVariable(CheckpointPathEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (File.Exists(env)) return env;
            if (Directory.Exists(env))
            {
                string candidate = Path.Combine(env, SafetensorsName);
                if (File.Exists(candidate)) return candidate;
            }
        }

        string conventional = Path.Combine(ConventionalDir, SafetensorsName);
        if (File.Exists(conventional)) return conventional;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(home))
        {
            string fallback = Path.Combine(home, UserProfileFallbackDir, SafetensorsName);
            if (File.Exists(fallback)) return fallback;
        }
        return null;
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnPromptPrefill()
    {
        string? checkpointPath = ResolveCheckpointPath();
        Skip.If(checkpointPath is null,
            $"ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
            + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        var cpuLoadWatch = Stopwatch.StartNew();
        var (cpuModel, cpuFile, config) = CheckpointGuard.LoadOrSkip(
            checkpointPath!, "ib-ssm/mamba3-370M-10BT checkpoint (CPU)",
            () => ModelLoader.LoadFromSafetensors(checkpointPath!));
        cpuLoadWatch.Stop();
        Assert.Equal(Architecture.Mamba3, config.Architecture);
        _output.WriteLine($"CPU load: {cpuLoadWatch.Elapsed.TotalSeconds:F1} s");

        var cudaLoadWatch = Stopwatch.StartNew();
        var (cudaModel, cudaSource, _) = CheckpointGuard.LoadOrSkip(
            checkpointPath!, "ib-ssm/mamba3-370M-10BT checkpoint (CUDA)",
            () => CudaModelLoader.LoadMamba3FromSafetensors(checkpointPath!));
        cudaLoadWatch.Stop();
        _output.WriteLine($"CUDA load: {cudaLoadWatch.Elapsed.TotalSeconds:F1} s");

        try
        {
            int[] tokenIds = [0, 100, 1000, 10000, 31999];
            int[] positions = [0, 1, 2, 3, 4];

            using ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            using ITensor cudaLogits = cudaModel.Forward(tokenIds, positions, deviceId: -1);

            float[] cpuLast = LastRow(cpuLogits, config.VocabSize);
            float[] cudaLast = LastRow(cudaLogits, config.VocabSize);

            AssertLogitsMatch(cpuLast, cudaLast, "prefill");
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    [SkippableFact]
    public void CudaDecode_MatchesCpuReference_PrefillThenDecode()
    {
        string? checkpointPath = ResolveCheckpointPath();
        Skip.If(checkpointPath is null,
            $"ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar}.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        var (cpuModel, cpuFile, config) = CheckpointGuard.LoadOrSkip(
            checkpointPath!, "ib-ssm/mamba3-370M-10BT checkpoint (CPU)",
            () => ModelLoader.LoadFromSafetensors(checkpointPath!));
        var (cudaModel, cudaSource, _) = CheckpointGuard.LoadOrSkip(
            checkpointPath!, "ib-ssm/mamba3-370M-10BT checkpoint (CUDA)",
            () => CudaModelLoader.LoadMamba3FromSafetensors(checkpointPath!));
        try
        {
            var cpuM3 = Assert.IsType<Mamba3TransformerModel>(cpuModel);

            int[] tokenIds = [0, 100, 1000];
            int[] positions = [0, 1, 2];
            int vocabSize = config.VocabSize;

            // CPU: prefill 2 + decode 1, state-threaded (mirrors DecodeMatchesPrefillOnRealCheckpoint).
            using var cpuState = new Mamba3State(config);
            using ITensor cpuSplitPrefill = cpuM3.Forward(tokenIds.AsSpan(0, 2), positions.AsSpan(0, 2), deviceId: -1, cpuState);
            using ITensor cpuDecodeTail = cpuM3.Forward(tokenIds.AsSpan(2, 1), positions.AsSpan(2, 1), deviceId: -1, cpuState);
            float[] cpuLast = LastRow(cpuDecodeTail, vocabSize);

            // CUDA: same split, state-threaded via CudaMamba3StateCache. deviceId=-1 here means
            // "host-resident output tensor" (see ForwardCore's doc comment, Task 9) — it does NOT
            // select which GPU runs the compute; that was fixed at LoadMamba3FromSafetensors time.
            using var cudaState = new CudaMamba3StateCache(config.Mamba3Config!, config.NumLayers);
            using ITensor cudaSplitPrefill = cudaModel.Forward(tokenIds.AsSpan(0, 2), positions.AsSpan(0, 2), deviceId: -1, cudaState);
            using ITensor cudaDecodeTail = cudaModel.Forward(tokenIds.AsSpan(2, 1), positions.AsSpan(2, 1), deviceId: -1, cudaState);
            float[] cudaLast = LastRow(cudaDecodeTail, vocabSize);

            AssertLogitsMatch(cpuLast, cudaLast, "prefill+decode");
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    private void AssertLogitsMatch(float[] cpuLast, float[] cudaLast, string label)
    {
        foreach (float v in cpuLast) Assert.True(float.IsFinite(v), $"[{label}] CPU logits contain NaN/Inf.");
        foreach (float v in cudaLast) Assert.True(float.IsFinite(v), $"[{label}] CUDA logits contain NaN/Inf.");

        float maxAbs = 0f;
        int worstIdx = 0;
        for (int i = 0; i < cpuLast.Length; i++)
        {
            float diff = MathF.Abs(cpuLast[i] - cudaLast[i]);
            if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
        }

        int cpuArg = ArgMax(cpuLast);
        int cudaArg = ArgMax(cudaLast);
        _output.WriteLine(
            $"[{label}] max_abs={maxAbs:E3} at idx {worstIdx} "
            + $"(cpu={cpuLast[worstIdx]:G6} cuda={cudaLast[worstIdx]:G6}); "
            + $"argmax: cpu={cpuArg} cuda={cudaArg}");

        Assert.True(maxAbs <= LogitsAbsTol,
            $"[{label}] L-inf logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} at idx {worstIdx}.");
        Assert.Equal(cpuArg, cudaArg);
    }

    private static unsafe float[] LastRow(ITensor logits, int vocabSize)
    {
        int seqLen = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice((seqLen - 1) * vocabSize, vocabSize).CopyTo(result);
        return result;
    }

    private static int ArgMax(float[] values)
    {
        int idx = 0;
        float best = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > best) { best = values[i]; idx = i; }
        return idx;
    }
}
