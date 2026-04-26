using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// End-to-end CPU↔CUDA parity for the MLA + (optional) MoE dispatch wired
/// into <see cref="CudaTransformerModel.Forward"/>. Writes a synthetic
/// DeepSeek-V2-style safetensors checkpoint, loads it through both the CPU
/// <see cref="TransformerModel"/> and the GPU <see cref="CudaTransformerModel"/>,
/// runs prefill on a fixed token sequence, and asserts the last-position
/// logits match within FP16 noise. The decisive evidence that the dispatch
/// glue actually works end-to-end (the kernel-level GPU helpers are already
/// covered separately by <see cref="CudaMlaForwardTests"/> +
/// <see cref="CudaMoeFfnTests"/>).
/// </summary>
[Trait("Category", "GPU")]
public sealed class CudaTransformerMlaForwardTests : IDisposable
{
    // Tiny but production-shaped MLA fixture. Two layers (both dense FFN
    // for the MLA-only test; layer-1 swapped to MoE for the MLA+MoE variant)
    // is enough to exercise per-layer pointer reuse without making logits
    // saturate. Heads / hidden / vocab kept small to keep prefill < 100 ms.
    private const int HiddenSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int QkNope = 4;
    private const int QkRope = 4;
    private const int VHead = 4;
    private const int KvLoraRank = 8;
    private const int IntermediateSize = 24;
    private const int MoeIntermediate = 24;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;

    /// <summary>
    /// FP16↔F32 conversion noise compounds across layers; this tolerance
    /// matches the loose-tol bound used by
    /// <c>MlaForward_DeepSeekV2LiteShapes_MatchesCpuOracle</c> (cuBLAS
    /// HGEMM tensor-core SGEMM order vs CPU scalar order over chained dot
    /// products — small per-element drift that grows with hidden size and
    /// layer count).
    /// </summary>
    private const float Tolerance = 5e-2f;

    private readonly string _scratch;

    public CudaTransformerMlaForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mla-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
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

    [SkippableFact]
    public void MlaForward_DenseFfn_LoRAQ_MatchesCpu()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var ptx = FindPtxDir();
        Skip.If(ptx == null, "PTX files not found");

        RunPrefillCompare(qLoraRank: 8, includeMoe: false, seed: 42, ptxDir: ptx!);
    }

    [SkippableFact]
    public void MlaForward_DenseFfn_MonolithicQ_MatchesCpu()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var ptx = FindPtxDir();
        Skip.If(ptx == null, "PTX files not found");

        // V2-Lite-style: q_lora_rank=0 (monolithic Q projection).
        RunPrefillCompare(qLoraRank: 0, includeMoe: false, seed: 7, ptxDir: ptx!);
    }

    [SkippableFact]
    public void MlaMoeForward_LoRAQ_MatchesCpu()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var ptx = FindPtxDir();
        Skip.If(ptx == null, "PTX files not found");

        // Both MLA dispatch and MoE dispatch active. Layer 0 stays dense
        // (DeepSeek-V2's first_k_dense_replace=1 convention) and layer 1
        // routes through MoE — exercises the per-layer MoE branch in the
        // dispatcher.
        RunPrefillCompare(qLoraRank: 8, includeMoe: true, seed: 99, ptxDir: ptx!);
    }

    private unsafe void RunPrefillCompare(int qLoraRank, bool includeMoe, int seed, string ptxDir)
    {
        string path = Path.Combine(_scratch, $"mla{qLoraRank}-moe{includeMoe}.safetensors");
        WriteFixture(path, qLoraRank, includeMoe, seed);

        var config = BuildConfig(qLoraRank, includeMoe);

        // Deterministic small token sequence.
        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];

        // ── CPU forward ──
        float[] cpuLastRow;
        using (var sf = SafetensorsFile.Open(path))
        using (var cpu = TransformerModel.LoadFromSafetensors(sf, config))
        {
            using ITensor logits = cpu.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(VocabSize, logits.Shape[1]);
            cpuLastRow = CopyRow(logits, tokenIds.Length - 1);
        }

        // ── GPU forward (uses the new MLA + (optional) MoE dispatch) ──
        float[] gpuLastRow;
        using (var sf = SafetensorsFile.Open(path))
        using (var gpu = CudaTransformerModel.LoadFromSafetensors(sf, config, deviceId: 0, ptxDir: ptxDir))
        {
            using ITensor logits = gpu.Forward(tokenIds, positions, deviceId: 0, kvCache: null);
            Assert.Equal(VocabSize, logits.Shape[1]);
            // GPU returns last-token only ([1, vocab]); CPU returns [seqLen, vocab].
            gpuLastRow = CopyRow(logits, 0);
        }

        AssertClose(cpuLastRow, gpuLastRow,
            label: $"MLA(qLora={qLoraRank}) MoE={includeMoe} prefill last-row");
    }

    private static unsafe float[] CopyRow(ITensor logits, int rowIndex)
    {
        int cols = logits.Shape.Rank == 1 ? logits.Shape[0] : logits.Shape[1];
        float[] row = new float[cols];
        new ReadOnlySpan<float>(
            (void*)(logits.DataPointer + (nint)((long)rowIndex * cols * sizeof(float))),
            cols).CopyTo(row);
        return row;
    }

    private static void AssertClose(float[] cpu, float[] gpu, string label)
    {
        Assert.Equal(cpu.Length, gpu.Length);

        // Surface NaN/Inf explicitly — tells us the dispatch is producing
        // garbage rather than just drifting outside tolerance.
        int gpuFinite = 0, cpuFinite = 0;
        for (int c = 0; c < cpu.Length; c++)
        {
            if (float.IsFinite(cpu[c])) cpuFinite++;
            if (float.IsFinite(gpu[c])) gpuFinite++;
        }
        Assert.True(cpuFinite == cpu.Length,
            $"{label}: CPU produced {cpu.Length - cpuFinite}/{cpu.Length} non-finite logits — fixture bug.");
        Assert.True(gpuFinite == gpu.Length,
            $"{label}: GPU produced {gpu.Length - gpuFinite}/{gpu.Length} non-finite logits. " +
            $"cpu=[{string.Join(", ", cpu.Select(v => v.ToString("F3")))}], " +
            $"gpu=[{string.Join(", ", gpu.Select(v => v.ToString("F3")))}]");

        float maxDiff = 0f;
        int worstCol = 0;
        for (int c = 0; c < cpu.Length; c++)
        {
            float d = MathF.Abs(cpu[c] - gpu[c]);
            if (d > maxDiff) { maxDiff = d; worstCol = c; }
        }
        Assert.True(maxDiff <= Tolerance,
            $"{label}: max |Δlogit| = {maxDiff:E3} at col {worstCol} " +
            $"(cpu={cpu[worstCol]:F4}, gpu={gpu[worstCol]:F4}) > {Tolerance:E3}; " +
            $"cpu=[{string.Join(", ", cpu.Select(v => v.ToString("F3")))}], " +
            $"gpu=[{string.Join(", ", gpu.Select(v => v.ToString("F3")))}]");
    }

    private static ModelConfig BuildConfig(int qLoraRank, bool includeMoe)
    {
        var mla = new MlaConfig
        {
            KvLoraRank = KvLoraRank,
            QLoraRank = qLoraRank,
            QkNopeHeadDim = QkNope,
            QkRopeHeadDim = QkRope,
            VHeadDim = VHead,
            RopeTheta = 10000.0f,
        };
        var rope = new RoPEConfig(Theta: 10000.0f, DimensionCount: QkNope + QkRope, Type: RoPEType.Norm);

        MoeConfig? moe = includeMoe
            ? new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = NumExpertsPerTok,
                MoeIntermediateSize = MoeIntermediate,
                NormTopKProb = true,
                // Layer 0 stays dense to exercise the per-layer alternation
                // (Qwen-MoE / DeepSeek-V2 convention via DecoderSparseStep).
                MlpOnlyLayers = new[] { 0 },
            }
            : null;

        return new ModelConfig
        {
            Architecture = Architecture.DeepSeekV2,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = QkNope + QkRope,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.MLA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            MlaConfig = mla,
            Moe = moe,
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int qLoraRank, bool includeMoe, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qkHead = QkNope + QkRope;
        int qTotal = NumHeads * qkHead;
        int kvADim = KvLoraRank + QkRope;
        int kvBOut = NumHeads * (QkNope + VHead);
        int oInput = NumHeads * VHead;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

            if (qLoraRank > 0)
            {
                AddRand(b, $"{prefix}.self_attn.q_a_proj.weight", [qLoraRank, HiddenSize], 0.1f, s + 2);
                AddRand(b, $"{prefix}.self_attn.q_a_layernorm.weight", [qLoraRank],
                        amplitude: 0.05f, seed: s + 3, center: 1.0f, jitter: 0.05f);
                AddRand(b, $"{prefix}.self_attn.q_b_proj.weight", [qTotal, qLoraRank], 0.1f, s + 4);
            }
            else
            {
                AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qTotal, HiddenSize], 0.1f, s + 2);
            }
            AddRand(b, $"{prefix}.self_attn.kv_a_proj_with_mqa.weight", [kvADim, HiddenSize], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.kv_a_layernorm.weight", [KvLoraRank],
                    amplitude: 0.05f, seed: s + 6, center: 1.0f, jitter: 0.05f);
            AddRand(b, $"{prefix}.self_attn.kv_b_proj.weight", [kvBOut, KvLoraRank], 0.1f, s + 7);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, oInput], 0.1f, s + 8);

            // FFN: layer 0 stays dense; layer 1 is MoE when includeMoe is on
            // (matches DeepSeek-V2's first_k_dense_replace=1 + sparse alternation).
            bool isMoeLayer = includeMoe && i > 0;
            if (isMoeLayer)
            {
                AddRand(b, $"{prefix}.mlp.gate.weight", [NumExperts, HiddenSize], 0.05f, s + 9);
                for (int e = 0; e < NumExperts; e++)
                {
                    AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight",
                            [MoeIntermediate, HiddenSize], 0.05f, s + 100 + e * 3);
                    AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight",
                            [MoeIntermediate, HiddenSize], 0.05f, s + 101 + e * 3);
                    AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight",
                            [HiddenSize, MoeIntermediate], 0.05f, s + 102 + e * 3);
                }
            }
            else
            {
                AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 9);
                AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.05f, s + 10);
                AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 11);
            }
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Deterministic small-magnitude cos-based fill — matches the pattern in
    /// <see cref="TransformerModelMlaForwardTests"/> so the CPU side's
    /// existing-tested flow exercises the same fixture surface.
    /// </summary>
    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            if (jitter > 0f) values[i] = center + jitter * cos;
            else values[i] = amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }
}
