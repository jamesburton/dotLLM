using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Full-model CPU-vs-GPU last-token-logits parity for a synthetic identity-MoTE-style
/// BitNet-ternary MoE checkpoint (issue #246). Unlike <see cref="CudaMoeFfnBitNetI2STests"/>
/// (which builds a <see cref="CudaMoeLayerWeights"/> directly and calls
/// <see cref="CudaMoeFfn.Forward"/> in isolation), this test exercises the FULL load path —
/// <see cref="CudaTransformerModel.LoadFromSafetensors"/> → <c>CudaWeights</c>'s
/// <c>lw.Moe!.IsBitNetI2S</c> dispatch → <see cref="CudaMoeWeightsLoader.LoadLayerBitNetI2S"/>
/// (the per-expert tail-scale-append upload) → the per-layer MoE forward dispatch inside
/// <c>CudaTransformerModel.Forward</c> (which already calls <see cref="CudaMoeFfn.Forward"/>
/// generically for any MoE layer, so no changes were needed there — see this class's remarks).
/// </summary>
/// <remarks>
/// <b>Fixture.</b> Mirrors <c>MoteBitNetMoeLoaderScaffoldTests.BuildBitNetMoeFixture</c> (same
/// dims: hidden=128, intermediate=256 — both multiples of 128 for the I2_S block invariant) but
/// keeps layer 1 as the only MoE layer (layer 0 forced dense via <c>mlp_only_layers=[0]</c>) so
/// the fixture also exercises the existing dense-BitNet CUDA attention/FFN path unmodified by
/// this issue.
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class CudaTransformerModelBitNetMoeTests : IDisposable
{
    // Wider tolerance than the isolated kernel-level parity in CudaMoeFfnBitNetI2STests
    // (1e-3): this compares FULL multi-layer forward logits (attention I2_S GEMV + MoE I2_S
    // GEMV + norms + lm_head), so per-layer F32 reduction-order drift compounds. Same order
    // of magnitude as CudaTransformerMlaForwardTests' whole-model tolerance (5e-2) for a
    // similarly-shaped multi-component parity gate.
    private const float Tolerance = 8e-2f;

    private const int Hidden = 128;
    private const int Intermediate = 256;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 32;
    private const int Vocab = 32;
    private const int NumExperts = 3;
    private const int TopK = 1;
    private const float Eps = 1e-5f;

    private readonly string _scratch;

    public CudaTransformerModelBitNetMoeTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mote-{Guid.NewGuid():N}");
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
    public void BitNetMoeModel_Decode_MatchesCpuOracle()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        string path = BuildBitNetMoeFixture(withGateBias: true, seed: 4242);
        var config = BuildBitNetMoeConfig();

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        float[] cpuLastRow;
        using (var sf = SafetensorsFile.Open(path))
        using (var cpu = TransformerModel.LoadFromSafetensors(sf, config))
        {
            using ITensor logits = cpu.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(Vocab, logits.Shape[1]);
            cpuLastRow = CopyRow(logits, tokenIds.Length - 1, Vocab);
        }

        float[] gpuLastRow;
        using (var sf = SafetensorsFile.Open(path))
        using (var gpu = CudaTransformerModel.LoadFromSafetensors(sf, config, deviceId: 0, ptxDir: ptxDir))
        {
            using ITensor logits = gpu.Forward(tokenIds, positions, deviceId: 0, kvCache: null);
            Assert.Equal(Vocab, logits.Shape[1]);
            // GPU decode-style forward returns last-token only ([1, vocab]).
            gpuLastRow = CopyRow(logits, 0, Vocab);
        }

        int cpuFinite = cpuLastRow.Count(float.IsFinite);
        int gpuFinite = gpuLastRow.Count(float.IsFinite);
        Assert.True(cpuFinite == Vocab, $"CPU produced {Vocab - cpuFinite}/{Vocab} non-finite logits — fixture bug.");
        Assert.True(gpuFinite == Vocab,
            $"GPU produced {Vocab - gpuFinite}/{Vocab} non-finite logits. "
          + $"cpu=[{string.Join(", ", cpuLastRow)}], gpu=[{string.Join(", ", gpuLastRow)}]");

        float maxDiff = 0f;
        int worstCol = 0;
        for (int c = 0; c < Vocab; c++)
        {
            float d = MathF.Abs(cpuLastRow[c] - gpuLastRow[c]);
            if (d > maxDiff) { maxDiff = d; worstCol = c; }
        }
        Assert.True(maxDiff <= Tolerance,
            $"max |Δlogit| = {maxDiff:E3} at col {worstCol} "
          + $"(cpu={cpuLastRow[worstCol]:F4}, gpu={gpuLastRow[worstCol]:F4}) > {Tolerance:E3}; "
          + $"cpu=[{string.Join(", ", cpuLastRow)}], gpu=[{string.Join(", ", gpuLastRow)}]");
    }

    private static float[] CopyRow(ITensor logits, int rowIndex, int cols)
    {
        float[] row = new float[cols];
        new ReadOnlySpan<float>(
            (void*)(logits.DataPointer + (nint)((long)rowIndex * cols * sizeof(float))),
            cols).CopyTo(row);
        return row;
    }

    // ── Fixture (trimmed copy of MoteBitNetMoeLoaderScaffoldTests.BuildBitNetMoeFixture, single
    // MoE layer + one forced-dense layer) ──
    private string BuildBitNetMoeFixture(bool withGateBias, int seed)
    {
        var rng = new Random(seed);
        var b = new SafetensorsFixtureBuilder();
        b.AddFloat32("model.embed_tokens.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));
        b.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));
        b.AddFloat32("lm_head.weight", [Vocab, Hidden], RandomVec(rng, Vocab * Hidden, 0.05f));

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;
        for (int i = 0; i < 2; i++)
        {
            string p = $"model.layers.{i}";
            b.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            b.AddFloat32($"{p}.self_attn.q_proj.weight", [qDim, Hidden], RandomVec(rng, qDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.k_proj.weight", [kvDim, Hidden], RandomVec(rng, kvDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.v_proj.weight", [kvDim, Hidden], RandomVec(rng, kvDim * Hidden, 0.05f));
            b.AddFloat32($"{p}.self_attn.o_proj.weight", [Hidden, qDim], RandomVec(rng, Hidden * qDim, 0.05f));
            b.AddFloat32($"{p}.self_attn.attn_sub_norm.weight", [Hidden], Ones(Hidden));

            if (i == 1)
            {
                // BitNet-MoE FFN: router + per-expert ternary {gate,up,down}_proj + ffn_sub_norm.
                b.AddFloat32($"{p}.mlp.gate.weight", [NumExperts, Hidden], RandomVec(rng, NumExperts * Hidden, 0.05f));
                if (withGateBias)
                    b.AddFloat32($"{p}.mlp.gate.bias", [NumExperts], RandomVec(rng, NumExperts, 0.1f));
                for (int e = 0; e < NumExperts; e++)
                {
                    b.AddFloat32($"{p}.mlp.experts.{e}.gate_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                    b.AddFloat32($"{p}.mlp.experts.{e}.up_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                    b.AddFloat32($"{p}.mlp.experts.{e}.down_proj.weight", [Hidden, Intermediate], RandomVec(rng, Hidden * Intermediate, 0.05f));
                    b.AddFloat32($"{p}.mlp.experts.{e}.ffn_sub_norm.weight", [Intermediate], Ones(Intermediate));
                }
            }
            else
            {
                // Dense BitNet FFN (forced dense via mlp_only_layers below).
                b.AddFloat32($"{p}.mlp.gate_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.up_proj.weight", [Intermediate, Hidden], RandomVec(rng, Intermediate * Hidden, 0.05f));
                b.AddFloat32($"{p}.mlp.down_proj.weight", [Hidden, Intermediate], RandomVec(rng, Hidden * Intermediate, 0.05f));
                b.AddFloat32($"{p}.mlp.ffn_sub_norm.weight", [Intermediate], Ones(Intermediate));
            }
        }

        string path = Path.Combine(_scratch, "bitnet-mote-cuda.safetensors");
        b.WriteTo(path);
        return path;
    }

    private static ModelConfig BuildBitNetMoeConfig()
        => new()
        {
            Architecture = Architecture.BitNet,
            ActivationFunction = ActivationFunction.ReluSquared,
            VocabSize = Vocab,
            HiddenSize = Hidden,
            IntermediateSize = Intermediate,
            NumLayers = 2,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 128,
            NormEpsilon = Eps,
            TiedEmbeddings = false,
            RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX),
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = Intermediate,
                NormTopKProb = true,
                DecoderSparseStep = 1,
                MlpOnlyLayers = [0],
            },
        };

    private static float[] RandomVec(Random rng, int n, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return v;
    }

    private static float[] Ones(int n)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = 1.0f;
        return v;
    }
}
