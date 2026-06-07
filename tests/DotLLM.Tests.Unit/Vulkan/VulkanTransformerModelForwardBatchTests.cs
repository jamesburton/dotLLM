using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity tests for the Phase 5f Vulkan <c>ForwardBatch</c> override
/// against per-seq Forward loop reference. Builds a tiny synthetic Llama-shape
/// dense GQA model (2 layers, hidden=64) and asserts that batched logits match
/// per-seq Forward outputs within the Vulkan F32 envelope (5e-3 abs / 1e-3 rel —
/// same bar as the existing <c>VulkanLoraForwardParityTests</c>).
/// </summary>
/// <remarks>
/// <para>
/// Tolerance: the batched path runs the same matmul kernels with seqLen
/// = Σ N_i instead of seqLen = N_i, so reduction-order drift between
/// <c>matmul_f32 [N, hidden] × [hidden, dim]</c> and <c>matmul_f32 [Σ N_i, hidden] × [hidden, dim]</c>
/// is bounded by the kernel's per-thread accumulation order — both
/// dispatch the same per-output-cell dot-product. Empirically logits match
/// well within the documented envelope.
/// </para>
/// <para>
/// Covers four cases:
/// <list type="number">
///   <item>Single sequence — ForwardBatch must equal Forward.</item>
///   <item>Two sequences with different prompts (mixed N_i ∈ {3, 5}).</item>
///   <item>Empty request list — ForwardBatch returns empty.</item>
///   <item>Mixed-length decode + prefill across 4 sequences (KvCache pre-seeded).</item>
/// </list>
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelForwardBatchTests : IDisposable
{
    private const int Hidden = 64;
    private const int NumHeads = 4;
    private const int HeadDim = 16;
    private const int IntermediateSize = 128;
    private const int VocabSize = 32;
    private const int NumLayers = 2;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly string _scratch;

    public VulkanTransformerModelForwardBatchTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-fwdbatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public unsafe void VulkanForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, "empty.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        using var sf = SafetensorsFile.Open(fixturePath);
        using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);

        var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1);
        Assert.NotNull(results);
        Assert.Empty(results);
    }

    [SkippableFact]
    public unsafe void VulkanForwardBatch_SingleSeq_EqualsForward_F32()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, "single.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        int[] tokenIds = [1, 5, 9, 3];
        int[] positions = [0, 1, 2, 3];

        // Reference: Forward with a single sequence (KvCache pre-seeded at position 0).
        float[] referenceLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var kv = model.CreateKvCache(cfg.MaxSequenceLength);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache: kv);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            referenceLogits = CopyLogits(logits);
        }

        // Under test: ForwardBatch with a single request.
        float[] batchedLogits;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var kv = model.CreateKvCache(cfg.MaxSequenceLength);
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokenIds.AsMemory(),
                    Positions = positions.AsMemory(),
                    KvCache = kv,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            Assert.Single(results);
            try
            {
                Assert.Equal(1, results[0].Shape[0]);
                Assert.Equal(VocabSize, results[0].Shape[1]);
                batchedLogits = CopyLogits(results[0]);
            }
            finally
            {
                foreach (var t in results) t.Dispose();
            }
        }

        AssertLogitsClose(referenceLogits, batchedLogits, "Single-seq batch vs Forward");
    }

    [SkippableFact]
    public unsafe void VulkanForwardBatch_TwoSeqs_DifferentPrompts_MatchesPerSeqLoop()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, "two-seqs.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        // Seq A: 3 tokens prefill from position 0
        int[] tokensA = [1, 4, 9];
        int[] positionsA = [0, 1, 2];
        // Seq B: 5 tokens prefill from position 0 with different content
        int[] tokensB = [7, 2, 11, 6, 3];
        int[] positionsB = [0, 1, 2, 3, 4];

        // Reference: two separate Forward calls on two separate models (clean state each).
        float[] referenceA, referenceB;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var kvA = model.CreateKvCache(cfg.MaxSequenceLength);
            using ITensor logitsA = model.Forward(tokensA, positionsA, deviceId: -1, kvCache: kvA);
            referenceA = CopyLogits(logitsA);

            using var kvB = model.CreateKvCache(cfg.MaxSequenceLength);
            using ITensor logitsB = model.Forward(tokensB, positionsB, deviceId: -1, kvCache: kvB);
            referenceB = CopyLogits(logitsB);
        }

        // Under test: ForwardBatch on the same two requests.
        float[] batchedA, batchedB;
        {
            using var sf = SafetensorsFile.Open(fixturePath);
            using var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir);
            using var kvA = model.CreateKvCache(cfg.MaxSequenceLength);
            using var kvB = model.CreateKvCache(cfg.MaxSequenceLength);
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(), KvCache = kvA,
                },
                new SequenceForwardRequest
                {
                    TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = kvB,
                },
            };
            var results = model.ForwardBatch(requests, deviceId: -1);
            try
            {
                Assert.Equal(2, results.Count);
                Assert.Equal(1, results[0].Shape[0]);
                Assert.Equal(VocabSize, results[0].Shape[1]);
                Assert.Equal(1, results[1].Shape[0]);
                Assert.Equal(VocabSize, results[1].Shape[1]);
                batchedA = CopyLogits(results[0]);
                batchedB = CopyLogits(results[1]);
            }
            finally
            {
                foreach (var t in results) t.Dispose();
            }
        }

        AssertLogitsClose(referenceA, batchedA, "Two-seq batch [A]");
        AssertLogitsClose(referenceB, batchedB, "Two-seq batch [B]");
    }

    [SkippableFact]
    public unsafe void VulkanForwardBatch_FourSeqs_MixedPrefillDecode_MatchesPerSeqLoop()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        string fixturePath = Path.Combine(_scratch, "four-seqs.safetensors");
        WriteSyntheticFixture(fixturePath);
        var cfg = BuildConfig();

        // Four sequences with different lengths and starting positions. Two of them
        // (S1, S3) are decode-step seqs (N_i=1) which read a pre-seeded KvCache; the
        // other two (S0, S2) are prefill seqs (N_i > 1) starting at position 0.
        // Token IDs are chosen distinct per seq to ensure the batched path doesn't
        // silently cross-contaminate.
        int[][] tokens =
        [
            [3, 5, 7],         // S0: prefill, 3 tokens
            [11],              // S1: decode, 1 token
            [2, 4, 6, 8, 10],  // S2: prefill, 5 tokens
            [1, 9],            // S3: prefill, 2 tokens
        ];
        int[][] positions =
        [
            [0, 1, 2],
            [4],               // S1 decode at position 4 — cache pre-seeded with positions [0..3]
            [0, 1, 2, 3, 4],
            [0, 1],
        ];

        // Helper: build a KvCache and prefill it to position[i][0]-1 if positive.
        // Per-seq logits via per-seq Forward.
        float[][] referenceLogits = new float[tokens.Length][];
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir))
        {
            for (int s = 0; s < tokens.Length; s++)
            {
                using var kv = model.CreateKvCache(cfg.MaxSequenceLength);
                int firstPos = positions[s][0];
                if (firstPos > 0)
                {
                    // Pre-seed with a synthetic warm-up sequence at positions [0..firstPos-1].
                    int[] warmupTokens = new int[firstPos];
                    int[] warmupPositions = new int[firstPos];
                    for (int t = 0; t < firstPos; t++) { warmupTokens[t] = (t * 7 + s) % VocabSize; warmupPositions[t] = t; }
                    using (model.Forward(warmupTokens, warmupPositions, deviceId: -1, kvCache: kv)) { }
                }
                using ITensor logits = model.Forward(tokens[s], positions[s], deviceId: -1, kvCache: kv);
                referenceLogits[s] = CopyLogits(logits);
            }
        }

        // Same logic but using ForwardBatch on a fresh model.
        float[][] batchedLogits = new float[tokens.Length][];
        using (var sf = SafetensorsFile.Open(fixturePath))
        using (var model = VulkanTransformerModel.LoadFromSafetensors(sf, cfg, spvDir))
        {
            // Each seq needs its own kvCache and matching prefill warm-up. Then run a single
            // ForwardBatch covering all four (matches the scheduler's continuous-batch
            // dispatch shape).
            var caches = new VulkanKvCache[tokens.Length];
            try
            {
                for (int s = 0; s < tokens.Length; s++)
                {
                    caches[s] = model.CreateKvCache(cfg.MaxSequenceLength);
                    int firstPos = positions[s][0];
                    if (firstPos > 0)
                    {
                        int[] warmupTokens = new int[firstPos];
                        int[] warmupPositions = new int[firstPos];
                        for (int t = 0; t < firstPos; t++) { warmupTokens[t] = (t * 7 + s) % VocabSize; warmupPositions[t] = t; }
                        using (model.Forward(warmupTokens, warmupPositions, deviceId: -1, kvCache: caches[s])) { }
                    }
                }

                var requests = new SequenceForwardRequest[tokens.Length];
                for (int s = 0; s < tokens.Length; s++)
                {
                    requests[s] = new SequenceForwardRequest
                    {
                        TokenIds = tokens[s].AsMemory(),
                        Positions = positions[s].AsMemory(),
                        KvCache = caches[s],
                    };
                }

                var results = model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    Assert.Equal(tokens.Length, results.Count);
                    for (int s = 0; s < tokens.Length; s++)
                    {
                        Assert.Equal(1, results[s].Shape[0]);
                        Assert.Equal(VocabSize, results[s].Shape[1]);
                        batchedLogits[s] = CopyLogits(results[s]);
                    }
                }
                finally
                {
                    foreach (var t in results) t.Dispose();
                }
            }
            finally
            {
                for (int s = 0; s < tokens.Length; s++) caches[s]?.Dispose();
            }
        }

        for (int s = 0; s < tokens.Length; s++)
        {
            AssertLogitsClose(referenceLogits[s], batchedLogits[s], $"Four-seq mixed batch [seq {s}, N={tokens[s].Length}]");
        }
    }

    private static void AssertLogitsClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        Assert.Equal(VocabSize, reference.Length);
        for (int c = 0; c < VocabSize; c++)
        {
            float r = reference[c];
            float a = actual[c];
            float diff = MathF.Abs(r - a);
            float bar = AbsTol + RelTol * MathF.Abs(r);
            Assert.True(diff <= bar,
                $"{label}: col={c}: reference={r:F6} vs actual={a:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig() => new()
    {
        Architecture = Architecture.Llama,
        VocabSize = VocabSize,
        HiddenSize = Hidden,
        IntermediateSize = IntermediateSize,
        NumLayers = NumLayers,
        NumAttentionHeads = NumHeads,
        NumKvHeads = NumHeads,
        HeadDim = HeadDim,
        MaxSequenceLength = 128,
        NormEpsilon = 1e-5f,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: HeadDim, Type: RoPEType.Norm),
    };

    private static void WriteSyntheticFixture(string path)
    {
        // Deterministic random fixture, same shape pattern as VulkanLoraForwardParityTests.
        var rng = new Random(42);
        var bld = new SafetensorsFixtureBuilder();
        bld.AddFloat32("model.embed_tokens.weight", [VocabSize, Hidden], RandomVec(rng, VocabSize * Hidden, 0.05f));
        bld.AddFloat32("model.norm.weight", [Hidden], Ones(Hidden));
        for (int i = 0; i < NumLayers; i++)
        {
            string p = $"model.layers.{i}";
            bld.AddFloat32($"{p}.input_layernorm.weight", [Hidden], Ones(Hidden));
            bld.AddFloat32($"{p}.post_attention_layernorm.weight", [Hidden], Ones(Hidden));
            bld.AddFloat32($"{p}.self_attn.q_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.k_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.v_proj.weight",
                [NumHeads * HeadDim, Hidden], RandomVec(rng, NumHeads * HeadDim * Hidden, 0.05f));
            bld.AddFloat32($"{p}.self_attn.o_proj.weight",
                [Hidden, NumHeads * HeadDim], RandomVec(rng, Hidden * NumHeads * HeadDim, 0.05f));
            bld.AddFloat32($"{p}.mlp.gate_proj.weight",
                [IntermediateSize, Hidden], RandomVec(rng, IntermediateSize * Hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.up_proj.weight",
                [IntermediateSize, Hidden], RandomVec(rng, IntermediateSize * Hidden, 0.05f));
            bld.AddFloat32($"{p}.mlp.down_proj.weight",
                [Hidden, IntermediateSize], RandomVec(rng, Hidden * IntermediateSize, 0.05f));
        }
        bld.AddFloat32("lm_head.weight", [VocabSize, Hidden], RandomVec(rng, VocabSize * Hidden, 0.05f));
        bld.WriteTo(path);
    }

    private static float[] RandomVec(Random rng, int n, float scale = 1.0f)
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
