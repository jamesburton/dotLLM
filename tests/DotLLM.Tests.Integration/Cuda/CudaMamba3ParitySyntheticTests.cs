using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// CPU-vs-CUDA forward parity for <see cref="CudaMamba3TransformerModel"/> (issue #346)
/// on a deterministic synthetic checkpoint — no external download required, runs in
/// every CI pass with a CUDA device. Mirrors
/// <c>IbSsmMamba3VulkanGenerationTests.Mamba3_VulkanForward_MatchesCpuReference_OnPromptPrefill</c>'s
/// comparison methodology (single one-shot prefill, last-token-focused tolerance —
/// see that class's remarks for why growing-context reprefill is NOT used for Mamba-3
/// parity tests: there is no public state-reset API this test needs to work around
/// here since both sides load fresh models per test).
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a second, two-chunk test exists.</b> Task 9's implementation (this plan's
/// forward-pass task) originally persisted the wrong tensor into the CUDA state cache's
/// K slot at the chunk boundary (C/"qRoped" instead of B/"kRoped" — a one-identifier
/// transcription slip caught and fixed only via deep review, see the plan's ledger).
/// That bug is invisible to a single-shot / single-chunk forward: the chunk-boundary
/// correction kernel only reads <c>k_state</c>/<c>v_state</c> at the START of a SECOND
/// call against the same <see cref="CudaMamba3StateCache"/> — a fresh cache's k/v state
/// is zero, so the correction term is zero regardless of which tensor a (hypothetical)
/// bug would have persisted after chunk 1. <see cref="CudaTwoChunkDecode_MatchesCpuReference_AndBoundaryContributionIsNonZero"/>
/// is therefore the dedicated regression gate: it runs two state-threaded calls through
/// <see cref="CudaMamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, CudaMamba3StateCache)"/>,
/// and — rather than just hoping the synthetic fixture happens to exercise the
/// <c>coef * kSum * v</c> boundary term — proves it does by an explicit ablation: cloning
/// the post-chunk-1 CUDA state, zeroing the clone's k_state/v_state (the "no persisted
/// boundary" condition), and asserting chunk 2's logits actually change when run against
/// the ablated clone vs. the real state. Only once that's established does it compare
/// chunk 2's real (non-ablated) logits against the CPU oracle running the identical
/// two-call schedule through <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMamba3ParitySyntheticTests : IDisposable
{
    private const int HiddenSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int HeadDim = 4;
    private const int Expand = 2;
    private const int StateSize = 8;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize;
    private const int NumRopeAngles = 2;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // Absolute logit tolerance for the single-shot comparison. NOT transplanted from
    // IbSsmMamba3VulkanGenerationTests' LogitsAbsTol=3.0f — that value is calibrated for
    // the REAL 370M checkpoint's O(10)-magnitude logits (paired with a top-K Jaccard
    // check, not used standalone), a ~300x different logit scale than this tiny
    // synthetic fixture's ~0.04-magnitude logits; an absolute tolerance does not survive
    // that scale change; 1e-2 would be ~25% of this fixture's own logit magnitude.
    // Calibrated instead from what this fixture actually produces: observed max_abs
    // 7.45E-9 pure F32 reduction-order noise (both backends run F32, no quantization),
    // tightened to 1e-6 (~130x margin). The mutation experiment for the two-chunk test
    // below (see TwoChunkLogitsAbsTol) showed the same drift order applies to the
    // single-shot path too, and that a loose absolute tolerance (the original 1e-2)
    // would silently pass a real regression of this fixture's own bug class — no reason
    // to leave this constant looser than the tightened one.
    private const float LogitsAbsTol = 1e-6f;

    // Floor for the two-chunk ablation delta (real state vs. k_state/v_state zeroed).
    // Both sides of this comparison are two deterministic CUDA runs on the same GPU —
    // there is no cross-backend or cross-run noise to hide behind, so any nonzero delta
    // is a genuine effect of the ablation, not F32 jitter. This tiny synthetic fixture's
    // tiny weight amplitudes (0.02-0.5) mean the coef*kSum*v term is itself small once
    // it has propagated through out_proj/residual/final-norm/lm_head — observed
    // max_abs=4.99E-7 on the (chunk1=3, chunk2=2) split used below (5+ orders of
    // magnitude above the ~1e-13 delta a truly no-op boundary correction would produce
    // from residual float non-associativity). Floor set ~5x below the observed value.
    private const float BoundaryContributionFloor = 1e-7f;

    // Tighter, separate tolerance for the two-chunk CPU-vs-CUDA comparison. The plain
    // 1e-2 LogitsAbsTol above is calibrated for the single-shot path, which never
    // exercises the chunk-boundary correction at all — far too loose to be a real gate
    // here, where the whole coef*kSum*v term this test exists to guard is itself only
    // ~5E-7 in final-logit terms (see BoundaryContributionFloor). Empirically verified
    // by temporarily reintroducing Task 9's original bug (persisting C/"qRoped" instead
    // of B/"kRoped" into kState — a one-line change in the C# launcher, not the .cu
    // kernel): correct code drifts 7.45E-9 vs the CPU oracle (same F32 noise floor as
    // the single-shot test); the bugged code drifts 1.661E-6 — about 223x more, but
    // still far under 1e-2, i.e. 1e-2 would have let the regression back in silently.
    // 1E-7 sits ~13x above the correct-code noise floor and ~16x below the bugged-code
    // drift, so it passes the correct implementation with margin while still failing
    // on reintroduction of the original bug.
    private const float TwoChunkLogitsAbsTol = 1e-7f;

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaMamba3ParitySyntheticTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mamba3-parity-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnSyntheticCheckpoint()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMamba3Checkpoint(modelPath, configPath);
        _output.WriteLine($"Synthesised tiny Mamba-3 checkpoint at: {modelPath}");

        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            int[] tokenIds = [0, 1, 2, 3, 5];
            int[] positions = [0, 1, 2, 3, 4];

            using ITensor cpuLogits = cpuModel.Forward(tokenIds, positions, deviceId: -1);
            using ITensor cudaLogits = cudaModel.Forward(tokenIds, positions, deviceId: -1);

            // CPU returns [seqLen, vocab] (per-position); CUDA returns [1, vocab]
            // (last token only) — same shape contract Vulkan uses. Compare the last row.
            Assert.Equal(2, cpuLogits.Shape.Rank);
            Assert.Equal(2, cudaLogits.Shape.Rank);
            Assert.Equal(tokenIds.Length, cpuLogits.Shape[0]);
            Assert.Equal(1, cudaLogits.Shape[0]);
            Assert.Equal(VocabSize, cpuLogits.Shape[1]);
            Assert.Equal(VocabSize, cudaLogits.Shape[1]);

            float[] cpuLast = ExtractRow(cpuLogits, tokenIds.Length - 1, VocabSize);
            float[] cudaLast = ExtractRow(cudaLogits, 0, VocabSize);

            float maxAbs = 0f;
            int worstIdx = 0;
            for (int i = 0; i < VocabSize; i++)
            {
                float diff = MathF.Abs(cpuLast[i] - cudaLast[i]);
                if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
            }
            _output.WriteLine(
                $"Last-token logit drift: max_abs={maxAbs:E3} at idx {worstIdx} "
                + $"(cpu={cpuLast[worstIdx]:G6} cuda={cudaLast[worstIdx]:G6})");

            int cpuArg = ArgMax(cpuLast);
            int cudaArg = ArgMax(cudaLast);
            _output.WriteLine($"Argmax: cpu={cpuArg} cuda={cudaArg}");

            Assert.True(maxAbs <= LogitsAbsTol,
                $"Last-token logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} at idx {worstIdx}.");
            Assert.Equal(cpuArg, cudaArg);
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    /// <summary>
    /// Two-chunk state-threaded decode parity — the dedicated regression gate for Task 9's
    /// chunk-boundary K/V persistence bug (see class remarks). Runs chunk 1 (primes
    /// <c>k_state</c>/<c>v_state</c>) then chunk 2 (reads them back via the chunk-boundary
    /// correction kernel) through <see cref="CudaMamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, CudaMamba3StateCache)"/>
    /// on the CUDA side and the equivalent <see cref="Mamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, Mamba3State)"/>
    /// schedule on the CPU side, then asserts:
    /// <list type="number">
    ///   <item><description>
    ///     The boundary term is actually exercised by this fixture — proven by an explicit
    ///     ablation (chunk 2 run again from a state clone with k_state/v_state forced to
    ///     zero) rather than assumed.
    ///   </description></item>
    ///   <item><description>
    ///     CUDA's (non-ablated) chunk-2 last-token logits match the CPU oracle's, within the
    ///     same documented F32-reduction-order tolerance as the single-shot test.
    ///   </description></item>
    /// </list>
    /// </summary>
    [SkippableFact]
    public void CudaTwoChunkDecode_MatchesCpuReference_AndBoundaryContributionIsNonZero()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        // ModelLoader.OpenSafetensorsAndConfig hardcodes the sibling filename to
        // "config.json" (derived from the weights directory, not from any path we
        // pass) — must match exactly, same as the single-shot test above. Each test
        // method gets its own `_scratch` instance (xUnit constructs a fresh test
        // class per [Fact]), so there's no collision with the other test's files.
        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMamba3Checkpoint(modelPath, configPath);
        _output.WriteLine($"Synthesised tiny Mamba-3 checkpoint at: {modelPath}");

        var (cpuModelBase, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        // Both ModelLoader.LoadFromSafetensors and CudaModelLoader.LoadMamba3FromSafetensors
        // dispatch Architecture.Mamba3 exclusively to these concrete types — the cast is safe
        // and needed for the Mamba3State-typed / CudaMamba3StateCache-typed overloads, which
        // are not on the IModel interface (Mamba3TransformerModelDecodeTests uses the same
        // concrete-type pattern for the CPU side).
        var cpuModel = (Mamba3TransformerModel)cpuModelBase;
        Mamba3Config m3 = config.Mamba3Config!;

        try
        {
            int[] tokens = [0, 1, 2, 3, 5];
            int[] positions = [0, 1, 2, 3, 4];
            ReadOnlySpan<int> chunk1Tokens = tokens.AsSpan(0, 3);
            ReadOnlySpan<int> chunk1Positions = positions.AsSpan(0, 3);
            ReadOnlySpan<int> chunk2Tokens = tokens.AsSpan(3, 2);
            ReadOnlySpan<int> chunk2Positions = positions.AsSpan(3, 2);

            // ---- CPU reference: prefill chunk 1, decode chunk 2 through the same state. ----
            using var cpuState = new Mamba3State(config);
            using (ITensor cpuChunk1 = cpuModel.Forward(chunk1Tokens, chunk1Positions, -1, cpuState))
            {
                // Discarded — only used to advance cpuState.
            }
            using ITensor cpuChunk2 = cpuModel.Forward(chunk2Tokens, chunk2Positions, -1, cpuState);
            float[] cpuChunk2Last = ExtractRow(cpuChunk2, chunk2Tokens.Length - 1, VocabSize);

            // ---- CUDA: identical split through CudaMamba3StateCache. ----
            using var cudaState = new CudaMamba3StateCache(m3, config.NumLayers);
            using (ITensor cudaChunk1 = cudaModel.Forward(chunk1Tokens, chunk1Positions, -1, cudaState))
            {
                // Discarded — only used to advance cudaState (writes k_state/v_state at
                // chunk 1's last token — the exact line Task 9's bug got wrong).
            }

            // Snapshot post-chunk-1 state, then build an ABLATED clone whose k_state/v_state
            // are forced to zero — the same "no persisted boundary" starting condition as a
            // brand-new cache. Comparing chunk 2 run from the real state vs. the ablated clone
            // isolates the chunk-boundary kernel's contribution.
            using CudaMamba3StateCache ablated = cudaState.Clone();
            for (int layer = 0; layer < config.NumLayers; layer++)
            {
                CudaDriverApi.cuMemsetD8_v2(ablated.GetKStatePtr(layer), 0,
                    (nuint)(ablated.KStateElementsPerLayer * sizeof(float))).ThrowOnError();
                CudaDriverApi.cuMemsetD8_v2(ablated.GetVStatePtr(layer), 0,
                    (nuint)(ablated.VStateElementsPerLayer * sizeof(float))).ThrowOnError();
            }

            using ITensor cudaChunk2 = cudaModel.Forward(chunk2Tokens, chunk2Positions, -1, cudaState);
            using ITensor cudaChunk2Ablated = cudaModel.Forward(chunk2Tokens, chunk2Positions, -1, ablated);

            float[] cudaChunk2Last = ExtractRow(cudaChunk2, 0, VocabSize);
            float[] cudaChunk2AblatedLast = ExtractRow(cudaChunk2Ablated, 0, VocabSize);

            // --- Assertion 1: the fixture actually exercises the boundary term on CUDA. ---
            float ablationMaxAbs = 0f;
            for (int i = 0; i < VocabSize; i++)
                ablationMaxAbs = MathF.Max(ablationMaxAbs, MathF.Abs(cudaChunk2Last[i] - cudaChunk2AblatedLast[i]));
            _output.WriteLine($"Chunk-2 boundary-term ablation delta (real vs k/v-zeroed state): max_abs={ablationMaxAbs:E3}");
            Assert.True(ablationMaxAbs > BoundaryContributionFloor,
                $"Zeroing k_state/v_state before chunk 2 changed logits by only {ablationMaxAbs:E3} "
                + $"(floor {BoundaryContributionFloor:E1}) — this fixture does not exercise the "
                + "chunk-boundary coef*kSum*v term, so this test cannot discriminate Task 9's "
                + "B-vs-C persistence bug from a correct implementation.");

            // --- Assertion 2: CPU and CUDA agree on chunk 2's (non-trivial) result. ---
            float maxAbs = 0f;
            int worstIdx = 0;
            for (int i = 0; i < VocabSize; i++)
            {
                float diff = MathF.Abs(cpuChunk2Last[i] - cudaChunk2Last[i]);
                if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
            }
            _output.WriteLine(
                $"Two-chunk decode chunk-2 last-token drift: max_abs={maxAbs:E3} at idx {worstIdx} "
                + $"(cpu={cpuChunk2Last[worstIdx]:G6} cuda={cudaChunk2Last[worstIdx]:G6})");

            int cpuArg = ArgMax(cpuChunk2Last);
            int cudaArg = ArgMax(cudaChunk2Last);
            _output.WriteLine($"Argmax: cpu={cpuArg} cuda={cudaArg}");

            Assert.True(maxAbs <= TwoChunkLogitsAbsTol,
                $"Two-chunk decode chunk-2 divergence {maxAbs:G6} > {TwoChunkLogitsAbsTol:G4} at idx {worstIdx}.");
            Assert.Equal(cpuArg, cudaArg);
        }
        finally
        {
            cudaModel.Dispose();
            cudaSource.Dispose();
            cpuModel.Dispose();
            cpuFile.Dispose();
        }
    }

    private static unsafe float[] ExtractRow(ITensor logits, int row, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, logits.Shape[0] * vocabSize);
        float[] result = new float[vocabSize];
        span.Slice(row * vocabSize, vocabSize).CopyTo(result);
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

    // ------------------------------------------------------------------
    // Fixture synthesis — same shape tuple and write strategy as
    // TinyMamba3SafetensorsLoadTests.WriteSyntheticMamba3Checkpoint.
    // ------------------------------------------------------------------

    private static void WriteSyntheticMamba3Checkpoint(string safetensorsPath, string configPath)
    {
        WriteConfigJson(configPath);
        WriteSafetensorsFixture(safetensorsPath);
    }

    /// <summary>
    /// Reusable entry point for other test classes (see
    /// <see cref="CudaUnsupportedArchitectureGuardTests.LoadFromSafetensors_Mamba3Checkpoint_ThrowsNotSupportedPointingAtDedicatedLoader"/>)
    /// that need the smallest fixture resolving to <c>Architecture.Mamba3</c>, without
    /// duplicating the synthetic-checkpoint writer a third time. No behavior change from
    /// <see cref="WriteSyntheticMamba3Checkpoint"/> — thin wrapper for call-site clarity.
    /// </summary>
    internal static void WriteMinimalMamba3CheckpointForGuardTest(string safetensorsPath, string configPath)
        => WriteSyntheticMamba3Checkpoint(safetensorsPath, configPath);

    private static void WriteConfigJson(string path)
    {
        using var fs = File.Create(path);
        using var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteString("model_type", "mamba3");
        writer.WriteNumber("hidden_size", HiddenSize);
        writer.WriteNumber("vocab_size", VocabSize);
        writer.WriteNumber("num_hidden_layers", NumLayers);
        writer.WriteNumber("num_heads", NumHeads);
        writer.WriteNumber("head_dim", HeadDim);
        writer.WriteNumber("expand", Expand);
        writer.WriteNumber("n_groups", 1);
        writer.WriteNumber("state_size", StateSize);
        writer.WriteNumber("chunk_size", 2);
        writer.WriteNumber("mimo_rank", 1);
        writer.WriteBoolean("is_mimo", false);
        writer.WriteBoolean("is_outproj_norm", false);
        writer.WriteBoolean("use_l2warp", false);
        writer.WriteBoolean("tie_word_embeddings", false);
        writer.WriteBoolean("rescale_prenorm_residual", true);
        writer.WriteBoolean("residual_in_fp32", true);
        writer.WriteNumber("A_floor", 1e-4);
        writer.WriteNumber("dt_init_floor", 1e-4);
        writer.WriteNumber("dt_min", 1e-3);
        writer.WriteNumber("dt_max", 0.1);
        writer.WriteNumber("norm_eps", 1e-5);
        writer.WriteNumber("rope_fraction", 0.5);
        writer.WriteNumber("max_position_embeddings", 32);
        writer.WriteEndObject();
    }

    private static void WriteSafetensorsFixture(string path)
    {
        var tensors = new List<(string Name, int[] Shape, float[] Values)>();

        AddSmall(tensors, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], amplitude: 0.05f, seed: 0);
        AddSmall(tensors, Mamba3TensorMapping.FinalNorm, [HiddenSize], amplitude: 0.5f, seed: 1);
        AddSmall(tensors, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], amplitude: 0.05f, seed: 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            AddSmall(tensors, Mamba3TensorMapping.LayerNorm(i), [HiddenSize], amplitude: 0.5f, seed: sBase + 0);
            AddSmall(tensors, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], amplitude: 0.02f, seed: sBase + 1);
            AddSmall(tensors, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], amplitude: 0.05f, seed: sBase + 2);
            AddSmall(tensors, Mamba3TensorMapping.BNorm(i), [StateSize], amplitude: 0.5f, seed: sBase + 3);
            AddSmall(tensors, Mamba3TensorMapping.CNorm(i), [StateSize], amplitude: 0.5f, seed: sBase + 4);
            AddSmall(tensors, Mamba3TensorMapping.BBias(i), [NumHeads, 1, StateSize], amplitude: 0.02f, seed: sBase + 5);
            AddSmall(tensors, Mamba3TensorMapping.CBias(i), [NumHeads, 1, StateSize], amplitude: 0.02f, seed: sBase + 6);
            AddSmall(tensors, Mamba3TensorMapping.D(i), [NumHeads], amplitude: 0.1f, seed: sBase + 7);
            AddSmall(tensors, Mamba3TensorMapping.DtBias(i), [NumHeads], amplitude: 0.02f, seed: sBase + 8);
        }

        WriteSafetensorsFile(path, tensors);
    }

    private static void AddSmall(List<(string, int[], float[])> sink, string name, int[] shape, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
        }
        sink.Add((name, shape, values));
    }

    private static void WriteSafetensorsFile(string path, List<(string Name, int[] Shape, float[] Values)> tensors)
    {
        using var headerMs = new MemoryStream();
        using (var w = new Utf8JsonWriter(headerMs, new JsonWriterOptions { Indented = false }))
        {
            w.WriteStartObject();
            long offset = 0;
            foreach (var (name, shape, values) in tensors)
            {
                long byteLen = values.Length * sizeof(float);
                w.WriteStartObject(name);
                w.WriteString("dtype", "F32");
                w.WritePropertyName("shape");
                w.WriteStartArray();
                foreach (int d in shape) w.WriteNumberValue(d);
                w.WriteEndArray();
                w.WritePropertyName("data_offsets");
                w.WriteStartArray();
                w.WriteNumberValue(offset);
                w.WriteNumberValue(offset + byteLen);
                w.WriteEndArray();
                w.WriteEndObject();
                offset += byteLen;
            }
            w.WriteEndObject();
        }
        byte[] headerBytes = headerMs.ToArray();

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        Span<byte> prefix = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(prefix, (ulong)headerBytes.Length);
        fs.Write(prefix);
        fs.Write(headerBytes);

        foreach (var (_, _, values) in tensors)
        {
            byte[] bytes = new byte[values.Length * sizeof(float)];
            for (int i = 0; i < values.Length; i++)
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
            fs.Write(bytes);
        }
    }
}
