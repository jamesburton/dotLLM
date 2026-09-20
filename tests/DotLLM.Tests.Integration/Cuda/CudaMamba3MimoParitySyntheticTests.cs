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
/// CPU-vs-CUDA forward parity for <see cref="CudaMamba3TransformerModel"/>'s MIMO path
/// (issue #346, Task 14) on a deterministic synthetic checkpoint — synthetic-only,
/// matching CPU/Vulkan's own MIMO coverage (no public MIMO checkpoint exists anywhere
/// in the codebase; see docs/ROADMAP.md step 60f). Mirrors
/// <see cref="CudaMamba3ParitySyntheticTests"/>'s SISO test structure exactly.
/// </summary>
/// <remarks>
/// <b>Why a second, two-chunk test exists.</b> This task's own brief reintroduced the
/// exact same B-vs-C persistence bug Task 9's SISO brief originally had (see
/// <see cref="CudaMamba3ParitySyntheticTests"/>'s class remarks and this plan's
/// progress.md): the MIMO chunk-boundary K-state persist step copied <c>_cDevice</c>
/// (C/"qRoped") into the state cache's K slot instead of <c>_bDevice</c> (B/"kRoped") —
/// confirmed against <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>'s persist step (the
/// <c>kRoped</c> parameter is bound to <c>bRHN</c> at the <c>ForwardMimo</c> call site,
/// Mamba3Block.cs:680) and fixed before commit. Per Task 11's SISO precedent, a
/// single-shot / single-chunk forward CANNOT discriminate this bug class — a fresh
/// state's k_state/v_state are zero, so the chunk-boundary correction term is zero
/// regardless of which tensor a (hypothetical) bug persisted after chunk 1.
/// <see cref="CudaTwoChunkMimoDecode_MatchesCpuReference_AndBoundaryContributionIsNonZero"/>
/// is the dedicated MIMO regression gate: two state-threaded calls through
/// <see cref="CudaMamba3TransformerModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, CudaMamba3StateCache)"/>,
/// with an explicit ablation (clone the post-chunk-1 CUDA state, zero the clone's
/// k_state/v_state, assert chunk 2's logits actually change) proving the fixture
/// exercises the boundary term before comparing chunk 2's real logits against the CPU
/// oracle running the identical two-call schedule.
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMamba3MimoParitySyntheticTests : IDisposable
{
    // Issue #385: widened from the degenerate SISO-inherited tuple
    // (NumHeads==HeadDim==4, HiddenSize==StateSize==8) so ALL of
    // NumHeads/HeadDim/StateSize/HiddenSize/MimoRank are pairwise distinct —
    // an H<->P (head/headDim) or a hidden<->state axis transposition now
    // produces an out-of-bounds index or a wrong-shape read instead of
    // silently staying in bounds. Expand dropped 2->1 to keep the loader's
    // hard invariant `num_heads*head_dim == expand*hidden_size` satisfied
    // (Mamba3ConfigExtractor.cs:97-100): NumHeads*HeadDim=12, Expand*HiddenSize=1*12=12.
    // HeadDim kept even (kernel-level MIMO SSD scan test uses headDim=64;
    // no known odd-headDim vectorization path is exercised anywhere else).
    private const int HiddenSize = 12;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumHeads = 2;
    private const int HeadDim = 6;
    private const int Expand = 1;
    private const int StateSize = 16;
    private const int MimoRank = 3;
    private const int DInner = NumHeads * HeadDim;
    private const int BcDim = StateSize * MimoRank;
    // int(state_size * rope_fraction=0.5) / 2 = int(16*0.5)/2 = 4 (Mamba3Config.NumRopeAngles).
    // Guard: 2*NumRopeAngles=8 <= StateSize=16 (LaunchMamba3DataRopeF32's second guard).
    private const int NumRopeAngles = 4;
    private const int DInProj = 2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;

    // Absolute logit tolerance for the single-shot comparison. Recalibrated for issue
    // #385's widened fixture (NumHeads=2/HeadDim=6/StateSize=16/HiddenSize=12/MimoRank=3,
    // all pairwise distinct — see field comments above) — NOT transplanted from the
    // pre-#385 degenerate-dims value (see Task 11's REVIEW fix round for why a
    // transplanted/uncalibrated constant is exactly the failure mode this project has
    // already shipped once). Observed max_abs = 5.588E-9 (RTX 3060, this fixture/tokens/
    // positions, 2026-08-14). Both backends run F32 (no quantization), so the residual
    // drift is pure reduction-order noise through the MIMO rank-sum path — same order of
    // magnitude as before widening (7.451E-9) and as CudaMamba3ParitySyntheticTests' SISO
    // LogitsAbsTol=1e-6. Kept at 1e-6 (~179x margin over observed).
    private const float LogitsAbsTol = 1e-6f;

    // Floor for the two-chunk ablation delta (real state vs. k_state/v_state zeroed).
    // Both sides of this comparison are two deterministic CUDA runs on the same GPU — no
    // cross-backend/cross-run noise to hide behind, so any nonzero delta is a genuine
    // effect of the ablation. Recalibrated for issue #385's widened fixture: observed
    // max_abs = 5.141E-7 (RTX 3060, 2026-08-14; was 4.470E-8 pre-widening — the larger
    // dims/rank give the rank-summed coef*kSum*v term more terms to sum, so a larger
    // boundary contribution is expected). Set ~5x below that (1e-7), matching
    // CudaMamba3ParitySyntheticTests.BoundaryContributionFloor's ~5x-below calibration
    // approach — still >13,000,000x above the ~1e-13 no-op-boundary float-non-associativity
    // floor the SISO comment cites, so this remains a real, non-noise gate.
    private const float BoundaryContributionFloor = 1e-7f;

    // Tighter, separate tolerance for the two-chunk CPU-vs-CUDA comparison — the MIMO
    // analog of CudaMamba3ParitySyntheticTests.TwoChunkLogitsAbsTol. The plain
    // LogitsAbsTol above never exercises the chunk-boundary correction (single-shot has
    // no persisted state) so is far too loose to be a real gate for the coef*kSum*v
    // (rank-summed) term this test exists to guard. Recalibrated for issue #385's widened
    // fixture: observed max_abs = 5.588E-9 (RTX 3060, 2026-08-14 — same order as the
    // single-shot path; the MIMO chunk-boundary term reproduces CPU exactly here). Kept at
    // 1e-7 (~18x margin over observed), matching Task 11's SISO TwoChunkLogitsAbsTol.
    // Re-verified by mutation (2026-08-14, reintroducing the B-vs-C chunk-boundary persist
    // swap at CudaMamba3TransformerModel.cs's MIMO branch, temporarily then reverted):
    // like the pre-widening fixture, this logit-level comparison does NOT reliably
    // discriminate the swap even at the new (wider, non-degenerate) dims — the boundary
    // term's own contribution to final logits is only ~5e-7 (see BoundaryContributionFloor)
    // and gets attenuated through out_proj/residual/final-norm/lm_head. KStateAbsTol below
    // is the assertion that actually catches that bug class (mutation-verified below) —
    // this constant is kept at the SISO-precedented value for logit-scale regressions
    // unrelated to the boundary term.
    private const float TwoChunkLogitsAbsTol = 1e-7f;

    // Tolerance for the direct raw k_state comparison (see class remarks). This buffer
    // holds post-RoPE B values (O(0.02)-to-O(0.7) magnitude after RMSNorm+bias+RoPE)
    // computed independently by CPU (Mamba3Block.ForwardMimo host math +
    // Mamba3DataRoPE.ExecuteCanonical) and CUDA (HostPrepareMimo host math +
    // mamba3_data_rope_f32.cu, an independent expf/cosf/sinf-based kernel) — a looser
    // tolerance than the logit-scale ones above because this is comparing raw
    // pre-out_proj/pre-norm values, not attenuated logits. Recalibrated for issue #385's
    // widened fixture: observed max_abs on the correct implementation (RTX 3060,
    // 2026-08-14) = 2.235E-7 (layer 0) / 3.278E-7 (layer 1) — pure F32 reduction-order
    // noise (was 1.192E-7 / 8.941E-8 pre-widening; larger StateSize/MimoRank sum more
    // terms so a modest increase is expected). Set to 3e-4 (~915x margin over the worst
    // observed 3.278E-7), matching the pre-widening ~840x-margin calibration approach.
    // MUTATION-VERIFIED (2026-08-14): reintroducing the B-vs-C swap at
    // CudaMamba3TransformerModel.cs's MIMO chunk-boundary persist (`_bDevice` ->
    // `_cDevice` at the `lastKSrc` assignment, temporarily then reverted — see class
    // remarks / Mamba3Block.cs:680's kRoped=bRHN binding) changes this to 7.347E-1
    // (layer 0), ~2450x over this tolerance and ~3.3M x above the correct-code noise
    // floor — this assertion, NOT the logit-level ones above, is what actually catches
    // that bug class for this fixture.
    private const float KStateAbsTol = 3e-4f;

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaMamba3MimoParitySyntheticTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mamba3-mimo-parity-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void CudaForward_MatchesCpuReference_OnSyntheticMimoCheckpoint()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMimoCheckpoint(modelPath, configPath);
        _output.WriteLine($"Synthesised tiny Mamba-3 MIMO checkpoint at: {modelPath}");

        var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            Assert.True(config.Mamba3Config!.IsMimo);
            AssertDerivedDims(config.Mamba3Config);

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
                $"MIMO last-token logit drift: max_abs={maxAbs:E3} at idx {worstIdx} "
                + $"(cpu={cpuLast[worstIdx]:G6} cuda={cudaLast[worstIdx]:G6})");

            int cpuArg = ArgMax(cpuLast);
            int cudaArg = ArgMax(cudaLast);
            _output.WriteLine($"Argmax: cpu={cpuArg} cuda={cudaArg}");

            Assert.True(maxAbs <= LogitsAbsTol,
                $"MIMO last-token logit divergence {maxAbs:G6} > {LogitsAbsTol:G4} at idx {worstIdx}.");
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
    /// Two-chunk state-threaded decode parity — the dedicated MIMO regression gate for
    /// this task's rediscovered chunk-boundary K/V persistence bug (see class remarks).
    /// Mirrors <see cref="CudaMamba3ParitySyntheticTests.CudaTwoChunkDecode_MatchesCpuReference_AndBoundaryContributionIsNonZero"/>
    /// exactly, on the MIMO checkpoint/model instead of SISO.
    /// </summary>
    [SkippableFact]
    public void CudaTwoChunkMimoDecode_MatchesCpuReference_AndBoundaryContributionIsNonZero()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        WriteSyntheticMimoCheckpoint(modelPath, configPath);
        _output.WriteLine($"Synthesised tiny Mamba-3 MIMO checkpoint at: {modelPath}");

        var (cpuModelBase, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
        var (cudaModel, cudaSource, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        var cpuModel = (Mamba3TransformerModel)cpuModelBase;
        Mamba3Config m3 = config.Mamba3Config!;
        Assert.True(m3.IsMimo);
        AssertDerivedDims(m3);

        try
        {
            int[] tokens = [0, 1, 2, 3, 5];
            int[] positions = [0, 1, 2, 3, 4];
            ReadOnlySpan<int> chunk1Tokens = tokens.AsSpan(0, 3);
            ReadOnlySpan<int> chunk1Positions = positions.AsSpan(0, 3);
            ReadOnlySpan<int> chunk2Tokens = tokens.AsSpan(3, 2);
            ReadOnlySpan<int> chunk2Positions = positions.AsSpan(3, 2);

            // ---- CPU reference: prefill chunk 1. ----
            using var cpuState = new Mamba3State(config);
            using (ITensor cpuChunk1 = cpuModel.Forward(chunk1Tokens, chunk1Positions, -1, cpuState))
            {
                // Discarded — only used to advance cpuState.
            }

            // ---- CUDA: identical chunk 1 through CudaMamba3StateCache (rank-strided K-state). ----
            using var cudaState = new CudaMamba3StateCache(m3, config.NumLayers);
            using (ITensor cudaChunk1 = cudaModel.Forward(chunk1Tokens, chunk1Positions, -1, cudaState))
            {
                // Discarded — only used to advance cudaState (writes rank-strided
                // k_state/v_state at chunk 1's last token — exactly where a MIMO-specific
                // stride bug in the [R, H, N] persist would hide).
            }

            // --- Direct k_state comparison, BEFORE either side's chunk 2 runs. ---
            // MUST happen here, before cpuState/cudaState advance to chunk 2 (a second
            // Forward call re-persists a NEW k_state for its own last token) — comparing
            // after chunk 2 would silently diff two different generations of the buffer.
            // This task's own brief reintroduced Task 9's B-vs-C chunk-boundary persist
            // bug (see class remarks) in the MIMO branch, fixed before commit. Empirically,
            // that bug does NOT reliably show up as a final-logit divergence on this tiny
            // fixture: the chunk-boundary term's own contribution to the final logits is
            // itself only ~1e-8 (see BoundaryContributionFloor/the ablation assertion
            // below), so swapping which per-rank tensor (B vs C, both O(0.02)-magnitude)
            // feeds it perturbs an already-tiny term by a comparable fraction — landing
            // within the same order as this fixture's own F32 reduction-order noise floor
            // (~7e-9) after propagating through out_proj/residual/final-norm/lm_head. A
            // direct comparison of the RAW persisted k_state buffer (pre-attenuation)
            // against the CPU oracle's own k_state is therefore the only test in this file
            // that reliably discriminates a wrong-tensor (or wrong-rank-stride) persist —
            // MUTATION-VERIFIED (see this task's report): reintroducing the C-vs-B swap
            // changes this raw comparison from ~1e-7 (F32 RoPE/host-math noise) to ~0.46
            // (B_bias/C_bias have independent, non-matching amplitudes/seeds).
            for (int layer = 0; layer < config.NumLayers; layer++)
            {
                int kElems = m3.MimoRank * NumHeads * StateSize; // [R, H, N] — matches Mamba3State.KState's layout.
                Assert.Equal(kElems, cudaState.KStateElementsPerLayer);
                float[] cudaKState = DownloadF32(cudaState.GetKStatePtr(layer), kElems);
                ReadOnlySpan<float> cpuKState = cpuState.KState(layer);
                Assert.Equal(kElems, cpuKState.Length);

                float kMaxAbs = 0f;
                for (int i = 0; i < kElems; i++)
                    kMaxAbs = MathF.Max(kMaxAbs, MathF.Abs(cpuKState[i] - cudaKState[i]));
                _output.WriteLine($"Layer {layer} persisted MIMO k_state drift (raw, pre-attenuation): max_abs={kMaxAbs:E3}");
                Assert.True(kMaxAbs <= KStateAbsTol,
                    $"Layer {layer} persisted k_state divergence {kMaxAbs:G6} > {KStateAbsTol:G4} — "
                    + "CUDA's chunk-boundary K persist does not match the CPU oracle's k_state.");
            }

            // ---- CPU reference: decode chunk 2 through the same (now chunk-1-primed) state. ----
            using ITensor cpuChunk2 = cpuModel.Forward(chunk2Tokens, chunk2Positions, -1, cpuState);
            float[] cpuChunk2Last = ExtractRow(cpuChunk2, chunk2Tokens.Length - 1, VocabSize);

            // Snapshot post-chunk-1 state, then build an ABLATED clone whose k_state/v_state
            // are forced to zero — the same "no persisted boundary" starting condition as a
            // brand-new cache. Comparing chunk 2 run from the real state vs. the ablated clone
            // isolates the chunk-boundary kernel's rank-summed contribution.
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
            _output.WriteLine($"Chunk-2 MIMO boundary-term ablation delta (real vs k/v-zeroed state): max_abs={ablationMaxAbs:E3}");
            Assert.True(ablationMaxAbs > BoundaryContributionFloor,
                $"Zeroing k_state/v_state before chunk 2 changed logits by only {ablationMaxAbs:E3} "
                + $"(floor {BoundaryContributionFloor:E1}) — this fixture does not exercise the MIMO "
                + "chunk-boundary rank-summed coef*kSum*v term, so this test cannot discriminate a "
                + "B-vs-C (or rank-stride) persistence bug from a correct implementation.");

            // --- Assertion 2: CPU and CUDA agree on chunk 2's (non-trivial) result. ---
            float maxAbs = 0f;
            int worstIdx = 0;
            for (int i = 0; i < VocabSize; i++)
            {
                float diff = MathF.Abs(cpuChunk2Last[i] - cudaChunk2Last[i]);
                if (diff > maxAbs) { maxAbs = diff; worstIdx = i; }
            }
            _output.WriteLine(
                $"Two-chunk MIMO decode chunk-2 last-token drift: max_abs={maxAbs:E3} at idx {worstIdx} "
                + $"(cpu={cpuChunk2Last[worstIdx]:G6} cuda={cudaChunk2Last[worstIdx]:G6})");

            int cpuArg = ArgMax(cpuChunk2Last);
            int cudaArg = ArgMax(cudaChunk2Last);
            _output.WriteLine($"Argmax: cpu={cpuArg} cuda={cudaArg}");

            Assert.True(maxAbs <= TwoChunkLogitsAbsTol,
                $"Two-chunk MIMO decode chunk-2 divergence {maxAbs:G6} > {TwoChunkLogitsAbsTol:G4} at idx {worstIdx}.");
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
    /// Issue #385 self-check: asserts this fixture's local constants match what
    /// <see cref="Mamba3Config"/>'s own formulas derive from the same primitive
    /// inputs (StateSize/NumHeads/HeadDim/NumGroups/RopeFraction/MimoRank) —
    /// catches the fixture's own arithmetic drifting from the production
    /// derivation it exists to exercise, before any forward pass runs.
    /// </summary>
    private static void AssertDerivedDims(Mamba3Config m3)
    {
        Assert.Equal(DInner, m3.DInner);
        Assert.Equal(BcDim, m3.BcDim);
        Assert.Equal(NumRopeAngles, m3.NumRopeAngles);
        Assert.Equal(DInProj, m3.InputProjectionDim);
    }

    /// <summary>Downloads a device buffer into a managed array via a D2H copy (test-only helper).</summary>
    private static unsafe float[] DownloadF32(nint devicePtr, int elementCount)
    {
        var host = new float[elementCount];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(elementCount * sizeof(float))).ThrowOnError();
        return host;
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

    private static void WriteSyntheticMimoCheckpoint(string safetensorsPath, string configPath)
    {
        using (var fs = File.Create(configPath))
        using (var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
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
            writer.WriteNumber("mimo_rank", MimoRank);
            writer.WriteBoolean("is_mimo", true);
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

        var tensors = new List<(string Name, int[] Shape, float[] Values)>();
        AddSmall(tensors, Mamba3TensorMapping.TokenEmbedding, [VocabSize, HiddenSize], 0.05f, 0);
        AddSmall(tensors, Mamba3TensorMapping.FinalNorm, [HiddenSize], 0.5f, 1);
        AddSmall(tensors, Mamba3TensorMapping.LmHead, [VocabSize, HiddenSize], 0.05f, 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int sBase = 10 * (i + 1);
            AddSmall(tensors, Mamba3TensorMapping.LayerNorm(i), [HiddenSize], 0.5f, sBase + 0);
            AddSmall(tensors, Mamba3TensorMapping.InProj(i), [DInProj, HiddenSize], 0.02f, sBase + 1);
            AddSmall(tensors, Mamba3TensorMapping.OutProj(i), [HiddenSize, DInner], 0.05f, sBase + 2);
            AddSmall(tensors, Mamba3TensorMapping.BNorm(i), [StateSize], 0.5f, sBase + 3);
            AddSmall(tensors, Mamba3TensorMapping.CNorm(i), [StateSize], 0.5f, sBase + 4);
            AddSmall(tensors, Mamba3TensorMapping.BBias(i), [NumHeads, MimoRank, StateSize], 0.02f, sBase + 5);
            AddSmall(tensors, Mamba3TensorMapping.CBias(i), [NumHeads, MimoRank, StateSize], 0.02f, sBase + 6);
            AddSmall(tensors, Mamba3TensorMapping.D(i), [NumHeads], 0.1f, sBase + 7);
            AddSmall(tensors, Mamba3TensorMapping.DtBias(i), [NumHeads], 0.02f, sBase + 8);
            // Issue #385: canonical init values are mimo_x ~ 1/R, mimo_z ~ 1, mimo_o ~ 1/R
            // (Mamba3TensorMapping doc comments), but a PURE constant fill is invariant
            // under index permutation — a [H,R,P] stride/transposition bug on the
            // consumption side (e.g. swapping the R and P axes, or a per-rank offset
            // error) reads a different element that holds the exact same value and is
            // therefore undetectable end-to-end. Perturb each with a small distinct
            // seeded-cosine ramp around its canonical center (mirrors the CPU MIMO
            // fixture's AddSeededCosinesAround / SafetensorsFixtureBuilder.cs) so a
            // wrong-index read is very likely to land on a numerically different value.
            AddSmallAround(tensors, Mamba3TensorMapping.MimoX(i), [NumHeads, MimoRank, HeadDim], 1f / MimoRank, 0.05f, sBase + 9);
            AddSmallAround(tensors, Mamba3TensorMapping.MimoZ(i), [NumHeads, MimoRank, HeadDim], 1f, 0.05f, sBase + 10);
            AddSmallAround(tensors, Mamba3TensorMapping.MimoO(i), [NumHeads, MimoRank, HeadDim], 1f / MimoRank, 0.05f, sBase + 11);
        }

        WriteSafetensorsFile(safetensorsPath, tensors);
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

    /// <summary>
    /// Issue #385: like <see cref="AddSmall"/> but centered on <paramref name="center"/>
    /// instead of zero — used for MimoX/MimoZ/MimoO, whose canonical inits are non-zero
    /// constants (1/R, 1, 1/R). Distinct <paramref name="seed"/> per tensor keeps the
    /// three perturbation patterns from coinciding.
    /// </summary>
    private static void AddSmallAround(List<(string, int[], float[])> sink, string name, int[] shape, float center, float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = center + amplitude * MathF.Cos(phi);
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
