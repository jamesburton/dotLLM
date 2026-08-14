using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models;
using Xunit;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Tests for <see cref="CudaMamba3TransformerModel.ForwardBatch"/> — the ONLY entrypoint
/// <c>ContinuousBatchScheduler</c> uses for this model. Filed as a final-review finding on
/// issue #346: the method has existed since Task 9's forward-pass work but had zero direct
/// test coverage (only exercised transitively, if at all, through scheduler-level tests that
/// never batch 2+ Mamba-3 sequences together).
/// </summary>
/// <remarks>
/// Mirrors
/// <see cref="DotLLM.Tests.Unit.Vulkan.VulkanMamba3TransformerModelForwardBatchTests"/>'s
/// four-case structure exactly (empty requests, single-seq parity with <c>Forward</c>, a
/// null-<c>MambaState</c> guard once 2+ requests are batched, and a 2-sequence run with
/// per-sequence state). Uses the internal fixture writer already shared by
/// <see cref="CudaMamba3ParitySyntheticTests"/>
/// (<see cref="CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest"/>)
/// rather than re-deriving the tensor shapes a third time — this file must live in
/// <c>DotLLM.Tests.Integration</c> (not <c>DotLLM.Tests.Unit</c>) for that <c>internal</c>
/// method to be visible.
/// <para>
/// <b>Why the 2-sequence case matters most here.</b> <see cref="CudaMamba3TransformerModel"/>
/// owns ONE set of device scratch buffers (<c>_hidden</c>, <c>_bDevice</c>, etc. — see
/// <c>EnsureScratchCapacity</c>/<c>FreeScratch</c>) shared across every <c>Forward</c> call,
/// including the sequential per-request loop inside <c>ForwardBatch</c>. A bug that let one
/// request's intermediate values leak into another's (e.g. a scratch buffer not being
/// re-written in full before the next request reads it, or a state-cache mixup between
/// requests) would silently corrupt one sequence's logits while looking perfectly fine in
/// every single-sequence test. <see cref="CudaForwardBatch_MultiSeq_PerSeqMambaState_MatchesReference"/>
/// catches that class of bug by comparing each batched sequence's logits against an
/// independent reference run of that SAME sequence alone (fresh model instance, own
/// <see cref="CudaMamba3StateCache"/>) — a cross-sequence leak would show up as only one of
/// the two comparisons failing (or both, with a similar bias), not a symmetric drift.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMamba3TransformerModelForwardBatchTests : IDisposable
{
    private const int VocabSize = 16; // must match CudaMamba3ParitySyntheticTests' fixture constants.

    // Same-GPU CUDA-vs-CUDA comparison (not cross-backend): both the reference run and the
    // batched run execute the identical kernels on the identical device, so the only
    // possible drift is F32 reduction-order noise from scratch-buffer reuse/growth
    // differences, not the CPU-vs-CUDA libdevice-vs-MathF 1-ULP transcendental gap
    // documented on the kernel-level tests (CudaMamba3DataRopeF32Tests etc.) or the
    // CudaMamba3ParitySyntheticTests' CPU-oracle comparisons. Calibrated the same way as
    // those two: start tight (1e-5), which is >100x looser than the 7.45e-9 pure-noise
    // floor CudaMamba3ParitySyntheticTests observed for a same-shape fixture, but far
    // below any O(1) real-bug signature (wrong index/stride/state-mixup).
    private const float LogitsAbsTol = 1e-5f;

    private readonly string _scratch;

    public CudaMamba3TransformerModelForwardBatchTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-m3-fwdbatch-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void CudaForwardBatch_EmptyRequests_ReturnsEmpty()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        (string modelPath, string configPath) = WriteFixture();
        CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(modelPath, configPath);

        var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            var results = model.ForwardBatch(Array.Empty<SequenceForwardRequest>(), deviceId: -1);
            Assert.NotNull(results);
            Assert.Empty(results);
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    [SkippableFact]
    public void CudaForwardBatch_SingleSeq_EqualsForward()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        (string modelPath, string configPath) = WriteFixture();
        CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(modelPath, configPath);

        int[] tokenIds = [0, 1, 2, 3, 5];
        int[] positions = [0, 1, 2, 3, 4];

        // Reference: plain single-sequence Forward on a fresh model.
        float[] reference;
        {
            var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
            try
            {
                using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
                reference = CopyLogits(logits);
            }
            finally
            {
                model.Dispose();
                source.Dispose();
            }
        }

        // Under test: ForwardBatch with a single request on a fresh model. MambaState is
        // null on the request — with < 2 requests the guard does not fire and the host
        // falls back to Forward's own ephemeral state, equivalent to the reference path.
        float[] batched;
        {
            var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
            try
            {
                var requests = new[]
                {
                    new SequenceForwardRequest
                    {
                        TokenIds = tokenIds.AsMemory(),
                        Positions = positions.AsMemory(),
                        KvCache = null!,
                    },
                };
                var results = model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    Assert.Single(results);
                    Assert.Equal(VocabSize, results[0].Shape[1]);
                    batched = CopyLogits(results[0]);
                }
                finally { foreach (var t in results) t.Dispose(); }
            }
            finally
            {
                model.Dispose();
                source.Dispose();
            }
        }

        AssertLogitsClose(reference, batched, "single-seq");
    }

    [SkippableFact]
    public void CudaForwardBatch_MultiSeq_NullMambaState_ThrowsArgument()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        (string modelPath, string configPath) = WriteFixture();
        CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(modelPath, configPath);

        int[] tokensA = [1, 2, 3];
        int[] positionsA = [0, 1, 2];
        int[] tokensB = [5, 7, 11];
        int[] positionsB = [0, 1, 2];

        var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
        try
        {
            var requests = new[]
            {
                new SequenceForwardRequest
                {
                    TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(), KvCache = null!,
                    // MambaState deliberately null — should trigger the guard once 2+
                    // requests are batched (silent cross-sequence corruption otherwise).
                },
                new SequenceForwardRequest
                {
                    TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(), KvCache = null!,
                },
            };

            var ex = Assert.Throws<ArgumentException>(() => model.ForwardBatch(requests, deviceId: -1));
            Assert.Contains("MambaState", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    [SkippableFact]
    public void CudaForwardBatch_MultiSeq_PerSeqMambaState_MatchesReference()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        (string modelPath, string configPath) = WriteFixture();
        CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(modelPath, configPath);

        int[] tokensA = [0, 1, 2, 3];
        int[] positionsA = [0, 1, 2, 3];
        int[] tokensB = [2, 0, 1, 3];
        int[] positionsB = [0, 1, 2, 3];

        // Reference: each sequence run alone through Forward on its OWN freshly loaded
        // model instance — the "truly isolated" baseline ForwardBatch + per-seq
        // CudaMamba3StateCache is meant to reproduce.
        float[] refA;
        {
            var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
            try
            {
                using ITensor logits = model.Forward(tokensA, positionsA, deviceId: -1);
                refA = CopyLogits(logits);
            }
            finally
            {
                model.Dispose();
                source.Dispose();
            }
        }
        float[] refB;
        {
            var (model, source, _) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
            try
            {
                using ITensor logits = model.Forward(tokensB, positionsB, deviceId: -1);
                refB = CopyLogits(logits);
            }
            finally
            {
                model.Dispose();
                source.Dispose();
            }
        }

        // Under test: ONE model instance (one set of shared scratch buffers), multi-seq
        // ForwardBatch with a distinct CudaMamba3StateCache per request. If the shared
        // scratch were reused incorrectly across the sequential per-request loop, or the
        // two state caches were mixed up, at least one of these would diverge from its
        // solo reference.
        float[] batA, batB;
        {
            var (model, source, config) = CudaModelLoader.LoadMamba3FromSafetensors(modelPath);
            CudaMamba3StateCache? stateA = null, stateB = null;
            try
            {
                stateA = new CudaMamba3StateCache(config.Mamba3Config!, config.NumLayers);
                stateB = new CudaMamba3StateCache(config.Mamba3Config!, config.NumLayers);
                var requests = new[]
                {
                    new SequenceForwardRequest
                    {
                        TokenIds = tokensA.AsMemory(), Positions = positionsA.AsMemory(),
                        KvCache = null!, MambaState = stateA,
                    },
                    new SequenceForwardRequest
                    {
                        TokenIds = tokensB.AsMemory(), Positions = positionsB.AsMemory(),
                        KvCache = null!, MambaState = stateB,
                    },
                };
                var results = model.ForwardBatch(requests, deviceId: -1);
                try
                {
                    Assert.Equal(2, results.Count);
                    batA = CopyLogits(results[0]);
                    batB = CopyLogits(results[1]);
                }
                finally { foreach (var t in results) t.Dispose(); }
            }
            finally
            {
                stateA?.Dispose();
                stateB?.Dispose();
                model.Dispose();
                source.Dispose();
            }
        }

        AssertLogitsClose(refA, batA, "seqA");
        AssertLogitsClose(refB, batB, "seqB");
    }

    private (string ModelPath, string ConfigPath) WriteFixture()
    {
        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        return (modelPath, configPath);
    }

    private static void AssertLogitsClose(float[] reference, float[] actual, string label)
    {
        Assert.Equal(reference.Length, actual.Length);
        for (int c = 0; c < reference.Length; c++)
        {
            float r = reference[c];
            float a = actual[c];
            float diff = MathF.Abs(r - a);
            Assert.True(diff <= LogitsAbsTol,
                $"{label} col={c}: reference={r:F6} vs batched={a:F6} (|diff|={diff:E3} > {LogitsAbsTol:E3})");
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
