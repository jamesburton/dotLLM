using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA coverage for issue #253 (Multi-Token Prediction self-speculative decoding), mirroring
/// <c>Qwen3HybridDenseMtpTests</c> (CPU): GGUF detection/loading of the trailing MTP/"NextN" head
/// on <see cref="CudaQwen3HybridDenseTransformerModel"/>, the MTP head's own CUDA forward pass
/// (<see cref="CudaMtpState"/> device-resident KV-cache + pending-hidden handoff), and an
/// engine-level integration test proving <see cref="DotLLM.Engine.MtpSpeculativeDecoder"/> — which
/// is completely unmodified/backend-agnostic — produces the same output sequence as plain greedy
/// decode when driven by the real CUDA model. Uses
/// <see cref="SyntheticQwen35HybridDenseMtpGguf"/> — the same tiny (F32, ~KB-scale) fixture the CPU
/// tests use, so no real Qwen3.6-MTP-GGUF download is needed.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpTests : IDisposable
{
    private readonly string _scratch;
    private readonly ITestOutputHelper _out;

    public CudaQwen3HybridDenseMtpTests(ITestOutputHelper output)
    {
        _out = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-qwen35-mtp-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string WriteFixture(bool withMtp, bool mtpHasOwnHeadTensors = true, string name = "qwen35-mtp.gguf") =>
        SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, name), withMtp: withMtp, mtpHasOwnHeadTensors: mtpHasOwnHeadTensors);

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
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

    // ── GGUF detection / zero-behavior-change ──────────────────────────────────

    [SkippableFact]
    public void LoadFromGguf_WithMtp_DetectsHeadAndCreatesState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(Architecture.Qwen3HybridDense, config.Architecture);
        Assert.Equal(1, config.NextnPredictLayers);
        Assert.Equal(SyntheticQwen35HybridDenseMtpGguf.BlockCount, config.NumLayers);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsMtp);
        using IMtpState? state = model.CreateMtpState();
        Assert.NotNull(state);
        Assert.IsType<CudaMtpState>(state);
    }

    [SkippableFact]
    public void LoadFromGguf_WithoutMtp_ZeroBehaviorChange()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: false);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        Assert.Equal(0, config.NextnPredictLayers);
        Assert.Equal(SyntheticQwen35HybridDenseMtpGguf.BlockCount, config.NumLayers);

        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.False(model.SupportsMtp);
        Assert.Null(model.CreateMtpState());

        Assert.Throws<NotSupportedException>(() =>
        {
            using var state = new CudaMtpState(hiddenSize: 1, numKvHeads: 1, headDim: 1, maxSteps: 1);
            model.ForwardMtp(state, tokenId: 0, position: 0);
        });
    }

    [SkippableFact]
    public void Forward_WithMtpStateCapture_ProducesIdenticalLogitsToPlainForward()
    {
        // The hidden-state capture is documented as a pure side effect — verify it byte-for-byte,
        // mirroring the CPU host's equivalent test.
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: true);
        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using var gguf1 = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf1.Metadata);
        using var model1 = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf1, config, deviceId: 0, ptxDir);
        using var kvCache1 = model1.CreateKvCache(maxSeqLen: 64);
        using ITensor plainLogits = model1.Forward(tokenIds, positions, deviceId: -1, kvCache1);

        // Two SEPARATE model instances (not model1.ResetSequenceState() + reuse): this model has a
        // GatedDeltaNet layer whose recurrent state is model-owned, not IKvCache-owned — a fresh
        // instance guarantees a genuinely independent starting state for the capture-vs-plain
        // comparison, matching CudaQwen3HybridDenseLastTokenLogitsOnlyTest's established pattern
        // for this exact class of model.
        using var gguf2 = GgufFile.Open(path);
        using var model2 = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf2, config, deviceId: 0, ptxDir);
        using var kvCache2 = model2.CreateKvCache(maxSeqLen: 64);
        using var mtpState = model2.CreateMtpState()!;
        using ITensor capturedLogits = model2.Forward(tokenIds, positions, deviceId: -1, kvCache2, adapter: null, mtpState);

        Assert.Equal(plainLogits.Shape[0], capturedLogits.Shape[0]);
        Assert.Equal(plainLogits.Shape[1], capturedLogits.Shape[1]);
        unsafe
        {
            int n = tokenIds.Length * config.VocabSize;
            var a = new ReadOnlySpan<float>((void*)plainLogits.DataPointer, n);
            var b = new ReadOnlySpan<float>((void*)capturedLogits.DataPointer, n);
            for (int i = 0; i < n; i++)
                Assert.Equal(a[i], b[i]); // byte-identical float compare — pure side effect claim
        }

        Assert.Equal(tokenIds.Length, mtpState.CapturedRowCount);
        Assert.Equal(config.HiddenSize, mtpState.HiddenSize);
    }

    // ── MTP head forward math ───────────────────────────────────────────────────

    [SkippableFact]
    public void ForwardMtp_ProducesFiniteLogitsAndAdvancesState()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: true);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        using var kvCache = model.CreateKvCache(maxSeqLen: 64);
        using var mtpState = (CudaMtpState)model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        Assert.Equal(0, mtpState.CurrentLength);

        using ITensor draft0 = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 2);
        Assert.Equal(1, draft0.Shape[0]);
        Assert.Equal(config.VocabSize, draft0.Shape[1]);
        Assert.Equal(1, mtpState.CurrentLength);
        AssertAllFinite(draft0, config.VocabSize);

        // Second autoregressive MTP step, seeded from the head's own output (not the trunk's).
        int argmax0 = ArgMax(draft0, config.VocabSize);
        using ITensor draft1 = model.ForwardMtp(mtpState, tokenId: argmax0, position: 3);
        Assert.Equal(2, mtpState.CurrentLength);
        AssertAllFinite(draft1, config.VocabSize);
    }

    [SkippableFact]
    public void ForwardMtp_Deterministic_SameInputsSameOutputs()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: true);

        float[] logitsA = RunSingleMtpStep(path, ptxDir!);
        float[] logitsB = RunSingleMtpStep(path, ptxDir!);

        Assert.Equal(logitsA.Length, logitsB.Length);
        for (int i = 0; i < logitsA.Length; i++)
            Assert.Equal(logitsA[i], logitsB[i]);
    }

    [SkippableFact]
    public void ForwardMtp_WithoutOwnHeadTensors_FallsBackToTrunkHeadAndEmbedding()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = WriteFixture(withMtp: true, mtpHasOwnHeadTensors: false, name: "qwen35-mtp-noheadtensors.gguf");
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsMtp);

        int[] tokenIds = [0, 1];
        int[] positions = [0, 1];
        using var kvCache = model.CreateKvCache(maxSeqLen: 64);
        using var mtpState = (CudaMtpState)model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        using ITensor draft = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 1);
        AssertAllFinite(draft, config.VocabSize);
    }

    // ── Engine-level integration: MtpSpeculativeDecoder (unmodified, backend-agnostic) against the
    //    real CUDA model ──────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// The central claim under test, ported from <c>MtpSpeculativeDecoderTests</c>'s mock-model
    /// version to the REAL CUDA model: MTP self-speculative decoding must produce the exact same
    /// output token sequence as plain greedy decode of the target model alone. Unlike the mock
    /// version (which hand-crafts "MTP disagrees with target" logits), the real MTP head and trunk
    /// are independently-weighted projections over an untrained (random-weight) tiny model, so
    /// disagreements between the MTP head's own guess and the trunk's argmax arise naturally —
    /// this exercises both the accept and the reject/correction path without needing a hand-crafted
    /// mock, while also proving <see cref="DotLLM.Engine.MtpSpeculativeDecoder"/> (completely
    /// unmodified by this CUDA work) really is backend-agnostic against a real CUDA <see cref="IModel"/>.
    /// </summary>
    /// <remarks>
    /// Uses <c>fullAttnInterval: 1</c> (an all-full-attention trunk, no GDN layer) deliberately —
    /// NOT the default mixed GDN+attention fixture the other tests in this class use. A separate,
    /// pre-existing gap was found while first writing this test against the default (GDN-containing)
    /// fixture: <see cref="DotLLM.Engine.MtpSpeculativeDecoder"/>'s (and the equivalent two-model
    /// <c>SpeculativeDecoder</c>'s) verify-batch forward runs SEVERAL candidate tokens through the
    /// trunk in one call, and rejected ones still permanently advance any recurrent (GDN/Mamba)
    /// layer's state — <see cref="IGdnState"/>'s own doc says outright "it has no position
    /// indexing", so unlike the attention KV-cache's <c>Rollback</c>, there is no way to undo that
    /// once the batched forward has run. This is an engine-layer limitation of speculative decoding
    /// against recurrent trunks in general (present identically on CPU — verified separately, not
    /// introduced by this CUDA work, and out of scope to fix here: it needs a real GDN-state
    /// checkpoint/restore design threaded through both decoders across CPU/CUDA/Vulkan). Isolating
    /// THIS test to an all-full-attention trunk keeps it a clean proof of the actual CUDA MTP
    /// correctness this task is scoped to; see the class remarks / final report for the full
    /// finding and recommendation to file it as its own follow-up issue.
    /// </remarks>
    [SkippableFact]
    public void DraftAndVerify_MatchesPlainGreedyDecode_OnRealCudaModel()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-fullattn.gguf"), withMtp: true, fullAttnInterval: 1);

        const int startToken = 1;
        const int totalNewTokens = 12;
        const int k = 3;

        List<int> speculative = RunSpeculative(path, ptxDir!, startToken, totalNewTokens, k);
        List<int> plain = RunPlainGreedy(path, ptxDir!, startToken, totalNewTokens);

        _out.WriteLine($"plain:       {string.Join(",", plain)}");
        _out.WriteLine($"speculative: {string.Join(",", speculative)}");

        Assert.Equal(plain, speculative);
    }

    /// <summary>
    /// CUDA counterpart of the CPU regression test for issue #287's fix
    /// (<c>MtpSpeculativeDecoderGdnStateTests.DraftAndVerify_AfterRejection_NextTokenLogitsMatchCleanReplay_RawFloats</c>).
    /// Unlike <see cref="DraftAndVerify_MatchesPlainGreedyDecode_OnRealCudaModel"/> above (which
    /// deliberately uses <c>fullAttnInterval: 1</c> to dodge the GDN-rollback gap that test's own
    /// remarks describe), this test uses the fixture's DEFAULT mixed GDN+attention layout and
    /// targets that gap directly: after running <see cref="MtpSpeculativeDecoder"/> through several
    /// rounds against the real <see cref="CudaQwen3HybridDenseTransformerModel"/> (hitting at least
    /// one genuine rejection), forwarding the still-pending last accepted/corrected token against
    /// the model's own internal, decoder-managed <see cref="CudaGdnStateCache"/> must produce
    /// BYTE-IDENTICAL logits to forwarding the exact same accepted-token history, one token at a
    /// time, through a fresh CUDA model instance. Before <see cref="CudaQwen3HybridDenseTransformerModel.CheckpointRecurrentState"/> /
    /// <see cref="CudaQwen3HybridDenseTransformerModel.RestoreRecurrentState"/> existed, this would
    /// diverge (rejected draft tokens' recurrent-state contribution leaked into the device-resident
    /// GDN state with no way to undo it); the CUDA <c>cuMemcpyDtoD_v2</c>-based checkpoint/restore
    /// fixes it identically to the CPU host.
    /// </summary>
    [SkippableFact]
    public void DraftAndVerify_AfterRejection_NextTokenLogitsMatchCleanReplay_OnRealCudaModel()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(
            Path.Combine(_scratch, "qwen35-mtp-gdn-cuda-logits.gguf"), withMtp: true);

        const int startToken = 3;
        const int k = 3;
        const int rounds = 3;

        List<int> acceptedHistory;
        int lastTokenPosition;
        float[] nextLogitsFromDecoder;
        bool anyRejection;

        // ── Run the (fixed) decoder through several rounds, then forward the still-pending last
        //    token against the SAME CUDA model instance's internal, decoder-managed GDN state. ──
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
            Assert.True(model.SupportsMtp);
            Assert.True(model.SupportsRecurrentStateCheckpoint);

            var decoder = new MtpSpeculativeDecoder(greedy: true);
            var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

            var generatedIds = new List<int> { startToken };
            using var kvCache = model.CreateKvCache(maxSeqLen: 64);
            using var mtpState = (CudaMtpState)model.CreateMtpState()!;

            // NOTE: deliberately NO manual prefill forward here — DraftAndVerify's own "catchup"
            // forward already seeds the KV-cache and trunk hidden state for `lastToken` at
            // `position` on round 1. A real recurrent (GDN) model must not be forwarded through the
            // same token/position twice (see MtpSpeculativeDecoderGdnStateTests' identical note).
            int position = 0;
            anyRejection = false;
            Span<int> outputBuffer = stackalloc int[k + 1];
            for (int r = 0; r < rounds; r++)
            {
                int kThisRound = Math.Min(k, 64 - position - 2);
                if (kThisRound <= 0) break;

                var result = decoder.DraftAndVerify(
                    model, kvCache, mtpState, pipeline, generatedIds,
                    constraint: null, position, vocabSize: config.VocabSize, numCandidates: kThisRound, outputBuffer);

                if (result.AcceptedCount <= result.DraftedCount)
                    anyRejection = true;

                _out.WriteLine(
                    $"[cuda-logits] round {r} @ position={position} k={kThisRound} accepted={result.AcceptedCount} " +
                    $"drafted={result.DraftedCount} out=[{string.Join(",", outputBuffer.Slice(0, result.AcceptedCount).ToArray())}]");

                for (int i = 0; i < result.AcceptedCount; i++)
                    generatedIds.Add(outputBuffer[i]);

                position += result.AcceptedCount;
            }

            Assert.True(anyRejection,
                "Test setup failure: no rejection occurred across the rounds run, so this comparison " +
                "cannot discriminate between correct and GDN-state-corrupted behavior.");

            acceptedHistory = generatedIds;
            lastTokenPosition = position;
            _out.WriteLine($"[cuda-logits] acceptedHistory=[{string.Join(",", acceptedHistory)}] lastTokenPosition={lastTokenPosition}");

            using ITensor nextLogits = model.Forward([acceptedHistory[^1]], [lastTokenPosition], deviceId: -1, kvCache);
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)nextLogits.DataPointer, config.VocabSize);
                nextLogitsFromDecoder = span.ToArray();
            }
        }

        // ── Clean reference: fresh CUDA model instance, forward EXACTLY the same accepted-token
        //    history, one token at a time — no decoder involved at all. ──
        float[] nextLogitsClean;
        {
            using var gguf = GgufFile.Open(path);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir!);
            using var kvCache = model.CreateKvCache(maxSeqLen: 64);

            ITensor? last = null;
            for (int pos = 0; pos < acceptedHistory.Count; pos++)
            {
                last?.Dispose();
                last = model.Forward([acceptedHistory[pos]], [pos], deviceId: -1, kvCache);
            }

            using ITensor probeLogits = last!;
            unsafe
            {
                var span = new ReadOnlySpan<float>((void*)probeLogits.DataPointer, config.VocabSize);
                nextLogitsClean = span.ToArray();
            }
        }

        Assert.Equal(nextLogitsClean.Length, nextLogitsFromDecoder.Length);
        for (int i = 0; i < nextLogitsClean.Length; i++)
            Assert.Equal(nextLogitsClean[i], nextLogitsFromDecoder[i]); // byte-identical float compare
    }

    private static float[] RunSingleMtpStep(string path, string ptxDir)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];
        using var kvCache = model.CreateKvCache(maxSeqLen: 64);
        using var mtpState = (CudaMtpState)model.CreateMtpState()!;
        using (ITensor _ = model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState)) { }
        mtpState.SeedFromCapturedRow(mtpState.CapturedRowCount - 1);

        using ITensor draft = model.ForwardMtp(mtpState, tokenId: tokenIds[^1], position: 2);
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)draft.DataPointer, config.VocabSize);
            return span.ToArray();
        }
    }

    private static List<int> RunPlainGreedy(string path, string ptxDir, int startToken, int totalNewTokens)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var kvCache = model.CreateKvCache(maxSeqLen: 64);

        var seq = new List<int> { startToken };
        int cur = startToken;
        for (int pos = 0; pos < totalNewTokens; pos++)
        {
            using ITensor logits = model.Forward([cur], [pos], deviceId: -1, kvCache);
            int argmax = ArgMax(logits, config.VocabSize);
            seq.Add(argmax);
            cur = argmax;
        }
        return seq;
    }

    private static List<int> RunSpeculative(
        string path, string ptxDir, int startToken, int totalNewTokens, int k)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        Assert.True(model.SupportsMtp);

        var decoder = new MtpSpeculativeDecoder(greedy: true);
        var pipeline = new SamplerPipeline(new InferenceOptions { Temperature = 0f });

        var generatedIds = new List<int> { startToken };
        using var kvCache = model.CreateKvCache(maxSeqLen: 64);
        using var mtpState = (CudaMtpState)model.CreateMtpState()!;

        // Prefill: seed the target KV-cache with the start token at position 0. DraftAndVerify's
        // own contract (see its remarks: "lastToken already occupies position in kvCacheTarget")
        // means `position` must equal lastToken's OWN KV-cache slot, not "prompt length + decoded
        // so far" naively read as generatedIds.Count -- so it starts at 0 (matching the prefill
        // call just above, not 1). Confirmed against SpeculativeDecoder's identical convention
        // (verifyPositions[0] = position, holding lastToken). The CPU mock-model engine tests this
        // was originally copied from never caught an off-by-one here because their mock Forward
        // ignores position entirely for logit computation; this real (position-sensitive,
        // RoPE-using) CUDA model does not tolerate it.
        using (ITensor _ = model.Forward([startToken], [0], deviceId: -1, kvCache)) { }

        int position = 0;
        Span<int> outputBuffer = stackalloc int[k + 1];
        int guard = 0;
        while (generatedIds.Count - 1 < totalNewTokens && guard++ < totalNewTokens * 4)
        {
            var result = decoder.DraftAndVerify(
                model, kvCache, mtpState, pipeline, generatedIds,
                constraint: null, position, vocabSize: config.VocabSize, numCandidates: k, outputBuffer);

            Assert.True(result.AcceptedCount > 0, "Every round must accept at least the corrected/bonus token.");

            for (int i = 0; i < result.AcceptedCount && generatedIds.Count - 1 < totalNewTokens; i++)
                generatedIds.Add(outputBuffer[i]);

            position += result.AcceptedCount;
        }

        return generatedIds.Take(totalNewTokens + 1).ToList();
    }

    private static void AssertAllFinite(ITensor tensor, int vocabSize)
    {
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, vocabSize);
            foreach (float v in span)
                Assert.True(float.IsFinite(v), $"non-finite logit: {v}");
        }
    }

    private static int ArgMax(ITensor tensor, int vocabSize)
    {
        unsafe
        {
            var span = new ReadOnlySpan<float>((void*)tensor.DataPointer, vocabSize);
            int best = 0;
            for (int i = 1; i < span.Length; i++)
                if (span[i] > span[best]) best = i;
            return best;
        }
    }
}
