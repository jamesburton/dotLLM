using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Issue #509 follow-up: proves the pinned staged H2D upload path is actually <i>reachable from a
/// real model load</i>, in both the on and the off position.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this exists as a separate class.</b> <c>CudaWeightUploadStagingTests</c> (unit) already
/// proves the staging transfer is byte-identical to the direct one, and it is a good test — but it
/// drives <see cref="CudaWeightUploadStaging"/> directly. A byte-identity proof of a component
/// says nothing about whether the production loader ever calls it. Every part of #509 could be
/// correct and the feature still be dead code: <c>CudaWeights</c> opens the scope but the decision
/// to stage is taken per tensor deep inside the upload, behind a size threshold, a support check
/// and a pinned-allocation failure latch. The handoff recorded exactly this gap — "the byte-identity
/// tests pass but do not prove the flag reaches a real upload".
/// </para>
/// <para>
/// <b>Why an A/B and not just an assertion that the counter moved.</b> A single opted-in arm
/// asserting <c>TotalStagedChunks &gt; 0</c> would also pass if staging were hard-wired on and the
/// flag ignored — which is the more dangerous defect of the two, since it would silently page-lock
/// memory for every user on a path nobody has measured. The off arm is what discriminates: the
/// same model, the same process, the same code, and the staged counters must not move at all while
/// the direct counter does. Together the two arms pin the flag to the behaviour in both directions.
/// </para>
/// <para>
/// <b>Why the in-process override rather than the environment variable.</b>
/// <c>CudaWeightUploadStaging.EnabledFromEnv</c> is a <c>static readonly</c> captured at type
/// initialisation, so a test that set <c>DOTLLM_CUDA_PINNED_UPLOAD</c> with
/// <c>Environment.SetEnvironmentVariable</c> would be read <i>after</i> the type was already
/// initialised by some earlier test in the same process and would silently prove nothing. The
/// <c>EnabledOverride</c> hook exists for this, and assigning it is why this class must live in a
/// serialized collection (issue #502's rule, enforced by <c>GpuCollectionGuardTests</c>).
/// </para>
/// <para>
/// <b>Counters are process-wide and never reset</b>, and CUDA classes share the process, so every
/// assertion here is on a <i>delta</i> across the load rather than an absolute value.
/// </para>
/// <para>
/// <b>First green run</b> (T5500, RTX 3060, Llama-3.2-1B Q8_0, 2026-09-23): opted in, 10 staging
/// chunks carrying 558,170,112 B at the default 67,108,864 B chunk; opted out, 0 staging chunks and
/// 1,592,066,048 B direct. Note the opted-in arm still sends most of the model direct — staging
/// only claims tensors above the one-chunk threshold, which is the documented design, so
/// <c>TotalDirectBytes</c> moving in the opted-in arm is expected and is deliberately not asserted
/// against.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CudaWeightUploadStagingWiringTests
{
    private readonly ITestOutputHelper _output;

    public CudaWeightUploadStagingWiringTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Loads one real checkpoint on CUDA twice — once opted in, once opted out — and asserts the
    /// staged counters move only in the opted-in arm while the direct counter moves only in the
    /// opted-out one.
    /// </summary>
    [SkippableFact]
    public void PinnedStagedUpload_IsReachedByARealModelLoad_AndOnlyWhenOptedIn()
    {
        FixtureLocation fixture = TestFixtureResolver.ResolveFile(
            "DOTLLM_LLAMA32_1B_Q8_0_GGUF", "bartowski", "Llama-3.2-1B-Instruct-GGUF",
            "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Skip.If(!fixture.Found, fixture.SkipMessage("Llama-3.2-1B Q8_0 GGUF"));

        string ptxDir = ResolvePtxDir();
        bool? savedEnabled = CudaWeightUploadStaging.EnabledOverride;

        try
        {
            // ── Arm A: opted IN ────────────────────────────────────────────────
            CudaWeightUploadStaging.EnabledOverride = true;
            Assert.True(CudaWeightUploadStaging.Enabled, "override did not take effect");

            long chunksBefore = CudaWeightUploadStaging.TotalStagedChunks;
            long stagedBefore = CudaWeightUploadStaging.TotalStagedBytes;
            LoadOnceOrSkip(fixture.Path!, ptxDir, "opted-in");
            long stagedChunks = CudaWeightUploadStaging.TotalStagedChunks - chunksBefore;
            long stagedBytes = CudaWeightUploadStaging.TotalStagedBytes - stagedBefore;

            _output.WriteLine(
                $"[opted-in] staged chunks +{stagedChunks}, staged bytes +{stagedBytes:N0} "
                + $"(chunk size {CudaWeightUploadStaging.ConfiguredChunkBytes:N0} B)");

            // The load must have pushed at least one tensor through the pinned path. If this fails
            // the feature is unreachable from a real load, which is the whole point of the test.
            Assert.True(stagedChunks > 0,
                "opted in, but a real CUDA model load issued zero staging chunks — the pinned "
                + "staged upload is not reachable from CudaWeights' upload path.");
            Assert.True(stagedBytes > 0, "staging chunks were issued but no bytes were recorded.");

            // Chunking actually loops: a tensor above the threshold is split, so bytes must exceed
            // what a single degenerate chunk would carry only if more than one chunk was issued.
            // (Not asserted as a strict multiple — the last chunk of a tensor is a partial one.)
            Assert.True(stagedBytes <= stagedChunks * CudaWeightUploadStaging.ConfiguredChunkBytes,
                $"recorded {stagedBytes} staged bytes across {stagedChunks} chunks, which exceeds "
                + "the configured chunk size — the counters disagree with the chunking loop.");

            // ── Arm B: opted OUT (the control) ─────────────────────────────────
            CudaWeightUploadStaging.EnabledOverride = false;
            Assert.False(CudaWeightUploadStaging.Enabled, "override did not take effect");

            chunksBefore = CudaWeightUploadStaging.TotalStagedChunks;
            long directBefore = CudaWeightUploadStaging.TotalDirectBytes;
            LoadOnceOrSkip(fixture.Path!, ptxDir, "opted-out");
            long stagedChunksOff = CudaWeightUploadStaging.TotalStagedChunks - chunksBefore;
            long directBytesOff = CudaWeightUploadStaging.TotalDirectBytes - directBefore;

            _output.WriteLine(
                $"[opted-out] staged chunks +{stagedChunksOff}, direct bytes +{directBytesOff:N0}");

            Assert.True(stagedChunksOff == 0,
                $"opted out, but the load still issued {stagedChunksOff} staging chunks — the "
                + "opt-in flag does not gate the production upload path.");
            Assert.True(directBytesOff > 0,
                "opted out, but no bytes took the direct path either — the load did not upload "
                + "anything, so this arm proves nothing.");
        }
        finally
        {
            CudaWeightUploadStaging.EnabledOverride = savedEnabled;
        }
    }

    /// <summary>
    /// Performs one real CUDA load of <paramref name="path"/> and disposes it, skipping the test
    /// (rather than failing it) when the host has too little VRAM — same convention as
    /// <see cref="RealGgufCudaParityTests"/>.
    /// </summary>
    private void LoadOnceOrSkip(string path, string ptxDir, string label)
    {
        using var gguf = CheckpointGuard.LoadOrSkip(path, $"GGUF checkpoint ({label})", () => GgufFile.Open(path));
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        CudaTransformerModel? model = null;
        try
        {
            model = CudaTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        }
        catch (DotLLM.Cuda.Interop.CudaException ex)
        {
            Skip.If(true,
                $"[{label}] CUDA load failed with {ex.Message}. Weights likely exceeded available "
                + "VRAM on this host. Re-run on a host with more VRAM.");
        }

        try
        {
            watch.Stop();
            _output.WriteLine($"[{label}] CUDA load {watch.Elapsed.TotalSeconds:F1} s");
        }
        finally
        {
            model?.Dispose();
        }
    }

    private static string ResolvePtxDir()
    {
        string? probe = AppContext.BaseDirectory;
        for (int i = 0; i < 8 && probe is not null; i++)
        {
            string candidate = Path.Combine(probe, "native", "ptx");
            if (Directory.Exists(candidate)) return candidate;
            probe = Path.GetDirectoryName(probe);
        }
        return Path.GetFullPath(Path.Combine(
            AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"));
    }
}
