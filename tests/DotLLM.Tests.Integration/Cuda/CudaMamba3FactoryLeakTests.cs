using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Regression coverage for issue #383: <see cref="CudaMamba3TransformerModel.LoadFromSafetensors"/>
/// used to create <c>CudaContext</c>/<c>CudaStream</c>/<c>CudaCublasHandle</c>/<c>CudaKernels</c>
/// and upload weight buffers with no try/catch — any exception partway through (device OOM, a
/// missing PTX module, a dtype/shape guard) leaked everything already created.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this test doesn't use a BF16 tensor (the repro the issue text names).</b>
/// <see cref="DotLLM.Models.Architectures.Mamba3WeightLoader.Load"/>'s <c>ResolveRequired</c>
/// already screens every tensor's dtype and folds a non-F32 tensor into
/// <c>Mamba3WeightLoadReport.MissingRequiredCount</c> (via <c>UnsupportedDType</c>), so
/// <c>LoadFromSafetensors</c>'s own <c>HasMissingRequired</c> guard throws
/// <see cref="InvalidDataException"/> BEFORE <c>CudaContext.Create</c> is ever called — nothing
/// has been allocated yet, so a BF16 checkpoint no longer reaches the leaking code path at all
/// (this looks like it changed independently of #383; <c>UploadF32</c>'s own dtype guard is now
/// dead code for every <c>ResolveRequired</c>-sourced tensor). This test instead uses a bogus
/// <c>ptxDir</c>, which makes <c>new CudaKernels(ptxDir)</c> throw a plain
/// <see cref="DirectoryNotFoundException"/> AFTER <c>CudaContext</c>/<c>CudaStream</c>/<c>CudaCublasHandle</c>
/// already exist — the exact resource triple named in the issue title, and a failure mode that
/// remains fully reachable (a misconfigured/missing PTX deployment).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMamba3FactoryLeakTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaMamba3FactoryLeakTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-mamba3-leak-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void LoadFromSafetensors_BadPtxDir_ThrowsAndLeaksNoDeviceMemory()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        string modelPath = Path.Combine(_scratch, "model.safetensors");
        string configPath = Path.Combine(_scratch, "config.json");
        CudaMamba3ParitySyntheticTests.WriteMinimalMamba3CheckpointForGuardTest(modelPath, configPath);

        string badPtxDir = Path.Combine(_scratch, "no-such-ptx-dir");

        const int Iterations = 5;
        nuint baselineFree = 0;

        // cuMemGetInfo (like every CUDA driver query) operates on whatever context is CURRENT
        // on the calling thread — and each failed LoadFromSafetensors call below both creates
        // AND (once the #383 fix disposes it in the catch path) destroys its own CudaContext,
        // leaving the thread with no current context afterwards. A dedicated, long-lived probe
        // context — created once, never touched by the factory under test — gives cuMemGetInfo
        // something valid to query; MakeCurrent() is called right before every query below since
        // the most recent CUDA call on this thread was the (now-destroyed) factory context.
        using var probeContext = CudaContext.Create(deviceId: 0);

        for (int i = 0; i < Iterations; i++)
        {
            var (cpuModel, cpuFile, config) = ModelLoader.LoadFromSafetensors(modelPath);
            try
            {
                // The model isn't needed for anything beyond resolving `config` + the opened
                // safetensors source the CUDA factory reads from.
                cpuModel.Dispose();

                Assert.Throws<DirectoryNotFoundException>(() =>
                    CudaMamba3TransformerModel.LoadFromSafetensors(cpuFile, config, deviceId: 0, ptxDir: badPtxDir));
            }
            finally
            {
                cpuFile.Dispose();
            }

            // Baseline AFTER the first failed load — the first CUDA call in this process pays
            // one-time driver/JIT warmup cost that is not representative of steady-state leak
            // behavior; measuring from here isolates the per-iteration effect.
            if (i == 0)
            {
                probeContext.MakeCurrent();
                CudaDriverApi.cuMemGetInfo_v2(out baselineFree, out _).ThrowOnError();
                _output.WriteLine($"Baseline free VRAM after 1st failed load: {baselineFree / (1024 * 1024)} MB");
            }
        }

        probeContext.MakeCurrent();
        CudaDriverApi.cuMemGetInfo_v2(out nuint freeAfter, out _).ThrowOnError();
        _output.WriteLine($"Free VRAM after {Iterations} failed loads: {freeAfter / (1024 * 1024)} MB");

        // Noise floor covers driver bookkeeping jitter across (Iterations-1) further failed
        // loads without hiding a real per-load leak. Pre-#383, each failed load leaked a live
        // CudaContext (which alone reserves tens of MB of device memory for driver structures)
        // plus a CudaStream/CudaCublasHandle — a real regression would show up as tens of MB
        // per iteration, i.e. hundreds of MB total, far past this floor.
        const long NoiseFloorBytes = 16L * 1024 * 1024;
        long drop = (long)baselineFree - (long)freeAfter;
        Assert.True(drop < NoiseFloorBytes,
            $"Free VRAM dropped by {drop / (1024.0 * 1024):F1} MB across {Iterations - 1} further failed loads " +
            $"(baseline {baselineFree / (1024 * 1024)} MB -> {freeAfter / (1024 * 1024)} MB) — likely a leak.");
    }
}
