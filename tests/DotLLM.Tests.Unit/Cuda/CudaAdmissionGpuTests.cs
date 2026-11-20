using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1848 on a real CUDA device: the allocation sites that need a live model (weights, forward scratch growth, the quadratic
/// G3 score scratch) are admitted against a fake free-memory source (<see cref="CudaAdmission.FreeBytesOverride"/>) and refused,
/// or (G3) fall back, instead of allocating. Each arm fails when its check is removed: the allocation then simply succeeds on a
/// card with free VRAM.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaAdmissionGpuTests
{
    private readonly ITestOutputHelper _out;
    public CudaAdmissionGpuTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        foreach (var dir in new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }

    private static string? FindGguf() =>
        TestFixtureResolver.ResolveFile("DOTLLM_QWEN25_05B_Q8_GGUF", "Qwen", "Qwen2.5-0.5B-Instruct-GGUF", "*q8_0.gguf").Path;

    private static (string Ptx, string Gguf) Require()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        string? path = FindGguf();
        Skip.If(path is null, "Qwen2.5-0.5B q8_0 GGUF not found.");
        return (ptxDir!, path!);
    }

    private static int[] Tokens(int n, int vocab) => Enumerable.Range(0, n).Select(i => 100 + (i * 37) % (vocab - 200)).ToArray();
    private static int[] Positions(int n) => Enumerable.Range(0, n).ToArray();

    private static unsafe float[] Read(DotLLM.Core.Tensors.ITensor t)
        => new ReadOnlySpan<float>((void*)t.DataPointer, checked((int)t.Shape.ElementCount)).ToArray();

    [SkippableFact]
    public void Weights_ThatDoNotFitFreeVram_AreRefused_NotWarned()
    {
        var (ptx, path) = Require();
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var before = CudaAdmission.FreeBytesOverride;
        CudaAdmission.FreeBytesOverride = () => 64L << 20;   // a card with 64 MiB free: the ~500 MiB of weights cannot fit
        try
        {
            var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx).Dispose());
            _out.WriteLine(ex.Message);
            Assert.StartsWith("Not enough free GPU memory for the model weights", ex.Message);
            Assert.Contains("--gpu-layers", ex.Message);
        }
        finally { CudaAdmission.FreeBytesOverride = before; }

        // The refused load released its context: a normal load right after works.
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        Assert.True(model.HasSpillMonitor);
    }

    [SkippableFact]
    public void ForwardScratchGrowth_IsRefusedBeforeTheOldBuffersAreFreed()
    {
        var (ptx, path) = Require();
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        int n = 300;
        float[] reference;
        using (var kv = model.CreateKvCache(512))
        using (var logits = model.Forward(Tokens(8, config.VocabSize), Positions(8), 0, kv)) { }

        using (var kv = model.CreateKvCache(512))
        {
            var before = CudaAdmission.FreeBytesOverride;
            CudaAdmission.FreeBytesOverride = () => 0;   // nothing free: the 8 -> 300-token growth must be refused
            try
            {
                var ex = Assert.Throws<DeviceMemoryAdmissionException>(() => model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv).Dispose());
                _out.WriteLine(ex.Message);
                Assert.StartsWith($"Not enough free GPU memory for the forward scratch for a {n}-token forward pass", ex.Message);
                Assert.Contains("--prefill-chunk", ex.Message);
            }
            finally { CudaAdmission.FreeBytesOverride = before; }
        }

        // The refusal left the state intact at its old capacity: the same forward now runs and is deterministic.
        using (var kv = model.CreateKvCache(512))
        using (var logits = model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv))
            reference = Read(logits);
        using (var kv = model.CreateKvCache(512))
        using (var logits = model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv))
            Assert.Equal(reference, Read(logits));
    }

    [SkippableFact]
    public void G3ScratchThatDoesNotFit_FallsBackToAttentionF16()
    {
        var (ptx, path) = Require();
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = CudaTransformerModel.LoadFromGguf(gguf, config, 0, ptx);
        bool savedEnabled = CudaG3Attention.Enabled, savedF16Flash = CudaTransformerModel.EnableF16FlashPrefill;
        int n = 400;
        try
        {
            CudaTransformerModel.EnableF16FlashPrefill = false;   // #1923: a G3 test; the F16 flash prefill would take the call first
            // Reference: G3 off (attention_f16), which also grows the forward scratch to n so the arm below allocates nothing else.
            CudaG3Attention.Enabled = false;
            float[] f16;
            using (var kv = model.CreateKvCache(512))
            using (var logits = model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv))
                f16 = Read(logits);

            CudaG3Attention.Enabled = true;
            Skip.IfNot(model.G3CanUse(n), "G3 does not apply to this model/device");
            float[] fallback;
            long refusals = CudaG3Attention.ScratchRefusalCount;
            using (var kv = model.CreateKvCache(512))
            {
                var before = CudaAdmission.FreeBytesOverride;
                CudaAdmission.FreeBytesOverride = () => 0;   // the KV cache is already allocated; only the G3 scratch asks now
                try
                {
                    using var logits = model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv);
                    fallback = Read(logits);
                }
                finally { CudaAdmission.FreeBytesOverride = before; }
            }
            long declined = CudaG3Attention.ScratchRefusalCount - refusals;
            _out.WriteLine($"G3 scratch refusals +{declined}");
            Assert.True(declined >= config.NumLayers, "every layer's G3 call should have been declined");
            Assert.Equal(f16, fallback);   // the fallback is exactly the attention_f16 path
            Assert.Equal(0, model.G3ScratchElems);

            // With room again, G3 runs (its output differs from attention_f16 in the last bits) and a large plane is released after.
            using (var kv = model.CreateKvCache(512))
            using (var logits = model.Forward(Tokens(n, config.VocabSize), Positions(n), 0, kv)) { }
            Assert.True(model.G3ScratchElems > 0, "G3 should have allocated its scratch when it fits");

            // #1848 review: a retained plane is given back when a KV cache is refused and the plane's bytes would make it fit. The
            // fake device is one byte short while the plane is held (so releasing the plane can help) and has room once it is gone,
            // so the cache is created iff the model released it.
            long kvBytes = CudaKvCache.BytesFor(DotLLM.Core.Attention.KvGeometry.FromConfig(config), 512);
            long shortBy1 = CudaAdmission.ReserveBytes + kvBytes - 1;
            var saved = CudaAdmission.FreeBytesOverride;
            CudaAdmission.FreeBytesOverride = () => model.G3ScratchElems > 0 ? shortBy1 : 4L << 30;
            try
            {
                using var kv = model.CreateKvCache(512);
                Assert.Equal(0, model.G3ScratchElems);
            }
            finally { CudaAdmission.FreeBytesOverride = saved; }
        }
        finally
        {
            CudaG3Attention.Enabled = savedEnabled;
            CudaTransformerModel.EnableF16FlashPrefill = savedF16Flash;
        }
    }
}
