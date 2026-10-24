using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Opt-in real-weights check (#1172): the 26 GB Q4_1 Flint file cannot fit a 12 GB GPU (that is F7), so this truncates the
/// model to its first <c>DOTLLM_FLINT_CUDA_LAYERS</c> layers (default 8: the 2 dense layers, sliding and NoPE full MoE
/// layers) and compares CPU vs CUDA logits on the same truncated stack, then times prefill / decode and reports VRAM and
/// host memory. The logits of a truncated stack are meaningless as language, but CPU-vs-CUDA agreement on them exercises
/// every real tensor shape (hidden 2048, head_dim 128, 4 KV heads, 256 experts, Q4_1 everywhere). Set
/// <c>DOTLLM_FLINT_GGUF</c> to the file.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaFlintRealWeightsProbe
{
    private readonly ITestOutputHelper _out;

    public CudaFlintRealWeightsProbe(ITestOutputHelper output) => _out = output;

    [SkippableFact]
    public void TruncatedStack_CpuVsCuda_AndMemory()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_FLINT_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "set DOTLLM_FLINT_GGUF to the Flint GGUF");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        int layers = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_FLINT_CUDA_LAYERS"), out int l) ? l : 8;

        using var gguf = GgufFile.Open(path!);
        ModelConfig full = GgufModelConfigExtractor.Extract(gguf.Metadata);
        ModelConfig cfg = full with { NumLayers = layers };
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tok.Encode("def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    if n < 2:\n        return n\n").ToArray();
        int n = ids.Length;
        int[] pos = Enumerable.Range(0, n).ToArray();
        _out.WriteLine($"layers {layers}/{full.NumLayers}, tokens {n}, sliding {string.Join(',', cfg.PerLayerSlidingWindow!.Take(layers).Select(w => w?.ToString() ?? "-"))}, "
            + $"NoPE {string.Join(',', cfg.NoRopeLayers!.Where(x => x < layers))}");

        float[] cpu;
        using (var model = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.Auto))
        {
            using ITensor t = model.Forward(ids, pos, deviceId: -1);
            cpu = Copy(t);
        }
        int vocab = full.VocabSize;

        CudaMemoryInfo.TryGet(0, out long free0, out long total);
        long priv0 = Process.GetCurrentProcess().PrivateMemorySize64;
        var sw = Stopwatch.StartNew();
        var (cuda, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0);
        double loadS = sw.Elapsed.TotalSeconds;
        using (cuda)
        {
            CudaMemoryInfo.TryGet(0, out long freeLoaded, out _);
            long privLoaded = Process.GetCurrentProcess().PrivateMemorySize64;
            using var kv = kvFactory(4096);
            CudaMemoryInfo.TryGet(0, out long freeKv, out _);

            int prefill = n - 8;
            sw.Restart();
            float[] pre;
            using (ITensor p = cuda.Forward(ids.AsSpan(0, prefill), pos.AsSpan(0, prefill), deviceId: -1, kv))
                pre = Copy(p);
            double prefillMs = sw.Elapsed.TotalMilliseconds;
            Report($"prefill@{prefill - 1}", cpu.AsSpan((prefill - 1) * vocab, vocab).ToArray(), pre);

            var decodeMs = new List<double>();
            for (int i = prefill; i < n; i++)
            {
                if (i == n - 1 && cuda is CudaTransformerModel ctm)
                    ctm.ProfilingEnabled = true;
                sw.Restart();
                using ITensor d = cuda.Forward(ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, kv);
                float[] row = Copy(d);
                decodeMs.Add(sw.Elapsed.TotalMilliseconds);
                Report($"decode@{i}", cpu.AsSpan(i * vocab, vocab).ToArray(), row);
            }
            CudaMemoryInfo.TryGet(0, out long freeRun, out _);
            long priv1 = Process.GetCurrentProcess().PrivateMemorySize64;

            if (cuda is CudaTransformerModel prof)
                _out.WriteLine($"profiled decode step: wall {decodeMs[^1]:F2} ms, GPU first-to-last launch {prof.LastGpuLaunchMs:F2} ms; per category "
                    + string.Join(", ", prof.LastCategoryMs.Select((ms, c) => $"{(CudaTransformerModel.ProfileCategory)c}={ms:F2}")));
            double steadyMs = decodeMs.Skip(2).SkipLast(1).Average();
            _out.WriteLine($"load {loadS:F1} s; prefill {prefill} tok {prefillMs:F0} ms (incl. first-call JIT); decode {steadyMs:F2} ms/token over {layers} layers "
                + $"=> {steadyMs / layers:F3} ms/layer");
            _out.WriteLine($"VRAM used (device-wide, of {total / 1048576} MiB): weights+state {(free0 - freeLoaded) / 1048576} MiB, "
                + $"+KV(4096) {(freeLoaded - freeKv) / 1048576} MiB, +run scratch {(freeKv - freeRun) / 1048576} MiB, total {(free0 - freeRun) / 1048576} MiB");
            _out.WriteLine($"host private bytes: before CUDA load {priv0 / 1048576} MiB, after load {privLoaded / 1048576} MiB, after run {priv1 / 1048576} MiB, "
                + $"peak working set {Process.GetCurrentProcess().PeakWorkingSet64 / 1048576} MiB");
        }

        // Calibration: does a plain cuMemAlloc of device memory show up in this process's private bytes (WDDM)?
        using (var ctx = CudaContext.Create(0))
        {
            long p0 = Process.GetCurrentProcess().PrivateMemorySize64;
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)(2L << 30)).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(d, 0, (nuint)(2L << 30)).ThrowOnError();
            long p1 = Process.GetCurrentProcess().PrivateMemorySize64;
            CudaDriverApi.cuMemFree_v2(d);
            _out.WriteLine($"calibration: 2048 MiB cuMemAlloc moves private bytes by {(p1 - p0) / 1048576} MiB (after model dispose: {p0 / 1048576} MiB)");
        }
    }

    private void Report(string what, float[] cpu, float[] cuda)
    {
        Assert.All(cuda, v => Assert.True(float.IsFinite(v), $"{what}: non-finite CUDA logit"));
        double maxDiff = 0, dot = 0, nc = 0, ng = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            maxDiff = Math.Max(maxDiff, Math.Abs(cpu[i] - cuda[i]));
            dot += (double)cpu[i] * cuda[i];
            nc += (double)cpu[i] * cpu[i];
            ng += (double)cuda[i] * cuda[i];
        }
        int topCpu = Array.IndexOf(cpu, cpu.Max()), topCuda = Array.IndexOf(cuda, cuda.Max());
        _out.WriteLine($"{what,-12} max|diff| {maxDiff:F4}  cos {dot / Math.Sqrt(nc * ng):F6}  KL(cpu||cuda) {Kl(cpu, cuda):E2}  top1 {topCpu}/{topCuda}");
    }

    private static double Kl(float[] p, float[] q)
    {
        double mp = p.Max(), mq = q.Max(), zp = 0, zq = 0;
        foreach (float v in p) zp += Math.Exp(v - mp);
        foreach (float v in q) zq += Math.Exp(v - mq);
        double kl = 0;
        for (int i = 0; i < p.Length; i++)
        {
            double lp = p[i] - mp - Math.Log(zp), lq = q[i] - mq - Math.Log(zq);
            kl += Math.Exp(lp) * (lp - lq);
        }
        return kl;
    }

    private static float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }
}
