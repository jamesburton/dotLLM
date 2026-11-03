using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Gguf;

public sealed class ScratchGlmProbe
{
    [SkippableFact]
    public void Probe()
    {
        string? path = Environment.GetEnvironmentVariable("GLM_PROBE_GGUF");
        Skip.If(string.IsNullOrEmpty(path), "scratch");
        string idsPath = Environment.GetEnvironmentVariable("GLM_PROBE_IDS")!;
        int row = int.Parse(Environment.GetEnvironmentVariable("GLM_PROBE_POS")!);
        int[] all = File.ReadAllText(idsPath).Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).Take(512).ToArray();
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = (TransformerModel)ModelLoader.CreateCpuModelFromGguf(gguf, cfg, new ThreadingConfig(32));
        int H = cfg.HiddenSize;
        float[] Run(int n, int layers)
        {
            int[] ids = all.Take(n).ToArray();
            int[] pos = Enumerable.Range(0, n).ToArray();
            var outp = new float[n * H];
            model.ForwardLayerWindow(ids, ReadOnlySpan<float>.Empty, pos, 0, layers, outp);
            return outp.AsSpan(row * H, H).ToArray();
        }
        foreach (int L in new[] { 3, 4, 5, 6, 8 })
        {
            var a = Run(512, L);
            var b = Run(512, L);
            double num = 0, den = 0, amax = 0; int amaxi = 0;
            for (int i = 0; i < H; i++)
            {
                double d = a[i] - b[i]; num += d * d; den += (double)a[i] * a[i];
                if (Math.Abs(a[i]) > amax) { amax = Math.Abs(a[i]); amaxi = i; }
            }
            Console.WriteLine($"PROBE layers={L}: relL2 diff(n={row + 1} vs n=512) = {Math.Sqrt(num / den):G3}  |h|rms={Math.Sqrt(den / H):G4} amax={amax:G4}@{amaxi}");
        }
    }
}
