using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.Gguf;

/// <summary>
/// One structured per-phase timing record (CSV-friendly). <see cref="ToCsv"/> emits
/// <c>phase,name,ms,tokensPerSec</c> (tokens/sec blank for non-prefill phases).
/// </summary>
public readonly record struct PhaseTiming(string Phase, string Name, double Milliseconds, double TokensPerSec)
{
    /// <summary>CSV line: <c>phase,name,ms,tokens_per_sec</c>.</summary>
    public string ToCsv() => $"{Phase},{Name},{Milliseconds:F3},{(TokensPerSec > 0 ? TokensPerSec.ToString("F1") : "")}";

    /// <summary>CSV header for a stream of <see cref="PhaseTiming"/> rows.</summary>
    public static string CsvHeader => "phase,name,ms,tokens_per_sec";
}

/// <summary>
/// Stopwatch-based per-phase timing harness for the synthetic Gemma-4 / DiffusionGemma fixture.
/// Times fixture generation, model load, a warmup forward, then N prefill forwards (with
/// tokens/sec) and M diffusion denoise steps, emitting one <see cref="PhaseTiming"/> per phase.
/// Reusable by the integration timing test and by <c>benchmarks/DotLLM.Benchmarks</c>.
/// </summary>
/// <remarks>
/// <para><b>Per-stage timing.</b> This is COARSE per-forward timing (Stopwatch around the whole
/// <c>model.Forward</c>). Finer per-stage (attention / FFN / MoE) timing would use the
/// <c>IInferenceHook</c>/<c>HookPoint</c> diagnostics seam; that is left to the dedicated
/// BenchmarkDotNet project where the measurement noise floor is controlled. Coarse per-forward
/// timing is sufficient to compare backends on the same generated <c>.gguf</c>.</para>
/// <para><b>Cross-backend.</b> The harness drives the CPU <see cref="IModel"/> from
/// <see cref="ModelLoader.LoadFromGguf(string, ThreadingConfig?, DiffusionConfig?)"/>. To time
/// the SAME fixture on Vulkan/CUDA/HIP, point those backends' GGUF loaders at the emitted
/// <c>.gguf</c> path (see <see cref="SyntheticGemma4Gguf"/> Write* helpers) and wrap their
/// forward in the same Stopwatch pattern, or register the fixture path with a
/// <c>BenchmarkDotNet</c> <c>[Benchmark]</c> parameterised by backend.</para>
/// </remarks>
public static class SyntheticGemma4Harness
{
    /// <summary>
    /// Runs the full timing sequence for one preset and returns the per-phase records. The
    /// fixture is generated to a temp file, loaded, warmed up, then timed.
    /// </summary>
    /// <param name="config">Size preset (e.g. <see cref="SyntheticGemma4Gguf.Tiny"/> / <c>.Bench</c>).</param>
    /// <param name="presetName">Label for the phase rows (e.g. "tiny").</param>
    /// <param name="prefillForwards">Number of timed prefill forwards.</param>
    /// <param name="prefillTokens">Prompt length per prefill forward.</param>
    /// <param name="diffusionSteps">Number of timed diffusion-style canvas forwards (Hybrid mask).</param>
    /// <param name="canvasLength">Canvas length for the diffusion forwards.</param>
    /// <param name="seed">PRNG seed for the fixture weights.</param>
    public static IReadOnlyList<PhaseTiming> Run(
        SyntheticGemma4Config config, string presetName,
        int prefillForwards = 5, int prefillTokens = 16,
        int diffusionSteps = 4, int canvasLength = 8, uint seed = 0xC0FFEEu)
    {
        var rows = new List<PhaseTiming>();
        string path = Path.Combine(Path.GetTempPath(), $"syn_gemma4_bench_{Guid.NewGuid():N}.gguf");
        try
        {
            // 1) Fixture generation (gemma4 AR variant).
            var sw = Stopwatch.StartNew();
            byte[] bytes = SyntheticGemma4Gguf.BuildGemma4(config, seed);
            File.WriteAllBytes(path, bytes);
            sw.Stop();
            rows.Add(new PhaseTiming("gen", presetName, sw.Elapsed.TotalMilliseconds, 0));

            // 2) Load.
            sw.Restart();
            var (model, gguf, modelConfig) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
            sw.Stop();
            rows.Add(new PhaseTiming("load", presetName, sw.Elapsed.TotalMilliseconds, 0));

            using (gguf)
            using (model)
            {
                int vocab = modelConfig.VocabSize;

                // 3) Warmup forward (JIT + first-touch page faults; not counted in prefill avg).
                int[] warm = MakeIds(prefillTokens, vocab, modelConfig);
                int[] warmPos = Positions(prefillTokens);
                sw.Restart();
                using (ITensor _ = model.Forward(warm, warmPos, -1, null, null, AttentionMaskSpec.Causal)) { }
                sw.Stop();
                rows.Add(new PhaseTiming("warmup", presetName, sw.Elapsed.TotalMilliseconds, 0));

                // 4) N prefill forwards (tokens/sec = prefillTokens / forward time).
                int[] pos = Positions(prefillTokens);
                double totalMs = 0;
                for (int it = 0; it < prefillForwards; it++)
                {
                    int[] ids = MakeIds(prefillTokens, vocab, modelConfig, salt: it + 1);
                    sw.Restart();
                    using (ITensor _ = model.Forward(ids, pos, -1, null, null, AttentionMaskSpec.Causal)) { }
                    sw.Stop();
                    double ms = sw.Elapsed.TotalMilliseconds;
                    totalMs += ms;
                    double tps = ms > 0 ? prefillTokens / (ms / 1000.0) : 0;
                    rows.Add(new PhaseTiming("prefill", $"{presetName}#{it}", ms, tps));
                }
                double avgMs = prefillForwards > 0 ? totalMs / prefillForwards : 0;
                double avgTps = avgMs > 0 ? prefillTokens / (avgMs / 1000.0) : 0;
                rows.Add(new PhaseTiming("prefill_avg", presetName, avgMs, avgTps));

                // 5) M diffusion-style canvas forwards: [prompt | canvas] under Hybrid(promptLen).
                int promptLen = Math.Max(1, prefillTokens / 2);
                int seqLen = promptLen + canvasLength;
                int[] diffPos = Positions(seqLen);
                int maskId = modelConfig.DiffusionConfig?.MaskTokenId ?? config.MaskTokenId;
                for (int s = 0; s < diffusionSteps; s++)
                {
                    int[] seq = MakeCanvasIds(promptLen, canvasLength, vocab, maskId, modelConfig, salt: s + 1);
                    sw.Restart();
                    using (ITensor _ = model.Forward(seq, diffPos, -1, null, null,
                        AttentionMaskSpec.Hybrid(promptLen))) { }
                    sw.Stop();
                    rows.Add(new PhaseTiming("diffusion_step", $"{presetName}#{s}", sw.Elapsed.TotalMilliseconds, 0));
                }
            }
        }
        finally
        {
            try { File.Delete(path); } catch { /* best-effort temp cleanup */ }
        }
        return rows;
    }

    private static int[] Positions(int n)
    {
        var p = new int[n];
        for (int i = 0; i < n; i++) p[i] = i;
        return p;
    }

    /// <summary>Deterministic in-range token ids (BOS first), avoiding any special tokens beyond BOS.</summary>
    private static int[] MakeIds(int n, int vocab, ModelConfig config, int salt = 0)
    {
        var ids = new int[n];
        int bos = BosId(config);
        ids[0] = bos;
        uint state = 0x1234567u + (uint)salt * 2654435761u;
        for (int i = 1; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ids[i] = (int)(state % (uint)vocab);
        }
        return ids;
    }

    private static int[] MakeCanvasIds(int promptLen, int canvas, int vocab, int maskId, ModelConfig config, int salt)
    {
        var ids = new int[promptLen + canvas];
        var prompt = MakeIds(promptLen, vocab, config, salt);
        Array.Copy(prompt, ids, promptLen);
        for (int i = promptLen; i < ids.Length; i++) ids[i] = maskId;
        return ids;
    }

    private static int BosId(ModelConfig config) => 2; // synthetic fixture uses BOS id 2
}
