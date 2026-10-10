using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Real-file QSA measurement harness (issue #819). Skipped unless <c>DOTLLM_QWEN4EXP_REAL_GGUF</c> names shard 1 of the released file
/// (e.g. the Unsloth UD-Q4_K_XL). One model load serves every probe: needle-in-a-haystack prompts from
/// <c>DOTLLM_QWEN4EXP_NEEDLE_DIR</c> (<c>needle_*.txt</c>, answer code in <c>needle_*.answer</c> or the file name table below), chunked-prefill and
/// decode throughput at several context depths, and the device memory held. GPU box only: hold <c>scripts/gpu-lock.sh</c>.
/// </summary>
[Trait("Category", "GPU")]
[Trait("Category", "RealFile")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpRealQsaTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpRealQsaTests(ITestOutputHelper output) => _out = output;

    private static readonly string[] Codes = ["plum-7391-harbor", "cedar-5520-lantern", "violet-8264-anchor"];

    private void Log(string s)
    {
        _out.WriteLine(s);
        string? path = Environment.GetEnvironmentVariable("DOTLLM_REAL_RESULTS");
        if (!string.IsNullOrEmpty(path)) File.AppendAllText(path, s + Environment.NewLine);
    }

    private static unsafe float[] LastRow(DotLLM.Core.Tensors.ITensor t)
    {
        using (t)
        {
            int v = t.Shape[1], rows = t.Shape[0];
            return new ReadOnlySpan<float>((void*)(t.DataPointer + (nint)((long)(rows - 1) * v * 4)), v).ToArray();
        }
    }

    private static int Argmax(float[] v) { int b = 0; for (int i = 1; i < v.Length; i++) if (v[i] > v[b]) b = i; return b; }

    [SkippableFact]
    public void LoadAndOneForward()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        Log($"[smoke] load {sw.Elapsed.TotalSeconds:F1}s, capacity {model.ContextCapacity}, QSA enabled {VulkanQwen4ExpQsa.Enabled}, {device.MemorySnapshot()}");
        using var st = model.CreateState();
        var row = LastRow(model.Forward(Enumerable.Range(1000, 64).ToArray(), Enumerable.Range(0, 64).ToArray(), -1, st));
        Log($"[smoke] forward ok, argmax {Argmax(row)}, {device.MemorySnapshot()}");
    }

    [SkippableFact]
    public void NeedlesThroughputAndMemory()
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_REAL_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_QWEN4EXP_REAL_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        string? needleDir = Environment.GetEnvironmentVariable("DOTLLM_QWEN4EXP_NEEDLE_DIR");
        int chunk = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_REAL_CHUNK"), out int c) && c > 0 ? c : 1024;

        var sw = Stopwatch.StartNew();
        using var gguf = GgufFile.Open(path!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var device = VulkanDevice.Create();
        using var model = VulkanQwen4ExpTransformerModel.BuildFromGguf(device, gguf, cfg, spvDir);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        Log($"[real] load {sw.Elapsed.TotalSeconds:F1}s, context capacity {model.ContextCapacity}, dense limit {model.DenseContextLimit}, {device.MemorySnapshot()}");

        int V = cfg.VocabSize;
        float[] Prefill(VulkanQwen4ExpSequenceState st, int[] ids, out double sec)
        {
            var t0 = Stopwatch.GetTimestamp();
            float[]? last = null;
            for (int a = 0; a < ids.Length; a += chunk)
            {
                int n = Math.Min(chunk, ids.Length - a);
                last = LastRow(model.Forward(ids.AsSpan(a, n), Enumerable.Range(a, n).ToArray(), -1, st));
            }
            sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            return last!;
        }

        // Warm-up (shader/pipeline first use, weight page-in) so the timed figures below are steady state.
        using (var warm = model.CreateState())
            Prefill(warm, Enumerable.Range(1000, 256).ToArray(), out _);

        // Same-session A/B of what the indexer costs at short context (decode at depth 1024, indexer off vs on, alternating).
        {
            var rngAb = new Random(7);
            int[] fill = Enumerable.Range(0, 1100).Select(_ => rngAb.Next(1000, 20000)).ToArray();
            VulkanQwen4ExpQsa.Enabled = false;
            using var off = model.CreateState();
            Prefill(off, fill.AsSpan(0, 1024).ToArray(), out _);
            VulkanQwen4ExpQsa.Enabled = true;
            using var on = model.CreateState();
            Prefill(on, fill.AsSpan(0, 1024).ToArray(), out _);
            double[] offMs = new double[3], onMs = new double[3];
            for (int round = 0; round < 3; round++)
            {
                foreach (bool useIndexer in new[] { false, true })
                {
                    VulkanQwen4ExpQsa.Enabled = useIndexer;
                    var st = useIndexer ? on : off;
                    int keep = st.Length;
                    var t0 = Stopwatch.GetTimestamp();
                    for (int i = 0; i < 24; i++)
                        LastRow(model.Forward([fill[(keep + i) % fill.Length]], [st.Length], -1, st));
                    double ms = Stopwatch.GetElapsedTime(t0).TotalMilliseconds / 24;
                    (useIndexer ? onMs : offMs)[round] = ms;
                }
            }
            VulkanQwen4ExpQsa.Enabled = true;
            Log($"[ab] decode @1024, ms/token: indexer OFF [{string.Join(", ", offMs.Select(x => x.ToString("F1")))}] ON [{string.Join(", ", onMs.Select(x => x.ToString("F1")))}]");
        }

        int passed = 0, total = 0;
        if (!string.IsNullOrEmpty(needleDir) && Directory.Exists(needleDir))
        {
            foreach (string file in Directory.GetFiles(needleDir, "needle_*.txt").OrderBy(x => x))
            {
                string name = Path.GetFileNameWithoutExtension(file);
                int idx = int.Parse(name[^1..]);
                string code = Codes[idx];
                int[] ids = tokenizer.Encode(File.ReadAllText(file));
                if (ids.Length + 40 > model.ContextCapacity)
                {
                    Log($"[real] {name}: {ids.Length} tokens exceed the capacity {model.ContextCapacity} - skipped");
                    continue;
                }
                using var st = model.CreateState();
                var logits = Prefill(st, ids, out double psec);
                var gen = new List<int>();
                var d0 = Stopwatch.GetTimestamp();
                for (int i = 0; i < 24; i++)
                {
                    int next = Argmax(logits);
                    gen.Add(next);
                    logits = LastRow(model.Forward([next], [st.Length], -1, st));
                }
                double dsec = Stopwatch.GetElapsedTime(d0).TotalSeconds;
                string text = tokenizer.Decode(gen.ToArray());
                bool ok = text.Contains(code, StringComparison.OrdinalIgnoreCase);
                total++; if (ok) passed++;
                Log($"[needle] {name}: {ids.Length} tokens, prefill {psec:F1}s ({ids.Length / psec:F0} tok/s), decode {24 / dsec:F1} tok/s, " +
                    $"answer \"{text.Replace("\n", " ")}\" expected \"{code}\" -> {(ok ? "PASS" : "FAIL")}");
            }
            Log($"[needle] {passed}/{total} passed");
        }

        // Decode throughput vs context depth (same state, growing): 1K, 2K (dense limit), 4K, 6K, 8K.
        using (var st = model.CreateState())
        {
            var rng = new Random(1);
            int[] filler = Enumerable.Range(0, model.ContextCapacity).Select(_ => rng.Next(1000, 20000)).ToArray();
            float[]? logits = null;
            int at = 0;
            foreach (int depth in new[] { 1024, 2048, 4096, 8000, 12000, 16000 })
            {
                if (depth + 40 > model.ContextCapacity) break;
                if (depth > at)
                {
                    var t0 = Stopwatch.GetTimestamp();
                    for (int a = at; a < depth; a += chunk)
                    {
                        int n = Math.Min(chunk, depth - a);
                        logits = LastRow(model.Forward(filler.AsSpan(a, n), Enumerable.Range(a, n).ToArray(), -1, st));
                        at = a + n;
                    }
                    double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
                    Log($"[prefill] to depth {depth}: {sec:F1}s");
                }
                var d0 = Stopwatch.GetTimestamp();
                const int Steps = 24;
                for (int i = 0; i < Steps; i++)
                {
                    logits = LastRow(model.Forward([filler[at % filler.Length]], [st.Length], -1, st));
                    at++;
                }
                double dsec = Stopwatch.GetElapsedTime(d0).TotalSeconds;
                Log($"[decode] depth ~{depth}: {Steps / dsec:F1} tok/s ({dsec / Steps * 1000:F1} ms/token), state length {st.Length}");
            }
        }
        Log($"[real] done in {sw.Elapsed.TotalSeconds:F0}s, {device.MemorySnapshot()}");
        if (total > 0) Assert.True(passed == total, $"needles {passed}/{total}");
    }
}
