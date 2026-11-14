using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models;
using DotLLM.Server;
using DotLLM.Tests.Unit.Hip;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Server;

[Collection("HipGpu")]
public sealed class HipSafetensorsApplicationTests(ITestOutputHelper output)
{
    [SkippableFact]
    [Trait("Category", "GPU")]
    public void RealHfConfigRoute_MatchesCpu_AndWrongRotaryLayoutIsDetected()
    {
        string? directory = Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_HF_MODEL");
        Skip.If(string.IsNullOrEmpty(directory), "Set DOTLLM_TEST_HIP_HF_MODEL to a genuine F16 HF checkpoint (offline BF16-to-F16 derivatives are allowed with provenance).");
        var (cpu, cpuSource) = DeviceModelLoader.LoadSafetensors(directory!, "cpu", null,
            new ThreadingConfig(1, EnableCallerPinning: false), _ => { }, _ => { });
        using (cpuSource)
        using (cpu.Model)
        {
            var (gpu, gpuSource) = DeviceModelLoader.LoadSafetensors(directory!, "hip:0", null, default, _ => { }, _ => { });
            using (gpuSource)
            using (gpu.Model)
            {
                var model = Assert.IsType<HipTransformerModel>(gpu.Model);
                var config = model.Config;
                Assert.Equal(RoPEType.NeoX, config.RoPEConfig!.Value.Type);
                Assert.Equal(RoPEType.NeoX, cpu.Model.Config.RoPEConfig!.Value.Type);
                var tokenizer = ModelLoader.LoadTokenizerFromHfDirectory(directory!)!;
                int[] input = tokenizer.Encode("The capital of France is Paris. The city is known for its architecture, museums, and");
                int[] firstInput = input;
                using var cpuCache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 128);
                using var gpuCache = model.CreateKvCache(128);
                int position = 0;
                double maxRelative = 0, maxKl = 0;
                int top1Mismatches = 0;
                double largestMismatchMargin = 0;
                float[] firstReference = [];
                for (int step = 0; step < 12; step++)
                {
                    int[] positions = Enumerable.Range(position, input.Length).ToArray();
                    using var reference = cpu.Model.Forward(input, positions, -1, cpuCache);
                    using var actual = model.Forward(input, positions, 0, gpuCache, lastTokenLogitsOnly: true);
                    float[] expected = ReadLast(reference, config.VocabSize), got = ReadLast(actual, config.VocabSize);
                    if (step == 0) firstReference = expected;
                    double relative = RelativeL2(expected, got), kl = Kl(expected, got);
                    maxRelative = Math.Max(maxRelative, relative); maxKl = Math.Max(maxKl, kl);
                    Assert.True(relative < 0.02, $"HF extracted-config step {step}: relative L2 {relative:G6}");
                    Assert.True(kl < 0.001, $"HF extracted-config step {step}: KL {kl:G6}");
                    int token = Array.IndexOf(expected, expected.Max());
                    int actualToken = Array.IndexOf(got, got.Max());
                    if (actualToken != token)
                    {
                        top1Mismatches++;
                        largestMismatchMargin = Math.Max(largestMismatchMargin, expected[token] - expected[actualToken]);
                        output.WriteLine($"Mismatch logits: CPU[{token}]={expected[token]:G9}, CPU[{actualToken}]={expected[actualToken]:G9}; HIP[{token}]={got[token]:G9}, HIP[{actualToken}]={got[actualToken]:G9}.");
                        Assert.True(IsF16NearTie(expected[token], expected[actualToken]),
                            $"HF step {step}: HIP winner trails CPU winner by more than one F16 ULP.");
                    }
                    output.WriteLine($"HF step {step}: L2={relative:G6}, KL={kl:G6}, CPU top1={token}, HIP top1={actualToken}, CPU logit margin={expected[token] - expected[actualToken]:G6}.");
                    position += input.Length;
                    input = [token];
                }

                // This deliberate alternative uses the GGUF adjacent-pair convention on unchanged HF Q/K rows.
                // Real data must distinguish it; an HF-shaped name-only copy of a GGUF could hide this error.
                using var wrong = HipTransformerModel.LoadFromSafetensors(gpuSource,
                    config with { RoPEConfig = config.RoPEConfig.Value with { Type = RoPEType.Norm } }, 0);
                using var wrongCache = wrong.CreateKvCache(128);
                using var wrongLogits = wrong.Forward(firstInput, Enumerable.Range(0, firstInput.Length).ToArray(), 0, wrongCache, lastTokenLogitsOnly: true);
                double wrongRelative = RelativeL2(firstReference, ReadLast(wrongLogits, config.VocabSize));
                Assert.True(wrongRelative > 0.02 && wrongRelative > maxRelative * 10,
                    $"The real HF fixture did not distinguish wrong rotary layout: correct max {maxRelative:G6}, wrong {wrongRelative:G6}");
                output.WriteLine($"Genuine HF config path: {12 - top1Mismatches}/12 top1, max relative L2={maxRelative:G6}, max KL={maxKl:G6}; wrong Norm layout relative L2={wrongRelative:G6}; largest top1 mismatch margin={largestMismatchMargin:G6}.");
                Assert.InRange(top1Mismatches, 0, 1);
            }
        }
    }

    // F16 logits can reverse near ties; use spacing at the CPU winner, including subnormals.
    private static bool IsF16NearTie(float winner, float contender)
        => winner - contender <= MathF.ScaleB(1f, Math.Max(-14, MathF.ILogB(MathF.Abs(winner))) - 10);

    [Theory]
    [InlineData(17.25f, 17.234375f, true)]
    [InlineData(17.25f, 17.234373f, false)]
    [InlineData(-17.25f, -17.265625f, true)]
    [InlineData(-17.25f, -17.265627f, false)]
    [InlineData(1.9990234375f, 1.998046875f, true)]
    [InlineData(1.9990234375f, 1.9970703125f, false)]
    [InlineData(0f, -0.000000059604644775390625f, true)]
    [InlineData(0f, -0.00000011920928955078125f, false)]
    public void F16NearTie_AcceptsOneUlp_AndRejectsLargerReversals(float winner, float contender, bool accepted)
        => Assert.Equal(accepted, IsF16NearTie(winner, contender));

    private static unsafe float[] ReadLast(ITensor tensor, int vocab)
        => new ReadOnlySpan<float>((float*)tensor.DataPointer + (tensor.Shape[0] - 1) * vocab, vocab).ToArray();

    private static double RelativeL2(float[] expected, float[] actual)
    {
        double error = 0, denominator = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"Nonfinite logit {i}");
            error += Math.Pow(expected[i] - actual[i], 2);
            denominator += expected[i] * (double)expected[i];
        }
        return Math.Sqrt(error / denominator);
    }

    private static double Kl(float[] expected, float[] actual)
    {
        double maxA = expected.Max(), maxB = actual.Max();
        double sumA = expected.Sum(x => Math.Exp(x - maxA)), sumB = actual.Sum(x => Math.Exp(x - maxB));
        double logA = maxA + Math.Log(sumA), logB = maxB + Math.Log(sumB), kl = 0;
        for (int i = 0; i < expected.Length; i++)
            kl += Math.Exp(expected[i] - logA) * (expected[i] - logA - actual[i] + logB);
        return kl;
    }

    [SkippableTheory]
    [InlineData("f16", "f16", 2064)]
    [InlineData("q8_0", "f16", 1584)]
    [InlineData("f16", "q8_0", 1584)]
    [InlineData("q8_0", "q8_0", 1104)]
    [Trait("Category", "GPU")]
    public async Task ServerRequests_PreserveCacheTypes_ReleasePagesOnCancellation_AndRemainIsolated(
        string key, string value, long expectedPageBytes)
    {
        bool available = OperatingSystem.IsWindows() && HipDevice.IsAvailable();
        if (Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED") is "1") Assert.True(available);
        Skip.IfNot(available, "Windows HIP/DirectStorage is unavailable.");
        using var environment = new HipQ8Environment(key is "q8_0" || value is "q8_0" ? "1" : null);
        using var fixture = new HfApplicationFixture();
        string directory = fixture.WriteCheckpoint("hip-cancel");
        using var state = ServerStartup.LoadModel(directory, new ServerOptions
        {
            Model = directory, ModelId = directory, Device = "hip:0", CacheTypeK = key, CacheTypeV = value,
            UsePaged = false, PromptCacheEnabled = false, Warmup = WarmupOptions.Disabled, MtpEnabled = false,
        });
        Assert.Equal("hip:0", state.Options.ResolvedDevice);
        var model = Assert.IsType<HipTransformerModel>(state.Model);
        Assert.Null(state.Scheduler);
        string baseline = await Generate(state); // The production generator must pass the selected ordinal and cache.

        var caches = new List<HipPagedKvCache>();
        var realFactory = state.KvCacheFactory!;
        // Observe the real server cache factory without changing the model, storage or request gate.
        state.Generator = new TextGenerator(model, state.Tokenizer!, (config, size) =>
        {
            var cache = Assert.IsType<HipPagedKvCache>(realFactory(config, size));
            Assert.Equal(KvCacheConfig.ParseDType(key), cache.KeyType);
            Assert.Equal(KvCacheConfig.ParseDType(value), cache.ValueType);
            Assert.Equal(0, cache.AllocatedBytes);
            caches.Add(cache);
            return cache;
        }, mtpEnabled: false, deviceId: model.DeviceId);

        long peak = 0;
        using (var cancelled = new CancellationTokenSource())
        {
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() => state.ExecuteAsync(state.Options.ModelId, async generator =>
            {
                await using var stream = generator.GenerateStreamingAsync("Hello",
                    new InferenceOptions { MaxTokens = 40, Temperature = 0 }, cancelled.Token).GetAsyncEnumerator();
                Assert.True(await stream.MoveNextAsync());
                var cache = Assert.Single(caches);
                peak = cache.AllocatedBytes;
                Assert.Equal(expectedPageBytes, peak);
                cancelled.Cancel();
                while (await stream.MoveNextAsync()) { }
            }, CancellationToken.None));
        }
        Assert.Equal(0, Assert.Single(caches).AllocatedBytes);
        Assert.Equal(0, caches[0].AllocatedPages);
        Assert.Equal(baseline, await Generate(state));
        Assert.Equal(baseline, await Generate(state));
        Assert.Equal(3, caches.Count);
        Assert.All(caches, cache => Assert.Equal(0, cache.AllocatedBytes));
        Assert.NotSame(caches[0], caches[1]);
        Assert.NotSame(caches[1], caches[2]);
        output.WriteLine($"HIP K={key} V={value}: initial requested KV bytes=0, cancellation peak={peak}, final=0; two isolated requests matched production baseline. Driver reservation is not included.");
        var source = state.CurrentSafetensors!;
        await state.UnloadAsync(null, all: true, CancellationToken.None);
        Assert.Throws<ObjectDisposedException>(() => source.GetTensorFileLocation("model.embed_tokens.weight"));
    }

    private static async Task<string> Generate(ServerState state)
    {
        var text = new StringBuilder();
        await state.ExecuteAsync(state.Options.ModelId, async generator =>
        {
            await foreach (string part in generator.GenerateStreamingAsync("Hello", new InferenceOptions { MaxTokens = 2, Temperature = 0 }))
                text.Append(part);
        }, CancellationToken.None);
        return text.ToString();
    }
}
