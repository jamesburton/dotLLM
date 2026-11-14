using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Engine.KvCache;
using DotLLM.Models;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Integration.Fixtures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.SafeTensors;

/// <summary>Optional real-checkpoint gate; no models or reference logits are stored in the repository.</summary>
public sealed unsafe class PackedQuantizedRealModelTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData("AWQ", "853cf974e9fd5a6f43d6084f371b90effb5f710fed18d5a7556ef4e3db84e5a2")]
    [InlineData("GPTQ", "ceef7773c19f5cbda3e5f688f99e045e07a879e70c19ecdb5d069b27c2d45b37")]
    public void TinyLlama_PrefillAndDecodeMatchPinnedProducerAndHuggingFace(string format, string expectedHash)
    {
        var fixture = TestFixtureResolver.ResolveDirectory([$"DOTLLM_TEST_TINYLLAMA_{format}"], "TheBloke",
            $"TinyLlama-1.1B-Chat-v1.0-{format}", ["model.safetensors", "config.json", "packed-reference.json"], null);
        Skip.If(!fixture.Found, fixture.SkipMessage("TinyLlama packed checkpoint and generate_packed_hf_reference.py output"));
        string path = Path.Combine(fixture.Path!, "model.safetensors");
        using (var stream = File.OpenRead(path))
            Assert.Equal(expectedHash, Convert.ToHexStringLower(SHA256.HashData(stream)));
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture.Path!, "packed-reference.json")));
        Assert.Equal(expectedHash, reference.RootElement.GetProperty("model_sha256").GetString());
        Assert.Equal("4.51.3", reference.RootElement.GetProperty("transformers").GetString());
        Assert.Equal(string.Equals(format, "AWQ", StringComparison.Ordinal) ? "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe"
            : "5a6a483784adc7d9ee55e62f6dc4fdaf3a642f56796d2a6b8afea0b2364d751d",
            reference.RootElement.GetProperty("producer_sha256").GetString());
        var watch = Stopwatch.StartNew();
        var (model, source, config) = ModelLoader.LoadFromSafetensors(path, new ThreadingConfig(4, EnableCallerPinning: false));
        using (source) using (model)
        using (var cache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 16))
        {
            var packed = Assert.IsType<PackedQuantizedSafetensorsSource>(source);
            Assert.Equal(154, packed.ProjectionCount);
            Assert.Equal(0, model.RepackedWeightBytes);
            double squareError = 0, squareReference = 0;
            float maxError = 0;
            int position = 0, batchIndex = 0;
            using var batches = reference.RootElement.GetProperty("tokens").EnumerateArray();
            foreach (var batch in batches)
            {
                using var tokenValues = batch.EnumerateArray();
                int[] tokens = tokenValues.Select(x => x.GetInt32()).ToArray();
                int[] positions = Enumerable.Range(position, tokens.Length).ToArray();
                using var actual = model.Forward(tokens, positions, -1, cache);
                var values = new ReadOnlySpan<float>((void*)actual.DataPointer, tokens.Length * config.VocabSize);
                var rows = reference.RootElement.GetProperty("logits")[batchIndex++];
                for (int t = 0; t < tokens.Length; t++)
                {
                    using var row = rows[t].EnumerateArray();
                    float[] expected = row.Select(x => x.GetSingle()).ToArray();
                    var observed = values.Slice(t * config.VocabSize, config.VocabSize);
                    Assert.Equal(ArgMax(expected), ArgMax(observed));
                    for (int i = 0; i < expected.Length; i++)
                    {
                        float error = Math.Abs(expected[i] - observed[i]);
                        maxError = Math.Max(maxError, error);
                        squareError += (double)error * error;
                        squareReference += (double)expected[i] * expected[i];
                        Assert.InRange(error, 0, 2e-4f * MathF.Sqrt(config.HiddenSize / 64f) * (1 + Math.Abs(expected[i])));
                    }
                }
                position += tokens.Length;
            }
            double relativeRms = Math.Sqrt(squareError / squareReference);
            Assert.InRange(relativeRms, 0, 2e-4);
            output.WriteLine($"{format}: max_abs={maxError:R}, relative_rms={relativeRms:R}, elapsed_s={watch.Elapsed.TotalSeconds:F3}, "
                + $"physical_tensor_bytes={source.Tensors.Sum(t => t.ByteCount)}, packed_bytes={packed.PackedBytes}, "
                + $"logical_parameters={packed.LogicalParameterCount}, descriptor_bytes={packed.ProjectionCount * 64}, "
                + $"kernel_accumulator=Vector256<float>, kernel_weight_scratch_bytes=0, inference_scratch_bytes={model.ComputeMemoryBytes}");
        }
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
