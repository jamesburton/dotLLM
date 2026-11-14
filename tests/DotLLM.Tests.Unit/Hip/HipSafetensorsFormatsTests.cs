using System.Security.Cryptography;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Hip;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Server;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Hip;

#pragma warning disable IDISP007 // DeviceModelLoader transfers ownership of both tuple results to these tests.
[Collection("HipGpu")]
public sealed class HipSafetensorsFormatsTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(true, false, false)]
    [InlineData(true, true, true)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [Trait("Category", "GPU")]
    public void PackedShardsAndTiedHeads_MatchIndependentDenseReference_AndOwnEveryComponent(bool awq, bool sharded, bool tied)
    {
        RequireGpu();
        string directory = Path.Combine(Path.GetTempPath(), $"dotllm-hip-formats-{Guid.NewGuid():N}");
        try
        {
            var fixture = new PackedQuantizedSafetensorsTests.Fixture(awq, tied: tied);
            if (!awq)
                fixture.Tensors["model.layers.0.mlp.gate_proj.bias"] = new("F16", [96], new byte[192]);
            string path = fixture.Write(Path.Combine(directory, "packed"), sharded);
            string referencePath = fixture.Write(Path.Combine(directory, "reference"), false, dense: true);
            var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
            var (reference, referenceSource, _) = ModelLoader.LoadFromSafetensors(referencePath,
                new ThreadingConfig(1, EnableCallerPinning: false));
            using (source) using (referenceSource) using (reference)
            {
                using var model = HipTransformerModel.LoadFromSafetensors(source, config);
                using var cache = model.CreateKvCache(16, 4);
                using var referenceCache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 16);
                long bytes = source.Tensors.Where(t => !t.Name.EndsWith(".bias", StringComparison.Ordinal)).Sum(t => t.ByteCount);
                Assert.Equal(bytes, model.StoredWeightBytes);
                Assert.True(model.WeightMemoryBytes >= bytes);
                foreach (var tensor in source.Tensors)
                {
                    var location = source.GetTensorFileLocation(tensor.Name);
                    using var file = File.OpenRead(location.Path);
                    file.Position = location.FileOffset;
                    byte[] actualBytes = new byte[checked((int)tensor.ByteCount)];
                    file.ReadExactly(actualBytes);
                    Assert.Equal(source.GetTensorSpan(tensor.Name).ToArray(), actualBytes);
                }
                int position = 0;
                foreach (int[] tokens in new[] { new[] { 1, 7, 13 }, [4], [17] })
                {
                    int[] positions = Enumerable.Range(position, tokens.Length).ToArray();
                    using var expected = reference.Forward(tokens, positions, -1, referenceCache);
                    using var actual = model.Forward(tokens, positions, 0, cache, true);
                    Check(ReadLast(expected, config.VocabSize), ReadLast(actual, config.VocabSize), $"{(awq ? "AWQ" : "GPTQ")} synthetic at {position}");
                    position += tokens.Length;
                }
#pragma warning disable IDISP016 // Verify the model frees device storage without disposing the caller's mappings.
                model.Dispose();
#pragma warning restore IDISP016
                Assert.Equal(0, model.StoredWeightBytes);
                Assert.Equal(0, model.WeightMemoryBytes);
                Assert.NotEqual(0, source.GetTensorPointer("model.embed_tokens.weight"));
            }
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void PackedValidation_RejectsRawSourcesWrongDimensionsExtrasAndMissingFileRanges(bool awq)
    {
        string directory = Path.Combine(Path.GetTempPath(), $"dotllm-hip-validation-{Guid.NewGuid():N}");
        try
        {
            var fixture = new PackedQuantizedSafetensorsTests.Fixture(awq);
            string path = fixture.Write(directory, false);
            var (source, config) = ModelLoader.OpenSafetensorsAndConfig(path);
            using (source)
            using (var raw = SafetensorsFile.Open(path))
            {
                HipWeights.Validate(source, config);
                Assert.Throws<InvalidDataException>(() => HipWeights.Validate(raw, config));
                Assert.Throws<InvalidDataException>(() => HipWeights.Validate(source, config with { IntermediateSize = 128 }));
                using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(directory, "config.json")));
                using var view = new SourceView(source, missingRanges: true);
                using var noRanges = PackedQuantizedSafetensorsSource.WrapIfQuantized(view, json.RootElement);
                Assert.Throws<NotSupportedException>(() => HipWeights.Validate(noRanges, config));
            }
            fixture.Tensors["model.unimplemented.weight"] = new("F16", [1], new byte[2]);
            var (extra, _) = ModelLoader.OpenSafetensorsAndConfig(fixture.Write(directory, false));
            using (extra) Assert.Throws<NotSupportedException>(() => HipWeights.Validate(extra, config));
        }
        finally { if (Directory.Exists(directory)) Directory.Delete(directory, true); }
    }

    [SkippableFact]
    [Trait("Category", "GPU")]
    public void OriginalBf16SmolLlama_PrefillDecodeMatchCpu_AndF16ReinterpretationFails()
    {
        RequireGpu();
        var fixture = TestFixtureResolver.ResolveDirectory(["DOTLLM_TEST_HIP_BF16_MODEL"], "HuggingFaceTB", "SmolLM2-135M-Instruct",
            ["config.json", "model.safetensors", "tokenizer.json"], null);
        Skip.If(!fixture.Found, fixture.SkipMessage("original BF16 SmolLM2 checkpoint"));
        VerifyHash(Path.Combine(fixture.Path!, "model.safetensors"), "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c");
        var (cpu, cpuSource) = DeviceModelLoader.LoadSafetensors(fixture.Path!, "cpu", null,
            new ThreadingConfig(1, EnableCallerPinning: false), _ => { }, _ => { });
        using (cpuSource) using (cpu.Model)
        {
            var (gpu, gpuSource) = DeviceModelLoader.LoadSafetensors(fixture.Path!, "hip:0", null, default, _ => { }, _ => { });
            using (gpuSource) using (gpu.Model)
            {
                var model = Assert.IsType<HipTransformerModel>(gpu.Model);
                var config = model.Config;
                Assert.All(gpuSource.Tensors, tensor => Assert.Equal(SafetensorsDType.BF16, tensor.DType));
                Assert.Equal(gpuSource.Tensors.Sum(t => t.ByteCount), model.StoredWeightBytes);
                var tokenizer = ModelLoader.LoadTokenizerFromHfDirectory(fixture.Path!)!;
                int[] first = tokenizer.Encode("The capital of France is Paris. The city is known for its architecture, museums, and");
                int[] input = first;
                using var cpuCache = new SimpleKvCache(config.NumLayers, config.NumKvHeads, config.HeadDim, 128);
                using var cache = model.CreateKvCache(128);
                int position = 0;
                float[] firstReference = [];
                for (int step = 0; step < 6; step++)
                {
                    int[] positions = Enumerable.Range(position, input.Length).ToArray();
                    using var expected = cpu.Model.Forward(input, positions, -1, cpuCache);
                    using var actual = model.Forward(input, positions, 0, cache, true);
                    float[] reference = ReadLast(expected, config.VocabSize);
                    if (step == 0) firstReference = reference;
                    Check(reference, ReadLast(actual, config.VocabSize), $"original BF16 step {step}");
                    position += input.Length;
                    input = [ArgMax(reference)];
                }
                using var wrongSource = new SourceView(gpuSource, reinterpretBf16: true);
                using var wrong = HipTransformerModel.LoadFromSafetensors(wrongSource, config);
                using var wrongResult = wrong.Forward(first, Enumerable.Range(0, first.Length).ToArray(), 0, null, true);
                double wrongError = RelativeL2(firstReference, ReadLast(wrongResult, config.VocabSize));
                output.WriteLine($"BF16 interpreted as F16: relative L2={wrongError:G6} (must fail the 0.02 correctness bound).");
                Assert.True(wrongError > 0.02);
            }
        }
    }

    [SkippableTheory]
    [InlineData("AWQ", "853cf974e9fd5a6f43d6084f371b90effb5f710fed18d5a7556ef4e3db84e5a2")]
    [InlineData("GPTQ", "ceef7773c19f5cbda3e5f688f99e045e07a879e70c19ecdb5d069b27c2d45b37")]
    [Trait("Category", "GPU")]
    public void RealPackedTinyLlama_MatchesPinnedProducerHfRows_AndWrongDeviceFileRangesFail(string format, string expectedHash)
    {
        RequireGpu();
        var fixture = TestFixtureResolver.ResolveDirectory([$"DOTLLM_TEST_TINYLLAMA_{format}"], "TheBloke",
            $"TinyLlama-1.1B-Chat-v1.0-{format}", ["model.safetensors", "config.json", "packed-reference.json"], null);
        Skip.If(!fixture.Found, fixture.SkipMessage("TinyLlama packed checkpoint and independently generated Hugging Face reference"));
        VerifyHash(Path.Combine(fixture.Path!, "model.safetensors"), expectedHash);
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture.Path!, "packed-reference.json")));
        Assert.Equal(expectedHash, reference.RootElement.GetProperty("model_sha256").GetString());
        Assert.Equal("4.51.3", reference.RootElement.GetProperty("transformers").GetString());
        Assert.Equal(string.Equals(format, "AWQ", StringComparison.Ordinal) ? "65eab3eabe3f55e300ffbab5feac59c49322d985f42dcda4e2288859fb9a4abe"
            : "5a6a483784adc7d9ee55e62f6dc4fdaf3a642f56796d2a6b8afea0b2364d751d",
            reference.RootElement.GetProperty("producer_sha256").GetString());
        var (loaded, source) = DeviceModelLoader.LoadSafetensors(fixture.Path!, "hip:0", null, default, _ => { }, _ => { });
        using (source) using (loaded.Model)
        {
            var model = Assert.IsType<HipTransformerModel>(loaded.Model);
            var packed = Assert.IsType<PackedQuantizedSafetensorsSource>(source);
            Assert.Equal(154, packed.ProjectionCount);
            Assert.Equal(source.Tensors.Where(t => !packed.IsZeroBias(t.Name)).Sum(t => t.ByteCount), model.StoredWeightBytes);
            using var cache = model.CreateKvCache(32);
            int position = 0, batchIndex = 0;
            float[] firstReference = [];
            int firstToken = 0;
            using var batches = reference.RootElement.GetProperty("tokens").EnumerateArray();
            foreach (var batch in batches)
            {
                int row = 0;
                using var tokenValues = batch.EnumerateArray();
                foreach (var token in tokenValues)
                {
                    using var rowValues = reference.RootElement.GetProperty("logits")[batchIndex][row++].EnumerateArray();
                    float[] expected = rowValues.Select(x => x.GetSingle()).ToArray();
                    if (position == 0) { firstReference = expected; firstToken = token.GetInt32(); }
                    using var actual = model.Forward([token.GetInt32()], [position], 0, cache);
                    Check(expected, ReadLast(actual, model.Config.VocabSize), $"{format} pinned HF row {position}");
                    position++;
                }
                batchIndex++;
            }
            Assert.Equal(6, position);
            using var json = JsonDocument.Parse(File.ReadAllText(Path.Combine(fixture.Path!, "config.json")));
            using var wrongView = new SourceView(source, wrongZeroRanges: true);
            using var wrongSource = PackedQuantizedSafetensorsSource.WrapIfQuantized(wrongView, json.RootElement);
            using var wrong = HipTransformerModel.LoadFromSafetensors(wrongSource, model.Config);
            using var wrongResult = wrong.Forward([firstToken], [0], 0);
            double wrongError = RelativeL2(firstReference, ReadLast(wrongResult, model.Config.VocabSize));
            output.WriteLine($"{format} original zero ranges replaced by qweight ranges: relative L2={wrongError:G6}; resident payload={model.StoredWeightBytes}, packed source={packed.PackedBytes} bytes.");
            Assert.True(wrongError > 0.02);
        }
    }

    private void Check(float[] expected, float[] actual, string context)
    {
        double relative = RelativeL2(expected, actual);
        double maxA = expected.Max(), maxB = actual.Max();
        double logA = maxA + Math.Log(expected.Sum(x => Math.Exp(x - maxA)));
        double logB = maxB + Math.Log(actual.Sum(x => Math.Exp(x - maxB)));
        double kl = expected.Select((x, i) => Math.Exp(x - logA) * (x - logA - actual[i] + logB)).Sum();
        output.WriteLine($"{context}: relative L2={relative:G6}, KL={kl:G6}, reference top1={ArgMax(expected)}, HIP top1={ArgMax(actual)}.");
        Assert.True(relative < 0.02, $"{context}: relative L2={relative:G6}");
        Assert.True(kl < 0.001, $"{context}: KL={kl:G6}");
        // F16 resident activations/logits can reverse close winners; the full vector and KL are the parity gates.
        Assert.True(expected[ArgMax(expected)] - expected[ArgMax(actual)] <= 0.03f, $"{context}: HIP winner is not a close reference contender.");
    }

    private static double RelativeL2(float[] expected, float[] actual)
    {
        double error = 0, scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (!float.IsFinite(actual[i])) return double.PositiveInfinity;
            error += Math.Pow(expected[i] - actual[i], 2);
            scale += expected[i] * (double)expected[i];
        }
        return Math.Sqrt(error / scale);
    }

    private static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());
    private static unsafe float[] ReadLast(ITensor tensor, int vocab)
        => new ReadOnlySpan<float>((float*)tensor.DataPointer + (tensor.Shape[0] - 1) * vocab, vocab).ToArray();
    private static void VerifyHash(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        Assert.Equal(expected, Convert.ToHexStringLower(SHA256.HashData(stream)));
    }
    private static void RequireGpu()
    {
        bool available = OperatingSystem.IsWindows() && HipDevice.IsAvailable();
        if (string.Equals(Environment.GetEnvironmentVariable("DOTLLM_TEST_HIP_REQUIRED"), "1", StringComparison.Ordinal)) Assert.True(available);
        Skip.IfNot(available, "Windows HIP/DirectStorage is unavailable.");
    }

    // Borrow mappings for deliberate substitutions; the real caller still owns and disposes the source.
    private sealed class SourceView : ISafetensorsTensorSource
    {
        private readonly ISafetensorsTensorSource _source;
        private readonly bool _missingRanges, _wrongZeroRanges;
        public SourceView(ISafetensorsTensorSource source, bool missingRanges = false, bool reinterpretBf16 = false, bool wrongZeroRanges = false)
        {
            _source = source; _missingRanges = missingRanges; _wrongZeroRanges = wrongZeroRanges;
            Tensors = source.Tensors.Select(t => reinterpretBf16 && t.DType == SafetensorsDType.BF16 ? t with { DType = SafetensorsDType.F16 } : t).ToArray();
            TensorsByName = Tensors.ToDictionary(t => t.Name, StringComparer.Ordinal);
        }
        public IReadOnlyList<SafetensorsTensorDescriptor> Tensors { get; }
        public IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName { get; }
        public nint GetTensorPointer(string name) => _source.GetTensorPointer(name);
        public ReadOnlySpan<byte> GetTensorSpan(string name) => _source.GetTensorSpan(name);
        public (string Path, long FileOffset) GetTensorFileLocation(string name)
            => _missingRanges ? throw new NotSupportedException("Deliberately no original ranges.")
                : _source.GetTensorFileLocation(_wrongZeroRanges && name.EndsWith(".qzeros", StringComparison.Ordinal)
                    ? name[..^".qzeros".Length] + ".qweight" : name);
        public void Dispose() { }
    }
}
#pragma warning restore IDISP007
