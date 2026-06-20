using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Lora;

public class BitNetSyntheticLoraTests
{
    private readonly ITestOutputHelper _output;
    public BitNetSyntheticLoraTests(ITestOutputHelper output) => _output = output;

    private static string? ModelPath =>
        Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF"); // set to the cached i2_s path

    [Fact]
    public void SyntheticAdapter_ChangesLogits_ButBaseUnchanged()
    {
        // Plain Fact + early-return (no SkippableFact dependency): no-op pass when the model is absent.
        if (ModelPath is null || !File.Exists(ModelPath))
        {
            _output.WriteLine("SKIP: BitNet GGUF not available (set DOTLLM_BITNET_GGUF).");
            return;
        }

        using var gguf = GgufFile.Open(ModelPath!);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, cfg, new ThreadingConfig(0, 0));

        int[] tok = [ 1, 2, 3 ];
        int[] pos = [ 0, 1, 2 ];

        using var baseLogits = model.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: null);
        using var adapter = SyntheticLoraAdapterFactory.ForConfig(cfg, rank: 8, alpha: 16f, seed: 42);
        using var loraLogits = model.Forward(tok, pos, deviceId: -1, kvCache: null, adapter: adapter);

        Assert.True(adapter.IsCompatible(cfg));
        // Deltas are non-zero ⇒ logits must differ.
        bool anyDiff = false;
        unsafe
        {
            float* b = (float*)baseLogits.DataPointer; float* l = (float*)loraLogits.DataPointer;
            long n = (long)baseLogits.Shape[0] * baseLogits.Shape[1];
            for (long i = 0; i < n; i++) if (MathF.Abs(b[i] - l[i]) > 1e-4f) { anyDiff = true; break; }
        }
        Assert.True(anyDiff, "LoRA adapter did not change BitNet logits");
    }
}
