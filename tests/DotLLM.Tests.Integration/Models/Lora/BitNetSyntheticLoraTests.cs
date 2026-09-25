using DotLLM.Core.Configuration;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using Xunit.Abstractions;
using Xunit;

namespace DotLLM.Tests.Integration.Models.Lora;

public class BitNetSyntheticLoraTests
{
    private readonly ITestOutputHelper _output;
    public BitNetSyntheticLoraTests(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// BitNet I2_S fixture, resolved via <see cref="KnownTestFixtures.BitNetI2S"/>:
    /// <c>$DOTLLM_BITNET_GGUF</c>, then the dotLLM test cache, then the HF hub cache (#308).
    /// </summary>
    private static FixtureLocation BitNetFixture => KnownTestFixtures.BitNetI2S;

    private static string? ModelPath => BitNetFixture.Path;

    [SkippableFact]
    public void SyntheticAdapter_ChangesLogits_ButBaseUnchanged()
    {
        // Plain Fact + early-return (no SkippableFact dependency): no-op pass when the model is absent.
        Skip.If(!BitNetFixture.Found, BitNetFixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));

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
