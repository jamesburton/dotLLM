using DotLLM.Core.Configuration;
using DotLLM.Core.Vision;
using DotLLM.Models.Vision;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Which encoder a model gets (#2062): its own device encoder through <see cref="IImageEncoderProvider"/>, or the CPU encoder when forced or when the model has none.</summary>
[Collection("ImageEncoderEnvironment")]
public sealed class Inkling2VisionEncoderRoutingTests
{
    private sealed class StubEncoder : IImageEncoder
    {
        public int EmbeddingDimension => 1;
        public int CountTokens(int width, int height) => 1;
        public ImageEmbeddings Encode(DecodedImage image, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public void Dispose() { }
    }

    private sealed class ProviderModel : IImageEncoderProvider
    {
        public string? Path;
        public IImageEncoder CreateImageEncoder(string mmprojPath, ImageOverLimit overLimit) { Path = mmprojPath; return new StubEncoder(); }
    }

    private sealed class DecliningProviderModel : IImageEncoderProvider
    {
        public bool Asked;
        public IImageEncoder CreateImageEncoder(string mmprojPath, ImageOverLimit overLimit) { Asked = true; throw new NotSupportedException("declined (test)"); }
    }

    private static T WithDevice<T>(string? value, Func<T> body)
    {
        string? old = Environment.GetEnvironmentVariable(VisionSupport.DeviceEnvVar);
        Environment.SetEnvironmentVariable(VisionSupport.DeviceEnvVar, value);
        try { return body(); }
        finally { Environment.SetEnvironmentVariable(VisionSupport.DeviceEnvVar, old); }
    }

    [Fact]
    public void ProviderModel_GetsItsOwnDeviceEncoder()
    {
        var model = new ProviderModel();
        using var enc = WithDevice<IImageEncoder>(null, () => VisionSupport.LoadEncoderFor(model, "x.gguf", new ThreadingConfig(1)));
        Assert.IsType<StubEncoder>(enc);
        Assert.Equal("x.gguf", model.Path);
    }

    [Fact]
    public void DeviceCpu_ForcesTheCpuEncoder_EvenForAProviderModel()
    {
        var model = new ProviderModel();
        // The CPU path opens the file: a missing one is the evidence that the provider was NOT asked.
        Assert.ThrowsAny<IOException>(() => WithDevice<IImageEncoder>("cpu", () => VisionSupport.LoadEncoderFor(model, "does-not-exist-mmproj.gguf", new ThreadingConfig(1))));
        Assert.Null(model.Path);
    }

    [Fact]
    public void ProviderThatDeclines_FallsBackToTheCpuEncoder()
    {
        // A device encoder that refuses the mmproj or the device (NotSupportedException: e.g. a non-Q8_0 mmproj on the Vulkan ViT) must not cost the model its image input:
        // the CPU encoder is tried next (it opens the file: the missing one is the evidence), instead of the provider's NotSupportedException reaching the caller.
        var model = new DecliningProviderModel();
        Assert.ThrowsAny<IOException>(() => WithDevice<IImageEncoder>(null, () => VisionSupport.LoadEncoderFor(model, "does-not-exist-mmproj.gguf", new ThreadingConfig(1))));
        Assert.True(model.Asked);
    }

    [Fact]
    public void ModelWithoutAProvider_UsesTheCpuEncoder()
        => Assert.ThrowsAny<IOException>(() => WithDevice<IImageEncoder>(null, () => VisionSupport.LoadEncoderFor(new object(), "does-not-exist-mmproj.gguf", new ThreadingConfig(1))));

    [Fact]
    public void DeviceEnvironment_ParsesTheValues_AndRejectsTypos()
    {
        Assert.False(VisionSupport.ForceCpuFromEnvironment(_ => null));
        Assert.False(VisionSupport.ForceCpuFromEnvironment(_ => " "));
        Assert.False(VisionSupport.ForceCpuFromEnvironment(_ => "auto"));
        Assert.False(VisionSupport.ForceCpuFromEnvironment(_ => "GPU"));
        Assert.True(VisionSupport.ForceCpuFromEnvironment(_ => " CPU "));
        var ex = Assert.Throws<ArgumentException>(() => VisionSupport.ForceCpuFromEnvironment(_ => "vulkan"));
        Assert.Contains("DOTLLM_VISION_DEVICE", ex.Message);
    }

    [Fact]
    public void MalformedDeviceValue_FailsLoadEncoderFor_InsteadOfBeingIgnored()
        => Assert.Throws<ArgumentException>(() => WithDevice<IImageEncoder>("cpuu", () => VisionSupport.LoadEncoderFor(new ProviderModel(), "x.gguf", new ThreadingConfig(1))));
}
