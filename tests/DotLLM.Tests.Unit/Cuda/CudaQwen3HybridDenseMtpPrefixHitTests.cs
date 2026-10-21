using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Engine;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1088 on CUDA: an agent turn that hits the recurrent prefix snapshot decodes with MTP, the head resumed from the
/// snapshot drafting what a cold run of the same prompt drafts, including on the turn whose hit grows the model-owned KV.
/// CPU twin: <c>TextGeneratorRecurrentPrefixMtpTests</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaQwen3HybridDenseMtpPrefixHitTests(ITestOutputHelper output) : IDisposable
{
    // The hit prefills only the suffix and the cold run the whole prompt, so the trunk's GEMMs run at different row counts
    // and the head's inputs differ in the last bits. A wrong carry row, length or rows moves the logits far more.
    private const float DraftLogitTolerance = 2e-3f;

    private readonly string _scratch = Directory.CreateTempSubdirectory("dotllm-cuda-prefix-mtp-").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void HitTurns_DecodeWithMtp_AndDraftWhatAColdRunDrafts()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "ptx");
        Skip.IfNot(Directory.Exists(ptxDir), "PTX files missing");

        string path = SyntheticQwen35HybridDenseMtpGguf.Write(Path.Combine(_scratch, "prefix-mtp.gguf"), withMtp: true, contextLength: 512);
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        // Two instances: each owns its KV storage, which a request on the other would overwrite.
        using var cachedModel = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);
        using var coldModel = CudaQwen3HybridDenseTransformerModel.LoadFromGguf(gguf, config, deviceId: 0, ptxDir);

        RecurrentPrefixMtpHarness.AssertHitTurnsDraftLikeColdRuns(
            cachedModel, (_, size) => cachedModel.CreateKvCache(size),
            coldModel, (_, size) => coldModel.CreateKvCache(size),
            DraftLogitTolerance, output.WriteLine);
    }
}
