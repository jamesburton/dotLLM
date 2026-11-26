using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Edge0;

/// <summary>
/// CLI auto-attach of the Edge0 MTP head (#2014): the resolver finds the <c>mtp/</c> folder only beside <c>model-skeleton.gguf</c>, attaches it through
/// <see cref="IEdge0MtpHeadAttachable"/>, and <c>DOTLLM_EDGE0_MTP=0</c> opts out (an explicit folder still attaches).
/// </summary>
[Collection("Edge0MtpEnv")]
public sealed class Edge0MtpAutoAttachTests : IDisposable
{
    private readonly string _root = Directory.CreateTempSubdirectory("edge0mtp").FullName;
    private readonly string? _oldEnv = Environment.GetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar);

    public void Dispose()
    {
        Environment.SetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar, _oldEnv);
        try { Directory.Delete(_root, true); } catch (IOException) { }
    }

    private sealed class StubEdge0Model : IModel, IEdge0MtpHeadAttachable
    {
        public ModelConfig Config { get; } = new() { Architecture = Architecture.Qwen4Exp, VocabSize = 8, HiddenSize = 256, NumLayers = 2, IntermediateSize = 1, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 1, MaxSequenceLength = 8 };
        public long ComputeMemoryBytes => 0;
        public bool HasMtpHead { get; private set; }
        public string? AttachedDir, AttachedFile;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void AttachEdge0MtpHead(string mtpDir) { AttachedDir = mtpDir; HasMtpHead = true; }
        public void AttachMtpHead(string path) { AttachedFile = path; HasMtpHead = true; }
        public void Dispose() { }
    }

    /// <summary>A model folder: <c>model-skeleton.gguf</c> plus (optionally) a complete <c>mtp/</c> folder.</summary>
    private string Folder(string name, bool skeleton = true, bool mtp = true)
    {
        string dir = Path.Combine(_root, name);
        Directory.CreateDirectory(dir);
        if (skeleton) File.WriteAllBytes(Path.Combine(dir, "model-skeleton.gguf"), [0]);
        if (mtp)
        {
            string m = Path.Combine(dir, "mtp");
            Directory.CreateDirectory(m);
            foreach (string f in new[] { "dense.txt", "dense.bin", "experts.bin", "draft_vocab.bin" }) File.WriteAllBytes(Path.Combine(m, f), [0]);
        }
        return dir;
    }

    [Fact]
    public void FindEdge0Pack_NeedsTheSkeletonAndACompleteMtpFolder()
    {
        string both = Folder("both");
        Assert.Equal(Path.Combine(both, "mtp"), Qwen4ExpMtpHeadResolver.FindEdge0Pack(Path.Combine(both, "model-skeleton.gguf")));
        Assert.Equal(Path.Combine(both, "mtp"), Qwen4ExpMtpHeadResolver.FindEdge0Pack(both));   // the folder form
        Assert.Null(Qwen4ExpMtpHeadResolver.FindEdge0Pack(Folder("noskel", skeleton: false)));   // an mtp/ with no skeleton is not an Edge0 model
        Assert.Null(Qwen4ExpMtpHeadResolver.FindEdge0Pack(Path.Combine(Folder("nomtp", mtp: false), "model-skeleton.gguf")));
        string other = Folder("other");
        File.WriteAllBytes(Path.Combine(other, "something-else.gguf"), [0]);
        Assert.Null(Qwen4ExpMtpHeadResolver.FindEdge0Pack(Path.Combine(other, "something-else.gguf")));   // a different file in the folder: not the skeleton
        string partial = Folder("partial");
        File.Delete(Path.Combine(partial, "mtp", "experts.bin"));
        Assert.Null(Qwen4ExpMtpHeadResolver.FindEdge0Pack(partial));
    }

    [Fact]
    public void TryAttach_AutoAttachesTheFolder_BesideTheSkeleton()
    {
        Environment.SetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar, null);
        string dir = Folder("auto");
        var model = new StubEdge0Model();
        string? got = Qwen4ExpMtpHeadResolver.TryAttach(model, Path.Combine(dir, "model-skeleton.gguf"), explicitPath: null, trunkBytes: 1);   // a 1-byte "trunk": the size rule must not apply
        Assert.Equal(Path.Combine(dir, "mtp"), got);
        Assert.Equal(got, model.AttachedDir);
        Assert.Null(model.AttachedFile);
        Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(model, Path.Combine(dir, "model-skeleton.gguf")));   // already attached
    }

    [Fact]
    public void OptOut_LeavesTheFolderAlone_ButAnExplicitFolderStillAttaches()
    {
        Environment.SetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar, "0");
        string dir = Folder("optout");
        string skel = Path.Combine(dir, "model-skeleton.gguf");
        var model = new StubEdge0Model();
        Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(model, skel));
        Assert.False(model.HasMtpHead);
        Assert.Null(Qwen4ExpMtpHeadResolver.FindAuto(model, skel));
        Assert.Equal(Path.Combine(dir, "mtp"), Qwen4ExpMtpHeadResolver.TryAttach(model, skel, Path.Combine(dir, "mtp")));
        Assert.True(model.HasMtpHead);
        Environment.SetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar, "1");
        Assert.Equal(Path.Combine(dir, "mtp"), Qwen4ExpMtpHeadResolver.FindAuto(new StubEdge0Model(), skel));   // anything but "0" leaves it on
    }

    [Fact]
    public void ModelsWithoutTheEdge0Interface_AreNeverGivenTheFolder()
    {
        Environment.SetEnvironmentVariable(Qwen4ExpMtpHeadResolver.Edge0EnvVar, null);
        string dir = Folder("cpu");
        var plain = new PlainHeadModel();
        Assert.Null(Qwen4ExpMtpHeadResolver.TryAttach(plain, Path.Combine(dir, "model-skeleton.gguf")));
        Assert.False(plain.HasMtpHead);
        Assert.Throws<NotSupportedException>(() => Qwen4ExpMtpHeadResolver.TryAttach(plain, Path.Combine(dir, "model-skeleton.gguf"), Path.Combine(dir, "mtp")));
    }

    private sealed class PlainHeadModel : IModel, IMtpHeadAttachable
    {
        public ModelConfig Config { get; } = new() { Architecture = Architecture.Qwen4Exp, VocabSize = 8, HiddenSize = 256, NumLayers = 2, IntermediateSize = 1, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 1, MaxSequenceLength = 8 };
        public long ComputeMemoryBytes => 0;
        public bool HasMtpHead { get; private set; }
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void AttachMtpHead(string path) => HasMtpHead = true;
        public void Dispose() { }
    }
}
