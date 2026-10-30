using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Separate-file MTP head for <c>qwen35</c> (issue #1185). <c>ggml-org/Qwen3.8-27B-GGUF</c> converts the trunk with <c>--no-mtp</c> and ships
/// the head as <c>mtp-Qwen3.8-27B-*.gguf</c>: the same <c>blk.{NumLayers}.*</c> full-attention + dense-FFN block and <c>nextn.*</c> tensors an
/// embedded head has, plus its own copies of <c>token_embd</c> / <c>output</c>, which are never read (the trunk's are used, as for an embedded
/// head without <c>nextn.embed_tokens</c> / <c>shared_head_head</c>).
/// </summary>
public sealed unsafe partial class Qwen3HybridDenseTransformerModel : IMtpHeadAttachable
{
    private GgufFile? _ownedMtpHeadFile;

    /// <inheritdoc/>
    public bool HasMtpHead => _mtpHead is not null;

    /// <inheritdoc/>
    public void AttachMtpHead(string path)
    {
        var file = GgufFile.Open(path);
        try { AttachMtpHead(file, ownsFile: true); }
        catch { file.Dispose(); throw; }
    }

    /// <summary>
    /// Attaches the MTP head in <paramref name="headFile"/> after checking it was converted from this trunk's checkpoint. The head's
    /// projections are read in their stored quantisation straight from <paramref name="headFile"/>'s mapping.
    /// </summary>
    /// <param name="headFile">An opened head GGUF; must outlive the model unless <paramref name="ownsFile"/>.</param>
    /// <param name="ownsFile">Dispose <paramref name="headFile"/> with the model.</param>
    /// <exception cref="InvalidOperationException">A head is already attached (embedded or separate).</exception>
    /// <exception cref="InvalidDataException">The file is not an MTP head for this trunk.</exception>
    public void AttachMtpHead(GgufFile headFile, bool ownsFile = false)
    {
        ArgumentNullException.ThrowIfNull(headFile);
        if (_mtpHead is not null) throw new InvalidOperationException("An MTP head is already attached.");
        ValidateMtpHeadFile(headFile, Config);
        _mtpHead = LoadMtpHeadIfPresent(new GgufDataBase(headFile), headFile.TensorsByName, Config with { NextnPredictLayers = 1 })
            ?? throw new InvalidDataException($"MTP head GGUF has no blk.{Config.NumLayers}.nextn.eh_proj.weight.");
        if (ownsFile) _ownedMtpHeadFile = headFile;
    }

    /// <summary>
    /// Throws unless <paramref name="headFile"/> is a single-block <c>qwen35</c> MTP head whose block sits right after
    /// <paramref name="trunk"/>'s layers and whose attention / FFN geometry matches the trunk's.
    /// </summary>
    /// <param name="headFile">The candidate head GGUF.</param>
    /// <param name="trunk">The loaded trunk's configuration.</param>
    /// <exception cref="InvalidDataException">The head does not belong to this trunk.</exception>
    public static void ValidateMtpHeadFile(GgufFile headFile, ModelConfig trunk)
    {
        ArgumentNullException.ThrowIfNull(headFile);
        ArgumentNullException.ThrowIfNull(trunk);
        ModelConfig head = GgufModelConfigExtractor.Extract(headFile.Metadata);
        var problems = new List<string>();
        void Check<T>(string what, T got, T want)
        {
            if (!EqualityComparer<T>.Default.Equals(got, want)) problems.Add($"{what} {got} (trunk {want})");
        }
        Check("architecture", head.Architecture, trunk.Architecture);
        Check("nextn_predict_layers", head.NextnPredictLayers, 1);
        Check("trunk layers", head.NumLayers, trunk.NumLayers);
        Check("hidden size", head.HiddenSize, trunk.HiddenSize);
        Check("attention heads", head.NumAttentionHeads, trunk.NumAttentionHeads);
        Check("KV heads", head.NumKvHeads, trunk.NumKvHeads);
        Check("head dim", head.HeadDim, trunk.HeadDim);
        Check("FFN width", head.IntermediateSize, trunk.IntermediateSize);
        if (problems.Count > 0)
            throw new InvalidDataException("The MTP head GGUF does not match the trunk: " + string.Join("; ", problems) + ".");
    }
}
