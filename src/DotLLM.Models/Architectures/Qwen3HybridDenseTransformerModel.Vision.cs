using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Image input of the Qwen3.5+ hybrid (Bonsai 2.5 27B with its Qwen3-VL mmproj, issue #2073): image rows replace the token embeddings of the
/// placeholder positions, and the attention layers rotate Q/K with the interleaved M-RoPE (sections [11, 11, 10, 0], ggml <c>IMROPE</c>)
/// on the image grid's (t, y, x) positions instead of the single slot position. The KV cache and the recurrent state are indexed by slot as always.
/// </summary>
public sealed unsafe partial class Qwen3HybridDenseTransformerModel : IImageInputModel
{
    private enum RopeMode { Plain, Shifted, Mrope }

    private IReadOnlyList<ImageSpan>? _pendingImages;
    private RopeMode _ropeMode = RopeMode.Plain;
    private int _ropeShift;
    private int[] _ropeT = [], _ropeY = [], _ropeX = [], _ropeScratch = [];
    private byte[]? _mropeAxis;
    private readonly ConditionalWeakTable<object, StrongBox<int>> _ropeDeltas = new();

    /// <summary>Test seam: deliberately wrong behaviour (CLAUDE.md "Prove the Test Can Fail").</summary>
    internal enum VisionMutant
    {
        /// <summary>The correct implementation.</summary>
        None,
        /// <summary>Image rows are not injected (the placeholder token embedding stays).</summary>
        NoInject,
        /// <summary>Plain single-axis RoPE at the slot position (no image grid positions, no delta).</summary>
        PlainRope,
        /// <summary>Image grid positions are used but the text after an image is not shifted back (no rope delta).</summary>
        NoDelta,
        /// <summary>The y and x axes are swapped inside images.</summary>
        SwapYx,
        /// <summary>Non-interleaved (sectioned) M-RoPE pair-to-axis mapping.</summary>
        SectionedRope,
    }

    internal VisionMutant Mutant { get; set; }

    /// <summary>Rope delta currently recorded for <paramref name="kvCache"/> (test seam).</summary>
    internal int RopeDeltaFor(IKvCache? kvCache) => _ropeDeltas.TryGetValue((object?)kvCache ?? this, out var b) ? b.Value : 0;

    /// <inheritdoc/>
    public ITensor ForwardWithImages(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IReadOnlyList<ImageSpan> images,
        IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        ArgumentNullException.ThrowIfNull(images);
        _pendingImages = images.Count > 0 ? images : null;
        try { return Forward(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly); }
        finally { _pendingImages = null; _ropeMode = RopeMode.Plain; }
    }

    /// <summary>Runs at the top of every trunk forward after the embeddings: injects image rows and decides the rope mode.</summary>
    private void PrepareVision(ReadOnlySpan<int> positions, IKvCache? kvCache, float* hidden, int hiddenSize)
    {
        object key = (object?)kvCache ?? this;
        int seqLen = positions.Length;
        if (_pendingImages is { } images)
        {
            if (Mutant != VisionMutant.NoInject)
                for (int r = 0; r < seqLen; r++)
                {
                    int p = positions[r];
                    for (int i = 0; i < images.Count; i++)
                    {
                        var sp = images[i];
                        if (p < sp.Start || p >= sp.End) continue;
                        if (sp.Embeddings.Dimension != hiddenSize)
                            throw new ArgumentException($"image embeddings are {sp.Embeddings.Dimension}-wide, the model hidden size is {hiddenSize}.", nameof(images));
                        sp.Embeddings.ReadRow(p - sp.Start, new Span<float>(hidden + (long)r * hiddenSize, hiddenSize));
                        break;
                    }
                }
            int delta = Mutant == VisionMutant.NoDelta ? 0 : MropePositions.TotalDelta(images);
            _ropeDeltas.AddOrUpdate(key, new StrongBox<int>(delta));
            if (Mutant == VisionMutant.PlainRope) { _ropeMode = RopeMode.Plain; return; }
            EnsureRopeBuffers(seqLen);
            MropePositions.Fill(positions, images, _ropeT, _ropeY, _ropeX);
            if (Mutant == VisionMutant.NoDelta)
            {
                // images keep their grid positions; text after them keeps the slot position
                for (int r = 0; r < seqLen; r++) if (_ropeT[r] == _ropeY[r] && _ropeY[r] == _ropeX[r]) _ropeT[r] = _ropeY[r] = _ropeX[r] = positions[r];
            }
            if (Mutant == VisionMutant.SwapYx) for (int r = 0; r < seqLen; r++) (_ropeY[r], _ropeX[r]) = (_ropeX[r], _ropeY[r]);
            _ropeMode = RopeMode.Mrope;
            return;
        }
        // No images in this call: a new sequence clears the delta; a continuation (decode, a later prefill chunk) applies it.
        int d = 0;
        if (positions[0] == 0) _ropeDeltas.Remove(key);
        else if (_ropeDeltas.TryGetValue(key, out var box)) d = box.Value;
        if (d == 0) { _ropeMode = RopeMode.Plain; return; }
        _ropeMode = RopeMode.Shifted;
        _ropeShift = d;
    }

    private void EnsureRopeBuffers(int n)
    {
        if (_ropeT.Length >= n) return;
        _ropeT = new int[n]; _ropeY = new int[n]; _ropeX = new int[n];
    }

    /// <summary>RoPE of one full-attention layer: plain, shifted by the rope delta, or M-RoPE on the image grid.</summary>
    private void ApplyAttentionRope(Span<float> q, Span<float> k, ReadOnlySpan<int> positions, int numHeads, int numKvHeads, int headDim)
    {
        switch (_ropeMode)
        {
            case RopeMode.Mrope:
                _rope.ApplyMrope(q, k, _ropeT.AsSpan(0, positions.Length), _ropeY.AsSpan(0, positions.Length), _ropeX.AsSpan(0, positions.Length),
                    MropeAxis(), numHeads, numKvHeads, headDim);
                break;
            case RopeMode.Shifted:
                if (_ropeScratch.Length < positions.Length) _ropeScratch = new int[positions.Length];
                for (int i = 0; i < positions.Length; i++) _ropeScratch[i] = positions[i] - _ropeShift;
                _rope.Apply(q, k, _ropeScratch.AsSpan(0, positions.Length), numHeads, numKvHeads, headDim, RoPEType.NeoX);
                break;
            default:
                _rope.Apply(q, k, positions, numHeads, numKvHeads, headDim, RoPEType.NeoX);
                break;
        }
    }

    private byte[] MropeAxis()
    {
        if (_mropeAxis is { } cached && Mutant != VisionMutant.SectionedRope) return cached;
        int[] sections = MropePositions.DefaultSections.ToArray();
        if (_gguf is not null)
        {
            string arch = _gguf.Metadata.GetStringOrDefault("general.architecture");
            string key = $"{arch}.rope.dimension_sections";
            if (_gguf.Metadata.ContainsKey(key))
            {
                // stored as an int32 / uint32 array depending on the converter
                try { sections = _gguf.Metadata.GetInt32Array(key); }
                catch { sections = Array.ConvertAll(_gguf.Metadata.GetUInt32Array(key), v => (int)v); }
            }
        }
        var axis = MropePositions.AxisPerPair(sections, _ropeDim / 2);
        if (Mutant == VisionMutant.SectionedRope)
        {
            // non-interleaved: the first s0 pairs t, the next s1 y, the next s2 x
            int s0 = sections[0], s1 = sections[1];
            for (int j = 0; j < axis.Length; j++) axis[j] = (byte)(j < s0 ? 0 : j < s0 + s1 ? 1 : 2);
            return axis;
        }
        _mropeAxis = axis;
        return axis;
    }
}
