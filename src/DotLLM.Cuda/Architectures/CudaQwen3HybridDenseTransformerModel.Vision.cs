using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>
/// Image input of the CUDA Qwen3.5+ hybrid (Bonsai 2.5 27B with its Qwen3-VL mmproj, issue #2073): image rows replace the token embeddings of
/// their slots, and the full-attention layers rotate Q/K with the interleaved M-RoPE (<c>rope_mrope_f32.cu</c>) on the image grid's (t, y, x)
/// positions; text after an image rotates at <c>slot - delta</c>. The KV cache and the recurrent state stay indexed by slot.
/// Same semantics as the CPU model (<c>Qwen3HybridDenseTransformerModel.Vision.cs</c>); the bookkeeping lives in <see cref="VisionRopeState"/>.
/// </summary>
/// <remarks>
/// CUDA status: BUILD-VERIFIED only. The M-RoPE PTX has to be generated with nvcc on a CUDA box (native/build_ptx.bat, or a DotLLM.Cuda build
/// with nvcc available); until it exists <see cref="CudaKernels.HasMropeF32"/> is false and a forward that needs it throws
/// <see cref="NotSupportedException"/>. A forward with a non-zero rope delta runs eagerly (never through a captured decode graph, whose
/// rope positions are the slot ints) and refuses MTP / DFlash rather than rotating at a wrong position.
/// </remarks>
public sealed unsafe partial class CudaQwen3HybridDenseTransformerModel : IImageInputModel
{
    private readonly VisionRopeState _vision = new();
    private IReadOnlyList<ImageSpan>? _pendingImages;
    private nint _mropeAxisDevice;
    private byte[]? _mropeAxis;

    /// <summary>Rope delta currently recorded for <paramref name="kvCache"/> (test seam).</summary>
    internal int RopeDeltaFor(IKvCache? kvCache) => _vision.DeltaFor((object?)kvCache ?? this);

    /// <inheritdoc/>
    public ITensor ForwardWithImages(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IReadOnlyList<ImageSpan> images,
        IKvCache? kvCache, bool lastTokenLogitsOnly)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(images);
        if (_isHeadOnly)
            throw new NotSupportedException("Image input is not supported on a head-only (partial-offload) instance.");
        if (images.Count > 0 && !_kernels.HasMropeF32)
            throw new NotSupportedException("Image input on CUDA needs rope_mrope_f32.ptx, which is not built (no nvcc on the dev host). " +
                "Regenerate the PTX on a CUDA box (native/build_ptx.bat) and rebuild DotLLM.Cuda.");
        _pendingImages = images.Count > 0 ? images : null;
        try { return Forward(tokenIds, positions, deviceId, kvCache, lastTokenLogitsOnly); }
        finally { _pendingImages = null; _vision.SetPlain(); }
    }

    /// <summary>
    /// Runs inside <c>ForwardTile</c> after the embedding gather (and Hadamard restore): writes the image rows (widened from F16 into the
    /// transient host embedding buffer, primal basis) over their slots, decides the rope mode and uploads the rope positions.
    /// </summary>
    private void PrepareVision(ReadOnlySpan<int> positions, IKvCache? kvCache, float* hostEmbed, int hiddenSize, bool mtpOrDflash, nint streamH)
    {
        var images = _pendingImages;
        if (images is not null)
        {
            for (int r = 0; r < positions.Length; r++)
            {
                int p = positions[r];
                for (int i = 0; i < images.Count; i++)
                {
                    var sp = images[i];
                    if (p < sp.Start || p >= sp.End) continue;
                    if (sp.Embeddings.Dimension != hiddenSize)
                        throw new ArgumentException($"image embeddings are {sp.Embeddings.Dimension}-wide, the model hidden size is {hiddenSize}.", nameof(images));
                    sp.Embeddings.ReadRow(p - sp.Start, new Span<float>(hostEmbed + (long)r * hiddenSize, hiddenSize));
                    break;
                }
            }
        }

        _vision.Prepare((object?)kvCache ?? this, positions, images);
        if (_vision.Mode == VisionRopeMode.Plain) return;
        if (mtpOrDflash)
            throw new NotSupportedException("MTP / DFlash speculation cannot honour the M-RoPE position shift of an image prompt (rope delta " +
                $"{_vision.LastDelta}); run image requests without speculative decoding.");

        var axes = _vision.Axes;
        fixed (int* pAxes = axes)
            CudaDriverApi.cuMemcpyHtoDAsync_v2(_state.RopePositionsDevice, (nint)pAxes, (nuint)((long)axes.Length * sizeof(int)), streamH).ThrowOnError();
        if (_vision.Mode == VisionRopeMode.Mrope && _mropeAxisDevice == 0)
        {
            byte[] axis = MropeAxisTable();
            _mropeAxisDevice = AllocDevice(axis.Length);
            fixed (byte* pAxis = axis) CopyHtoD(_mropeAxisDevice, (nint)pAxis, axis.Length);
        }
    }

    private byte[] MropeAxisTable()
    {
        if (_mropeAxis is { } cached) return cached;
        int[] sections = MropePositions.DefaultSections.ToArray();
        if (_gguf is not null)
        {
            string arch = _gguf.Metadata.GetStringOrDefault("general.architecture");
            string key = $"{arch}.rope.dimension_sections";
            if (_gguf.Metadata.ContainsKey(key))
            {
                try { sections = _gguf.Metadata.GetInt32Array(key); }
                catch { sections = Array.ConvertAll(_gguf.Metadata.GetUInt32Array(key), v => (int)v); }
            }
        }
        return _mropeAxis = MropePositions.AxisPerPair(sections, _ropeDim / 2);
    }

    /// <summary>RoPE of one full-attention layer's Q/K: plain slot positions, <c>slot - delta</c>, or M-RoPE on the image grid.</summary>
    private void LaunchAttentionRope(nint q, nint k, int seqLen, int numHeads, int numKvHeads, int headDim, nint streamH)
    {
        switch (_vision.Mode)
        {
            case VisionRopeMode.Mrope:
                if (t_capturingDecode) throw new InvalidOperationException("M-RoPE cannot be captured into a decode graph.");
                _kernels.LaunchMropeF32(q, k, _state.RopePositionsDevice, _mropeAxisDevice, seqLen, numHeads, numKvHeads, headDim,
                    _ropeDim, _ropeTheta, streamH, ropeInvFreq: _ropeYarnInvFreq, ropeMscale: _ropeYarnMscale);
                break;
            case VisionRopeMode.Shifted:
                if (t_capturingDecode) throw new InvalidOperationException("A shifted rope position cannot be captured into a decode graph.");
                _kernels.LaunchRoPEF32(q, k, _state.RopePositionsDevice, seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, 1, streamH,
                    ropeInvFreq: _ropeYarnInvFreq, ropeMscale: _ropeYarnMscale);
                break;
            default:
                _kernels.LaunchRoPEF32(q, k, _state.PositionsDevice, seqLen, numHeads, numKvHeads, headDim, _ropeDim, _ropeTheta, 1, streamH,
                    ropeInvFreq: _ropeYarnInvFreq, ropeMscale: _ropeYarnMscale);
                break;
        }
    }

    /// <summary>Throws while the latest sequence carries a rope delta: the MTP / DFlash drafters rotate at the slot position.</summary>
    private void ThrowIfRopeDeltaActive()
    {
        if (_vision.LastDelta != 0)
            throw new NotSupportedException($"MTP / DFlash drafting is not supported after an image prompt (rope delta {_vision.LastDelta}): " +
                "the drafter would rotate at the slot position, not the M-RoPE position.");
    }
}
