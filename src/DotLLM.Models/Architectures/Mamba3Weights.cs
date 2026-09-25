using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Resolved weight handles for a Mamba-3 checkpoint — one
/// <see cref="Mamba3TensorHandle"/> per tensor declared in
/// <see cref="Mamba3TensorMapping"/>. Produced by
/// <see cref="Mamba3WeightLoader"/>.
/// </summary>
/// <remarks>
/// <para>
/// This record is the boundary between Stage D2 (weight loading) and
/// Stage D3 (model forward). It contains no compute logic: every
/// non-null handle points at F32 row-major data either inside a mapped
/// <see cref="SafeTensors.SafetensorsFile"/> or a loader-owned
/// 64-byte-aligned scratch allocation (see <see cref="Mamba3TensorHandle"/>).
/// </para>
/// <para>
/// <b>Tying.</b> When the underlying checkpoint sets
/// <c>tie_word_embeddings=true</c>, <see cref="LmHead"/>'s pointer and
/// shape are aliased to <see cref="TokenEmbedding"/>'s and its
/// <see cref="Mamba3TensorHandle.OwnsMemory"/> is <c>false</c> to avoid a
/// double-free on <see cref="Dispose"/>.
/// </para>
/// <para>
/// <b>Vulkan-only quant overlay.</b> The matmul-target projections (<see cref="LmHead"/>
/// plus the per-layer <see cref="Mamba3LayerWeights.InProj"/> and
/// <see cref="Mamba3LayerWeights.OutProj"/>) carry optional raw-quant byte overlay
/// pointers that are zero on production load paths and only populated by tests / a
/// future quant-aware loader. When set, the Vulkan upload keeps the raw quant blocks
/// on device (gated on the contraction axis being a multiple of the format's group
/// size — 32 for Q8_0, 256 for Q4_K / Q5_K / Q6_K) and dispatches the matmuls through the matching
/// kernel — same two-mode storage policy as the standard transformer at
/// <c>VulkanWeights</c>. The CPU forward continues to consume the F32 handle
/// (<see cref="LmHead"/>, etc.); when an overlay is set the F32 source must already
/// carry values equivalent to dequantising the raw quant bytes so the CPU-vs-Vulkan
/// comparison is fair. Small per-layer tensors (norms, biases, D, dt_bias, MIMO
/// per-rank weights) are NOT overlaid — they stay F32 in every mode (matches what
/// production GGUFs ship for SSMs). The overlay slots use the historical "Q8" naming
/// because Q8_0 was the first quant type wired through; they actually carry raw bytes
/// for whichever format the companion <c>*QuantTypeOverlay</c> field declares —
/// Q8_0, Q4_K, Q5_K, or Q6_K (Phase 1 of the K-quant work, now complete for the
/// Vulkan matmul kernels — coopmat variants and the remaining K-quant formats
/// (Q2_K, Q3_K) remain follow-up tickets).
/// </para>
/// </remarks>
public sealed class Mamba3Weights : IDisposable
{
    private bool _disposed;

    /// <summary>Token-embedding matrix, shape <c>[vocab_size, hidden_size]</c>.</summary>
    public required Mamba3TensorHandle TokenEmbedding { get; set; }

    /// <summary>Final pre-LM-head RMSNorm gain, shape <c>[hidden_size]</c>.</summary>
    public required Mamba3TensorHandle FinalNorm { get; set; }

    /// <summary>
    /// LM head matrix, shape <c>[vocab_size, hidden_size]</c>. When the
    /// checkpoint ties embeddings to the LM head, this handle's pointer and
    /// shape are aliased to <see cref="TokenEmbedding"/> with
    /// <see cref="Mamba3TensorHandle.OwnsMemory"/> set to <c>false</c>.
    /// </summary>
    /// <remarks>
    /// Settable so tests / a future quant-aware loader can rewire the F32 source pointer
    /// to a buffer holding dequantised-from-Q8_0 values when attaching the
    /// <see cref="LmHeadQ8Ptr"/> overlay; production load paths set this once at
    /// construction and never touch it again.
    /// </remarks>
    public required Mamba3TensorHandle LmHead { get; set; }

    /// <summary>Optional raw-quant bytes for the LM head (<c>[vocab, hidden]</c>).
    /// Zero when <see cref="LmHead"/> stays F32 on device. When non-zero the Vulkan
    /// upload keeps the raw quant blocks (Q8_0, Q4_K, Q5_K, or Q6_K, declared via
    /// <see cref="LmHeadQuantTypeOverlay"/>); the CPU oracle continues reading
    /// <see cref="LmHead"/>'s F32 data, which must hold values equivalent to
    /// dequantising the raw bytes for parity. Production loaders never set this.</summary>
    public nint LmHeadQ8Ptr { get; set; }

    /// <summary>Storage type of the <see cref="LmHeadQ8Ptr"/> overlay. One of
    /// <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is set.</summary>
    public QuantizationType LmHeadQuantTypeOverlay { get; set; } = QuantizationType.F32;

    /// <summary>Per-layer weights, ordered by physical layer index.</summary>
    public required Mamba3LayerWeights[] Layers { get; init; }

    /// <summary>
    /// Optional per-layer Q8_0 overlays for the matmul-target projections
    /// <c>in_proj</c> and <c>out_proj</c>. Null on production load paths; tests
    /// allocate one entry per layer and populate the relevant raw-byte pointers
    /// to drive the Vulkan Q8_0 matmul path. Length must equal
    /// <see cref="Layers"/>.Length when non-null. The <see cref="Mamba3LayerWeights"/>
    /// record struct itself stays F32-only / immutable — this side-array is the
    /// mutable opt-in slot.
    /// </summary>
    public Mamba3LayerQuantOverlay[]? LayerOverlays { get; set; }

    /// <summary>
    /// Structured diagnostics produced at load time: one entry per tensor
    /// the loader touched or was expected to touch. Callers should check
    /// <see cref="Mamba3WeightLoadReport.HasMissingRequired"/> before using
    /// the weights for a forward pass.
    /// </summary>
    public required Mamba3WeightLoadReport Report { get; init; }

    /// <summary>
    /// Releases every loader-owned <see cref="NativeMemory.AlignedAlloc"/>
    /// allocation wrapped by this record. Tensor handles whose
    /// <see cref="Mamba3TensorHandle.OwnsMemory"/> is <c>false</c> (mmap
    /// views, tied-embedding aliases) are left alone — the underlying
    /// <see cref="SafeTensors.SafetensorsFile"/> remains the lifetime
    /// anchor for those.
    /// </summary>
    public unsafe void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        FreeIfOwned(TokenEmbedding);
        FreeIfOwned(FinalNorm);
        if (LmHead.OwnsMemory && LmHead.Pointer != TokenEmbedding.Pointer)
            FreeIfOwned(LmHead);

        foreach (var layer in Layers)
        {
            FreeIfOwned(layer.Norm);
            FreeIfOwned(layer.InProj);
            FreeIfOwned(layer.OutProj);
            FreeIfOwned(layer.BNorm);
            FreeIfOwned(layer.CNorm);
            FreeIfOwned(layer.BBias);
            FreeIfOwned(layer.CBias);
            FreeIfOwned(layer.D);
            FreeIfOwned(layer.DtBias);
            FreeIfOwned(layer.MimoX);
            FreeIfOwned(layer.MimoZ);
            FreeIfOwned(layer.MimoO);
        }
    }

    private static unsafe void FreeIfOwned(Mamba3TensorHandle h)
    {
        if (h.OwnsMemory && h.Pointer != nint.Zero)
            NativeMemory.AlignedFree((void*)h.Pointer);
    }
}

/// <summary>
/// Per-layer Mamba-3 mixer + norm weight handles, matching the 9-entry
/// SISO (or 12-entry MIMO) per-layer signature declared in
/// <see cref="Mamba3TensorMapping"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>SISO layout.</b> <see cref="BBias"/> / <see cref="CBias"/> carry their
/// HF on-disk 3-D shape (<c>[num_heads, 1, state_size]</c>); the block-level
/// consumer squeezes the singleton middle axis when it uses them.
/// <see cref="MimoX"/> / <see cref="MimoZ"/> / <see cref="MimoO"/> are
/// <see cref="Mamba3TensorHandle.Empty"/> for SISO.
/// </para>
/// <para>
/// <b>MIMO layout.</b> <see cref="BBias"/> / <see cref="CBias"/> carry the
/// rank-expanded shape <c>[num_heads, mimo_rank, state_size]</c>.
/// <see cref="MimoX"/> / <see cref="MimoZ"/> / <see cref="MimoO"/> are each
/// populated with shape <c>[num_heads, mimo_rank, head_dim]</c>.
/// </para>
/// </remarks>
public readonly record struct Mamba3LayerWeights(
    Mamba3TensorHandle Norm,
    Mamba3TensorHandle InProj,
    Mamba3TensorHandle OutProj,
    Mamba3TensorHandle BNorm,
    Mamba3TensorHandle CNorm,
    Mamba3TensorHandle BBias,
    Mamba3TensorHandle CBias,
    Mamba3TensorHandle D,
    Mamba3TensorHandle DtBias,
    Mamba3TensorHandle MimoX = default,
    Mamba3TensorHandle MimoZ = default,
    Mamba3TensorHandle MimoO = default);

/// <summary>
/// Optional per-layer quant overlay for the matmul-target projections of one Mamba-3
/// layer. Holds raw-byte pointers that the Vulkan upload may keep on device verbatim
/// when the corresponding <see cref="QuantizationType"/> is one of the supported
/// formats (Q8_0 — group size 32 — or Q4_K / Q5_K / Q6_K — group size 256, Phase 1
/// of the K-quant work) and the contraction axis is a multiple of that group size.
/// Production load
/// paths leave this at the default (F32 / null pointers); tests populate it to drive
/// the Vulkan quantised matmul kernels.
/// </summary>
/// <remarks>
/// <para>
/// The CPU forward pass continues to read the F32 handles on
/// <see cref="Mamba3LayerWeights"/>; when <see cref="InProjQuantTypeOverlay"/> /
/// <see cref="OutProjQuantTypeOverlay"/> is set to a quantised format the F32 source
/// must already carry values equivalent to dequantising the raw quant bytes so the
/// CPU-vs-Vulkan comparison is fair.
/// </para>
/// <para>
/// The slot names use "Q8" for historical reasons (Q8_0 was the first quant type
/// wired through); the slots actually carry raw bytes for whichever format the
/// companion <c>*QuantTypeOverlay</c> field declares.
/// </para>
/// </remarks>
public sealed class Mamba3LayerQuantOverlay
{
    /// <summary>Optional raw-quant bytes for <c>in_proj</c>
    /// (<c>[d_in_proj, hidden_size]</c>). Format declared by
    /// <see cref="InProjQuantTypeOverlay"/>.</summary>
    public nint InProjQ8Ptr;

    /// <summary>Storage type of <see cref="InProjQ8Ptr"/>. One of
    /// <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is set.</summary>
    public QuantizationType InProjQuantTypeOverlay = QuantizationType.F32;

    /// <summary>Optional raw-quant bytes for <c>out_proj</c>
    /// (<c>[hidden_size, d_inner]</c>). Format declared by
    /// <see cref="OutProjQuantTypeOverlay"/>.</summary>
    public nint OutProjQ8Ptr;

    /// <summary>Storage type of <see cref="OutProjQ8Ptr"/>. One of
    /// <see cref="QuantizationType.Q8_0"/>, <see cref="QuantizationType.Q4_K"/>,
    /// <see cref="QuantizationType.Q5_K"/>, or <see cref="QuantizationType.Q6_K"/>;
    /// <see cref="QuantizationType.F32"/> when no overlay is set.</summary>
    public QuantizationType OutProjQuantTypeOverlay = QuantizationType.F32;
}

/// <summary>
/// Opaque pointer + shape metadata for a single Mamba-3 weight tensor.
/// </summary>
/// <remarks>
/// <para>
/// The pointer is either:
/// </para>
/// <list type="bullet">
///   <item>a view into a memory-mapped <see cref="SafeTensors.SafetensorsFile"/>
///   (<c>OwnsMemory = false</c>), or</item>
///   <item>a loader-owned 64-byte-aligned allocation via
///   <see cref="NativeMemory.AlignedAlloc"/> (<c>OwnsMemory = true</c>).</item>
/// </list>
/// <para>
/// Data is always F32 row-major in the Stage D2 path — the
/// <paramref name="SourceDType"/> field is recorded for diagnostics and is
/// the hook for a future bf16 ingest.
/// </para>
/// </remarks>
/// <param name="Pointer">Absolute pointer to the first element.</param>
/// <param name="Shape">Tensor shape (row-major); never null.</param>
/// <param name="SourceDType">Dtype token the descriptor advertised.</param>
/// <param name="OwnsMemory">
/// <c>true</c> if the pointer must be released via
/// <see cref="NativeMemory.AlignedFree"/> at dispose time; <c>false</c> if
/// it aliases a shared mmap view.
/// </param>
public readonly record struct Mamba3TensorHandle(
    nint Pointer,
    int[] Shape,
    SafeTensors.SafetensorsDType SourceDType,
    bool OwnsMemory)
{
    /// <summary>A null/empty handle (used for tensors the loader could not find).</summary>
    public static Mamba3TensorHandle Empty { get; } =
        new(nint.Zero, Array.Empty<int>(), SafeTensors.SafetensorsDType.Unknown, false);

    /// <summary>Whether this handle was actually populated.</summary>
    public bool IsPopulated => Pointer != nint.Zero;

    /// <summary>Total element count (product of <see cref="Shape"/>).</summary>
    public long ElementCount
    {
        get
        {
            if (Shape is null) return 0;
            long n = 1;
            for (int i = 0; i < Shape.Length; i++) n *= Shape[i];
            return n;
        }
    }
}

/// <summary>
/// Why a tensor the loader expected wasn't populated.
/// </summary>
public enum Mamba3TensorIssueKind
{
    /// <summary>Tensor was present and shape-validated successfully.</summary>
    Ok = 0,

    /// <summary>Tensor name was not in the safetensors header.</summary>
    Missing,

    /// <summary>Tensor was present but its shape did not match the config.</summary>
    ShapeMismatch,

    /// <summary>Tensor was present but its dtype is not yet supported.</summary>
    UnsupportedDType,
}

/// <summary>
/// Per-tensor diagnostic entry. Combines cleanly to build structured
/// error reports without throwing eagerly on the first missing tensor.
/// </summary>
/// <param name="TensorName">Canonical HF name (from <see cref="Mamba3TensorMapping"/>).</param>
/// <param name="Kind">Classification of the finding.</param>
/// <param name="Detail">Human-readable explanation.</param>
/// <param name="IsRequired">
/// Whether the consumer must have this tensor before a forward pass. For
/// Stage D2 every mapped tensor is required, but e.g. a tied LM head
/// (absent <see cref="Mamba3TensorMapping.LmHead"/>) is optional when the
/// config sets <c>tie_word_embeddings=true</c>.
/// </param>
public readonly record struct Mamba3TensorDiagnostic(
    string TensorName,
    Mamba3TensorIssueKind Kind,
    string Detail,
    bool IsRequired);

/// <summary>
/// Structured load-time diagnostics produced by
/// <see cref="Mamba3WeightLoader"/>.
/// </summary>
public sealed class Mamba3WeightLoadReport
{
    /// <summary>All diagnostics recorded during load (includes successes).</summary>
    public required IReadOnlyList<Mamba3TensorDiagnostic> Entries { get; init; }

    /// <summary>Number of tensors loaded successfully.</summary>
    public int LoadedCount { get; init; }

    /// <summary>
    /// Number of required tensors the loader could not produce — either
    /// because they were missing, shape-mismatched, or had an unsupported
    /// dtype.
    /// </summary>
    public int MissingRequiredCount { get; init; }

    /// <summary>
    /// <c>true</c> iff at least one required tensor is not populated.
    /// Callers MUST short-circuit on this before attempting a forward pass.
    /// </summary>
    public bool HasMissingRequired => MissingRequiredCount > 0;

    /// <summary>
    /// Enumerates only the <see cref="Entries"/> that signal a problem
    /// (everything with <c>Kind != Ok</c>).
    /// </summary>
    public IEnumerable<Mamba3TensorDiagnostic> Problems =>
        Entries.Where(e => e.Kind != Mamba3TensorIssueKind.Ok);
}
