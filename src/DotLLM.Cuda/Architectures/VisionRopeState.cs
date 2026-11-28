using System.Runtime.CompilerServices;
using DotLLM.Core.Vision;

namespace DotLLM.Cuda.Architectures;

/// <summary>How the full-attention layers rotate Q/K for one forward (issue #2073).</summary>
internal enum VisionRopeMode
{
    /// <summary>The slot position, as without images.</summary>
    Plain,
    /// <summary>The slot position minus the sequence's rope delta (text after an image).</summary>
    Shifted,
    /// <summary>Interleaved M-RoPE on the image grid's (t, y, x) positions.</summary>
    Mrope,
}

/// <summary>
/// The GPU-free half of the CUDA Qwen3.5 hybrid model's image input: which rope mode a forward uses, the per-sequence "rope delta"
/// bookkeeping and the host-side position arrays the model uploads. Mirrors <c>Qwen3HybridDenseTransformerModel.PrepareVision</c> (CPU)
/// so the two cannot drift: with images the delta is <see cref="MropePositions.TotalDelta"/>; without images a forward at slot 0 starts a new
/// sequence and clears it, and any other forward (decode, verify, a later prefill chunk) rotates at <c>slot - delta</c>.
/// </summary>
internal sealed class VisionRopeState
{
    private readonly ConditionalWeakTable<object, StrongBox<int>> _deltas = new();
    private int[] _axes = [];

    /// <summary>Rope mode of the last <see cref="Prepare"/>.</summary>
    public VisionRopeMode Mode { get; private set; }

    /// <summary>The delta subtracted from slots when <see cref="Mode"/> is <see cref="VisionRopeMode.Shifted"/>.</summary>
    public int Shift { get; private set; }

    /// <summary>Rows of the last <see cref="Prepare"/>.</summary>
    public int Rows { get; private set; }

    /// <summary>
    /// The delta of the most recently prepared sequence (0 once a new sequence starts). Paths that cannot honour a rope delta
    /// (MTP, DFlash) refuse to run while this is non-zero.
    /// </summary>
    public int LastDelta { get; private set; }

    /// <summary>
    /// Host positions of the last <see cref="Prepare"/>: <see cref="VisionRopeMode.Mrope"/> lays out <c>[t][y][x]</c> as three runs of
    /// <see cref="Rows"/> ints; <see cref="VisionRopeMode.Shifted"/> holds <c>slot - delta</c> in the first <see cref="Rows"/> ints.
    /// </summary>
    public ReadOnlySpan<int> Axes => _axes.AsSpan(0, Mode == VisionRopeMode.Mrope ? 3 * Rows : Mode == VisionRopeMode.Shifted ? Rows : 0);

    /// <summary>The delta recorded for <paramref name="key"/> (0 when none).</summary>
    public int DeltaFor(object key) => _deltas.TryGetValue(key, out var b) ? b.Value : 0;

    /// <summary>Back to <see cref="VisionRopeMode.Plain"/> without touching the recorded deltas (paths that never carry images).</summary>
    public void SetPlain() { Mode = VisionRopeMode.Plain; Shift = 0; Rows = 0; }

    /// <summary>Decides the rope mode of a forward over <paramref name="slots"/> and fills the host position arrays.</summary>
    /// <param name="key">The sequence's identity (its KV cache, or the model when there is none).</param>
    /// <param name="slots">Absolute slot positions of the rows.</param>
    /// <param name="images">The request's image spans (absolute slots), or <see langword="null"/> / empty for none.</param>
    public void Prepare(object key, ReadOnlySpan<int> slots, IReadOnlyList<ImageSpan>? images)
    {
        int n = slots.Length;
        Rows = n;
        if (images is { Count: > 0 })
        {
            int delta = MropePositions.TotalDelta(images);
            _deltas.AddOrUpdate(key, new StrongBox<int>(delta));
            LastDelta = delta;
            if (_axes.Length < 3 * n) _axes = new int[3 * n];
            MropePositions.Fill(slots, images, _axes.AsSpan(0, n), _axes.AsSpan(n, n), _axes.AsSpan(2 * n, n));
            Mode = VisionRopeMode.Mrope;
            Shift = 0;
            return;
        }

        int d = 0;
        if (n > 0 && slots[0] == 0) { _deltas.Remove(key); LastDelta = 0; }
        else { d = DeltaFor(key); LastDelta = d; }
        if (d == 0) { Mode = VisionRopeMode.Plain; Shift = 0; return; }
        Mode = VisionRopeMode.Shifted;
        Shift = d;
        if (_axes.Length < n) _axes = new int[n];
        for (int i = 0; i < n; i++) _axes[i] = slots[i] - d;
    }
}
