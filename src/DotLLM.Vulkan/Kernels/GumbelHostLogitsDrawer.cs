using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Vulkan.Kernels;

/// <summary>
/// Generic device Gumbel draw for Vulkan hosts that have no in-place LM-head hook (#1435: Gemma4, Qwen4Exp, Qwen3-MoE-hybrid, Nemotron-H).
/// The host's ordinary forward returns its logits (already read back); this class uploads the rows and runs
/// <see cref="GumbelArgMaxRowsKernel"/> over them, so the O(vocabulary) Philox noise + log work (~17 ms per row at a 248k vocabulary on the
/// CPU) moves to the GPU and only one token id per row comes back. The draw is bit-for-bit <c>GumbelNoise.Sample</c> of the same logits.
/// </summary>
/// <remarks>
/// Cost vs the dense host's native hook: one extra upload of <c>rows x vocab</c> F32 (a UMA host-to-device copy, ~1 MB per row) on top of the
/// readback the forward already did. That is a bounce the in-place hook avoids; it is accepted here because the host loop it replaces is
/// 10-100x more expensive and no in-place LM-head hook exists on these hosts yet.
/// </remarks>
public sealed class GumbelHostLogitsDrawer : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly GumbelArgMaxRowsKernel _kernel;
    private VulkanDevice.Buffer? _logits, _ids, _params;
    private int _capacityRows, _capacityVocab;
    private float[] _staging = [];
    private uint[] _packed = [];
    private int[] _idsHost = [];

    /// <summary>Draws made through this instance (a perturbation counter for tests).</summary>
    public int Draws { get; private set; }

    /// <summary>Rows per launch this drawer handles.</summary>
    public const int MaxRows = 64;

    /// <summary>Creates a drawer on <paramref name="device"/>.</summary>
    public GumbelHostLogitsDrawer(VulkanDevice device, string spvDir)
    {
        _device = device;
        _kernel = GumbelArgMaxRowsKernel.Create(device, spvDir);
    }

    /// <summary>True when the per-row shader is present in <paramref name="spvDir"/>.</summary>
    public static bool IsAvailable(string? spvDir)
        => GumbelArgMaxRowsKernel.IsAvailable(spvDir) && Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_GUMBEL") != "0";

    /// <summary>
    /// <c>IModel.ForwardGumbelSampled</c> for a host without an in-place hook: runs the host's own verify forward, then draws every row.
    /// </summary>
    public void ForwardGumbelSampled(IModel model, ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, IKvCache? kvCache,
        IMtpState? mtpState, bool recurrentRowSnapshots, ulong seed, int firstTokenIndex, float invTemperature, GumbelFilter filter,
        Span<int> sampledIds)
    {
        int rows = tokenIds.Length;
        if (rows > MaxRows) throw new NotSupportedException($"{rows} rows exceed the device draw limit {MaxRows}.");
        using ITensor logits = recurrentRowSnapshots
            ? model.ForwardWithRecurrentSnapshots(tokenIds, positions, deviceId: -1, kvCache, mtpState)
            : model.Forward(tokenIds, positions, deviceId: -1, kvCache, adapter: null, mtpState);
        int vocab = model.Config.VocabSize;
        if (logits.Shape.Rank < 2 || logits.Shape[0] < rows)
            throw new InvalidOperationException($"Expected {rows} logits rows, the forward returned shape {logits.Shape}.");
        EnsureCapacity(rows, vocab);
        unsafe { new ReadOnlySpan<float>((void*)logits.DataPointer, rows * vocab).CopyTo(_staging); }
        for (int r = 0; r < rows; r++)
            GumbelArgMaxRowsKernel.PackRow(_packed.AsSpan(r * 8, 8), seed, firstTokenIndex + r, invTemperature, filter.TopK, filter.MinP);
        Launch(rows, vocab, sampledIds);
    }

    /// <summary>
    /// Draws one row per tensor in <paramref name="logits"/> (each tensor's FIRST row, the decode row) with its own <paramref name="draws"/> entry.
    /// </summary>
    public void DrawBatch(IReadOnlyList<ITensor> logits, int vocab, ReadOnlySpan<GumbelDraw> draws, Span<int> sampledIds)
    {
        for (int first = 0; first < logits.Count; first += MaxRows)
        {
            int rows = Math.Min(MaxRows, logits.Count - first);
            EnsureCapacity(rows, vocab);
            for (int r = 0; r < rows; r++)
            {
                unsafe { new ReadOnlySpan<float>((void*)logits[first + r].DataPointer, vocab).CopyTo(_staging.AsSpan(r * vocab, vocab)); }
                var d = draws[first + r];
                GumbelArgMaxRowsKernel.PackRow(_packed.AsSpan(r * 8, 8), d.Seed, d.TokenIndex, d.InvTemperature, d.TopK, d.MinP);
            }
            Launch(rows, vocab, sampledIds.Slice(first, rows));
        }
    }

    private void EnsureCapacity(int rows, int vocab)
    {
        if (_logits is not null && _capacityRows >= rows && _capacityVocab >= vocab) return;
        _logits?.Dispose(); _ids?.Dispose(); _params?.Dispose();
        _capacityRows = Math.Max(rows, 16);
        _capacityVocab = vocab;
        _logits = _device.Allocate((long)_capacityRows * vocab * sizeof(float));
        _ids = _device.AllocateHostReadback((long)_capacityRows * sizeof(uint));
        _params = _device.Allocate((long)_capacityRows * GumbelArgMaxRowsKernel.RowParamBytes);
        _staging = new float[(long)_capacityRows * vocab];
        _packed = new uint[_capacityRows * 8];
        _idsHost = new int[_capacityRows];
        _kernel.InvalidateDescriptorCache();
    }

    private void Launch(int rows, int vocab, Span<int> sampledIds)
    {
        _device.Upload(_staging.AsSpan(0, rows * vocab), _logits!);
        _device.Upload(MemoryMarshal.AsBytes(_packed.AsSpan(0, rows * 8)), _params!);
        _kernel.Launch(_logits!, _ids!, _params!, rows, vocab, vocab);
        _device.Download(_ids!, MemoryMarshal.Cast<int, float>(_idsHost.AsSpan(0, rows)));
        _idsHost.AsSpan(0, rows).CopyTo(sampledIds);
        Draws++;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _kernel.Dispose();
        _logits?.Dispose(); _ids?.Dispose(); _params?.Dispose();
    }
}
