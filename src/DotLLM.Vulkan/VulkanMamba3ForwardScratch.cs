using DotLLM.Core.Models;
using DotLLM.Models.Architectures;

namespace DotLLM.Vulkan;

/// <summary>
/// Per-forward scratch buffers for the Mamba-3 Vulkan path (SISO and MIMO). Sized for the
/// maximum <c>seqLen</c> seen so far; grows monotonically with <see cref="EnsureCapacity"/>.
/// Mirrors <see cref="VulkanNemotronHForwardState"/> in slot management but with the
/// Mamba-3-specific scratch shapes (Proj, X, Z, Dt, Adt, Trap, Gamma, Scale, AnglesRaw,
/// B, C, QkPreDot, YScan).
/// </summary>
/// <remarks>
/// <para>
/// Two of the SSM-prep scratch buffers (the per-token preprocessing tables — DT, ADT, Trap,
/// Gamma, Scale, AnglesRaw, B, C, X, Z, QkPreDot) are populated host-side after downloading
/// the in_proj output, then uploaded back. They are therefore allocated host-visible
/// (<see cref="VulkanDevice.Allocate"/>) — same pattern NemotronH uses for its
/// <c>PositionsBuffer</c>. The downloaded <see cref="Proj"/> buffer is also host-visible
/// since the host needs to read it.
/// </para>
/// <para>
/// The remaining scratch (HiddenState pair, NormOutput, BlockOut, Logits) is device-local —
/// touched only by GPU kernels.
/// </para>
/// <para>
/// <b>Rank-aware B / C.</b> The B and C buffers are sized at
/// <c>seqLen · max(1, mimoRank) · n_head · d_state</c> so that a single allocation covers
/// the canonical SISO <c>[T, H, N]</c> and MIMO <c>[T, R, H, N]</c> layouts. The MIMO
/// scan kernel reads them with stride <c>R·H·N</c> per token; SISO collapses R=1 trivially.
/// </para>
/// </remarks>
internal sealed class VulkanMamba3ForwardScratch : IDisposable
{
    private readonly VulkanDevice _device;
    private readonly int _hiddenSize;
    private readonly int _dInProj;
    private readonly int _dInner;
    private readonly int _nHead;
    // n_head * effectiveRank * d_state — effectiveRank == max(1, mimoRank) so SISO
    // (r==1) and MIMO share the same allocation footprint per token.
    private readonly int _bcWidth;
    private readonly int _numRopeAngles;
    private readonly int _vocabSize;

    private int _capacitySeqLen;
    private bool _disposed;

    // ── Hidden / residual rotation pair (mirrors VulkanNemotronHForwardState) ──
    private readonly VulkanDevice.Buffer[] _hiddenSlots = new VulkanDevice.Buffer[2];
    private int _hiddenIdx;
    public VulkanDevice.Buffer HiddenState => _hiddenSlots[_hiddenIdx];
    public VulkanDevice.Buffer Residual => _hiddenSlots[_hiddenIdx];
    public VulkanDevice.Buffer AddScratch => _hiddenSlots[_hiddenIdx ^ 1];
    public void RotateHiddenSlot() => _hiddenIdx ^= 1;
    public void ResetHiddenSlot() => _hiddenIdx = 0;

    public VulkanDevice.Buffer NormOutput { get; private set; } = null!;
    public VulkanDevice.Buffer BlockOut { get; private set; } = null!;

    // ── In-projection output (device-local; downloaded for CPU prep) ──
    public VulkanDevice.Buffer Proj { get; private set; } = null!;             // [seqLen, d_in_proj]

    // ── Host-prepared per-token tables (host-visible; uploaded each layer) ──
    public VulkanDevice.Buffer X { get; private set; } = null!;                // [seqLen, d_inner]
    public VulkanDevice.Buffer Z { get; private set; } = null!;                // [seqLen, d_inner]
    public VulkanDevice.Buffer Dt { get; private set; } = null!;               // [seqLen, n_head]
    public VulkanDevice.Buffer Adt { get; private set; } = null!;              // [seqLen, n_head]
    public VulkanDevice.Buffer Gamma { get; private set; } = null!;            // [seqLen, n_head]
    public VulkanDevice.Buffer Scale { get; private set; } = null!;            // [seqLen, n_head]
    public VulkanDevice.Buffer AnglesRaw { get; private set; } = null!;        // [seqLen, num_rope_angles]
    public VulkanDevice.Buffer B { get; private set; } = null!;                // [seqLen, R, n_head, d_state] (R=1 for SISO)
    public VulkanDevice.Buffer C { get; private set; } = null!;                // [seqLen, R, n_head, d_state] (R=1 for SISO)
    public VulkanDevice.Buffer QkPreDot { get; private set; } = null!;         // [seqLen, n_head]
    public VulkanDevice.Buffer YScan { get; private set; } = null!;            // [seqLen, n_head, head_dim]

    // Host-visible per-head coefficient buffer for the streaming-chunk boundary
    // adjustment: coef[h] = dt[0, h] · (1 - trap[0, h]). Sized to nHead — does
    // NOT scale with seqLen, so it lives outside the AllocateForCapacity grow path
    // and is allocated once in the constructor + freed in Dispose.
    public VulkanDevice.Buffer BoundaryCoef { get; private set; } = null!;     // [n_head]

    // ── Final logits (1-row, last token only) ──
    public VulkanDevice.Buffer Logits { get; private set; }
    public VulkanDevice.Buffer LastTokenHidden { get; private set; } = null!;  // [hidden]

    public long AllocatedBytes { get; private set; }

    public VulkanMamba3ForwardScratch(VulkanDevice device, ModelConfig config, int initialSeqLen)
    {
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(config);
        Mamba3Config m3 = config.Mamba3Config
            ?? throw new ArgumentException(
                "ModelConfig.Mamba3Config must be populated for VulkanMamba3ForwardScratch.",
                nameof(config));

        _device = device;
        _hiddenSize = config.HiddenSize;
        _dInProj = m3.InputProjectionDim;
        _dInner = m3.DInner;
        _nHead = m3.NumHeads;
        // Rank-aware B/C width — SISO collapses R to 1; MIMO uses the canonical [T, R, H, N]
        // layout the MIMO scan kernel expects.
        int effectiveRank = m3.IsMimo ? m3.MimoRank : 1;
        _bcWidth = _nHead * effectiveRank * m3.StateSize;
        _numRopeAngles = m3.NumRopeAngles;
        _vocabSize = config.VocabSize;

        Logits = device.Allocate((long)_vocabSize * sizeof(float));
        // Per-head boundary-adjustment coefficient buffer — fixed-size (no seqLen
        // dependence), so allocated once here and reused across every Forward.
        BoundaryCoef = device.Allocate((long)_nHead * sizeof(float));
        AllocateForCapacity(Math.Max(1, initialSeqLen));
    }

    public bool EnsureCapacity(int seqLen)
    {
        if (seqLen <= _capacitySeqLen) return false;
        ReleaseLayerScratch();
        AllocateForCapacity(seqLen);
        return true;
    }

    private void AllocateForCapacity(int seqLen)
    {
        long hiddenBytes = (long)seqLen * _hiddenSize * sizeof(float);
        long projBytes = (long)seqLen * _dInProj * sizeof(float);
        long innerBytes = (long)seqLen * _dInner * sizeof(float);
        long headBytes = (long)seqLen * _nHead * sizeof(float);
        long bcBytes = (long)seqLen * _bcWidth * sizeof(float);
        long angBytes = (long)seqLen * _numRopeAngles * sizeof(float);

        _hiddenSlots[0] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenSlots[1] = _device.AllocateDeviceLocal(hiddenBytes);
        _hiddenIdx = 0;
        NormOutput = _device.AllocateDeviceLocal(hiddenBytes);
        BlockOut = _device.AllocateDeviceLocal(hiddenBytes);
        LastTokenHidden = _device.AllocateDeviceLocal((long)_hiddenSize * sizeof(float));

        // Proj is host-visible — we map and download it after the in_proj matmul so the
        // host can run the per-token preprocessing block. (Compute shaders read host-visible
        // memory just fine; on UMA this is zero-copy, on discrete GPUs there's a PCIe hop
        // but correctness is unaffected. Quant-prep shaders are a future perf follow-up.)
        Proj = _device.Allocate(projBytes);

        // Host-visible tables — written by host, read by the data-RoPE / SSD scan kernels.
        X = _device.Allocate(innerBytes);
        Z = _device.Allocate(innerBytes);
        Dt = _device.Allocate(headBytes);
        Adt = _device.Allocate(headBytes);
        Gamma = _device.Allocate(headBytes);
        Scale = _device.Allocate(headBytes);
        AnglesRaw = _device.Allocate(angBytes);
        B = _device.Allocate(bcBytes);
        C = _device.Allocate(bcBytes);
        QkPreDot = _device.Allocate(headBytes);
        YScan = _device.Allocate(innerBytes);

        _capacitySeqLen = seqLen;
        AllocatedBytes = hiddenBytes * 3 + (long)_hiddenSize * sizeof(float)
                       + projBytes
                       + innerBytes * 3            // X, Z, YScan
                       + headBytes * 5             // Dt, Adt, Gamma, Scale, QkPreDot
                       + angBytes
                       + bcBytes * 2               // B, C
                       + (long)_vocabSize * sizeof(float)
                       + (long)_nHead * sizeof(float);   // BoundaryCoef (per-head, no seqLen scale)
    }

    private void ReleaseLayerScratch()
    {
        _hiddenSlots[0]?.Dispose();
        _hiddenSlots[1]?.Dispose();
        _hiddenSlots[0] = null!;
        _hiddenSlots[1] = null!;
        NormOutput?.Dispose();
        BlockOut?.Dispose();
        LastTokenHidden?.Dispose();
        Proj?.Dispose();
        X?.Dispose();
        Z?.Dispose();
        Dt?.Dispose();
        Adt?.Dispose();
        Gamma?.Dispose();
        Scale?.Dispose();
        AnglesRaw?.Dispose();
        B?.Dispose();
        C?.Dispose();
        QkPreDot?.Dispose();
        YScan?.Dispose();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        ReleaseLayerScratch();
        Logits?.Dispose();
        BoundaryCoef?.Dispose();
    }
}
