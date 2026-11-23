using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// The GenerationSession control's 1-ULP residual perturbation, one instrument for every backend (#1919, James 2026-11-22 "add GPU seams
/// now"): a ONE-SHOT test-only residual modifier that moves every element of one token row of the residual (<see cref="Row"/>) after one
/// layer (<see cref="Layer"/>) by exactly <see cref="Ulps"/> ULPs away from zero in the residual's storage dtype: F32 on CPU, the Vulkan
/// models and the CUDA hybrids, F16 on the CUDA dense model. It rides the model's own PostLayer seam (<see cref="IResidualModifierHost"/>
/// on CPU, <see cref="IVulkanResidualModifierHost"/>, <see cref="ICudaResidualModifierHost"/>), so a model with no seam yields no arm.
/// Why the whole row, not one element: a single-element nudge is absorbed on some families (the Q8 activation quantizer downstream turns it
/// into the same step), so the row is the smallest unit whose observability holds on every backend and family; this is asserted bitwise by
/// <c>OneUlpArm_IsObservable</c> on each of them.
/// </summary>
internal abstract class ResidualUlpArm : IDisposable
{
    /// <summary>Layer the nudge follows. 0 = the first layer: the perturbation then runs through the whole trunk.</summary>
    public const int Layer = 0;
    /// <summary>Token row nudged (the first token of the first forward).</summary>
    public const int Row = 0;
    /// <summary>ULPs added. The "nudge zero ULPs" mutant sets 0 and OneUlpArm_IsObservable must fail.</summary>
    public static int Ulps = 1;

    /// <summary>Times the model actually called the perturber (an inert hook is distinguished from an absorbed perturbation by this).</summary>
    public int Fired { get; protected set; }

    /// <summary>The arm for <paramref name="model"/>, or null when the model has no residual-modifier seam.</summary>
    public static ResidualUlpArm? For(IModel model) => model switch
    {
        IResidualModifierHost cpu => new CpuArm(cpu),
        IVulkanResidualModifierHost vk => new VulkanArm(vk),
        ICudaResidualModifierHost cuda => new CudaArm(cuda),
        _ => null,
    };

    /// <summary>Layer the installed perturber follows; <see cref="int.MaxValue"/> makes it attached but never applicable (the inert-when-unset A/B).</summary>
    protected int TargetLayer = Layer;

    /// <summary>Installs a fresh one-shot perturber after <paramref name="layer"/>, remembering the modifier that was attached.</summary>
    public void Attach(int layer = Layer) { TargetLayer = layer; AttachCore(); }

    /// <summary>Per-backend install of the perturber (see <see cref="Attach"/>).</summary>
    protected abstract void AttachCore();

    /// <summary>Detaches whatever modifier the model has (a fixture counter included) and returns it for <see cref="RestoreModifier"/>: the "hook never attached" state.</summary>
    public abstract object? ClearModifier();

    /// <summary>Puts back what <see cref="ClearModifier"/> returned.</summary>
    public abstract void RestoreModifier(object? saved);

    /// <summary>Restores the modifier that was attached before <see cref="Attach"/> (null when none).</summary>
    public abstract void Detach();

    /// <summary>Whether a perturber is installed now.</summary>
    public abstract bool Attached { get; }

    public virtual void Dispose() => Detach();

    private static float Nudge(float v)
    {
        for (int i = 0; i < Ulps; i++) v = v >= 0 ? MathF.BitIncrement(v) : MathF.BitDecrement(v);
        return v;
    }

    // ── CPU ────────────────────────────────────────────────────────────────────────────────────────
    private sealed class CpuArm(IResidualModifierHost host) : ResidualUlpArm
    {
        private IResidualModifier? _saved;
        private bool _attached;
        public override bool Attached => _attached;
        public override object? ClearModifier() { var m = host.ResidualModifier; host.ResidualModifier = null; return m; }
        public override void RestoreModifier(object? saved) => host.ResidualModifier = (IResidualModifier?)saved;

        protected override void AttachCore()
        {
            if (_attached) return;
            _saved = host.ResidualModifier;
            host.ResidualModifier = new Perturber(this);
            _attached = true;
        }

        public override void Detach()
        {
            if (!_attached) return;
            host.ResidualModifier = _saved;
            _attached = false;
        }

        private sealed class Perturber(CpuArm arm) : IResidualModifier
        {
            private bool _done;
            public bool AppliesTo(int layerIndex) => layerIndex == arm.TargetLayer && !_done;
            public void ModifyResidual(Span<float> residual, int layerIndex, int tokenCount)
            {
                if (_done || layerIndex != arm.TargetLayer) return;
                _done = true;
                arm.Fired++;
                int hidden = residual.Length / tokenCount;
                for (int i = 0; i < hidden; i++) residual[Row * hidden + i] = Nudge(residual[Row * hidden + i]);
            }
        }
    }

    // ── Vulkan ─────────────────────────────────────────────────────────────────────────────────────
    private sealed class VulkanArm : ResidualUlpArm
    {
        private readonly IVulkanResidualModifierHost _host;
        private readonly ResidualUlpNudgeF32Kernel _kernel;
        private IVulkanResidualModifier? _saved;
        private bool _attached;
        public override bool Attached => _attached;
        public override object? ClearModifier() { var m = _host.ResidualModifier; _host.ResidualModifier = null; return m; }
        public override void RestoreModifier(object? saved) => _host.ResidualModifier = (IVulkanResidualModifier?)saved;

        public VulkanArm(IVulkanResidualModifierHost host)
        {
            _host = host;
            VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
            _kernel = ResidualUlpNudgeF32Kernel.Create(host.ResidualDevice, spvDir);
        }

        protected override void AttachCore()
        {
            if (_attached) return;
            _saved = _host.ResidualModifier;
            _host.ResidualModifier = new Perturber(this);
            _attached = true;
        }

        public override void Detach()
        {
            if (!_attached) return;
            _host.ResidualModifier = _saved;
            _attached = false;
        }

        public override void Dispose()
        {
            Detach();
            _kernel.Dispose();
        }

        private sealed class Perturber(VulkanArm arm) : IVulkanResidualModifier
        {
            private bool _done;
            public bool AppliesTo(int layerIndex) => layerIndex == arm.TargetLayer && !_done;

            public void Record(nint cmdBuf, VulkanDevice.Buffer residual, int layerIndex, int tokenCount, int hiddenSize)
            {
                if (_done || layerIndex != arm.TargetLayer) return;
                _done = true;
                arm.Fired++;
                // One-shot and the only recorder on this kernel: a fresh descriptor cache per recording cannot alias a stale buffer handle.
                arm._kernel.InvalidateDescriptorCache();
                // The shader adds 1 to the bit pattern (one ULP away from zero): Ulps > 1 records it again.
                for (int u = 0; u < Ulps; u++)
                {
                    if (u > 0) KernelSupport.ComputeToComputeBarrier(cmdBuf);
                    arm._kernel.Record(cmdBuf, residual, Row * hiddenSize, hiddenSize);
                }
            }

            public void InvalidateDescriptorCache() => arm._kernel.InvalidateDescriptorCache();
        }
    }

    // ── CUDA ───────────────────────────────────────────────────────────────────────────────────────
    private sealed class CudaArm(ICudaResidualModifierHost host) : ResidualUlpArm
    {
        private ICudaResidualModifier? _saved;
        private bool _attached;
        public override bool Attached => _attached;
        public override object? ClearModifier() { var m = host.ResidualModifier; host.ResidualModifier = null; return m; }
        public override void RestoreModifier(object? saved) => host.ResidualModifier = (ICudaResidualModifier?)saved;

        protected override void AttachCore()
        {
            if (_attached) return;
            _saved = host.ResidualModifier;
            host.ResidualModifier = new Perturber(this);
            _attached = true;
        }

        public override void Detach()
        {
            if (!_attached) return;
            host.ResidualModifier = _saved;
            _attached = false;
        }

        /// <summary>Test-only, so it may sync: stream sync, D2H of the one row, nudge each element's bit pattern, H2D, stream sync.</summary>
        private sealed unsafe class Perturber(CudaArm arm) : ICudaResidualModifier
        {
            private bool _done;
            public bool AppliesTo(int layerIndex) => layerIndex == arm.TargetLayer && !_done;

            public void Apply(nint residualDevicePtr, CudaResidualDType dtype, int layerIndex, int tokenCount, int hiddenSize, nint stream)
            {
                if (_done || layerIndex != arm.TargetLayer) return;
                _done = true;
                arm.Fired++;
                int elem = dtype == CudaResidualDType.F16 ? sizeof(ushort) : sizeof(float);
                nuint bytes = (nuint)(hiddenSize * elem);
                nint rowDev = residualDevicePtr + (nint)((long)Row * hiddenSize * elem);
                byte[] row = new byte[(int)bytes];
                fixed (byte* p = row)
                {
                    CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                    CudaDriverApi.cuMemcpyDtoHAsync_v2((nint)p, rowDev, bytes, stream).ThrowOnError();
                    CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                    for (int i = 0; i < hiddenSize; i++)
                    {
                        // Sign-magnitude: +1 on the bit pattern is one ULP away from zero for either sign (BitIncrement/BitDecrement on the CPU).
                        if (dtype == CudaResidualDType.F16) ((ushort*)p)[i] = (ushort)(((ushort*)p)[i] + Ulps);
                        else ((uint*)p)[i] = (uint)(((uint*)p)[i] + Ulps);
                    }
                    CudaDriverApi.cuMemcpyHtoDAsync_v2(rowDev, (nint)p, bytes, stream).ThrowOnError();
                    CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                }
            }
        }
    }
}
