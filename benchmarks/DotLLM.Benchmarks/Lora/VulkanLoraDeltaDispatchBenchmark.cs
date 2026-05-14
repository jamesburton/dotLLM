using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Engines;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Benchmarks.Lora;

/// <summary>
/// Vulkan-side fused vs. un-fused LoRA delta dispatch comparison. Replicates
/// the per-decode-token dispatch shape of TinyLlama-1.1B
/// (22 layers × 7 LoRA-adapted projections per layer) at the four "common"
/// adapter shapes — q/k/v/o (hidden=2048 → 2048), gate/up (2048 → 5632),
/// down (5632 → 2048).
/// </summary>
/// <remarks>
/// <para>
/// Baseline: the original 4-dispatch chain
/// <c>matmul(B) → matmul(A) → add → vkCmdCopyBuffer</c> per delta site.
/// </para>
/// <para>
/// Fused: a single <see cref="LoraDeltaGemvFusedF32Kernel"/> dispatch that
/// performs <c>y[t,m] += sum_r A[m,r] · dot(B[r,:], x[t,:])</c> in place.
/// </para>
/// <para>
/// Each iteration submits the full per-token dispatch sequence
/// (154 LoRA delta sites for TinyLlama, scaled-down here so the bench runs
/// in seconds instead of minutes) inside one command buffer and waits on the
/// fence. Wall-clock difference is dominated by per-dispatch fixed cost
/// (descriptor binding, push constants, barrier flushes) at decode-path
/// shapes — the math itself is small.
/// </para>
/// <para>
/// Self-skipping: returns immediately when <c>VulkanDevice.IsAvailable</c>
/// is false or the SPV directory isn't found. BenchmarkDotNet has no native
/// "skip" — the bench reports a meaningless tiny number that the runner
/// treats as a no-op iteration.
/// </para>
/// </remarks>
[MemoryDiagnoser]
[SimpleJob(RunStrategy.Throughput, warmupCount: 3, iterationCount: 5, invocationCount: 4)]
public class VulkanLoraDeltaDispatchBenchmark
{
    /// <summary>LoRA rank — 16 is the most common PEFT default.</summary>
    [Params(8, 16, 32)]
    public int Rank { get; set; }

    /// <summary>How many layers to mimic per submitted batch (TinyLlama=22). Keep small for bench wall-clock.</summary>
    [Params(22)]
    public int LayerCount { get; set; }

    // TinyLlama-1.1B projection shapes.
    private const int Hidden = 2048;          // q/k/v/o input + output, gate/up input, down output
    private const int Intermediate = 5632;    // gate/up output, down input
    private const int SeqLen = 1;             // decode path

    private VulkanDevice? _device;
    private string? _spvDir;
    private LoraDeltaGemvFusedF32Kernel? _fused;
    private MatMulF32Kernel? _matmul;
    private AddKernel? _add;

    // Per-projection upload sites: [B, A, x, y, scratchTmp, scratchDelta, scratchSum].
    // 7 LoRA targets per layer, × LayerCount layers, all sized to one of two shapes.
    private VulkanDevice.Buffer[]? _bBufs;
    private VulkanDevice.Buffer[]? _aBufs;
    private VulkanDevice.Buffer[]? _xBufs;
    private VulkanDevice.Buffer[]? _yBufs;
    private VulkanDevice.Buffer[]? _tmpBufs;
    private VulkanDevice.Buffer[]? _deltaBufs;
    private VulkanDevice.Buffer[]? _sumBufs;
    private (int InputDim, int OutputDim)[]? _shapes;

    private VulkanDevice.SubmitContext? _submit;

    private static readonly string[] ProjOrder =
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];

    [GlobalSetup]
    public void Setup()
    {
        if (!VulkanDevice.IsAvailable())
            return;

        _spvDir = FindSpvDir();
        if (_spvDir is null) return;

        _device = VulkanDevice.Create();
        _fused = LoraDeltaGemvFusedF32Kernel.Create(_device, _spvDir);
        _matmul = MatMulF32Kernel.Create(_device, _spvDir);
        _add = AddKernel.Create(_device, _spvDir);

        int sites = LayerCount * ProjOrder.Length;
        _bBufs = new VulkanDevice.Buffer[sites];
        _aBufs = new VulkanDevice.Buffer[sites];
        _xBufs = new VulkanDevice.Buffer[sites];
        _yBufs = new VulkanDevice.Buffer[sites];
        _tmpBufs = new VulkanDevice.Buffer[sites];
        _deltaBufs = new VulkanDevice.Buffer[sites];
        _sumBufs = new VulkanDevice.Buffer[sites];
        _shapes = new (int, int)[sites];

        var rng = new Random(0xCAFE + Rank * 31 + LayerCount * 17);
        for (int layer = 0; layer < LayerCount; layer++)
        {
            for (int p = 0; p < ProjOrder.Length; p++)
            {
                int idx = layer * ProjOrder.Length + p;
                (int inputDim, int outputDim) = ShapeFor(ProjOrder[p]);
                _shapes[idx] = (inputDim, outputDim);

                _bBufs[idx] = AllocAndFill(_device, rng, (long)Rank * inputDim);
                _aBufs[idx] = AllocAndFill(_device, rng, (long)outputDim * Rank);
                _xBufs[idx] = AllocAndFill(_device, rng, (long)SeqLen * inputDim);
                _yBufs[idx] = AllocAndFill(_device, rng, (long)SeqLen * outputDim);
                _tmpBufs[idx] = _device.AllocateDeviceLocal((long)SeqLen * Rank * sizeof(float));
                _deltaBufs[idx] = _device.AllocateDeviceLocal((long)SeqLen * outputDim * sizeof(float));
                _sumBufs[idx] = _device.AllocateDeviceLocal((long)SeqLen * outputDim * sizeof(float));
            }
        }

        _submit = _device.CreateSubmitContext();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _submit?.Dispose();
        if (_bBufs is not null)
        {
            for (int i = 0; i < _bBufs.Length; i++)
            {
                _bBufs[i].Dispose();
                _aBufs![i].Dispose();
                _xBufs![i].Dispose();
                _yBufs![i].Dispose();
                _tmpBufs![i].Dispose();
                _deltaBufs![i].Dispose();
                _sumBufs![i].Dispose();
            }
        }
        _add?.Dispose();
        _matmul?.Dispose();
        _fused?.Dispose();
        _device?.Dispose();
    }

    /// <summary>Baseline: the four-dispatch chain per delta site.</summary>
    [Benchmark(Baseline = true)]
    public void UnfusedFourDispatch()
    {
        if (_device is null || _submit is null) return;

        _submit.Begin();
        var cmdBuf = _submit.CommandBuffer;
        for (int i = 0; i < _shapes!.Length; i++)
        {
            var (inputDim, outputDim) = _shapes[i];
            _matmul!.Record(cmdBuf, _bBufs![i], _xBufs![i], _tmpBufs![i],
                m: Rank, k: inputDim, n: SeqLen);
            KernelSupport_ComputeBarrier(cmdBuf);
            _matmul.Record(cmdBuf, _aBufs![i], _tmpBufs[i], _deltaBufs![i],
                m: outputDim, k: Rank, n: SeqLen);
            KernelSupport_ComputeBarrier(cmdBuf);
            _add!.Record(cmdBuf, _yBufs![i], _deltaBufs[i], _sumBufs![i], SeqLen * outputDim);
            KernelSupport_TransferBarrier(cmdBuf);
            CopyBackInto(cmdBuf, _sumBufs[i], _yBufs[i], (long)SeqLen * outputDim * sizeof(float));
            KernelSupport_TransferToComputeBarrier(cmdBuf);
        }
        _submit.SubmitAndWait();
    }

    /// <summary>Fused: one dispatch per delta site.</summary>
    [Benchmark]
    public void FusedSingleDispatch()
    {
        if (_device is null || _submit is null) return;

        _submit.Begin();
        var cmdBuf = _submit.CommandBuffer;
        for (int i = 0; i < _shapes!.Length; i++)
        {
            var (inputDim, outputDim) = _shapes[i];
            _fused!.Record(cmdBuf, _xBufs![i], _bBufs![i], _aBufs![i], _yBufs![i], _tmpBufs![i],
                seqLen: SeqLen, inputDim: inputDim, outputDim: outputDim, rank: Rank);
            KernelSupport_ComputeBarrier(cmdBuf);
        }
        _submit.SubmitAndWait();
    }

    private static (int InputDim, int OutputDim) ShapeFor(string proj) => proj switch
    {
        "q_proj" or "k_proj" or "v_proj" or "o_proj" => (Hidden, Hidden),
        "gate_proj" or "up_proj"                     => (Hidden, Intermediate),
        "down_proj"                                  => (Intermediate, Hidden),
        _ => throw new ArgumentOutOfRangeException(nameof(proj)),
    };

    private static unsafe VulkanDevice.Buffer AllocAndFill(VulkanDevice device, Random rng, long elementCount)
    {
        long bytes = elementCount * sizeof(float);
        var buf = device.AllocateDeviceLocal(bytes);
        var data = new float[elementCount];
        for (long i = 0; i < elementCount; i++)
            data[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
        // Stage through a host-visible buffer.
        using var staging = device.Allocate(bytes);
        var dataBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(data.AsSpan());
        device.UploadToDeviceLocal(dataBytes, staging, buf);
        return buf;
    }

    private static unsafe void KernelSupport_ComputeBarrier(nint cmdBuf)
        => KernelSupport.ComputeToComputeBarrier(cmdBuf);

    private static unsafe void KernelSupport_TransferBarrier(nint cmdBuf)
        => KernelSupport.ComputeToTransferBarrier(cmdBuf);

    private static unsafe void KernelSupport_TransferToComputeBarrier(nint cmdBuf)
        => KernelSupport.TransferToComputeBarrier(cmdBuf);

    private static unsafe void CopyBackInto(nint cmdBuf, VulkanDevice.Buffer src, VulkanDevice.Buffer dst, long bytes)
    {
        var region = new DotLLM.Vulkan.Interop.VkBufferCopy
        {
            srcOffset = 0,
            dstOffset = 0,
            size = (ulong)bytes,
        };
        DotLLM.Vulkan.Interop.VulkanApi.vkCmdCopyBuffer(cmdBuf, src.Handle, dst.Handle, 1, region);
    }

    private static string? FindSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv")),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "..", "native", "vulkan", "spv")),
        };
        foreach (var c in candidates)
            if (Directory.Exists(c) && File.Exists(Path.Combine(c, "lora_delta_gemv_fused_f32.spv")))
                return c;
        return null;
    }
}
