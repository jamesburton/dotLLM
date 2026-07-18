using System.Diagnostics;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Scaffold + baseline for a possible Vulkan "RoPE + KV-cache write" fusion
/// (tracked informally alongside <c>.docs/KERNEL_MAP.md</c> §5/§6/§11 and the
/// CUDA precedent <c>native/kernels/fused_rope_kv_write.cu</c> /
/// <c>CudaTransformerModel.cs:976-989</c>, which already fuses this pair for
/// decode). No fused Vulkan kernel exists yet — this file has two parts:
/// </summary>
/// <remarks>
/// <para>
/// <b>Part 1 (skipped, contract-only):</b> <see cref="FusedRopeKvWrite_MatchesTwoDispatchBaseline"/>
/// documents the exact numerical contract a future fused kernel must satisfy,
/// oracled against the same CPU reference the existing unfused parity test
/// (<c>VulkanRopeF32KernelTests</c>) uses. Un-skip and fill in once/if a
/// <c>RopeKvWriteF32Kernel</c> (or similarly named) fused kernel + <c>.comp</c>
/// shader is implemented.
/// </para>
/// <para>
/// <b>Part 2 (active):</b> <see cref="CurrentTwoDispatchPath_DecodeShapeTiming"/>
/// and <see cref="CurrentTwoDispatchPath_PrefillShapeTiming"/> measure the
/// CURRENT two-dispatch structure's wall-clock cost (RoPE compute dispatch +
/// COMPUTE→COMPUTE barrier + 2× <c>vkCmdCopyBuffer</c> TRANSFER + TRANSFER→COMPUTE
/// barrier) for a realistic SmolLM-135M-shaped layer, one submit per iteration
/// (matching the granularity a fused kernel would collapse to a single dispatch).
/// This is the baseline a fused implementation should be compared against.
/// </para>
/// <para>
/// <b>Current call site</b> (confirmed by reading the source, no code changed here):
/// <c>VulkanTransformerModel.cs:2432-2434</c> (prefill path) and
/// <c>:2967-2969</c> / <c>:4071-4076</c> (per-layer / MoE decode paths) record
/// <c>_rope.Record(cmdBuf, Q, K, positions, ...)</c> (a single COMPUTE dispatch —
/// <c>RopeF32Kernel.Record</c>, <c>src/DotLLM.Vulkan/Kernels/RopeF32Kernel.cs:130-197</c>,
/// backed by <c>native/vulkan/shaders/rope_f32.comp</c>), immediately followed
/// (<c>VulkanTransformerModel.cs:2984-2985</c>) by:
/// <code>
/// BarrierComputeToCompute(cmdBuf);              // RoPE(K) write -&gt; KV copy read
/// vkCache.RecordUpdate(cmdBuf, _state.K, _state.V, positions, seqLen, layer);
/// BarrierTransferToCompute(cmdBuf);              // KV copy write -&gt; attention read
/// </code>
/// <c>VulkanKvCache.RecordUpdate</c> (<c>src/DotLLM.Vulkan/VulkanKvCache.cs:194-233</c>)
/// is a PURE <c>vkCmdCopyBuffer</c> memory copy (TRANSFER queue-family work, not a
/// compute shader) — one contiguous-range copy for K and one for V when positions
/// are ascending-contiguous (the decode/normal-prefill case), or a per-row copy
/// loop otherwise (out-of-order positions, e.g. some batched/speculative paths).
/// No quantization and no page-table indirection happen in this copy — the cache
/// buffer is a flat <c>[maxSeqLen, kvStride]</c> row-major FP32 buffer
/// (<c>VulkanKvCache.cs:35-54</c>), so the destination row offset is a single
/// multiply (<c>pos * rowBytes</c>). This is why fusion looks like a buffer-rebind,
/// not new numerical work: <c>rope_f32.comp</c> already computes <c>i0</c>/<c>i1</c>
/// indices into a flat row-major K buffer at <c>[t * numKvHeads * headDim + head *
/// headDim + ...]</c> (<c>rope_f32.comp:114</c>); a fused shader would instead bind
/// the KV-cache K/V buffers as additional bindings and compute the SAME index
/// arithmetic against <c>cache_row_offset = pos * kvStride</c> (mirroring the CUDA
/// kernel's <c>region</c> scheme in <c>fused_rope_kv_write.cu:17-21,96-148</c>) —
/// no new math, no new synchronization primitive, just writing K (rotated) and V
/// (pass-through copy) directly into the cache instead of into <c>_state.K</c>/
/// <c>_state.V</c> scratch. Q still writes to scratch (attention reads Q from
/// <c>_state.Q</c>/<c>PerSeqQ</c>, never from the cache).
/// </para>
/// <para>
/// <b>What would block a naive fusion (checked, none apply to the common case):</b>
/// <list type="bullet">
///   <item><b>Non-contiguous positions</b> — <c>RecordUpdate</c>'s per-row-copy branch
///   (<c>VulkanKvCache.cs:225-233</c>) handles out-of-order/gapped positions
///   (batched multi-sequence forwards with interleaved slots). A fused shader
///   would need a per-thread position lookup instead of the current single
///   contiguous <c>startPos</c> push-constant — an easy generalization (index by
///   <c>positions[t]</c> per token, same as the existing RoPE shader already does
///   for the angle computation) but it IS an extra input the fused kernel needs
///   beyond the current unfused RoPE shader's contract.</item>
///   <item><b>Per-layer KV stride variance</b> — Gemma-4's sliding/global layers use
///   different <c>numKvHeads</c>×<c>headDim</c> per layer (<c>KvGeometry</c>,
///   <c>VulkanKvCache.cs:42-46</c>). Not a blocker — already a per-call push-constant
///   in both the RoPE shader and <c>RecordUpdate</c>; a fused kernel just needs
///   <c>kvStride</c> alongside <c>headDim</c>/<c>numKvHeads</c>.</item>
///   <item><b>Quantized KV cache</b> (<c>VulkanTurboQuantKvCache</c>) — this path is
///   NOT a plain copy; it's RoPE(K) -&gt; encode-to-codes -&gt; dequant-to-scratch
///   (<c>VulkanTransformerModel.cs:3016-3030</c>), a genuinely different shape of
///   fusion (RoPE + quantize, not RoPE + copy). Out of scope for a first fused
///   kernel — gate the fusion on <c>kvCache is VulkanKvCache</c> only, matching the
///   CUDA precedent's <c>kvCache is CudaKvCache</c> gate
///   (<c>CudaTransformerModel.cs:977</c>).</item>
///   <item><b>MLA / <c>MlaVulkanKvCache</c></b> — separate code path
///   (<c>RecordMlaLayer</c>, <c>VulkanTransformerModel.cs:3763-3777</c>) with its own
///   RoPE-on-latent kernel (<c>RopeMlaF32Kernel</c>) and split KV write
///   (<c>MlaKvSplitF32Kernel</c>). Not addressed by this scaffold.</item>
/// </list>
/// The common dense/GQA decode-and-prefill path against a plain
/// <see cref="VulkanKvCache"/> has NO blocker beyond the contiguous-vs-per-row
/// branch above — implementation difficulty is assessed as low (new shader +
/// buffer rebind, not a new synchronization design), similar in scale to the
/// already-landed CUDA fusion.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanRopeKvWriteFusionParityTests
{
    private readonly ITestOutputHelper _output;

    public VulkanRopeKvWriteFusionParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ------------------------------------------------------------------
    // Part 1 — contract-only scaffold for the (not yet implemented) fused kernel.
    // ------------------------------------------------------------------

    /// <summary>
    /// Documents the expected contract of a future fused RoPE+KV-write kernel
    /// (working name: <c>RopeKvWriteF32Kernel</c> / <c>rope_kv_write_f32.comp</c>)
    /// and the oracle it must satisfy. NOT implemented yet — kept skipped so CI
    /// stays green; un-skip once the kernel exists.
    /// </summary>
    /// <remarks>
    /// Intended contract (mirrors the CUDA <c>fused_rope_kv_write_f16</c> kernel,
    /// adapted to dotLLM's FP32 Vulkan KV cache):
    /// <list type="number">
    ///   <item>
    ///   Inputs: <c>Q</c> scratch buffer [seqLen, numHeads*headDim] (in/out, rotated
    ///   in place — unchanged from today), <c>K</c> scratch buffer [seqLen,
    ///   numKvHeads*headDim] (read-only source for rotation), <c>V</c> scratch buffer
    ///   [seqLen, numKvHeads*headDim] (read-only, no rotation), <c>positions</c>
    ///   [seqLen] int32 (RoPE angle source — same as today), K-cache buffer
    ///   [maxSeqLen, kvStride] and V-cache buffer [maxSeqLen, kvStride] for the
    ///   layer (NEW bindings vs today's RoPE shader), and either a single
    ///   <c>startPos</c> push-constant (contiguous case) or a per-token
    ///   <c>cachePositions</c> buffer (non-contiguous case, see the "non-contiguous
    ///   positions" remark on the class).
    ///   </item>
    ///   <item>
    ///   Outputs: <c>Q</c> rotated in place (bit-identical to
    ///   <see cref="RopeF32Kernel"/>'s current Q output). K-cache row at
    ///   <c>cachePos</c> receives the SAME rotated K values <see cref="RopeF32Kernel"/>
    ///   currently writes back into the K scratch buffer (bit-identical, same
    ///   angle formula, same NeoX/Norm pairing). V-cache row at <c>cachePos</c>
    ///   receives an unrotated copy of the V scratch row (bit-identical to what
    ///   <see cref="VulkanKvCache.RecordUpdate"/> currently produces via
    ///   <c>vkCmdCopyBuffer</c>).
    ///   </item>
    ///   <item>
    ///   Oracle: reuse the SAME CPU reference the unfused test already uses —
    ///   <see cref="RoPE.ExecuteScalar"/> / <see cref="RoPE.Execute"/> (NeoX) driven
    ///   by <see cref="RoPE.PrecomputeFrequencyTableScalar"/> — for the Q/K rotation
    ///   half of the oracle, PLUS a byte-for-byte copy check of V into the expected
    ///   cache row offset. Tolerance should match <c>VulkanRopeF32KernelTests</c>
    ///   (abs 1e-4 / rel 1e-3) since the rotation math is unchanged; the KV-cache
    ///   write portion has zero tolerance (it's a copy, not arithmetic).
    ///   </item>
    ///   <item>
    ///   Parity gate: run the SAME kernel-shape matrix as
    ///   <c>VulkanRopeF32KernelTests.Launch_MatchesCpuReference_Norm/_NeoX</c>
    ///   (MHA and GQA, short and long seqLen, Llama-3 theta) PLUS an explicit
    ///   decode shape (seqLen=1) since that's the only shape the CUDA fusion
    ///   targets and where this fusion pays off most.
    ///   </item>
    ///   <item>
    ///   Gate: only fire when <c>kvCache is VulkanKvCache</c> (not TurboQuant, not
    ///   MLA) — matching the CUDA precedent's <c>kvCache is CudaKvCache</c> gate.
    ///   </item>
    /// </list>
    /// </remarks>
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;

    /// <summary>
    /// Bit-for-bit (within the standard RoPE tolerance) parity between the fused
    /// <see cref="RopeKvWriteF32Kernel"/> single dispatch and the CURRENT two-dispatch
    /// path (<see cref="RopeF32Kernel"/> + <see cref="VulkanKvCache.RecordUpdate"/>) run
    /// side by side against independent cache instances. Also spot-checks against the CPU
    /// <see cref="RoPE.ExecuteScalar"/> oracle for Q/K, exactly as <c>VulkanRopeF32KernelTests</c>
    /// does, plus a byte-for-byte V-cache-row check.
    /// </summary>
    [SkippableTheory]
    // (seqLen, numHeads, numKvHeads, headDim, theta, positionOffset) — decode (seqLen=1),
    // short/long prefill (contiguous positions), GQA and MHA, Llama-3 theta, and a non-zero
    // positionOffset case (append past an already-populated cache prefix, matching a
    // continuing-decode call).
    [InlineData(1, 9, 3, 64, 10000f, 0)]     // decode, SmolLM-135M GQA shape
    [InlineData(1, 9, 3, 64, 10000f, 17)]    // decode continuing at position 17 (non-zero offset)
    [InlineData(4, 2, 2, 64, 10000f, 0)]     // short prefill, MHA
    [InlineData(4, 9, 3, 64, 10000f, 0)]     // short prefill, GQA
    [InlineData(256, 9, 3, 64, 10000f, 0)]   // long prefill, GQA — SmolLM-135M prefill shape
    [InlineData(1, 32, 8, 128, 500000f, 0)]  // decode, Llama-3 style theta
    [InlineData(8, 4, 2, 64, 10000f, 5)]     // prefill continuing at a non-zero cache offset
    public void FusedRopeKvWrite_MatchesTwoDispatchBaseline(
        int seqLen, int numHeads, int numKvHeads, int headDim, float theta, int positionOffset)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(
            RopeKvWriteF32Kernel.IsAvailable(spvDir),
            "rope_kv_write_f32.spv not built — run native/vulkan/build.ps1.");

        int ropeDim = headDim; // rotate the full head
        int halfDim = ropeDim / 2;
        int maxSeqLen = positionOffset + seqLen + 1;

        var rng = new Random(0x380F0 + seqLen * 13 + numHeads * 7 + headDim + positionOffset);
        float[] q = RandomFloats(rng, seqLen * numHeads * headDim);
        float[] k = RandomFloats(rng, seqLen * numKvHeads * headDim);
        float[] v = RandomFloats(rng, seqLen * numKvHeads * headDim);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = positionOffset + i;

        // CPU reference (Norm/interleaved) via scalar path using pre-computed tables —
        // same oracle VulkanRopeF32KernelTests uses. RoPE.ExecuteScalar indexes the
        // table by ABSOLUTE position (positions[t], not t), so the table must cover
        // [0, positionOffset+seqLen) — sizing it to seqLen alone (as the zero-offset
        // callers can get away with) throws ArgumentOutOfRangeException once
        // positionOffset > 0.
        int tableLen = positionOffset + seqLen;
        float[] cosTable = new float[tableLen * halfDim];
        float[] sinTable = new float[tableLen * halfDim];
        RoPE.PrecomputeFrequencyTableScalar(tableLen, headDim, theta, cosTable, sinTable);

        float[] qExpected = (float[])q.Clone();
        float[] kExpectedRotated = (float[])k.Clone();
        RoPE.ExecuteScalar(
            qExpected.AsSpan(), kExpectedRotated.AsSpan(), positions,
            numHeads, numKvHeads, headDim, ropeDim,
            cosTable, sinTable);

        using var device = VulkanDevice.Create();

        int qElems = seqLen * numHeads * headDim;
        int kElems = seqLen * numKvHeads * headDim;
        int kvStride = numKvHeads * headDim;

        // ── Baseline: unfused RopeF32Kernel + VulkanKvCache.RecordUpdate, on its own cache. ──
        using var ropeKernel = RopeF32Kernel.Create(device, spvDir);
        using var bufQBaseline = device.Allocate((long)qElems * sizeof(float));
        using var bufKBaseline = device.Allocate((long)kElems * sizeof(float));
        using var bufVBaseline = device.Allocate((long)kElems * sizeof(float));
        using var bufPos = device.Allocate((long)seqLen * sizeof(int));
        var kvCacheBaseline = new VulkanKvCache(device, numLayers: 1, numKvHeads: numKvHeads, headDim: headDim, maxSeqLen: maxSeqLen);

        device.Upload(q.AsSpan(), bufQBaseline);
        device.Upload(k.AsSpan(), bufKBaseline);
        device.Upload(v.AsSpan(), bufVBaseline);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(positions.AsSpan()), bufPos);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            ropeKernel.Record(ctx.CommandBuffer, bufQBaseline, bufKBaseline, bufPos,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: ropeDim, theta: theta,
                variant: RopeF32Kernel.Variant.Norm);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kvCacheBaseline.RecordUpdate(ctx.CommandBuffer, bufKBaseline, bufVBaseline, positions, seqLen, layerIndex: 0);
            KernelSupport.TransferToComputeBarrier(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }

        float[] qBaselineActual = new float[qElems];
        device.Download(bufQBaseline, qBaselineActual);
        float[] kCacheBaseline = new float[maxSeqLen * kvStride];
        float[] vCacheBaseline = new float[maxSeqLen * kvStride];
        device.Download(kvCacheBaseline.GetKeysBuffer(0), kCacheBaseline);
        device.Download(kvCacheBaseline.GetValuesBuffer(0), vCacheBaseline);

        // ── Fused: RopeKvWriteF32Kernel, on an independent cache. ──
        using var fusedKernel = RopeKvWriteF32Kernel.Create(device, spvDir);
        using var bufQFused = device.Allocate((long)qElems * sizeof(float));
        using var bufKFused = device.Allocate((long)kElems * sizeof(float));
        using var bufVFused = device.Allocate((long)kElems * sizeof(float));
        var kvCacheFused = new VulkanKvCache(device, numLayers: 1, numKvHeads: numKvHeads, headDim: headDim, maxSeqLen: maxSeqLen);

        device.Upload(q.AsSpan(), bufQFused);
        device.Upload(k.AsSpan(), bufKFused);
        device.Upload(v.AsSpan(), bufVFused);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            fusedKernel.Record(ctx.CommandBuffer,
                bufQFused, bufKFused, bufVFused, bufPos,
                kvCacheFused.GetKeysBuffer(0), kvCacheFused.GetValuesBuffer(0),
                startPos: positionOffset, kvStride: kvStride,
                seqLen: seqLen, numHeads: numHeads, numKvHeads: numKvHeads,
                headDim: headDim, ropeDim: ropeDim, theta: theta,
                variant: RopeF32Kernel.Variant.Norm);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        kvCacheFused.AdvanceCurrentLength(positions, seqLen);

        float[] qFusedActual = new float[qElems];
        device.Download(bufQFused, qFusedActual);
        float[] kCacheFused = new float[maxSeqLen * kvStride];
        float[] vCacheFused = new float[maxSeqLen * kvStride];
        device.Download(kvCacheFused.GetKeysBuffer(0), kCacheFused);
        device.Download(kvCacheFused.GetValuesBuffer(0), vCacheFused);

        // Q: fused kernel vs CPU oracle AND vs the unfused baseline's Q output.
        AssertClose(qExpected, qFusedActual, "Q vs CPU oracle");
        AssertClose(qBaselineActual, qFusedActual, "Q fused vs unfused baseline");

        // K-cache: fused kernel's cache rows vs the unfused baseline's cache rows
        // (both already contain the SAME rotated K — the CPU oracle for K was computed
        // above into kExpectedRotated but is not directly indexed against cache rows
        // here since the baseline path already validates that mapping via VulkanRopeF32KernelTests).
        AssertClose(kCacheBaseline, kCacheFused, "K-cache fused vs unfused baseline");

        // Spot-check K-cache rows against the CPU oracle directly (rotated K values written at
        // [positionOffset + t, :]).
        for (int t = 0; t < seqLen; t++)
        {
            int cacheRow = positionOffset + t;
            for (int e = 0; e < kvStride; e++)
            {
                float expected = kExpectedRotated[t * kvStride + e];
                float actual = kCacheFused[cacheRow * kvStride + e];
                float diff = MathF.Abs(expected - actual);
                float rel = diff / MathF.Max(MathF.Abs(expected), 1e-7f);
                Assert.True(diff <= AbsTol || rel <= RelTol,
                    $"K-cache[{cacheRow},{e}]: expected {expected}, actual {actual} (diff={diff}, rel={rel})");
            }
        }

        // V-cache: zero-tolerance copy check — fused vs unfused baseline, AND fused vs the
        // raw V source (no rotation should ever touch V).
        Assert.Equal(vCacheBaseline, vCacheFused);
        for (int t = 0; t < seqLen; t++)
        {
            int cacheRow = positionOffset + t;
            for (int e = 0; e < kvStride; e++)
            {
                Assert.Equal(v[t * kvStride + e], vCacheFused[cacheRow * kvStride + e]);
            }
        }
    }

    // ------------------------------------------------------------------
    // Part 2 — active baseline timing of the CURRENT two-dispatch path.
    // ------------------------------------------------------------------

    // SmolLM-135M shape (per .docs/KERNEL_MAP.md §12 / issue #143 campaign):
    // 30L / hidden=576 / 9 Q heads / 3 KV heads / headDim=64 / vocab=49152.
    // One layer's RoPE+KV-write is what we time here; the campaign's per-token
    // decode overhead numbers are the reference point for whether a fusion's
    // saving would be noticeable end-to-end.
    private const int NumHeads = 9;
    private const int NumKvHeads = 3;
    private const int HeadDim = 64;
    private const float Theta = 10000f;

    private const int WarmupIters = 32;
    private const int TimedIters = 256;

    /// <summary>
    /// Times the current decode-shape (seqLen=1) two-dispatch RoPE+KV-write path:
    /// one <see cref="RopeF32Kernel"/> COMPUTE dispatch, one COMPUTE→COMPUTE
    /// barrier, two <c>vkCmdCopyBuffer</c> TRANSFER copies (K and V) via
    /// <see cref="VulkanKvCache.RecordUpdate"/>, one TRANSFER→COMPUTE barrier —
    /// all in a single submit per iteration (one <c>vkQueueSubmit</c> + fence
    /// wait), matching the granularity a fused single-dispatch kernel would
    /// collapse this to. This is the PER LAYER PER TOKEN cost at decode — the
    /// highest-frequency operation in the system. Multiply by layer count for
    /// the full-model per-token saving a fusion could plausibly capture (upper
    /// bound; actual end-to-end saving also depends on how much of this is
    /// already hidden behind other dispatches' latency).
    /// </summary>
    [SkippableFact]
    public void CurrentTwoDispatchPath_DecodeShapeTiming()
    {
        RunAndReport(seqLen: 1, contiguousPositions: true, label: "decode (seqLen=1)");
    }

    /// <summary>
    /// Same as <see cref="CurrentTwoDispatchPath_DecodeShapeTiming"/> but for a
    /// prefill-shaped chunk (seqLen=32, contiguous positions) — the PER LAYER
    /// (not per-token) cost at prefill, i.e. what a fusion would save once per
    /// chunk rather than once per token.
    /// </summary>
    [SkippableFact]
    public void CurrentTwoDispatchPath_PrefillShapeTiming()
    {
        RunAndReport(seqLen: 32, contiguousPositions: true, label: "prefill (seqLen=32)");
    }

    private void RunAndReport(int seqLen, bool contiguousPositions, string label)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        using var rope = RopeF32Kernel.Create(device, spvDir);

        int qElems = seqLen * NumHeads * HeadDim;
        int kElems = seqLen * NumKvHeads * HeadDim;
        int maxSeqLen = seqLen + 1; // room for one decode-append past any prior positions

        using var bufQ = device.Allocate((long)qElems * sizeof(float));
        using var bufK = device.Allocate((long)kElems * sizeof(float));
        using var bufV = device.Allocate((long)kElems * sizeof(float));
        using var bufPos = device.Allocate((long)seqLen * sizeof(int));
        var kvCache = new VulkanKvCache(device, numLayers: 1, numKvHeads: NumKvHeads, headDim: HeadDim, maxSeqLen: Math.Max(maxSeqLen, 512));

        var rng = new Random(0x2026_0717);
        float[] qInit = new float[qElems];
        float[] kInit = new float[kElems];
        float[] vInit = new float[kElems];
        for (int i = 0; i < qElems; i++) qInit[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kElems; i++) kInit[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kElems; i++) vInit[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        device.Upload(qInit.AsSpan(), bufQ);
        device.Upload(kInit.AsSpan(), bufK);
        device.Upload(vInit.AsSpan(), bufV);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(positions.AsSpan()), bufPos);

        void RecordOneIteration(nint cmdBuf)
        {
            // Mirrors VulkanTransformerModel.cs:2967-2985 exactly: RoPE dispatch,
            // COMPUTE->COMPUTE barrier, KV-cache RecordUpdate (2x vkCmdCopyBuffer),
            // TRANSFER->COMPUTE barrier.
            rope.Record(cmdBuf, bufQ, bufK, bufPos,
                seqLen: seqLen, numHeads: NumHeads, numKvHeads: NumKvHeads,
                headDim: HeadDim, ropeDim: HeadDim, theta: Theta,
                variant: RopeF32Kernel.Variant.Norm);
            KernelSupport.ComputeToComputeBarrier(cmdBuf);
            kvCache.RecordUpdate(cmdBuf, bufK, bufV, positions, seqLen, layerIndex: 0);
            KernelSupport.TransferToComputeBarrier(cmdBuf);
        }

        // Warmup — let the GPU clock ramp before timing (avoids the cold-launch
        // clock-ramp confound documented in project memory
        // vulkan-perf-ab-clock-ramp-confound).
        for (int i = 0; i < WarmupIters; i++)
        {
            using var warmCtx = device.CreateSubmitContext();
            warmCtx.Begin();
            RecordOneIteration(warmCtx.CommandBuffer);
            warmCtx.SubmitAndWait();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TimedIters; i++)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            RecordOneIteration(ctx.CommandBuffer);
            ctx.SubmitAndWait();
        }
        sw.Stop();

        double usPerIter = sw.Elapsed.TotalMicroseconds / TimedIters;
        _output.WriteLine(
            $"[rope+kv-write baseline] {label}: {usPerIter:F1} us/iteration "
            + $"(1 RoPE dispatch + 1 barrier + 2 vkCmdCopyBuffer + 1 barrier, {TimedIters} iters). "
            + "This is the CURRENT two-dispatch cost a future fused single-dispatch kernel would replace. "
            + "For reference, .docs/KERNEL_MAP.md's #145 fused rmsnorm+quantize precedent measured "
            + "+3% end-to-end from removing one dispatch+barrier pair at a similarly hot call site — "
            + "treat this per-iteration number the same way: multiply by layer count and compare against "
            + "measured per-token decode time (issue #143 campaign) before assuming a proportional win, "
            + "since some of this cost may already overlap with adjacent GPU work rather than being pure "
            + "added latency.");

        // Sanity assertion only — this is a measurement, not a regression gate.
        // A meaningfully broken submit path (e.g. device lost) would make this
        // trivially fail; we don't assert a specific latency bound since iGPU
        // clocks vary run-to-run (see vulkan-perf-ab-clock-ramp-confound memory).
        Assert.True(usPerIter > 0, "Timed loop should report positive elapsed time.");
    }

    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // [-1, 1]
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, string tensorName)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"{tensorName}: Numerical drift exceeded tolerance: " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
