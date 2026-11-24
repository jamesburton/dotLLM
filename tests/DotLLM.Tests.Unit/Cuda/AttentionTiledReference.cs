using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// The CPU decode oracle for GPU attention tests: the per-head tiled kernel, pinned on this thread (#1927). The pointer
/// <see cref="Attention.Execute(float*, float*, float*, float*, int, int, int, int, int, int, ComputeThreadPool?, int?, float, float[])"/>
/// otherwise decodes past 2,048 keys with the split-KV kernel (F16 partials, about 8e-5 from the tiled result), which is not the
/// reference these tests were tuned against.
/// </summary>
internal static unsafe class AttentionTiledReference
{
    public static void Execute(float* q, float* k, float* v, float* output, int seqQ, int seqKv, int numHeads, int numKvHeads,
                               int headDim, int positionOffset, ComputeThreadPool? pool, int? slidingWindowSize = null,
                               float softCap = 0f, float[]? sinks = null)
    {
        using var tiled = Attention.OverrideDecodeSplitKvOnThisThread(DecodeSplitKvPartials.Off);
        Attention.Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, pool,
            slidingWindowSize, softCap, sinks);
    }
}
