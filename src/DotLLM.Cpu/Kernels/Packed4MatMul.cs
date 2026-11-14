using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Bounded CPU execution of mapped AutoAWQ GEMM4 and legacy AutoGPTQ4 projections.
/// Accumulates with F32 activations; neither model weights nor scales are widened in storage.
/// Eight outputs accumulate in a vector register; there is no explicit F32 weight or accumulator scratch.
/// </summary>
public static unsafe class Packed4MatMul
{
    /// <summary>c[token * m + row] = dot(weight[row, :], b[token * k :]).</summary>
    public static void Gemm(Packed4Weight* weight, float* b, float* c, int m, int k, int n,
        ComputeThreadPool? pool = null)
    {
        if (weight is null || b is null || c is null)
            throw new ArgumentException("Packed projection and activation pointers must be non-null.");
        if (m <= 0 || k <= 0 || n <= 0 || weight->OutputFeatures != m || weight->InputFeatures != k
            || m % 8 != 0 || weight->GroupSize <= 0 || k % weight->GroupSize != 0)
            throw new ArgumentException("Packed projection dimensions do not match the validated descriptor.");
        if (weight->Layout is not (QuantizationType.Awq4 or QuantizationType.Gptq4)
            || weight->QWeight is null || weight->Scales is null || weight->QZeros is null
            || (weight->Layout == QuantizationType.Gptq4 && (weight->GroupIndices is null || k % 32 != 0)))
            throw new ArgumentException("Invalid packed AWQ/GPTQ descriptor.");
        var context = new Context { Weight = weight, Input = b, Output = c, Tasks = checked((m / 8) * n) };
        if (pool is null) Worker((nint)(&context), 0, 1);
        else pool.Dispatch((nint)(&context), &Worker);
    }

    private struct Context
    {
        public Packed4Weight* Weight;
        public float* Input;
        public float* Output;
        public int Tasks;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Worker(nint address, int threadIndex, int threadCount)
    {
        ref var context = ref Unsafe.AsRef<Context>((void*)address);
        var w = context.Weight;
        int m = w->OutputFeatures, k = w->InputFeatures, blocks = m / 8;
        int start = (int)((long)context.Tasks * threadIndex / threadCount);
        int end = (int)((long)context.Tasks * (threadIndex + 1) / threadCount);
        bool awq = w->Layout == QuantizationType.Awq4;
        for (int task = start; task < end; task++)
        {
            int token = task / blocks, block = task % blocks, output = block * 8;
            var sums = Vector256<float>.Zero;
            float* input = context.Input + (long)token * k;
            for (int column = 0; column < k; column++)
            {
                int group = awq ? column / w->GroupSize : w->GroupIndices[column];
                Half* scales = w->Scales + (long)group * m + output;
                uint zeros = w->QZeros[(long)group * blocks + block];
                uint packed = awq ? w->QWeight[(long)column * blocks + block] : 0;
                uint* gptq = w->QWeight + (long)(column / 8) * m + output;
                // Decode directly into a transient vector, without storing widened weights or
                // partial sums. The final store below is the existing F32 activation output.
                var decoded = Vector256.Create(
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 0),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 1),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 2),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 3),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 4),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 5),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 6),
                    DecodeLane(awq, packed, zeros, gptq, scales, column, 7));
                sums += Vector256.Create(input[column]) * decoded;
            }
            sums.Store(context.Output + (long)token * m + output);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DecodeLane(bool awq, uint packed, uint zeros, uint* gptq,
        Half* scales, int column, int lane)
    {
        // AutoAWQ pack order [0,2,4,6,1,3,5,7]: logical lane -> packed lane.
        int shift = awq ? ((lane % 2) * 4 + lane / 2) * 4 : lane * 4;
        int zero = (int)((zeros >> shift) & 15) + (awq ? 0 : 1);
        int code = awq ? (int)((packed >> shift) & 15)
            : (int)((gptq[lane] >> ((column % 8) * 4)) & 15);
        return (code - zero) * (float)scales[lane];
    }
}
