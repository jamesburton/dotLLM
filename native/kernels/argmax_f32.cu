// Device argmax over an FP32 vector — issue #486 (MTP greedy draft: one int to the host instead of
// a 248k-float logits row).
//
//   out[0] = argmax(x[0..n))
//
// Contract — identical to System.Numerics.Tensors.TensorPrimitives.IndexOfMax, which the host
// decoder uses on the full-logits path:
//   * the largest value wins; on a tie the LOWEST index wins;
//   * if any value is NaN, the index of the first NaN is returned;
//   * +0 ranks above -0 (they compare equal as floats, so the sign decides).
// am_better below is a strict total order on (value, index) pairs under those rules, so the
// result does not depend on the order in which the reduction combines candidates.
//
// Launch: grid = 1, block = 1024 (ARGMAX_THREADS), no dynamic shared memory. One block reads the
// vector with coalesced loads (thread t takes t, t+1024, ...); for a 248320-entry vocabulary that is
// ~1 MB at single-SM bandwidth, a few tens of microseconds — against ~0.6 ms for the D2H it replaces.
//
// Build (CUDA 12.8, default flags, compute_75 like the rest of the tree):
//   nvcc -ptx -arch=compute_75 -o native/ptx/argmax_f32.ptx native/kernels/argmax_f32.cu

#include <stdint.h>
#include <math.h>

#define ARGMAX_THREADS 1024
#define AM_NEG_INF __int_as_float((int)0xff800000u)

// True when candidate (va, ia) must replace (vb, ib).
__device__ __forceinline__ bool am_better(float va, int ia, float vb, int ib)
{
    const bool na = isnan(va), nb = isnan(vb);
    if (na || nb)
        return na && (!nb || ia < ib);          // first NaN wins; a NaN beats any number
    if (va != vb)
        return va > vb;
    const bool sa = signbit(va) != 0, sb = signbit(vb) != 0;
    if (sa != sb)
        return !sa;                             // equal values of opposite sign: only +0 / -0
    return ia < ib;                             // exact tie: lowest index
}

extern "C" __global__ void __launch_bounds__(ARGMAX_THREADS) argmax_f32(
    const float* __restrict__ x,
    const int n,
    int* __restrict__ out)
{
    __shared__ float sv[ARGMAX_THREADS / 32];
    __shared__ int si[ARGMAX_THREADS / 32];

    // Sentinel: loses to every real element (-inf with an index past any real one).
    float bv = AM_NEG_INF;
    int bi = 0x7FFFFFFF;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = x[i];
        if (am_better(v, i, bv, bi)) { bv = v; bi = i; }
    }

    for (int o = 16; o > 0; o >>= 1)
    {
        const float ov = __shfl_down_sync(0xFFFFFFFF, bv, o);
        const int oi = __shfl_down_sync(0xFFFFFFFF, bi, o);
        if (am_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();

    if (wid == 0)
    {
        const int nw = blockDim.x >> 5;
        bv = lane < nw ? sv[lane] : AM_NEG_INF;
        bi = lane < nw ? si[lane] : 0x7FFFFFFF;
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_down_sync(0xFFFFFFFF, bv, o);
            const int oi = __shfl_down_sync(0xFFFFFFFF, bi, o);
            if (am_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out[0] = bi;
    }
}
