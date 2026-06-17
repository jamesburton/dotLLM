// Causal softmax over column-major [s x s] FP16 attention scores, in-place.
//
// Companion kernel for the cuBLAS tensor-core prefill-attention path (G3):
//   QK^T  -> cublasGemmStridedBatchedEx -> scores (col-major [s x s], ldc=s, one plane per query head)
//   THIS  -> causal softmax over scores
//   P * V -> cublasGemmStridedBatchedEx -> out
//
// Score layout: per query head hq, plane base = hq * s * s. Element (tq, tk) lives
// at plane_base + tq + tk * s  (the QK GEMM has m=s so tq is the contiguous row
// axis with leading dim ldc=s). A softmax row is a fixed query tq over varying key
// tk, so it is STRIDED BY s in memory. The reduction here walks that strided axis.
//
// Causal mask (pure prefill, position_offset 0): query tq attends only to keys
// tk <= tq. Two things this kernel MUST get right because the downstream P*V GEMM
// sums over the FULL key axis tk = 0..s-1:
//   1. Entries with tk > tq are written as exactly 0 (not left untouched / not
//      -inf), so they contribute nothing to P*V.
//   2. The running max is taken over tk in [0, tq] only — a stray large
//      non-causal score would otherwise shift the max and underflow the row.
//
// The QK GEMM already applied the 1/sqrt(head_dim) scale via its alpha, so the
// scale is NOT re-applied here.
//
// FP32 internal accumulation (max-subtract + exp + normalize), matching the
// online-softmax math in attention.cu. One block per (query head, query row).
// expf is used, so this file is compiled precise (kept out of build_ptx.bat's
// FAST_MATH list, like softmax.cu).

#include <cuda_fp16.h>
#include <float.h>

extern "C" __global__ void __launch_bounds__(256) attention_softmax_causal_f16(
    half* __restrict__ scores,
    const int s,
    const int num_heads)
{
    int block_id = blockIdx.x;
    int total_blocks = num_heads * s;
    if (block_id >= total_blocks) return;

    int hq = block_id / s;
    int tq = block_id % s;

    // Plane for this query head; row (fixed tq) is strided by s along the key axis.
    half* plane = scores + (size_t)hq * s * s;
    half* row = plane + tq;  // element (tq, tk) at row[tk * s]

    int causal_len = tq + 1;  // keys 0..tq are valid

    // Pass 1: max over the causal prefix only.
    float max_val = -FLT_MAX;
    for (int tk = threadIdx.x; tk < causal_len; tk += blockDim.x)
    {
        float v = __half2float(row[(size_t)tk * s]);
        if (v > max_val) max_val = v;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
        if (other > max_val) max_val = other;
    }

    __shared__ float warp_vals[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_vals[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        max_val = (lane < num_warps) ? warp_vals[lane] : -FLT_MAX;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            float other = __shfl_down_sync(0xFFFFFFFF, max_val, offset);
            if (other > max_val) max_val = other;
        }
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Pass 2: exp(x - max) over the causal prefix, store back, accumulate sum.
    float sum_exp = 0.0f;
    for (int tk = threadIdx.x; tk < causal_len; tk += blockDim.x)
    {
        float e = expf(__half2float(row[(size_t)tk * s]) - max_val);
        sum_exp += e;
        row[(size_t)tk * s] = __float2half(e);
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);

    if (lane == 0) warp_vals[warp_id] = sum_exp;
    __syncthreads();

    if (warp_id == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        sum_exp = (lane < num_warps) ? warp_vals[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
    }

    __shared__ float shared_sum_inv;
    if (threadIdx.x == 0) shared_sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    __syncthreads();
    float sum_inv = shared_sum_inv;

    // Pass 3a: normalize the causal prefix in place.
    for (int tk = threadIdx.x; tk < causal_len; tk += blockDim.x)
        row[(size_t)tk * s] = __float2half(__half2float(row[(size_t)tk * s]) * sum_inv);

    // Pass 3b: zero the masked (non-causal) tail so P*V's full-key sum ignores it.
    for (int tk = causal_len + threadIdx.x; tk < s; tk += blockDim.x)
        row[(size_t)tk * s] = __float2half(0.0f);
}

// Coalesced variant: ONE THREAD per softmax row (per query head, per query token).
//
// The one-block-per-row variant above reads row[tk*s] with consecutive threads in a
// block striding by s along the key axis — fully uncoalesced (each 32-byte transaction
// delivers 2 useful bytes), which caps the cuBLAS+softmax path well below the GEMM-only
// floor. Here thread t owns global row = blockIdx.x*blockDim.x + threadIdx.x, decoded as
// hq = row / s, tq = row % s. Consecutive threads own consecutive tq, whose row starts
// (plane_base + tq) are CONSECUTIVE addresses, so each strided read across a warp lands
// in one cache line — coalesced. No shuffles, no shared memory; the whole row reduction
// is serial within the owning thread (s up to a few thousand is fine — memory-bound).
extern "C" __global__ void __launch_bounds__(256) attention_softmax_causal_coalesced_f16(
    half* __restrict__ scores,
    const int s,
    const int num_heads)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= num_heads * s) return;

    int hq = row_id / s;
    int tq = row_id % s;

    half* row = scores + (size_t)hq * s * s + tq;  // element (tq, tk) at row[tk * s]
    int causal_len = tq + 1;

    // Pass 1: max over the causal prefix.
    float max_val = -FLT_MAX;
    for (int tk = 0; tk < causal_len; tk++)
    {
        float v = __half2float(row[(size_t)tk * s]);
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp(x - max), store back, accumulate sum.
    float sum_exp = 0.0f;
    for (int tk = 0; tk < causal_len; tk++)
    {
        float e = expf(__half2float(row[(size_t)tk * s]) - max_val);
        sum_exp += e;
        row[(size_t)tk * s] = __float2half(e);
    }

    float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // Pass 3a: normalize the causal prefix.
    for (int tk = 0; tk < causal_len; tk++)
        row[(size_t)tk * s] = __float2half(__half2float(row[(size_t)tk * s]) * sum_inv);

    // Pass 3b: zero the masked tail so P*V's full-key sum ignores it.
    for (int tk = causal_len; tk < s; tk++)
        row[(size_t)tk * s] = __float2half(0.0f);
}

// FP32-scores variant of the coalesced kernel (one thread per softmax row).
//
// Reads the QK scores in FP32 (the QK GEMM keeps COMPUTE_32F and writes CUDA_R_32F),
// runs the same causal max/exp/normalize math in FP32, and writes the normalized probs
// to a SEPARATE FP16 buffer for the P*V GEMM to consume. This removes the dominant
// numeric error of the all-FP16 path: rounding the wide-range pre-softmax scores to FP16
// before exp() (which amplifies the rounding). Post-softmax probs live in [0,1] where
// FP16 rel error is ~5e-4, so writing the output in FP16 keeps PV on tensor cores at
// negligible cost. Layout matches the FP16 variant: scores/probs are per-head col-major
// [s x s], element (tq, tk) at plane_base + tq + tk*s.
extern "C" __global__ void __launch_bounds__(256) attention_softmax_causal_coalesced_f32in_f16out(
    const float* __restrict__ scores,
    half* __restrict__ probs,
    const int s,
    const int num_heads)
{
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= num_heads * s) return;

    int hq = row_id / s;
    int tq = row_id % s;

    const float* srow = scores + (size_t)hq * s * s + tq;  // element (tq, tk) at srow[tk * s]
    half* prow = probs + (size_t)hq * s * s + tq;
    int causal_len = tq + 1;

    // Pass 1: max over the causal prefix (FP32 scores).
    float max_val = -FLT_MAX;
    for (int tk = 0; tk < causal_len; tk++)
    {
        float v = srow[(size_t)tk * s];
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp(x - max), accumulate sum (kept in FP32; not stored yet).
    float sum_exp = 0.0f;
    for (int tk = 0; tk < causal_len; tk++)
        sum_exp += expf(srow[(size_t)tk * s] - max_val);

    float sum_inv = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    // Pass 3a: write normalized probs to FP16 (recompute exp from FP32 scores so the
    // only FP16 rounding is on the final [0,1] probability).
    for (int tk = 0; tk < causal_len; tk++)
        prow[(size_t)tk * s] = __float2half(expf(srow[(size_t)tk * s] - max_val) * sum_inv);

    // Pass 3b: zero the masked tail so P*V's full-key sum ignores it.
    for (int tk = causal_len; tk < s; tk++)
        prow[(size_t)tk * s] = __float2half(0.0f);
}
