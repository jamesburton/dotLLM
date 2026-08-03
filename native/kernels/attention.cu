// Tiled attention kernel for dotLLM with online softmax.
// Q[seqQ, numHeads * headDim], K[seqKv, numKvHeads * headDim], V same as K.
// All FP16 data, FP32 accumulation for numerical stability.
// GQA: KV head broadcast via group_size = num_heads / num_kv_heads.
// Causal masking + optional sliding window.
// One block per (query_token, query_head) pair.
//
// Key optimizations over naive implementation:
// 1. Q loaded into shared memory once (not re-read per KV position)
// 2. Parallel warp-shuffle reductions (not serial thread-0 scan)
// 3. Tiled online softmax — bounded shared memory O(TILE_KV + headDim)
//    regardless of sequence length (no crash at long contexts)
//
// Two entry points share the body:
//   attention_f16        — scalar seq_kv / position_offset (eager / prefill).
//   attention_f16_dyn    — reads seq_kv / position_offset from device pointers.
//                          Used by CUDA-Graphs decode replay where launch params
//                          are baked into the graph at instantiate time but the
//                          KV length grows by 1 per replay. Zero extra sync —
//                          host bumps a 4-byte cuMemcpyHtoD before each launch.
//
// Issue #218: attention_f16_dyn was measurably slower per launch than
// attention_f16 at matched seq_kv (ncu: +25% duration, +25% warp cycles per
// issued instruction, CTA-barrier wait up ~3.4 cycles — see
// .perf-runs/ncu-2026-07-28/README.md). SASS inspection (ptxas -arch=sm_86,
// cuobjdump --dump-sass) showed the two seq_kv_ptr/position_offset_ptr LDGs were
// ALREADY scheduled as early as physically possible (the first two real
// instructions after the block-bounds early-exit, ~50-90 independent
// instructions before first use) — so "issue the read earlier" was not
// actionable, that part of the original hypothesis is refuted. What SASS did
// show: every one of the 256 threads (8 warps) independently executes its own
// copy of both LDGs, even though the address is block-uniform — 8x the
// redundant memory latency exposure of the scalar entry point, which instead
// reads from the (effectively free) constant/parameter bank. The fix below has
// only thread 0 dereference the two pointers once and broadcast the values via
// shared memory, timed to resolve at the body's PRE-EXISTING first
// __syncthreads() (after the Q-vector load + accumulator init) rather than
// introducing a new barrier — so the single load's latency overlaps with work
// every thread already has to do, instead of 8 independent per-warp stalls.
// Templated on a compile-time bool so attention_f16's SASS is provably
// unaffected (the dead branches fold away entirely for DeviceIndirect=false).

#include <cuda_fp16.h>
#include <float.h>

#define TILE_KV 256

// Body shared by both entry points. Inlined into each. seq_kv_ptr/position_offset_ptr
// are only read (by thread 0) when DeviceIndirect is true; pass nullptr otherwise.
template <bool DeviceIndirect>
__device__ __forceinline__ void attention_f16_body(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset,
    int sliding_window,
    const int* __restrict__ seq_kv_ptr,
    const int* __restrict__ position_offset_ptr);

extern "C" __global__ void __launch_bounds__(256) attention_f16(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)
{
    attention_f16_body<false>(q, k, v, output, seq_q, seq_kv, num_heads, num_kv_heads,
                       head_dim, position_offset, sliding_window, nullptr, nullptr);
}

// Graph-friendly entry point: seq_kv and position_offset are dereferenced from
// 4-byte device buffers. Host increments these via cuMemcpyHtoD between
// cuGraphLaunch calls (~1 µs vs 22 µs/launch on WDDM). The dereference itself
// happens inside attention_f16_body<true> (thread 0 only, broadcast via shared
// memory) — see the file header comment for why.
extern "C" __global__ void __launch_bounds__(256) attention_f16_dyn(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    const int seq_q,
    const int* __restrict__ seq_kv_ptr,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int* __restrict__ position_offset_ptr,
    const int sliding_window)
{
    attention_f16_body<true>(q, k, v, output, seq_q, /*seq_kv resolved post-barrier*/ 0,
                       num_heads, num_kv_heads, head_dim,
                       /*position_offset resolved post-barrier*/ 0, sliding_window,
                       seq_kv_ptr, position_offset_ptr);
}

template <bool DeviceIndirect>
__device__ __forceinline__ void attention_f16_body(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    int seq_q,
    int seq_kv,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position_offset,
    int sliding_window,
    const int* __restrict__ seq_kv_ptr,
    const int* __restrict__ position_offset_ptr)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    // Shared memory layout (fixed size, independent of seq_kv):
    //   q_shared[head_dim]     — Q vector cached for reuse
    //   score_tile[TILE_KV]    — attention scores for current tile
    //   out_accum[head_dim]    — weighted V accumulator
    //   warp_scratch[32]       — reduction workspace (only indices [0, nw) are ever
    //                            used by the reductions below, nw = ceil(256/32) = 8;
    //                            indices [30, 31] are permanently dead space, reused
    //                            below as the seq_kv/position_offset broadcast slot)
    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Issue #218: for the dyn entry point, only thread 0 dereferences seq_kv_ptr /
    // position_offset_ptr (one LDG each, not one per warp) and stashes them in the
    // dead tail of warp_scratch. This issues concurrently with every thread's Q-load
    // + accumulator-init work below and is consumed only after that work's
    // pre-existing __syncthreads() — no new barrier is introduced.
    int* bcast = (int*)(warp_scratch + 30);
    if (DeviceIndirect && threadIdx.x == 0)
    {
        bcast[0] = seq_kv_ptr[0];
        bcast[1] = position_offset_ptr[0];
    }

    // Step 1: Load Q vector into shared memory (FP16 → FP32)
    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);

    // Initialize output accumulator
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    if (DeviceIndirect)
    {
        seq_kv = bcast[0];
        position_offset = bcast[1];
    }

    // Absolute position for causal masking
    int pos_q = position_offset + tq;

    // Step 2: Process KV in tiles with online softmax
    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Compute Q·K scores for this tile
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            // Causal mask: only attend to positions <= current query position
            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            // Sliding window: skip if outside window
            if (sliding_window > 0 && pos_q - tkv >= sliding_window)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            const half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * __half2float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        __syncthreads();

        // 2b. Find tile max via parallel warp-shuffle reduction
        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));

        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        // 2c. Online softmax: rescale running accumulators
        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        // 2d. Compute attention weights: exp(score - max)
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        // Reduce tile_sum
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // 2e. Accumulate weighted V
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    const half* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * __half2float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Step 3: Final normalize and write output
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}

// Issue #200: direct block-table-read paged decode attention. Identical online-softmax
// math to attention_f16 (same tiled algorithm, same shared-memory layout) — the ONLY
// change is how a KV position's row address is resolved. Non-paged kernels compute a flat
// `k + tkv * kv_stride` offset into one contiguous buffer; this kernel instead resolves
// `tkv` to a (block, offset) pair and follows a device pointer out of a small per-layer
// array of block base addresses, exactly mirroring CudaKvBlockPool/CudaKvBlockTable's
// scattered block storage (see docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md §6). This lets a
// paged decode step skip PagedKvCache/CudaPagedKvCache's staging-buffer gather (no D2D copy
// of KV bytes before every decode step) — only a tiny host array of block base pointers
// needs to be refreshed (and only when a new block is allocated, roughly every
// `block_size` tokens), not the full KV content.
//
// Deliberately a full separate entry point rather than a templated variant of
// attention_f16_body (same choice already made for attention_pos_f16 above) — this is a new,
// unvalidated kernel (opt-in via DOTLLM_ATTN_PAGED_NATIVE=1, see CudaKernels.cs), and keeping
// it fully separate means it can never perturb the already-validated attention_f16/_dyn SASS.
//
// k_block_ptrs / v_block_ptrs: device arrays of `ceil(seq_kv / block_size)` device pointers,
// one per logical KV block for THIS sequence and THIS layer (CudaKvBlockPool.GetKeyPtr /
// GetValuePtr resolved host-side per block, uploaded via a tiny H2D copy — see
// CudaPagedKvCache.PrepareNativeBlockPtrs). Each block's storage is `block_size` rows of
// `kv_stride` half elements, matching CudaKvBlockPool's per-block layout exactly.
extern "C" __global__ void __launch_bounds__(256) attention_f16_paged(
    const half* __restrict__ q,
    const half* const* __restrict__ k_block_ptrs,
    const half* const* __restrict__ v_block_ptrs,
    half* __restrict__ output,
    const int seq_q,
    const int seq_kv,
    const int block_size,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int position_offset,
    const int sliding_window)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    extern __shared__ float smem3[];
    float* q_shared    = smem3;
    float* score_tile  = smem3 + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    int pos_q = position_offset + tq;

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        // 2a. Compute Q·K scores for this tile. Block-table indirection: resolve tkv ->
        // (logical block, offset-in-block) -> device pointer -> row address. This is the
        // ONLY change versus attention_f16_body's flat `k + tkv * kv_stride` addressing.
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;

            if (tkv > pos_q)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }
            if (sliding_window > 0 && pos_q - tkv >= sliding_window)
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            int logical_block = tkv / block_size;
            int offset_in_block = tkv % block_size;
            const half* k_block = k_block_ptrs[logical_block];
            const half* k_vec = k_block + (size_t)offset_in_block * kv_stride + hkv * head_dim;

            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * __half2float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        __syncthreads();

        // 2b. Find tile max via parallel warp-shuffle reduction
        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));

        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        // 2c. Online softmax: rescale running accumulators
        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        // 2d. Compute attention weights: exp(score - max)
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // 2e. Accumulate weighted V — same block-table indirection as the K read above.
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    int logical_block = tkv / block_size;
                    int offset_in_block = tkv % block_size;
                    const half* v_block = v_block_ptrs[logical_block];
                    const half* v_vec = v_block + (size_t)offset_in_block * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * __half2float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}

extern "C" __global__ void __launch_bounds__(256) attention_pos_f16(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    const int* __restrict__ positions,
    const int seq_q,
    const int seq_kv,
    const int num_heads,
    const int num_kv_heads,
    const int head_dim,
    const int sliding_window)
{
    int block_id = blockIdx.x;
    int total_blocks = seq_q * num_heads;
    if (block_id >= total_blocks) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;

    int group_size = num_heads / num_kv_heads;
    int hkv = hq / group_size;

    float scale = rsqrtf((float)head_dim);

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;
    int pos_q = positions[tq];

    extern __shared__ float smem2[];
    float* q_shared    = smem2;
    float* score_tile  = smem2 + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    const half* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = __half2float(q_vec[d]);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = 0; t_start < seq_kv; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > seq_kv) t_end = seq_kv;
        int tile_len = t_end - t_start;

        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv > sliding_window))
            {
                score_tile[t] = -FLT_MAX;
                continue;
            }

            const half* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * __half2float(k_vec[d]);

            score_tile[t] = score * scale;
        }
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));

        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? expf(running_max - new_max) : 0.0f;

        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;

        running_max = new_max;
        __syncthreads();

        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? expf(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();

        if (warp_id == 0)
        {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                if (score_tile[t] > 0.0f)
                {
                    int tkv = t_start + t;
                    const half* v_vec = v + (size_t)tkv * kv_stride + hkv * head_dim;
                    v_acc += score_tile[t] * __half2float(v_vec[d]);
                }
            }
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;

    half* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = __float2half(out_accum[d] * sum_inv);
}
