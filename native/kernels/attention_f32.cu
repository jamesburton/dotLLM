// Tiled attention with FP32 Q/K/V/output and online softmax.
//
// Softmax uses Schraudolph's IEEE-754 bit-trick approximation of expf, matching the
// CPU oracle's DotLLM.Cpu.Kernels.FastMath.ExpSumAndStore. The CPU side has used the
// fast-exp path since the kernel's inception; switching CUDA to precise expf made the
// two backends disagree by ~1% (5e-3 abs on attention output) on synthetic-fixture
// parity. The bit-trick keeps both backends bit-near-equivalent without a CPU-side
// accuracy regression. Constants C0/C1 must stay in sync with FastMath.cs.
//
//   exp(x) ≈ bitcast_int_to_float((int)(x * C0 + C1)),   x ≤ 0 only (no overflow guard)
//
// C0 = 2^23 / ln(2), C1 = (127 - 0.0579) * 2^23. Applied only to softmax `expf` calls
// where the argument is always ≤ 0 by construction (max-subtracted scores).

#include <float.h>
#include <math.h>
#include <cooperative_groups.h>

#define TILE_KV 256

// Schraudolph fast-exp constants (mirror FastMath.cs).
#define FASTEXP_C0 12102203.0f
#define FASTEXP_C1 1064866805.0f
#define FASTEXP_MIN_CLAMP -87.3f

__device__ __forceinline__ float fast_exp_neg(float x)
{
    // Caller contract: x ≤ 0 (max-subtracted softmax scores). Clamp the lower bound
    // to keep the integer cast inside the IEEE-754 normal range. Use float-to-int
    // truncation (toward zero) to match the C# scalar `(int)x` and the SIMD
    // ConvertToVector*Int32WithTruncation paths in FastMath.cs — round-to-nearest
    // would introduce a sub-ULP bias.
    x = fmaxf(x, FASTEXP_MIN_CLAMP);
    int bits = __float2int_rz(fmaf(x, FASTEXP_C0, FASTEXP_C1));
    return __int_as_float(bits);
}

extern "C" __global__ void __launch_bounds__(256) attention_f32(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_q, const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window)
{
    int block_id = blockIdx.x;
    if (block_id >= seq_q * num_heads) return;

    int tq = block_id / num_heads;
    int hq = block_id % num_heads;
    int hkv = hq / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);
    int pos_q = position_offset + tq;

    int q_stride = num_heads * head_dim;
    int kv_stride = num_kv_heads * head_dim;

    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Load Q into shared memory
    const float* q_vec = q + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = q_vec[d];

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

        // Compute scores
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            { score_tile[t] = -FLT_MAX; continue; }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * k_vec[d];
            score_tile[t] = score * scale;
        }
        __syncthreads();

        // Tile max reduction
        float tile_max = -FLT_MAX;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
            tile_max = fmaxf(tile_max, score_tile[t]);

        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        if (lane == 0) warp_scratch[warp_id] = tile_max;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
        }
        if (threadIdx.x == 0) warp_scratch[0] = tile_max;
        __syncthreads();
        tile_max = warp_scratch[0];

        // Online softmax rescale
        float new_max = fmaxf(running_max, tile_max);
        float correction = (running_max > -FLT_MAX + 1.0f)
                           ? fast_exp_neg(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        // Attention weights
        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? fast_exp_neg(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        // Accumulate weighted V
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
                if (score_tile[t] > 0.0f)
                    v_acc += score_tile[t] * (v + (size_t)(t_start + t) * kv_stride + hkv * head_dim)[d];
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Normalize and write
    float sum_inv = (running_sum > 1e-10f) ? (1.0f / running_sum) : 0.0f;
    float* out_vec = output + (size_t)tq * q_stride + hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_vec[d] = out_accum[d] * sum_inv;
}

// ─── OPT-IN split-KV ("Flash-Decoding") cooperative-groups variant (issue #183) ────────────
//
// `attention_f32`'s grid is `seq_q * num_heads` only. For decode (seq_q==1) that means grid ==
// num_heads — for real Bonsai-27B (qwen35.attention.head_count=24) that's SMALLER than the RTX
// 3060's 28 SMs, so several SMs get zero work, and the KV-tile loop above
// (`for t_start = 0; t_start < seq_kv; t_start += TILE_KV`) runs entirely SEQUENTIALLY within one
// block per head with zero parallelism across the KV dimension. `.docs/handoff.md`'s "Depth-
// dependent attention finding" measured this kernel growing +151% (0.043ms -> 0.108ms/call) from
// context depth 0 to 256 — this is exactly the published "Flash-Decoding" scenario (Dao et al.,
// the FlashAttention team): a single (or few) query tokens attending to many cached KV positions,
// where the query/head-only grid underfills the GPU. The fix: split the KV dimension across
// ADDITIONAL blocks, each computing a PARTIAL (running_max, running_sum, weighted_output) over its
// own KV sub-range using the exact same online-softmax tiling as `attention_f32` above, then merge
// partials with the standard rule (equivalent to what `attention_f32`'s own per-tile rescale above
// already does, one step at a time, generalized to N-way combine):
//   given partials {(m_i, l_i, o_i)} for i in [0, SPLIT):
//     m = max_i(m_i)
//     l = sum_i( l_i * exp(m_i - m) )
//     o = sum_i( o_i * exp(m_i - m) )      (component-wise, o_i is UNNORMALIZED weighted-V sum)
//     final = o / l
//
// ─── Design choice: single cooperative launch, not two kernel launches ─────────────────────
// Modeled both before implementing (per this investigation's standing "model before implementing"
// rule — see pq2_0_gemv.cu's header for ~10 precedents where skipping this step cost real
// regressions). This kernel is called 16x/layer/step for Bonsai-27B (16 full-attention layers,
// `qwen35.full_attention_interval=4` of 64 total layers) — much less frequently than
// gdn_scan_step_f32's 48x/layer/step (issue #180). A naive two-launch design (partial-compute
// kernel + separate combine kernel, no grid.sync needed) would add 1 extra launch x 16 layers =
// 16 extra launches/decode token. Per-launch overhead on this WDDM host has been measured/quoted
// elsewhere in this investigation at roughly 5-22us (the higher end from `CudaDecodeGraph.cs`'s
// CUDA-graph-capture motivation) — so 16 extra launches cost an estimated ~80-350us/decode token.
// At the ~17-18 tok/s baseline (~55-59ms/decode token), that overhead alone is small in absolute
// terms (~0.15-0.6%) but is NOT small relative to what this specific kernel can plausibly recover:
// `attn-6c-core` is itself only ~3-10% of total decode time even at depth 256-1024 (16 calls x
// 0.1-0.4ms/call ~= 1.6-6.4ms of a ~55-59ms token), so a second launch's overhead could eat a
// double-digit percentage of whatever the split saves — unlike GDN's 48-launch case where the
// *kernel itself* is a much larger share of decode time, making 2-launch overhead comparatively
// cheaper there too, but still rejected. Given the two designs' overhead is comparable in kind
// here, and the #180/#181 cooperative-launch precedent is proven safe and practical on this exact
// GPU (reuses `cuLaunchCooperativeKernel` / `cuOccupancyMaxActiveBlocksPerMultiprocessor`, already
// wired in `CudaDriverApi.cs`), single-launch cooperative is the safer choice to preserve real
// end-to-end gain: it adds ZERO launches (same call count as `attention_f32`), trading that for one
// `grid.sync()` instead — and GDN's own split4 data showed grid.sync overhead is small in absolute
// terms (split=1 ~65-68us -> split=4 ~48-49us net win, despite 2 grid.sync calls in that kernel;
// this one needs only ONE).
//
// ─── Why ONE grid.sync() suffices here (simpler than GDN's two) ───────────────────────────
// GDN's coop kernel has 3 sequential phases (decay+retrieve -> combine -> write+read) needing two
// rendezvous points. Here, EVERY block (including the ones that will later combine) performs the
// identical partial-compute phase first; only AFTER all partials are written do the "leader" blocks
// (kv_split index 0 for each head) read every split's partial and produce the final normalized
// output. That is exactly one write-then-read dependency across the whole grid -> one grid.sync().
//
// ─── Fixed SPLIT=4, gated by a minimum seq_kv threshold at the C# call site ────────────────
// Unlike GDN's split (which divides a fixed d_state=128), this split divides the KV sequence
// range, which varies every decode step as context grows — there is no fixed-size constraint
// forcing a particular SPLIT. A fixed SPLIT=4 (mirroring #180 exactly) keeps the design and its
// occupancy/safety-check story simple: the safety check only needs to verify `num_heads*4` blocks
// can be co-resident (computed once per model shape, cached). The C# call site is responsible for
// only invoking this kernel once `seq_kv` is large enough that splitting is worth the grid.sync +
// combine overhead (small seq_kv means each split only saves a few iterations of the per-tile
// weighted-V accumulation loop — not worth 4x the blocks + a grid-wide barrier) — see
// `CudaKernels.AttentionSplitKvMinSeqKv` / the `ForwardFullAttnBody` call site.
//
// ─── Precision: reuses fast_exp_neg exactly, no precise expf anywhere in the merge ─────────
// The combine step's `exp(m_i - m)` always has a non-positive argument (m = max over splits), so
// it satisfies `fast_exp_neg`'s caller contract exactly like every other softmax exponential in
// this file. Introducing precise `expf` here would reintroduce the ~1% CPU/GPU divergence this
// file's header already documents fixing — independent of whatever NEW tolerance the cross-block
// reassociation itself requires (see below).
//
// ─── Correctness: reassociation, NOT the same "recurrent state" story as GDN's split4 ──────
// Splitting KV necessarily reassociates the float accumulation (independent partial sums combined
// instead of one long sequential accumulation) — mathematically equal, not bit-identical, same as
// GDN's split4. BUT: unlike GDN's state matrix (which IS the model's recurrent memory, carried
// step-to-step and hence compounding drift), this kernel's inputs each decode step are the
// (exact, unperturbed) KV cache plus this step's query — the KV cache holds each layer's ORIGINAL
// K/V vectors, computed from the pre-attention hidden state, so a numerical perturbation in THIS
// layer's attention OUTPUT does not feed back into what gets cached for this same layer at this
// position. The only cross-step pathway is discrete: a different logit distribution could change
// which token gets sampled, branching the whole generation — not a continuous numerical drift
// accumulating in a persistent FP32 state the way GDN's does. See
// `CudaAttentionF32SplitKvTests.cs` for the empirical many-step characterization confirming
// (or refuting) this expectation — this is flagged as a genuinely different, worth-checking-for-
// real property, not assumed. CONFIRMED (not just theorized): a 300-consecutive-decode-step run
// (seqKv growing 256->555, one new KV row appended per step, matching real generation) found max
// abs diff vs the CPU oracle does NOT compound — first-10-steps average 4.6e-3 vs last-3-steps
// average 3.1e-3 (ratio 0.68x, i.e. slightly DECREASING, not growing) — a genuinely different,
// more reassuring result than GDN split4's compounding (which reached ~1-2% within 6 steps).
//
// ─── Real bench result: INCONCLUSIVE at the depths safely measurable on this host ──────────
// `dotnet run bench` A/B (real Bonsai-27B, RTX 3060, `-p 8 -n 48`, multiple rounds):
//   depth=256  (seqKv~256-264): baseline median 17.59 / best 17.71 tok/s vs split-KV median 17.38 /
//              best 17.71 tok/s — split-KV very slightly WORSE on median, tied on best. Each split
//              only gets ~64 KV rows here (256/ATTN_KV_SPLIT) — well under one TILE_KV tile — so
//              the extra grid.sync()+combine overhead has little sequential work left to amortize
//              against, the same "small win canceled by overhead" shape as GDN's split=2 (#180).
//   depth=512  (seqKv~512-520), 2 rounds: baseline medians 16.73/16.85 (mean 16.79), best 17.27/
//              17.30 (mean 17.29); split-KV medians 16.90/16.73 (mean 16.82), best 17.36/17.44
//              (mean 17.40) — median is a coin flip (+0.2% aggregate, inside this host's own
//              documented 2-8% run-to-run noise floor), but split-KV's BEST-of-8-reps edged
//              baseline's best in every round measured (+0.6% to +0.8%) — a weak, GDN-#180-shaped
//              signal ("never lost by the best-of-rep metric") but far short of a clear win.
//   depth>=768: COULD NOT BE SAFELY MEASURED on this host — both the unmodified baseline path
//              (confirmed independently, `DOTLLM_ATTN_SPLIT_KV` unset) and the split-KV path hung
//              for 3-4+ minutes at depth 768 and depth 1024 (100% GPU util, ~15-62W power draw,
//              VRAM near this card's 12GB ceiling — the same signature `.docs/handoff.md` already
//              flagged for `DOTLLM_HYBRID_PROFILE=1`+`--depth 1024`, but reproduced here WITHOUT
//              profiling and on the plain baseline path too, generalizing that prior finding to a
//              broader depth>=768 instability, not something this change introduced or is specific
//              to). This is exactly the depth range where the motivating profiling data
//              (attn-6c-core growing +151% from depth 0->256, presumably continuing to grow past
//              256) predicts the split-KV design should show its clearest win — and it is the one
//              range this session could not validate. Do not read the depth 256/512 numbers above
//              as the final verdict on this kernel's value; they are what could be safely measured,
//              not necessarily where the effect is largest.
//
// VERDICT: correctness-validated, zero-risk (opt-in, default-OFF, proper safety-gated fallback,
// zero change to any default code path), but NOT a demonstrated performance win at the depths this
// session could safely test — an honest "partial/incomplete given the scope" outcome per this
// investigation's stated culture, not a clean win (unlike #168/#170/#180/#181) or a clean negative
// (unlike #172/#173, which showed no benefit at ANY tested condition). Kept as opt-in infrastructure
// specifically because (a) it is real, tested, zero-default-risk engineering that would be wasted
// effort to discard, (b) the "best-of-rep never lost" signal at depth 512 hints at a real, if small,
// effect, and (c) a future session with either more VRAM headroom (to safely reach depth>=768-1024)
// or a fix for the underlying depth>=768 hang could re-run this exact A/B and get a decisive answer
// without redoing any of the design/correctness work. Do NOT flip `EnableAttentionSplitKv`'s default
// without that follow-up validation.
#define ATTN_KV_SPLIT 4

extern "C" __global__ void attention_f32_split_kv(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window,
    float* __restrict__ partial_max,   // [num_heads, ATTN_KV_SPLIT]
    float* __restrict__ partial_sum,   // [num_heads, ATTN_KV_SPLIT]
    float* __restrict__ partial_out)   // [num_heads, ATTN_KV_SPLIT, head_dim]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int hq = blockIdx.x;
    int s  = blockIdx.y;
    int hkv = hq / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);
    int pos_q = position_offset; // seq_q == 1 (decode-only kernel)

    int kv_stride = num_kv_heads * head_dim; // q has no stride use here: seqQ==1, so q_vec below indexes by head only

    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // This split's contiguous KV sub-range: [kv_lo, kv_hi).
    int chunk = (seq_kv + ATTN_KV_SPLIT - 1) / ATTN_KV_SPLIT;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    const float* q_vec = q + (size_t)hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = q_vec[d];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = kv_lo; t_start < kv_hi; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > kv_hi) t_end = kv_hi;
        int tile_len = t_end - t_start;

        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            { score_tile[t] = -FLT_MAX; continue; }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * k_vec[d];
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
        if (warp_id == 0) {
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
                           ? fast_exp_neg(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? fast_exp_neg(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
                if (score_tile[t] > 0.0f)
                    v_acc += score_tile[t] * (v + (size_t)(t_start + t) * kv_stride + hkv * head_dim)[d];
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Publish this split's partial (UNNORMALIZED — no division by running_sum here).
    if (threadIdx.x == 0)
    {
        partial_max[(size_t)hq * ATTN_KV_SPLIT + s] = running_max;
        partial_sum[(size_t)hq * ATTN_KV_SPLIT + s] = running_sum;
    }
    float* partial_out_vec = partial_out + ((size_t)hq * ATTN_KV_SPLIT + s) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        partial_out_vec[d] = out_accum[d];

    grid.sync();   // ALL blocks (every head, every split) must have published before any combine.

    if (s != 0) return; // only the split=0 block per head performs the final combine.

    __shared__ float s_combined_max;
    __shared__ float s_combined_sum;
    if (threadIdx.x == 0)
    {
        float m = -FLT_MAX;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
            m = fmaxf(m, partial_max[(size_t)hq * ATTN_KV_SPLIT + i]);

        float l = 0.0f;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
        {
            float mi = partial_max[(size_t)hq * ATTN_KV_SPLIT + i];
            float li = partial_sum[(size_t)hq * ATTN_KV_SPLIT + i];
            float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
            l += li * w;
        }
        s_combined_max = m;
        s_combined_sum = l;
    }
    __syncthreads();

    float m = s_combined_max;
    float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float o = 0.0f;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
        {
            float mi = partial_max[(size_t)hq * ATTN_KV_SPLIT + i];
            float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
            float oi = partial_out[((size_t)hq * ATTN_KV_SPLIT + i) * head_dim + d];
            o += oi * w;
        }
        output[(size_t)hq * head_dim + d] = o * sum_inv;
    }
}

// ─── Issue #226 spike: fp64 cross-split COMBINE only, fast_exp_neg untouched ───────────────────────
//
// #222 found attention_f32_split_kv's real-generation divergence from baseline (#183's known,
// accepted "not bit-exact" reassociation tradeoff, quantified at generation scale): a genuine
// argmax flip at decode depth 257 (the first step split-KV can engage), fully compounding from
// there (774/775 subsequent tokens differ), plus a +0.30% post-gate perplexity regression.
// attention_f32.cu's own header (see above) already documents the root cause as the COMBINE step's
// cross-split reassociation (independent partial sums merged, not one sequential accumulation) --
// NOT the per-tile fast_exp_neg approximation, which is identical, in the identical accumulation
// order, in both the baseline attention_f32 and every split-KV variant.
//
// This is a byte-for-byte copy of attention_f32_split_kv EXCEPT the combine block (after
// grid.sync(), guarded by `if (s != 0) return`) accumulates the cross-split partial_sum ("l") and
// partial_out ("o") merges in double precision instead of float -- fast_exp_neg itself still
// computes and returns a float (untouched, per issue #226's explicit scope), only the SUMMATION of
// its already-computed float outputs across the (up to) ATTN_KV_SPLIT=4 terms happens in double.
// This isolates the one specific hypothesis #226 asks about: does the reassociation error in this
// specific 4-term combine sum (not the exp approximation, not the per-tile accumulation within a
// split) meaningfully explain the #222 divergence.
//
// Kept as a SEPARATE kernel (not a modification of attention_f32_split_kv in place) specifically so
// both can be A/B'd directly without a rebuild-swap dance -- opt-in via DOTLLM_ATTN_SPLIT_KV_HP=1,
// mutually exclusive with (and takes priority over, when both would apply) plain split-KV. See
// issue #226 for the correctness/precision/perf verdict once measured.
extern "C" __global__ void attention_f32_split_kv_hp(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window,
    float* __restrict__ partial_max,   // [num_heads, ATTN_KV_SPLIT]
    float* __restrict__ partial_sum,   // [num_heads, ATTN_KV_SPLIT]
    float* __restrict__ partial_out)   // [num_heads, ATTN_KV_SPLIT, head_dim]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int hq = blockIdx.x;
    int s  = blockIdx.y;
    int hkv = hq / (num_heads / num_kv_heads);
    float scale = rsqrtf((float)head_dim);
    int pos_q = position_offset; // seq_q == 1 (decode-only kernel)

    int kv_stride = num_kv_heads * head_dim; // q has no stride use here: seqQ==1, so q_vec below indexes by head only

    extern __shared__ float smem[];
    float* q_shared    = smem;
    float* score_tile  = smem + head_dim;
    float* out_accum   = score_tile + TILE_KV;
    float* warp_scratch = out_accum + head_dim;

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // This split's contiguous KV sub-range: [kv_lo, kv_hi).
    int chunk = (seq_kv + ATTN_KV_SPLIT - 1) / ATTN_KV_SPLIT;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    const float* q_vec = q + (size_t)hq * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        q_shared[d] = q_vec[d];
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        out_accum[d] = 0.0f;
    __syncthreads();

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    for (int t_start = kv_lo; t_start < kv_hi; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > kv_hi) t_end = kv_hi;
        int tile_len = t_end - t_start;

        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            if (tkv > pos_q || (sliding_window > 0 && pos_q - tkv >= sliding_window))
            { score_tile[t] = -FLT_MAX; continue; }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score += q_shared[d] * k_vec[d];
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
        if (warp_id == 0) {
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
                           ? fast_exp_neg(running_max - new_max) : 0.0f;
        running_sum *= correction;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[d] *= correction;
        running_max = new_max;
        __syncthreads();

        float tile_sum = 0.0f;
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
            float w = (score_tile[t] > -FLT_MAX + 1.0f)
                      ? fast_exp_neg(score_tile[t] - running_max) : 0.0f;
            score_tile[t] = w;
            tile_sum += w;
        }
        for (int off = warpSize / 2; off > 0; off >>= 1)
            tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
        if (lane == 0) warp_scratch[warp_id] = tile_sum;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + warpSize - 1) / warpSize;
            tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[0] = tile_sum;
        }
        __syncthreads();
        running_sum += warp_scratch[0];

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_acc = 0.0f;
            for (int t = 0; t < tile_len; t++)
                if (score_tile[t] > 0.0f)
                    v_acc += score_tile[t] * (v + (size_t)(t_start + t) * kv_stride + hkv * head_dim)[d];
            out_accum[d] += v_acc;
        }
        __syncthreads();
    }

    // Publish this split's partial (UNNORMALIZED — no division by running_sum here).
    if (threadIdx.x == 0)
    {
        partial_max[(size_t)hq * ATTN_KV_SPLIT + s] = running_max;
        partial_sum[(size_t)hq * ATTN_KV_SPLIT + s] = running_sum;
    }
    float* partial_out_vec = partial_out + ((size_t)hq * ATTN_KV_SPLIT + s) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        partial_out_vec[d] = out_accum[d];

    grid.sync();   // ALL blocks (every head, every split) must have published before any combine.

    if (s != 0) return; // only the split=0 block per head performs the final combine.

    // ── fp64 combine (issue #226): only this block differs from attention_f32_split_kv ──
    __shared__ float s_combined_max;
    __shared__ double s_combined_sum;
    if (threadIdx.x == 0)
    {
        float m = -FLT_MAX;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
            m = fmaxf(m, partial_max[(size_t)hq * ATTN_KV_SPLIT + i]);

        double l = 0.0;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
        {
            float mi = partial_max[(size_t)hq * ATTN_KV_SPLIT + i];
            float li = partial_sum[(size_t)hq * ATTN_KV_SPLIT + i];
            float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
            l += (double)li * (double)w;
        }
        s_combined_max = m;
        s_combined_sum = l;
    }
    __syncthreads();

    float m = s_combined_max;
    double sum_d = s_combined_sum;
    double sum_inv = (sum_d > 1e-10) ? (1.0 / sum_d) : 0.0;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        double o = 0.0;
        for (int i = 0; i < ATTN_KV_SPLIT; i++)
        {
            float mi = partial_max[(size_t)hq * ATTN_KV_SPLIT + i];
            float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
            float oi = partial_out[((size_t)hq * ATTN_KV_SPLIT + i) * head_dim + d];
            o += (double)oi * (double)w;
        }
        output[(size_t)hq * head_dim + d] = (float)(o * sum_inv);
    }
}

// ─── OPT-IN GQA-group + split-KV composed kernel (issues #197 + #198) ──────────────────────────────
//
// `attention_f32_split_kv` above (issue #183) grids one block per QUERY head (grid.x=numHeads).
// For real GQA models (Bonsai-27B: numHeads=24, numKvHeads=4, group=numHeads/numKvHeads=6) the
// `group` query heads sharing one KV head each independently re-read the SAME K/V rows from
// global memory every decode step -- issue #198's ncu-driven finding. The naive fix ("grid =
// numKvHeads instead of numHeads") was checked and rejected as a standalone change: numKvHeads=4
// is SMALLER than the already-underfilling numHeads=24 grid (only 4 of 28 SMs would get any work
// at all). The two issues' brainstorms concluded GQA-group batching and split-KV MUST compose:
// grid = (numKvHeads, kv_split) -- this kernel.
//
// Each block owns one KV head (`hkv = blockIdx.x`) and one KV sub-range (`s = blockIdx.y`), and
// internally register-blocks the QK/PV loops across the `group = numHeads/numKvHeads` query
// heads sharing that KV head: every K/V element is read from global memory exactly ONCE per
// (hkv, s, tile position) and reused `group` times from registers (`scores[MAX_GQA_GROUP]`,
// `v_acc[MAX_GQA_GROUP]` -- a compile-time cap on a RUNTIME-loop-bounded register array, needed
// only because CUDA C local arrays must have a compile-time size; MAX_GQA_GROUP=8 covers every
// GQA ratio this project currently targets, including Bonsai-27B's 6). Shared memory
// (`q_shared`/`out_accum`/`score_tile`/`running_max_s`/`running_sum_s`, all sized by the RUNTIME
// `group` via the C# launcher, NOT MAX_GQA_GROUP) is fully dynamic -- unlike
// `attention_flash_mma.cu`'s prefill kernel, this does NOT stage K/V tiles into shared memory (a
// full [256]x[256] K or V tile at Bonsai's headDim=256 would be 256KB, many times over Ampere's
// shared-memory budget); only Q/out/scores/softmax-state are shared, K/V stay register-blocked
// via repeated global reads amortized across the group, matching #198's "register-blocking, not
// shared-tiling" framing.
//
// Each of the `group` query heads keeps FULLY INDEPENDENT online-softmax state
// (`running_max_s`/`running_sum_s`/`out_accum`, one slot per head in shared memory) -- heads
// share KV reads, never softmax state. The max/sum block-wide reductions (identical shuffle-tree
// code to `attention_f32`/`attention_f32_split_kv` above) run sequentially, once per head in the
// group, per KV tile -- the one place this design does not parallelize across the group (see
// issue #198 §5 for why: parallelizing this is a plausible v2, not attempted here).
//
// ─── Correctness: bit-exact per query head, same non-goal as #183's kernel ────────────────────
// For a FIXED head g, the register-blocked QK loop visits d=0..headDim-1 in the same order as
// `attention_f32`'s scalar loop (just interleaved with other heads' independent accumulators);
// the register-blocked PV loop visits t=0..tileLen-1 in the same order; the max/sum reductions
// reuse the identical shuffle-tree code. Grouping changes WHICH iterations share a K/V global
// read, never the order of operations within any one head's accumulation -- so at `kv_split==1`
// this kernel special-cases the trivial one-way combine (skips the `fast_exp_neg` reweighting
// entirely, since with exactly one partial there is nothing to reassociate) and is expected to be
// BIT-EXACT vs `attention_f32` for that case (validated directly, not just asserted -- see
// `CudaAttentionF32GqaSplitTests.cs`). At `kv_split>1` this kernel inherits EXACTLY
// `attention_f32_split_kv`'s already-characterized reassociation tolerance (same combine formula,
// same partial-buffer layout `[numHeads, kv_split]` / `[numHeads, kv_split, headDim]`, indexed by
// `hq = hkv*group + g`) -- no new tolerance category.
//
// ─── kv_split is a RUNTIME parameter, not a hardcoded constant (issue #197) ────────────────
// Unlike `attention_f32_split_kv`'s compile-time `ATTN_KV_SPLIT=4`, this kernel takes `kv_split`
// as a kernel argument, sized by `CudaKernels.ComputeAttentionKvSplit` (an occupancy-target
// heuristic modeled on Vulkan's `VulkanSplitKvAttentionKernel.ComputeSplits`/issue #347 in FORM
// only -- the constants are re-derived for CUDA's cooperative-launch co-residency ceiling, which
// is a hard requirement unlike Vulkan's soft dispatch-count target; see CudaKernels.cs's
// `ComputeAttentionKvSplit` doc for the full reasoning). Callers MUST clamp `kv_split` to
// `MaxSafeAttentionGqaSplit`'s result (queried via `cuOccupancyMaxActiveBlocksPerMultiprocessor`,
// same mechanism `IsAttentionSplitKvSafe` already uses) before launch -- exceeding the
// cooperative-launch co-residency ceiling is a hard CUDA error, not a soft perf regression.
#define MAX_GQA_GROUP 8

extern "C" __global__ void __launch_bounds__(256) attention_f32_gqa_split_kv(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ output,
    const int seq_kv,
    const int num_heads, const int num_kv_heads, const int head_dim,
    const int position_offset, const int sliding_window,
    const int kv_split,
    float* __restrict__ partial_max,   // [num_heads, kv_split]
    float* __restrict__ partial_sum,   // [num_heads, kv_split]
    float* __restrict__ partial_out)   // [num_heads, kv_split, head_dim]
{
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();

    int hkv = blockIdx.x;
    int s   = blockIdx.y;
    int group = num_heads / num_kv_heads;
    float scale = rsqrtf((float)head_dim);
    int pos_q = position_offset; // seq_q == 1 (decode-only kernel)

    int kv_stride = num_kv_heads * head_dim;

    // Dynamic shared layout, sized by the RUNTIME `group` (C# launcher computes sharedBytes from
    // the actual numHeads/numKvHeads ratio, not MAX_GQA_GROUP -- see LaunchAttentionF32GqaSplit).
    extern __shared__ float smem[];
    float* q_shared      = smem;                              // [group][head_dim]
    float* out_accum     = q_shared + group * head_dim;        // [group][head_dim]
    float* score_tile    = out_accum + group * head_dim;       // [group][TILE_KV]
    float* warp_scratch  = score_tile + group * TILE_KV;       // [32] -- reused per-head, sequential
    float* running_max_s = warp_scratch + 32;                  // [group]
    float* running_sum_s = running_max_s + group;              // [group]

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Load Q for every head in this block's GQA group; zero out_accum; init softmax state.
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;
        const float* q_vec = q + (size_t)hq * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            q_shared[g * head_dim + d] = q_vec[d];
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            out_accum[g * head_dim + d] = 0.0f;
        if (threadIdx.x == 0) { running_max_s[g] = -FLT_MAX; running_sum_s[g] = 0.0f; }
    }
    __syncthreads();

    // This split's contiguous KV sub-range: [kv_lo, kv_hi).
    int chunk = (seq_kv + kv_split - 1) / kv_split;
    int kv_lo = s * chunk;
    int kv_hi = kv_lo + chunk;
    if (kv_hi > seq_kv) kv_hi = seq_kv;
    if (kv_lo > seq_kv) kv_lo = seq_kv;

    for (int t_start = kv_lo; t_start < kv_hi; t_start += TILE_KV)
    {
        int t_end = t_start + TILE_KV;
        if (t_end > kv_hi) t_end = kv_hi;
        int tile_len = t_end - t_start;

        // -- Register-blocked QK: each K element read from global ONCE, reused `group` times. --
        for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
        {
            int tkv = t_start + t;
            bool masked = (tkv > pos_q) || (sliding_window > 0 && pos_q - tkv >= sliding_window);
            if (masked)
            {
                for (int g = 0; g < group; g++) score_tile[g * TILE_KV + t] = -FLT_MAX;
                continue;
            }

            const float* k_vec = k + (size_t)tkv * kv_stride + hkv * head_dim;
            float scores[MAX_GQA_GROUP];
            #pragma unroll
            for (int g = 0; g < MAX_GQA_GROUP; g++) scores[g] = 0.0f;
            for (int d = 0; d < head_dim; d++)
            {
                float kd = k_vec[d]; // ONE global read, shared across the group below
                for (int g = 0; g < group; g++)
                    scores[g] += q_shared[g * head_dim + d] * kd;
            }
            for (int g = 0; g < group; g++)
                score_tile[g * TILE_KV + t] = scores[g] * scale;
        }
        __syncthreads();

        // -- Per-head max/sum reduction + online-softmax rescale, sequential over the group. --
        // (identical shuffle-tree code to attention_f32/attention_f32_split_kv above, just
        // indexed by g -- see this kernel's header for why this loop is not itself parallelized).
        for (int g = 0; g < group; g++)
        {
            float tile_max = -FLT_MAX;
            for (int t = threadIdx.x; t < tile_len; t += blockDim.x)
                tile_max = fmaxf(tile_max, score_tile[g * TILE_KV + t]);
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
            if (lane == 0) warp_scratch[warp_id] = tile_max;
            __syncthreads();
            if (warp_id == 0) {
                int nw = (blockDim.x + warpSize - 1) / warpSize;
                tile_max = (lane < nw) ? warp_scratch[lane] : -FLT_MAX;
                for (int off = warpSize / 2; off > 0; off >>= 1)
                    tile_max = fmaxf(tile_max, __shfl_down_sync(0xFFFFFFFF, tile_max, off));
            }
            if (threadIdx.x == 0) warp_scratch[0] = tile_max;
            __syncthreads();
            tile_max = warp_scratch[0];

            float running_max = running_max_s[g];
            float new_max = fmaxf(running_max, tile_max);
            float correction = (running_max > -FLT_MAX + 1.0f)
                               ? fast_exp_neg(running_max - new_max) : 0.0f;
            float running_sum = running_sum_s[g] * correction;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                out_accum[g * head_dim + d] *= correction;
            __syncthreads();

            float tile_sum = 0.0f;
            for (int t = threadIdx.x; t < tile_len; t += blockDim.x) {
                float w = (score_tile[g * TILE_KV + t] > -FLT_MAX + 1.0f)
                          ? fast_exp_neg(score_tile[g * TILE_KV + t] - new_max) : 0.0f;
                score_tile[g * TILE_KV + t] = w;
                tile_sum += w;
            }
            for (int off = warpSize / 2; off > 0; off >>= 1)
                tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
            if (lane == 0) warp_scratch[warp_id] = tile_sum;
            __syncthreads();
            if (warp_id == 0) {
                int nw = (blockDim.x + warpSize - 1) / warpSize;
                tile_sum = (lane < nw) ? warp_scratch[lane] : 0.0f;
                for (int off = warpSize / 2; off > 0; off >>= 1)
                    tile_sum += __shfl_down_sync(0xFFFFFFFF, tile_sum, off);
                if (lane == 0) warp_scratch[0] = tile_sum;
            }
            __syncthreads();
            running_sum += warp_scratch[0];

            if (threadIdx.x == 0) { running_max_s[g] = new_max; running_sum_s[g] = running_sum; }
            // NOTE (issue #230): no __syncthreads() here. This barrier existed only to guard
            // `warp_scratch` reuse across loop iterations (next g's max-phase write at the top of
            // this loop vs this g's sum-phase read just above) -- a WAR hazard on the SCRATCH
            // BUFFER, not a real data dependency. It is already provably redundant:
            //   - Within one head, `warp_scratch` is next touched by the NEXT head's max-phase
            //     write, which is immediately followed by a REAL barrier (the `__syncthreads()`
            //     after the `warp_scratch[warp_id] = tile_max` write, a few lines below the top of
            //     this loop). That barrier requires every thread in the block to have reached it,
            //     which (single-threaded-per-thread program order) is only possible after every
            //     thread has already finished this g's entire body, INCLUDING the read above --
            //     so the WAR hazard is already closed by that barrier, transitively.
            //   - On the LAST iteration (g == group-1), nothing between here and the PV phase
            //     below touches `warp_scratch`, `score_tile[g]`, or `out_accum[g]` for this head --
            //     those were already made block-visible by the real barrier a few lines up (the one
            //     guarding `warp_scratch[lane] = tile_sum` / the cross-warp sum combine), so the PV
            //     phase's reads are safe without an additional rendezvous here.
            // ncu (`.perf-runs/ncu-2026-07-30-post197198/`) attributed 44.2% of this kernel's
            // per-instruction stall cycles to CTA-barrier waits; SASS inspection (ptxas -arch=sm_86
            // + cuobjdump --dump-sass / nvdisasm -g, no elevation needed, matching issue #218's
            // verify-before-trusting precedent) confirmed this per-head reduction loop -- run
            // sequentially `group` times per KV tile, as the kernel's own header already flags as
            // the one place this design doesn't parallelize across the group -- accounts for the
            // large majority of the kernel's dynamic barrier count (36 of ~39-51 barrier hits per
            // decode step at Bonsai-27B's group=6). This specific barrier was one of six per
            // iteration; removing it is a pure dependency-graph correction (no change to any
            // floating-point value, order, or the kv_split==1 bit-exactness contract) validated by
            // CudaAttentionF32GqaSplitTests (all group sizes 1..8) and
            // CudaAttentionSplitKvGenerationParityTests.
        }

        // -- Register-blocked PV: each V element read from global ONCE, reused `group` times. --
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float v_acc[MAX_GQA_GROUP];
            #pragma unroll
            for (int g = 0; g < MAX_GQA_GROUP; g++) v_acc[g] = 0.0f;
            for (int t = 0; t < tile_len; t++)
            {
                float v_val = (v + (size_t)(t_start + t) * kv_stride + hkv * head_dim)[d]; // ONE global read
                for (int g = 0; g < group; g++)
                {
                    float w = score_tile[g * TILE_KV + t];
                    if (w > 0.0f) v_acc[g] += w * v_val;
                }
            }
            for (int g = 0; g < group; g++)
                out_accum[g * head_dim + d] += v_acc[g];
        }
        __syncthreads();
    }

    // Publish this split's partial (UNNORMALIZED) for EACH of this block's group heads, using
    // the EXACT SAME [numHeads, kv_split] / [numHeads, kv_split, headDim] layout
    // attention_f32_split_kv already defined -- the combine phase below (and the C# scratch
    // allocator) depend on it being unchanged, just indexed by hq = hkv*group + g.
    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;
        if (threadIdx.x == 0)
        {
            partial_max[(size_t)hq * kv_split + s] = running_max_s[g];
            partial_sum[(size_t)hq * kv_split + s] = running_sum_s[g];
        }
        float* partial_out_vec = partial_out + ((size_t)hq * kv_split + s) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
            partial_out_vec[d] = out_accum[g * head_dim + d];
    }

    grid.sync();   // ALL blocks (every kv head, every split) must have published before combine.

    if (s != 0) return; // only the split=0 block per KV head performs the final combine.

    __shared__ float s_combined_max;
    __shared__ float s_combined_sum;

    for (int g = 0; g < group; g++)
    {
        int hq = hkv * group + g;

        if (kv_split == 1)
        {
            // Trivial one-way combine: nothing to reassociate, skip fast_exp_neg(0) entirely so
            // this path is bit-exact vs the un-split accumulation (see header "Correctness").
            if (threadIdx.x == 0)
            {
                s_combined_max = partial_max[(size_t)hq * kv_split + 0];
                s_combined_sum = partial_sum[(size_t)hq * kv_split + 0];
            }
            __syncthreads();
            float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;
            const float* partial_out_vec = partial_out + ((size_t)hq * kv_split + 0) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
                output[(size_t)hq * head_dim + d] = partial_out_vec[d] * sum_inv;
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0)
        {
            float m = -FLT_MAX;
            for (int i = 0; i < kv_split; i++)
                m = fmaxf(m, partial_max[(size_t)hq * kv_split + i]);

            float l = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float li = partial_sum[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
                l += li * w;
            }
            s_combined_max = m;
            s_combined_sum = l;
        }
        __syncthreads();

        float m = s_combined_max;
        float sum_inv = (s_combined_sum > 1e-10f) ? (1.0f / s_combined_sum) : 0.0f;

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        {
            float o = 0.0f;
            for (int i = 0; i < kv_split; i++)
            {
                float mi = partial_max[(size_t)hq * kv_split + i];
                float w = (mi > -FLT_MAX + 1.0f) ? fast_exp_neg(mi - m) : 0.0f;
                float oi = partial_out[((size_t)hq * kv_split + i) * head_dim + d];
                o += oi * w;
            }
            output[(size_t)hq * head_dim + d] = o * sum_inv;
        }
        __syncthreads();
    }
}
