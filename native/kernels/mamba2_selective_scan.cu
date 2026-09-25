// native/kernels/mamba2_selective_scan.cu
//
// Mamba2 selective state-space scan (NVIDIA Nemotron-H). Bit-order-faithful port of
// DotLLM.Cpu.Kernels.Mamba2SelectiveScan.Execute, fused with:
//   - the raw-dt + dt_bias add and GUARDED softplus (Mamba2SelectiveScan.SoftPlus's exact
//     three-branch form: x>20 -> x; x<-20 -> exp(x); else log(1+exp(x))) — NOT the unguarded
//     form gdn_decay_f32 uses (that kernel's own CPU oracle has no guard; this one's does).
//   - the D-skip term (CPU step 7 of NemotronHTransformerModel.ForwardSsmBody: y += x*D[h]),
//     a per-element op safe to fuse since it adds no cross-thread reduction.
//
// Do NOT generalize this into a shared kernel with any future Mamba3/SSD scan (issue #346) —
// Mamba3's recurrence rotates B/C through RoPE and fuses a pre-RoPE skip-dot + gate that this
// Mamba2 recurrence does not have. Confirmed independently by reading both CPU references.
//
// ── Layouts (matches the CPU reference exactly) ─────────────────────────────
//   state  : [n_head, head_dim, d_state]  row-major, in/out
//   x      : [seq_len, d_inner]           row-major (d_inner = n_head*head_dim)
//   dt_raw : [seq_len, n_head]            row-major, NOT yet bias-added
//   dt_bias: [n_head]
//   a      : [n_head]                     scalar A per head (GGUF stores it negative)
//   d      : [n_head]                     scalar D per head (skip connection)
//   b, c   : [seq_len, n_group, d_state]  row-major
//   y      : [seq_len, d_inner]           row-major, out (already includes the D-skip term)
//
// ── Parallelization ──────────────────────────────────────────────────────────
// One block per head (gridDim.x = n_head). One thread per head-channel
// (blockDim.x = head_dim; host launch requires head_dim <= 256, this kernel is
// __launch_bounds__(256)). Each thread i owns state row state[h, i, 0..d_state) and walks
// t = 0..seq_len-1 sequentially (state depends on t-1), and for each t walks
// k = 0..d_state-1 sequentially in-register — the SAME nesting and accumulation order as the
// CPU's `for t / for h / for i / for k` loop nest (h is the block, i is the thread, t and k are
// the two sequential loops each thread runs). B/C are shared across every head in a group, so
// each block (== one head) stages this token's b/c group slice into shared memory once per t,
// exactly like gdn_scan_step_f32 stages k_shared/q_shared once per call.
//
// group index: g = h / (n_head / n_group) (heads_per_group heads share one B/C group).

extern "C" __global__ void __launch_bounds__(256) mamba2_selective_scan_f32(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ dt_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    const float* __restrict__ d,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ y,
    const int n_head, const int head_dim, const int d_state, const int n_group, const int seq_len)
{
    int h = blockIdx.x;
    if (h >= n_head) return;
    int i = threadIdx.x; // this thread's channel within the head; valid range [0, head_dim)

    int d_inner = n_head * head_dim;
    int heads_per_group = n_head / n_group;
    int g = h / heads_per_group;

    extern __shared__ float smem[];
    float* b_shared = smem;             // [d_state]
    float* c_shared = smem + d_state;   // [d_state]

    float a_h = a[h];
    float d_h = d[h];
    float dtb_h = dt_bias[h];

    // state row for this (h, i) — only meaningful while i < head_dim, which host launch
    // guarantees (blockDim.x == head_dim exactly).
    float* state_row = state + ((size_t)h * head_dim + i) * (size_t)d_state;

    for (int t = 0; t < seq_len; t++)
    {
        const float* b_row = b + ((size_t)t * n_group + g) * (size_t)d_state;
        const float* c_row = c + ((size_t)t * n_group + g) * (size_t)d_state;
        for (int k = threadIdx.x; k < d_state; k += blockDim.x)
        {
            b_shared[k] = b_row[k];
            c_shared[k] = c_row[k];
        }
        __syncthreads();

        // dt = dt_raw + dt_bias, then GUARDED softplus (Mamba2SelectiveScan.SoftPlus exactly).
        float dt_val = dt_raw[(size_t)t * n_head + h] + dtb_h;
        float dt_sp;
        if (dt_val > 20.0f)      dt_sp = dt_val;
        else if (dt_val < -20.0f) dt_sp = expf(dt_val);
        else                      dt_sp = logf(1.0f + expf(dt_val));

        // A is stored negative by the GGUF converter; exp(dt_sp * a_h) is in (0,1) -> decay.
        float dA = expf(dt_sp * a_h);

        float x_val = x[(size_t)t * d_inner + (size_t)h * head_dim + i];
        float x_dt = x_val * dt_sp;

        float sumf = 0.0f;
        for (int k = 0; k < d_state; k++)
        {
            float s = state_row[k] * dA + b_shared[k] * x_dt;
            state_row[k] = s;
            sumf += s * c_shared[k];
        }

        // D-skip fused in (CPU step 7): elementwise, no reduction, safe to fuse.
        y[(size_t)t * d_inner + (size_t)h * head_dim + i] = sumf + x_val * d_h;

        __syncthreads(); // before next t's shared-memory overwrite
    }
}
