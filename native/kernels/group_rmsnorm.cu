// native/kernels/group_rmsnorm.cu
//
// Per-group RMS normalization (NVIDIA Nemotron-H Mamba2 SSM output norm). Structural copy of
// per_head_rmsnorm_f32.cu with ONE change: weight is indexed per-GROUP (weight[g*group_dim+i]),
// not shared across all groups (per_head_rmsnorm_f32's weight[i] broadcast is WRONG for this use
// — NemotronH's ssm_norm.weight is [n_group*group_dim], each group has its own gain slice; see
// NemotronHTransformerModel.ForwardSsmBody step 9, ssmW.NormWeight.AsSpan(g*groupDim, groupDim)).
//
// Layout: x is [seq_len, n_group, group_dim] row-major (n_group*group_dim == d_inner).
// weight is [n_group, group_dim] row-major (== ssm_norm.weight, GGUF shape [d_inner]).
//
// Warp-shuffle tree reduction (same precision philosophy as per_head_rmsnorm_f32 and every other
// RMSNorm-family CUDA kernel in this codebase) — tolerance-based parity with the CPU's sequential
// RmsNorm.Execute reduction, not bit-exact.

extern "C" __global__ void __launch_bounds__(256) group_rmsnorm_f32(
    float* __restrict__ x, const float* __restrict__ weight,
    const float eps, const int seq_len, const int n_group, const int group_dim)
{
    int block_id = blockIdx.x;
    int t = block_id / n_group, g = block_id % n_group;
    if (t >= seq_len) return;

    float* vec = x + (size_t)t * n_group * group_dim + (size_t)g * group_dim;
    const float* w = weight + (size_t)g * group_dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) { float v = vec[i]; sum_sq += v * v; }
    for (int off = warpSize / 2; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float ws[32]; int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    if (lane == 0) ws[wid] = sum_sq; __syncthreads();
    if (wid == 0) { int nw = (blockDim.x + warpSize - 1) / warpSize; sum_sq = (lane < nw) ? ws[lane] : 0.0f;
        for (int off = warpSize / 2; off > 0; off >>= 1) sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off); }
    __shared__ float ri; if (threadIdx.x == 0) ri = rsqrtf(sum_sq / (float)group_dim + eps); __syncthreads();
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x)
        vec[i] = vec[i] * ri * w[i];
}
