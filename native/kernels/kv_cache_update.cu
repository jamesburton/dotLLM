#include <cuda_fp16.h>

extern "C" __global__ void __launch_bounds__(256) kv_cache_update_pos_f16(
    const half* __restrict__ key,
    const half* __restrict__ value,
    half* __restrict__ cache_key,
    half* __restrict__ cache_value,
    const int* __restrict__ positions,
    const int kv_stride)
{
    int pos = positions[0];
    half* key_dst = cache_key + (size_t)pos * kv_stride;
    half* value_dst = cache_value + (size_t)pos * kv_stride;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < kv_stride; i += blockDim.x * gridDim.x)
    {
        key_dst[i] = key[i];
        value_dst[i] = value[i];
    }
}
