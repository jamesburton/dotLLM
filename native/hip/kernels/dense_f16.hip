// Dense single-token HIP primitives. Resident weights stay F16/BF16/packed; reductions stay in registers.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__device__ __forceinline__ float sum32(float value)
{
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down(value, offset, 32);
    return __shfl(value, 0, 32);
}

extern "C" __global__ void dense_embed_f16(__half* output, const __half* weights, int token, int dim)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) output[i] = weights[(size_t)token * dim + i];
}

extern "C" __global__ void dense_convert_f32_f16(__half* output, const float* input, int count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(input[i]);
}

extern "C" __global__ void dense_rmsnorm_f16(
    __half* output, const __half* input, const __half* weight, int dim, float epsilon)
{
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)dim; i += 32)
    {
        float value = __half2float(input[i]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum32(sum) / dim + epsilon);
    for (size_t i = threadIdx.x; i < (size_t)dim; i += 32)
        output[i] = __float2half(__half2float(input[i]) * inverse * __half2float(weight[i]));
}

extern "C" __global__ void dense_linear_f16(
    __half* output, const __half* input, const __half* weights, int outDim, int inDim)
{
    int row = blockIdx.x;
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)inDim; i += 32)
        sum = fmaf(__half2float(input[i]), __half2float(weights[(size_t)row * inDim + i]), sum);
    sum = sum32(sum);
    if (threadIdx.x == 0) output[row] = __float2half(sum);
}

__device__ __forceinline__ float bf16_value(unsigned short value)
{
    return __uint_as_float((unsigned int)value << 16);
}

extern "C" __global__ void dense_embed_bf16(__half* output, const unsigned short* weights, int token, int dim)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) output[i] = __float2half(bf16_value(weights[(size_t)token * dim + i]));
}

extern "C" __global__ void dense_rmsnorm_bf16(
    __half* output, const __half* input, const unsigned short* weight, int dim, float epsilon)
{
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)dim; i += 32)
    {
        float value = __half2float(input[i]);
        sum = fmaf(value, value, sum);
    }
    float inverse = rsqrtf(sum32(sum) / dim + epsilon);
    for (size_t i = threadIdx.x; i < (size_t)dim; i += 32)
        output[i] = __float2half(__half2float(input[i]) * inverse * bf16_value(weight[i]));
}

extern "C" __global__ void dense_linear_bf16(
    __half* output, const __half* input, const unsigned short* weights, int outDim, int inDim)
{
    int row = blockIdx.x;
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)inDim; i += 32)
        sum = fmaf(__half2float(input[i]), bf16_value(weights[(size_t)row * inDim + i]), sum);
    sum = sum32(sum);
    if (threadIdx.x == 0) output[row] = __float2half(sum);
}

extern "C" __global__ void dense_linear_packed4(
    __half* output, const __half* input, const unsigned int* qweight, const __half* scales,
    const unsigned int* qzeros, const int* groupIndices, int outDim, int inDim, int groupSize, int awq)
{
    int row = blockIdx.x, lane = row % 8;
    // AutoAWQ packs logical outputs [0,2,4,6,1,3,5,7]; GPTQ packs consecutive K values.
    int zeroShift = (awq ? (lane % 2) * 4 + lane / 2 : lane) * 4;
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)inDim; i += 32)
    {
        int group = awq ? i / groupSize : groupIndices[i];
        unsigned int packed = awq ? qweight[i * (outDim / 8) + row / 8]
            : qweight[(i / 8) * outDim + row];
        int code = (packed >> (awq ? zeroShift : (i % 8) * 4)) & 15;
        int zero = ((qzeros[(size_t)group * (outDim / 8) + row / 8] >> zeroShift) & 15) + (awq ? 0 : 1);
        float weight = (code - zero) * __half2float(scales[(size_t)group * outDim + row]);
        sum = fmaf(__half2float(input[i]), weight, sum);
    }
    sum = sum32(sum);
    if (threadIdx.x == 0) output[row] = __float2half(sum);
}

extern "C" __global__ void dense_add_f16(__half* output, const __half* left, const __half* right, int count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2half(__half2float(left[i]) + __half2float(right[i]));
}

extern "C" __global__ void dense_swiglu_f16(__half* output, const __half* gate, const __half* up, int count)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
    {
        float value = __half2float(gate[i]);
        output[i] = __float2half((value / (1.0f + expf(-value))) * __half2float(up[i]));
    }
}

extern "C" __global__ void dense_rope_f16(
    __half* q, __half* k, int qHeads, int kvHeads, int headDim, int rotaryDim,
    int position, float theta, float scale, int neox)
{
    size_t pair = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int pairs = rotaryDim / 2;
    if (pair >= (qHeads + kvHeads) * pairs) return;
    int head = pair / pairs;
    int j = pair % pairs;
    __half* data = head < qHeads ? q + (size_t)head * headDim : k + (size_t)(head - qHeads) * headDim;
    int first = neox ? j : 2 * j;
    int second = neox ? j + pairs : first + 1;
    float angle = position * scale * powf(theta, -2.0f * j / rotaryDim);
    float sine, cosine;
    sincosf(angle, &sine, &cosine);
    float x = __half2float(data[first]), y = __half2float(data[second]);
    data[first] = __float2half(x * cosine - y * sine);
    data[second] = __float2half(x * sine + y * cosine);
}

extern "C" __global__ void dense_write_kv_f16(
    __half* const* kPages, __half* const* vPages, const __half* k, const __half* v,
    int pageTokens, int kvDim, int position)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < kvDim)
    {
        int page = position / pageTokens;
        size_t offset = (size_t)(position % pageTokens) * kvDim + i;
        kPages[page][offset] = k[i];
        vPages[page][offset] = v[i];
    }
}

extern "C" __global__ void dense_attention_f16(
    __half* output, const __half* q, const __half* const* kPages, const __half* const* vPages,
    int pageTokens, int qHeads, int kvHeads, int headDim, int length, float scale)
{
    int head = blockIdx.x;
    int lane = threadIdx.x;
    int kvHead = head / (qHeads / kvHeads);
    int kvDim = kvHeads * headDim;
    // shortcut: at most 256 channels per head, add a separately qualified kernel for larger heads.
    float query[8], result[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int d = lane + 32 * j;
        query[j] = d < headDim ? __half2float(q[(size_t)head * headDim + d]) : 0.0f;
        result[j] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (int token = 0; token < length; ++token)
    {
        int page = token / pageTokens;
        size_t offset = (size_t)(token % pageTokens) * kvDim + kvHead * headDim;
        const __half* key = kPages[page] + offset;
        const __half* value = vPages[page] + offset;
        float score = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int d = lane + 32 * j;
            if (d < headDim) score = fmaf(query[j], __half2float(key[d]), score);
        }
        score = sum32(score) * scale;
        float nextMaximum = fmaxf(maximum, score);
        float oldScale = expf(maximum - nextMaximum);
        float probability = expf(score - nextMaximum);
        denominator = denominator * oldScale + probability;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int d = lane + 32 * j;
            if (d < headDim)
                result[j] = result[j] * oldScale + probability * __half2float(value[d]);
        }
        maximum = nextMaximum;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int d = lane + 32 * j;
        if (d < headDim) output[(size_t)head * headDim + d] = __float2half(result[j] / denominator);
    }
}


// GGUF block layouts match DotLLM.Cpu.Kernels.Dequantize: Q4_0 stores low nibbles
// for channels 0..15 and high nibbles for channels 16..31; Q8_0 bytes are signed.
__device__ __forceinline__ float packed_weight(const unsigned char* row, size_t index, int type)
{
    int within = index % 32;
    const unsigned char* block = row + (index / 32) * (type == 2 ? 18 : 34);
    float scale = __half2float(*reinterpret_cast<const __half*>(block));
    int quant;
    if (type == 2)
    {
        unsigned char pair = block[2 + within % 16];
        quant = (within < 16 ? pair & 15 : pair >> 4) - 8;
    }
    else quant = reinterpret_cast<const signed char*>(block + 2)[within];
    return scale * quant;
}

extern "C" __global__ void dense_embed_packed(
    __half* output, const unsigned char* weights, int token, int dim, int type)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t rowBytes = (size_t)(dim / 32) * (type == 2 ? 18 : 34);
    if (i < dim) output[i] = __float2half(packed_weight(weights + (size_t)token * rowBytes, i, type));
}

extern "C" __global__ void dense_linear_packed(
    __half* output, const __half* input, const unsigned char* weights, int outDim, int inDim, int type)
{
    int row = blockIdx.x;
    size_t rowBytes = (size_t)(inDim / 32) * (type == 2 ? 18 : 34);
    const unsigned char* weightRow = weights + (size_t)row * rowBytes;
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < (size_t)inDim; i += 32)
        sum = fmaf(__half2float(input[i]), packed_weight(weightRow, i, type), sum);
    sum = sum32(sum);
    if (threadIdx.x == 0) output[row] = __float2half(sum);
}

__device__ __forceinline__ float max32(float value)
{
    for (int offset = 16; offset; offset >>= 1)
        value = fmaxf(value, __shfl_down(value, offset, 32));
    return __shfl(value, 0, 32);
}

__device__ __forceinline__ void write_kv_block(unsigned char* row, const __half* input, int block, int type)
{
    int lane = threadIdx.x;
    float value = __half2float(input[(size_t)block * 32 + lane]);
    if (type == 0)
    {
        reinterpret_cast<__half*>(row)[(size_t)block * 32 + lane] = __float2half(value);
        return;
    }
    float maximum = max32(fabsf(value));
    float scale = maximum / 127.0f;
    float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    unsigned char* destination = row + (size_t)block * 34;
    if (lane == 0) *reinterpret_cast<__half*>(destination) = __float2half(scale);
    // MathF.Round in the CPU reference uses ties-to-even; clamp symmetrically.
    int quant = __float2int_rn(value * inverse);
    quant = quant < -127 ? -127 : (quant > 127 ? 127 : quant);
    reinterpret_cast<signed char*>(destination + 2)[lane] = (signed char)quant;
}

extern "C" __global__ void dense_write_kv_packed(
    unsigned char* const* kPages, unsigned char* const* vPages, const __half* k, const __half* v,
    int pageTokens, int kvDim, int position, int keyType, int valueType)
{
    int page = position / pageTokens;
    size_t row = position % pageTokens;
    size_t keyRowBytes = keyType == 0 ? (size_t)kvDim * 2 : (size_t)(kvDim / 32) * 34;
    size_t valueRowBytes = valueType == 0 ? (size_t)kvDim * 2 : (size_t)(kvDim / 32) * 34;
    write_kv_block(kPages[page] + row * keyRowBytes, k, blockIdx.x, keyType);
    write_kv_block(vPages[page] + row * valueRowBytes, v, blockIdx.x, valueType);
}

__device__ __forceinline__ float kv_value(const unsigned char* row, size_t index, int type)
{
    if (type == 0) return __half2float(reinterpret_cast<const __half*>(row)[index]);
    const unsigned char* block = row + (index / 32) * 34;
    return __half2float(*reinterpret_cast<const __half*>(block))
        * reinterpret_cast<const signed char*>(block + 2)[index % 32];
}

extern "C" __global__ void dense_attention_packed(
    __half* output, const __half* q, const unsigned char* const* kPages, const unsigned char* const* vPages,
    int pageTokens, int qHeads, int kvHeads, int headDim, int length, float scale, int keyType, int valueType)
{
    int head = blockIdx.x;
    int lane = threadIdx.x;
    int kvHead = head / (qHeads / kvHeads);
    int kvDim = kvHeads * headDim;
    size_t keyRowBytes = keyType == 0 ? (size_t)kvDim * 2 : (size_t)(kvDim / 32) * 34;
    size_t valueRowBytes = valueType == 0 ? (size_t)kvDim * 2 : (size_t)(kvDim / 32) * 34;
    float query[8], result[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int d = lane + 32 * j;
        query[j] = d < headDim ? __half2float(q[(size_t)head * headDim + d]) : 0.0f;
        result[j] = 0.0f;
    }
    float maximum = -INFINITY, denominator = 0.0f;
    for (int token = 0; token < length; ++token)
    {
        int page = token / pageTokens;
        size_t row = token % pageTokens;
        const unsigned char* key = kPages[page] + row * keyRowBytes;
        const unsigned char* value = vPages[page] + row * valueRowBytes;
        float score = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int d = lane + 32 * j;
            if (d < headDim) score = fmaf(query[j], kv_value(key, (size_t)kvHead * headDim + d, keyType), score);
        }
        score = sum32(score) * scale;
        float nextMaximum = fmaxf(maximum, score);
        float oldScale = expf(maximum - nextMaximum);
        float probability = expf(score - nextMaximum);
        denominator = denominator * oldScale + probability;
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int d = lane + 32 * j;
            if (d < headDim)
                result[j] = result[j] * oldScale + probability * kv_value(value, (size_t)kvHead * headDim + d, valueType);
        }
        maximum = nextMaximum;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int d = lane + 32 * j;
        if (d < headDim) output[(size_t)head * headDim + d] = __float2half(result[j] / denominator);
    }
}
