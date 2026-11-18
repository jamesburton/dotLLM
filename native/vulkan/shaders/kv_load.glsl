// K / V cache element loads shared by the attention shaders (issue #1102).
//
// NOT compiled on its own: the .glsl extension keeps it out of build.{sh,ps1}'s *.comp glob. Every attention body that reads the
// KV cache includes it in place of its own BufK / BufV declarations and reads elements through KV_K(i) / KV_V(i), where i is the
// element index into a row-major [seq_kv, num_kv_heads * head_dim] cache. The .comp wrapper picks the storage:
//
//   (default)  F32: one float per element, the declarations the shaders always had (the F32 pipelines compile to the same SPIR-V).
//   KV_F16     F16: two halves per 32-bit word, the even element in the low half, as convert_f32_to_f16_range.comp writes them.
//              uint-backed, so no 16-bit storage device feature is needed. unpackHalf2x16 is exact, so a kernel reading F16 KV
//              computes on exactly the values an F32 kernel reads from a cache holding the F16-rounded rows (the parity oracle).
//              Nothing is widened into a scratch: each element is converted in the load that consumes it. 2 bytes per element.
//   KV_Q8      Q8_0 (issue: Flint Q8_0 KV): 36-byte blocks of 32 elements, [float scale][32 x int8], rows of num_kv_heads * head_dim elements
//              (a multiple of 32), so element i lives in block i >> 5 at byte 4 + (i & 31) of that block (word 9 * block + 1 + ((i & 31) >> 2)).
//              Dequantised in the load that consumes it (int8 * scale, both exact in f32); nothing is widened into a scratch. 1.125 bytes per element.
//   KV_Q4      Q4_0 (issue #1884): 20-byte blocks of 32 elements, [float scale][16 bytes], the KV convention CPU (KvQuantize) and CUDA
//              (attention_kvq.cu, #956) use: ADJACENT PAIRS, byte j = element 2j (low nibble) | element 2j + 1 (high nibble), value =
//              (nibble - 8) * scale. Element i of a block is in word 1 + ((i & 31) >> 3), byte ((i & 31) >> 1) & 3, nibble i & 1. The
//              scale is F32 here (CPU / CUDA store F16), as Vulkan's Q8_0 does. 0.625 bytes per element.
//
// K and V pick their storage independently (issue #1884, mixed --cache-type-k / -v): KV_K_F16 | KV_K_Q8 | KV_K_Q4 and the KV_V_*
// twins. KV_F16 / KV_Q8 / KV_Q4 set both sides.
#if defined(KV_F16)
#define KV_K_F16 1
#define KV_V_F16 1
#endif
#if defined(KV_Q8)
#define KV_K_Q8 1
#define KV_V_Q8 1
#endif
#if defined(KV_Q4)
#define KV_K_Q4 1
#define KV_V_Q4 1
#endif

float kvLoadQ4(uint w, uint scaleWord, uint e) {
    uint nib = (w >> ((((e >> 1u) & 3u) << 3u) + ((e & 1u) << 2u))) & 15u;
    return (float(nib) - 8.0) * uintBitsToFloat(scaleWord);
}

#if defined(KV_K_Q8)
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uint kw[]; };
float kvq8K(uint i) {
    uint blk = i >> 5u, e = i & 31u;
    return float(bitfieldExtract(int(kw[blk * 9u + 1u + (e >> 2u)]), int((e & 3u) << 3u), 8)) * uintBitsToFloat(kw[blk * 9u]);
}
#define KV_K(i) kvq8K(i)
#elif defined(KV_K_Q4)
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uint kw[]; };
float kvK(uint i) { uint blk = i >> 5u, e = i & 31u; return kvLoadQ4(kw[blk * 5u + 1u + (e >> 3u)], kw[blk * 5u], e); }
#define KV_K(i) kvK(i)
#elif defined(KV_K_F16)
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uint kw[]; };    // [seq_kv, num_kv_heads * head_dim] F16, 2 per word
#define KV_K(i) unpackHalf2x16(kw[(i) >> 1u] >> (((i) & 1u) << 4u)).x
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { float k[]; };    // [seq_kv, num_kv_heads * head_dim]
#define KV_K(i) k[i]
#endif

#if defined(KV_V_Q8)
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uint vw[]; };
float kvq8V(uint i) {
    uint blk = i >> 5u, e = i & 31u;
    return float(bitfieldExtract(int(vw[blk * 9u + 1u + (e >> 2u)]), int((e & 3u) << 3u), 8)) * uintBitsToFloat(vw[blk * 9u]);
}
#define KV_V(i) kvq8V(i)
#elif defined(KV_V_Q4)
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uint vw[]; };
float kvV(uint i) { uint blk = i >> 5u, e = i & 31u; return kvLoadQ4(vw[blk * 5u + 1u + (e >> 3u)], vw[blk * 5u], e); }
#define KV_V(i) kvV(i)
#elif defined(KV_V_F16)
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uint vw[]; };    // [seq_kv, num_kv_heads * head_dim] F16, 2 per word
#define KV_V(i) unpackHalf2x16(vw[(i) >> 1u] >> (((i) & 1u) << 4u)).x
#else
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { float v[]; };    // [seq_kv, num_kv_heads * head_dim]
#define KV_V(i) v[i]
#endif
