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
//   KV_TQ      TurboQuant (issue #1892), read IN CODE SPACE: per head vector (head_dim elements) TQ_CODE_UINTS 32-bit words of
//              TQ_MSE_BITS-bit centroid codes packed LSB first (coordinate c at bit c * TQ_MSE_BITS of the vector's words; a code may
//              straddle two words, vectors never share a word) plus one norm, so element i = (hv, c) with hv = i / head_dim,
//              c = i % head_dim is centroid[code(hv, c)] * norm(hv). That is the cached vector in the rotated space, NOT the original
//              one: the host rotates the query into K code space before the body runs (tq_rotate_q) and the output back out of V code
//              space after it (tq_rotate_o), once per dispatch. Nothing is decoded into a scratch and nothing is added to shared
//              memory: the centroid table stays a buffer.
//              Besides the code words at bindings 1 / 2 it binds the K norms, V norms and centroids at the FIXED bindings 12, 13 and
//              14 (no body may use them), and takes the layout as specialization constants at ids 20..23, so the loader folds at
//              pipeline compile time and costs a body no push-constant register. Each layer's code buffer ends in one guard word
//              (the straddle read of the last vector's last code).
//              Norm formats (TqNormFormat): 1 = F16, 2 = BF16, two vectors per word, the even one in the low half; 0 = F32 (reference).
//              KV_K4 / KV_V4 (i a multiple of 4) and kvTqK8 / kvTqV8 (i a multiple of 8) load consecutive elements of one head vector
//              with one norm read, for the bodies that load in groups.
//
// K and V pick their storage independently (issue #1884, mixed --cache-type-k / -v): KV_K_F16 | KV_K_Q8 | KV_K_Q4 | KV_K_TQ and the
// KV_V_* twins. KV_F16 / KV_Q8 / KV_Q4 / KV_TQ set both sides.
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
#if defined(KV_TQ)
#define KV_K_TQ 1
#define KV_V_TQ 1
#endif

#if defined(KV_K_TQ) || defined(KV_V_TQ)
layout(constant_id = 20) const uint TQ_MSE_BITS    = 4u;    // bits per coordinate, 1..8
layout(constant_id = 21) const uint TQ_CODE_UINTS  = 16u;   // 32-bit words per head vector
layout(constant_id = 22) const uint TQ_NORM_FORMAT = 1u;    // 0 = F32 (reference), 1 = F16, 2 = BF16
layout(constant_id = 23) const uint TQ_HD_LOG2     = 7u;    // log2(head_dim)
const uint TQ_CODE_MASK = (1u << TQ_MSE_BITS) - 1u;
const bool TQ_STRADDLE  = (32u % TQ_MSE_BITS) != 0u;        // a code can cross a word boundary (3, 5, 6, 7 bits)
layout(set = 0, binding = 14, std430) readonly buffer BufTqCent { float tqCent[]; };   // [1 << TQ_MSE_BITS], scaled by 1/sqrt(d)

uint tqNormWord(uint hv) { return TQ_NORM_FORMAT == 0u ? hv : hv >> 1u; }
float tqNorm(uint w, uint hv) {
    if (TQ_NORM_FORMAT == 0u) return uintBitsToFloat(w);
    uint h = (w >> ((hv & 1u) << 4u)) & 0xFFFFu;
    return TQ_NORM_FORMAT == 1u ? unpackHalf2x16(h).x : uintBitsToFloat(h << 16u);
}
// The code at bit sh of word w0; w1 is the next word (only read into the result when the code straddles).
uint tqCode(uint w0, uint w1, uint sh) {
    uint c = w0 >> sh;
    if (TQ_STRADDLE) c |= (w1 << 1u) << (31u - sh);   // two shifts: sh == 0 must contribute nothing
    return c & TQ_CODE_MASK;
}
// Four consecutive codes starting at bit sh of w0 (4 * TQ_MSE_BITS <= 32 bits, possibly crossing into w1), times the norm.
vec4 tqGroup4(uint w0, uint w1, uint sh, float norm) {
    uint f = (w0 >> sh) | ((w1 << 1u) << (31u - sh));
    return vec4(tqCent[f & TQ_CODE_MASK], tqCent[(f >> TQ_MSE_BITS) & TQ_CODE_MASK],
                tqCent[(f >> (2u * TQ_MSE_BITS)) & TQ_CODE_MASK], tqCent[(f >> (3u * TQ_MSE_BITS)) & TQ_CODE_MASK]) * norm;
}
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
#elif defined(KV_K_TQ)
layout(set = 0, binding = 1, std430)  readonly buffer BufK     { uint kw[]; };  // [seq_kv * num_kv_heads, TQ_CODE_UINTS] code words + 1 guard word
layout(set = 0, binding = 12, std430) readonly buffer BufKNorm { uint kn[]; };  // [seq_kv * num_kv_heads] norms, TQ_NORM_FORMAT
float kvTqK(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    return tqCent[tqCode(kw[w], kw[w + 1u], bit & 31u)] * tqNorm(kn[tqNormWord(hv)], hv);
}
vec4 kvTqK4(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    return tqGroup4(kw[w], kw[w + 1u], bit & 31u, tqNorm(kn[tqNormWord(hv)], hv));
}
mat2x4 kvTqK8(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS, bit2 = bit + 4u * TQ_MSE_BITS;
    uint b = hv * TQ_CODE_UINTS, w = b + (bit >> 5u), w2 = b + (bit2 >> 5u);
    float norm = tqNorm(kn[tqNormWord(hv)], hv);
    return mat2x4(tqGroup4(kw[w], kw[w + 1u], bit & 31u, norm), tqGroup4(kw[w2], kw[w2 + 1u], bit2 & 31u, norm));
}
#define KV_K(i) kvTqK(i)
#define KV_K4(i) kvTqK4(i)
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
#elif defined(KV_V_TQ)
layout(set = 0, binding = 2, std430)  readonly buffer BufV     { uint vw[]; };  // [seq_kv * num_kv_heads, TQ_CODE_UINTS] code words + 1 guard word
layout(set = 0, binding = 13, std430) readonly buffer BufVNorm { uint vn[]; };  // [seq_kv * num_kv_heads] norms, TQ_NORM_FORMAT
float kvTqV(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    return tqCent[tqCode(vw[w], vw[w + 1u], bit & 31u)] * tqNorm(vn[tqNormWord(hv)], hv);
}
vec4 kvTqV4(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS;
    uint w = hv * TQ_CODE_UINTS + (bit >> 5u);
    return tqGroup4(vw[w], vw[w + 1u], bit & 31u, tqNorm(vn[tqNormWord(hv)], hv));
}
mat2x4 kvTqV8(uint i) {
    uint hv = i >> TQ_HD_LOG2, bit = (i & ((1u << TQ_HD_LOG2) - 1u)) * TQ_MSE_BITS, bit2 = bit + 4u * TQ_MSE_BITS;
    uint b = hv * TQ_CODE_UINTS, w = b + (bit >> 5u), w2 = b + (bit2 >> 5u);
    float norm = tqNorm(vn[tqNormWord(hv)], hv);
    return mat2x4(tqGroup4(vw[w], vw[w + 1u], bit & 31u, norm), tqGroup4(vw[w2], vw[w2 + 1u], bit2 & 31u, norm));
}
#define KV_V(i) kvTqV(i)
#define KV_V4(i) kvTqV4(i)
#elif defined(KV_V_F16)
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uint vw[]; };    // [seq_kv, num_kv_heads * head_dim] F16, 2 per word
#define KV_V(i) unpackHalf2x16(vw[(i) >> 1u] >> (((i) & 1u) << 4u)).x
#else
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { float v[]; };    // [seq_kv, num_kv_heads * head_dim]
#define KV_V(i) v[i]
#endif
