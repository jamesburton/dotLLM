// K / V cache element loads shared by the attention shaders (issue #1102).
//
// NOT compiled on its own: the .glsl extension keeps it out of build.{sh,ps1}'s *.comp glob. Every attention body that reads the
// KV cache includes it in place of its own BufK / BufV declarations and reads elements through KV_K(i) / KV_V(i), where i is the
// element index into a row-major [seq_kv, num_kv_heads * head_dim] cache (KV_K2 / KV_V2: elements i and i + 1 for an EVEN i, one
// word in F16). The .comp wrapper picks the storage:
//
//   (default)  F32: one float per element, the declarations the shaders always had (the F32 pipelines compile to the same SPIR-V).
//   KV_F16     F16: two halves per 32-bit word, the even element in the low half, as convert_f32_to_f16_range.comp writes them.
//              uint-backed, so no 16-bit storage device feature is needed. unpackHalf2x16 is exact, so a kernel reading F16 KV
//              computes on exactly the values an F32 kernel reads from a cache holding the F16-rounded rows (the parity oracle).
//              Nothing is widened into a scratch: each element is converted in the load that consumes it. 2 bytes per element.
#if defined(KV_F16)
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { uint kw[]; };    // [seq_kv, num_kv_heads * head_dim] F16, 2 per word
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { uint vw[]; };    // [seq_kv, num_kv_heads * head_dim] F16, 2 per word
#define KV_K(i) unpackHalf2x16(kw[(i) >> 1u] >> (((i) & 1u) << 4u)).x
#define KV_V(i) unpackHalf2x16(vw[(i) >> 1u] >> (((i) & 1u) << 4u)).x
#define KV_K2(i) unpackHalf2x16(kw[(i) >> 1u])
#define KV_V2(i) unpackHalf2x16(vw[(i) >> 1u])
#else
layout(set = 0, binding = 1, std430) readonly  buffer BufK   { float k[]; };    // [seq_kv, num_kv_heads * head_dim]
layout(set = 0, binding = 2, std430) readonly  buffer BufV   { float v[]; };    // [seq_kv, num_kv_heads * head_dim]
#define KV_K(i) k[i]
#define KV_V(i) v[i]
#define KV_K2(i) vec2(k[i], k[(i) + 1u])
#define KV_V2(i) vec2(v[i], v[(i) + 1u])
#endif
