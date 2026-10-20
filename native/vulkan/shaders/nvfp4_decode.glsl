// NVFP4 (GGUF type 40, #1038) decode helpers shared by matmul_nvfp4_f32_gemv/gemm.
//
// Block = 36 bytes = 9 uints (always 4-byte aligned, so a row stride of blocks*36 is too):
//   word 0      = d[4]  — one unsigned UE4M3 scale byte per 16-element sub-block
//   words 1..8  = qs[32] — sub-block s owns qs[8s .. 8s+7]; byte j holds element 16s+j in its
//                 low nibble and element 16s+8+j in its high nibble (llama.cpp dequantize_row_nvfp4).
// value = KVALUES_FP4[nibble] * ue4m3_half(d[s]); the per-tensor X.scale multiplies the output.
// Requires a `weight[]` uint buffer declared by the including shader.

const uint NVFP4_BLOCK_UINTS = 9u;
const uint NVFP4_GROUP_SIZE  = 64u;

// E2M1 values doubled (ggml kvalues_mxfp4); the matching 0.5 lives in ue4m3_half.
const float KVALUES_FP4[16] = float[16](
     0.0,  1.0,  2.0,  3.0,  4.0,  6.0,  8.0,  12.0,
     0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0
);

// Literal port of ggml_ue4m3_to_fp32: 0x00 and 0x7F decode to 0, bit 7 is ignored (not a sign).
float ue4m3_half(uint x) {
    if (x == 0u || x == 0x7Fu) return 0.0;
    int e = int((x >> 3u) & 0xFu);
    float m = float(x & 7u);
    float raw = (e == 0) ? ldexp(m, -9) : ldexp(1.0 + m / 8.0, e - 7);
    return raw * 0.5;
}
