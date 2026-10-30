// block_q8_1 keeps its (d, s) pair in fp16; the CPU Q8_1 dot (#1159) and llama.cpp multiply by the rounded values. dotLLM's Q8_1
// quantizers store F32, so the Q4_1 MMQs (#1235) round at staging. Done on the bits (round to nearest even at 10 mantissa bits,
// fp16 subnormal spacing below 2^-14) because drivers fold unpackHalf2x16(packHalf2x16(x)) to x (measured on NVIDIA). Values past
// the fp16 range stay finite here, where the CPU's would become inf.
float roundToF16(float x) {
    if (abs(x) < 6.103515625e-05) return roundEven(x * 16777216.0) * (1.0 / 16777216.0);
    uint u = floatBitsToUint(x);
    return uintBitsToFloat((u + 0x0FFFu + ((u >> 13u) & 1u)) & 0xFFFFE000u);
}

vec2 roundToF16(vec2 v) { return vec2(roundToF16(v.x), roundToF16(v.y)); }
