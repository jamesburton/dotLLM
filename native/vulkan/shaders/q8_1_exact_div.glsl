// Correctly-rounded (IEEE) scale arithmetic for the Q8_1 activation quantizers (#1340).
//
// Contract shared with the CPU reference (MatMul.QuantizeF32ToQ8_1*, == llama.cpp quantize_row_q8_1_ref):
//   d  = amax / 127        (IEEE division)
//   id = 1 / d             (IEEE reciprocal)
//   q  = roundHalfToEven(x * id), clamped to [-127, 127]
//
// RDNA has no divider: a plain GLSL `a / 127.0` or `1.0 / d` lowers to a multiply by an approximate reciprocal (Vulkan permits 2.5 ULP),
// so `d` (or `id`) lands one ULP off the CPU value and the lanes that sit on a rounding boundary flip by one integer - which then moves
// every output row of the next projection. Both helpers below use the Markstein correction (one fma residual step) so the result is the
// IEEE-correct quotient regardless of how the driver lowers the first approximation. `precise` stops the compiler from folding the
// correction away or re-associating it.

// 1/127 rounded to nearest float (0x3c010204).
const float Q81_RCP_127 = 0.007874016;   // shortest decimal that round-trips to 0x3c010204

float q81DivBy127(float a) {
    // Residual of the first quotient without relying on a fused fma (the driver may lower fma(-q, 127, a) to mul+add, which makes the
    // residual vanish and silently returns the unrefined quotient - measured on gfx1151): 127 * q = 128 * q - q, and 128 * q is exact,
    // so a fast-two-sum of (128q, -q) gives 127 * q = s + err exactly, and a - s is exact (Sterbenz).
    precise float q = a * Q81_RCP_127;
    precise float p = q * 128.0;
    precise float s = p - q;
    precise float z = s - p;
    precise float err = -q - z;
    precise float r = (a - s) - err;
    precise float rq = r * Q81_RCP_127;
    precise float q2 = q + rq;
    return q2;
}
float q81Recip(float d) {
    precise float y0 = 1.0 / d;
    precise float e = fma(-d, y0, 1.0);
    precise float y1 = fma(y0, e, y0);
    return y1;
}
