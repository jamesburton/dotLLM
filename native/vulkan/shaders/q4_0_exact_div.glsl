// Correctly-rounded (IEEE) scale for the Q4_0 KV quantizer (#1500): d = amax / 7.
//
// Same construction as q81DivBy127 in q8_1_exact_div.glsl (read that file for the why): 7 * q = 8 * q - q, and 8 * q is
// exact, so a fast-two-sum of (8q, -q) gives the residual of the first quotient without relying on a fused fma. Kept in
// its own include so the Q8_1 shaders' SPIR-V stays byte-identical (an extra global const shifts every id in a module).
// The reciprocal helper q81Recip is shared; include q8_1_exact_div.glsl first.

// 1/7 rounded to nearest float (0x3e124925).
const float Q40_RCP_7 = 0.14285715;   // shortest decimal that round-trips to 0x3e124925

float q40DivBy7(float a) {
    precise float q = a * Q40_RCP_7;
    precise float p = q * 8.0;
    precise float s = p - q;
    precise float z = s - p;
    precise float err = -q - z;
    precise float r = (a - s) - err;
    precise float rq = r * Q40_RCP_7;
    precise float q2 = q + rq;
    return q2;
}
