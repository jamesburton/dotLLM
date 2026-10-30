// Shared by gumbel_argmax_f32.comp and gumbel_noise_dump_f32.comp (#1382). Mirrors DotLLM.Engine.Samplers.GumbelNoise bit for bit.
const uint M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
const uint W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;

const float P0 = 1.0, P1 = -0.49999988, P2 = 0.33333325, P3 = -0.25001585, P4 = 0.20002072;
const float P5 = -0.16609126, P6 = 0.14180080, P7 = -0.13240842, P8 = 0.12914170, P9 = -0.07638592;
const float LN2HI = 6.9313812256e-01, LN2LO = 9.0580006145e-06, SQRT2 = 1.41421356;

uint philoxWord(uint seedLo, uint seedHi, uint tokenIndex, uint index) {
    uint c0 = index, c1 = tokenIndex, c2 = 0u, c3 = 0u;
    uint k0 = seedLo, k1 = seedHi;
    for (int r = 0; r < 10; r++) {
        uint hi0, lo0, hi1, lo1;
        umulExtended(M0, c0, hi0, lo0);
        umulExtended(M1, c2, hi1, lo1);
        uint n0 = hi1 ^ c1 ^ k0;
        uint n1 = lo1;
        uint n2 = hi0 ^ c3 ^ k1;
        uint n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += W0; k1 += W1;
    }
    return c0;
}

// GumbelNoise.Log for a positive normal x.
precise float plog(float x) {
    uint bits = floatBitsToUint(x);
    int e = int(bits >> 23) - 127;
    precise float m = uintBitsToFloat((bits & 0x007FFFFFu) | 0x3F800000u);
    if (m > SQRT2) { m = m * 0.5; e++; }
    precise float f = m - 1.0;
    precise float acc = P9;
    acc = acc * f + P8; acc = acc * f + P7; acc = acc * f + P6; acc = acc * f + P5;
    acc = acc * f + P4; acc = acc * f + P3; acc = acc * f + P2; acc = acc * f + P1; acc = acc * f + P0;
    precise float l1p = f * acc;
    precise float de = float(e);
    precise float hi = de * LN2HI;
    precise float lo = de * LN2LO;
    precise float tail = l1p + lo;
    precise float r = hi + tail;
    return r;
}

// Gumbel(0,1) noise for a Philox output word: GumbelNoise.FromUniform(GumbelNoise.Uniform24(w)), bit for bit.
precise float gumbelFromWord(uint w) {
    uint mnt = max(w >> 8, 1u);
    precise float u = float(mnt) * (1.0 / 16777216.0);
    precise float a = -plog(u);
    precise float g = -plog(a);
    return g;
}

// G(seed, tokenIndex, index): GumbelNoise.Noise.
precise float gumbelNoise(uint seedLo, uint seedHi, uint tokenIndex, uint index) {
    return gumbelFromWord(philoxWord(seedLo, seedHi, tokenIndex, index));
}
