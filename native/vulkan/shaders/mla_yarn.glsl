// YaRN-ramped RoPE inverse frequency for the MLA decoupled-rope slice (#1278). Mirrors RoPE.ComputeYarnInverseFrequencies
// (CPU): inv = inter * ramp + extra * (1 - ramp), extra = theta^-(2i/d), inter = 1 / (factor * theta^(2i/d)),
// ramp = clamp((i - low) / (high - low), 0, 1). factor <= 1 returns the plain extra frequency (legacy path, bit-identical).
float mlaRopeFreq(uint pair, uint ropeDim, float theta, float yarnFactor, float yarnLow, float yarnHigh) {
    float expn = float(2u * pair) / float(ropeDim);
    float p = pow(theta, expn);
    float freqExtra = 1.0 / p;
    if (yarnFactor <= 1.0) return freqExtra;
    float freqInter = 1.0 / (yarnFactor * p);
    float ramp = clamp((float(pair) - yarnLow) / (yarnHigh - yarnLow), 0.0, 1.0);
    return freqInter * ramp + freqExtra * (1.0 - ramp);
}
