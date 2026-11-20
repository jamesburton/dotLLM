using System;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.KvCache.Codecs;
using DotLLM.Tests.Unit.Engine.KvCache;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity test for <see cref="VulkanTurboQuantKvCache"/> against the CPU <see cref="TurboQuantKvCache"/>.
/// Both store the same head-vectors as TurboQuant MSE codes + norm and dequantize to fp32; the GPU
/// path (encode shader → dequant shader) should reproduce the CPU path (codec encode → decode) to
/// within tight float tolerance — the encode kernel matched CPU codes 100% and the dequant kernel is
/// bit-exact, so the reconstructed K/V should match. A wrong addressing / seed / scratch bug in the
/// cache would diverge by O(norm).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanTurboQuantKvCacheTests
{
    private const ulong Seed = 0xC0FFEE_4B2CUL;
    private const ulong VSeedXor = 0xD1B54A32D192ED03UL; // must match TurboQuantKvCache's V-seed derivation

    private readonly ITestOutputHelper _output;
    public VulkanTurboQuantKvCacheTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(64, 4, 3, 2, 5)]    // headDim, bits, numKvHeads, numLayers, seqLen
    [InlineData(128, 4, 8, 2, 6)]   // Llama-3.1-8B geometry
    [InlineData(128, 3, 2, 1, 4)]
    public void GpuCache_MatchesCpuCache(int headDim, int bits, int numKvHeads, int numLayers, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int stride = numKvHeads * headDim;
        int maxSeqLen = seqLen + 2;
        var rng = new Random(0xCA5_E77 ^ (headDim * 7 + bits * 3 + numKvHeads + numLayers));

        // Codec constants (K and V use independent rotations, same as the CPU cache derives).
        var codecK = new TurboQuantCodec(headDim, bits, Seed, useQjl: false);
        var codecV = new TurboQuantCodec(headDim, bits, Seed ^ VSeedXor, useQjl: false);

        using var cpu = new TurboQuantKvCache(numLayers, numKvHeads, headDim, maxSeqLen, bits, Seed, useQjl: false);
        using var device = VulkanDevice.Create();
        using var gpu = new VulkanTurboQuantKvCache(
            device, spvDir, numLayers, numKvHeads, headDim, maxSeqLen, codecK.MseBits,
            codecK.Centroids, codecK.RotationSigns, codecV.RotationSigns, codecK.InvSqrtD);

        var positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        double globalMax = 0;
        long mism = 0, total = 0;
        double sumAbs = 0;

        for (int layer = 0; layer < numLayers; layer++)
        {
            float[] hostK = RandomRows(rng, seqLen, stride, numKvHeads, headDim, salt: layer * 2);
            float[] hostV = RandomRows(rng, seqLen, stride, numKvHeads, headDim, salt: layer * 2 + 1);

            // CPU cache update + reconstruct.
            float[] expK = new float[seqLen * stride];
            float[] expV = new float[seqLen * stride];
            fixed (float* pk = hostK)
            fixed (float* pv = hostV)
            {
                cpu.Update(new TensorRef(seqLen, stride, DType.Float32, -1, (nint)pk),
                           new TensorRef(seqLen, stride, DType.Float32, -1, (nint)pv), positions, layer);
                var kRef = cpu.GetKeysRef(layer);
                var vRef = cpu.GetValuesRef(layer);
                new ReadOnlySpan<float>((void*)kRef.DataPointer, seqLen * stride).CopyTo(expK);
                new ReadOnlySpan<float>((void*)vRef.DataPointer, seqLen * stride).CopyTo(expV);
            }

            // GPU cache update (upload activations to device) + reconstruct.
            using var kDev = device.Allocate((long)seqLen * stride * sizeof(float));
            using var vDev = device.Allocate((long)seqLen * stride * sizeof(float));
            device.Upload(new ReadOnlySpan<float>(hostK), kDev);
            device.Upload(new ReadOnlySpan<float>(hostV), vDev);
            gpu.UpdateSync(kDev, vDev, positions, seqLen, layer);

            var gotK = new float[seqLen * stride];
            var gotV = new float[seqLen * stride];
            gpu.DequantKeysToHost(layer, gotK);
            gpu.DequantValuesToHost(layer, gotV);

            (double mx, long mm, double sa) = Compare(expK, gotK);
            globalMax = Math.Max(globalMax, mx); mism += mm; sumAbs += sa;
            (mx, mm, sa) = Compare(expV, gotV);
            globalMax = Math.Max(globalMax, mx); mism += mm; sumAbs += sa;
            total += 2L * seqLen * stride;
        }

        double mismFrac = (double)mism / total;
        double meanAbs = sumAbs / total;
        _output.WriteLine($"d={headDim} b={bits} heads={numKvHeads} layers={numLayers} seq={seqLen}: " +
                          $"maxAbs={globalMax:E3} meanAbs={meanAbs:E3} mismatchFrac={mismFrac:P3}");

        // codes match CPU 100% + dequant is bit-exact ⇒ expect near-exact; allow a tiny margin for
        // rare encode boundary flips.
        Assert.True(mismFrac < 0.005, $"too many element mismatches: {mismFrac:P3}");
        Assert.True(meanAbs < 1e-3, $"mean abs diff too high: {meanAbs:E3}");
    }

    /// <summary>
    /// #1892: stored norm bytes equal <see cref="TurboQuantNormCodec.Encode(TqNormFormat, ReadOnlySpan{float}, Span{byte})"/> of the
    /// encode-time norm, in every format the GPU supports, and the dequant reads them back.
    /// </summary>
    [SkippableTheory]
    [InlineData(TqNormFormat.F32Reference)]
    [InlineData(TqNormFormat.F16)]
    [InlineData(TqNormFormat.Bf16)]
    public void Oracle_NormsRoundTripThroughTheSharedCodec(TqNormFormat format)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int hd = 128, bits = 4;
        var corpus = TurboQuantByteExactCorpus.Build(bits, hd, halfInputs: false);
        var (_, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(corpus.K, format);

        using var device = VulkanDevice.Create();
        var ck = corpus.K.Codec; var cv = corpus.V.Codec;
        using var gpu = new VulkanTurboQuantKvCache(device, spvDir, 1, TurboQuantByteExactCorpus.NumKvHeads, hd,
            TurboQuantByteExactCorpus.SeqLen, ck.MseBits, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD, format);
        Assert.Equal(format, gpu.NormFormat);
        Upload(device, corpus, out var kDev, out var vDev);
        using (kDev) using (vDev)
        {
            var positions = new int[TurboQuantByteExactCorpus.SeqLen];
            for (int i = 0; i < positions.Length; i++) positions[i] = i;
            gpu.UpdateSync(kDev, vDev, positions, positions.Length, 0);

            var got = new byte[cpuNorms.Length];
            gpu.DownloadNormBytes(0, values: false, got);
            Assert.Equal(cpuNorms, got);

            // and the dequant decodes them. Recover the norm the SHADER applied directly: the scratch is norm * unitDirection, so
            // at the coordinate where the CPU unit direction is largest, scratch / unit is the decoded norm to ~1e-7. That is
            // compared against TurboQuantNormCodec.Decode at a relative 1e-5, which a flushed F16 subnormal norm (decoded 0, a 100%
            // error) cannot pass. The former 1e-4 absolute-scaled dequant comparison could not see that, because a subnormal norm
            // contributes < 1e-4 to every element either way.
            var recon = new float[positions.Length * TurboQuantByteExactCorpus.NumKvHeads * hd];
            gpu.DequantKeysToHost(0, recon);
            int n = TurboQuantByteExactCorpus.SeqLen * TurboQuantByteExactCorpus.NumKvHeads;
            var codes = new byte[ck.CodeBytesPerVector];
            var unit = new float[hd];
            int subnormals = 0;
            for (int v = 0; v < n; v++)
            {
                float norm = TurboQuantNormCodec.Decode(format, cpuNorms, v);
                ck.Encode(corpus.K.Data.AsSpan(v * hd, hd), codes);
                ck.Decode(codes, 1.0f, unit);
                int j = 0;
                for (int i = 1; i < hd; i++) if (Math.Abs(unit[i]) > Math.Abs(unit[j])) j = i;
                float shaderNorm = recon[v * hd + j] / unit[j];
                Assert.True(Math.Abs(shaderNorm - norm) <= 1e-5f * norm,
                    $"v={v}: decoded norm cpu={norm:R} shader={shaderNorm:R}");
                if (format == TqNormFormat.F16 && norm > 0f && norm < 6.103515625e-5f) subnormals++;
            }
            if (format == TqNormFormat.F16)
                Assert.True(subnormals > 0, "the corpus has no F16-subnormal norm, so a flush-to-zero decode would pass unseen");
        }
    }

    /// <summary>
    /// #1892 gate: encode is byte-identical to the CPU codec (codes AND F16 norm bytes) on random K/V with exact
    /// F32 distance ties injected after normalise+rotate. CUDA is checked against the same CPU reference on framework.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(ByteExactConfigs))]
    public void Encode_IsByteIdenticalToCpu_Vulkan(int bits, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var corpus = TurboQuantByteExactCorpus.Build(bits, headDim, halfInputs: false);
        var format = TurboQuantLayout.DefaultNorm;

        using var device = VulkanDevice.Create();
        var ck = corpus.K.Codec; var cv = corpus.V.Codec;
        using var gpu = new VulkanTurboQuantKvCache(device, spvDir, 1, TurboQuantByteExactCorpus.NumKvHeads, headDim,
            TurboQuantByteExactCorpus.SeqLen, ck.MseBits, ck.Centroids, ck.RotationSigns, cv.RotationSigns, ck.InvSqrtD);
        Assert.Equal(TqNormFormat.F16, gpu.NormFormat);
        Upload(device, corpus, out var kDev, out var vDev);
        using (kDev) using (vDev)
        {
            int seq = TurboQuantByteExactCorpus.SeqLen;
            var positions = new int[seq];
            for (int i = 0; i < seq; i++) positions[i] = i;
            gpu.UpdateSync(kDev, vDev, positions, seq, 0);

            int n = seq * TurboQuantByteExactCorpus.NumKvHeads;
            int cu = gpu.CodeUintsPerVec;
            int nb = TurboQuantNormCodec.BytesPerNorm(format);
            string failures = "";
            foreach (bool values in new[] { false, true })
            {
                var side = values ? corpus.V : corpus.K;
                var (cpuCodes, cpuNorms) = TurboQuantByteExactCorpus.CpuEncode(side, format);
                var gotCodes = new uint[n * cu];
                var gotNorms = new byte[n * nb];
                gpu.DownloadCodeUints(0, values, gotCodes);
                gpu.DownloadNormBytes(0, values, gotNorms);
                var (cb, nbad, total) = TurboQuantByteExactCorpus.Compare(cpuCodes, cpuNorms,
                    MemoryMarshal.AsBytes(gotCodes.AsSpan()), gotNorms, cu * 4, nb);
                _output.WriteLine($"VULKAN bits={bits} hd={headDim} {(values ? "V" : "K")}: code mismatches {cb}/{total}, norm mismatches {nbad}/{total} (tie coords in corpus: {side.TieCoords})");
                if (cb != 0 || nbad != 0) failures += $"{(values ? "V" : "K")}: {cb} code / {nbad} norm of {total}; ";
            }
            Assert.True(failures.Length == 0, $"Vulkan encode differs from the CPU codec at bits={bits} hd={headDim}: {failures}");
        }
    }

    public static System.Collections.Generic.IEnumerable<object[]> ByteExactConfigs() => TurboQuantByteExactCorpus.BitsAndHeadDims();

    private static void Upload(VulkanDevice device, TurboQuantByteExactCorpus.Corpus corpus,
                               out VulkanDevice.Buffer kDev, out VulkanDevice.Buffer vDev)
    {
        kDev = device.Allocate((long)corpus.K.Data.Length * sizeof(float));
        vDev = device.Allocate((long)corpus.V.Data.Length * sizeof(float));
        device.Upload(new ReadOnlySpan<float>(corpus.K.Data), kDev);
        device.Upload(new ReadOnlySpan<float>(corpus.V.Data), vDev);
    }

    private static (double maxAbs, long mismatches, double sumAbs) Compare(float[] expected, float[] actual)
    {
        double maxAbs = 0, sumAbs = 0;
        long mismatches = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = Math.Abs((double)expected[i] - actual[i]);
            sumAbs += d;
            if (d > maxAbs) maxAbs = d;
            if (d > 2e-3 + 1e-4 * Math.Abs(expected[i])) mismatches++;
        }
        return (maxAbs, mismatches, sumAbs);
    }

    // seqLen rows of [stride]; each head a distinct-norm Gaussian direction.
    private static float[] RandomRows(Random rng, int seqLen, int stride, int numKvHeads, int headDim, int salt)
    {
        var buf = new float[seqLen * stride];
        for (int pos = 0; pos < seqLen; pos++)
            for (int h = 0; h < numKvHeads; h++)
            {
                float target = (pos + 1) * 9f + h * 2.5f + 1f + salt;
                int off = pos * stride + h * headDim;
                double sq = 0;
                for (int i = 0; i < headDim; i++)
                {
                    double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
                    float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
                    buf[off + i] = g; sq += (double)g * g;
                }
                float scale = target / (float)Math.Sqrt(sq);
                for (int i = 0; i < headDim; i++) buf[off + i] *= scale;
            }
        return buf;
    }
}
