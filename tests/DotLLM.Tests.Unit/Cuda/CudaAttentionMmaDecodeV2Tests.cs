using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #988: parity + microbenchmark harness for the v2 (warp-independent, register-resident)
/// tensor-core decode attention kernel at the real Bonsai-2 27B shape (24 q-heads / 4 kv-heads /
/// head_dim 256). Parity references: the v1 kernel (<c>DOTLLM_ATTN_MMA_DECODE_V2=0</c> path) and an
/// F64 CPU reference over the same FP16-rounded inputs.
/// <para>
/// The microbenchmark is opt-in: set <c>DOTLLM_ATTN_MICROBENCH=1</c> (optional
/// <c>DOTLLM_ATTN_MICROBENCH_SEQKV=512,4096,16384,65536</c>, <c>DOTLLM_ATTN_MICROBENCH_ITERS</c>).
/// Run with <c>--logger "console;verbosity=detailed"</c> to see the table.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaAttentionMmaDecodeV2Tests
{
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;
    private const int Group = NumHeads / NumKvHeads;

    private readonly ITestOutputHelper _out;
    public CudaAttentionMmaDecodeV2Tests(ITestOutputHelper output) => _out = output;

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    private sealed class Rig : IDisposable
    {
        public CudaContext Ctx = null!;
        public CudaStream Stream = null!;
        public CudaKernels Kernels = null!;
        public CudaAttentionMmaDecodeGqaSplit Kernel = null!;
        public void Dispose() { Stream.Dispose(); Kernels.Dispose(); Ctx.Dispose(); }
    }

    private static Rig? TrySetUp(out string? skip)
    {
        skip = null;
        if (!IsCudaDriverPresent()) { skip = "No CUDA GPU available"; return null; }
        string? ptxDir = FindPtxDir();
        if (ptxDir == null) { skip = "PTX files not found"; return null; }
        var rig = new Rig { Ctx = CudaContext.Create(0), Stream = CudaStream.Create() };
        rig.Kernels = new CudaKernels(ptxDir);
        if (!rig.Kernels.HasAttentionMmaDecodeGqaSplit)
        {
            skip = "mma decode kernel not present";
            rig.Dispose();
            return null;
        }
        rig.Kernel = new CudaAttentionMmaDecodeGqaSplit(rig.Kernels);
        CudaAttentionMmaDecodeGqaSplit.Enabled = true;
        return rig;
    }

    private static ushort[] RandomHalf(Random rng, long n, float magnitude)
    {
        var h = new ushort[n];
        for (long i = 0; i < n; i++)
            h[i] = BitConverter.HalfToUInt16Bits((Half)((float)(rng.NextDouble() * 2.0 - 1.0) * magnitude));
        return h;
    }

    private static nint UploadF16(ushort[] host)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)((long)host.Length * 2)).ThrowOnError();
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)((long)host.Length * 2)).ThrowOnError();
        return d;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static float[] Download(nint src, int elems)
    {
        var host = new float[elems];
        fixed (float* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(elems * 4)).ThrowOnError();
        return host;
    }

    private static double[] CpuReference(ushort[] q, ushort[] k, ushort[] v, int seqKv)
    {
        var o = new double[NumHeads * HeadDim];
        var sc = new double[seqKv];
        int kvRow = NumKvHeads * HeadDim;
        for (int h = 0; h < NumHeads; h++)
        {
            int hk = h / Group;
            double mx = double.NegativeInfinity;
            for (int j = 0; j < seqKv; j++)
            {
                double a = 0;
                for (int d = 0; d < HeadDim; d++)
                    a += (double)(float)BitConverter.UInt16BitsToHalf(q[h * HeadDim + d])
                       * (double)(float)BitConverter.UInt16BitsToHalf(k[(long)j * kvRow + hk * HeadDim + d]);
                a /= 16.0;
                sc[j] = a;
                if (a > mx) mx = a;
            }
            double sum = 0;
            for (int j = 0; j < seqKv; j++) { sc[j] = Math.Exp(sc[j] - mx); sum += sc[j]; }
            for (int j = 0; j < seqKv; j++)
            {
                double w = sc[j] / sum;
                for (int d = 0; d < HeadDim; d++)
                    o[h * HeadDim + d] += w * (double)(float)BitConverter.UInt16BitsToHalf(v[(long)j * kvRow + hk * HeadDim + d]);
            }
        }
        return o;
    }

    private static float[] RunOnce(Rig rig, bool v2, nint dQ, nint dK, nint dV, int seqKv, int kvSplitOverride,
        out int kvSplitUsed)
    {
        bool prev = CudaKernels.MmaDecodeV2Enabled;
        CudaKernels.MmaDecodeV2Enabled = v2;
        nint dOut = 0, dPm = 0, dPs = 0, dPo = 0;
        try
        {
            int split = kvSplitOverride > 0 ? kvSplitOverride : rig.Kernel.ComputeSafeKvSplit(NumKvHeads, Group, seqKv);
            Assert.True(split >= 1);
            kvSplitUsed = split;
            dOut = Alloc(NumHeads * HeadDim * 4);
            dPm = Alloc((long)NumHeads * split * 4);
            dPs = Alloc((long)NumHeads * split * 4);
            dPo = Alloc((long)NumHeads * split * HeadDim * 4);
            rig.Kernel.Run(dQ, dK, dV, dOut, seqKv, NumHeads, NumKvHeads, split, dPm, dPs, dPo, rig.Stream.Handle);
            rig.Stream.Synchronize();
            return Download(dOut, NumHeads * HeadDim);
        }
        finally
        {
            CudaKernels.MmaDecodeV2Enabled = prev;
            foreach (var p in new[] { dOut, dPm, dPs, dPo }) if (p != 0) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    [SkippableTheory]
    [InlineData(1, 0)]
    [InlineData(15, 0)]
    [InlineData(16, 0)]
    [InlineData(17, 0)]
    [InlineData(255, 0)]
    [InlineData(256, 0)]
    [InlineData(4096, 0)]
    [InlineData(16385, 0)]
    [InlineData(4096, 1)]
    [InlineData(2051, 3)]
    public void V2_MatchesV1AndF64Reference_AtBonsaiShape(int seqKv, int forcedSplit)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        Skip.IfNot(CudaKernels.MmaDecodeV2Enabled && rig!.Kernels.HasAttentionMmaDecodeGqaSplit, "v2 disabled");
        bool prev = CudaKernels.MmaDecodeV2Enabled;
        try
        {
            CudaKernels.MmaDecodeV2Enabled = true;
            Skip.IfNot(rig!.Kernels.MmaDecodeV2Active, "v2 kernel not loaded (stale PTX)");
            var rng = new Random(0x988 ^ seqKv);
            var q = RandomHalf(rng, NumHeads * HeadDim, 1.0f);
            var k = RandomHalf(rng, (long)seqKv * NumKvHeads * HeadDim, 1.0f);
            var v = RandomHalf(rng, (long)seqKv * NumKvHeads * HeadDim, 1.0f);
            nint dQ = UploadF16(q), dK = UploadF16(k), dV = UploadF16(v);
            try
            {
                int s1, s2;
                var o1 = RunOnce(rig, false, dQ, dK, dV, seqKv, forcedSplit, out s1);
                var o2 = RunOnce(rig, true, dQ, dK, dV, seqKv, forcedSplit, out s2);
                var r = CpuReference(q, k, v, seqKv);
                double d12 = 0, d1r = 0, d2r = 0;
                for (int i = 0; i < r.Length; i++)
                {
                    d12 = Math.Max(d12, Math.Abs(o1[i] - o2[i]));
                    d1r = Math.Max(d1r, Math.Abs(o1[i] - r[i]));
                    d2r = Math.Max(d2r, Math.Abs(o2[i] - r[i]));
                    Assert.True(Math.Abs(o2[i] - r[i]) <= 5e-3 + 5e-3 * Math.Abs(r[i]),
                        $"v2 deviates from F64 ref at {i}: {o2[i]} vs {r[i]} (seqKv={seqKv})");
                }
                _out.WriteLine($"seqKv={seqKv,6} split v1={s1} v2={s2}  max|v1-v2|={d12:E2}  max|v1-ref|={d1r:E2}  max|v2-ref|={d2r:E2}");
                Assert.True(d12 < 5e-3, $"v1 vs v2 diverge: {d12}");
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(dQ); CudaDriverApi.cuMemFree_v2(dK); CudaDriverApi.cuMemFree_v2(dV);
            }
        }
        finally { CudaKernels.MmaDecodeV2Enabled = prev; }
    }

    [SkippableFact]
    public void Microbench_V1_vs_V2()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH") == "1", "set DOTLLM_ATTN_MICROBENCH=1");
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        var seqList = (Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_SEQKV") ?? "512,4096,16384,65536")
            .Split(',').Select(int.Parse).ToArray();
        int iters = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_ITERS"), out int it) ? it : 48;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("| seqKv | variant | kvSplit | ms/launch | eff GB/s |");
        sb.AppendLine("|---:|---|---:|---:|---:|");
        foreach (int seqKv in seqList)
        {
            long kvElems = (long)seqKv * NumKvHeads * HeadDim;
            long perCopy = kvElems * 2 * 2;
            int copies = (int)Math.Clamp(700L * 1024 * 1024 / perCopy, 1, 16);
            var rng = new Random(seqKv);
            var q = RandomHalf(rng, NumHeads * HeadDim, 1.0f);
            var k = RandomHalf(rng, kvElems, 1.0f);
            var v = RandomHalf(rng, kvElems, 1.0f);
            nint dQ = UploadF16(q);
            var dK = new nint[copies]; var dV = new nint[copies];
            for (int c = 0; c < copies; c++) { dK[c] = UploadF16(k); dV[c] = UploadF16(v); }
            nint dOut = Alloc(NumHeads * HeadDim * 4);
            nint dPm = Alloc(NumHeads * 64 * 4), dPs = Alloc(NumHeads * 64 * 4), dPo = Alloc((long)NumHeads * 64 * HeadDim * 4);
            try
            {
                foreach (bool v2 in new[] { false, true })
                {
                    bool prev = CudaKernels.MmaDecodeV2Enabled;
                    CudaKernels.MmaDecodeV2Enabled = v2;
                    try
                    {
                        if (v2 && !rig!.Kernels.MmaDecodeV2Active) continue;
                        int split = rig!.Kernel.ComputeSafeKvSplit(NumKvHeads, Group, seqKv);
                        var splits = new List<int> { split };
                        if (v2 && Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_SWEEP") == "1")
                            { int ms = rig.Kernels.MaxSafeAttentionMmaDecodeGqaSplit(NumKvHeads, HeadDim, Group); foreach (int sp in new[] { 1, 2, 3, 4, 5, 6, 7, 8, 10, 14 }) if (sp != split && sp <= ms) splits.Add(sp); }
                        foreach (int sp in splits)
                        {
                            double best = double.MaxValue;
                            for (int rep = 0; rep < 4; rep++)
                            {
                                for (int w = 0; w < 4; w++)
                                    rig.Kernel.Run(dQ, dK[w % copies], dV[w % copies], dOut, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                                rig.Stream.Synchronize();
                                var sw = Stopwatch.StartNew();
                                for (int i = 0; i < iters; i++)
                                    rig.Kernel.Run(dQ, dK[i % copies], dV[i % copies], dOut, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                                rig.Stream.Synchronize();
                                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
                            }
                            double gbps = (double)kvElems * 2 * 2 / (best * 1e-3) / 1e9;
                            sb.AppendLine($"| {seqKv} | {(v2 ? "v2" : "v1")} | {sp} | {best:F3} | {gbps:F0} |");
                        }
                    }
                    finally { CudaKernels.MmaDecodeV2Enabled = prev; }
                }
            }
            finally
            {
                for (int c = 0; c < copies; c++) { CudaDriverApi.cuMemFree_v2(dK[c]); CudaDriverApi.cuMemFree_v2(dV[c]); }
                foreach (var p in new[] { dQ, dOut, dPm, dPs, dPo }) CudaDriverApi.cuMemFree_v2(p);
            }
        }
        _out.WriteLine(sb.ToString());
        string? outFile = Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_OUT");
        if (!string.IsNullOrEmpty(outFile)) File.AppendAllText(outFile, sb.ToString());
    }

    // ── Quantized-KV (fused) coverage -- issue #988 step 3 ──────────────────────────────────

    private static int BlockBytes(KvCacheDType t) => t == KvCacheDType.Q8_0 ? 34 : 18;

    /// <summary>Random KV-format blocks for <paramref name="elems"/> logical elements (multiple of 32).</summary>
    private static byte[] RandomQuantBlocks(Random rng, long elems, KvCacheDType t)
    {
        long blocks = elems / 32;
        int bb = BlockBytes(t);
        var buf = new byte[blocks * bb];
        for (long b = 0; b < blocks; b++)
        {
            ushort sc = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.03f));
            buf[b * bb] = (byte)(sc & 0xFF);
            buf[b * bb + 1] = (byte)(sc >> 8);
            for (int i = 2; i < bb; i++) buf[b * bb + i] = (byte)rng.Next(256);
        }
        return buf;
    }

    private static nint UploadBytes(byte[] host)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)host.Length).ThrowOnError();
        fixed (byte* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)host.Length).ThrowOnError();
        return d;
    }

    private static readonly (KvCacheDType K, KvCacheDType V)[] KvqCombos =
    {
        (KvCacheDType.Q8_0, KvCacheDType.Q8_0), (KvCacheDType.Q4_0, KvCacheDType.Q4_0),
        (KvCacheDType.Q8_0, KvCacheDType.Q4_0), (KvCacheDType.Q4_0, KvCacheDType.Q8_0),
    };

    [SkippableTheory]
    [InlineData(17)]
    [InlineData(255)]
    [InlineData(4096)]
    [InlineData(16385)]
    public void Kvq_FusedBitIdenticalToScratchDequant_V2(int seqKv)
    {
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        bool prev = CudaKernels.MmaDecodeV2Enabled;
        try
        {
            CudaKernels.MmaDecodeV2Enabled = true;
            Skip.IfNot(rig!.Kernels.MmaDecodeV2Active, "v2 kernel not loaded");
            long kvElems = (long)seqKv * NumKvHeads * HeadDim;
            foreach (var (kd, vd) in KvqCombos)
            {
                Skip.IfNot(rig.Kernels.HasAttentionMmaDecodeKvq(kd, vd), "kvq kernel not loaded");
                var rng = new Random(0x988 + seqKv + (int)kd * 7 + (int)vd);
                var q = RandomHalf(rng, NumHeads * HeadDim, 1.0f);
                nint dQ = UploadF16(q);
                nint dKq = UploadBytes(RandomQuantBlocks(rng, kvElems, kd));
                nint dVq = UploadBytes(RandomQuantBlocks(rng, kvElems, vd));
                nint dKs = Alloc(kvElems * 2), dVs = Alloc(kvElems * 2);
                try
                {
                    rig.Kernels.LaunchKvDequantToF16(dKq, kd, dKs, (int)kvElems, rig.Stream.Handle);
                    rig.Kernels.LaunchKvDequantToF16(dVq, vd, dVs, (int)kvElems, rig.Stream.Handle);
                    rig.Stream.Synchronize();
                    int split = rig.Kernel.ComputeSafeKvSplitKvq(kd, vd, NumKvHeads, seqKv);
                    Assert.True(split >= 1);
                    foreach (bool v2 in new[] { true, false })
                    {
                        CudaKernels.MmaDecodeV2Enabled = v2;
                        int sp = rig.Kernel.ComputeSafeKvSplitKvq(kd, vd, NumKvHeads, seqKv);
                        nint dOutS = Alloc(NumHeads * HeadDim * 4), dOutF = Alloc(NumHeads * HeadDim * 4);
                        nint dPm = Alloc(NumHeads * sp * 4), dPs = Alloc(NumHeads * sp * 4), dPo = Alloc((long)NumHeads * sp * HeadDim * 4);
                        try
                        {
                            rig.Kernel.Run(dQ, dKs, dVs, dOutS, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                            rig.Kernel.RunKvq(dQ, dKq, dVq, kd, vd, dOutF, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                            rig.Stream.Synchronize();
                            var a = Download(dOutS, NumHeads * HeadDim);
                            var b = Download(dOutF, NumHeads * HeadDim);
                            int diff = 0;
                            for (int i = 0; i < a.Length; i++)
                                if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i])) diff++;
                            _out.WriteLine($"seqKv={seqKv} {kd}/{vd} v2={v2} split={sp}: fused vs scratch bit-diff elements = {diff}");
                            Assert.Equal(0, diff);
                        }
                        finally
                        {
                            foreach (var pp in new[] { dOutS, dOutF, dPm, dPs, dPo }) CudaDriverApi.cuMemFree_v2(pp);
                        }
                    }
                }
                finally
                {
                    foreach (var pp in new[] { dQ, dKq, dVq, dKs, dVs }) CudaDriverApi.cuMemFree_v2(pp);
                }
            }
        }
        finally { CudaKernels.MmaDecodeV2Enabled = prev; }
    }

    [SkippableFact]
    public void Microbench_Kvq_Fused_vs_Scratch()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH") == "1", "set DOTLLM_ATTN_MICROBENCH=1");
        using var rig = TrySetUp(out string? skip);
        Skip.If(rig == null, skip);
        var seqList = (Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_SEQKV") ?? "4096,16384,65536")
            .Split(',').Select(int.Parse).ToArray();
        int iters = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_ATTN_MICROBENCH_ITERS"), out int it) ? it : 24;
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("| seqKv | KV fmt | path | kvSplit | ms/launch |");
        sb.AppendLine("|---:|---|---|---:|---:|");
        foreach (int seqKv in seqList)
        {
            long kvElems = (long)seqKv * NumKvHeads * HeadDim;
            var rng = new Random(seqKv);
            var q = RandomHalf(rng, NumHeads * HeadDim, 1.0f);
            nint dQ = UploadF16(q);
            nint dOut = Alloc(NumHeads * HeadDim * 4);
            nint dPm = Alloc(NumHeads * 64 * 4), dPs = Alloc(NumHeads * 64 * 4), dPo = Alloc((long)NumHeads * 64 * HeadDim * 4);
            nint dKs = Alloc(kvElems * 2), dVs = Alloc(kvElems * 2);
            try
            {
                foreach (var kd in new[] { KvCacheDType.Q8_0, KvCacheDType.Q4_0 })
                {
                    nint dKq = UploadBytes(RandomQuantBlocks(rng, kvElems, kd));
                    nint dVq = UploadBytes(RandomQuantBlocks(rng, kvElems, kd));
                    try
                    {
                        foreach (string path in new[] { "scratch+f16-v1", "scratch+f16-v2", "fused-v1", "fused-v2" })
                        {
                            bool v2 = path.EndsWith("v2");
                            bool fused = path.StartsWith("fused");
                            bool prev = CudaKernels.MmaDecodeV2Enabled;
                            CudaKernels.MmaDecodeV2Enabled = v2;
                            try
                            {
                                if (v2 && !rig!.Kernels.MmaDecodeV2Active) continue;
                                int sp = fused ? rig!.Kernel.ComputeSafeKvSplitKvq(kd, kd, NumKvHeads, seqKv)
                                               : rig!.Kernel.ComputeSafeKvSplit(NumKvHeads, Group, seqKv);
                                double best = double.MaxValue;
                                for (int rep = 0; rep < 3; rep++)
                                {
                                    var sw = Stopwatch.StartNew();
                                    for (int i = 0; i < iters; i++)
                                    {
                                        if (fused)
                                            rig.Kernel.RunKvq(dQ, dKq, dVq, kd, kd, dOut, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                                        else
                                        {
                                            rig.Kernels.LaunchKvDequantToF16(dKq, kd, dKs, (int)kvElems, rig.Stream.Handle);
                                            rig.Kernels.LaunchKvDequantToF16(dVq, kd, dVs, (int)kvElems, rig.Stream.Handle);
                                            rig.Kernel.Run(dQ, dKs, dVs, dOut, seqKv, NumHeads, NumKvHeads, sp, dPm, dPs, dPo, rig.Stream.Handle);
                                        }
                                    }
                                    rig.Stream.Synchronize();
                                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
                                }
                                sb.AppendLine($"| {seqKv} | {kd} | {path} | {sp} | {best:F3} |");
                            }
                            finally { CudaKernels.MmaDecodeV2Enabled = prev; }
                        }
                    }
                    finally { CudaDriverApi.cuMemFree_v2(dKq); CudaDriverApi.cuMemFree_v2(dVq); }
                }
            }
            finally
            {
                foreach (var pp in new[] { dQ, dOut, dPm, dPs, dPo, dKs, dVs }) CudaDriverApi.cuMemFree_v2(pp);
            }
        }
        _out.WriteLine(sb.ToString());
    }
}
