using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Engine.KvCache.Codecs;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// #1924 (James 2026-11-22): the tensor-core TurboQuant prefill entries must not read a key row past <c>seq_kv</c>. Every other fixture
/// allocates the cache with spare capacity, so a missing tail clamp reads finite rows inside the allocation whose keys are then masked:
/// the mutants "staged code-row clamp removed" (m10) and "prefetched norm clamp removed" (m11) survived the whole suite. Here the K/V
/// buffers hold exactly <c>seq_kv</c> rows (not a multiple of 16, so the last tile is partial), and:
/// <list type="bullet">
/// <item><see cref="FlashPrefill_ExactSizeCache_PoisonedTail_IsIgnored"/>: everything past row seq_kv is poisoned (NaN norms, all-ones
/// codes). The output must be finite and bit-identical to the same launch with a zero tail. A norm read past the end turns
/// p * n_v = 0 * NaN into NaN.</item>
/// <item><see cref="FlashPrefill_CodesEndAtUnmappedPage_NoOutOfBoundsRead"/>: the K and V code buffers end 16 bytes before an unmapped
/// page (CUDA virtual memory management: one mapped granule at the start of a two-granule reservation). Any code row read past seq_kv
/// faults (CUDA_ERROR_ILLEGAL_ADDRESS at the synchronize). The 16 bytes cover TurboQuantLayout's guard word.</item>
/// </list>
/// Every entry is run: Legacy, Specialized (E1) and Staged (E3), forced through <see cref="CudaKernels.TqPrefillVariantOverride"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe partial class CudaTurboQuantPrefillTailTests : IDisposable
{
    private const ulong Seed = 0x7A3C_91E5_2B44UL;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly ITestOutputHelper _out;

    public CudaTurboQuantPrefillTailTests(ITestOutputHelper output)
    {
        _out = output;
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string ptx = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "ptx"));
        if (Directory.Exists(ptx)) _kernels = new CudaKernels(ptx);
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    private void Require(int hd)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels is null, "PTX files not found");
        Assert.True(_kernels!.HasAttentionFlashPrefillKvTq(hd), $"tensor-core TurboQuant prefill hd{hd} not loaded (stale PTX / pre-Ampere?)");
    }

    private static IEnumerable<CudaKernels.TqPrefillVariant> Entries(CudaKernels k, int hd, int bits)
    {
        yield return CudaKernels.TqPrefillVariant.Legacy;
        Assert.True(k.HasAttentionFlashPrefillKvTqSpecialized(hd, bits), $"_hd{hd}_tq{bits} not loaded");
        yield return CudaKernels.TqPrefillVariant.Specialized;
        if (hd * bits / 32 % 4 == 0)
        {
            Assert.True(k.HasAttentionFlashPrefillKvTqStaged(hd, bits), $"_hd{hd}_tq{bits}_cpa not loaded");
            yield return CudaKernels.TqPrefillVariant.Staged;
        }
    }

    private static nint Alloc(long bytes) { CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)Math.Max(bytes, 16)).ThrowOnError(); return p; }
    private static void Up<T>(nint dst, T[] src) where T : unmanaged
    { fixed (T* p = src) CudaDriverApi.cuMemcpyHtoD_v2(dst, (nint)p, (nuint)(src.Length * sizeof(T))).ThrowOnError(); }
    private static void Down<T>(T[] dst, nint src) where T : unmanaged
    { fixed (T* p = dst) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, src, (nuint)(dst.Length * sizeof(T))).ThrowOnError(); }

    /// <summary>Host data for one case: random codes and F16 norms for seq_kv rows, random F16 Q, the codec's centroids.</summary>
    private sealed class Case
    {
        public required int Bits, Hd, NumHeads, NumKvHeads, SeqQ, SeqKv, Cu;
        public required uint[] KCodes, VCodes;
        public required Half[] KNorms, VNorms, Q;
        public required float[] Centroids;
    }

    private static Case MakeCase(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        int seqKv = offset + seqQ, cu = new TurboQuantLayout(hd, numKvHeads, bits, TqNormFormat.F16).CodeUintsPerVec;
        Assert.NotEqual(0, seqKv % 16);   // the last KV tile must be partial, or there is no tail to read
        var r = new Random(bits * 131 + hd + seqKv);
        uint[] Codes() { var c = new uint[seqKv * numKvHeads * cu]; for (int i = 0; i < c.Length; i++) c[i] = (uint)r.NextInt64(0, 1L << 32); return c; }
        Half[] Norms() { var n = new Half[seqKv * numKvHeads]; for (int i = 0; i < n.Length; i++) n[i] = (Half)(0.5 + 1.5 * r.NextDouble()); return n; }
        var q = new Half[seqQ * numHeads * hd];
        for (int i = 0; i < q.Length; i++) q[i] = (Half)(2.0 * (r.NextDouble() - 0.5));
        return new Case
        {
            Bits = bits, Hd = hd, NumHeads = numHeads, NumKvHeads = numKvHeads, SeqQ = seqQ, SeqKv = seqKv, Cu = cu,
            KCodes = Codes(), VCodes = Codes(), KNorms = Norms(), VNorms = Norms(), Q = q,
            Centroids = new TurboQuantCodec(hd, bits, Seed).Centroids.ToArray(),
        };
    }

    private Half[] Launch(Case c, CudaKernels.TqPrefillVariant entry, nint kc, nint vc, nint kn, nint vn)
    {
        nint dQ = Alloc(c.Q.Length * 2L), dO = Alloc(c.Q.Length * 2L), dC = Alloc(c.Centroids.Length * 4L);
        try
        {
            Up(dQ, c.Q); Up(dC, c.Centroids);
            CudaDriverApi.cuMemsetD8_v2(dO, 0, (nuint)(c.Q.Length * 2)).ThrowOnError();
            long spec0 = CudaKernels.AttentionFlashPrefillKvTqSpecLaunches, cpa0 = CudaKernels.AttentionFlashPrefillKvTqCpaLaunches;
            CudaKernels.TqPrefillVariantOverride = entry;
            try
            {
                _kernels!.LaunchAttentionFlashPrefillKvTq(dQ, kc, vc, kn, vn, dC, dO, c.SeqQ, c.SeqKv, c.NumHeads, c.NumKvHeads, c.Hd,
                    c.SeqKv - c.SeqQ, c.Bits, c.Cu, TqNormFormat.F16, _stream!.Handle);
            }
            finally { CudaKernels.TqPrefillVariantOverride = null; }
            _stream!.Synchronize();   // an out-of-bounds read surfaces here as CUDA_ERROR_ILLEGAL_ADDRESS
            Assert.Equal(entry == CudaKernels.TqPrefillVariant.Specialized ? 1 : 0, CudaKernels.AttentionFlashPrefillKvTqSpecLaunches - spec0);
            Assert.Equal(entry == CudaKernels.TqPrefillVariant.Staged ? 1 : 0, CudaKernels.AttentionFlashPrefillKvTqCpaLaunches - cpa0);
            var o = new Half[c.Q.Length];
            Down(o, dO);
            return o;
        }
        finally { CudaDriverApi.cuMemFree_v2(dQ); CudaDriverApi.cuMemFree_v2(dO); CudaDriverApi.cuMemFree_v2(dC); }
    }

    public static TheoryData<int, int, int, int, int, int> TailCases => new()
    {
        // bits, hd, numHeads, numKvHeads, seqQ, offset  (seq_kv = offset + seqQ, never a multiple of 16)
        { 4, 128, 32, 8, 40, 37 },   // Qwen3-4B geometry, seq_kv 77: last tile holds 13 keys
        { 3, 128, 32, 8, 40, 37 },
        { 2, 128, 8, 2, 40, 37 },
        { 8, 128, 8, 2, 20, 13 },    // seq_kv 33
        { 4, 64, 6, 2, 20, 13 },
        { 3, 64, 6, 2, 20, 13 },     // hd64 tq3: no staged entry (24-byte rows)
    };

    [SkippableTheory]
    [MemberData(nameof(TailCases))]
    public void FlashPrefill_ExactSizeCache_PoisonedTail_IsIgnored(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Require(hd);
        var c = MakeCase(bits, hd, numHeads, numKvHeads, seqQ, offset);
        int tailRows = 32;                                           // past the partial tile and one more full tile
        int rowWords = c.NumKvHeads * c.Cu, rowNorms = c.NumKvHeads;
        long codeWords = (long)(c.SeqKv + tailRows) * rowWords + 4, normCount = (long)(c.SeqKv + tailRows) * rowNorms + 2;
        nint kc = Alloc(codeWords * 4), vc = Alloc(codeWords * 4), kn = Alloc(normCount * 2), vn = Alloc(normCount * 2);
        try
        {
            void Fill(bool poison)
            {
                var kcw = new uint[codeWords]; var vcw = new uint[codeWords];
                var knh = new ushort[normCount]; var vnh = new ushort[normCount];
                Array.Copy(c.KCodes, kcw, c.KCodes.Length); Array.Copy(c.VCodes, vcw, c.VCodes.Length);
                for (int i = 0; i < c.KNorms.Length; i++) { knh[i] = BitConverter.HalfToUInt16Bits(c.KNorms[i]); vnh[i] = BitConverter.HalfToUInt16Bits(c.VNorms[i]); }
                if (poison)
                {
                    for (long i = c.KCodes.Length; i < codeWords; i++) { kcw[i] = 0xFFFFFFFFu; vcw[i] = 0xFFFFFFFFu; }
                    for (long i = c.KNorms.Length; i < normCount; i++) { knh[i] = 0x7E00; vnh[i] = 0x7E00; }   // F16 quiet NaN
                }
                Up(kc, kcw); Up(vc, vcw); Up(kn, knh); Up(vn, vnh);
            }
            foreach (var entry in Entries(_kernels!, hd, bits))
            {
                Fill(poison: false);
                var clean = Launch(c, entry, kc, vc, kn, vn);
                Fill(poison: true);
                var poisoned = Launch(c, entry, kc, vc, kn, vn);
                int bad = 0, firstBad = -1;
                for (int i = 0; i < clean.Length; i++)
                {
                    Assert.True(Half.IsFinite(clean[i]), $"{entry}: clean output not finite at {i}");
                    if (BitConverter.HalfToUInt16Bits(clean[i]) != BitConverter.HalfToUInt16Bits(poisoned[i])) { bad++; if (firstBad < 0) firstBad = i; }
                }
                _out.WriteLine($"tq{bits} hd{hd} {numHeads}/{numKvHeads} seqKv {c.SeqKv} {entry}: {bad} of {clean.Length} outputs moved by the poisoned tail");
                Assert.True(bad == 0, $"{entry}: the poisoned rows past seq_kv changed {bad} outputs (first at {firstBad}: {clean[Math.Max(firstBad, 0)]} -> {poisoned[Math.Max(firstBad, 0)]})");
            }
        }
        finally { CudaDriverApi.cuMemFree_v2(kc); CudaDriverApi.cuMemFree_v2(vc); CudaDriverApi.cuMemFree_v2(kn); CudaDriverApi.cuMemFree_v2(vn); }
    }

    // ── CUDA virtual memory management: a buffer whose end touches an unmapped page ──────────────────────────────────────────

    [StructLayout(LayoutKind.Sequential)]
    private struct CuMemLocation { public int Type; public int Id; }

    [StructLayout(LayoutKind.Sequential)]
    private struct CuMemAllocationProp
    {
        public int Type;                     // CU_MEM_ALLOCATION_TYPE_PINNED = 1
        public int RequestedHandleTypes;     // none
        public CuMemLocation Location;       // CU_MEM_LOCATION_TYPE_DEVICE = 1
        public nint Win32HandleMetaData;
        public byte CompressionType, GpuDirectRdmaCapable;
        public ushort Usage;
        public uint Reserved;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct CuMemAccessDesc { public CuMemLocation Location; public int Flags; }   // CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3

    private const string Nvcuda = "nvcuda";
    [LibraryImport(Nvcuda)] private static partial int cuDeviceGetAttribute(out int value, int attrib, int device);
    [LibraryImport(Nvcuda)] private static partial int cuMemGetAllocationGranularity(out nuint granularity, in CuMemAllocationProp prop, int option);
    [LibraryImport(Nvcuda)] private static partial int cuMemAddressReserve(out nint ptr, nuint size, nuint alignment, nint addr, ulong flags);
    [LibraryImport(Nvcuda)] private static partial int cuMemCreate(out ulong handle, nuint size, in CuMemAllocationProp prop, ulong flags);
    [LibraryImport(Nvcuda)] private static partial int cuMemMap(nint ptr, nuint size, nuint offset, ulong handle, ulong flags);
    [LibraryImport(Nvcuda)] private static partial int cuMemSetAccess(nint ptr, nuint size, in CuMemAccessDesc desc, nuint count);
    [LibraryImport(Nvcuda)] private static partial int cuMemUnmap(nint ptr, nuint size);
    [LibraryImport(Nvcuda)] private static partial int cuMemRelease(ulong handle);
    [LibraryImport(Nvcuda)] private static partial int cuMemAddressFree(nint ptr, nuint size);

    /// <summary>One mapped granule at the start of a two-granule reservation; the second granule stays unmapped.</summary>
    private sealed class GuardedBuffer : IDisposable
    {
        public nint Base; public nuint Granule; public ulong Handle;
        public nint EndOfMapping => Base + (nint)Granule;
        public void Dispose()
        {
            if (Handle != 0) { cuMemUnmap(Base, Granule); cuMemRelease(Handle); }
            if (Base != 0) cuMemAddressFree(Base, 2 * Granule);
        }
    }

    private static GuardedBuffer Guarded(long bytes)
    {
        CudaDriverApi.cuDeviceGet(out int dev, 0).ThrowOnError();
        var prop = new CuMemAllocationProp { Type = 1, Location = new CuMemLocation { Type = 1, Id = dev } };
        cuMemGetAllocationGranularity(out nuint g, prop, 0).ThrowOnError();
        nuint gran = (nuint)((bytes + (long)g - 1) / (long)g) * g;
        var b = new GuardedBuffer { Granule = gran };
        cuMemAddressReserve(out b.Base, 2 * gran, g, 0, 0).ThrowOnError();
        cuMemCreate(out b.Handle, gran, prop, 0).ThrowOnError();
        cuMemMap(b.Base, gran, 0, b.Handle, 0).ThrowOnError();
        cuMemSetAccess(b.Base, gran, new CuMemAccessDesc { Location = prop.Location, Flags = 3 }, 1).ThrowOnError();
        return b;
    }

    [SkippableTheory]
    [MemberData(nameof(TailCases))]
    public void FlashPrefill_CodesEndAtUnmappedPage_NoOutOfBoundsRead(int bits, int hd, int numHeads, int numKvHeads, int seqQ, int offset)
    {
        Require(hd);
        CudaDriverApi.cuDeviceGet(out int dev, 0).ThrowOnError();
        cuDeviceGetAttribute(out int vmm, 102, dev).ThrowOnError();   // CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
        Skip.If(vmm == 0, "CUDA virtual memory management is not supported on this device/driver");
        var c = MakeCase(bits, hd, numHeads, numKvHeads, seqQ, offset);
        long codeBytes = (long)c.KCodes.Length * 4;
        const int GuardBytes = 16;   // TurboQuantLayout's trailing guard word, rounded to keep the code base 16-byte aligned
        using var gk = Guarded(codeBytes + GuardBytes);
        using var gv = Guarded(codeBytes + GuardBytes);
        nint kc = gk.EndOfMapping - (nint)(codeBytes + GuardBytes), vc = gv.EndOfMapping - (nint)(codeBytes + GuardBytes);
        Assert.Equal(0L, (long)kc % 16);
        Up(kc, c.KCodes); Up(vc, c.VCodes);
        CudaDriverApi.cuMemsetD8_v2(kc + (nint)codeBytes, 0, GuardBytes).ThrowOnError();
        CudaDriverApi.cuMemsetD8_v2(vc + (nint)codeBytes, 0, GuardBytes).ThrowOnError();
        nint kn = Alloc(c.KNorms.Length * 2L + 64), vn = Alloc(c.VNorms.Length * 2L + 64);
        try
        {
            Up(kn, c.KNorms); Up(vn, c.VNorms);
            foreach (var entry in Entries(_kernels!, hd, bits))
            {
                var o = Launch(c, entry, kc, vc, kn, vn);   // throws (illegal address) if a code row past seq_kv is read
                foreach (var h in o) Assert.True(Half.IsFinite(h), $"{entry}: non-finite output");
                _out.WriteLine($"tq{bits} hd{hd} seqKv {c.SeqKv} {entry}: codes end {GuardBytes} B before an unmapped page, no fault");
            }
        }
        finally { CudaDriverApi.cuMemFree_v2(kn); CudaDriverApi.cuMemFree_v2(vn); }
    }
}
