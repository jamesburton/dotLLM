using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Equivalence tests for the per-row K-quant embedding lookup kernels added in
/// commit 'cuda-embedding-rowlookup'. These compare the output of
/// <see cref="CudaKernels.LaunchEmbeddingLookup"/> (per-row dequant on lookup)
/// against the legacy "dequant whole table to FP16, then index" path. The two
/// must be bit-identical because the dequant arithmetic per super-block is
/// identical — only the iteration order changes.
/// </summary>
[Trait("Category", "GPU")]
public class CudaEmbeddingLookupKQuantTests : IDisposable
{
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    public CudaEmbeddingLookupKQuantTests()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();

        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);
    }

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

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K)]
    [InlineData(QuantizationType.Q5_K)]
    [InlineData(QuantizationType.Q6_K)]
    public unsafe void EmbeddingLookup_KQuant_MatchesBulkDequant(QuantizationType qt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX files not found");
        Skip.IfNot(_kernels!.HasEmbeddingLookup(qt, hiddenSize: 256),
            $"Per-row {qt} embedding kernel not available in PTX");

        // Tiny synthetic vocab/hidden — large enough to exercise multiple
        // super-blocks per row but small enough to verify cheaply.
        const int vocab = 17;
        const int hidden = 512;          // 2 super-blocks per row
        long rowBytes = Dequantize.RowByteSize(hidden, qt);
        long tableBytes = rowBytes * vocab;

        // Random quantized bytes — the kernels treat them as opaque, and we only
        // care that per-row and bulk-dequant agree on the SAME bytes.
        var rng = new Random(1234);
        byte[] tableHost = new byte[tableBytes];
        rng.NextBytes(tableHost);

        // Token id list: include duplicates and out-of-order ids to confirm the
        // gather pattern is correct.
        int[] idsHost = new[] { 0, 5, 16, 3, 5, 11, 0, 2 };
        int seqLen = idsHost.Length;

        nint s = _stream!.Handle;

        // 1) Allocate device buffers
        CudaDriverApi.cuMemAlloc_v2(out nint dTable, (nuint)tableBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dIds, (nuint)(seqLen * sizeof(int))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dOutPerRow, (nuint)(seqLen * hidden * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dBulkF16, (nuint)(vocab * hidden * sizeof(ushort))).ThrowOnError();

        try
        {
            // Upload table + ids
            fixed (byte* p = tableHost)
                CudaDriverApi.cuMemcpyHtoD_v2(dTable, (nint)p, (nuint)tableBytes).ThrowOnError();
            fixed (int* p = idsHost)
                CudaDriverApi.cuMemcpyHtoD_v2(dIds, (nint)p, (nuint)(seqLen * sizeof(int))).ThrowOnError();

            // Per-row path: dequant only the seqLen rows we need
            _kernels!.LaunchEmbeddingLookup(dTable, qt, dIds, dOutPerRow, seqLen, hidden, s);

            // Reference: bulk-dequant the whole table, then gather rows on the host
            _kernels!.LaunchDequantToF16(dTable, qt, dBulkF16, vocab * hidden, s);
            _stream!.Synchronize();

            ushort[] perRowOut = new ushort[seqLen * hidden];
            ushort[] bulkOut = new ushort[vocab * hidden];
            fixed (ushort* p = perRowOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dOutPerRow,
                    (nuint)(seqLen * hidden * sizeof(ushort))).ThrowOnError();
            fixed (ushort* p = bulkOut)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dBulkF16,
                    (nuint)(vocab * hidden * sizeof(ushort))).ThrowOnError();

            // Compare bit-for-bit. Both kernels do the same FP arithmetic on
            // identical input bytes — there should be zero drift.
            for (int t = 0; t < seqLen; t++)
            {
                int id = idsHost[t];
                for (int i = 0; i < hidden; i++)
                {
                    ushort gotBits = perRowOut[t * hidden + i];
                    ushort wantBits = bulkOut[id * hidden + i];
                    Assert.True(gotBits == wantBits,
                        $"{qt}: token {t} (id={id}) elem {i}: per-row 0x{gotBits:X4} != bulk 0x{wantBits:X4}");
                }
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dTable);
            CudaDriverApi.cuMemFree_v2(dIds);
            CudaDriverApi.cuMemFree_v2(dOutPerRow);
            CudaDriverApi.cuMemFree_v2(dBulkF16);
        }
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
