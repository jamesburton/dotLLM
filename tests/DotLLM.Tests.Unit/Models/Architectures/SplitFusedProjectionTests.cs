using System.Runtime.InteropServices;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Defensive-audit tests for <see cref="SafetensorsTensorResolver.SplitFusedProjection"/>.
/// Phi-3 ships <c>self_attn.qkv_proj.weight</c> (row-fused Q/K/V) and
/// <c>mlp.gate_up_proj.weight</c> (row-fused gate/up). HF
/// <c>modeling_phi3.py</c> slices these in a specific order:
/// <list type="bullet">
///   <item><description>qkv: Q rows <c>[0, n_heads * head_dim)</c>, K rows
///     <c>[n_heads*head_dim, +n_kv_heads*head_dim)</c>, V rows the last
///     <c>n_kv_heads * head_dim</c>.</description></item>
///   <item><description>gate_up: <c>chunk(2, dim=-1)</c> on the output dim,
///     which makes gate = rows <c>[0, I)</c>, up = rows
///     <c>[I, 2I)</c>.</description></item>
/// </list>
/// These tests construct a tiny fake fused tensor where row <c>i</c> is the
/// constant <c>i</c>, run <see cref="SafetensorsTensorResolver.SplitFusedProjection"/>,
/// and verify that each slab recovers the HF row range exactly.
/// </summary>
public sealed class SplitFusedProjectionTests
{
    /// <summary>
    /// Phi-3.5-mini uses MHA — <c>n_heads == n_kv_heads = 32</c>. Here we
    /// pick small but unequal Q / K / V widths so any accidental swap would
    /// show up as a whole-slab content mismatch.
    /// </summary>
    [Fact]
    public unsafe void PhiQkvSplit_MatchesHfRowOrder()
    {
        // Synthetic shape: Q=8 rows, K=4 rows, V=4 rows, hidden=3 cols.
        // Row i is filled with the value i across all columns, so we can
        // spot-check that slab 0 holds rows 0..7, slab 1 holds 8..11,
        // slab 2 holds 12..15.
        const int qRows = 8;
        const int kRows = 4;
        const int vRows = 4;
        const int hidden = 3;
        const int totalRows = qRows + kRows + vRows;

        using var fake = FakeSafetensorsSource.WithRowIndexedFusedTensor(
            "model.layers.0.self_attn.qkv_proj.weight", totalRows, hidden);

        var owned = new List<nint>();
        try
        {
            SafetensorsTensorResolver.SplitFusedProjection(
                fake,
                "model.layers.0.self_attn.qkv_proj.weight",
                new[] { qRows, kRows, vRows },
                hidden,
                owned,
                out var partPtrs);

            Assert.Equal(3, partPtrs.Length);
            AssertSlabRows(partPtrs[0], rowStart: 0, rowCount: qRows, cols: hidden);
            AssertSlabRows(partPtrs[1], rowStart: qRows, rowCount: kRows, cols: hidden);
            AssertSlabRows(partPtrs[2], rowStart: qRows + kRows, rowCount: vRows, cols: hidden);
        }
        finally
        {
            foreach (var p in owned)
                NativeMemory.AlignedFree((void*)p);
        }
    }

    /// <summary>
    /// Phi-3 MLP <c>gate_up_proj</c>: HF does
    /// <c>gate, up = gate_up.chunk(2, dim=-1)</c>. With weights
    /// <c>[2I, hidden]</c> that makes gate = rows <c>[0, I)</c>,
    /// up = rows <c>[I, 2I)</c>.
    /// </summary>
    [Fact]
    public unsafe void PhiGateUpSplit_MatchesHfChunkOrder()
    {
        const int intermediate = 6;
        const int hidden = 3;
        const int totalRows = 2 * intermediate;

        using var fake = FakeSafetensorsSource.WithRowIndexedFusedTensor(
            "model.layers.0.mlp.gate_up_proj.weight", totalRows, hidden);

        var owned = new List<nint>();
        try
        {
            SafetensorsTensorResolver.SplitFusedProjection(
                fake,
                "model.layers.0.mlp.gate_up_proj.weight",
                new[] { intermediate, intermediate },
                hidden,
                owned,
                out var partPtrs);

            Assert.Equal(2, partPtrs.Length);
            // Gate first (rows 0..I), then up (rows I..2I). Opposite order
            // would manifest as SwiGLU computing act(up)*gate instead of
            // act(gate)*up — subtle but silently wrong.
            AssertSlabRows(partPtrs[0], rowStart: 0, rowCount: intermediate, cols: hidden);
            AssertSlabRows(partPtrs[1], rowStart: intermediate, rowCount: intermediate, cols: hidden);
        }
        finally
        {
            foreach (var p in owned)
                NativeMemory.AlignedFree((void*)p);
        }
    }

    /// <summary>
    /// Asserts that the slab at <paramref name="slabPtr"/> holds
    /// <paramref name="rowCount"/> rows where row <c>j</c> is filled with
    /// the constant <c>rowStart + j</c> across <paramref name="cols"/>
    /// columns.
    /// </summary>
    private static unsafe void AssertSlabRows(nint slabPtr, int rowStart, int rowCount, int cols)
    {
        float* p = (float*)slabPtr;
        for (int j = 0; j < rowCount; j++)
        {
            float expected = rowStart + j;
            for (int c = 0; c < cols; c++)
            {
                float actual = p[(long)j * cols + c];
                Assert.Equal(expected, actual);
            }
        }
    }

    /// <summary>
    /// Minimal in-memory <see cref="ISafetensorsTensorSource"/> exposing a
    /// single row-indexed F32 tensor. The backing buffer is owned by this
    /// object and freed on <see cref="Dispose"/>.
    /// </summary>
    private sealed unsafe class FakeSafetensorsSource : ISafetensorsTensorSource
    {
        private readonly Dictionary<string, SafetensorsTensorDescriptor> _byName;
        private readonly List<SafetensorsTensorDescriptor> _all;
        private readonly nint _basePtr;
        private readonly long _byteCount;
        private readonly string _name;

        private FakeSafetensorsSource(string name, int rows, int cols)
        {
            _name = name;
            long elements = (long)rows * cols;
            _byteCount = elements * sizeof(float);
            _basePtr = (nint)NativeMemory.AlignedAlloc((nuint)_byteCount, 64);

            // Fill row i with the scalar i so splits are trivially verifiable.
            float* f = (float*)_basePtr;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    f[(long)r * cols + c] = r;

            var desc = new SafetensorsTensorDescriptor(
                Name: name,
                DType: SafetensorsDType.F32,
                Shape: new[] { rows, cols },
                DataBeginOffset: 0,
                DataEndOffset: _byteCount);

            _byName = new Dictionary<string, SafetensorsTensorDescriptor>
            {
                [name] = desc,
            };
            _all = new List<SafetensorsTensorDescriptor> { desc };
        }

        public static FakeSafetensorsSource WithRowIndexedFusedTensor(string name, int rows, int cols)
            => new(name, rows, cols);

        public IReadOnlyList<SafetensorsTensorDescriptor> Tensors => _all;
        public IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName => _byName;

        public nint GetTensorPointer(string name)
        {
            if (name != _name)
                throw new KeyNotFoundException(name);
            return _basePtr;
        }

        public ReadOnlySpan<byte> GetTensorSpan(string name)
        {
            if (name != _name)
                throw new KeyNotFoundException(name);
            return new ReadOnlySpan<byte>((void*)_basePtr, (int)_byteCount);
        }

        public void Dispose()
        {
            if (_basePtr != 0)
                NativeMemory.AlignedFree((void*)_basePtr);
        }
    }
}
