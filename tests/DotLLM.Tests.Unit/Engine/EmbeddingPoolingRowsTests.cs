using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Embeddings;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>#1406: the default <see cref="IEmbeddingModel.ForwardHiddenForPooling"/> delivers exactly the rows pooling reads.</summary>
public sealed unsafe class EmbeddingPoolingRowsTests
{
    private const int H = 4;

    private sealed class Fake : IEmbeddingModel
    {
        public ITensor ForwardHidden(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
        {
            var t = UnmanagedTensor.Allocate(new TensorShape(tokenIds.Length, H), DType.Float32, deviceId: -1);
            var s = new Span<float>((void*)t.DataPointer, tokenIds.Length * H);
            for (int r = 0; r < tokenIds.Length; r++)
                for (int c = 0; c < H; c++) s[r * H + c] = tokenIds[r] * 10 + c;
            return t;
        }
    }

    [Theory]
    [InlineData(PoolingType.Last, 1)]
    [InlineData(PoolingType.Cls, 1)]
    [InlineData(PoolingType.Mean, 3)]
    public void Delivers_only_needed_rows_and_pools_identically(PoolingType pooling, int expectedRows)
    {
        IEmbeddingModel m = new Fake();
        int[] seq = [1, 2, 3];
        var viaRows = new float[H];
        int gotRows = -1;
        m.ForwardHiddenForPooling([seq], pooling, (_, rows) =>
        {
            gotRows = rows.Length / H;
            EmbeddingPooler.Pool(rows, gotRows, H, pooling, viaRows);
        }, 0);
        Assert.Equal(expectedRows, gotRows);

        var full = new float[H];
        using (var h = m.ForwardHidden(seq, [0, 1, 2], 0))
            EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, 3 * H), 3, H, pooling, full);
        Assert.Equal(full, viaRows);
    }
}
