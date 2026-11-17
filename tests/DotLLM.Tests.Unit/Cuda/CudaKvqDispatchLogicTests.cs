using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1850: which quantized-KV attention calls take the tensor-core prefill (pure; no GPU). Everything outside the gate runs
/// the scalar in-place kernel, which is always correct, so the gate only has to reject what the tensor-core kernel cannot do.
/// </summary>
public sealed class CudaKvqDispatchLogicTests
{
    private static bool Use(int seqLen = 600, int seqKv = 600, int offset = 0, int sliding = 0, bool sinks = false, int window = 0,
        int heads = 32, int kvHeads = 8, int headDim = 128, bool enabled = true, bool loaded = true)
        => CudaTransformerModel.UseKvqFlashPrefill(enabled, loaded, seqLen, seqKv, offset, sliding, sinks, window, heads, kvHeads, headDim);

    [Fact]
    public void ColdAndOffsetPrefills_UseTheTensorCoreKernel()
    {
        Assert.True(Use());                                         // TwIL-8B cold prefill
        Assert.True(Use(seqLen: 442, seqKv: 3407, offset: 2965));   // #1847 suffix prefill over a reused prefix
        Assert.True(Use(heads: 14, kvHeads: 2, headDim: 64));       // Qwen2.5-0.5B, group 7
        Assert.True(Use(seqLen: CudaTransformerModel.KvqFlashPrefillMinRows, seqKv: CudaTransformerModel.KvqFlashPrefillMinRows));
    }

    [Fact]
    public void EverythingElse_RunsTheScalarKernel()
    {
        Assert.False(Use(seqLen: 1, seqKv: 601, offset: 600));                 // decode
        Assert.False(Use(seqLen: 15, seqKv: 15));                              // below one q-tile
        Assert.False(Use(seqLen: 600, seqKv: 700));                            // queries are not the newest rows
        Assert.False(Use(sliding: 4096));
        Assert.False(Use(sinks: true));
        Assert.False(Use(window: 64));                                         // FP16 window ring
        Assert.False(Use(headDim: 256));
        Assert.False(Use(heads: 36, kvHeads: 4));                              // group 9 > 8
        Assert.False(Use(heads: 30, kvHeads: 8));                              // uneven GQA
        Assert.False(Use(enabled: false));                                     // DOTLLM_CUDA_KVQ_FLASH_PREFILL=0
        Assert.False(Use(loaded: false));                                      // pre-Ampere / missing PTX
    }
}
