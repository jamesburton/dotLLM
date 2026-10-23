using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda.Architectures;

/// <summary>Short-convolution and unbiased sigmoid-routing kernels used by the LFM2 host.</summary>
internal sealed unsafe class CudaLfm2Kernels : IDisposable
{
    private readonly CudaModule _module;
    private readonly nint _conv, _route, _gemvQ4, _embeddingQ4, _quantQ8, _gemmQ4Q8;
    public CudaLfm2Kernels(string ptxDir)
    {
        _module = CudaModule.LoadFromFile(Path.Combine(ptxDir, "lfm2.ptx"));
        try { _conv = _module.GetFunction("lfm2_shortconv_f32"); _route = _module.GetFunction("lfm2_sigmoid_topk_f32"); _gemvQ4 = _module.GetFunction("lfm2_gemv_q4_0_f32"); _embeddingQ4 = _module.GetFunction("lfm2_embedding_q4_0_f32"); _quantQ8 = _module.GetFunction("lfm2_quant_q8_1_f32"); _gemmQ4Q8 = _module.GetFunction("lfm2_gemm_q4_0_q8_1"); }
        catch { _module.Dispose(); throw; }
    }
    public void Conv(nint state, nint proj, nint weights, nint output, int taps, int hidden, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &state, &proj, &weights, &output, &taps, &hidden, &tokens };
        CudaDriverApi.cuLaunchKernel(_conv, (uint)((hidden + 127) / 128), 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void Route(nint logits, nint bias, nint indices, nint weights, int experts, int topk, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &logits, &bias, &indices, &weights, &experts, &topk, &tokens };
        CudaDriverApi.cuLaunchKernel(_route, (uint)((tokens + 31) / 32), 1, 1, 32, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void GemvQ4(nint weights, nint input, nint output, int rows, int cols, nint stream)
    {
        void** args = stackalloc void*[] { &weights, &input, &output, &rows, &cols };
        CudaDriverApi.cuLaunchKernel(_gemvQ4, (uint)((rows + 3) / 4), 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void EmbeddingQ4(nint weights, nint ids, nint output, int hidden, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &weights, &ids, &output, &hidden, &tokens };
        CudaDriverApi.cuLaunchKernel(_embeddingQ4, (uint)(((long)hidden * tokens + 127) / 128), 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void QuantizeQ8(nint input, nint output, int blocks, nint stream)
    {
        void** args = stackalloc void*[] { &input, &output, &blocks };
        CudaDriverApi.cuLaunchKernel(_quantQ8, (uint)((blocks + 3) / 4), 1, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void GemmQ4Q8(nint weights, nint input, nint output, int rows, int cols, int tokens, nint stream)
    {
        void** args = stackalloc void*[] { &weights, &input, &output, &rows, &cols };
        CudaDriverApi.cuLaunchKernel(_gemmQ4Q8, (uint)((rows + 3) / 4), (uint)tokens, 1, 128, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }
    public void Dispose() => _module.Dispose();
}
