namespace DotLLM.Cuda;

/// <summary>
/// Which kernel runs an NVFP4 decode (single-token) GEMV on CUDA (#1061); see
/// <see cref="CudaKernels.Nvfp4DecodeGemvMode"/>.
/// </summary>
public enum Nvfp4DecodeGemv
{
    /// <summary>
    /// The dp4a kernel for weights of at least <see cref="CudaKernels.Nvfp4Dp4aMinWeightBytes"/>, the FP32-activation
    /// kernel below that (where the dp4a path's extra quantize launch costs more than it saves).
    /// </summary>
    Auto,

    /// <summary>Always the dp4a kernel on int8 activations (when it is loaded).</summary>
    Dp4a,

    /// <summary>Always #1038's FP32-activation kernel.</summary>
    F32,
}
