using System.Runtime.CompilerServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Issue #505: installs <see cref="CudaLibraryResolver"/> as this module loads, before anything can
/// try to resolve a P/Invoke in it.
/// </summary>
/// <remarks>
/// <para>The JIT resolves a <c>[LibraryImport]</c> target when it <b>compiles</b> the method that
/// references it, not when the call executes. A method that merely mentions
/// <see cref="CudaDriverApi"/> therefore loads <c>"cuda"</c> before its own first statement runs —
/// so a guard like <c>Skip.IfNot(CudaDevice.IsAvailable())</c> inside that method is too late to
/// protect it. Whether the load succeeds then depends on whether some earlier caller happened to
/// have registered the resolver, which is to say on ordering:
/// <c>CudaAttentionF16PagedPerfHarness</c> passed inside the full CUDA suite and failed with
/// <c>DllNotFoundException</c> when run as the only filter.</para>
/// <para>A module initializer runs before the first access to any type in the module, so the mapping
/// (<c>"cuda"</c> → <c>nvcuda.dll</c>, <c>"cublas"</c> → <c>cublas64_*.dll</c>) is in place for every
/// caller — production, test and benchmark — without each one having to isolate its driver calls in
/// a <c>NoInlining</c> helper the way <c>CudaDevice.ProbeGpuCount</c> does.</para>
/// <para>Registering costs nothing on a machine with no CUDA: it installs a callback and loads no
/// library. The resolver is only consulted if something actually P/Invokes.</para>
/// </remarks>
internal static class CudaModuleInitializer
{
    // CA2255 warns off [ModuleInitializer] in libraries because a library should not impose
    // start-up work on its consumers. Installing a DllImport resolver is the documented exception:
    // it is the only hook that runs before the JIT can resolve a P/Invoke in this module (#505), it
    // loads nothing, and a consumer that never touches the backend pays one delegate registration.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Initialize() => CudaLibraryResolver.Register();
}
