using System.Runtime.CompilerServices;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Issue #505: installs <see cref="VulkanLibraryResolver"/> as this module loads. See
/// <c>CudaModuleInitializer</c> for why registration at first <i>call</i> is too late — the JIT
/// resolves a <c>[LibraryImport]</c> target when it compiles the referencing method.
/// </summary>
internal static class VulkanModuleInitializer
{
    // CA2255 warns off [ModuleInitializer] in libraries because a library should not impose
    // start-up work on its consumers. Installing a DllImport resolver is the documented exception:
    // it is the only hook that runs before the JIT can resolve a P/Invoke in this module (#505), it
    // loads nothing, and a consumer that never touches the backend pays one delegate registration.
#pragma warning disable CA2255
    [ModuleInitializer]
#pragma warning restore CA2255
    internal static void Initialize() => VulkanLibraryResolver.Register();
}
