using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Vulkan.Interop;

/// <summary>
/// Resolves the "vulkan-1" library name to platform-specific Vulkan loader binaries.
/// Windows: vulkan-1.dll. Linux: libvulkan.so.1. macOS: libvulkan.dylib (via MoltenVK).
/// </summary>
internal static class VulkanLibraryResolver
{
    private static readonly bool Registered;

    /// <summary>
    /// Issue #504: see <c>CudaLibraryResolver</c>'s note — the flag-based guard claimed the flag
    /// before installing the resolver, so a concurrent caller could return early and then P/Invoke
    /// <c>vulkan-1</c> with no mapping. The type-initialization lock closes that window.
    /// </summary>
    static VulkanLibraryResolver()
    {
        NativeLibrary.SetDllImportResolver(
            typeof(VulkanLibraryResolver).Assembly,
            ResolveVulkanLibrary);
        Registered = true;
    }

    /// <summary>
    /// Ensures the resolver is installed; guaranteed complete on return. The body reads
    /// <see cref="Registered"/> so the type-initialization trigger cannot be elided.
    /// </summary>
    internal static void Register()
    {
        if (!Registered)
            throw new InvalidOperationException("Vulkan library resolver registration did not complete.");
    }

    private static nint ResolveVulkanLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != "vulkan-1") return 0;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (NativeLibrary.TryLoad("vulkan-1.dll", out nint h)) return h;
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            // MoltenVK ships as libvulkan.dylib (plus libMoltenVK.dylib).
            if (NativeLibrary.TryLoad("libvulkan.dylib", out nint h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.1.dylib", out nint h2)) return h2;
        }
        else
        {
            if (NativeLibrary.TryLoad("libvulkan.so.1", out nint h)) return h;
            if (NativeLibrary.TryLoad("libvulkan.so", out nint h2)) return h2;
        }

        return 0; // fall through to default resolution
    }
}
