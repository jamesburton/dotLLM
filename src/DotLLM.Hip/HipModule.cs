using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// Loads a HIP code-object file (.co / .hsaco) into a module and caches kernel
/// function handles. The ROCm runtime JIT-finalizes the code object for the
/// current GPU on first load where needed.
/// </summary>
public sealed class HipModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new();

    /// <summary>
    /// Loads a HIP module from a file path.
    /// </summary>
    /// <param name="coPath">Path to the .co / .hsaco code-object file.</param>
    public static HipModule LoadFromFile(string coPath)
    {
        byte[] bytes = File.ReadAllBytes(coPath);
        return LoadFromBytes(bytes);
    }

    /// <summary>
    /// Loads a HIP module from a byte array (raw code-object bytes).
    /// </summary>
    /// <param name="bytes">Code-object bytes (ELF container; no null terminator needed).</param>
    public static HipModule LoadFromBytes(byte[] bytes)
    {
        var module = new HipModule();
        unsafe
        {
            fixed (byte* p = bytes)
            {
                HipDriverApi.hipModuleLoadData(out module._module, (nint)p)
                    .ThrowOnError();
            }
        }
        return module;
    }

    /// <summary>
    /// Gets a kernel function handle by name. Caches the result for subsequent calls.
    /// </summary>
    /// <param name="name">The <c>extern "C"</c> kernel function name.</param>
    public nint GetFunction(string name)
    {
        if (!_functions.TryGetValue(name, out nint func))
        {
            HipDriverApi.hipModuleGetFunction(out func, _module, name)
                .ThrowOnError();
            _functions[name] = func;
        }
        return func;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint module = Interlocked.Exchange(ref _module, 0);
        if (module != 0)
        {
            HipDriverApi.hipModuleUnload(module);
            _functions.Clear();
        }
    }
}
