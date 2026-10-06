using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using DotLLM.Cuda.Interop;
using DotLLM.Hip.Interop;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Interop;

/// <summary>
/// Issue #504: each backend's library resolver must be installed by the time <c>Register()</c>
/// returns, on every calling thread — not merely "eventually, by whichever thread got there first".
/// </summary>
/// <remarks>
/// <para>The bug being guarded against: <c>Register()</c> used
/// <c>if (Interlocked.Exchange(ref _registered, 1) != 0) return;</c> and only then called
/// <see cref="NativeLibrary.SetDllImportResolver"/>. A second thread arriving inside that window
/// returned having installed nothing, and its next P/Invoke asked the OS for the bare name — which
/// is exactly what the resolver exists to translate (<c>"cuda"</c> → <c>nvcuda.dll</c>,
/// <c>"vulkan-1"</c> → <c>vulkan-1.dll</c>, <c>"amdhip64"</c> → <c>amdhip64.dll</c>). It surfaced on
/// the T5500 as <c>DllNotFoundException: Unable to load DLL 'cuda'</c> in one full-suite run and not
/// the next, on a box where 771 other CUDA tests passed.</para>
/// <para><b>What this test can and cannot do.</b> It is a <i>contract</i> test, not a deterministic
/// reproducer: it asserts the observable invariant (after <c>Register()</c> returns, the resolver is
/// installed) rather than trying to lose a race on purpose. Against the old code it would fail only
/// when it happened to hit the window. The deterministic evidence that the window is gone is the
/// code shape — registration now runs in an explicit static constructor, so the CLR's
/// type-initialization lock blocks every other thread until it has finished.</para>
/// <para>The oracle is <see cref="NativeLibrary.SetDllImportResolver"/> itself: it permits exactly
/// one resolver per assembly and throws <see cref="InvalidOperationException"/> on a second attempt.
/// A throw therefore proves one is already installed. The attempt has no side effect when it throws,
/// so this does not disturb the resolvers the rest of the suite depends on.</para>
/// <para>No GPU, driver or native library is required: nothing here loads a library, it only
/// inspects the registration.</para>
/// </remarks>
public sealed class LibraryResolverRegistrationTests
{
    private const int Threads = 32;

    public static TheoryData<string> Backends() => new() { "cuda", "vulkan", "hip" };

    [Theory]
    [MemberData(nameof(Backends))]
    public void Register_InstallsTheResolverBeforeItReturns_OnEveryThread(string backend)
    {
        (Action register, System.Reflection.Assembly assembly) = backend switch
        {
            "cuda" => ((Action)CudaLibraryResolver.Register, typeof(CudaLibraryResolver).Assembly),
            "vulkan" => ((Action)VulkanLibraryResolver.Register, typeof(VulkanLibraryResolver).Assembly),
            "hip" => ((Action)HipLibraryResolver.Register, typeof(HipLibraryResolver).Assembly),
            _ => throw new ArgumentOutOfRangeException(nameof(backend)),
        };

        var failures = new ConcurrentBag<string>();
        using var start = new Barrier(Threads);

        var threads = new Thread[Threads];
        for (int i = 0; i < Threads; i++)
        {
            int id = i;
            threads[i] = new Thread(() =>
            {
                start.SignalAndWait();
                register();

                // Register() has returned on THIS thread, so a resolver must already be installed —
                // whichever thread installed it. If none is, this call succeeds instead of throwing,
                // which is both the assertion failure and (harmlessly) the missing registration.
                try
                {
                    NativeLibrary.SetDllImportResolver(assembly, static (_, _, _) => 0);
                    failures.Add($"thread {id}: Register() returned with no resolver installed");
                }
                catch (InvalidOperationException)
                {
                    // Expected: one is already installed.
                }
            });
        }

        foreach (Thread t in threads) t.Start();
        foreach (Thread t in threads) t.Join(TimeSpan.FromSeconds(30));

        Assert.True(failures.IsEmpty,
            $"{backend}: {string.Join("; ", failures)} — a caller that returns from Register() must be "
            + "able to P/Invoke immediately (#504).");
    }
}
