using System.Diagnostics;
using DotLLM.Vulkan.Interop;

namespace DotLLM.Vulkan;

/// <summary>What a <see cref="VulkanHeadroomCanary"/> probe concluded.</summary>
internal enum HeadroomOutcome
{
    /// <summary>Every requested byte was allocated AND touched by a GPU copy: the headroom exists right now.</summary>
    Fits,
    /// <summary>The driver reported out-of-memory (at allocation or at the first submit that needed the memory resident).</summary>
    DoesNotFit,
    /// <summary>The probe could not run (no second device, unexpected failure): no information, fall back to the arithmetic estimate.</summary>
    Unavailable,
}

/// <summary>Result of one canary probe.</summary>
internal readonly record struct HeadroomProbeResult(HeadroomOutcome Outcome, long RequestedBytes, long ReachedBytes, double Milliseconds, string? Detail)
{
    /// <summary>One-line description for the attach trace and the refusal text.</summary>
    public string Describe()
    {
        static string G(long b) => $"{b / (double)(1L << 30):F2} GiB";
        return Outcome switch
        {
            HeadroomOutcome.Fits => $"canary allocated and touched {G(ReachedBytes)} of {G(RequestedBytes)} on a second device in {Milliseconds:F0} ms",
            HeadroomOutcome.DoesNotFit => $"canary reached {G(ReachedBytes)} of {G(RequestedBytes)} before the driver ran out of memory ({Detail})",
            _ => $"canary unavailable ({Detail})",
        };
    }
}

/// <summary>The thing a canary allocates on: one call allocates a chunk and forces it resident with a GPU copy. Throws on out-of-memory.</summary>
internal interface IHeadroomCanaryTarget : IDisposable
{
    /// <summary>Allocates <paramref name="bytes"/> device memory, keeps it, and runs a GPU copy over it.</summary>
    void AllocateAndTouch(long bytes);
}

/// <summary>
/// Measures real free device headroom without risking the model's device (#820). On this UMA box a process (together with everything else on the
/// machine) can hold only about 0.65 x RAM of GPU memory, the figure DRIFTS (81.4 GiB measured at one time, 73.9 GiB an hour later, nothing else
/// running), the allocation that crosses it still succeeds, and the NEXT submit fails with out-of-device-memory - after which every allocation on
/// that <c>VkDevice</c> fails, even after the memory is released. No static formula can see the drift, so the question "will N more bytes fit?" is
/// put to a SECOND <see cref="VulkanDevice"/> in the same process: it allocates N bytes in chunks and copies over each one, so the driver must make
/// them resident. If that device hits the limit, only it is poisoned (measured: the model's device keeps working, and so does a third device made
/// afterwards); on success the memory is freed again and the real upload follows. The probe sees what the arithmetic cannot: this process's other
/// devices, other processes, and the current OS limit.
/// </summary>
internal static class VulkanHeadroomCanary
{
    /// <summary>Chunk size: large enough to be quick (a 3 GiB probe is ~25 allocations), small enough that the failing point is located within 128 MiB.</summary>
    internal const long ChunkBytes = 128L << 20;

    /// <summary>Allocates and touches <paramref name="bytes"/> on a second device bound to the same GPU as <paramref name="like"/>.</summary>
    public static HeadroomProbeResult Probe(VulkanDevice like, long bytes)
    {
        ArgumentNullException.ThrowIfNull(like);
        VulkanDevice second;
        try { second = like.PhysicalDeviceIndex >= 0 ? VulkanDevice.Create(like.PhysicalDeviceIndex) : VulkanDevice.Create(); }
        catch (Exception e) { return new(HeadroomOutcome.Unavailable, bytes, 0, 0, "cannot create a second device: " + e.Message); }
        return Run(new SecondDeviceTarget(second), bytes, ChunkBytes);   // the target owns (and disposes) the second device
    }

    /// <summary>The probe loop, separated from the device so the decision logic is unit-testable.</summary>
    internal static HeadroomProbeResult Run(IHeadroomCanaryTarget target, long bytes, long chunk)
    {
        var sw = Stopwatch.StartNew();
        long reached = 0;
        using (target)
        {
            try
            {
                while (reached < bytes)
                {
                    long n = Math.Min(chunk, bytes - reached);
                    target.AllocateAndTouch(n);
                    reached += n;
                }
                return new(HeadroomOutcome.Fits, bytes, reached, sw.Elapsed.TotalMilliseconds, null);
            }
            catch (VulkanException e)
            {
                return new(HeadroomOutcome.DoesNotFit, bytes, reached, sw.Elapsed.TotalMilliseconds, FirstLine(e.Message));
            }
            catch (Exception e)
            {
                return new(HeadroomOutcome.Unavailable, bytes, reached, sw.Elapsed.TotalMilliseconds, e.GetType().Name + ": " + FirstLine(e.Message));
            }
        }
    }

    private static string FirstLine(string s)
    {
        int i = s.IndexOf('\n');
        string t = i >= 0 ? s[..i] : s;
        return t.Length > 160 ? t[..160] : t;
    }

    private sealed class SecondDeviceTarget(VulkanDevice device) : IHeadroomCanaryTarget
    {
        private readonly List<VulkanDevice.Buffer> _buffers = new();
        private VulkanDevice.Buffer? _first;
        private long _firstBytes;

        public void AllocateAndTouch(long bytes)
        {
            var b = device.AllocateDeviceLocal(bytes);
            _buffers.Add(b);
            if (_first is null)
            {
                _first = b;
                _firstBytes = bytes;
                return;
            }
            // Copy over the whole new allocation so it must be resident; the first chunk is made resident by copying back into it once.
            ulong n = (ulong)Math.Min(bytes, _firstBytes);
            device.CopyBufferSynchronous(_first, b, n);
            if (_buffers.Count == 2) device.CopyBufferSynchronous(b, _first, n);
        }

        public void Dispose()
        {
            foreach (var b in _buffers) b.Dispose();
            _buffers.Clear();
            device.Dispose();
        }
    }
}
