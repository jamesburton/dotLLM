using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Interop;

/// <summary>
/// Issue #1068: the spill warning's trigger logic, through the monitor's counter seam (no GPU, no PDH). The first read is the
/// baseline; each further <see cref="GpuSpillMonitor.WarnStepBytes"/> of shared usage above it warns once.
/// </summary>
public sealed class GpuSpillMonitorTests
{
    private const long MiB = 1L << 20;

    [Fact]
    public void WarnsOncePerStep_AboveTheFirstReading()
    {
        long shared = 76 * MiB;   // T5500 idle Shared Usage after load
        var warnings = new List<string>();
        var monitor = new GpuSpillMonitor(() => shared, warnings.Add);

        monitor.Check("load");                    // baseline
        shared += 255 * MiB;
        monitor.Check("below one step");
        Assert.Empty(warnings);

        shared += 2 * MiB;                         // 257 MiB above
        monitor.Check("KV growth");
        Assert.Single(warnings);
        Assert.Contains("257 MiB", warnings[0]);
        Assert.Contains("KV growth", warnings[0]);
        Assert.Contains("Sysmem Fallback Policy", warnings[0]);

        monitor.Check("same level");
        Assert.Single(warnings);                   // no repeat at the same step

        shared += 1800 * MiB;                      // ~2 GiB above, as on the owner's card
        monitor.Check("decode");
        Assert.Equal(2, warnings.Count);
        Assert.Equal(2057 * MiB, monitor.SpilledBytes);
    }

    [Fact]
    public void UnknownCounter_NeverWarns()
    {
        var warnings = new List<string>();
        var monitor = new GpuSpillMonitor(() => null, warnings.Add);
        monitor.Check("a");
        monitor.Check("b");
        Assert.Empty(warnings);
        Assert.Equal(0, monitor.SpilledBytes);
    }

    [Fact]
    public void AThrowingCounterRead_IsSwallowed()
    {
        var monitor = new GpuSpillMonitor(() => throw new InvalidOperationException("pdh"), _ => { });
        monitor.Check("a");
        Assert.Equal(0, monitor.SpilledBytes);
    }

    [Fact]
    public void RealReader_OnThisMachine_DoesNotThrow()
    {
        // Constructs the PDH reader (Windows) or nothing (elsewhere) and reads once. No GPU work is done.
        using var monitor = new GpuSpillMonitor(() => "test");
        monitor.Check("smoke");
        monitor.Check("smoke");
        Assert.True(monitor.SpilledBytes >= 0);
    }
}
