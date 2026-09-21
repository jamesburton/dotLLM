using DotLLM.Tray.Autostart;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// Covers #455's autostart acceptance criterion: off until explicitly enabled, and cleanly
/// removable.
/// </summary>
/// <remarks>
/// <b>Discrimination (#417), by mutation.</b> Two mutants, both died:
/// <list type="number">
///   <item>
///     Changing <c>Disable()</c> to write an empty string instead of deleting failed
///     <see cref="Disable_RemovesTheEntryEntirelyRatherThanBlankingIt"/> — an empty <c>Run</c>
///     value still lists in Task Manager's Startup tab, so "cleanly removable" would be false
///     while <c>IsEnabled()</c> happily reported off.
///   </item>
///   <item>
///     Dropping the quoting from <c>Enable()</c> failed
///     <see cref="Enable_QuotesThePathSoASpaceDoesNotSplitTheCommand"/>.
///   </item>
/// </list>
/// </remarks>
public sealed class AutostartManagerTests
{
    private sealed class FakeAutostartStore : IAutostartStore
    {
        internal Dictionary<string, string> Values { get; } = new(StringComparer.OrdinalIgnoreCase);

        public string? Read(string valueName) => Values.GetValueOrDefault(valueName);

        public void Write(string valueName, string commandLine) => Values[valueName] = commandLine;

        public void Delete(string valueName) => Values.Remove(valueName);
    }

    [Fact]
    public void AFreshInstall_HasNoAutostartEntryAtAll()
    {
        // Not "a disabled entry" — no entry. Constructing the manager must not write anything.
        var store = new FakeAutostartStore();
        var manager = new AutostartManager(store);

        Assert.False(manager.IsEnabled());
        Assert.Empty(store.Values);
    }

    [Fact]
    public void Enable_QuotesThePathSoASpaceDoesNotSplitTheCommand()
    {
        var store = new FakeAutostartStore();
        var manager = new AutostartManager(store);

        manager.Enable(@"C:\Program Files\dotLLM\dotllm-tray.exe");

        var command = Assert.Single(store.Values).Value;
        Assert.Equal(@"""C:\Program Files\dotLLM\dotllm-tray.exe""", command);
        Assert.True(manager.IsEnabled());
    }

    [Fact]
    public void Enable_AppendsArgumentsOutsideTheQuotedPath()
    {
        var store = new FakeAutostartStore();
        var manager = new AutostartManager(store);

        manager.Enable(@"C:\dotLLM\tray.exe", "--minimized");

        Assert.Equal(@"""C:\dotLLM\tray.exe"" --minimized", store.Values[manager.ValueName]);
    }

    [Fact]
    public void Enable_Twice_OverwritesRatherThanLeavingAStaleSecondEntry()
    {
        var store = new FakeAutostartStore();
        var manager = new AutostartManager(store);

        manager.Enable(@"C:\old\tray.exe");
        manager.Enable(@"C:\new\tray.exe");

        Assert.Single(store.Values);
        Assert.Contains(@"C:\new\tray.exe", manager.CurrentCommandLine()!, StringComparison.Ordinal);
    }

    [Fact]
    public void Disable_RemovesTheEntryEntirelyRatherThanBlankingIt()
    {
        var store = new FakeAutostartStore();
        var manager = new AutostartManager(store);
        manager.Enable(@"C:\dotLLM\tray.exe");

        manager.Disable();

        Assert.False(manager.IsEnabled());
        // The acceptance criterion is "cleanly removable". A surviving empty value is not clean.
        Assert.Empty(store.Values);
    }

    [Fact]
    public void Disable_IsIdempotent()
    {
        var manager = new AutostartManager(new FakeAutostartStore());
        manager.Disable();
        manager.Disable();
        Assert.False(manager.IsEnabled());
    }

    [Fact]
    public void Enable_RejectsAnEmptyPath()
    {
        var manager = new AutostartManager(new FakeAutostartStore());
        Assert.Throws<ArgumentException>(() => manager.Enable(""));
    }

    [Fact]
    public void AnAlreadyQuotedPath_IsNotDoubleQuoted()
    {
        Assert.Equal(@"""C:\a b\t.exe""", AutostartManager.Quote(@"""C:\a b\t.exe"""));
    }

    /// <summary>
    /// The one test that touches the real registry, under a throwaway value name it deletes again.
    /// </summary>
    /// <remarks>
    /// The fake above proves the <i>logic</i>; it cannot prove that
    /// <see cref="RegistryAutostartStore"/> actually reaches
    /// <c>HKCU\Software\Microsoft\Windows\CurrentVersion\Run</c>. A wrong key path would pass
    /// every other test in this class and silently do nothing on a real machine.
    /// </remarks>
    [SkippableFact]
    public void RegistryStore_RoundTripsAgainstTheRealRunKey()
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "Registry autostart is Windows-only.");

        var valueName = "dotLLM Tray Test " + Guid.NewGuid().ToString("N");
        var store = new RegistryAutostartStore();
        var manager = new AutostartManager(store, valueName);
        try
        {
            Assert.False(manager.IsEnabled());

            manager.Enable(@"C:\Program Files\dotLLM\dotllm-tray.exe");
            Assert.True(manager.IsEnabled());
            Assert.Equal(@"""C:\Program Files\dotLLM\dotllm-tray.exe""", manager.CurrentCommandLine());

            manager.Disable();
            Assert.False(manager.IsEnabled());
            Assert.Null(manager.CurrentCommandLine());
        }
        finally
        {
            // Never leave a startup entry behind on the developer's machine, even on failure.
            store.Delete(valueName);
        }
    }
}
