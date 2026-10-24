using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DotLLM.Core.Moe;

/// <summary>
/// What a cached calibration is valid for (#1071): the GPU, its PCIe slot, the CPU, the model file (path, size and modification
/// time stand in for a content hash) and the CPU leg's thread count. Any change re-runs the calibration.
/// </summary>
/// <param name="Gpu">GPU name.</param>
/// <param name="PciLink">PCIe identity of the GPU (bus id; the link the copies use).</param>
/// <param name="Cpu">CPU model name.</param>
/// <param name="ModelPath">Full path of the model file.</param>
/// <param name="ModelBytes">Size of the model file.</param>
/// <param name="ModelMtimeUtcTicks">Last write time of the model file (UTC ticks).</param>
/// <param name="CpuThreads">CPU cold-leg worker threads.</param>
public sealed record MoeCalibrationKey(string Gpu, string PciLink, string Cpu, string ModelPath, long ModelBytes,
    long ModelMtimeUtcTicks, int CpuThreads)
{
    /// <summary>The key as one string (the JSON property name of the record).</summary>
    /// <returns>The key string.</returns>
    public override string ToString() => string.Join('|',
        Gpu, PciLink, Cpu, ModelPath, ModelBytes.ToString(CultureInfo.InvariantCulture),
        ModelMtimeUtcTicks.ToString(CultureInfo.InvariantCulture), CpuThreads.ToString(CultureInfo.InvariantCulture));

    /// <summary>
    /// Builds the key for a model file (size and modification time read from disk; zero when the file cannot be read).
    /// </summary>
    /// <param name="gpu">GPU name.</param>
    /// <param name="pciLink">PCIe identity.</param>
    /// <param name="modelPath">Model file path (null or empty = unknown).</param>
    /// <param name="cpuThreads">CPU cold-leg worker threads.</param>
    /// <returns>The key.</returns>
    public static MoeCalibrationKey ForFile(string gpu, string pciLink, string? modelPath, int cpuThreads)
    {
        string path = "", cpu = CpuModelName();
        long bytes = 0, mtime = 0;
        if (!string.IsNullOrEmpty(modelPath))
        {
            try
            {
                var fi = new FileInfo(modelPath);
                path = fi.FullName;
                if (fi.Exists)
                {
                    bytes = fi.Length;
                    mtime = fi.LastWriteTimeUtc.Ticks;
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                path = modelPath;
            }
        }
        return new MoeCalibrationKey(gpu, pciLink, cpu, path, bytes, mtime, cpuThreads);
    }

    /// <summary>
    /// The CPU model name (Windows registry <c>ProcessorNameString</c>, Linux <c>/proc/cpuinfo</c> "model name"), else the
    /// architecture and logical processor count.
    /// </summary>
    /// <returns>The name.</returns>
    public static string CpuModelName()
    {
        try
        {
            if (OperatingSystem.IsWindows())
            {
                using var k = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(@"HARDWARE\DESCRIPTION\System\CentralProcessor\0");
                if (k?.GetValue("ProcessorNameString") is string name && name.Trim().Length > 0) return name.Trim();
            }
            else if (File.Exists("/proc/cpuinfo"))
            {
                foreach (string line in File.ReadLines("/proc/cpuinfo"))
                    if (line.StartsWith("model name", StringComparison.Ordinal) && line.IndexOf(':') is var c and > 0)
                        return line[(c + 1)..].Trim();
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or System.Security.SecurityException)
        {
            // fall through to the generic name
        }
        return $"{System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture} x{Environment.ProcessorCount}";
    }
}

/// <summary>CPU time of one expert of a layout, measured at n = 1 and n = 8 tokens (single-threaded, medians).</summary>
/// <param name="UpBytes">Up-matrix bytes of the layout.</param>
/// <param name="DownBytes">Down-matrix bytes of the layout.</param>
/// <param name="CpuUs1">Median microseconds at one token.</param>
/// <param name="CpuUs8">Median microseconds at eight tokens.</param>
public readonly record struct MoeLayoutTiming(long UpBytes, long DownBytes, double CpuUs1, double CpuUs8)
{
    /// <summary>The layout.</summary>
    public ExpertLayout Layout => new(UpBytes, DownBytes);
}

/// <summary>One load-time calibration (#1071, design #1063 section 6): the inputs of <see cref="MoeCostModel"/>.</summary>
/// <param name="H2DBytesPerSecond">Pinned host-to-device bandwidth (16 MiB copies).</param>
/// <param name="SyncRoundTripUs">Median D2H + event-sync round trip.</param>
/// <param name="Layouts">CPU timings per expert layout.</param>
/// <param name="MeasuredUtc">When it was measured.</param>
public sealed record MoeCalibrationRecord(double H2DBytesPerSecond, double SyncRoundTripUs, IReadOnlyList<MoeLayoutTiming> Layouts,
    DateTime MeasuredUtc)
{
    /// <summary>The cost model: CPU <c>a + b·n</c> fitted from n = 1 and 8, device time from the bytes (<see cref="MoeCostModel.DefaultGpuCost"/>).</summary>
    /// <returns>The model.</returns>
    public MoeCostModel ToCostModel()
        => new(H2DBytesPerSecond, SyncRoundTripUs,
            [.. Layouts.Select(t => new MoeLayoutCost(t.Layout, LinearCost.FitFrom1And8(t.CpuUs1, t.CpuUs8),
                MoeCostModel.DefaultGpuCost(t.UpBytes + t.DownBytes)))]);

    /// <summary>True when the record has a timing for every layout in <paramref name="layouts"/>.</summary>
    /// <param name="layouts">Layouts the model needs.</param>
    /// <returns>True when complete.</returns>
    public bool Covers(IEnumerable<ExpertLayout> layouts) => layouts.All(l => Layouts.Any(t => t.Layout == l));

    /// <summary>One log line: link, round trip and the CPU timings.</summary>
    /// <returns>The text.</returns>
    public string Describe()
    {
        var c = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.Append(c, $"H2D {H2DBytesPerSecond / 1e9:F2} GB/s, sync round trip {SyncRoundTripUs:F0} us; CPU us/expert (n=1 / n=8):");
        foreach (var t in Layouts)
            sb.Append(c, $" [{(t.UpBytes + t.DownBytes) / 1e6:F2} MB: {t.CpuUs1:F0} / {t.CpuUs8:F0}]");
        return sb.ToString();
    }
}

/// <summary>
/// The per-machine calibration cache (#1071): a JSON file of records keyed by <see cref="MoeCalibrationKey"/>, by default
/// <c>%LOCALAPPDATA%\dotllm\moe-calibration.json</c>. Reads never throw (a missing or corrupt file is a cache miss); writes
/// are best-effort.
/// </summary>
public static class MoeCalibrationStore
{
    /// <summary>Format version of the file.</summary>
    public const int Version = 1;

    /// <summary>The default cache path.</summary>
    public static string DefaultPath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.DoNotVerify),
        "dotllm", "moe-calibration.json");

    /// <summary>The record cached for <paramref name="key"/>, or null.</summary>
    /// <param name="path">Cache file.</param>
    /// <param name="key">The key.</param>
    /// <returns>The record, or null when absent, unreadable, written by another format <see cref="Version"/>, or holding a
    /// measurement no cost model can use (a non-positive or non-finite bandwidth, latency or CPU time).</returns>
    public static MoeCalibrationRecord? TryLoad(string path, MoeCalibrationKey key)
    {
        try
        {
            if (!File.Exists(path)) return null;
            using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
            if (!doc.RootElement.TryGetProperty("version", out var v) || v.ValueKind != JsonValueKind.Number
                || v.GetInt32() != Version) return null;
            if (!doc.RootElement.TryGetProperty("entries", out var entries) || entries.ValueKind != JsonValueKind.Object) return null;
            if (!entries.TryGetProperty(key.ToString(), out var e) || e.ValueKind != JsonValueKind.Object) return null;
            var r = Parse(e);
            return IsUsable(r) ? r : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException or InvalidOperationException
            or FormatException or KeyNotFoundException)
        {
            return null;
        }
    }

    /// <summary>Stores <paramref name="record"/> under <paramref name="key"/>, keeping every other entry.</summary>
    /// <param name="path">Cache file.</param>
    /// <param name="key">The key.</param>
    /// <param name="record">The record.</param>
    /// <returns>True when the file was written.</returns>
    public static bool TrySave(string path, MoeCalibrationKey key, MoeCalibrationRecord record)
    {
        try
        {
            var entries = new SortedDictionary<string, string>(StringComparer.Ordinal);   // key -> raw JSON of the record
            if (File.Exists(path))
            {
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
                    if (doc.RootElement.TryGetProperty("entries", out var old) && old.ValueKind == JsonValueKind.Object)
                        foreach (var p in old.EnumerateObject()) entries[p.Name] = p.Value.GetRawText();
                }
                catch (JsonException)
                {
                    // a corrupt cache is replaced
                }
            }
            entries[key.ToString()] = Serialize(record);

            using var ms = new MemoryStream();
            using (var w = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
            {
                w.WriteStartObject();
                w.WriteNumber("version", Version);
                w.WriteStartObject("entries");
                foreach (var (k, raw) in entries)
                {
                    w.WritePropertyName(k);
                    w.WriteRawValue(raw);
                }
                w.WriteEndObject();
                w.WriteEndObject();
            }
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            string tmp = $"{path}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp";   // concurrent writers never share it
            File.WriteAllBytes(tmp, ms.ToArray());
            File.Move(tmp, path, overwrite: true);
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static string Serialize(MoeCalibrationRecord r)
    {
        using var ms = new MemoryStream();
        using (var w = new Utf8JsonWriter(ms))
        {
            w.WriteStartObject();
            w.WriteNumber("h2dBytesPerSecond", r.H2DBytesPerSecond);
            w.WriteNumber("syncRoundTripUs", r.SyncRoundTripUs);
            w.WriteString("measuredUtc", r.MeasuredUtc.ToString("O", CultureInfo.InvariantCulture));
            w.WriteStartArray("layouts");
            foreach (var t in r.Layouts)
            {
                w.WriteStartObject();
                w.WriteNumber("upBytes", t.UpBytes);
                w.WriteNumber("downBytes", t.DownBytes);
                w.WriteNumber("cpuUs1", t.CpuUs1);
                w.WriteNumber("cpuUs8", t.CpuUs8);
                w.WriteEndObject();
            }
            w.WriteEndArray();
            w.WriteEndObject();
        }
        return Encoding.UTF8.GetString(ms.ToArray());
    }

    // A cached zero bandwidth would make every copy infinitely expensive (every cold expert to the CPU); re-measure instead.
    private static bool IsUsable(MoeCalibrationRecord r)
    {
        static bool Pos(double x) => double.IsFinite(x) && x > 0;
        return Pos(r.H2DBytesPerSecond) && double.IsFinite(r.SyncRoundTripUs) && r.SyncRoundTripUs >= 0
            && r.Layouts.All(t => Pos(t.CpuUs1) && Pos(t.CpuUs8));
    }

    private static MoeCalibrationRecord Parse(JsonElement e)
    {
        var layouts = new List<MoeLayoutTiming>();
        foreach (var t in e.GetProperty("layouts").EnumerateArray())
            layouts.Add(new MoeLayoutTiming(t.GetProperty("upBytes").GetInt64(), t.GetProperty("downBytes").GetInt64(),
                t.GetProperty("cpuUs1").GetDouble(), t.GetProperty("cpuUs8").GetDouble()));
        var when = DateTime.Parse(e.GetProperty("measuredUtc").GetString()!, CultureInfo.InvariantCulture,
            DateTimeStyles.RoundtripKind);
        return new MoeCalibrationRecord(e.GetProperty("h2dBytesPerSecond").GetDouble(),
            e.GetProperty("syncRoundTripUs").GetDouble(), layouts, when);
    }
}
