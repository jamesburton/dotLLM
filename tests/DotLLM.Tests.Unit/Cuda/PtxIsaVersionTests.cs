using System.Globalization;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Guards the PTX ISA version of the checked-in PTX files. These are tracked build
/// artifacts, so regenerating one with a newer CUDA toolkit silently raises the
/// <c>.version</c> directive and makes the module unloadable on older drivers
/// (<c>CUDA_ERROR_UNSUPPORTED_PTX_VERSION</c>).
/// </summary>
/// <remarks>
/// <para>
/// The baseline is PTX ISA <b>8.7</b>, emitted by CUDA 12.8 — the toolkit issue #124
/// standardised on for the committed tree. CUDA 13.1 emits <c>.version 9.1</c>, which
/// fails to load on any driver older than 13.1 (it bit a Kaggle T4 on CUDA 13.0).
/// Issue #318: commit 5d724b8d regressed 12 of the 57 committed files back to 9.1
/// because it was built with the 13.1 toolkit.
/// </para>
/// <para>
/// This complements, and is deliberately independent of, the <c>.target</c>
/// architecture: a file can target <c>sm_75</c> and still be unloadable because its
/// ISA version is too new. The two directives fail on different drivers.
/// </para>
/// </remarks>
public sealed class PtxIsaVersionTests
{
    /// <summary>Highest PTX ISA version the committed artifacts may declare (CUDA 12.8).</summary>
    private const int MaxMajor = 8;

    /// <summary>Highest minor within <see cref="MaxMajor"/>.</summary>
    private const int MaxMinor = 7;

    [Fact]
    public void CheckedInPtx_DeclaresPortableIsaVersion()
    {
        string ptxDir = FindPtxDir();
        string[] files = Directory.GetFiles(ptxDir, "*.ptx");
        Assert.NotEmpty(files);

        List<string> offenders = [];

        foreach (string file in files)
        {
            (int major, int minor)? version = ReadIsaVersion(file);
            Assert.True(version.HasValue, $"{Path.GetFileName(file)} declares no parseable .version directive.");

            (int major, int minor) = version!.Value;

            // Numeric comparison, not string: "10.0" sorts below "8.7" lexically.
            if (major > MaxMajor || (major == MaxMajor && minor > MaxMinor))
                offenders.Add($"{Path.GetFileName(file)}: .version {major}.{minor}");
        }

        Assert.True(
            offenders.Count == 0,
            $"{offenders.Count} of {files.Length} committed PTX files declare a PTX ISA version above " +
            $"{MaxMajor}.{MaxMinor} (CUDA 12.8) and will fail to load with CUDA_ERROR_UNSUPPORTED_PTX_VERSION " +
            "on older drivers:" + Environment.NewLine +
            string.Join(Environment.NewLine, offenders) + Environment.NewLine +
            "Regenerate with native/build_ptx.bat using the CUDA 12.8 toolkit " +
            @"(on the T5500 CUDA box: set ""CUDA_PATH=E:\CUDA_v12.8.1"" first — the toolkit on the " +
            "default CUDA_PATH is 13.1 and emits .version 9.1). See issues #124 and #318.");
    }

    /// <summary>
    /// Returns the <c>major.minor</c> of the first <c>.version</c> directive in a PTX file,
    /// or <see langword="null"/> if the file declares none.
    /// </summary>
    private static (int Major, int Minor)? ReadIsaVersion(string file)
    {
        foreach (string raw in File.ReadLines(file))
        {
            string line = raw.Trim();
            if (!line.StartsWith(".version", StringComparison.Ordinal))
                continue;

            string value = line.AsSpan(".version".Length).Trim().ToString();
            string[] parts = value.Split('.');
            if (parts.Length >= 2
                && int.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out int major)
                && int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out int minor))
            {
                return (major, minor);
            }

            return null;
        }

        return null;
    }

    private static string FindPtxDir()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            string candidate = Path.Combine(dir.FullName, "native", "ptx");
            if (Directory.Exists(candidate))
                return candidate;
            dir = dir.Parent;
        }

        throw new DirectoryNotFoundException("Could not locate native/ptx from the test output directory.");
    }
}
