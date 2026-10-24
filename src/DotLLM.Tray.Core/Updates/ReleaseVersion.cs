using System.Globalization;

namespace DotLLM.Tray.Updates;

/// <summary>
/// A SemVer 2.0 version, enough of it to order dotLLM's MinVer-produced tags.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="System.Version"/> cannot do this job. dotLLM versions its releases with MinVer
/// (<c>MinVerTagPrefix=v</c>, <c>MinVerDefaultPreReleaseIdentifiers=preview.0</c>), so real
/// version strings look like <c>v0.4.0</c>, <c>0.4.0-preview.0.12</c> and
/// <c>0.4.0-preview.0.12+abc1234</c>. <c>Version.Parse("0.4.0-preview.0.12")</c> throws, and
/// naive string comparison puts <c>0.4.0-preview.0.12</c> <i>after</i> <c>0.4.0</c> — which would
/// tell a user on a final release that a stale prerelease is an upgrade.
/// </para>
/// <para>
/// The rules implemented are SemVer's precedence rules: build metadata after <c>+</c> is ignored
/// entirely; a version with a prerelease suffix sorts <i>before</i> the same core version without
/// one; prerelease identifiers compare left to right, numeric identifiers numerically and below
/// alphanumeric ones.
/// </para>
/// </remarks>
public sealed class ReleaseVersion : IComparable<ReleaseVersion>, IEquatable<ReleaseVersion>
{
    private ReleaseVersion(int major, int minor, int patch, string[] prerelease, string original)
    {
        Major = major;
        Minor = minor;
        Patch = patch;
        Prerelease = prerelease;
        Original = original;
    }

    /// <summary>Major component.</summary>
    public int Major { get; }

    /// <summary>Minor component.</summary>
    public int Minor { get; }

    /// <summary>Patch component.</summary>
    public int Patch { get; }

    /// <summary>Dot-separated prerelease identifiers, empty for a final release.</summary>
    public IReadOnlyList<string> Prerelease { get; }

    /// <summary>The string this was parsed from.</summary>
    public string Original { get; }

    /// <summary>Whether this is a prerelease.</summary>
    public bool IsPrerelease => Prerelease.Count > 0;

    /// <summary>
    /// Parses a version, tolerating a leading <c>v</c> and trailing <c>+build</c> metadata.
    /// Returns null when the string is not a version at all.
    /// </summary>
    /// <param name="value">The text to parse.</param>
    public static ReleaseVersion? Parse(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return null;

        var text = value.Trim();
        var original = text;

        if (text.StartsWith('v') || text.StartsWith('V'))
            text = text[1..];

        // Build metadata is explicitly excluded from precedence by SemVer, and
        // AssemblyInformationalVersion carries it ("0.4.0+abc1234") by default.
        var plus = text.IndexOf('+', StringComparison.Ordinal);
        if (plus >= 0)
            text = text[..plus];

        string[] prerelease = [];
        var dash = text.IndexOf('-', StringComparison.Ordinal);
        if (dash >= 0)
        {
            prerelease = text[(dash + 1)..].Split('.', StringSplitOptions.RemoveEmptyEntries);
            text = text[..dash];
        }

        var parts = text.Split('.');
        if (parts.Length is < 1 or > 4)
            return null;

        if (!TryPart(parts, 0, out var major) || !TryPart(parts, 1, out var minor) || !TryPart(parts, 2, out var patch))
            return null;

        return new ReleaseVersion(major, minor, patch, prerelease, original);

        static bool TryPart(string[] parts, int index, out int value)
        {
            if (index >= parts.Length)
            {
                value = 0;
                return true;
            }

            return int.TryParse(parts[index], NumberStyles.None, CultureInfo.InvariantCulture, out value);
        }
    }

    /// <inheritdoc />
    public int CompareTo(ReleaseVersion? other)
    {
        if (other is null)
            return 1;

        var core = Major.CompareTo(other.Major);
        if (core != 0)
            return core;
        core = Minor.CompareTo(other.Minor);
        if (core != 0)
            return core;
        core = Patch.CompareTo(other.Patch);
        if (core != 0)
            return core;

        // A prerelease sorts before the release it leads to: 1.0.0-preview.1 < 1.0.0.
        if (Prerelease.Count == 0 && other.Prerelease.Count == 0)
            return 0;
        if (Prerelease.Count == 0)
            return 1;
        if (other.Prerelease.Count == 0)
            return -1;

        var shared = Math.Min(Prerelease.Count, other.Prerelease.Count);
        for (var i = 0; i < shared; i++)
        {
            var comparison = CompareIdentifier(Prerelease[i], other.Prerelease[i]);
            if (comparison != 0)
                return comparison;
        }

        return Prerelease.Count.CompareTo(other.Prerelease.Count);
    }

    /// <inheritdoc />
    public bool Equals(ReleaseVersion? other) => CompareTo(other) == 0;

    /// <inheritdoc />
    public override bool Equals(object? obj) => obj is ReleaseVersion other && Equals(other);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Major);
        hash.Add(Minor);
        hash.Add(Patch);
        foreach (var identifier in Prerelease)
            hash.Add(identifier, StringComparer.Ordinal);
        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString() => Original;

    private static int CompareIdentifier(string left, string right)
    {
        var leftNumeric = int.TryParse(left, NumberStyles.None, CultureInfo.InvariantCulture, out var leftValue);
        var rightNumeric = int.TryParse(right, NumberStyles.None, CultureInfo.InvariantCulture, out var rightValue);

        // "Numeric identifiers always have lower precedence than alphanumeric identifiers."
        if (leftNumeric && rightNumeric)
            return leftValue.CompareTo(rightValue);
        if (leftNumeric)
            return -1;
        if (rightNumeric)
            return 1;
        return string.CompareOrdinal(left, right);
    }
}
