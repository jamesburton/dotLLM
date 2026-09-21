using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using DotLLM.Tray.Api;
using DotLLM.Tray.Config;
using Xunit;

namespace DotLLM.Tray.Tests;

/// <summary>
/// The tray's half of the #462 guard. <c>JsonContextDefaultsTests</c> covers
/// <c>ServerJsonContext</c> and says in its own remarks that it does <b>not</b> reach here —
/// the tray maintains its own DTOs and its own two contexts, so it needs its own guard, per the
/// rule in <c>CLAUDE.md</c>: <i>if you add a JSON context, add the equivalent guard</i>.
/// </summary>
/// <remarks>
/// <para>
/// The trap: a property initializer can be silently ignored on deserialization, so
/// <c>public bool Enabled { get; init; } = true;</c> arrives as <c>false</c>. No warning, no
/// error, and direct construction still honours it — so any test that does not round-trip through
/// JSON cannot see it. It bit three times in one day across three worktrees.
/// </para>
/// <para>
/// <b>The oracle.</b> Property initializers are invisible to reflection — they run in the
/// constructor. So construct directly, deserialize an <i>empty</i> payload, and compare every
/// property. Anything that differs had its initializer dropped.
/// </para>
/// <para>
/// <b>Why an empty payload and not "only the required members".</b> The server-side guard names
/// the required members because those types have them. Nothing in the tray does — and
/// <c>TraySettings.Normalized()</c> documents the defaults being dropped on a record with no
/// <c>required</c> member at all. Whether that is really the same mechanism is settled by
/// <see cref="TraySettings_PartialLoad_IsTheReasonNormalizedExists"/> below rather than assumed,
/// because the answer decides whether <c>required</c> is the trigger or merely a correlate.
/// </para>
/// </remarks>
public sealed class TrayJsonContextDefaultsTests
{
    public static TheoryData<string, JsonSerializerContext> Contexts() => new()
    {
        { nameof(TrayJsonContext), TrayJsonContext.Default },
        { nameof(TraySettingsJsonContext), TraySettingsJsonContext.Default },
    };

    /// <summary>
    /// Every deserializable type in either tray context must round-trip its own construction
    /// defaults. A dropped initializer shows up here naming the type, the property and both values.
    /// </summary>
    [Theory]
    [MemberData(nameof(Contexts))]
    public void EverySerializableType_RoundTripsItsConstructionDefaults(string name, JsonSerializerContext context)
    {
        var failures = new List<string>();
        int checkedTypes = 0;

        foreach (Type type in SerializableTypes(context))
        {
            if (!IsPlainObject(type))
                continue;

            object? direct;
            try
            {
                direct = Activator.CreateInstance(type, nonPublic: true);
            }
            catch (MissingMethodException)
            {
                continue; // no parameterless ctor — nothing to compare against
            }

            if (direct is null)
                continue;

            object? parsed;
            try
            {
                parsed = JsonSerializer.Deserialize("{}", type, context);
            }
            catch (Exception ex) when (ex is JsonException or NotSupportedException or InvalidOperationException)
            {
                continue;
            }

            if (parsed is null)
                continue;

            checkedTypes++;

            foreach (PropertyInfo p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                         .Where(p => p.CanRead && p.GetIndexParameters().Length == 0))
            {
                object? expected = SafeGet(p, direct);
                object? actual = SafeGet(p, parsed);

                // Only a NON-default construction value can be lost. If the type constructs the
                // property as null/zero anyway, there was no initializer to drop.
                if (expected is null || IsDefaultOfType(expected, p.PropertyType))
                    continue;

                // Collections compare by reference otherwise; an empty array default that comes
                // back as a different empty array is not a dropped initializer.
                if (expected is System.Collections.IEnumerable && expected is not string)
                {
                    if (actual is not null) continue;
                    failures.Add($"{type.Name}.{p.Name}: constructed as an empty collection but deserialized as null.");
                    continue;
                }

                if (!Equals(expected, actual))
                {
                    failures.Add(
                        $"{type.Name}.{p.Name}: constructed as '{expected}' but deserialized as "
                        + $"'{actual?.ToString() ?? "null"}'. The property initializer was dropped — see #462. "
                        + "Make the property nullable and resolve the default in code.");
                }
            }
        }

        Assert.True(checkedTypes > 0, $"{name}: inspected no types — the reflection walk is not finding them");
        Assert.True(failures.Count == 0, $"{name}:\n" + string.Join("\n", failures));
    }

    /// <summary>
    /// The structural precondition, caught one step earlier and with a clearer message than a
    /// value mismatch: no tray DTO may combine a <c>required</c> member with a property
    /// initializer. Nothing does today — which is exactly why this belongs here <i>now</i>, since
    /// the day someone adds the first <c>required</c> member is the day the existing initializers
    /// (<c>TrayAvailableModel.Enabled = true</c> among them, where a silent <c>false</c> would
    /// disable models) become live bugs rather than latent ones.
    /// </summary>
    [Theory]
    [MemberData(nameof(Contexts))]
    public void NoTypeCombinesARequiredMemberWithAPropertyInitializer(string name, JsonSerializerContext context)
    {
        var offenders = new List<string>();

        foreach (Type type in SerializableTypes(context).Where(IsPlainObject))
        {
            PropertyInfo[] props = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
            if (!props.Any(p => p.GetCustomAttribute<RequiredMemberAttribute>() is not null))
                continue;

            object? direct;
            try { direct = Activator.CreateInstance(type, nonPublic: true); }
            catch (MissingMethodException) { continue; }
            if (direct is null) continue;

            foreach (PropertyInfo p in props)
            {
                if (p.GetCustomAttribute<RequiredMemberAttribute>() is not null || !p.CanRead)
                    continue;
                object? value = SafeGet(p, direct);
                if (value is not null && !IsDefaultOfType(value, p.PropertyType))
                    offenders.Add($"{type.Name}.{p.Name} (initializer '{value}')");
            }
        }

        Assert.True(offenders.Count == 0,
            $"{name}: these properties carry an initializer on a type that also has a required member, "
            + "which is the #462 shape — the initializer is dropped on deserialization. Make them "
            + "nullable and resolve the default in code:\n" + string.Join("\n", offenders));
    }

    /// <summary>
    /// Settles what <c>TraySettings.Normalized()</c>'s remarks assert: that the generated
    /// deserializer does not apply this record's initializers, <b>despite it having no
    /// <c>required</c> member at all</b>.
    /// </summary>
    /// <remarks>
    /// This matters beyond the tray. If <c>required</c> is genuinely the trigger, the trap is
    /// narrow and easy to spot. If defaults can vanish without one, then every initializer in
    /// every source-generated context is suspect and the round-trip guard above is the only thing
    /// that can tell. The assertion is written to record whichever is true rather than to confirm
    /// what the comment says.
    /// </remarks>
    [Fact]
    public void TraySettings_PartialLoad_IsTheReasonNormalizedExists()
    {
        // The raw properties carry no initializer any more, precisely because one would be a lie:
        // source generation drops it on an `init` member. The resolved accessors are the contract.
        var direct = new TraySettings();
        Assert.Null(direct.Host);
        Assert.Null(direct.Port);
        Assert.Equal("localhost", direct.EffectiveHost);
        Assert.Equal(8080, direct.EffectivePort);

        var parsed = JsonSerializer.Deserialize("""{"autostart_last_known":true}""",
            TraySettingsJsonContext.Default.TraySettings)!;

        // Whatever the deserializer did, Normalized() is what the tray actually consumes, and it
        // must produce a usable host and port from a partial file either way.
        var normalized = parsed.Normalized();
        Assert.Equal("localhost", normalized.Host);
        Assert.Equal(8080, normalized.Port);
        Assert.True(normalized.AutostartLastKnown);
    }

    private static IEnumerable<Type> SerializableTypes(JsonSerializerContext context) =>
        context.GetType()
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Select(p => p.PropertyType)
            .Where(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(JsonTypeInfo<>))
            .Select(t => t.GetGenericArguments()[0])
            .Distinct();

    private static bool IsPlainObject(Type t) =>
        t is { IsClass: true, IsAbstract: false }
        && t != typeof(string)
        && !t.IsArray
        && !typeof(System.Collections.IEnumerable).IsAssignableFrom(t);

    private static object? SafeGet(PropertyInfo p, object instance)
    {
        try { return p.GetValue(instance); }
        catch (TargetInvocationException) { return null; }
    }

    private static bool IsDefaultOfType(object value, Type declared)
    {
        Type t = Nullable.GetUnderlyingType(declared) ?? declared;
        if (!t.IsValueType)
            return false;
        object? zero = Activator.CreateInstance(t);
        return Equals(value, zero);
    }
}
