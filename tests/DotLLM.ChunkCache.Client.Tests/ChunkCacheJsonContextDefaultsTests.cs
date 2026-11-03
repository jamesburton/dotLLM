using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using Xunit;

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>
/// The #462 guard for the client's JSON contexts (dotLLM CLAUDE.md: <i>if you add a JSON context, add the equivalent
/// guard</i>). Same oracle as <c>TrayJsonContextDefaultsTests</c>: construct directly, deserialize <c>{}</c>, compare —
/// property initializers are invisible to reflection, and source generation silently drops them on <c>init</c> members.
/// </summary>
public sealed class ChunkCacheJsonContextDefaultsTests
{
    public static TheoryData<string> Contexts() => new() { nameof(ChunkCacheJsonContext), nameof(ChunkCacheHttpJsonContext) };

    private static JsonSerializerContext ContextOf(string name) => name switch
    {
        nameof(ChunkCacheJsonContext) => ChunkCacheJsonContext.Default,
        nameof(ChunkCacheHttpJsonContext) => ChunkCacheHttpJsonContext.Default,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    /// <summary>Every deserializable type round-trips its own construction defaults through <c>{}</c>.</summary>
    [Theory]
    [MemberData(nameof(Contexts))]
    public void EverySerializableType_RoundTripsItsConstructionDefaults(string name)
    {
        JsonSerializerContext context = ContextOf(name);
        var failures = new List<string>();
        int checkedTypes = 0;

        foreach (Type type in SerializableTypes(context).Where(IsPlainObject))
        {
            object direct = Activator.CreateInstance(type, nonPublic: true)
                ?? throw new InvalidOperationException(type.Name);
            object parsed = JsonSerializer.Deserialize("{}", type, context)!;
            checkedTypes++;

            foreach (PropertyInfo p in Properties(type))
            {
                object? expected = p.GetValue(direct);
                object? actual = p.GetValue(parsed);
                if (expected is null || IsDefaultOfType(expected, p.PropertyType))
                    continue;
                if (!Equals(expected, actual))
                    failures.Add($"{type.Name}.{p.Name}: constructed as '{expected}' but deserialized as '{actual ?? "null"}' (#462).");
            }
        }

        Assert.True(checkedTypes >= 3, $"{name}: inspected {checkedTypes} types; the reflection walk is not finding them");
        Assert.True(failures.Count == 0, $"{name}:\n" + string.Join("\n", failures));
    }

    /// <summary>
    /// The structural form: no <c>init</c> property of a DTO carries an initializer. Every type in these contexts must
    /// be constructible without arguments, so a positional record (whose defaults the guard could not see) fails here.
    /// </summary>
    [Theory]
    [MemberData(nameof(Contexts))]
    public void NoInitOnlyPropertyCarriesAnInitializer(string name)
    {
        var offenders = new List<string>();
        foreach (Type type in SerializableTypes(ContextOf(name)).Where(IsPlainObject))
        {
            object? direct;
            try { direct = Activator.CreateInstance(type, nonPublic: true); }
            catch (MissingMethodException) { offenders.Add($"{type.Name} has no parameterless constructor"); continue; }

            foreach (PropertyInfo p in Properties(type).Where(IsInitOnly))
            {
                object? value = p.GetValue(direct);
                if (value is not null && !IsDefaultOfType(value, p.PropertyType))
                    offenders.Add($"{type.Name}.{p.Name} (initializer '{value}')");
            }
        }

        Assert.True(offenders.Count == 0, $"{name}: make these nullable and resolve the default in code (#462):\n" + string.Join("\n", offenders));
    }

    /// <summary>The resolved defaults the nullable properties stand for (spec 7.2).</summary>
    [Fact]
    public void EffectiveDefaults_MatchSpec()
    {
        var evict = JsonSerializer.Deserialize("{}", ChunkCacheJsonContext.Default.EvictRequest)!;
        Assert.Equal("orphans", evict.EffectiveMode);
        Assert.True(evict.EffectiveDryRun);

        var select = JsonSerializer.Deserialize("{}", ChunkCacheJsonContext.Default.SelectRequest)!;
        Assert.Equal("symbol-512", select.EffectiveChunker);
        Assert.Equal(1, select.EffectiveHops);
        Assert.False(select.EffectiveIncludeText);
        Assert.Equal("document", select.EffectiveOrder);

        var status = JsonSerializer.Deserialize("{}", ChunkCacheJsonContext.Default.StatusRequest)!;
        Assert.Equal("symbol-512", status.EffectiveChunker);
        Assert.False(status.EffectiveListMissing);

        // [JsonIgnore]d accessors never reach the wire.
        string json = JsonSerializer.Serialize(new EvictRequest(), ChunkCacheJsonContext.Default.EvictRequest);
        Assert.Equal("{}", json);
    }

    private static IEnumerable<PropertyInfo> Properties(Type type) =>
        type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.CanRead && p.GetIndexParameters().Length == 0 && p.GetCustomAttribute<JsonIgnoreAttribute>() is null);

    private static bool IsInitOnly(PropertyInfo p) =>
        p.SetMethod is { } setter
        && setter.ReturnParameter.GetRequiredCustomModifiers().Any(m => string.Equals(m.FullName, "System.Runtime.CompilerServices.IsExternalInit", StringComparison.Ordinal));

    private static IEnumerable<Type> SerializableTypes(JsonSerializerContext context) =>
        context.GetType()
            .GetProperties(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
            .Select(p => p.PropertyType)
            .Where(t => t.IsGenericType && t.GetGenericTypeDefinition() == typeof(JsonTypeInfo<>))
            .Select(t => t.GetGenericArguments()[0])
            .Distinct();

    private static bool IsPlainObject(Type t) =>
        t is { IsClass: true, IsAbstract: false }
        && t != typeof(string)
        && !t.IsArray
        && !typeof(System.Collections.IEnumerable).IsAssignableFrom(t);

    private static bool IsDefaultOfType(object value, Type declared)
    {
        Type t = Nullable.GetUnderlyingType(declared) ?? declared;
        return t.IsValueType && Equals(value, Activator.CreateInstance(t));
    }
}
