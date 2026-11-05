using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using DotLLM.ChunkReuse.Store;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The <c>JsonContextDefaultsTests</c> guard (dotLLM CLAUDE.md, #462) for the store's source-generated contexts
/// (<see cref="ServerLedgerJsonContext"/>, <see cref="ChunkIndexJsonContext"/>, <see cref="StoreRegistryJsonContext"/>): a property initializer on an <c>init</c>
/// property is silently dropped on deserialization. Every DTO reachable from a context must have no <c>init</c> setter, and
/// must round-trip its construction defaults through <c>{}</c>.
/// </summary>
public sealed class ChunkReuseJsonContextDefaultsTests
{
    public static TheoryData<string> Contexts => new() { nameof(ServerLedgerJsonContext), nameof(ChunkIndexJsonContext), nameof(StoreRegistryJsonContext) };

    private static JsonSerializerContext ContextOf(string name) => name switch
    {
        nameof(ServerLedgerJsonContext) => ServerLedgerJsonContext.Default,
        nameof(ChunkIndexJsonContext) => ChunkIndexJsonContext.Default,
        nameof(StoreRegistryJsonContext) => StoreRegistryJsonContext.Default,
        _ => throw new ArgumentException(name),
    };

    [Theory]
    [MemberData(nameof(Contexts))]
    public void EveryDto_HasNoInitSetter_AndRoundTripsItsConstructionDefaults(string contextName)
    {
        JsonSerializerContext context = ContextOf(contextName);
        // Every type the generator emitted a JsonTypeInfo<T> property for; DTOs are the DotLLM classes among them.
        var roots = context.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.PropertyType.IsGenericType && p.PropertyType.GetGenericTypeDefinition() == typeof(JsonTypeInfo<>))
            .Select(p => p.PropertyType.GetGenericArguments()[0])
            .Where(t => t.IsClass && t.Namespace?.StartsWith("DotLLM", StringComparison.Ordinal) == true).ToList();
        Assert.NotEmpty(roots);

        var seen = new HashSet<Type>();
        var queue = new Queue<Type>(roots);
        var failures = new List<string>();
        while (queue.Count > 0)
        {
            Type type = queue.Dequeue();
            if (!seen.Add(type)) continue;
            JsonTypeInfo info = context.GetTypeInfo(type) ?? throw new InvalidOperationException($"{type} not in {contextName}");
            foreach (JsonPropertyInfo p in info.Properties) Enqueue(p.PropertyType, queue);
            if (info.Kind != JsonTypeInfoKind.Object) continue;

            foreach (PropertyInfo prop in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                MethodInfo? setter = prop.SetMethod;
                if (setter is not null && setter.ReturnParameter.GetRequiredCustomModifiers().Contains(typeof(IsExternalInit)))
                    failures.Add($"{type.Name}.{prop.Name} has an init setter");
            }
            object direct = Activator.CreateInstance(type, nonPublic: true)!;
            object parsed = JsonSerializer.Deserialize("{}", info)!;
            foreach (PropertyInfo prop in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
                if (!Equals(prop.GetValue(direct), prop.GetValue(parsed)))
                    failures.Add($"{type.Name}.{prop.Name}: constructed {prop.GetValue(direct) ?? "null"}, from {{}} {prop.GetValue(parsed) ?? "null"}");
        }
        Assert.True(failures.Count == 0, string.Join("\n", failures));
        Assert.True(seen.Count >= 2, $"walked only {seen.Count} types");
    }

    private static void Enqueue(Type t, Queue<Type> queue)
    {
        Type u = Nullable.GetUnderlyingType(t) ?? t;
        if (u.IsGenericType && u.GetGenericTypeDefinition() == typeof(List<>)) u = u.GetGenericArguments()[0];
        if (u.IsClass && u != typeof(string) && u.Namespace?.StartsWith("DotLLM", StringComparison.Ordinal) == true) queue.Enqueue(u);
    }

    [Fact]
    public void LedgerContext_RefusesDuplicateProperties()
    {
        Assert.False(ServerLedgerJsonContext.Default.Options.AllowDuplicateProperties);
        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize("""{"formatVersion": 1, "formatVersion": 2}""", ServerLedgerJsonContext.Default.LedgerDocument));
    }
}
