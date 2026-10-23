using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Guards the whole <see cref="ServerJsonContext"/> surface against a System.Text.Json
/// source-generation trap (#462): <b>a property initializer on an <c>init</c>-only property is
/// silently ignored on deserialization.</b>
/// </summary>
/// <remarks>
/// <para>
/// So <c>public bool Stream { get; init; } = true;</c> deserializes as <c>false</c>, and
/// <c>public string Name { get; init; } = "";</c> as <c>null</c> on a non-nullable reference type.
/// There is no warning and no error — the default simply does not apply, and direct construction
/// still honours it, so any test that does not round-trip through JSON cannot see the bug.
/// </para>
/// <para>
/// <b>The trigger is the <c>init</c> accessor</b>, not a <c>required</c> member and not
/// <c>record</c> — measured four ways in <c>TrayJsonContextDefaultsTests</c> on .NET 10:
/// <c>init</c> drops on both records and plain classes, <c>set</c> is honoured, adding a
/// <c>required</c> member changes nothing, and reflection-based serialization is unaffected. This
/// remark previously said <c>required</c> was the cause; the first three hits all happened to sit
/// on types that also had one, which made a correlate look causal.
/// </para>
/// <para>
/// It bit four times: <c>ModelPullRequest.Stream</c> (#454 — every default pull silently became
/// non-streaming), <c>TraySettings</c> (#455 — a partial config gave <c>Host=null, Port=0</c>),
/// <c>ChatCompletionRequest.N</c> (#460 — latent only because nothing read it), and the tray's
/// entire client DTO surface, where <c>TrayAvailableModel.Enabled = true</c> meant every available
/// model could arrive <b>disabled</b>.
/// </para>
/// <para>
/// <b>How it can be general.</b> Property initializers are not visible to reflection — they run
/// in the constructor. But <c>required</c> is enforced by the <i>compiler</i>, not the runtime, so
/// <see cref="Activator.CreateInstance(Type)"/> bypasses the requirement and the initializers
/// still run. That gives an oracle: construct directly, deserialize a payload that names only the
/// required members, and compare every other property.
/// </para>
/// <para>
/// <b>Scope, stated honestly.</b> This test passing is <i>not</i> evidence that the server's
/// <c>init</c> DTOs are clean — only that the <b>deserialized</b> ones are. The
/// <c>ResponseOnly</c> list below exempts ~25 write-only types, which genuinely cannot lose a
/// default they never deserialize; if any of them ever becomes a request shape, it leaves this
/// guard's scope silently.
/// </para>
/// </remarks>
public sealed class JsonContextDefaultsTests
{
    /// <summary>
    /// Every deserializable type in <see cref="ServerJsonContext"/> must round-trip its own
    /// construction defaults. A property whose initializer is dropped shows up here as a
    /// mismatch naming the type, the property, and both values.
    /// </summary>
    [Fact]
    public void EverySerializableType_RoundTripsItsConstructionDefaults()
    {
        var failures = new List<string>();
        int checkedTypes = 0;

        foreach (Type type in SerializableTypes())
        {
            if (!IsPlainObject(type))
                continue;
            if (ResponseOnly.Contains(type.Name))
                continue;

            object? direct;
            try
            {
                // `required` is a compile-time contract, so this runs the initializers.
                direct = Activator.CreateInstance(type, nonPublic: true);
            }
            catch (MissingMethodException)
            {
                continue; // no parameterless ctor — nothing to compare against
            }

            if (direct is null)
                continue;

            PropertyInfo[] props = type
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(p => p.CanRead && p.GetIndexParameters().Length == 0)
                .ToArray();

            PropertyInfo[] required = props.Where(IsRequired).ToArray();

            // Name only the required members, so every OTHER property must come back as its
            // construction default. If it does not, an initializer was dropped.
            string json = "{"
                + string.Join(",", required.Select(p => $"\"{JsonNameOf(p)}\":null"))
                + "}";

            object? parsed;
            try
            {
                parsed = JsonSerializer.Deserialize(json, type, ServerJsonContext.Default);
            }
            catch (Exception ex) when (ex is JsonException or NotSupportedException or InvalidOperationException)
            {
                // Not constructible from this shape (e.g. a required value type that rejects
                // null). Out of scope rather than silently "passing".
                continue;
            }

            if (parsed is null)
                continue;

            checkedTypes++;

            foreach (PropertyInfo p in props)
            {
                if (IsRequired(p))
                    continue;

                object? expected = SafeGet(p, direct);
                object? actual = SafeGet(p, parsed);

                // Only a NON-default construction value can be lost. If the type constructs the
                // property as null/zero anyway there was no initializer to drop.
                if (expected is null || IsDefaultOfType(expected, p.PropertyType))
                    continue;

                if (!Equals(expected, actual))
                {
                    failures.Add(
                        $"{type.Name}.{p.Name}: constructed as '{expected}' but deserialized as "
                        + $"'{actual ?? "null"}'. The property initializer was dropped — see #462. "
                        + "Make the property nullable and resolve the default in code.");
                }
            }
        }

        Assert.True(checkedTypes > 10, $"only inspected {checkedTypes} types — the reflection walk is not finding the context's types");
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>
    /// The types the context actually generated metadata for. Read from the generated
    /// <see cref="JsonTypeInfo{T}"/> properties rather than from
    /// <c>[JsonSerializable]</c> attributes, because that is the set the server really
    /// serializes with — an attribute the generator skipped would otherwise be inspected here
    /// and never exercised in production.
    /// </summary>

    /// <summary>
    /// Types the server only ever <b>writes</b>. An initializer cannot be dropped on a value
    /// that is never deserialized, so these are out of scope.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>This is an exemption list, not an allowlist — deliberately.</b> A new type is checked
    /// by default, so forgetting to classify one fails safe. The same inversion is what makes
    /// <c>RateLimitMiddleware</c>'s metering policy hold, and it has already caught three
    /// unclassified routes.
    /// </para>
    /// <para>
    /// <b>Residual risk, stated:</b> a type exempted here that later becomes deserialized —
    /// because a new endpoint accepts it, or a client library reuses it — silently leaves this
    /// guard's scope. If you start deserializing one of these, remove it from this list first.
    /// Clients that deserialize these shapes (the Web UI, the tray) maintain their OWN DTOs and
    /// need their own guard; <c>TrayJsonContext</c> is not covered here.
    /// </para>
    /// </remarks>
    private static readonly HashSet<string> ResponseOnly = new(StringComparer.Ordinal)
    {
        // OpenAI response shapes
        "ChatCompletionResponse", "ChatCompletionChunk", "CompletionResponse", "CompletionChunk",
        "ModelListResponse", "ModelInfoDto", "PropsResponse", "ErrorResponse",
        "EmbeddingResponse", "EmbeddingData",
        // Anthropic response + stream-event shapes (#448/#449)
        "AnthropicMessageResponse", "AnthropicErrorResponse", "AnthropicPingEvent",
        "AnthropicMessageStartEvent", "AnthropicMessageDeltaEvent", "AnthropicMessageStopEvent",
        "AnthropicContentBlockStartEvent", "AnthropicContentBlockDeltaEvent",
        "AnthropicContentBlockStopEvent",
        // Management/settings response shapes (#454)
        "AvailableModelDto", "BackendInfoDto", "SettingsDto", "SettingsUpdateResponse",
        "ModelUnloadResponse",
    };

    private static IEnumerable<Type> SerializableTypes() =>
        typeof(ServerJsonContext)
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

    private static bool IsRequired(PropertyInfo p) =>
        p.GetCustomAttribute<RequiredMemberAttribute>() is not null;

    private static string JsonNameOf(PropertyInfo p) =>
        p.GetCustomAttribute<JsonPropertyNameAttribute>()?.Name ?? p.Name;

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
