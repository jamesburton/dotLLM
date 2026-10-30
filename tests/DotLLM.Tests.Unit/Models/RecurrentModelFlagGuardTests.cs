using System.Reflection;
using DotLLM.Core.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Models;

/// <summary>
/// <c>TextGenerator</c> (and the scheduler) decide whether to re-zero a model's recurrent state from
/// <see cref="IModel.RequiresPerSequenceState"/>. A model that implements <see cref="IModel.ResetSequenceState"/>
/// but leaves the flag at its <c>false</c> default is therefore never reset between requests (issue #615: the CUDA
/// Qwen3.5 hybrids flipped a repeated greedy answer from A to B). This pins the pairing for every IModel in the
/// product assemblies, so the next recurrent architecture cannot repeat it.
/// </summary>
public sealed class RecurrentModelFlagGuardTests
{
    private static readonly string[] Assemblies =
    [
        "DotLLM.Models", "DotLLM.Cpu", "DotLLM.Cuda", "DotLLM.Vulkan",
    ];

    [Fact]
    public void EveryModelDeclaringResetSequenceState_AlsoDeclaresRequiresPerSequenceState()
    {
        var offenders = new List<string>();
        int checkedTypes = 0;

        foreach (string name in Assemblies)
        {
            Assembly asm;
            try { asm = Assembly.Load(name); }
            catch (FileNotFoundException) { continue; }

            foreach (Type t in asm.GetTypes())
            {
                if (t.IsAbstract || t.IsInterface || !typeof(IModel).IsAssignableFrom(t)) continue;

                bool declaresReset = t.GetMethod(nameof(IModel.ResetSequenceState),
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly, Type.EmptyTypes) is not null;
                if (!declaresReset) continue;
                checkedTypes++;

                bool declaresFlag = t.GetProperty(nameof(IModel.RequiresPerSequenceState),
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly) is not null;
                if (!declaresFlag) offenders.Add(t.FullName!);
            }
        }

        Assert.True(checkedTypes >= 4, $"guard looked at only {checkedTypes} types; the reflection scan is not finding the models");
        Assert.True(offenders.Count == 0,
            "These models implement ResetSequenceState() but do not declare RequiresPerSequenceState => true, so " +
            "TextGenerator never resets their recurrent state between requests: " + string.Join(", ", offenders));
    }
}
