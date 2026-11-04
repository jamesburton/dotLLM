using System.Text;
using System.Text.RegularExpressions;
using DotLLM.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cpu;

/// <summary>
/// #1623: source-level guards for the CPU feature-gating convention (docs/CPU_FEATURE_GATING.md). They exist because the bug class
/// they catch (an AVX2 call reachable on a CPU or runtime config without AVX2) is invisible on every machine that has AVX2: it
/// passed CI on #1617 and failed only on the one leg that disables the wide ISAs.
///
/// <list type="number">
/// <item><b>One mechanism.</b> No raw <c>X.IsSupported</c> / <c>VectorN.IsHardwareAccelerated</c> outside <see cref="CpuFeatures"/>, so
/// <c>DOTLLM_CPU_ISA</c> reaches every dispatch.</item>
/// <item><b>Entry points are guarded or tier-named.</b> A non-private method that calls ISA intrinsics must either consult
/// <see cref="CpuFeatures"/> itself or say in its name which tier it is (<c>...Avx2</c>, <c>...Sse</c>, <c>...Vnni</c>, <c>...Fma</c>, ...), which
/// marks it as a leaf that a dispatcher must guard.</item>
/// <item><b>Dispatchers guard.</b> A non-private method that calls a tier-named kernel must itself consult <see cref="CpuFeatures"/> (or a
/// predicate member defined in terms of it), unless it is tier-named too.</item>
/// </list>
/// The scan is a deliberately small heuristic (regex + brace matching, no Roslyn): it errs on the side of reporting, and the fix is
/// either the guard or a rename that states the tier. The dynamic backstop is the CI leg that runs the whole suite with the wide ISAs off.
/// </summary>
public sealed class CpuGatingConventionTests(ITestOutputHelper output)
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    private static readonly Regex RawCheck = new(
        @"\b(?:Avx|Avx2|Fma|AvxVnni|Avx512F|Avx512BW|Avx512CD|Avx512DQ|Avx512VL|Avx512Vbmi|Avx10v1|Sse|Sse2|Sse3|Ssse3|Sse41|Sse42|Bmi1|Bmi2|Popcnt|Lzcnt|Vector128|Vector256|Vector512)\s*\.\s*(?:IsSupported|IsHardwareAccelerated)\b",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture, Timeout);

    private static readonly Regex IsaCall = new(
        @"\b(?:Avx512[A-Za-z0-9]*|AvxVnni|Avx2|Avx|Fma|Sse41|Sse42|Sse3|Sse2|Ssse3|Sse|Bmi1|Bmi2|Popcnt|Lzcnt)\.(?!IsSupported)[A-Z][A-Za-z0-9]*\s*[(<]",
        RegexOptions.Compiled | RegexOptions.ExplicitCapture, Timeout);

    private static readonly Regex TierName = new(@"(Avx2|Avx512|Avx|Vnni|Sse|Ssse3|Sse2|Sse41|Fma|Simd)", RegexOptions.Compiled | RegexOptions.ExplicitCapture, Timeout);

    private static readonly Regex MethodDecl = new(
        @"^(?<indent>[ \t]+)(?:\[[^\]\r\n]*\]\s*)*(?<mods>(?:(?:public|internal|private|protected|static|unsafe|partial|readonly|extern|override|sealed)\s+)+)[\w<>\[\],\.\?\* ]+?\s+(?<name>\w+)\s*(?:<[^>\r\n]*>)?\(",
        RegexOptions.Compiled | RegexOptions.Multiline | RegexOptions.ExplicitCapture, Timeout);

    private static string? FindSourceRoot()
    {
        for (string? d = AppContext.BaseDirectory; d is not null; d = Path.GetDirectoryName(d))
            if (File.Exists(Path.Combine(d, "src", "DotLLM.Cpu", "DotLLM.Cpu.csproj"))) return Path.Combine(d, "src");
        return null;
    }

    private sealed record Method(string File, int Line, string Name, bool IsPrivate, string Body);

    private static string StripComments(string s) => Regex.Replace(s, @"//[^\r\n]*", "", RegexOptions.ExplicitCapture, Timeout);

    private static IEnumerable<string> SourceFiles(string src)
        => Directory.EnumerateFiles(src, "*.cs", SearchOption.AllDirectories)
            .Where(f => !f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}")
                     && !f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}"));

    private static List<Method> Methods(string src)
    {
        var all = new List<Method>();
        foreach (string f in SourceFiles(src))
        {
            string text = File.ReadAllText(f);
            if (!text.Contains("System.Runtime.Intrinsics", StringComparison.Ordinal)) continue;
            foreach (Match m in MethodDecl.Matches(text))
            {
                if (m.Groups["indent"].Value.Replace("\t", "    ", StringComparison.Ordinal).Length > 8) continue; // local function: private by construction
                string head = text.Substring(m.Index, Math.Min(200, text.Length - m.Index));
                if (head.Contains(" class ") || head.Contains(" struct ") || head.TrimStart().StartsWith("//")) continue;
                int i = m.Index + m.Length, depth = 0, bodyStart = -1;
                // Find the body: first '{' or '=>' before the next ';' at paren depth 0.
                int paren = 1;
                for (; i < text.Length && paren > 0; i++) { if (text[i] == '(') paren++; else if (text[i] == ')') paren--; }
                int j = i;
                while (j < text.Length && text[j] != '{' && text[j] != ';' && !(text[j] == '=' && j + 1 < text.Length && text[j + 1] == '>')) j++;
                if (j >= text.Length || text[j] == ';') continue; // abstract / extern / declaration only
                int end;
                if (text[j] == '=') { end = text.IndexOf(';', j); if (end < 0) continue; bodyStart = j; }
                else
                {
                    bodyStart = j;
                    int k = j;
                    for (; k < text.Length; k++)
                    {
                        if (text[k] == '{') depth++;
                        else if (text[k] == '}') { depth--; if (depth == 0) break; }
                    }
                    end = k;
                }
                string body = StripComments(text.Substring(bodyStart, Math.Max(0, end - bodyStart)));
                int line = text.AsSpan(0, m.Index).Count('\n') + 1;
                all.Add(new Method(Path.GetRelativePath(src, f).Replace('\\', '/'), line, m.Groups["name"].Value,
                    Regex.IsMatch(m.Groups["mods"].Value, @"\bprivate\b", RegexOptions.ExplicitCapture, Timeout), body));
            }
        }
        return all;
    }

    [SkippableFact]
    public void NoRawIsaChecksOutsideCpuFeatures()
    {
        string? src = FindSourceRoot();
        Skip.If(src is null, "source tree not available next to the test binaries");
        var bad = new List<string>();
        foreach (string f in SourceFiles(src!))
        {
            if (Path.GetFileName(f) == "CpuFeatures.cs") continue;
            string[] lines = File.ReadAllLines(f);
            for (int i = 0; i < lines.Length; i++)
            {
                string code = StripComments(lines[i]);
                if (RawCheck.IsMatch(code)) bad.Add($"{Path.GetRelativePath(src!, f).Replace('\\', '/')}:{i + 1}: {lines[i].Trim()}");
            }
        }
        foreach (string b in bad) output.WriteLine(b);
        Assert.True(bad.Count == 0,
            "Use CpuFeatures.<Flag> instead of a raw IsSupported / IsHardwareAccelerated (so DOTLLM_CPU_ISA applies):\n" + string.Join('\n', bad));
    }

    [SkippableFact]
    public void NonPrivateIntrinsicMethods_AreGuardedOrNameTheirTier()
    {
        string? src = FindSourceRoot();
        Skip.If(src is null, "source tree not available next to the test binaries");
        var bad = Methods(src!)
            .Where(m => !m.IsPrivate && IsaCall.IsMatch(m.Body) && !TierName.IsMatch(m.Name) && !m.Body.Contains("CpuFeatures.", StringComparison.Ordinal))
            .Select(m => $"{m.File}:{m.Line} {m.Name}").ToList();
        foreach (string b in bad) output.WriteLine(b);
        Assert.True(bad.Count == 0,
            "These non-private methods call ISA intrinsics but neither consult CpuFeatures nor name their tier (rename ...Avx2/...Sse/..., or guard them):\n"
            + string.Join('\n', bad));
    }

    [SkippableFact]
    public void NonPrivateCallersOfTierKernels_ConsultCpuFeatures()
    {
        string? src = FindSourceRoot();
        Skip.If(src is null, "source tree not available next to the test binaries");
        List<Method> methods = Methods(src!);

        var tierKernels = methods
            .Where(m => TierName.IsMatch(m.Name) && IsaCall.IsMatch(m.Body) && !m.Name.Contains("Scalar", StringComparison.Ordinal))
            .Select(m => m.Name).ToHashSet();

        // Predicate members (bool properties / fields / methods) defined in terms of CpuFeatures, transitively, count as guards.
        var defs = new List<(string Name, string Text)>();
        foreach (string f in SourceFiles(src!))
        {
            string text = StripComments(File.ReadAllText(f));
            foreach (Match m in Regex.Matches(text, @"\bbool\s+(?<n>\w+)\s*(?:\([^)]*\))?\s*(?:=>|=|\{)(?<rest>[^;]*;)", RegexOptions.ExplicitCapture, Timeout))
                defs.Add((m.Groups["n"].Value, m.Groups["rest"].Value));
        }
        var predicates = new HashSet<string>();
        for (bool grew = true; grew;)
        {
            grew = false;
            foreach ((string name, string rest) in defs)
                if (!predicates.Contains(name)
                    && (rest.Contains("CpuFeatures", StringComparison.Ordinal) || predicates.Any(p => rest.Contains(p, StringComparison.Ordinal))))
                    grew |= predicates.Add(name);
        }

        output.WriteLine($"guard predicates: {predicates.Count}");
        var bad = new List<string>();
        foreach (Method m in methods)
        {
            if (m.IsPrivate || TierName.IsMatch(m.Name)) continue;
            if (m.Body.Contains("CpuFeatures.", StringComparison.Ordinal) || predicates.Any(p => Regex.IsMatch(m.Body, @"\b" + p + @"\b", RegexOptions.ExplicitCapture, Timeout))) continue;
            string[] called = tierKernels.Where(k => k != m.Name && Regex.IsMatch(m.Body, @"\b" + k + @"\s*(?:<[^>]*>)?\(", RegexOptions.ExplicitCapture, Timeout)).ToArray();
            if (called.Length > 0) bad.Add($"{m.File}:{m.Line} {m.Name} -> {string.Join(", ", called.Take(4))}");
        }
        foreach (string b in bad) output.WriteLine(b);
        Assert.True(bad.Count == 0,
            "These non-private methods call tier-named intrinsic kernels without consulting CpuFeatures (or a predicate built on it):\n"
            + string.Join('\n', bad));
    }
}
