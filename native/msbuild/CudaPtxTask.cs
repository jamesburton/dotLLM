// Inline MSBuild task (RoslynCodeTaskFactory) behind native/msbuild/CudaPtx.targets. Issue #1635.
// Kept to C# 7.3 syntax: Visual Studio's .NET Framework MSBuild compiles it too.
//
// Contract: in the default "auto" mode this task must never fail the build because the machine's
// CUDA SDK or host compiler is newer than the repo baseline. It picks the most compatible
// toolchain it can find, escalates to override flags only on a host-compiler failure, keeps the
// emitted PTX at or below the baseline ISA version, and otherwise warns and leaves committed
// PTX alone. See docs/CUDA.md, "Building on newer CUDA SDKs".
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Build.Framework;
using Microsoft.Build.Utilities;

public class DotLLMCompileCudaPtx : Task
{
    [Required] public ITaskItem[] Kernels { get; set; }
    [Required] public string PtxOutputDir { get; set; }
    [Required] public string RepoRoot { get; set; }
    [Required] public string Mode { get; set; }
    public string ToolkitPath { get; set; }
    public string MsvcBinDir { get; set; }
    [Required] public string VersionPolicy { get; set; }
    [Required] public string BaselineCudaVersion { get; set; }
    [Required] public string BaselinePtxVersion { get; set; }
    public bool ForceAll { get; set; }
    public bool AllowUnsupportedCompiler { get; set; }
    [Output] public ITaskItem[] NewPtxFiles { get; set; }

    private const string StlOverride = "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH";
    private const string SetupHint =
        "Install the baseline side by side with scripts/setup-cuda-baseline.ps1, or see docs/CUDA.md \"Building on newer CUDA SDKs\".";

    private static readonly bool IsWindows = Path.DirectorySeparatorChar == '\\';
    private static readonly Regex ReleaseRx = new Regex(@"release (\d+)\.(\d+)", RegexOptions.Compiled);
    private static readonly Regex PtxVersionRx = new Regex(@"^\.version\s+(\d+)\.(\d+)", RegexOptions.Compiled | RegexOptions.Multiline);
    private static readonly Regex ToolchainErrorRx = new Regex(
        @"unsupported Microsoft Visual Studio version|STL1002|_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH|"
        + @"unsupported OS|targets unsupported|unsupported GNU version|unsupported clang version|"
        + @"Cannot find compiler|cl\.exe.*not (found|recognized)|Unsupported gpu architecture|"
        + @"host_config\.h|yvals_core\.h|Microsoft Visual Studio\\[^\r\n]*\\include\\|traditional preprocessor|"
        + @"libcu\+\+ requires|CCCL requires",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    private sealed class Toolkit
    {
        public string Root, Nvcc, Ptxas, Source;
        public Version Version;
        public override string ToString() { return "CUDA " + Version + " (" + Root + ", via " + Source + ")"; }
    }

    private sealed class Host
    {
        public string BinDir;
        public Version Version;
        public override string ToString() { return BinDir == null ? "default host compiler" : "MSVC " + Version; }
    }

    private sealed class Attempt
    {
        public Toolkit Toolkit;
        public Host Host;
        public bool Overrides;
        public string Key { get { return Toolkit.Root + "|" + (Host == null ? "" : Host.BinDir) + "|" + Overrides; } }
        public override string ToString() { return Toolkit + " + " + Host + (Overrides ? " + override flags" : ""); }
    }

    private bool Strict { get { return string.Equals(Mode, "compile", StringComparison.OrdinalIgnoreCase); } }
    private Version _basePtx;
    private Version _baseCuda;
    private Attempt _good;
    private readonly List<string> _degraded = new List<string>();

    private void Problem(string message)
    {
        if (Strict) Log.LogError("dotLLM CUDA: " + message);
        else Log.LogWarning("dotLLM CUDA: " + message);
    }

    public override bool Execute()
    {
        NewPtxFiles = new ITaskItem[0];
        string mode = (Mode ?? "auto").Trim().ToLowerInvariant();
        if (mode != "auto" && mode != "committed" && mode != "compile")
        {
            Log.LogWarning("dotLLM CUDA: unknown DotLLMCudaBuildMode '" + Mode + "'; using 'auto'.");
            mode = "auto";
            Mode = "auto";
        }
        string policy = (VersionPolicy ?? "pin").Trim().ToLowerInvariant();
        if (policy != "pin" && policy != "native" && policy != "strict")
        {
            Log.LogWarning("dotLLM CUDA: unknown DotLLMPtxVersionPolicy '" + VersionPolicy + "'; using 'pin'.");
            policy = "pin";
        }
        VersionPolicy = policy;
        if (!Version.TryParse(BaselinePtxVersion, out _basePtx)) _basePtx = new Version(8, 7);
        if (!Version.TryParse(BaselineCudaVersion, out _baseCuda)) _baseCuda = new Version(12, 8);

        List<ITaskItem> stale = SelectStale(mode);
        if (stale.Count == 0)
        {
            Log.LogMessage(MessageImportance.Low, "dotLLM CUDA: all PTX up to date; nvcc not needed.");
            return true;
        }

        if (mode == "committed")
        {
            foreach (ITaskItem k in stale)
            {
                string ptx = PtxPath(k);
                if (!File.Exists(ptx))
                    Log.LogWarning("dotLLM CUDA: " + Path.GetFileName(ptx) + " is missing and DotLLMCudaBuildMode=committed; "
                        + "kernel '" + Name(k) + "' will be unavailable at runtime.");
                else
                    Log.LogMessage(MessageImportance.High, "dotLLM CUDA: " + Name(k) + ".cu changed but DotLLMCudaBuildMode=committed; using the existing PTX.");
            }
            return true;
        }

        List<Toolkit> toolkits = DiscoverToolkits();
        if (toolkits.Count == 0)
        {
            foreach (ITaskItem k in stale)
            {
                bool has = File.Exists(PtxPath(k));
                Problem(Name(k) + ".cu needs compiling but no CUDA toolkit (nvcc) was found"
                    + (has ? "; using the existing, possibly stale, PTX. " : "; the kernel will be unavailable at runtime. ")
                    + SetupHint);
            }
            return !Log.HasLoggedErrors;
        }
        Log.LogMessage(MessageImportance.Normal, "dotLLM CUDA: toolkit candidates: " + string.Join("; ", toolkits.Select(t => t.ToString())));
        List<Host> hosts = DiscoverHosts();
        if (IsWindows)
            Log.LogMessage(MessageImportance.Normal, "dotLLM CUDA: host compiler candidates: " + string.Join("; ", hosts.Select(h => h.ToString() + (h.BinDir == null ? "" : " (" + h.BinDir + ")"))));

        var created = new List<ITaskItem>();
        foreach (ITaskItem k in stale)
        {
            bool existed = File.Exists(PtxPath(k));
            if (CompileOne(k, toolkits, hosts) && !existed)
                created.Add(new TaskItem(PtxPath(k)));
        }
        NewPtxFiles = created.ToArray();

        if (_degraded.Count > 0)
            Log.LogWarning("dotLLM CUDA: PTX was produced without the baseline toolchain (CUDA " + _baseCuda
                + " / PTX ISA " + _basePtx + "): " + string.Join("; ", _degraded)
                + ". It is valid, but do not commit it without checking. " + SetupHint);
        return !Log.HasLoggedErrors;
    }

    // ── stale selection ─────────────────────────────────────────────────────────────────────

    private string Name(ITaskItem k) { return Path.GetFileNameWithoutExtension(k.ItemSpec); }
    private string PtxPath(ITaskItem k) { return Path.Combine(PtxOutputDir, Name(k) + ".ptx"); }

    private List<ITaskItem> SelectStale(string mode)
    {
        var result = new List<ITaskItem>();
        HashSet<string> changed = mode == "compile" ? null : GitChangedPaths();
        foreach (ITaskItem k in Kernels)
        {
            string cu = k.GetMetadata("FullPath");
            string ptx = PtxPath(k);
            if (ForceAll || !File.Exists(ptx)) { result.Add(k); continue; }
            bool newer = File.GetLastWriteTimeUtc(cu) > File.GetLastWriteTimeUtc(ptx);
            if (!newer) continue;
            // The silent-recompile hazard: a checkout can leave an UNMODIFIED .cu newer than its
            // committed .ptx, and recompiling it with whatever toolkit is installed rewrites a
            // committed file (historically at the wrong ISA version: #124, #318). In auto mode a
            // .cu git reports as unchanged keeps its committed PTX.
            if (changed != null && !changed.Contains(NormalizeKey(cu)))
            {
                Log.LogMessage(MessageImportance.Low, "dotLLM CUDA: " + Name(k) + ".cu is newer than its PTX but unmodified per git; keeping the committed PTX.");
                continue;
            }
            result.Add(k);
        }
        return result;
    }

    private static string NormalizeKey(string path)
    {
        string full = Path.GetFullPath(path).Replace('\\', '/');
        return IsWindows ? full.ToLowerInvariant() : full;
    }

    /// <summary>Full paths of files git reports as modified/added/untracked, or null when git is unavailable.</summary>
    private HashSet<string> GitChangedPaths()
    {
        string stdout;
        int ec = Run("git", "-C \"" + RepoRoot.TrimEnd('\\', '/') + "\" status --porcelain --untracked-files=all -- native", 60000, out stdout);
        if (ec != 0) return null;
        var set = new HashSet<string>();
        string top;
        if (Run("git", "-C \"" + RepoRoot.TrimEnd('\\', '/') + "\" rev-parse --show-toplevel", 30000, out top) != 0) return null;
        top = top.Trim();
        foreach (string raw in stdout.Split('\n'))
        {
            string line = raw.TrimEnd('\r');
            if (line.Length < 4) continue;
            string p = line.Substring(3);
            int arrow = p.IndexOf(" -> ", StringComparison.Ordinal);
            if (arrow >= 0) p = p.Substring(arrow + 4);
            p = p.Trim('"');
            set.Add(NormalizeKey(Path.Combine(top, p)));
        }
        return set;
    }

    // ── toolchain discovery ─────────────────────────────────────────────────────────────────

    private List<Toolkit> DiscoverToolkits()
    {
        var roots = new List<KeyValuePair<string, string>>(); // root, source
        if (!string.IsNullOrWhiteSpace(ToolkitPath))
        {
            roots.Add(new KeyValuePair<string, string>(ToolkitPath.Trim(), "DotLLMCudaToolkitPath"));
        }
        else
        {
            string bv = _baseCuda.Major + "_" + _baseCuda.Minor;
            AddRoot(roots, Environment.GetEnvironmentVariable("CUDA_PATH_V" + bv), "CUDA_PATH_V" + bv + " (baseline)");
            foreach (string r in StandardRoots())
                if (DirVersion(r) == new Version(_baseCuda.Major, _baseCuda.Minor))
                    AddRoot(roots, r, "baseline install dir");
            AddRoot(roots, Environment.GetEnvironmentVariable("CUDA_PATH"), "CUDA_PATH");
            AddRoot(roots, Environment.GetEnvironmentVariable("CUDA_HOME"), "CUDA_HOME");
            string onPath = FindOnPath(IsWindows ? "nvcc.exe" : "nvcc");
            if (onPath != null) AddRoot(roots, Path.GetDirectoryName(Path.GetDirectoryName(onPath)), "PATH");
            var others = new List<string>();
            foreach (System.Collections.DictionaryEntry e in Environment.GetEnvironmentVariables())
                if (((string)e.Key).StartsWith("CUDA_PATH_V", StringComparison.OrdinalIgnoreCase)) others.Add((string)e.Value);
            others.AddRange(StandardRoots());
            foreach (string r in others.OrderByDescending(r => DirVersion(r) ?? new Version(0, 0)))
                AddRoot(roots, r, "installed toolkit");
        }

        var result = new List<Toolkit>();
        var seen = new HashSet<string>();
        foreach (var kv in roots)
        {
            string root;
            try { root = Path.GetFullPath(kv.Key.Trim().Trim('"')).TrimEnd('\\', '/'); } catch { continue; }
            if (!seen.Add(NormalizeKey(root))) continue;
            string nvcc = Path.Combine(root, "bin", IsWindows ? "nvcc.exe" : "nvcc");
            if (!File.Exists(nvcc))
            {
                if (kv.Value == "DotLLMCudaToolkitPath")
                    Problem("DotLLMCudaToolkitPath='" + ToolkitPath + "' has no bin/nvcc.");
                continue;
            }
            string outText;
            if (Run(nvcc, "--version", 60000, out outText) != 0) continue;
            Match m = ReleaseRx.Match(outText);
            if (!m.Success) continue;
            result.Add(new Toolkit
            {
                Root = root,
                Nvcc = nvcc,
                Ptxas = Path.Combine(root, "bin", IsWindows ? "ptxas.exe" : "ptxas"),
                Version = new Version(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value)),
                Source = kv.Value,
            });
        }
        return result;
    }

    private static void AddRoot(List<KeyValuePair<string, string>> roots, string root, string source)
    {
        if (!string.IsNullOrWhiteSpace(root)) roots.Add(new KeyValuePair<string, string>(root, source));
    }

    private static IEnumerable<string> StandardRoots()
    {
        var list = new List<string>();
        try
        {
            if (IsWindows)
            {
                string pf = Environment.GetEnvironmentVariable("ProgramFiles") ?? @"C:\Program Files";
                string baseDir = Path.Combine(pf, "NVIDIA GPU Computing Toolkit", "CUDA");
                if (Directory.Exists(baseDir)) list.AddRange(Directory.GetDirectories(baseDir, "v*"));
            }
            else
            {
                if (Directory.Exists("/usr/local")) list.AddRange(Directory.GetDirectories("/usr/local", "cuda-*"));
                if (Directory.Exists("/usr/local/cuda")) list.Add("/usr/local/cuda");
                if (Directory.Exists("/opt/cuda")) list.Add("/opt/cuda");
            }
        }
        catch { }
        return list;
    }

    private static Version DirVersion(string root)
    {
        if (string.IsNullOrEmpty(root)) return null;
        Match m = Regex.Match(Path.GetFileName(root.TrimEnd('\\', '/')), @"(\d+)\.(\d+)");
        return m.Success ? new Version(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value)) : null;
    }

    private static string FindOnPath(string exe)
    {
        foreach (string dir in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            try
            {
                string p = Path.Combine(dir.Trim().Trim('"'), exe);
                if (File.Exists(p)) return p;
            }
            catch { }
        }
        return null;
    }

    /// <summary>
    /// Host compilers to try, best first. Windows: an explicit DotLLMMsvcBinDir only; otherwise the
    /// newest MSVC toolset below 14.50 (what every CUDA 12.x/13.0/13.1 accepts) then the newest
    /// 14.50+ (VS 2026; needs CUDA 13.2+ or the override flags). Non-Windows: nvcc's default.
    /// </summary>
    private List<Host> DiscoverHosts()
    {
        var hosts = new List<Host>();
        if (!IsWindows) { hosts.Add(new Host()); return hosts; }
        if (!string.IsNullOrWhiteSpace(MsvcBinDir))
        {
            string dir = MsvcBinDir.Trim().Trim('"').TrimEnd('\\');
            if (!File.Exists(Path.Combine(dir, "cl.exe")))
                Problem("DotLLMMsvcBinDir='" + MsvcBinDir + "' has no cl.exe.");
            hosts.Add(new Host { BinDir = dir, Version = MsvcVersionOf(dir) ?? new Version(0, 0) });
            return hosts;
        }
        var found = new Dictionary<string, Host>(StringComparer.OrdinalIgnoreCase);
        string vcTools = Environment.GetEnvironmentVariable("VCToolsInstallDir");
        if (!string.IsNullOrEmpty(vcTools)) AddHost(found, Path.Combine(vcTools, "bin", "Hostx64", "x64"));
        string pf86 = Environment.GetEnvironmentVariable("ProgramFiles(x86)") ?? @"C:\Program Files (x86)";
        string vswhere = Path.Combine(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe");
        string outText;
        if (File.Exists(vswhere) && Run(vswhere, "-products * -find \"VC\\Tools\\MSVC\\*\\bin\\Hostx64\\x64\\cl.exe\"", 60000, out outText) == 0)
            foreach (string line in outText.Split('\n'))
                if (line.Trim().EndsWith("cl.exe", StringComparison.OrdinalIgnoreCase))
                    AddHost(found, Path.GetDirectoryName(line.Trim()));
        // Fallback scan (vswhere missing, or instances it does not report).
        foreach (string pf in new[] { pf86, Environment.GetEnvironmentVariable("ProgramFiles") ?? @"C:\Program Files" })
        {
            try
            {
                string vs = Path.Combine(pf, "Microsoft Visual Studio");
                if (!Directory.Exists(vs)) continue;
                foreach (string year in Directory.GetDirectories(vs))
                    foreach (string edition in Directory.GetDirectories(year))
                    {
                        string msvc = Path.Combine(edition, "VC", "Tools", "MSVC");
                        if (!Directory.Exists(msvc)) continue;
                        foreach (string ver in Directory.GetDirectories(msvc, "14.*"))
                            AddHost(found, Path.Combine(ver, "bin", "Hostx64", "x64"));
                    }
            }
            catch { }
        }
        var cut = new Version(14, 50);
        Host below = found.Values.Where(h => h.Version < cut).OrderByDescending(h => h.Version).FirstOrDefault();
        Host above = found.Values.Where(h => h.Version >= cut).OrderByDescending(h => h.Version).FirstOrDefault();
        if (below != null) hosts.Add(below);
        if (above != null) hosts.Add(above);
        if (hosts.Count == 0) hosts.Add(new Host()); // let nvcc look for cl.exe on PATH
        return hosts;
    }

    private static void AddHost(Dictionary<string, Host> found, string binDir)
    {
        try
        {
            binDir = Path.GetFullPath(binDir).TrimEnd('\\');
            if (!File.Exists(Path.Combine(binDir, "cl.exe")) || found.ContainsKey(binDir)) return;
            Version v = MsvcVersionOf(binDir);
            if (v != null) found[binDir] = new Host { BinDir = binDir, Version = v };
        }
        catch { }
    }

    private static Version MsvcVersionOf(string binDir)
    {
        // ...\VC\Tools\MSVC\<14.xx.yyyyy>\bin\Hostx64\x64
        Match m = Regex.Match(binDir, @"\\MSVC\\(\d+)\.(\d+)", RegexOptions.IgnoreCase);
        return m.Success ? new Version(int.Parse(m.Groups[1].Value), int.Parse(m.Groups[2].Value)) : null;
    }

    // ── compile ladder ──────────────────────────────────────────────────────────────────────

    private IEnumerable<Attempt> Ladder(List<Toolkit> toolkits, List<Host> hosts)
    {
        var seen = new HashSet<string>();
        if (_good != null) { seen.Add(_good.Key); yield return _good; }
        foreach (Toolkit tk in toolkits)
        {
            foreach (bool overrides in new[] { AllowUnsupportedCompiler, true })
                foreach (Host h in hosts)
                {
                    var a = new Attempt { Toolkit = tk, Host = h, Overrides = overrides };
                    if (seen.Add(a.Key)) yield return a;
                }
        }
    }

    private bool CompileOne(ITaskItem k, List<Toolkit> toolkits, List<Host> hosts)
    {
        string name = Name(k);
        string cu = k.GetMetadata("FullPath");
        string target = PtxPath(k);
        string arch = string.IsNullOrEmpty(k.GetMetadata("Arch")) ? "compute_75" : k.GetMetadata("Arch");
        string sm = arch.Replace("compute_", "sm_");
        string flags = string.Join(" ", new[] { k.GetMetadata("FastMathFlag"), k.GetMetadata("FmadFlag"), k.GetMetadata("CxxStdFlag") }
            .Where(f => !string.IsNullOrWhiteSpace(f)));
        Directory.CreateDirectory(PtxOutputDir);
        string tmpDir = Path.Combine(Path.GetTempPath(), "dotllm-ptx-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(tmpDir);
        var notes = new List<string>();
        string fallbackText = null; Attempt fallbackAttempt = null; // above-baseline PTX we could not pin
        try
        {
            Log.LogMessage(MessageImportance.High, "dotLLM CUDA: compiling " + name + ".cu -> PTX (" + arch + ")");
            foreach (Attempt a in Ladder(toolkits, hosts))
            {
                string outPtx = Path.Combine(tmpDir, name + ".ptx");
                if (File.Exists(outPtx)) File.Delete(outPtx);
                var args = new StringBuilder();
                args.Append("-ptx -arch=").Append(arch).Append(' ').Append(flags);
                if (a.Overrides) args.Append(" -allow-unsupported-compiler ").Append(StlOverride);
                // CUDA 13's CCCL (pulled in by cooperative_groups.h) hard-errors on MSVC's traditional
                // preprocessor; measured on CUDA 13.3 + MSVC 14.44 for every -std=c++17 kernel (#1635).
                if (IsWindows && a.Toolkit.Version.Major >= 13) args.Append(" -Xcompiler /Zc:preprocessor");
                if (a.Host != null && a.Host.BinDir != null) args.Append(" -ccbin \"").Append(a.Host.BinDir).Append('"');
                args.Append(" -o \"").Append(outPtx).Append("\" \"").Append(cu).Append('"');
                string output;
                int ec = Run(a.Toolkit.Nvcc, args.ToString(), 600000, out output);
                if (ec != 0 || !File.Exists(outPtx))
                {
                    if (!ToolchainErrorRx.IsMatch(output))
                    {
                        // A real kernel error: other toolchains will not fix it.
                        Problem(name + ".cu failed to compile with " + a + " (not a toolchain-version problem):\n" + Tail(output, 25)
                            + (File.Exists(target) ? "\nKeeping the existing PTX." : "\nThe kernel will be unavailable at runtime."));
                        return false;
                    }
                    notes.Add(a + ": " + FirstMatchLine(output));
                    continue;
                }

                string text = File.ReadAllText(outPtx);
                Match vm = PtxVersionRx.Match(text);
                Version ver = vm.Success ? new Version(int.Parse(vm.Groups[1].Value), int.Parse(vm.Groups[2].Value)) : null;
                if (ver == null) { notes.Add(a + ": emitted PTX has no .version"); continue; }

                if (ver > _basePtx)
                {
                    if (VersionPolicy == "native")
                    {
                        _degraded.Add(name + ": .version " + ver + " kept (DotLLMPtxVersionPolicy=native; needs a driver supporting CUDA " + a.Toolkit.Version + ")");
                    }
                    else if (VersionPolicy == "pin")
                    {
                        string pinned = PinVersion(text, ver, a.Toolkit);
                        string why;
                        if (ValidateWithPtxas(a.Toolkit, pinned, sm, tmpDir, out why))
                        {
                            text = pinned;
                            _degraded.Add(name + ": " + a + ", .version " + ver + " pinned to " + _basePtx + " (ptxas-validated)");
                        }
                        else
                        {
                            notes.Add(a + ": .version " + ver + " cannot be pinned to " + _basePtx + ": " + why);
                            if (fallbackText == null) { fallbackText = text; fallbackAttempt = a; }
                            continue;
                        }
                    }
                    else
                    {
                        notes.Add(a + ": .version " + ver + " exceeds the baseline " + _basePtx + " (DotLLMPtxVersionPolicy=strict)");
                        if (fallbackText == null) { fallbackText = text; fallbackAttempt = a; }
                        continue;
                    }
                }
                else if (a.Toolkit.Version != new Version(_baseCuda.Major, _baseCuda.Minor) || a.Overrides)
                {
                    _degraded.Add(name + ": " + a + ", .version " + ver);
                }

                if (a.Overrides && !AllowUnsupportedCompiler)
                    Log.LogWarning("dotLLM CUDA: " + name + ".cu needed -allow-unsupported-compiler " + StlOverride + " with " + a
                        + ". This toolkit does not support the available host compiler; an MSVC toolset below 14.50 (e.g. v14.44, VS 2022) avoids the overrides. " + SetupHint);
                _good = a;
                WriteIfChanged(target, text);
                Log.LogMessage(MessageImportance.High, "dotLLM CUDA:   " + name + ".ptx <- " + a);
                return true;
            }

            // Nothing produced acceptable PTX.
            string detail = notes.Count == 0 ? "" : "\n  " + string.Join("\n  ", notes);
            if (fallbackText != null && !Strict && !File.Exists(target))
            {
                WriteIfChanged(target, fallbackText);
                Log.LogWarning("dotLLM CUDA: " + name + ".ptx written at its native ISA version by " + fallbackAttempt
                    + " because no toolchain produced PTX at or below " + _basePtx + ". It loads only on drivers that support that"
                    + " version — DO NOT COMMIT it. " + SetupHint + detail);
                return true;
            }
            Problem(name + ".cu: no toolchain produced acceptable PTX"
                + (File.Exists(target) ? "; keeping the existing PTX. " : "; the kernel will be unavailable at runtime. ")
                + SetupHint + detail);
            return false;
        }
        finally
        {
            try { Directory.Delete(tmpDir, true); } catch { }
        }
    }

    private string PinVersion(string text, Version from, Toolkit tk)
    {
        string pinned = PtxVersionRx.Replace(text, ".version " + _basePtx.Major + "." + _basePtx.Minor, 1);
        int at = PtxVersionRx.Match(pinned).Index;
        return pinned.Insert(at, "// dotLLM: .version pinned from " + from + " to " + _basePtx + " (emitted by CUDA " + tk.Version
            + ", validated by its ptxas; docs/CUDA.md \"Building on newer CUDA SDKs\")\n");
    }

    /// <summary>
    /// ptxas rejects a PTX file that uses any feature newer than its declared .version
    /// ("Feature ... requires PTX ISA .version X or later"), so assembling the pinned text proves
    /// it is valid ISA at the baseline version — what an older driver's JIT will check.
    /// </summary>
    private bool ValidateWithPtxas(Toolkit tk, string pinnedText, string sm, string tmpDir, out string why)
    {
        why = null;
        if (!File.Exists(tk.Ptxas)) { why = "no ptxas next to nvcc"; return false; }
        string ptx = Path.Combine(tmpDir, "pinned.ptx");
        string cubin = Path.Combine(tmpDir, "pinned.cubin");
        File.WriteAllText(ptx, pinnedText);
        string output;
        int ec = Run(tk.Ptxas, "-arch=" + sm + " -o \"" + cubin + "\" \"" + ptx + "\"", 600000, out output);
        if (ec == 0) return true;
        why = FirstMatchLine(output);
        return false;
    }

    private static void WriteIfChanged(string path, string text)
    {
        if (File.Exists(path) && File.ReadAllText(path) == text) return;
        File.WriteAllText(path, text);
    }

    private static string FirstMatchLine(string output)
    {
        foreach (string l in output.Split('\n'))
            if (Regex.IsMatch(l, "error|fatal|unsupported|requires", RegexOptions.IgnoreCase))
                return l.Trim();
        return Tail(output, 1).Trim();
    }

    private static string Tail(string s, int lines)
    {
        string[] all = s.Replace("\r", "").Split('\n').Where(l => l.Trim().Length > 0).ToArray();
        return string.Join("\n", all.Skip(Math.Max(0, all.Length - lines)));
    }

    private int Run(string exe, string args, int timeoutMs, out string output)
    {
        var sb = new StringBuilder();
        try
        {
            var psi = new ProcessStartInfo(exe, args)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            using (var p = new Process { StartInfo = psi })
            {
                p.OutputDataReceived += (s, e) => { if (e.Data != null) lock (sb) sb.AppendLine(e.Data); };
                p.ErrorDataReceived += (s, e) => { if (e.Data != null) lock (sb) sb.AppendLine(e.Data); };
                p.Start();
                p.BeginOutputReadLine();
                p.BeginErrorReadLine();
                if (!p.WaitForExit(timeoutMs))
                {
                    try { p.Kill(); } catch { }
                    output = sb.ToString() + "\n(timed out)";
                    return -1;
                }
                p.WaitForExit();
                output = sb.ToString();
                return p.ExitCode;
            }
        }
        catch (Exception ex)
        {
            output = ex.Message;
            return -1;
        }
    }
}
