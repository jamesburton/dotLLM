using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #484. Covers <see cref="CudaKernels.ResolveAndValidatePtxDirectory"/> — the pre-flight
/// check every CUDA model factory now runs <b>before</b> <c>CudaContext.Create</c>, so that a bad
/// or incomplete PTX deployment cannot orphan a context/stream/cuBLAS handle.
/// </summary>
/// <remarks>
/// These tests need no GPU and no CUDA driver: the validator is pure file-system I/O, which is the
/// whole point of it running before any driver call. They therefore stay out of
/// <see cref="CudaCollection"/> and run on every machine, including CI without NVIDIA hardware.
/// </remarks>
public sealed class CudaKernelsPtxValidationTests : IDisposable
{
    private readonly string _scratch;

    /// <summary>Creates an isolated scratch directory for the synthetic PTX trees below.</summary>
    public CudaKernelsPtxValidationTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-ptx-validate-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private string MakeCompletePtxDir(string name)
    {
        string dir = Path.Combine(_scratch, name);
        Directory.CreateDirectory(dir);
        foreach (string f in CudaKernels.RequiredPtxFiles)
            File.WriteAllText(Path.Combine(dir, f), "// placeholder\n");
        return dir;
    }

    [Fact]
    public void MissingDirectory_ThrowsDirectoryNotFound()
    {
        string missing = Path.Combine(_scratch, "no-such-ptx-dir");

        // DirectoryNotFoundException specifically: both bad-ptxDir leak regression tests
        // (CudaMamba3FactoryLeakTests, CudaNemotronHTransformerModelForwardTests) assert on that
        // exact type, and they used to get it incidentally from File.ReadAllBytes inside the
        // CudaKernels constructor. Moving the failure earlier must not change the type.
        var ex = Assert.Throws<DirectoryNotFoundException>(
            () => CudaKernels.ResolveAndValidatePtxDirectory(missing));
        Assert.Contains(missing, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void CompleteDirectory_ReturnsItUnchanged()
    {
        string dir = MakeCompletePtxDir("complete");
        Assert.Equal(dir, CudaKernels.ResolveAndValidatePtxDirectory(dir));
    }

    [Fact]
    public void DirectoryMissingOneRequiredFile_ThrowsFileNotFoundNamingIt()
    {
        // Discriminating against a "does the directory exist?"-only check: the directory is
        // present and holds 32 of the 33 required modules. A validator that only stats the
        // directory would pass this and let the load reach CudaContext.Create — exactly the
        // ordering #484 removes.
        string dir = MakeCompletePtxDir("incomplete");
        string victim = CudaKernels.RequiredPtxFiles[^1];
        File.Delete(Path.Combine(dir, victim));

        var ex = Assert.Throws<FileNotFoundException>(
            () => CudaKernels.ResolveAndValidatePtxDirectory(dir));
        Assert.Contains(victim, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EmptyDirectory_ThrowsFileNotFound()
    {
        string dir = Path.Combine(_scratch, "empty");
        Directory.CreateDirectory(dir);
        Assert.Throws<FileNotFoundException>(() => CudaKernels.ResolveAndValidatePtxDirectory(dir));
    }

    [Fact]
    public void NullPtxDir_ResolvesToBaseDirectoryPtx()
    {
        // Null must resolve exactly as the old `ptxDir ??= Path.Combine(AppContext.BaseDirectory,
        // "ptx")` lines did. The resolved path is reported through the exception when that
        // directory is absent (the common case on a machine without a native PTX build), and
        // returned directly when it is present.
        string expected = Path.Combine(AppContext.BaseDirectory, "ptx");
        if (Directory.Exists(expected) && CudaKernels.RequiredPtxFiles.All(f => File.Exists(Path.Combine(expected, f))))
        {
            Assert.Equal(expected, CudaKernels.ResolveAndValidatePtxDirectory(null));
        }
        else
        {
            var ex = Assert.ThrowsAny<IOException>(() => CudaKernels.ResolveAndValidatePtxDirectory(null));
            Assert.Contains(expected, ex.Message, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Keeps <see cref="CudaKernels.RequiredPtxFiles"/> honest by parsing the constructor body and
    /// asserting set equality with every <b>unconditional</b> <c>CudaModule.LoadFromFile</c> call.
    /// </summary>
    /// <remarks>
    /// Without this, the list is a second source of truth that silently rots the next time a
    /// required module is added to the constructor: the pre-flight check would pass, the load
    /// would reach <c>CudaContext.Create</c> and then throw — re-opening precisely the window
    /// #484 closed. Optional modules (loaded behind a <c>File.Exists</c> guard) are excluded by
    /// construction: they are loaded from a local path variable, never from an inline
    /// <c>Path.Combine(ptxDir, "...")</c> argument.
    /// </remarks>
    [SkippableFact]
    public void RequiredPtxFiles_MatchesUnconditionalConstructorLoads()
    {
        string? src = FindCudaKernelsSource();
        Skip.If(src is null, "CudaKernels.cs not reachable from the test output directory.");

        string text = File.ReadAllText(src!);
        int ctorStart = text.IndexOf("public CudaKernels(string ptxDir)", StringComparison.Ordinal);
        Assert.True(ctorStart >= 0, "Could not locate the CudaKernels constructor.");
        string ctorBody = ExtractBracedBlock(text, ctorStart);

        var found = ScanUnconditionalLoads(ctorBody);

        Assert.NotEmpty(found);
        var declared = CudaKernels.RequiredPtxFiles.ToHashSet(StringComparer.Ordinal);

        Assert.Equal(CudaKernels.RequiredPtxFiles.Count, declared.Count); // no duplicates in the list
        Assert.True(found.SetEquals(declared),
            "CudaKernels.RequiredPtxFiles is out of sync with the constructor. "
            + $"Loaded but not declared: [{string.Join(", ", found.Except(declared).Order())}]. "
            + $"Declared but not loaded: [{string.Join(", ", declared.Except(found).Order())}].");
    }

    /// <summary>
    /// Collects the file names of every bare
    /// <c>CudaModule.LoadFromFile(Path.Combine(ptxDir, "name.ptx"))</c> call in
    /// <paramref name="ctorBody"/>. The inline <c>Path.Combine(ptxDir, "...")</c> argument is what
    /// makes a load unconditional: optional modules go through a local path variable that was
    /// first tested with <c>File.Exists</c>, so they never match this shape.
    /// </summary>
    private static HashSet<string> ScanUnconditionalLoads(string ctorBody)
    {
        const string Prefix = "CudaModule.LoadFromFile(Path.Combine(ptxDir, \"";
        var found = new HashSet<string>(StringComparer.Ordinal);
        int i = 0;
        while ((i = ctorBody.IndexOf(Prefix, i, StringComparison.Ordinal)) >= 0)
        {
            int nameStart = i + Prefix.Length;
            int nameEnd = ctorBody.IndexOf('"', nameStart);
            Assert.True(nameEnd > nameStart, "Unterminated PTX file name literal in the constructor.");
            found.Add(ctorBody[nameStart..nameEnd]);
            i = nameEnd;
        }
        return found;
    }

    private static string ExtractBracedBlock(string text, int fromIndex)
    {
        int open = text.IndexOf('{', fromIndex);
        Assert.True(open >= 0);
        int depth = 0;
        for (int i = open; i < text.Length; i++)
        {
            if (text[i] == '{') depth++;
            else if (text[i] == '}' && --depth == 0) return text[open..i];
        }
        throw new InvalidOperationException("Unbalanced braces while scanning the constructor body.");
    }

    private static string? FindCudaKernelsSource()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "src", "DotLLM.Cuda", "CudaKernels.cs");
            if (File.Exists(candidate)) return candidate;
            dir = dir.Parent;
        }
        return null;
    }
}
