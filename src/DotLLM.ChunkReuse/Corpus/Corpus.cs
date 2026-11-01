using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using DotLLM.Tokenizers;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>A file the corpus loader skipped, and why (<c>generated</c>, <c>test</c> or <c>build-output</c>).</summary>
public sealed record ExcludedFile(string Path, string Reason);

/// <summary>
/// A chunked C# code corpus: source files in ordinal path order, their chunks (fixed once by an <see cref="IChunker"/>), the
/// model tokenizer, and — built lazily for the task generator — one Roslyn compilation over every file with metadata
/// references to the BCL only (the running runtime's own assemblies). Symbols that need any other reference do not resolve
/// and are skipped downstream, never guessed.
/// </summary>
public sealed class Corpus
{
    /// <summary>
    /// The global usings the .NET SDK generates for <c>&lt;ImplicitUsings&gt;enable&lt;/ImplicitUsings&gt;</c> in
    /// <c>Microsoft.NET.Sdk</c> projects (written to <c>obj/</c>, which the loader excludes).
    /// </summary>
    public const string SdkImplicitUsings =
        "global using global::System;\nglobal using global::System.Collections.Generic;\nglobal using global::System.IO;\n" +
        "global using global::System.Linq;\nglobal using global::System.Net.Http;\nglobal using global::System.Threading;\n" +
        "global using global::System.Threading.Tasks;\n";

    private readonly SourceFile[] _files;
    private readonly Dictionary<string, int> _fileIndex;
    private readonly Chunk[][] _chunksByFile;
    private readonly Chunk[] _chunks;
    private readonly bool _implicitUsings;
    private CSharpCompilation? _compilation;
    private Dictionary<string, SyntaxTree>? _trees;
    private Dictionary<string, List<(string Name, int Position)>>? _declarations;

    /// <summary>Chunks every file (ordinal path order).</summary>
    /// <param name="files">The files; paths must be unique.</param>
    /// <param name="chunker">Chunker applied to each file.</param>
    /// <param name="tokenizer">The tokenizer the chunker uses (tasks encode prefixes and continuations with it).</param>
    /// <param name="implicitUsings">Add <see cref="SdkImplicitUsings"/> to the compilation (most current SDK projects enable them).</param>
    /// <param name="excluded">Files the loader skipped, for the record.</param>
    public Corpus(IEnumerable<SourceFile> files, IChunker chunker, ITokenizer tokenizer, bool implicitUsings = true,
        IEnumerable<ExcludedFile>? excluded = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(chunker);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _files = files.OrderBy(f => f.Path, StringComparer.Ordinal).ToArray();
        _fileIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < _files.Length; i++)
            if (!_fileIndex.TryAdd(_files[i].Path, i))
                throw new ArgumentException($"Duplicate file path '{_files[i].Path}'.", nameof(files));
        Chunker = chunker;
        Tokenizer = tokenizer;
        _implicitUsings = implicitUsings;
        _chunksByFile = _files.Select(f => chunker.Chunk(f).ToArray()).ToArray();
        _chunks = _chunksByFile.SelectMany(c => c).ToArray();
        Excluded = (excluded ?? []).ToArray();
    }

    /// <summary>
    /// Loads every <paramref name="pattern"/> file under <paramref name="directory"/>, skipping build output (<c>bin</c>,
    /// <c>obj</c>, <c>.git</c>, <c>.vs</c>, <c>node_modules</c>), test and benchmark projects (a directory named <c>test</c>,
    /// <c>tests</c>, <c>*.Tests</c>, <c>*.Test</c>, <c>*Tests</c>, <c>benchmarks</c> or <c>*.Benchmarks</c>) and generated
    /// code (<see cref="SymbolChunker.IsGenerated"/>). Text is read as UTF-8 with CRLF → LF.
    /// </summary>
    public static Corpus FromDirectory(string directory, IChunker chunker, ITokenizer tokenizer, string pattern = "*.cs",
        bool implicitUsings = true)
    {
        var files = new List<SourceFile>();
        var excluded = new List<ExcludedFile>();
        foreach (string full in Directory.EnumerateFiles(directory, pattern, SearchOption.AllDirectories))
        {
            string rel = Path.GetRelativePath(directory, full).Replace('\\', '/');
            string[] dirs = rel.Split('/')[..^1];
            if (dirs.Any(d => d is "bin" or "obj" or ".git" or ".vs" or "node_modules"))
            {
                excluded.Add(new ExcludedFile(rel, "build-output"));
                continue;
            }
            if (dirs.Any(IsTestDirectory))
            {
                excluded.Add(new ExcludedFile(rel, "test"));
                continue;
            }
            var file = new SourceFile(rel, File.ReadAllText(full).Replace("\r\n", "\n", StringComparison.Ordinal));
            if (SymbolChunker.IsGenerated(file))
            {
                excluded.Add(new ExcludedFile(rel, "generated"));
                continue;
            }
            files.Add(file);
        }
        return new Corpus(files, chunker, tokenizer, implicitUsings, excluded.OrderBy(e => e.Path, StringComparer.Ordinal));
    }

    private static bool IsTestDirectory(string name)
    {
        string d = name.ToLowerInvariant();
        return d is "test" or "tests" or "benchmarks" || d.EndsWith(".tests", StringComparison.Ordinal)
            || d.EndsWith(".test", StringComparison.Ordinal) || d.EndsWith("tests", StringComparison.Ordinal)
            || d.EndsWith(".benchmarks", StringComparison.Ordinal);
    }

    /// <summary>The corpus files in ordinal path order.</summary>
    public IReadOnlyList<SourceFile> Files => _files;

    /// <summary>Files the loader skipped.</summary>
    public IReadOnlyList<ExcludedFile> Excluded { get; }

    /// <summary>The chunker used.</summary>
    public IChunker Chunker { get; }

    /// <summary>The tokenizer used.</summary>
    public ITokenizer Tokenizer { get; }

    /// <summary>All chunks: files in ordinal order, chunks in text order.</summary>
    public IReadOnlyList<Chunk> Chunks => _chunks;

    /// <summary>Sum of chunk token counts.</summary>
    public long TotalTokens => _chunks.Sum(c => (long)c.TokenIds.Length);

    /// <summary>Chunks of file <paramref name="path"/> in text order.</summary>
    public IReadOnlyList<Chunk> ChunksOf(string path) => _chunksByFile[_fileIndex[path]];

    /// <summary>The file at <paramref name="path"/>.</summary>
    public SourceFile FileAt(string path) => _files[_fileIndex[path]];

    /// <summary>The chunk of <paramref name="path"/> holding character <paramref name="offset"/>, or null.</summary>
    public Chunk? ChunkAt(string path, int offset)
    {
        foreach (Chunk c in ChunksOf(path))
            if (offset >= c.StartOffset && offset < c.EndOffset) return c;
        return null;
    }

    /// <summary>The text of <paramref name="chunk"/> (a slice of its file).</summary>
    public string TextOf(Chunk chunk) => FileAt(chunk.FilePath).Text[chunk.StartOffset..chunk.EndOffset];

    /// <summary>
    /// SHA-256 (lowercase hex) of the chunked corpus: the chunker name, then per chunk its path (UTF-8, NUL-terminated), its
    /// token count and its ids (little-endian int32). Identifies the exact chunks a run used.
    /// </summary>
    public string ChunkIdHash
    {
        get
        {
            using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            sha.AppendData(Encoding.UTF8.GetBytes(Chunker.Name + "\0"));
            foreach (Chunk c in _chunks)
            {
                sha.AppendData(Encoding.UTF8.GetBytes(c.FilePath + "\0"));
                sha.AppendData(BitConverter.GetBytes(c.TokenIds.Length));
                sha.AppendData(MemoryMarshal.AsBytes(c.TokenIds.AsSpan()));
            }
            return Convert.ToHexStringLower(sha.GetHashAndReset());
        }
    }

    /// <summary>The Roslyn compilation over every corpus file (+ implicit usings), BCL references only. Built on first use.</summary>
    internal CSharpCompilation Compilation
    {
        get
        {
            if (_compilation is not null) return _compilation;
            (_compilation, _trees) = CreateCompilation(_files, _implicitUsings);
            return _compilation;
        }
    }

    /// <summary>
    /// One Roslyn compilation over <paramref name="files"/> (+ <see cref="SdkImplicitUsings"/> when
    /// <paramref name="implicitUsings"/>), BCL references only, and each file's syntax tree by path (the implicit-usings tree
    /// is not a corpus file and is not in the map).
    /// </summary>
    internal static (CSharpCompilation Compilation, Dictionary<string, SyntaxTree> Trees) CreateCompilation(
        IEnumerable<SourceFile> files, bool implicitUsings)
    {
        var trees = new Dictionary<string, SyntaxTree>(StringComparer.Ordinal);
        foreach (SourceFile f in files)
            if (!trees.TryAdd(f.Path, CSharpSyntaxTree.ParseText(f.Text, SymbolChunker.ParseOptions, f.Path)))
                throw new ArgumentException($"Duplicate file path '{f.Path}'.", nameof(files));
        IEnumerable<SyntaxTree> all = trees.Values;
        if (implicitUsings)
            all = all.Append(CSharpSyntaxTree.ParseText(SdkImplicitUsings, SymbolChunker.ParseOptions, "<implicit-usings>"));
        var compilation = CSharpCompilation.Create("corpus", all, BclReferences.Value,
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary, allowUnsafe: true,
                nullableContextOptions: NullableContextOptions.Enable, concurrentBuild: false));
        return (compilation, trees);
    }

    /// <summary>The syntax tree of <paramref name="path"/> in <see cref="Compilation"/>.</summary>
    internal SyntaxTree TreeOf(string path)
    {
        _ = Compilation;
        return _trees![path];
    }

    /// <summary>True when <paramref name="path"/> is a corpus file.</summary>
    internal bool Contains(string path) => _fileIndex.ContainsKey(path);

    /// <summary>Whether <paramref name="chunk"/> declares a member or type named <paramref name="name"/> (identifier inside it).</summary>
    internal bool Declares(Chunk chunk, string name)
    {
        _declarations ??= BuildDeclarations();
        if (!_declarations.TryGetValue(chunk.FilePath, out var decls)) return false;
        foreach ((string n, int pos) in decls)
            if (pos >= chunk.StartOffset && pos < chunk.EndOffset && string.Equals(n, name, StringComparison.Ordinal)) return true;
        return false;
    }

    private Dictionary<string, List<(string, int)>> BuildDeclarations()
    {
        var map = new Dictionary<string, List<(string, int)>>(StringComparer.Ordinal);
        foreach (SourceFile f in _files)
        {
            var list = new List<(string, int)>();
            foreach (SyntaxNode n in TreeOf(f.Path).GetRoot().DescendantNodes())
            {
                SyntaxToken id = n switch
                {
                    MethodDeclarationSyntax m => m.Identifier,
                    BaseTypeDeclarationSyntax t => t.Identifier,
                    PropertyDeclarationSyntax p => p.Identifier,
                    EventDeclarationSyntax e => e.Identifier,
                    DelegateDeclarationSyntax d => d.Identifier,
                    VariableDeclaratorSyntax v when v.Parent?.Parent is BaseFieldDeclarationSyntax => v.Identifier,
                    LocalFunctionStatementSyntax l => l.Identifier,
                    _ => default,
                };
                if (id.RawKind != 0) list.Add((id.ValueText, id.SpanStart));
            }
            map[f.Path] = list;
        }
        return map;
    }

    private static readonly Lazy<MetadataReference[]> BclReferences = new(() =>
    {
        string runtimeDir = RuntimeEnvironment.GetRuntimeDirectory();
        string tpa = AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES") as string ?? "";
        return tpa.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries)
            .Where(p => string.Equals(Path.GetDirectoryName(p)?.TrimEnd(Path.DirectorySeparatorChar),
                                      runtimeDir.TrimEnd(Path.DirectorySeparatorChar), StringComparison.OrdinalIgnoreCase))
            .Where(p =>
            {
                string n = Path.GetFileName(p);
                return n.StartsWith("System.", StringComparison.OrdinalIgnoreCase) || n is "mscorlib.dll" or "netstandard.dll"
                    || n.StartsWith("Microsoft.Win32.", StringComparison.OrdinalIgnoreCase) || n == "Microsoft.CSharp.dll"
                    || n == "Microsoft.VisualBasic.dll";
            })
            .Where(p => !p.EndsWith(".Native.dll", StringComparison.OrdinalIgnoreCase))
            .Order(StringComparer.OrdinalIgnoreCase)
            .Select(p => (MetadataReference)MetadataReference.CreateFromFile(p))
            .ToArray();
    });
}
