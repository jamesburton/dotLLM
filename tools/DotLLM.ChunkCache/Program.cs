using System.Globalization;
using System.Text;
using DotLLM.ChunkCache;

// chunkcache: see ChunkCacheCli.Usage for the verbs and the exit codes (0 ok, 1 error, 2 usage, 3 strict failure).
// stdout and stderr are UTF-8 without a BOM whatever the console code page, so non-ASCII paths and chunk text survive a
// redirected pipe (a wrapper parsing --json).
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var utf8 = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
using var stdout = new StreamWriter(Console.OpenStandardOutput(), utf8) { AutoFlush = true };
using var stderr = new StreamWriter(Console.OpenStandardError(), utf8) { AutoFlush = true };
if (args.Length > 0 && args[0] == "mcp")
{
    // The MCP server owns stdin/stdout as raw frames (UTF-8, newline-delimited JSON-RPC); only --help prints text there.
    using Stream stdin = Console.OpenStandardInput();
    using var frames = Console.OpenStandardOutput();
    // Anything else that writes to Console.Out in THIS process (libraries, analyzers and generators the solution scanner loads)
    // goes to stderr: a stray line on stdout would corrupt the frame stream. Roslyn's MSBuild BuildHost is a child process
    // talking over its own pipe; this redirect does not cover a child that inherits and writes to stdout.
    Console.SetOut(stderr);
    using var cancel = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancel.Cancel(); };
    return await McpCommand.RunAsync(args[1..], stdin, frames, stderr, stdout, Environment.CurrentDirectory, null, null, cancel.Token);
}
return ChunkCacheCli.Run(args, stdout, stderr);
