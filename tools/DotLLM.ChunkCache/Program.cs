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
return ChunkCacheCli.Run(args, stdout, stderr);
