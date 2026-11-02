using System.Globalization;
using DotLLM.ChunkCache;

// chunkcache: see ChunkCacheCli.Usage for the verbs and the exit codes (0 ok, 1 error, 2 usage, 3 strict failure).
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
return ChunkCacheCli.Run(args, Console.Out, Console.Error);
