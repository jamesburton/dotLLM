using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;
using DotLLM.Benchmarks.Columns;
using DotLLM.Benchmarks.Profile;

// Subcommands not understood by BenchmarkSwitcher are routed first.
if (args.Length > 0 && args[0] == "profile-cuda-decode")
{
    return CudaDecodeProfile.Run(args[1..]);
}

var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddColumn(new PrefillTokPerSecColumn())
    .AddColumn(new DecodeTokPerSecColumn());

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
return 0;
