using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains.InProcess.Emit;
using DotLLM.Benchmarks.Columns;
using DotLLM.Benchmarks.Profile;

// Subcommands not understood by BenchmarkSwitcher are routed first.
if (args.Length > 0 && args[0] == "profile-cuda-decode")
{
    return CudaDecodeProfile.Run(args[1..]);
}
if (args.Length > 0 && args[0] == "profile-vulkan-host-import")
{
    return VulkanHostImportProfile.Run(args[1..]);
}

// Mutator job: run benchmarks in-process so BDN skips the separate subprocess
// build step entirely. This avoids the 2-minute build timeout that DotLLM.Cuda's
// PTX compilation exceeds. AsMutator() patches attribute-defined jobs rather than
// adding a duplicate run.
var inProcessMutator = new Job("InProcess")
    .WithToolchain(InProcessEmitToolchain.Instance)
    .AsMutator();

var config = ManualConfig.Create(DefaultConfig.Instance)
    .AddColumn(new PrefillTokPerSecColumn())
    .AddColumn(new DecodeTokPerSecColumn())
    .AddJob(inProcessMutator);

BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args, config);
return 0;
