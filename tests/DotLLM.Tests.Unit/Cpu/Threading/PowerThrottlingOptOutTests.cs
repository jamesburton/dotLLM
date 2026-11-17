using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Threading;

/// <summary>
/// Every thread that runs a <see cref="ComputeThreadPool"/> dispatch (each worker and the caller) must have opted out of
/// Windows execution-speed power throttling (EcoQoS), as llama.cpp's compute threads do (#1675). The check reads the
/// state back from the OS with <c>GetThreadInformation</c>, so it fails if the request was never made or made wrongly.
/// </summary>
public sealed unsafe class PowerThrottlingOptOutTests
{
    private const uint ExecutionSpeed = 0x1;

    private struct Ctx
    {
        public uint* Control;
        public uint* State;
        public int* Ok;
    }

    private static void Record(nint ctxPtr, int threadIdx, int threadCount)
    {
        var ctx = (Ctx*)ctxPtr;
        var s = CpuAffinity.QueryExecutionSpeedThrottling();
        if (s is { } v)
        {
            ctx->Control[threadIdx] = v.ControlMask;
            ctx->State[threadIdx] = v.StateMask;
            ctx->Ok[threadIdx] = 1;
        }
    }

    [SkippableTheory]
    [InlineData(DispatchMode.EventBased)]
    [InlineData(DispatchMode.SpinWait)]
    public void EveryDispatchThread_HasOptedOutOfExecutionSpeedThrottling(DispatchMode mode)
    {
        Skip.IfNot(OperatingSystem.IsWindows(), "Windows power throttling only.");
        Skip.IfNot(CpuAffinity.PowerThrottleOptOutEnabled, "DOTLLM_CPU_POWER_THROTTLE_OPTOUT=0 in this environment.");
        Skip.If(CpuAffinity.QueryExecutionSpeedThrottling() is null, "GetThreadInformation(ThreadPowerThrottling) unsupported.");

        const int threads = 4;
        var observed = new (uint Control, uint State, int Ok)[threads];
        (uint Control, uint State)? callerBefore = null;

        // A fresh dedicated caller thread, explicitly put INTO EcoQoS first: the pool must take it back out.
        // (A fresh thread also has a fresh per-thread "already opted out" flag.)
        var caller = new Thread(() =>
        {
            uint* control = stackalloc uint[threads];
            uint* state = stackalloc uint[threads];
            int* ok = stackalloc int[threads];
            Assert.True(OperatingSystem.IsWindows() && CpuAffinity.SetExecutionSpeedThrottling(throttled: true));
            callerBefore = CpuAffinity.QueryExecutionSpeedThrottling();

            using var pool = new ComputeThreadPool(threads, topology: null, new ThreadingConfig(threads, DecodeThreadCount: threads));
            pool.SetDispatchMode(mode);
            var ctx = new Ctx { Control = control, State = state, Ok = ok };
            for (int i = 0; i < threads; i++) { control[i] = 0xFFFF; state[i] = 0xFFFF; ok[i] = 0; }
            pool.Dispatch((nint)(&ctx), &Record);
            for (int i = 0; i < threads; i++) observed[i] = (control[i], state[i], ok[i]);
        });
        caller.Start();
        caller.Join();

        // The pre-condition the test depends on: the caller really was throttled before the pool touched it.
        Assert.Equal((ExecutionSpeed, ExecutionSpeed), callerBefore);
        for (int i = 0; i < threads; i++)
        {
            Assert.True(observed[i].Ok == 1, $"thread {i} did not run or could not query");
            Assert.True((observed[i].Control & ExecutionSpeed) != 0, $"thread {i}: EXECUTION_SPEED not under explicit control (control=0x{observed[i].Control:X})");
            Assert.True((observed[i].State & ExecutionSpeed) == 0, $"thread {i}: still throttled (state=0x{observed[i].State:X})");
        }
    }
}
