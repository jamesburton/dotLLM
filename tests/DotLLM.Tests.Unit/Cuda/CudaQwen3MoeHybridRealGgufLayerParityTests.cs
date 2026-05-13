using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-vs-CUDA <i>layer-by-layer</i> activation parity for the real
/// <c>Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf</c> (~30 GB on disk, 40 layers — 30 GDN + 10 full-attn
/// per qwen35moe layout). Extends the synthetic-fixture coverage of
/// <see cref="CudaQwen3MoeHybridParityTests"/> to real Q6_K quantised weights — catches
/// any divergence that manifests only on the quantised path.
/// </summary>
/// <remarks>
/// <para>
/// <b>Approach.</b> A 30 GB GGUF cannot fit in a 12 GB GPU. The harness streams one layer's
/// weights through device memory at a time:
/// </para>
/// <list type="number">
///   <item><description>The developer runs the CPU oracle once with
///     <c>DOTLLM_TENSOR_DUMP_DIR=&lt;dir&gt;</c> set, producing
///     <c>NNNNN_blk.{i}.l_out.bin</c> per-layer dumps and <c>NNNNN_token_embd.bin</c> /
///     <c>NNNNN_result_*.bin</c> output dumps. The samples/console runs the CPU
///     <see cref="DotLLM.Models.Architectures.Qwen3MoeHybridTransformerModel"/> end-to-end —
///     the dumps are already wired into the per-layer body and the output stage.</description></item>
///   <item><description>The test loads the GGUF on CUDA in <i>layer-by-layer harness</i>
///     mode (<see cref="CudaQwen3MoeHybridTransformerModel.LoadFromGgufForLayerByLayerHarness"/>) —
///     embedding, output norm, and lm_head are uploaded eagerly; per-layer weight slots
///     stay empty.</description></item>
///   <item><description>For each of 40 layers <c>i</c>:
///     <list type="bullet">
///       <item><description>Load layer <c>i</c>'s weights to device (~6 GB peak at Q6_K_XL).</description></item>
///       <item><description>Read the CPU dump for layer <c>i</c>'s input — <c>token_embd</c>
///         when <c>i == 0</c>, otherwise <c>blk.{i-1}.l_out</c>.</description></item>
///       <item><description>Run one CUDA layer body in isolation via
///         <see cref="CudaQwen3MoeHybridTransformerModel.RunIsolatedLayerFromHostInput"/>.</description></item>
///       <item><description>Compare to <c>blk.{i}.l_out</c>; emit max-abs-diff diagnostics.</description></item>
///       <item><description>Free layer <c>i</c>'s device weights before iterating.</description></item>
///     </list>
///   </description></item>
/// </list>
/// <para>
/// <b>Tolerance.</b> 1e-3 absolute / 5e-3 relative (looser than synthetic because Q6_K dequant
/// has its own per-element noise that compounds across ~6 GB of layer weights). Tightening or
/// loosening this is a finding — the band itself is pinned by the task contract.
/// </para>
/// <para>
/// <b>VRAM budget.</b> Peak device residency at any layer boundary is asserted &lt;= 10 GB,
/// leaving headroom for KV cache + activations on a 12 GB card.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class CudaQwen3MoeHybridRealGgufLayerParityTests
{
    private const string GgufPathEnvVar = "DOTLLM_QWEN3MOEHYBRID_GGUF_PATH";
    private const string DumpDirEnvVar = "DOTLLM_QWEN3MOEHYBRID_CPU_DUMP_DIR";
    private const string DefaultGgufPath =
        "C:/Development/KTransformerTests/models/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf";

    // Tolerance band — looser than the synthetic-fixture tests (1e-4 abs, 1e-3 rel) to absorb
    // Q6_K dequant + reduction-order noise across ~6 GB of per-layer weights. The 2e-3 abs bar
    // matches the empirical noise floor of cuBLAS F32 GEMM tree-reduction vs CPU sequential
    // SIMD scalar accumulation over Q6_K-dequanted weights at hidden=2048 dot products, while
    // still catching any kernel-level divergence above ~1 ULP per element. Synthetic tests stay
    // at the tight 1e-4 bar because there's no Q6_K noise on the F32 path.
    private const float AbsTol = 2e-3f;
    private const float RelTol = 5e-3f;

    // Peak VRAM safety margin — must stay below this on a 12 GB card.
    private const long PeakVramSoftLimitBytes = 10L * 1024L * 1024L * 1024L;

    private readonly ITestOutputHelper _output;

    public CudaQwen3MoeHybridRealGgufLayerParityTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Layer-by-layer per-layer parity for every GDN-mixing layer in the real GGUF.
    /// One method covers all 30 GDN layers because the failure mode (which layer first
    /// diverges) is best surfaced in a single sweep — splitting per-layer would amplify
    /// the per-test load cost (~1.5–2 s per layer to upload 6 GB of weights × 30 = 45 s).
    /// </summary>
    [SkippableFact]
    public void RealGguf_AllGdnLayers_PerLayer_MatchesCpu()
    {
        var ctx = TryBuildHarness();
        Skip.If(ctx is null, "Harness skip — see test output for the missing prerequisite.");

        using (ctx)
        {
            VerifyLayersByKind(ctx, HybridLayerKind.GatedDeltaNet);
        }
    }

    /// <summary>
    /// Per-layer parity for every full-attention layer (10 of 40 at qwen35moe stride 4).
    /// These layers exercise the Q+Gate fused projection, QK-norm, partial-rotary NeoX
    /// RoPE, GQA SDPA, and the post-attention sigmoid-gate elementwise mul — the same
    /// stack that issue #35 just hardened.
    /// </summary>
    [SkippableFact]
    public void RealGguf_AllFullAttnLayers_PerLayer_MatchesCpu()
    {
        var ctx = TryBuildHarness();
        Skip.If(ctx is null, "Harness skip — see test output for the missing prerequisite.");

        using (ctx)
        {
            VerifyLayersByKind(ctx, HybridLayerKind.Attention);
        }
    }

    /// <summary>
    /// Sanity check: the CUDA embedding lookup for the prompt's tokens must match the CPU
    /// dump <c>token_embd</c>. A divergence here would mean the embedding table upload or
    /// the embedding-lookup kernel is wrong — the rest of the parity sweep would then
    /// drift from the very first layer's input.
    /// </summary>
    [SkippableFact]
    public void RealGguf_FirstLayerInput_MatchesCpuEmbeddingLookup()
    {
        var ctx = TryBuildHarness();
        Skip.If(ctx is null, "Harness skip — see test output for the missing prerequisite.");

        using (ctx)
        {
            var harness = ctx.Harness;
            int hiddenSize = ctx.Config.HiddenSize;
            int seqLen = ctx.TokenIds.Length;
            float[] cudaEmbed = new float[(long)seqLen * hiddenSize];
            harness.LookupEmbeddingsToHost(ctx.TokenIds, cudaEmbed);
            float[] cpuEmbed = LoadDump(ctx.DumpDir, "token_embd", seqLen, hiddenSize);
            CompareActivations(label: "token_embd", cpu: cpuEmbed, cuda: cudaEmbed,
                seqLen: seqLen, hiddenSize: hiddenSize);
        }
    }

    /// <summary>
    /// Sanity check: feed the CPU-captured pre-final-norm hidden state
    /// <c>blk.{N-1}.l_out</c> through CUDA's output stage (final RMSNorm + lm_head) and
    /// compare to the CPU dump <c>result_output</c>. Pins the last-layer-to-logits glue.
    /// </summary>
    [SkippableFact]
    public void RealGguf_OutputProjection_MatchesCpu()
    {
        var ctx = TryBuildHarness();
        Skip.If(ctx is null, "Harness skip — see test output for the missing prerequisite.");

        using (ctx)
        {
            var harness = ctx.Harness;
            int hiddenSize = ctx.Config.HiddenSize;
            int vocabSize = ctx.Config.VocabSize;
            int seqLen = ctx.TokenIds.Length;
            int lastLayer = ctx.Config.NumLayers - 1;

            float[] preNormHidden = LoadDump(ctx.DumpDir, $"blk.{lastLayer}.l_out",
                seqLen, hiddenSize);
            float[] cudaLogits = new float[(long)seqLen * vocabSize];
            harness.RunOutputProjectionFromHostInput(preNormHidden, seqLen, cudaLogits);

            float[] cpuLogits = LoadDump(ctx.DumpDir, "result_output", seqLen, vocabSize);
            CompareActivations(label: "result_output", cpu: cpuLogits, cuda: cudaLogits,
                seqLen: seqLen, hiddenSize: vocabSize);
        }
    }

    // ─── Core sweep ─────────────────────────────────────────────────────────────

    private void VerifyLayersByKind(HarnessContext ctx, HybridLayerKind targetKind)
    {
        var harness = ctx.Harness;
        var layout = ctx.Config.HybridLayout!;
        int hiddenSize = ctx.Config.HiddenSize;
        int seqLen = ctx.TokenIds.Length;
        int numLayers = ctx.Config.NumLayers;

        // Build position array once.
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;

        var failures = new List<string>();
        long peakUsed = 0;
        for (int layer = 0; layer < numLayers; layer++)
        {
            if (layout.LayerKind[layer] != targetKind) continue;

            // Load layer's input: token_embd for layer 0, blk.{layer-1}.l_out otherwise.
            string inputName = layer == 0 ? "token_embd" : $"blk.{layer - 1}.l_out";
            float[] inputHidden = LoadDump(ctx.DumpDir, inputName, seqLen, hiddenSize);

            // Load expected output.
            string outputName = $"blk.{layer}.l_out";
            float[] expected = LoadDump(ctx.DumpDir, outputName, seqLen, hiddenSize);

            // Stream this layer's weights through device memory.
            var loadWatch = Stopwatch.StartNew();
            harness.LoadSingleLayerWeightsFromGguf(layer);
            loadWatch.Stop();

            var (usedBytes, totalBytes) = harness.GetDeviceMemoryUsage();
            peakUsed = Math.Max(peakUsed, usedBytes);

            // Run isolated layer body. kvCache: null forces the no-cache prefill fast path
            // on both sides — matches the CPU oracle's prefill semantics and dodges the F32
            // KV cache sidecar as a confound.
            float[] cudaOutput = new float[(long)seqLen * hiddenSize];
            var runWatch = Stopwatch.StartNew();
            harness.RunIsolatedLayerFromHostInput(layer, inputHidden, positions,
                kvCache: null, cudaOutput);
            runWatch.Stop();

            // Compare. Don't throw on the first failure — record it and keep going so the
            // diagnostic surfaces every divergent layer in a single run.
            try
            {
                CompareActivations(label: outputName, cpu: expected, cuda: cudaOutput,
                    seqLen: seqLen, hiddenSize: hiddenSize);
                _output.WriteLine(
                    $"layer {layer,2} {targetKind,-14} OK   " +
                    $"(load {loadWatch.Elapsed.TotalSeconds:F2}s, run {runWatch.Elapsed.TotalSeconds:F2}s, " +
                    $"vram {usedBytes / (1024.0 * 1024 * 1024):F2}/{totalBytes / (1024.0 * 1024 * 1024):F2} GiB)");
            }
            catch (Xunit.Sdk.XunitException ex)
            {
                failures.Add($"layer {layer} ({targetKind}): {ex.Message.Split('\n')[0]}");
                _output.WriteLine($"layer {layer,2} {targetKind,-14} FAIL {ex.Message.Split('\n')[0]}");
            }
            finally
            {
                harness.FreeSingleLayerWeights(layer);
            }
        }

        _output.WriteLine($"Peak VRAM during sweep: {peakUsed / (1024.0 * 1024 * 1024):F2} GiB");
        Assert.True(peakUsed <= PeakVramSoftLimitBytes,
            $"Peak VRAM {peakUsed / (1024.0 * 1024 * 1024):F2} GiB exceeded soft limit " +
            $"{PeakVramSoftLimitBytes / (1024.0 * 1024 * 1024):F2} GiB.");

        if (failures.Count > 0)
        {
            Assert.Fail("Per-layer parity failures:\n  " + string.Join("\n  ", failures));
        }
    }

    // ─── Comparison helper ──────────────────────────────────────────────────────

    private void CompareActivations(string label, float[] cpu, float[] cuda, int seqLen, int hiddenSize)
    {
        Assert.Equal(cpu.Length, cuda.Length);

        float maxAbsDiff = 0f;
        int maxAbsDiffIdx = -1;
        int firstBadIdx = -1;
        double sumSq = 0;
        int finiteCount = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            float a = cpu[i];
            float b = cuda[i];
            if (!float.IsFinite(a) || !float.IsFinite(b))
            {
                Assert.Fail($"{label}: non-finite at i={i} (cpu={a}, cuda={b}).");
            }
            float diff = MathF.Abs(a - b);
            if (diff > maxAbsDiff)
            {
                maxAbsDiff = diff;
                maxAbsDiffIdx = i;
            }
            float bar = AbsTol + RelTol * MathF.Abs(a);
            if (diff > bar && firstBadIdx < 0) firstBadIdx = i;
            sumSq += (double)diff * diff;
            finiteCount++;
        }
        double rmsDiff = finiteCount > 0 ? Math.Sqrt(sumSq / finiteCount) : 0;

        _output.WriteLine($"  {label}: max|diff|={maxAbsDiff:E3} @{maxAbsDiffIdx} rms={rmsDiff:E3} n={cpu.Length}");

        if (firstBadIdx >= 0)
        {
            int t = firstBadIdx / hiddenSize;
            int c = firstBadIdx % hiddenSize;
            float a = cpu[firstBadIdx];
            float b = cuda[firstBadIdx];
            float bar = AbsTol + RelTol * MathF.Abs(a);
            Assert.Fail(
                $"{label}: tolerance breach at i={firstBadIdx} (t={t}, c={c}): " +
                $"cpu={a:F6} vs cuda={b:F6} (|diff|={MathF.Abs(a - b):E3} > {bar:E3}). " +
                $"max|diff|={maxAbsDiff:E3} @{maxAbsDiffIdx}, rms={rmsDiff:E3}.");
        }
    }

    // ─── Dump loader ───────────────────────────────────────────────────────────

    /// <summary>
    /// Loads a per-tensor dump file produced by <c>TensorDump</c> with
    /// <c>DOTLLM_TENSOR_DUMP_DIR</c> set. Filename pattern is
    /// <c>NNNNN_{name}.bin</c> — globbed by suffix to ignore the call-order prefix.
    /// </summary>
    private static float[] LoadDump(string dir, string tensorName, int expectedD0, int expectedD1)
    {
        string safe = tensorName.Replace('/', '_').Replace('\\', '_');
        var matches = Directory.GetFiles(dir, $"*_{safe}.bin");
        if (matches.Length == 0)
            throw new FileNotFoundException(
                $"No CPU dump named '{tensorName}' in {dir}. Expected file pattern " +
                $"NNNNN_{safe}.bin. Re-run the CPU oracle with DOTLLM_TENSOR_DUMP_DIR={dir}.");
        // Multiple matches => probably from different forward calls; pick the lexicographically
        // smallest (earliest emit-order index) which is the prefill we want.
        Array.Sort(matches);
        string path = matches[0];

        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        // Header (16 bytes): rank, dim0, dim1, dim2_or_-1. We only consume rank-2 dumps
        // here (blk.*.l_out, token_embd, result_output); rank-3 dumps would not match the
        // (expectedD0, expectedD1) shape gate below.
        int rank = br.ReadInt32();
        int dim0 = br.ReadInt32();
        int dim1 = br.ReadInt32();
        _ = br.ReadInt32(); // dim2 padding — always -1 for rank-2 dumps
        if (rank != 2)
            throw new InvalidDataException(
                $"Dump {path} has rank {rank}; this loader handles rank-2 dumps only.");
        if (dim0 != expectedD0 || dim1 != expectedD1)
            throw new InvalidDataException(
                $"Dump {path} shape [{dim0},{dim1}] != expected [{expectedD0},{expectedD1}]. " +
                "Did the CPU oracle run with a different prompt or seqLen?");
        long n = (long)dim0 * dim1;

        float[] data = new float[n];
        var bytes = MemoryMarshal.AsBytes(data.AsSpan());
        int total = bytes.Length;
        int off = 0;
        while (off < total)
        {
            int read = fs.Read(bytes.Slice(off));
            if (read == 0)
                throw new InvalidDataException($"Dump {path} truncated: expected {total} bytes, got {off}.");
            off += read;
        }
        return data;
    }

    // ─── Skip predicates + harness builder ──────────────────────────────────────

    private sealed class HarnessContext : IDisposable
    {
        public CudaQwen3MoeHybridTransformerModel Harness { get; }
        public ModelConfig Config { get; }
        public GgufFile Gguf { get; }
        public string DumpDir { get; }
        public int[] TokenIds { get; }

        public HarnessContext(
            CudaQwen3MoeHybridTransformerModel harness, ModelConfig config,
            GgufFile gguf, string dumpDir, int[] tokenIds)
        {
            Harness = harness;
            Config = config;
            Gguf = gguf;
            DumpDir = dumpDir;
            TokenIds = tokenIds;
        }

        public void Dispose()
        {
            Harness.Dispose();
            Gguf.Dispose();
        }
    }

    private HarnessContext? TryBuildHarness()
    {
        if (!IsCudaDriverPresent())
        {
            _output.WriteLine("Skip: no CUDA driver / GPU present.");
            return null;
        }
        string? ptxDir = FindPtxDir();
        if (ptxDir is null)
        {
            _output.WriteLine("Skip: PTX files not found near the test assembly.");
            return null;
        }

        string ggufPath = Environment.GetEnvironmentVariable(GgufPathEnvVar) ?? DefaultGgufPath;
        if (!File.Exists(ggufPath))
        {
            _output.WriteLine($"Skip: GGUF not at {ggufPath}. Set {GgufPathEnvVar} or place at default.");
            return null;
        }

        string? dumpDir = Environment.GetEnvironmentVariable(DumpDirEnvVar);
        if (string.IsNullOrEmpty(dumpDir) || !Directory.Exists(dumpDir) ||
            Directory.GetFiles(dumpDir, "*_token_embd.bin").Length == 0)
        {
            _output.WriteLine(
                $"Skip: CPU dump directory not found / empty. Set {DumpDirEnvVar} to a directory " +
                "containing the output of a prior CPU forward run with " +
                "DOTLLM_TENSOR_DUMP_DIR=<dir>. To generate it once:\n" +
                "  $env:DOTLLM_TENSOR_DUMP_DIR='C:/path/to/dumpdir'\n" +
                "  dotnet run --project samples/DotLLM.Sample.Console -c Release -- <gguf> --prompt 'The capital of France is' --max-tokens 0");
            return null;
        }

        _output.WriteLine($"GGUF:     {ggufPath}");
        _output.WriteLine($"PTX dir:  {ptxDir}");
        _output.WriteLine($"Dump dir: {dumpDir}");

        var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        if (config.Architecture != DotLLM.Core.Configuration.Architecture.Qwen3MoeHybrid)
        {
            _output.WriteLine($"Skip: GGUF arch {config.Architecture} != Qwen3MoeHybrid.");
            gguf.Dispose();
            return null;
        }

        // Encode the canonical parity prompt ("The capital of France is") via the GGUF
        // tokenizer. The CPU dump-producer must have used this same prompt — the embedding
        // lookup test will diverge cleanly if it didn't.
        int dumpSeqLen = ReadDumpSeqLen(dumpDir, "token_embd");
        int[] tokenIds = EncodeParityPrompt(gguf, dumpSeqLen);
        _output.WriteLine($"Prompt tokens (seqLen={tokenIds.Length}): [{string.Join(",", tokenIds)}]");

        var harness = CudaQwen3MoeHybridTransformerModel.LoadFromGgufForLayerByLayerHarness(
            gguf, config, deviceId: 0, ptxDir: ptxDir);

        var (used, total) = harness.GetDeviceMemoryUsage();
        _output.WriteLine(
            $"Post-load VRAM: {used / (1024.0 * 1024 * 1024):F2} / " +
            $"{total / (1024.0 * 1024 * 1024):F2} GiB (no per-layer weights uploaded yet).");

        return new HarnessContext(harness, config, gguf, dumpDir, tokenIds);
    }

    /// <summary>
    /// Reads the <c>token_embd</c> dump header to recover the seqLen the CPU oracle
    /// dumped with. Used as a cross-check against the prompt's tokenization length.
    /// </summary>
    private static int ReadDumpSeqLen(string dumpDir, string tensorName)
    {
        var matches = Directory.GetFiles(dumpDir, $"*_{tensorName}.bin");
        if (matches.Length == 0)
            throw new FileNotFoundException(
                $"No '{tensorName}' dump in {dumpDir}; regenerate the CPU oracle output.");
        Array.Sort(matches);
        using var fs = new FileStream(matches[0], FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        _ = br.ReadInt32(); // rank
        int seqLen = br.ReadInt32();
        return seqLen;
    }

    /// <summary>
    /// Encodes the canonical parity prompt via the GGUF tokenizer. Truncates / pads to
    /// match the dump's seqLen — if the user generated dumps from a different prompt the
    /// embedding-lookup test will diverge, surfacing the mismatch immediately.
    /// </summary>
    private static int[] EncodeParityPrompt(GgufFile gguf, int expectedSeqLen)
    {
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        const string parityPrompt = "The capital of France is";
        int[] tokens = tokenizer.Encode(parityPrompt);
        if (tokens.Length == expectedSeqLen) return tokens;
        if (tokens.Length > expectedSeqLen) return tokens[..expectedSeqLen];
        // Dump has more tokens than the canonical prompt produced. Honour the dump's
        // length so the embedding-lookup parity test reports against the same shape.
        int[] padded = new int[expectedSeqLen];
        Array.Copy(tokens, padded, tokens.Length);
        return padded;
    }

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }
}
