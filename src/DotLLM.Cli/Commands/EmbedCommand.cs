using System.ComponentModel;
using System.Globalization;
using System.Text;
using DotLLM.Cli.Helpers;
using DotLLM.Core.Models;
using DotLLM.Server;
using DotLLM.Engine.Embeddings;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// <c>dotllm embed &lt;model&gt; [text...]</c> — prints one embedding per input as a JSON array on its own
/// line (JSON Lines). With no text arguments, reads one input per line from stdin. CPU, or Vulkan for decoder embedders (#1406):
/// encoders (BERT / nomic-bert, CPU only) and decoder checkpoints that implement <see cref="IEmbeddingModel"/>.
/// </summary>
internal sealed class EmbedCommand : Command<EmbedCommand.Settings>
{
    public sealed class Settings : CommandSettings
    {
        [CommandArgument(0, "<model>")]
        [Description("Path to a GGUF file or HuggingFace repo ID.")]
        public string Model { get; set; } = string.Empty;

        [CommandArgument(1, "[text]")]
        [Description("Text(s) to embed. When omitted, one input per line is read from stdin.")]
        public string[] Text { get; set; } = [];

        [CommandOption("--quant|-q")]
        [Description("Quantization to select when <model> is a HuggingFace repo.")]
        public string? Quant { get; set; }

        [CommandOption("--pooling")]
        [Description("Pooling override: last, mean or cls. Default: the model's declared pooling.")]
        public string? Pooling { get; set; }

        [CommandOption("--add-eos")]
        [Description("Force (true) or suppress (false) appending the EOS token. Default: appended for last-token pooling on decoder embedders (Qwen3-Embedding).")]
        public bool? AddEos { get; set; }

        [CommandOption("--dimensions")]
        [Description("Truncate to the leading N components (Matryoshka), then renormalise.")]
        public int? Dimensions { get; set; }

        [CommandOption("--no-normalize")]
        [Description("Return un-normalised vectors.")]
        [DefaultValue(false)]
        public bool NoNormalize { get; set; }

        [CommandOption("--device|-d")]
        [Description("Compute device: 'cpu' (default) or 'vulkan' (decoder embedders such as Qwen3-Embedding and BERT encoders with F16 weights, #741; nomic-bert / gemma-embedding are CPU-only). An explicit GPU that cannot be honoured is an error, never a silent CPU run.")]
        [DefaultValue("cpu")]
        public string Device { get; set; } = "cpu";

        [CommandOption("--threads")]
        [Description("CPU threads (0 = auto).")]
        [DefaultValue(0)]
        public int Threads { get; set; }
    }

    public override unsafe int Execute(CommandContext context, Settings s)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        string? path = GgufFileResolver.Resolve(s.Model, s.Quant);
        if (path is null) { Console.Error.WriteLine($"Model not found: {s.Model}"); return 1; }

        var inputs = s.Text.Length > 0 ? s.Text.ToList() : ReadStdin();
        if (inputs.Count == 0) { Console.Error.WriteLine("No input text (pass text arguments or pipe lines on stdin)."); return 1; }

        PoolingType? requested = null;
        if (!string.IsNullOrEmpty(s.Pooling))
        {
            requested = s.Pooling.ToLowerInvariant() switch
            {
                "last" => PoolingType.Last, "mean" => PoolingType.Mean, "cls" => PoolingType.Cls, _ => null,
            };
            if (requested is null) { Console.Error.WriteLine("--pooling must be last, mean or cls."); return 1; }
        }

        if (DeviceCli.Validate(s.Device) is { } deviceError) { Console.Error.WriteLine(deviceError); return 1; }
        var deviceSpec = DeviceSpec.Parse(s.Device);
        var threading = new DotLLM.Core.Configuration.ThreadingConfig(s.Threads, s.Threads);
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        IModel model;
        try
        {
            bool isEncoder = config.Architecture is DotLLM.Core.Configuration.Architecture.Bert or DotLLM.Core.Configuration.Architecture.NomicBert;
            if (isEncoder && deviceSpec.IsGpu && !(deviceSpec.Kind == DeviceKind.Vulkan && config.Architecture is DotLLM.Core.Configuration.Architecture.Bert or DotLLM.Core.Configuration.Architecture.NomicBert))
            {
                // #741: only architecture 'bert' (F16 weights) has a GPU encoder, and only on Vulkan. Loud refusal, never a CPU run.
                throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(
                    s.Device, Path.GetFileName(path), 0,
                    $"{config.Architecture} encoder embedding models have no GPU backend on '{s.Device}' (#741 / #1471; only 'bert' and 'nomic-bert' on vulkan); use --device cpu. Decoder embedders such as Qwen3-Embedding run on vulkan"));
            }
            if (isEncoder && deviceSpec.IsGpu)
            {
                try { model = DotLLM.Vulkan.VulkanBertEncoderModel.LoadFromGguf(DotLLM.Vulkan.VulkanModelLoader.SharedDevice, gguf, config, DotLLM.Vulkan.VulkanModelLoader.ResolveSpvDir()); }
                catch (NotSupportedException ex)
                {
                    throw new DeviceUnavailableException(DeviceSelector.ExplicitFailureMessage(s.Device, Path.GetFileName(path), 0, ex.Message));
                }
            }
            else
            if (isEncoder || deviceSpec.Kind == DeviceKind.Cpu)
            {
                model = ModelLoader.CreateCpuModelFromGguf(gguf, config, threading);
            }
            else
            {
                var (loaded, outcome) = DeviceCli.Load(gguf, config, s.Device, null, threading, path);
                model = loaded.Model;
                DeviceCli.Report(outcome, json: true);   // stderr: stdout stays pure JSON Lines
            }
        }
        catch (DeviceUnavailableException ex)
        {
            DeviceCli.PrintError(ex.Message, json: true);
            gguf.Dispose();
            return 1;
        }
        using (gguf)
        using (model)
        {
            if (model is not IEmbeddingModel emb)
            {
                Console.Error.WriteLine($"{config.Architecture} does not support embedding extraction.");
                return 1;
            }
            var tokenizer = GgufTokenizerFactory.Load(gguf.Metadata);
            var pooling = EmbeddingPooler.Resolve(requested, emb.DeclaredPoolingType);
            if (pooling is PoolingType.None or PoolingType.Rank)
            {
                Console.Error.WriteLine($"Model declares pooling '{pooling}'; pass --pooling last|mean|cls.");
                return 1;
            }
            int hidden = config.HiddenSize;
            int outDims = s.Dimensions ?? hidden;
            if (outDims < 1 || outDims > hidden) { Console.Error.WriteLine($"--dimensions must be 1..{hidden}."); return 1; }

            foreach (string text in inputs)
            {
                int[] tokens = EmbeddingPooler.AppendEos(tokenizer.Encode(text).ToArray(), emb.DeclaredPoolingType, emb.AppendsEosForLastPooling, tokenizer.EosTokenId, s.AddEos);
                if (tokens.Length == 0 || tokens.Length > config.MaxSequenceLength)
                {
                    Console.Error.WriteLine($"Input has {tokens.Length} tokens (valid: 1..{config.MaxSequenceLength}).");
                    return 1;
                }
                var vec = new float[hidden];
                // Only the rows pooling reads come back (one row for last/cls on a GPU), #1406.
                emb.ForwardHiddenForPooling([tokens], pooling,
                    (_, rows) => EmbeddingPooler.Pool(rows, rows.Length / hidden, hidden, pooling, vec), deviceId: 0);
                var outv = outDims < hidden ? vec.AsSpan(0, outDims).ToArray() : vec;
                if (!s.NoNormalize) EmbeddingPooler.L2Normalize(outv);

                var sb = new StringBuilder("[");
                for (int i = 0; i < outv.Length; i++)
                {
                    if (i > 0) sb.Append(',');
                    sb.Append(outv[i].ToString("R", CultureInfo.InvariantCulture));
                }
                Console.Out.WriteLine(sb.Append(']'));
            }
        }
        return 0;
    }

    private static List<string> ReadStdin()
    {
        var list = new List<string>();
        if (!Console.IsInputRedirected) return list;
        // Console.In decodes with the OEM code page on Windows, which mangles non-ASCII text; read raw UTF-8.
        using var reader = new StreamReader(Console.OpenStandardInput(), new UTF8Encoding(false));
        string? line;
        while ((line = reader.ReadLine()) is not null)
            if (line.Length > 0) list.Add(line);
        return list;
    }
}
