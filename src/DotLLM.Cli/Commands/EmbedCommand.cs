using System.ComponentModel;
using System.Globalization;
using System.Text;
using DotLLM.Core.Models;
using DotLLM.Engine.Embeddings;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using Spectre.Console.Cli;

namespace DotLLM.Cli.Commands;

/// <summary>
/// <c>dotllm embed &lt;model&gt; [text...]</c> — prints one embedding per input as a JSON array on its own
/// line (JSON Lines). With no text arguments, reads one input per line from stdin. CPU only (#740):
/// encoders (BERT / nomic-bert) and decoder checkpoints that implement <see cref="IEmbeddingModel"/>.
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

        [CommandOption("--dimensions")]
        [Description("Truncate to the leading N components (Matryoshka), then renormalise.")]
        public int? Dimensions { get; set; }

        [CommandOption("--no-normalize")]
        [Description("Return un-normalised vectors.")]
        [DefaultValue(false)]
        public bool NoNormalize { get; set; }

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

        var (model, gguf, config) = ModelLoader.LoadFromGguf(path, new DotLLM.Core.Configuration.ThreadingConfig(s.Threads, s.Threads));
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
                int[] tokens = tokenizer.Encode(text).ToArray();
                if (tokens.Length == 0 || tokens.Length > config.MaxSequenceLength)
                {
                    Console.Error.WriteLine($"Input has {tokens.Length} tokens (valid: 1..{config.MaxSequenceLength}).");
                    return 1;
                }
                var positions = new int[tokens.Length];
                for (int i = 0; i < positions.Length; i++) positions[i] = i;
                var vec = new float[hidden];
                using (var h = emb.ForwardHidden(tokens, positions, deviceId: 0))
                    EmbeddingPooler.Pool(new ReadOnlySpan<float>((void*)h.DataPointer, tokens.Length * hidden), tokens.Length, hidden, pooling, vec);
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
