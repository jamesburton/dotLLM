using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

public static partial class GgufModelConfigExtractor
{
    /// <summary>
    /// Extracts the <see cref="Qwen4ExpConfig"/> from <c>qwen4exp</c> GGUF metadata.
    /// </summary>
    /// <remarks>
    /// Keys are the ones llama.cpp reads in <c>src/models/qwen4exp.cpp</c> (<c>load_arch_hparams</c>), with the same
    /// required/optional split and the same sanity checks, verified against the real
    /// <c>unsloth/Qwen3.8-Flash-Next-GGUF</c> header:
    /// <list type="bullet">
    ///   <item><c>{arch}.hyper_connection.count</c> (required, must be &gt; 1), <c>.low_rank</c> (required, &gt; 0)</item>
    ///   <item><c>{arch}.attention.indexer.head_count</c> / <c>.key_length</c> / <c>.top_k</c> (required, each &gt; 0)</item>
    ///   <item><c>{arch}.attention.compress_ratios</c> (optional per-block i32 array; all non-zero entries equal, &gt; 1, dividing top_k)</item>
    ///   <item><c>{arch}.rope.dimension_sections</c> (required, 4 entries)</item>
    ///   <item>PLE (only if <c>{arch}.ple.layers</c> is non-empty): <c>ple.ngram_size</c>, <c>ple.heads_per_ngram</c>,
    ///     <c>ple.conv_kernel</c>, <c>ple.eos_token_id</c>, <c>embedding_length_per_layer_input</c> (all required),
    ///     <c>ple.image_token_id</c> (optional), and the exact-uint64 arrays <c>ple.layer_multipliers</c>,
    ///     <c>ple.head_offsets</c>, <c>ple.head_vocab_sizes</c> (at least ngram_size / heads entries)</item>
    /// </list>
    /// </remarks>
    private static Qwen4ExpConfig ExtractQwen4ExpConfig(
        GgufMetadata metadata, string arch, int numLayers, int numTrunkLayers, HybridLayerLayout trunkLayout)
    {
        int hcCount = RequiredPositive(metadata, $"{arch}.hyper_connection.count");
        if (hcCount <= 1)
            throw new InvalidDataException(
                $"'{arch}.hyper_connection.count' must be greater than one (a single stream has nothing to mix), got {hcCount}.");
        int hcRank = RequiredPositive(metadata, $"{arch}.hyper_connection.low_rank");

        int idxHeads = RequiredPositive(metadata, $"{arch}.attention.indexer.head_count");
        int idxKeyLen = RequiredPositive(metadata, $"{arch}.attention.indexer.key_length");
        int idxTopK = RequiredPositive(metadata, $"{arch}.attention.indexer.top_k");

        // Per-block pooling ratios. llama.cpp's get_key_or_arr(.., n_layer_all, required=false): absent is allowed.
        int[] ratios = ReadIntegerArray(metadata, $"{arch}.attention.compress_ratios");
        if (ratios.Length != 0 && ratios.Length != numLayers)
            throw new InvalidDataException(
                $"'{arch}.attention.compress_ratios' has {ratios.Length} entries but the model has {numLayers} blocks.");

        int blockSize = 0;
        for (int il = 0; il < ratios.Length; il++)
        {
            int r = ratios[il];
            if (r < 0)
                throw new InvalidDataException($"'{arch}.attention.compress_ratios'[{il}] is negative ({r}).");
            if (r == 0) continue;
            if (blockSize != 0 && r != blockSize)
                throw new InvalidDataException(
                    $"QSA layers must share one compress ratio, got {blockSize} and {r} (block {il}).");
            blockSize = r;
        }
        if (blockSize == 1 || (blockSize > 0 && idxTopK % blockSize != 0))
            throw new InvalidDataException(
                $"QSA needs a compress ratio above 1 that divides the token budget, got ratio {blockSize} and top_k {idxTopK}.");

        // The ratio table and the GDN/attention interval describe the same layers; a disagreement means one of them is wrong.
        if (ratios.Length != 0)
        {
            for (int il = 0; il < numTrunkLayers; il++)
            {
                bool isAttention = trunkLayout.LayerKind[il] == HybridLayerKind.Attention;
                if ((ratios[il] != 0) != isAttention)
                    throw new InvalidDataException(
                        $"Block {il} is a {(isAttention ? "full-attention (QSA)" : "linear-attention")} layer by " +
                        $"'{arch}.full_attention_interval' but '{arch}.attention.compress_ratios'[{il}] is {ratios[il]}.");
            }
        }

        int[] sections = ReadIntegerArray(metadata, $"{arch}.rope.dimension_sections");
        if (sections.Length != 4)
            throw new InvalidDataException(
                $"'{arch}.rope.dimension_sections' is required and must have 4 entries, got {sections.Length}.");

        return new Qwen4ExpConfig
        {
            HyperConnectionCount = hcCount,
            HyperConnectionLowRank = hcRank,
            IndexerHeadCount = idxHeads,
            IndexerKeyLength = idxKeyLen,
            IndexerTopK = idxTopK,
            CompressRatios = ratios,
            IndexerBlockSize = blockSize,
            RopeSections = sections,
            Ple = ExtractQwen4ExpPle(metadata, arch, numLayers, numTrunkLayers, trunkLayout),
        };
    }

    private static Qwen4ExpPleConfig? ExtractQwen4ExpPle(
        GgufMetadata metadata, string arch, int numLayers, int numTrunkLayers, HybridLayerLayout trunkLayout)
    {
        // "if the key group is absent every field stays zero" (llama.cpp) — no PLE module.
        int[] layers = ReadIntegerArray(metadata, $"{arch}.ple.layers");
        if (layers.Length == 0)
            return null;

        // hparams holds one set of hash constants, so llama.cpp cannot represent several PLE modules.
        if (layers.Length != 1)
            throw new NotSupportedException(
                $"'{arch}.ple.layers' lists {layers.Length} layers, but only one PLE layer is supported (llama.cpp shares one set of hash constants).");
        int layer = layers[0];
        if (layer < 0 || layer >= numLayers)
            throw new InvalidDataException($"PLE layer {layer} is out of range (the model has {numLayers} blocks).");
        // The PLE conv history lives in the recurrent state, which only linear-attention layers have.
        if (layer >= numTrunkLayers || trunkLayout.LayerKind[layer] == HybridLayerKind.Attention)
            throw new InvalidDataException($"PLE layer {layer} is not a linear-attention (Gated-DeltaNet) layer.");

        int ngram = RequiredInt(metadata, $"{arch}.ple.ngram_size");
        int headsPerNgram = RequiredInt(metadata, $"{arch}.ple.heads_per_ngram");
        int convKernel = RequiredPositive(metadata, $"{arch}.ple.conv_kernel");
        int eos = RequiredInt(metadata, $"{arch}.ple.eos_token_id");
        int rowDim = RequiredPositive(metadata, $"{arch}.embedding_length_per_layer_input");
        int? image = metadata.ContainsKey($"{arch}.ple.image_token_id")
            ? RequiredInt(metadata, $"{arch}.ple.image_token_id")
            : null;

        if (ngram < 2)
            throw new InvalidDataException($"PLE n-gram size {ngram} is out of range (must be at least 2).");
        long numHeads = (long)(ngram - 1) * headsPerNgram;
        if (numHeads <= 0 || numHeads > 1024)
            throw new InvalidDataException($"PLE head count {numHeads} is out of range.");

        ulong[] multipliers = RequireUInt64Array(metadata, $"{arch}.ple.layer_multipliers", ngram);
        ulong[] offsets = RequireUInt64Array(metadata, $"{arch}.ple.head_offsets", (int)numHeads);
        ulong[] vocabs = RequireUInt64Array(metadata, $"{arch}.ple.head_vocab_sizes", (int)numHeads);
        for (int h = 0; h < numHeads; h++)
        {
            if (vocabs[h] == 0)
                throw new InvalidDataException($"PLE head {h} has a zero vocab size.");
            if (offsets[h] > long.MaxValue || vocabs[h] > long.MaxValue || offsets[h] + vocabs[h] > long.MaxValue)
                throw new InvalidDataException($"PLE head {h} range does not fit a signed 64-bit row index.");
        }

        return new Qwen4ExpPleConfig
        {
            Layers = layers,
            NgramSize = ngram,
            HeadsPerNgram = headsPerNgram,
            ConvKernel = convKernel,
            EosTokenId = eos,
            ImageTokenId = image,
            RowDim = rowDim,
            LayerMultipliers = multipliers,
            HeadOffsets = offsets,
            HeadVocabSizes = vocabs,
        };
    }

    private static int RequiredInt(GgufMetadata metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var v))
            throw new InvalidDataException($"GGUF architecture 'qwen4exp' requires metadata key '{key}'; the file does not carry it.");
        long value = v.Value switch
        {
            uint u => u,
            int i => i,
            ushort us => us,
            ulong ul when ul <= int.MaxValue => (long)ul,
            long l => l,
            _ => throw new InvalidDataException($"GGUF metadata key '{key}' has type {v.Type}, expected an integer."),
        };
        if (value < 0 || value > int.MaxValue)
            throw new InvalidDataException($"GGUF metadata key '{key}' value {value} is out of range.");
        return (int)value;
    }

    private static int RequiredPositive(GgufMetadata metadata, string key)
    {
        int value = RequiredInt(metadata, key);
        if (value == 0)
            throw new InvalidDataException($"'{key}' must be greater than zero, got 0.");
        return value;
    }

    /// <summary>Reads a signed/unsigned 32-bit integer array (any writer's choice of element type); empty when absent.</summary>
    private static int[] ReadIntegerArray(GgufMetadata metadata, string key)
    {
        if (!metadata.TryGetValue(key, out var entry))
            return [];
        if (entry.Type != GgufValueType.Array)
            throw new InvalidDataException($"GGUF metadata key '{key}' has type {entry.Type}, expected an integer array.");
        // NB: exact runtime-type tests, not `is T[]` patterns — the CLR lets an int[] pass as uint[] (and long[] as ulong[]),
        // which would reinterpret negative values as huge unsigned ones without any error.
        Type type = entry.Value.GetType();
        if (type == typeof(int[]))
            return (int[])entry.Value;
        if (type == typeof(uint[]))
        {
            var src = (uint[])entry.Value;
            var r = new int[src.Length];
            for (int i = 0; i < r.Length; i++) r[i] = checked((int)src[i]);
            return r;
        }
        if (type == typeof(long[]))
        {
            var src = (long[])entry.Value;
            var r = new int[src.Length];
            for (int i = 0; i < r.Length; i++) r[i] = checked((int)src[i]);
            return r;
        }
        if (type == typeof(ushort[]))
        {
            var src = (ushort[])entry.Value;
            var r = new int[src.Length];
            for (int i = 0; i < r.Length; i++) r[i] = src[i];
            return r;
        }
        throw new InvalidDataException($"GGUF metadata key '{key}' is not an integer array ({type.Name}).");
    }

    /// <summary>
    /// Reads an unsigned 64-bit array bit-exactly. An array written with a signed element type is accepted only if no entry is
    /// negative (a reinterpretation would otherwise silently change the hash constants).
    /// </summary>
    private static ulong[] RequireUInt64Array(GgufMetadata metadata, string key, int minLength)
    {
        if (!metadata.TryGetValue(key, out var entry))
            throw new InvalidDataException($"GGUF architecture 'qwen4exp' requires metadata key '{key}'; the file does not carry it.");

        // Exact runtime-type tests (see ReadIntegerArray): `is ulong[]` would also accept a long[] and reinterpret negatives.
        Type type = entry.Value.GetType();
        ulong[] values;
        if (type == typeof(ulong[]))
        {
            values = (ulong[])entry.Value;
        }
        else if (type == typeof(long[]))
        {
            var src = (long[])entry.Value;
            values = new ulong[src.Length];
            for (int i = 0; i < src.Length; i++)
            {
                if (src[i] < 0)
                    throw new InvalidDataException($"'{key}'[{i}] is negative ({src[i]}); the n-gram hash constants are unsigned.");
                values[i] = (ulong)src[i];
            }
        }
        else if (type == typeof(uint[]))
        {
            var src = (uint[])entry.Value;
            values = new ulong[src.Length];
            for (int i = 0; i < src.Length; i++) values[i] = src[i];
        }
        else if (type == typeof(int[]))
        {
            var src = (int[])entry.Value;
            values = new ulong[src.Length];
            for (int i = 0; i < src.Length; i++)
            {
                if (src[i] < 0)
                    throw new InvalidDataException($"'{key}'[{i}] is negative ({src[i]}); the n-gram hash constants are unsigned.");
                values[i] = (ulong)src[i];
            }
        }
        else
        {
            throw new InvalidDataException($"GGUF metadata key '{key}' is not an integer array ({type.Name}).");
        }

        // llama.cpp: get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops.
        if (values.Length < minLength)
            throw new InvalidDataException($"'{key}' has {values.Length} entries, but at least {minLength} are required.");
        return values;
    }
}
