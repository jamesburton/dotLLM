using System.IO.Compression;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Core.Vision;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Vision;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>Model-free tests of the image input plumbing (issue #2035): preprocessing, decoding, placeholders, chat content parts, engine hand-off.</summary>
public sealed class Inkling2VisionPlumbingTests
{
    private readonly ITestOutputHelper _output;
    public Inkling2VisionPlumbingTests(ITestOutputHelper output) => _output = output;

    // ───────────── preprocessing ─────────────

    [Theory]
    [InlineData(190, 110, 200, 120, 15)]    // padded up to a multiple of 40
    [InlineData(640, 640, 640, 640, 256)]
    [InlineData(80, 80, 80, 80, 4)]
    [InlineData(1, 1, 40, 40, 1)]
    [InlineData(41, 40, 80, 40, 2)]
    public void PlanSize_AlignsToPatchTimesMerge_AndCountsMergedTokens(int w, int h, int pw, int ph, int tokens)
    {
        var (W, H, _, _) = Inkling2ImagePreprocessor.PlanSize(w, h);
        Assert.Equal((pw, ph), (W, H));
        Assert.Equal(tokens, Inkling2ImagePreprocessor.CountTokens(w, h));
    }

    [Fact]
    public void PlanSize_BoundsOversizedImagesToTheTokenLimit()
    {
        foreach (var (w, h) in new[] { (4000, 4000), (8000, 600), (600, 9000) })
        {
            int tokens = Inkling2ImagePreprocessor.CountTokens(w, h);
            Assert.InRange(tokens, 1, Inkling2ImagePreprocessor.MaxTokens);
        }
        // below the limit nothing is resized: 2500 x 2500 is 63 x 63 = 3969 tokens
        Assert.Equal(63 * 63, Inkling2ImagePreprocessor.CountTokens(2500, 2500));
    }

    [Fact]
    public void Normalize_UsesHalfMeanStd_AndPadsAtMinusOneRaw()
    {
        // 2x1 image padded to 40x40: pixel (0,0) = 255 -> +1, pixel (1,0) = 0 -> -1, the padding is (-1 - 127.5) / 127.5 = -1.0078 (the reference's quirk).
        var img = new DecodedImage(2, 1, [255, 255, 255, 0, 0, 0]);
        var buf = new float[40 * 40 * 3];
        var (w, h) = Inkling2ImagePreprocessor.Normalize(img, buf);
        Assert.Equal((40, 40), (w, h));
        Assert.Equal(1f, buf[0]); Assert.Equal(1f, buf[2]);
        Assert.Equal(-1f, buf[3]); Assert.Equal(-1f, buf[5]);
        float pad = (-1f - 127.5f) / 127.5f;
        Assert.Equal(pad, buf[6]);                       // (2, 0)
        Assert.Equal(pad, buf[40 * 3 + 0]);              // (0, 1)
        Assert.NotEqual(-1f, pad);
        // channel order is kept (R, G, B interleaved)
        var rgb = new DecodedImage(1, 1, [255, 0, 51]);
        Inkling2ImagePreprocessor.Normalize(rgb, buf);
        Assert.Equal((255f - 127.5f) / 127.5f, buf[0]); Assert.Equal(-1f, buf[1]); Assert.Equal((51f - 127.5f) / 127.5f, buf[2]);
    }

    // ───────────── decoding ─────────────

    private static byte[] MakePng(int w, int h, byte[] rgb, int ctype = 2, Func<int, int>? filterFor = null)
    {
        int channels = ctype == 6 ? 4 : 3;
        var raw = new List<byte>();
        var prev = new byte[w * channels];
        for (int y = 0; y < h; y++)
        {
            var line = new byte[w * channels];
            for (int x = 0; x < w; x++)
            {
                for (int c = 0; c < 3; c++) line[x * channels + c] = rgb[(y * w + x) * 3 + c];
                if (channels == 4) line[x * 4 + 3] = 200;   // alpha is dropped by the decoder
            }
            int f = filterFor?.Invoke(y) ?? 0;
            raw.Add((byte)f);
            for (int i = 0; i < line.Length; i++)
            {
                int a = i >= channels ? line[i - channels] : 0, b = prev[i], c = i >= channels ? prev[i - channels] : 0;
                int pred = f switch
                {
                    0 => 0, 1 => a, 2 => b, 3 => (a + b) >> 1,
                    _ => PaethRef(a, b, c),
                };
                raw.Add((byte)(line[i] - pred));
            }
            prev = line;
        }
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, CompressionLevel.Fastest, leaveOpen: true)) z.Write(raw.ToArray());
        var idat = ms.ToArray();
        var png = new List<byte> { 0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A };
        void Chunk(string type, byte[] body)
        {
            png.AddRange(BitConverter.GetBytes(System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(body.Length)));
            png.AddRange(System.Text.Encoding.ASCII.GetBytes(type)); png.AddRange(body); png.AddRange(new byte[4]);   // CRC not checked
        }
        var ihdr = new List<byte>();
        ihdr.AddRange(BitConverter.GetBytes(System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(w)));
        ihdr.AddRange(BitConverter.GetBytes(System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(h)));
        ihdr.AddRange(new byte[] { 8, (byte)ctype, 0, 0, 0 });
        Chunk("IHDR", ihdr.ToArray()); Chunk("IDAT", idat); Chunk("IEND", []);
        return png.ToArray();
    }

    private static int PaethRef(int a, int b, int c)
    {
        int p = a + b - c, pa = Math.Abs(p - a), pb = Math.Abs(p - b), pc = Math.Abs(p - c);
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
    }

    [Theory]
    [InlineData(2)]
    [InlineData(6)]
    public void Png_AllFiveFilters_AndAlphaDropped_DecodeExactly(int ctype)
    {
        const int w = 7, h = 10;
        var rgb = new byte[w * h * 3];
        new Random(5).NextBytes(rgb);
        var png = MakePng(w, h, rgb, ctype, y => y % 5);
        var img = ImageDecoder.Decode(png);
        Assert.Equal((w, h), (img.Width, img.Height));
        Assert.Equal(rgb, img.Rgb);
    }

    [Fact]
    public void Ppm_And_DataUri_Decode_AndJpegIsRefusedByName()
    {
        var ppm = System.Text.Encoding.ASCII.GetBytes("P6\n# comment\n2 1\n255\n").Concat(new byte[] { 1, 2, 3, 4, 5, 6 }).ToArray();
        var img = ImageDecoder.Decode(ppm);
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6 }, img.Rgb);
        var viaUri = ImageDecoder.DecodeDataUri("data:image/x-portable-pixmap;base64," + Convert.ToBase64String(ppm));
        Assert.Equal(img.Rgb, viaUri.Rgb);
        Assert.Equal(img.ContentHash, viaUri.ContentHash);
        var ex = Assert.Throws<NotSupportedException>(() => ImageDecoder.Decode([0xFF, 0xD8, 0xFF, 0xE0, 0, 0, 0, 0, 0, 0]));
        Assert.Contains("JPEG", ex.Message);
        Assert.Throws<NotSupportedException>(() => ImageDecoder.DecodeDataUri("data:image/png,rawbytes"));
    }

    [Fact]
    public void ContentHash_DistinguishesImagesOfEqualShapeAndShapesOfEqualBytes()
    {
        var a = new DecodedImage(2, 1, [1, 2, 3, 4, 5, 6]);
        var b = new DecodedImage(2, 1, [1, 2, 3, 4, 5, 7]);
        var c = new DecodedImage(1, 2, [1, 2, 3, 4, 5, 6]);
        Assert.NotEqual(a.ContentHash, b.ContentHash);
        Assert.NotEqual(a.ContentHash, c.ContentHash);
    }

    // ───────────── placeholders ─────────────

    [Fact]
    public void Expand_WidensEachPlaceholder_AndReportsStarts()
    {
        var (ids, starts) = ImagePlaceholders.Expand([1, 9, 2, 9, 3], 9, [3, 2]);
        Assert.Equal(new[] { 1, 9, 9, 9, 2, 9, 9, 3 }, ids);
        Assert.Equal(new[] { 1, 5 }, starts);
    }

    [Fact]
    public void Expand_RefusesAMismatchedPlaceholderCount()
    {
        Assert.Throws<ArgumentException>(() => ImagePlaceholders.Expand([1, 9, 2], 9, [1, 1]));   // fewer placeholders than images
        Assert.Throws<ArgumentException>(() => ImagePlaceholders.Expand([1, 9, 9], 9, [1]));      // an unmatched placeholder would be fed as a token
    }

    // ───────────── chat content parts ─────────────

    [Fact]
    public void ChatTemplate_RendersImageParts_AsTheReferenceTemplateDoes()
    {
        var tpl = new JinjaChatTemplate(DotLLM.Tests.Unit.Tokenizers.ChatTemplates.ReferenceRenderCases.Fixture("inkling2-chat-template.jinja"), "", "");
        var msg = new ChatMessage
        {
            Role = "user",
            Content = "Describe this image in one sentence.",
            ContentParts =
            [
                new ChatContentPart { Type = "image_url", ImageUrl = "data:image/png;base64,AAAA" },
                new ChatContentPart { Type = "text", Text = "Describe this image in one sentence." },
            ],
        };
        string r = tpl.Apply([msg], new ChatTemplateOptions { AddGenerationPrompt = true });
        // expected: transformers' own render of the same messages (generate_inkling2_render_cases.py / Fixtures)
        Assert.Equal("<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_image|><|unused_200054|><|end_message|>"
                     + "<|message_user|><|content_text|>Describe this image in one sentence.<|end_message|>", r);
        // the text-only message keeps rendering as a plain string
        string plain = tpl.Apply([new ChatMessage { Role = "user", Content = "Hi" }], new ChatTemplateOptions { AddGenerationPrompt = true });
        Assert.DoesNotContain("<|content_image|>", plain);
    }

    [Fact]
    public void MessageContent_ReadsStringOrParts_AndWritesBackTheSameShape()
    {
        var plain = JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"user","content":"hi"}""")!;
        Assert.Equal("hi", plain.Content!.Text);
        Assert.Null(plain.Content.Parts);
        Assert.Contains("\"content\":\"hi\"", JsonSerializer.Serialize(plain));

        var parts = JsonSerializer.Deserialize<ChatMessageDto>(
            """{"role":"user","content":[{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"b"}]}""")!;
        Assert.Equal("ab", parts.Content!.Text);
        Assert.Equal(["data:image/png;base64,AA=="], parts.Content.ImageUrls.ToArray());
        string round = JsonSerializer.Serialize(parts);
        Assert.Contains("\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}", round);

        Assert.Throws<JsonException>(() => JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"user","content":[{"type":"input_audio"}]}"""));
        var nul = JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"assistant","content":null}""")!;
        Assert.Null(nul.Content);
    }

    [Fact]
    public void RequestConverter_KeepsTextOnlyArraysAsStrings_AndCarriesImageParts()
    {
        var textOnly = JsonSerializer.Deserialize<ChatMessageDto>("""{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}""")!;
        var withImage = JsonSerializer.Deserialize<ChatMessageDto>(
            """{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"q"}]}""")!;
        var msgs = DotLLM.Server.RequestConverter.ToMessages([textOnly, withImage]);
        Assert.Equal("ab", msgs[0].Content); Assert.Null(msgs[0].ContentParts);
        Assert.Equal("q", msgs[1].Content);
        Assert.Equal(["image_url", "text"], msgs[1].ContentParts!.Select(p => p.Type).ToArray());
        Assert.Equal(["data:image/png;base64,AA=="], DotLLM.Server.RequestConverter.ImageUrls([textOnly, withImage]).ToArray());
    }

    // ───────────── engine hand-off ─────────────

    private sealed class ImageStubModel : IModel, IImageInputModel
    {
        public const int Vocab = 32, HeadDim = 4;
        public readonly List<(int[] Tokens, int[] Positions, ImageSpan[] Spans)> ImageCalls = [];
        public int PlainForwards;
        public ModelConfig Config { get; } = new()
        {
            VocabSize = Vocab, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 8, IntermediateSize = 32,
            HeadDim = HeadDim, MaxSequenceLength = 256, Architecture = Architecture.Llama,
        };
        public long ComputeMemoryBytes => 0;
        public void Dispose() { }
        private unsafe ITensor Logits(int rows)
        {
            var t = UnmanagedTensor.Allocate(new TensorShape(rows, Vocab), DType.Float32, -1);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < rows * Vocab; i++) p[i] = -10f;
            for (int r = 0; r < rows; r++) p[r * Vocab + 5] = 10f;
            return t;
        }
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IKvCache? kv) { PlainForwards++; Touch(kv, t, p); return Logits(t.Length); }
        public ITensor ForwardWithImages(ReadOnlySpan<int> t, ReadOnlySpan<int> p, int d, IReadOnlyList<ImageSpan> images, bool last)
        {
            ImageCalls.Add((t.ToArray(), p.ToArray(), images.ToArray()));
            return Logits(t.Length);
        }
        private static unsafe void Touch(IKvCache? kv, ReadOnlySpan<int> t, ReadOnlySpan<int> p)
        {
            nuint bytes = (nuint)(t.Length * HeadDim * sizeof(float));
            nint k = (nint)System.Runtime.InteropServices.NativeMemory.AlignedAlloc(bytes, 64);
            System.Runtime.InteropServices.NativeMemory.Clear((void*)k, bytes);
            kv!.Update(new TensorRef(t.Length, HeadDim, DType.Float32, -1, k), new TensorRef(t.Length, HeadDim, DType.Float32, -1, k), p, 0);
            System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)k);
        }
    }

    private sealed class WordTokenizer : ITokenizer
    {
        public int[] Encode(string text) => text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        public string Decode(ReadOnlySpan<int> ids) => string.Join(' ', ids.ToArray());
        public string Decode(ReadOnlySpan<int> ids, bool strip) => Decode(ids);
        public string DecodeToken(int id) => id.ToString();
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public int VocabSize => ImageStubModel.Vocab;
        public int CountTokens(string text) => Encode(text).Length;
    }

    [Fact]
    public void Generator_WidensPlaceholders_PassesSpansToTheModel_AndRetainsNoPrefix()
    {
        var model = new ImageStubModel();
        using var emb1 = new ImageEmbeddings(2, 1, 8, "h1");    // 2 tokens
        using var emb2 = new ImageEmbeddings(3, 1, 8, "h2");    // 3 tokens
        var prefixCache = new PrefixCache(4);
        var gen = new TextGenerator(model, new WordTokenizer(), kvCacheFactory: (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg, false), size),
            prefixCache: prefixCache);
        var options = new InferenceOptions { MaxTokens = 2, Temperature = 0, Images = [emb1, emb2], ImagePlaceholderTokenId = 9 };
        var r = gen.Generate("3 9 4 9 6", options);
        Assert.Single(model.ImageCalls);
        var call = model.ImageCalls[0];
        Assert.Equal(new[] { 3, 9, 9, 4, 9, 9, 9, 6 }, call.Tokens);
        Assert.Equal(Enumerable.Range(0, 8).ToArray(), call.Positions);
        Assert.Equal([1, 4], call.Spans.Select(s => s.Start).ToArray());
        Assert.Equal([2, 3], call.Spans.Select(s => s.Embeddings.Tokens).ToArray());
        Assert.Equal(8, r.PromptTokenCount);
        // a second request with the SAME token ids but other images must not reuse anything: both prefill the full prompt through the image path
        using var other = new ImageEmbeddings(2, 1, 8, "other");
        gen.Generate("3 9 4 9 6", new InferenceOptions { MaxTokens = 2, Temperature = 0, Images = [other, emb2], ImagePlaceholderTokenId = 9 });
        Assert.Equal(2, model.ImageCalls.Count);
        Assert.Equal(8, model.ImageCalls[1].Tokens.Length);
        Assert.Equal(0, prefixCache.EntryCount);   // nothing retained: a token-keyed hit would serve another image's KV
    }

    [Fact]
    public void Generator_RefusesImagesWhenTheModelCannotTakeThem_AndOnAPlaceholderCountMismatch()
    {
        var gen = new TextGenerator(new ImageStubModel(), new WordTokenizer(), kvCacheFactory: (cfg, size) => new SimpleKvCache(KvGeometry.FromConfig(cfg, false), size));
        using var emb = new ImageEmbeddings(1, 1, 8, "h");
        Assert.Throws<ArgumentException>(() => gen.Generate("3 4", new InferenceOptions { MaxTokens = 1, Images = [emb], ImagePlaceholderTokenId = 9 }));
        Assert.Throws<ArgumentException>(() => gen.Generate("3 9", new InferenceOptions { MaxTokens = 1, Images = [emb] }));
    }
}
