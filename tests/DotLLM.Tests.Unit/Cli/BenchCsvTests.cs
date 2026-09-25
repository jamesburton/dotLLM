using DotLLM.Cli.Benchmarking;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

public sealed class BenchCsvTests
{
    [Fact]
    public void Header_Matches_Results_Csv_Header()
    {
        Assert.Equal(
            "date,host,device,backend,runtime,runtime_version,model,quant,pp_tok_s,tg_tok_s,tg_ctx_depth,settings,notes",
            BenchCsv.Header);
    }

    [Fact]
    public void FormatRow_Produces_Column_Aligned_Row()
    {
        string row = BenchCsv.FormatRow(
            new DateOnly(2026, 7, 16), "strix-halo", "Zen5-16c", "cpu", "dev-81bec9f6",
            "SmolLM-135M", "Q8_0", ppTokS: 231.4, tgTokS: 56.71, tgCtxDepth: 512,
            settings: "bench --device cpu -p 512 -n 128 -r 5",
            notes: "pp512/tg128; tg best 63.9");

        Assert.Equal(
            "2026-07-16,strix-halo,Zen5-16c,cpu,dotLLM,dev-81bec9f6,SmolLM-135M,Q8_0,231,56.7,512," +
            "\"bench --device cpu -p 512 -n 128 -r 5\",\"pp512/tg128; tg best 63.9\"",
            row);
    }

    [Fact]
    public void FormatRow_Has_Same_Column_Count_As_Header_Even_With_Commas_In_Free_Text()
    {
        string row = BenchCsv.FormatRow(
            new DateOnly(2026, 1, 2), "host", "dev", "vulkan", "dev-abc",
            "Model", "Q4_K_M", 12.3, 4.56, 640,
            settings: "a, b, c", notes: "x, y");

        // Split respecting quotes: count unquoted commas.
        int columns = 1;
        bool inQuotes = false;
        foreach (char c in row)
        {
            if (c == '"') inQuotes = !inQuotes;
            else if (c == ',' && !inQuotes) columns++;
        }
        Assert.Equal(BenchCsv.Header.Split(',').Length, columns);
    }

    [Fact]
    public void FormatRow_Escapes_Embedded_Quotes_In_Free_Text()
    {
        string row = BenchCsv.FormatRow(
            new DateOnly(2026, 1, 2), "h", "d", "cpu", "v",
            "m", "q", 1, 1, 0, settings: "say \"hi\"", notes: "n");

        Assert.Contains("\"say \"\"hi\"\"\"", row, StringComparison.Ordinal);
    }

    [Fact]
    public void FormatRow_Sanitizes_Commas_In_Bare_Fields()
    {
        string row = BenchCsv.FormatRow(
            new DateOnly(2026, 1, 2), "h,x", "d", "cpu", "v",
            "m", "q", 1, 1, 0, settings: "s", notes: "n");

        Assert.StartsWith("2026-01-02,h;x,", row, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("SmolLM-135M.Q8_0.gguf", null, "Q8_0")]
    [InlineData("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf", null, "Q4_K_M")]
    [InlineData("Llama-3.2-3B-Instruct.IQ4_XS.gguf", null, "IQ4_XS")]
    [InlineData("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", null, "UD-Q4_K_M")]
    [InlineData("model.gguf", null, "unknown")]
    [InlineData("model.gguf", "Q6_K", "Q6_K")]
    public void InferQuantLabel_Reads_Filename_Suffix(string fileName, string? flag, string expected)
    {
        Assert.Equal(expected, BenchEnvironment.InferQuantLabel(fileName, flag));
    }

    [Theory]
    [InlineData("SmolLM-135M.Q8_0.gguf", "Q8_0", "SmolLM-135M")]
    [InlineData("gemma-4-26B-A4B-it-UD-Q4_K_M.gguf", "UD-Q4_K_M", "gemma-4-26B-A4B-it")]
    [InlineData("model.gguf", "unknown", "model")]
    public void InferModelName_Strips_Quant_Suffix(string fileName, string quant, string expected)
    {
        Assert.Equal(expected, BenchEnvironment.InferModelName(fileName, quant));
    }

    [Theory]
    [InlineData("dev", "dev")]
    [InlineData("main", "main")]
    [InlineData("issue/140-dotllm-bench-cli", "issue140")]
    [InlineData("feature/mamba-3", "featuremamba")]
    [InlineData(null, "dev")]
    [InlineData("HEAD", "dev")]
    public void TagFromBranch_Compacts_Branch_Names(string? branch, string expected)
    {
        Assert.Equal(expected, BenchEnvironment.TagFromBranch(branch));
    }
}
