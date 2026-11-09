using Spectre.Console.Cli;

var app = new CommandApp();

app.Configure(DotLLM.Cli.CliConfiguration.Configure);

try
{
    DotLLM.Cli.Helpers.DirectStoragePrompt.OfferIfMissing(args);
    return app.Run(args);
}
catch (Exception ex) when (string.Equals(
    Environment.GetEnvironmentVariable("DOTLLM_DEBUG_ERRORS"), "1", StringComparison.Ordinal))
{
    Console.Error.WriteLine(ex.ToString());
    return 1;
}
