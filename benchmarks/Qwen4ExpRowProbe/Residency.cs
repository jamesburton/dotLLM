using DotLLM.Models.Gguf;
using DotLLM.Vulkan;

/// <summary>
/// <c>residency gguf</c>: loads the model through the Vulkan loader, runs one 1-token and one 64-token forward, and prints this process's own
/// device accounting (VulkanDevice.TotalLiveBytes + the per-heap VK_EXT_memory_budget snapshot) after load and after the forwards (#1018).
/// </summary>
internal static class Residency
{
    public static void Run(string[] args)
    {
        using var gguf = GgufFile.Open(args[1]);
        var cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        using var dev = VulkanDevice.Create();
        var (model, kvFactory) = VulkanModelLoader.CreateFromGguf(dev, gguf, cfg, Path.Combine(AppContext.BaseDirectory, "spv"), 0);
        Console.WriteLine($"[residency] {model.GetType().Name} after load: live {dev.TotalLiveBytes() / 1073741824.0:F2} GiB | {dev.MemorySnapshot()}");
        var ids = tk.Encode(string.Concat(Enumerable.Repeat("The capital of France is Paris. ", 16))).ToArray();
        using var cache = kvFactory(ids.Length + 8);
        using (var t = model.Forward(ids.AsSpan(0, 64), Enumerable.Range(0, 64).ToArray(), 0, cache)) { }
        using (var t = model.Forward(ids.AsSpan(64, 1), new[] { 64 }, 0, cache)) { }
        Console.WriteLine($"[residency] after forwards: live {dev.TotalLiveBytes() / 1073741824.0:F2} GiB | {dev.MemorySnapshot()}");
        (model as IDisposable)?.Dispose();
    }
}
