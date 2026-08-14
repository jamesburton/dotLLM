using DotLLM.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Regression coverage for <see cref="CheckpointGuard"/> (issue #384): a checkpoint that
/// is present but unreadable (partial in-flight download or locked by another process)
/// must SKIP, while a complete-but-corrupt checkpoint must still FAIL — the discriminating
/// requirement issue #384 calls out explicitly. Exercises the guard directly against small
/// synthetic safetensors byte layouts (not a real multi-GB fixture), so it runs
/// unconditionally with no <c>DOTLLM_*</c> gate.
/// </summary>
public sealed class CheckpointGuardTests
{
    [Fact]
    public void LoadOrSkip_TruncatedHeaderLengthPrefix_SkipsInsteadOfFailing()
    {
        // An 8-byte header-length prefix (declaring a 50-byte JSON header) with NO header
        // bytes following it — exactly the shape of a download interrupted right after the
        // first 8 bytes landed on disk. SafetensorsFile.Open throws InvalidDataException
        // with the "would read past EOF" truncation signature.
        string path = TempPath("truncated");
        try
        {
            File.WriteAllBytes(path, BitConverter.GetBytes(50UL));

            var ex = Assert.Throws<Xunit.SkipException>(() =>
                CheckpointGuard.LoadOrSkip(path, "regression fixture", () => SafetensorsFile.Open(path)));
            Assert.Contains(
                "checkpoint present but unreadable (partial download or locked)",
                ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadOrSkip_FileLockedByAnotherHandle_SkipsInsteadOfFailing()
    {
        // Simulates a downloader still holding the file open for writing: the content
        // doesn't matter (SafetensorsFile.Open never gets past its own FileStream
        // constructor), only that another handle denies FileShare.Read.
        string path = TempPath("locked");
        try
        {
            File.WriteAllBytes(path, new byte[8]);

            using var exclusiveHandle = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);

            var ex = Assert.Throws<Xunit.SkipException>(() =>
                CheckpointGuard.LoadOrSkip(path, "regression fixture", () => SafetensorsFile.Open(path)));
            Assert.Contains(
                "checkpoint present but unreadable (partial download or locked)",
                ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LoadOrSkip_CompleteButCorruptFile_PropagatesOriginalFailure()
    {
        // The header-length prefix matches the actual header bytes present (nothing is
        // truncated — every declared offset is satisfiable), but the header itself is not
        // valid JSON: real corruption, not a partial download. This is the discriminating
        // case from issue #384 task item 3 and must NOT be converted to a skip.
        string path = TempPath("corrupt");
        try
        {
            byte[] header = "{invalid}"u8.ToArray();
            byte[] lengthPrefix = BitConverter.GetBytes((ulong)header.Length);
            byte[] content = new byte[lengthPrefix.Length + header.Length];
            Buffer.BlockCopy(lengthPrefix, 0, content, 0, lengthPrefix.Length);
            Buffer.BlockCopy(header, 0, content, lengthPrefix.Length, header.Length);
            File.WriteAllBytes(path, content);

            var ex = Assert.Throws<InvalidDataException>(() =>
                CheckpointGuard.LoadOrSkip(path, "regression fixture", () => SafetensorsFile.Open(path)));
            Assert.DoesNotContain("checkpoint present but unreadable", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static string TempPath(string label)
        => Path.Combine(Path.GetTempPath(), $"dotllm-384-{label}-{Guid.NewGuid():N}.safetensors");
}
