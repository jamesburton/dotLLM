using System.Threading;
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

    [Fact]
    public void LoadOrSkip_BoundsOvershootOnStableUnlockedFile_PropagatesOriginalFailure()
    {
        // The adversarial case a first pass of this guard got wrong: a self-consistent
        // header (valid length prefix, valid JSON, correct dtype/shape) whose data_offsets
        // were independently corrupted to point past the file — e.g. a bit flip, disk
        // corruption, or a loader bug, on an otherwise COMPLETE file. This produces the
        // exact same "exceed data section length" message a genuinely truncated download
        // does, so the message alone cannot discriminate the two. Because the file is
        // stable (not growing) and unlocked, CheckpointGuard must NOT skip this — it must
        // fail loudly, since silently skipping it would mask real corruption.
        string path = TempPath("bounds-overshoot-stable");
        try
        {
            WriteBoundsOvershootSafetensorsFile(path, declaredEnd: 1_000_000, actualDataBytes: 4);

            var ex = Assert.Throws<InvalidDataException>(() =>
                CheckpointGuard.LoadOrSkip(path, "regression fixture", () => SafetensorsFile.Open(path)));
            Assert.Contains("exceed data section length", ex.Message, StringComparison.Ordinal);
            Assert.DoesNotContain("checkpoint present but unreadable", ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public async Task LoadOrSkip_BoundsOvershootWhileFileStillGrowing_SkipsInsteadOfFailing()
    {
        // Same bounds-overshoot header shape as the stable case above, but this time a
        // background writer keeps appending to the file for the duration of the test —
        // simulating an active downloader. CheckpointGuard's own in-flight recheck should
        // observe the size changing and treat it as a genuine partial download -> skip.
        string path = TempPath("bounds-overshoot-growing");
        try
        {
            WriteBoundsOvershootSafetensorsFile(path, declaredEnd: 1_000_000, actualDataBytes: 4);

            using var stop = new CancellationTokenSource();
            Task writer = Task.Run(() =>
            {
                while (!stop.IsCancellationRequested)
                {
                    try
                    {
                        using var fs = new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
                        fs.WriteByte(0);
                    }
                    catch (IOException)
                    {
                        // Transient sharing conflict with CheckpointGuard's own lock probe —
                        // retry on the next iteration.
                    }
                    Thread.Sleep(20);
                }
            });

            try
            {
                var ex = Assert.Throws<Xunit.SkipException>(() =>
                    CheckpointGuard.LoadOrSkip(path, "regression fixture", () => SafetensorsFile.Open(path)));
                Assert.Contains(
                    "checkpoint present but unreadable (partial download or locked)",
                    ex.Message, StringComparison.Ordinal);
            }
            finally
            {
                stop.Cancel();
                await writer.WaitAsync(TimeSpan.FromSeconds(5));
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>
    /// Writes a safetensors file whose header is fully self-consistent (valid length
    /// prefix, valid JSON, single F32 tensor "w") but whose declared <c>data_offsets</c>
    /// end lands at <paramref name="declaredEnd"/> — far past the actual data section,
    /// which is only <paramref name="actualDataBytes"/> long. Reproduces both "a checkpoint
    /// whose header finished downloading but whose tensor bytes did not" and "a complete
    /// file with corrupted offsets" — the two scenarios <see cref="CheckpointGuard"/> must
    /// tell apart using signals other than the exception message.
    /// </summary>
    private static void WriteBoundsOvershootSafetensorsFile(string path, long declaredEnd, int actualDataBytes)
    {
        string json = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,"
            + declaredEnd.ToString(System.Globalization.CultureInfo.InvariantCulture) + "]}}";
        byte[] header = System.Text.Encoding.UTF8.GetBytes(json);
        byte[] lengthPrefix = BitConverter.GetBytes((ulong)header.Length);
        byte[] content = new byte[lengthPrefix.Length + header.Length + actualDataBytes];
        Buffer.BlockCopy(lengthPrefix, 0, content, 0, lengthPrefix.Length);
        Buffer.BlockCopy(header, 0, content, lengthPrefix.Length, header.Length);
        File.WriteAllBytes(path, content);
    }

    private static string TempPath(string label)
        => Path.Combine(Path.GetTempPath(), $"dotllm-384-{label}-{Guid.NewGuid():N}.safetensors");
}
