namespace DotLLM.Models.Gguf;

/// <summary>
/// Parses only the metadata and the tensor table of a GGUF, without mapping or validating the tensor data section. Works on a
/// header-only range read (the first few MB of a remote file), which is how real-file contracts are checked without downloading
/// the weights (issue #1643: the Ripple-65B headers). Never use the result to load weights.
/// </summary>
public static class GgufHeaderOnly
{
    /// <summary>Reads the header, metadata and tensor descriptors of <paramref name="path"/>.</summary>
    /// <param name="path">A GGUF file, or any prefix of one that contains the whole header and tensor table.</param>
    /// <returns>The metadata and the tensor table.</returns>
    /// <exception cref="InvalidDataException">The prefix is too short or the bytes are not GGUF.</exception>
    public static (GgufMetadata Metadata, IReadOnlyList<GgufTensorDescriptor> Tensors) Read(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        return Read(fs);
    }

    /// <summary>Reads the header, metadata and tensor descriptors from <paramref name="stream"/> (positioned at the file start).</summary>
    /// <param name="stream">The source.</param>
    /// <returns>The metadata and the tensor table.</returns>
    public static (GgufMetadata Metadata, IReadOnlyList<GgufTensorDescriptor> Tensors) Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
        try
        {
            var header = GgufReader.ReadHeader(reader);
            var metadata = new GgufMetadata(GgufReader.ReadMetadata(reader, header));
            var tensors = GgufReader.ReadTensorInfos(reader, header);
            return (metadata, tensors);
        }
        catch (EndOfStreamException ex)
        {
            throw new InvalidDataException("The GGUF prefix ends inside the header or tensor table; read a larger prefix.", ex);
        }
    }
}
