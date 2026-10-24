namespace DotLLM.Vulkan;

/// <summary>
/// The 16-bit / 8-bit storage and float16 / int8 arithmetic feature bits a device supports; <see cref="VulkanDevice"/> enables every
/// one of them at creation (issue #374).
/// </summary>
/// <param name="StorageBuffer16BitAccess">16-bit types in storage buffers (<c>GL_EXT_shader_16bit_storage</c>).</param>
/// <param name="UniformAndStorageBuffer16BitAccess">16-bit types in uniform and storage buffers.</param>
/// <param name="StoragePushConstant16">16-bit types in push constants.</param>
/// <param name="ShaderFloat16">float16 arithmetic (<c>GL_EXT_shader_explicit_arithmetic_types_float16</c>).</param>
/// <param name="ShaderInt8">int8 arithmetic (<c>GL_EXT_shader_explicit_arithmetic_types_int8</c>).</param>
/// <param name="StorageBuffer8BitAccess">8-bit types in storage buffers (<c>GL_EXT_shader_8bit_storage</c>).</param>
/// <param name="UniformAndStorageBuffer8BitAccess">8-bit types in uniform and storage buffers.</param>
/// <param name="StoragePushConstant8">8-bit types in push constants.</param>
public readonly record struct VulkanNumericFeatures(
    bool StorageBuffer16BitAccess,
    bool UniformAndStorageBuffer16BitAccess,
    bool StoragePushConstant16,
    bool ShaderFloat16,
    bool ShaderInt8,
    bool StorageBuffer8BitAccess,
    bool UniformAndStorageBuffer8BitAccess,
    bool StoragePushConstant8)
{
    /// <summary>True when any 16-bit storage bit is supported (and therefore enabled).</summary>
    public bool Any16BitStorage => StorageBuffer16BitAccess || UniformAndStorageBuffer16BitAccess || StoragePushConstant16;

    /// <summary>True when any 8-bit storage bit is supported (and therefore enabled).</summary>
    public bool Any8BitStorage => StorageBuffer8BitAccess || UniformAndStorageBuffer8BitAccess || StoragePushConstant8;
}
