using System.Runtime.InteropServices;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for the VK_EXT_external_memory_host zero-copy weight import path
/// (<see cref="HostVisibleBuffer"/> + <see cref="VulkanDevice.TryWrapHostVisible"/>).
/// All tests are gated on the presence of a Vulkan loader/device — they
/// skip cleanly on hosts without one. The probe + fallback behavior is
/// validated regardless of whether the driver supports the extension; the
/// import-succeeds path additionally requires
/// <see cref="VulkanDevice.HasExternalMemoryHost"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class HostVisibleBufferTests
{
    [SkippableFact]
    public void Probe_DoesNotCrash_WhenExtensionAbsent()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        // The probe runs unconditionally in VulkanDevice.Create. Pass criterion
        // is simply that we got here — i.e. ProbeExternalMemoryHost did not
        // throw on a driver that doesn't expose the extension, and the device
        // constructed successfully whether or not the extension was enabled.
        using var device = VulkanDevice.Create();
        Assert.NotNull(device);
        // HasExternalMemoryHost is true XOR false — either is valid; we just
        // need a consistent state.
        if (device.HasExternalMemoryHost)
        {
            Assert.True(device.MinImportedHostPointerAlignment > 0,
                "Driver reports HasExternalMemoryHost but zero alignment — would divide by zero in import.");
            // x86-64 page size is 4096 in practice. Anything else is exotic
            // (typically larger — POWER9 is 64KiB) but we only assert on the
            // common case to flag suspicious driver behavior.
            Assert.True(device.MinImportedHostPointerAlignment <= 65536,
                $"Unusually large minImportedHostPointerAlignment: {device.MinImportedHostPointerAlignment}");
        }
        else
        {
            Assert.Equal(0u, (uint)device.MinImportedHostPointerAlignment);
        }
    }

    [SkippableFact]
    public unsafe void TryCreate_ReturnsNull_WhenExtensionAbsent()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        Skip.If(device.HasExternalMemoryHost,
            "Driver supports VK_EXT_external_memory_host — testing absent-path requires a host without it.");

        // Allocate a page-aligned 4 KiB host buffer; even if alignment is
        // hypothetically satisfied, the absence of the extension must short
        // out the call.
        const int pageBytes = 4096;
        void* host = NativeMemory.AlignedAlloc(pageBytes, pageBytes);
        try
        {
            var buf = HostVisibleBuffer.TryCreate(device, (nint)host, pageBytes);
            Assert.Null(buf);
        }
        finally
        {
            NativeMemory.AlignedFree(host);
        }
    }

    [SkippableFact]
    public unsafe void TryCreate_ImportsAlignedPointer_WhenSupported()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not support VK_EXT_external_memory_host on this host.");

        ulong alignment = device.MinImportedHostPointerAlignment;
        Assert.True(alignment > 0);

        // Allocate one alignment-sized chunk so the start pointer is trivially
        // aligned. Round to int for AlignedAlloc.
        nuint chunkBytes = (nuint)alignment;
        void* host = NativeMemory.AlignedAlloc(chunkBytes, (nuint)alignment);
        try
        {
            // Fill with a pattern so we could in principle verify by mapping
            // the GPU buffer back via a compute shader — left out here since
            // wiring a one-shot copy kernel is overkill for a smoke test.
            new Span<byte>(host, (int)chunkBytes).Fill(0xA5);

            using var buf = HostVisibleBuffer.TryCreate(device, (nint)host, (long)chunkBytes);
            Assert.NotNull(buf);
            Assert.Equal((long)chunkBytes, buf!.Size);
            Assert.Equal(0L, buf.BindOffset);
            Assert.Equal((nint)host, buf.ImportedHostPointer);
            Assert.Equal(MemoryDomain.HostVisibleZeroCopy, buf.Domain);
            Assert.NotEqual(0, buf.Handle);
            Assert.NotEqual(0, buf.Memory);
        }
        finally
        {
            NativeMemory.AlignedFree(host);
        }
    }

    [SkippableFact]
    public unsafe void TryCreate_HandlesUnalignedPointer_ByPageRounding()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not support VK_EXT_external_memory_host on this host.");

        ulong alignment = device.MinImportedHostPointerAlignment;
        // Allocate two pages so we can pick an unaligned-but-still-in-bounds
        // start pointer inside the first page.
        nuint allocBytes = (nuint)alignment * 2;
        void* host = NativeMemory.AlignedAlloc(allocBytes, (nuint)alignment);
        try
        {
            // Offset by 128 bytes into the page — not page-aligned.
            const int offset = 128;
            nint unaligned = (nint)host + offset;
            long logicalSize = (long)alignment; // one full page worth, fits within the 2-page alloc

            using var buf = HostVisibleBuffer.TryCreate(device, unaligned, logicalSize);
            Assert.NotNull(buf);
            Assert.Equal(logicalSize, buf!.Size);
            // BindOffset must equal the original misalignment so the shader
            // descriptor reads logical byte 0 at the requested pointer.
            Assert.Equal((long)offset, buf.BindOffset);
            // Imported host pointer must be the page-aligned base, NOT the
            // unaligned input.
            Assert.Equal((nint)host, buf.ImportedHostPointer);
            // Imported size covers offset+logicalSize rounded up to alignment.
            Assert.True(buf.ImportedSize >= offset + logicalSize);
            Assert.Equal(0L, buf.ImportedSize % (long)alignment);
        }
        finally
        {
            NativeMemory.AlignedFree(host);
        }
    }

    [SkippableFact]
    public void Buffer_FromHostImport_FlagsAsHostImported()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasExternalMemoryHost,
            "Driver does not support VK_EXT_external_memory_host on this host.");

        ulong alignment = device.MinImportedHostPointerAlignment;
        unsafe
        {
            void* host = NativeMemory.AlignedAlloc((nuint)alignment, (nuint)alignment);
            try
            {
                using var buf = device.TryWrapHostVisible((nint)host, (long)alignment);
                Assert.NotNull(buf);
                Assert.True(buf!.IsHostImported);
                Assert.Equal((long)alignment, buf.Size);
                Assert.NotEqual(0, buf.Handle);
            }
            finally
            {
                NativeMemory.AlignedFree(host);
            }
        }
    }

    [SkippableFact]
    public void Buffer_FromDeviceLocal_NotHostImported()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/device.");

        using var device = VulkanDevice.Create();
        using var buf = device.AllocateDeviceLocal(4096);
        Assert.False(buf.IsHostImported);
        Assert.NotEqual(0, buf.Handle);
    }
}
