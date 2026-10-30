# d3d12-vk-import-spike (#1250)

Standalone C++ spike (mingw g++ + Vulkan SDK + Windows D3D12/DXGI + DirectStorage 1.3 from NuGet `Microsoft.Direct3D.DirectStorage`).
Not built by the solution. It compares, with a weight-reuse compute shader (`reuse.comp`), a device-local buffer, a
shared D3D12 DEFAULT heap imported into Vulkan, anonymous host-pointer import, a file-backed writable host-pointer import of a
GGUF, and a DirectStorage-filled D3D12 heap. Build: `glslc -O reuse.comp -o reuse.spv`, then
`g++ -O2 -std=c++17 -I$VULKAN_SDK/Include -I<ds>/native/include spike.cpp -o spike.exe -L$VULKAN_SDK/Lib -lvulkan-1 -ld3d12 -ldxgi -ldxguid -lpsapi`
and put `dstorage.dll`, `dstoragecore.dll`, `reuse.spv` next to it. Run: `spike.exe 0x1002 <file.gguf> [nofile]`
(`nofile` skips the file-backed arm, which otherwise holds the GGUF open for write and makes DirectStorage's OpenFile fail with a sharing violation).
Take the GPU lock first. Results: `.docs/1250-attribution.md` and the #1250 comments.
