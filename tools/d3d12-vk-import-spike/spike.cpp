// #1250 spike: can Vulkan import a D3D12 shared heap, which memory type does it get, and is it as fast as
// device-local memory under weight-reuse access? Compares: device-local | D3D12 DEFAULT heap | D3D12 CUSTOM L0
// write-back heap | VK_EXT_external_memory_host (anonymous RW pages; the status-quo import).
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <psapi.h>
#include <dstorage.h>
static ID3D12Heap* g_lastHeap = nullptr;
static const GUID IID_DSFactory = {0x6924ea0c,0xc3cd,0x4826,{0xb1,0x0a,0xf6,0x4f,0x4e,0xd9,0x27,0xc1}};
static const GUID IID_DSFile = {0x5de95e7b,0x955a,0x4868,{0xa7,0x3c,0x24,0x3b,0x29,0xf4,0xb8,0xda}};
static const GUID IID_DSQueue = {0xcfdbd83f,0x9e06,0x4fda,{0x8e,0xa5,0x69,0x04,0x21,0x37,0xf4,0x9b}};
static void memLine(const char* tag) {
    MEMORYSTATUSEX ms{sizeof ms}; GlobalMemoryStatusEx(&ms);
    PROCESS_MEMORY_COUNTERS_EX pc{}; GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pc, sizeof pc);
    printf("  [mem %s] availphys=%llu MiB  proc: ws=%llu MiB private=%llu MiB\n", tag, (unsigned long long)(ms.ullAvailPhys >> 20),
           (unsigned long long)(pc.WorkingSetSize >> 20), (unsigned long long)(pc.PrivateUsage >> 20));
}
#define CK(x) do { VkResult r_ = (x); if (r_ < 0) { printf("FAIL %s = %d (line %d)\n", #x, r_, __LINE__); exit(1);} } while (0)

static VkInstance inst; static VkPhysicalDevice pd; static VkDevice dev; static VkQueue q; static uint32_t qfi;
static VkPhysicalDeviceMemoryProperties mp;
static PFN_vkGetMemoryWin32HandlePropertiesKHR pGetHandleProps;
static PFN_vkGetMemoryHostPointerPropertiesEXT pGetHostProps;

static void printTypes(const char* tag, uint32_t bits) {
    printf("  [%s] memoryTypeBits=0x%X ->", tag, bits);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if (bits & (1u << i)) {
        auto f = mp.memoryTypes[i].propertyFlags;
        printf(" %u{heap%u%s%s%s%s}", i, mp.memoryTypes[i].heapIndex, (f & 1) ? " DEVLOCAL" : "", (f & 2) ? " HV" : "",
               (f & 8) ? " CACHED" : "", (f & 0x40) ? " UNCACHED_AMD" : "");
    }
    printf("\n");
}
struct Buf { VkBuffer b = 0; VkDeviceMemory m = 0; uint32_t type = ~0u; const char* name = ""; };

// want/avoid filter the memory type; for imports `forceType` (if != ~0) overrides the choice.
static Buf makeBuf(VkDeviceSize size, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid,
                   VkExternalMemoryHandleTypeFlagBits ext, void* importPtrOrHandle, bool isHandle, uint32_t forceType = ~0u) {
    Buf r;
    VkExternalMemoryBufferCreateInfo eb{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    eb.handleTypes = ext;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = ext ? &eb : nullptr; bi.size = size; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    CK(vkCreateBuffer(dev, &bi, nullptr, &r.b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, r.b, &req);
    uint32_t bits = req.memoryTypeBits;
    printTypes("buffer reqs", bits);
    if (ext) {
        if (isHandle) {
            VkMemoryWin32HandlePropertiesKHR hp{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
            VkResult pr = pGetHandleProps(dev, ext, (HANDLE)importPtrOrHandle, &hp);
            if (pr < 0) { printf("  vkGetMemoryWin32HandlePropertiesKHR failed %d\n", pr); r.b = 0; return r; }
            printTypes("handle props", hp.memoryTypeBits); bits &= hp.memoryTypeBits;
        } else {
            VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            CK(pGetHostProps(dev, ext, importPtrOrHandle, &hp)); printTypes("host ptr props", hp.memoryTypeBits); bits &= hp.memoryTypeBits;
        }
    }
    uint32_t pick = ~0u;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        auto f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) != want || (f & avoid)) continue;
        pick = i; break;
    }
    if (forceType != ~0u && (bits & (1u << forceType))) pick = forceType;
    if (pick == ~0u) { printf("  no memory type (bits=0x%X)\n", bits); vkDestroyBuffer(dev, r.b, nullptr); r.b = 0; return r; }
    VkImportMemoryWin32HandleInfoKHR iw{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR};
    VkImportMemoryHostPointerInfoEXT ih{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = std::max<VkDeviceSize>(size, req.size); ai.memoryTypeIndex = pick;
    if (ext && isHandle) { iw.handleType = ext; iw.handle = (HANDLE)importPtrOrHandle; ai.pNext = &iw; }
    if (ext && !isHandle) { ih.handleType = ext; ih.pHostPointer = importPtrOrHandle; ai.pNext = &ih; }
    VkResult ar = vkAllocateMemory(dev, &ai, nullptr, &r.m);
    if (ar < 0) { printf("  vkAllocateMemory(type %u) failed %d\n", pick, ar); vkDestroyBuffer(dev, r.b, nullptr); r.b = 0; return r; }
    CK(vkBindBufferMemory(dev, r.b, r.m, 0));
    r.type = pick; return r;
}

int main(int argc, char** argv) {
    const VkDeviceSize S = 512ull << 20; // weights
    const uint32_t chunkBytes = 64 << 10, chunkVec4 = chunkBytes / 16;
    uint32_t chunks = (uint32_t)(S / chunkBytes);
    int tilesList[] = {1, 4, 8, 16};
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &app;
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr); std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    uint32_t want = argc > 1 ? (uint32_t)strtoul(argv[1], 0, 0) : 0x1002;
    pd = 0; VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    for (auto p : pds) {
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        vkGetPhysicalDeviceProperties2(p, &p2);
        printf("VK device: %s vendor=0x%X\n", p2.properties.deviceName, p2.properties.vendorID);
        if (!pd && p2.properties.vendorID == want) pd = p;
    }
    if (!pd) { printf("no device\n"); return 1; }
    VkPhysicalDeviceProperties2 pp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2}; pp.pNext = &idp; vkGetPhysicalDeviceProperties2(pd, &pp);
    printf("using %s, LUID valid=%d\n", pp.properties.deviceName, (int)idp.deviceLUIDValid);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) printf("type %u heap%u flags=0x%X\n", i, mp.memoryTypes[i].heapIndex, mp.memoryTypes[i].propertyFlags);
    uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr); std::vector<VkQueueFamilyProperties> qf(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float prio = 1; VkDeviceQueueCreateInfo dq{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; dq.queueFamilyIndex = qfi; dq.queueCount = 1; dq.pQueuePriorities = &prio;
    const char* exts[] = {"VK_KHR_external_memory", "VK_KHR_external_memory_win32", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &dq; dci.enabledExtensionCount = 3; dci.ppEnabledExtensionNames = exts;
    CK(vkCreateDevice(pd, &dci, nullptr, &dev)); vkGetDeviceQueue(dev, qfi, 0, &q);
    pGetHandleProps = (PFN_vkGetMemoryWin32HandlePropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryWin32HandlePropertiesKHR");
    pGetHostProps = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");

    IDXGIFactory4* fac; CreateDXGIFactory1(IID_PPV_ARGS(&fac));
    LUID luid; memcpy(&luid, idp.deviceLUID, sizeof luid);
    IDXGIAdapter* ad = nullptr; if (FAILED(fac->EnumAdapterByLuid(luid, IID_PPV_ARGS(&ad)))) { printf("no DXGI adapter for LUID\n"); return 1; }
    DXGI_ADAPTER_DESC adesc; ad->GetDesc(&adesc); wprintf(L"D3D12 adapter: %s\n", adesc.Description);
    ID3D12Device* d3d; if (FAILED(D3D12CreateDevice(ad, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d)))) { printf("D3D12CreateDevice failed\n"); return 1; }
    D3D12_FEATURE_DATA_ARCHITECTURE arch{}; d3d->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof arch);
    printf("D3D12 UMA=%d CacheCoherentUMA=%d\n", arch.UMA, arch.CacheCoherentUMA);
    auto mkHeap = [&](const char* tag, D3D12_HEAP_PROPERTIES hp) -> HANDLE {
        D3D12_HEAP_DESC hd{}; hd.SizeInBytes = S; hd.Properties = hp; hd.Alignment = 64 << 10;
        hd.Flags = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        ID3D12Heap* h; HRESULT hr = d3d->CreateHeap(&hd, IID_PPV_ARGS(&h)); g_lastHeap = h;
        if (FAILED(hr)) { printf("CreateHeap(%s) failed 0x%X\n", tag, (unsigned)hr); return nullptr; }
        HANDLE sh; hr = d3d->CreateSharedHandle(h, nullptr, GENERIC_ALL, nullptr, &sh);
        if (FAILED(hr)) { printf("CreateSharedHandle(%s) failed 0x%X\n", tag, (unsigned)hr); return nullptr; }
        return sh; };
    D3D12_HEAP_PROPERTIES hpDef{}; hpDef.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_HEAP_PROPERTIES hpL0{}; hpL0.Type = D3D12_HEAP_TYPE_CUSTOM; hpL0.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK; hpL0.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    HANDLE hDef = mkHeap("DEFAULT", hpDef), hL0 = mkHeap("CUSTOM L0 WB", hpL0);

    std::vector<Buf> bufs;
    const auto NONE = (VkExternalMemoryHandleTypeFlagBits)0;
    printf("\n== device-local (vkAllocateMemory)\n");
    { Buf b = makeBuf(S, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, NONE, nullptr, false); b.name = "device-local"; bufs.push_back(b); }
    auto HEAPBIT = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP_BIT;
    if (hDef) { printf("== import D3D12 DEFAULT heap\n"); Buf b = makeBuf(S, 0, 0, HEAPBIT, hDef, true); b.name = "d3d12-default"; bufs.push_back(b); }
    if (hL0) { printf("== import D3D12 CUSTOM L0 write-back heap\n"); Buf b = makeBuf(S, 0, 0, HEAPBIT, hL0, true); b.name = "d3d12-custom-L0"; bufs.push_back(b); }
    printf("== VK_EXT_external_memory_host (anonymous RW pages)\n");
    void* anon = VirtualAlloc(nullptr, S, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    { Buf b = makeBuf(S, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, anon, false); b.name = "host-ptr-import"; bufs.push_back(b); }

    if (argc > 2 && !(argc > 3 && strcmp(argv[3], "nofile") == 0)) {
        printf("== VK_EXT_external_memory_host (FILE-BACKED writable view of %s)\n", argv[2]);
        HANDLE fh = CreateFileA(argv[2], GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (fh == INVALID_HANDLE_VALUE) printf("  open failed %lu\n", GetLastError());
        else {
            HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READWRITE, 0, 0, nullptr);
            // skip the first 1 GiB (header/metadata region is irrelevant); view start must be 64 KiB aligned
            void* view = mh ? MapViewOfFile(mh, FILE_MAP_WRITE, 0, 1u << 30, S) : nullptr;
            if (!view) printf("  map failed %lu\n", GetLastError());
            else {
                volatile unsigned char sink = 0; for (VkDeviceSize o = 0; o < S; o += 4096) sink += ((unsigned char*)view)[o]; // warm
                Buf b = makeBuf(S, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, 0, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view, false);
                b.name = "host-ptr-FILE"; bufs.push_back(b);
            }
        }
    }
    if (argc > 2) {
        printf("== DirectStorage: file -> D3D12 DEFAULT shared heap buffer -> Vulkan import\n");
        HANDLE dsh = mkHeap("DS DEFAULT", hpDef); ID3D12Heap* dsHeap = g_lastHeap;
        HMODULE dl = LoadLibraryA("dstorage.dll");
        auto pGet = dl ? (decltype(&DStorageGetFactory))GetProcAddress(dl, "DStorageGetFactory") : nullptr;
        if (!dsh || !pGet) printf("  no heap or dstorage.dll missing (%p)\n", (void*)dl);
        else {
            D3D12_RESOURCE_DESC rd{}; rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER; rd.Width = S; rd.Height = 1; rd.DepthOrArraySize = 1; rd.MipLevels = 1;
            rd.SampleDesc.Count = 1; rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            ID3D12Resource* res; HRESULT hr = d3d->CreatePlacedResource(dsHeap, 0, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
            IDStorageFactory* fac2 = nullptr; if (SUCCEEDED(hr)) hr = pGet(IID_DSFactory, (void**)&fac2);
            IDStorageFile* dfile = nullptr; wchar_t wp[1024]; MultiByteToWideChar(CP_UTF8, 0, argv[2], -1, wp, 1024);
            if (SUCCEEDED(hr)) hr = fac2->OpenFile(wp, IID_DSFile, (void**)&dfile);
            IDStorageQueue* dq2 = nullptr; DSTORAGE_QUEUE_DESC qd{}; qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY; qd.Priority = DSTORAGE_PRIORITY_NORMAL; qd.Device = d3d;
            if (SUCCEEDED(hr)) hr = fac2->CreateQueue(&qd, IID_DSQueue, (void**)&dq2);
            ID3D12Fence* fnc = nullptr; if (SUCCEEDED(hr)) hr = d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fnc));
            if (FAILED(hr)) printf("  DS setup failed 0x%X\n", (unsigned)hr);
            else {
                memLine("before DS");
                const UINT64 fileOff = 1ull << 30; const UINT32 reqBytes = 16u << 20;
                auto t0 = std::chrono::high_resolution_clock::now();
                for (UINT64 o = 0; o < S; o += reqBytes) {
                    DSTORAGE_REQUEST r{}; r.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE; r.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
                    r.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                    r.Source.File.Source = dfile; r.Source.File.Offset = fileOff + o; r.Source.File.Size = reqBytes;
                    r.Destination.Buffer.Resource = res; r.Destination.Buffer.Offset = o; r.Destination.Buffer.Size = reqBytes; r.UncompressedSize = reqBytes;
                    dq2->EnqueueRequest(&r);
                }
                dq2->EnqueueSignal(fnc, 1); dq2->Submit();
                HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr); fnc->SetEventOnCompletion(1, ev); WaitForSingleObject(ev, INFINITE);
                double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                DSTORAGE_ERROR_RECORD er{}; dq2->RetrieveErrorRecord(&er);
                printf("  DS read %llu MiB in %.2f s = %.2f GB/s, first failure hr=0x%X\n", (unsigned long long)(S >> 20), s, (double)S / s / 1e9, (unsigned)er.FirstFailure.HResult);
                memLine("after DS");
                Buf b = makeBuf(S, 0, 0, HEAPBIT, dsh, true); b.name = "d3d12-DS-filled";
                if (b.b) {
                    // verify first 4 MiB + last 4 MiB against the file via a host-visible copy
                    VkDeviceSize vb = 4u << 20;
                    Buf hostb = makeBuf(vb, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, NONE, nullptr, false);
                    bool ok = true;
                    for (int part = 0; part < 2 && hostb.b; part++) {
                        VkDeviceSize so = part ? S - vb : 0;
                        VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; VkCommandPoolCreateInfo pc2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pc2.queueFamilyIndex = qfi;
                        VkCommandPool cp2; CK(vkCreateCommandPool(dev, &pc2, nullptr, &cp2)); a.commandPool = cp2; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
                        VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &a, &cb)); VkCommandBufferBeginInfo bi2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi2));
                        VkBufferCopy rg{so, 0, vb}; vkCmdCopyBuffer(cb, b.b, hostb.b, 1, &rg); CK(vkEndCommandBuffer(cb));
                        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb; CK(vkQueueSubmit(q, 1, &si, 0)); CK(vkQueueWaitIdle(q));
                        void* mp2; CK(vkMapMemory(dev, hostb.m, 0, vb, 0, &mp2));
                        std::vector<char> ref(vb); std::ifstream rf(argv[2], std::ios::binary); rf.seekg(fileOff + so); rf.read(ref.data(), vb);
                        bool same = memcmp(ref.data(), mp2, vb) == 0; printf("  verify %s 4 MiB: %s\n", part ? "tail" : "head", same ? "IDENTICAL" : "MISMATCH"); ok &= same;
                        vkUnmapMemory(dev, hostb.m);
                    }
                    bufs.push_back(b);
                }
            }
        }
    }
    std::ifstream f("reuse.spv", std::ios::binary); std::vector<char> spv((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize = spv.size(); smi.pCode = (uint32_t*)spv.data();
    VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, nullptr, &sm));
    VkDescriptorSetLayoutBinding lb[2]{}; for (int i = 0; i < 2; i++) { lb[i].binding = i; lb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; lb[i].descriptorCount = 1; lb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dli.bindingCount = 2; dli.pBindings = lb;
    VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pli.setLayoutCount = 1; pli.pSetLayouts = &dsl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.layout = pl;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, 0, 1, &cpi, nullptr, &pipe));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 32; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dp));
    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, nullptr, &cp));
    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence; CK(vkCreateFence(dev, &fi, nullptr, &fence));

    uint32_t maxTiles = 16; VkDeviceSize outSize = (VkDeviceSize)chunks * maxTiles * 256 * 4;
    Buf out = makeBuf(outSize, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, NONE, nullptr, false);

    auto submit = [&](VkCommandBuffer cb) { VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        CK(vkResetFences(dev, 1, &fence)); CK(vkQueueSubmit(q, 1, &si, fence)); CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, ~0ull)); };
    auto newCb = [&]() { VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; a.commandPool = cp; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
        VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &a, &cb)); VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi)); return cb; };

    printf("\n== reuse benchmark: %llu MiB weights, %u KiB chunks; GB/s = tiles*S / time (higher = faster)\n", (unsigned long long)(S >> 20), chunkBytes >> 10);
    printf("%-18s type", "buffer"); for (int t : tilesList) printf("  tiles=%-3d", t); printf("\n");
    for (auto& b : bufs) {
        if (!b.b) continue;
        { VkCommandBuffer cb = newCb(); vkCmdFillBuffer(cb, b.b, 0, S, 0x01010101u); CK(vkEndCommandBuffer(cb)); submit(cb); }
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dai.descriptorPool = dp; dai.descriptorSetCount = 1; dai.pSetLayouts = &dsl;
        VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
        VkDescriptorBufferInfo di[2] = {{b.b, 0, S}, {out.b, 0, outSize}}; VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; i++) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &di[i]; }
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
        printf("%-18s %4u", b.name, b.type);
        for (int tiles : tilesList) {
            double best = 1e9;
            for (int rep = 0; rep < 6; rep++) {
                VkCommandBuffer cb = newCb(); vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
                uint32_t pc[2] = {chunkVec4, (uint32_t)tiles}; vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
                vkCmdDispatch(cb, chunks * tiles, 1, 1); CK(vkEndCommandBuffer(cb));
                auto t0 = std::chrono::high_resolution_clock::now(); submit(cb);
                double s = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
                if (rep > 0) best = std::min(best, s);
            }
            printf("  %9.1f", (double)tiles * S / best / 1e9);
        }
        printf("\n");
    }
    return 0;
}
