#:property AllowUnsafeBlocks=true

// Standalone Vulkan device-local allocation probe.
//
// WHY THIS EXISTS (see .docs/BONSAI2_27B_SUPPORT.md "Vulkan allocator bug"):
// loading Ternary-Bonsai-2-27B (PQ2_0, ~7.2 GB packed) through dotLLM's Vulkan
// backend dies with VK_ERROR_OUT_OF_DEVICE_MEMORY on a single ~356 MB
// DEVICE_LOCAL allocation, while the heap it targets (heap[1], 96 GiB on this
// Strix Halo part) reports 91.20 GiB budget / 0 usage from a *separate*
// process, and llama.cpp allocates 5.53 GiB on that same heap minutes later.
//
// This program deliberately does NOT reference DotLLM.Vulkan. It talks to
// vulkan-1 directly, so a failure here indicts the driver / allocation pattern
// and a success here indicts dotLLM's allocation path. The decisive instrument
// is VK_EXT_memory_budget queried IN THIS PROCESS before and after every
// allocation -- vulkaninfo's numbers describe vulkaninfo's process, not ours.
//
// Run:
//   dotnet run tools/vk-alloc-probe/vk_alloc_probe.cs -- [options]
// Options:
//   --mode info|ramp|cumul|bigblocks|mapwrite|all   (default: all)
//   --type <N>            memory type index to allocate from (default: auto = first
//                         strictly-DEVICE_LOCAL type, mirroring AllocateInternal)
//   --chunk <MiB>         cumul chunk size (default 8)
//   --count <N>           cumul max chunks (default 1200 -- 851 tensors + slack)
//   --host-pressure <GiB> touch this much host RAM before probing (regime (b))
//   --no-buffer           allocate memory without creating/binding a VkBuffer (control)

using System.Diagnostics;
using System.Runtime.InteropServices;
using static VkApi;

const int VK_SUCCESS = 0;
const int VK_ERROR_OUT_OF_HOST_MEMORY = -1;
const int VK_ERROR_OUT_OF_DEVICE_MEMORY = -2;

// ---- CLI -------------------------------------------------------------------
string mode = "all";
int typeIndexArg = -1;
long chunkMiB = 8;
int maxCount = 1200;
double hostPressureGiB = 0;
bool noBuffer = false;

for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--mode": mode = args[++i]; break;
        case "--type": typeIndexArg = int.Parse(args[++i]); break;
        case "--chunk": chunkMiB = long.Parse(args[++i]); break;
        case "--count": maxCount = int.Parse(args[++i]); break;
        case "--host-pressure": hostPressureGiB = double.Parse(args[++i]); break;
        case "--no-buffer": noBuffer = true; break;
        default: Console.Error.WriteLine($"unknown arg {args[i]}"); return 2;
    }
}

// ---- host pressure regime (b) ---------------------------------------------
// Regime (a) is a clean process; regime (b) reproduces dotLLM's host-side
// footprint (mmap'd GGUF + staging) without any of dotLLM's Vulkan code.
nint hostBlock = 0;
if (hostPressureGiB > 0)
{
    nuint hostBlockBytes = (nuint)(hostPressureGiB * 1024 * 1024 * 1024);
    Console.WriteLine($"[pressure] committing {hostPressureGiB:F1} GiB of host RAM, touching every page...");
    unsafe
    {
        byte* p = (byte*)NativeMemory.Alloc(hostBlockBytes);
        if (p == null) { Console.Error.WriteLine("host allocation failed"); return 3; }
        hostBlock = (nint)p;
        for (nuint off = 0; off < hostBlockBytes; off += 4096) p[off] = 1;
    }
    Console.WriteLine($"[pressure] resident; working set = {Environment.WorkingSet / (1024 * 1024)} MiB");
}

// ---- instance --------------------------------------------------------------
nint appNamePtr = Marshal.StringToHGlobalAnsi("vk_alloc_probe");
nint instance;
unsafe
{
    var appInfo = new VkApplicationInfo
    {
        sType = 0,
        pApplicationName = appNamePtr,
        applicationVersion = 1,
        pEngineName = appNamePtr,
        engineVersion = 1,
        apiVersion = (1u << 22) | (2u << 12), // VK_API_VERSION_1_2 -- 1.1+ needed for MemoryProperties2
    };
    var ici = new VkInstanceCreateInfo { sType = 1, pApplicationInfo = (nint)(&appInfo) };
    Check(vkCreateInstance(&ici, 0, out instance), "vkCreateInstance");
}

// ---- physical device -------------------------------------------------------
nint phys;
unsafe
{
    uint count = 0;
    Check(vkEnumeratePhysicalDevices(instance, ref count, null), "vkEnumeratePhysicalDevices(count)");
    if (count == 0) { Console.Error.WriteLine("no Vulkan physical devices"); return 3; }
    nint* devs = stackalloc nint[(int)count];
    Check(vkEnumeratePhysicalDevices(instance, ref count, devs), "vkEnumeratePhysicalDevices");
    phys = devs[0];

    byte* props = stackalloc byte[1024];
    new Span<byte>(props, 1024).Clear();
    vkGetPhysicalDeviceProperties(phys, props);
    uint api = *(uint*)props;
    // deviceName[256] sits at offset 20 (apiVersion, driverVersion, vendorID, deviceID, deviceType).
    string name = Marshal.PtrToStringAnsi((nint)(props + 20)) ?? "?";
    Console.WriteLine($"device: {name}  (driver api {api >> 22}.{(api >> 12) & 0x3FF}.{api & 0xFFF})");
    if (count > 1) Console.WriteLine($"note: {count} physical devices present; probing device 0");
}

// ---- VK_EXT_memory_budget available? --------------------------------------
bool budgetExt = false;
unsafe
{
    uint extCount = 0;
    vkEnumerateDeviceExtensionProperties(phys, 0, ref extCount, null);
    byte* ext = stackalloc byte[(int)extCount * 260];
    vkEnumerateDeviceExtensionProperties(phys, 0, ref extCount, ext);
    for (uint i = 0; i < extCount; i++)
        if ((Marshal.PtrToStringAnsi((nint)(ext + i * 260)) ?? "") == "VK_EXT_memory_budget")
            budgetExt = true;
}
Console.WriteLine($"VK_EXT_memory_budget: {(budgetExt ? "present" : "ABSENT (budget/usage columns unavailable)")}");

// ---- memory properties -----------------------------------------------------
var (types, heaps) = ReadMemoryProperties(phys);
Console.WriteLine();
Console.WriteLine("memory heaps:");
for (int h = 0; h < heaps.Count; h++)
    Console.WriteLine($"  heap[{h}] size={Mib(heaps[h].size),9:N0} MiB  flags=0x{heaps[h].flags:X} {HeapFlagNames(heaps[h].flags)}");
Console.WriteLine("memory types:");
for (int t = 0; t < types.Count; t++)
    Console.WriteLine($"  type[{t,2}] heap={types[t].heapIndex} flags=0x{types[t].flags:X2} {TypeFlagNames(types[t].flags)}");

// The type AllocateInternal picks for weights: strictly DEVICE_LOCAL, NOT host-visible.
int autoType = -1;
for (int t = 0; t < types.Count && autoType < 0; t++)
    if ((types[t].flags & 0x1) != 0 && (types[t].flags & 0x2) == 0) autoType = t;
if (autoType < 0)
    for (int t = 0; t < types.Count && autoType < 0; t++)
        if ((types[t].flags & 0x1) != 0) autoType = t;
if (autoType < 0) { Console.Error.WriteLine("no DEVICE_LOCAL memory type"); return 3; }

int probeType = typeIndexArg >= 0 ? typeIndexArg : autoType;
Console.WriteLine();
Console.WriteLine($"strict-DEVICE_LOCAL type (what AllocateInternal picks) = {autoType}; "
                + $"probing type {probeType} on heap {types[probeType].heapIndex}");

// ---- logical device --------------------------------------------------------
nint device;
unsafe
{
    uint qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, ref qcount, null);
    byte* qfp = stackalloc byte[(int)qcount * 24];
    vkGetPhysicalDeviceQueueFamilyProperties(phys, ref qcount, qfp);
    uint family = 0;
    for (uint i = 0; i < qcount; i++)
        if ((*(uint*)(qfp + i * 24) & 0x2) != 0) { family = i; break; } // VK_QUEUE_COMPUTE_BIT

    float prio = 1.0f;
    var qci = new VkDeviceQueueCreateInfo
    { sType = 2, queueFamilyIndex = family, queueCount = 1, pQueuePriorities = (nint)(&prio) };

    nint extName = Marshal.StringToHGlobalAnsi("VK_EXT_memory_budget");
    nint* extArr = stackalloc nint[1];
    extArr[0] = extName;

    var dci = new VkDeviceCreateInfo
    {
        sType = 3,
        queueCreateInfoCount = 1,
        pQueueCreateInfos = (nint)(&qci),
        enabledExtensionCount = budgetExt ? 1u : 0u,
        ppEnabledExtensionNames = budgetExt ? (nint)extArr : 0,
    };
    Check(vkCreateDevice(phys, &dci, 0, out device), "vkCreateDevice");
}

PrintBudget("baseline");
Console.WriteLine();

var allocs = new List<Alloc>();
try
{
    if (mode is "ramp" or "all") Ramp();
    if (mode is "cumul" or "all") Cumulative();
    if (mode is "bigblocks" or "all") BigBlocks();
    if (mode is "mapwrite" or "all") MapWrite();
    if (mode is not ("info" or "ramp" or "cumul" or "bigblocks" or "mapwrite" or "all"))
    {
        Console.Error.WriteLine($"unknown mode {mode}");
        return 2;
    }
}
finally
{
    FreeAll();
    vkDestroyDevice(device, 0);
    vkDestroyInstance(instance, 0);
    unsafe { if (hostBlock != 0) NativeMemory.Free((void*)hostBlock); }
}
return 0;

// =========================== experiments ====================================

// (1) Per-allocation size ceiling: ONE allocation of increasing size, freed
//     immediately. With nothing else resident this finds any static per-allocation
//     cap below maxMemoryAllocationSize (2 GiB on this part).
void Ramp()
{
    Console.WriteLine("=== ramp: single allocation, doubling size, freed immediately ===");
    for (long mib = 1; mib <= 2048; mib *= 2)
    {
        long bytes = mib * 1024 * 1024;
        var (res, a) = TryAlloc(bytes, probeType);
        Console.WriteLine($"  {mib,5} MiB -> {ResName(res)}");
        if (res == VK_SUCCESS) Free(a);
        else { Console.WriteLine($"  ceiling lies between {mib / 2} and {mib} MiB"); break; }
    }
    // The sizes the real load actually died on.
    foreach (long b in new long[] { 240_000_000, 356_515_840, 1_000_000_000 })
    {
        var (res, a) = TryAlloc(b, probeType);
        Console.WriteLine($"  {b,13:N0} B (a size the real load failed on) -> {ResName(res)}");
        if (res == VK_SUCCESS) Free(a);
    }
    PrintBudget("after ramp");
    Console.WriteLine();
}

// (2) Count vs cumulative bytes. dotLLM makes one allocation per tensor (851 for
//     this model); ggml-vulkan suballocates a few ~2 GiB blocks instead. A wall at
//     a COUNT means the arena is the fix; a wall at a cumulative BYTE total means a
//     commit cap and the arena will not help.
void Cumulative()
{
    Console.WriteLine($"=== cumul: up to {maxCount} x {chunkMiB} MiB, none freed ===");
    long bytes = chunkMiB * 1024 * 1024;
    long total = 0;
    for (int i = 0; i < maxCount; i++)
    {
        var (res, a) = TryAlloc(bytes, probeType);
        if (res != VK_SUCCESS)
        {
            Console.WriteLine($"  FAILED at allocation #{i + 1} (cumulative {total / (1024 * 1024)} MiB): {ResName(res)}");
            PrintBudget("  at failure");
            Forensics(bytes);
            break;
        }
        allocs.Add(a);
        total += bytes;
        if ((i + 1) % 64 == 0) PrintBudget($"  #{i + 1,4} cumulative {total / (1024 * 1024),6} MiB");
    }
    Console.WriteLine($"  reached {allocs.Count} allocations, {total / (1024 * 1024)} MiB");
    PrintBudget("  after cumul");
    FreeAll();
    PrintBudget("  after freeing cumul");
    Console.WriteLine();
}

// (3) The same order of magnitude in a handful of large blocks -- the ggml-vulkan
//     shape. Succeeding here where (2) failed is direct evidence for the arena.
void BigBlocks()
{
    Console.WriteLine("=== bigblocks: 4 x ~2 GiB (the ggml-vulkan shape) ===");
    long bytes = 2L * 1024 * 1024 * 1024 - 1024 * 1024; // just under maxMemoryAllocationSize
    long total = 0;
    for (int i = 0; i < 4; i++)
    {
        var (res, a) = TryAlloc(bytes, probeType);
        if (res != VK_SUCCESS)
        {
            Console.WriteLine($"  FAILED at block #{i + 1} (cumulative {total / (1024 * 1024)} MiB): {ResName(res)}");
            PrintBudget("  at failure");
            break;
        }
        allocs.Add(a);
        total += bytes;
        PrintBudget($"  block #{i + 1} cumulative {total / (1024 * 1024),6} MiB");
    }
    Console.WriteLine($"  reached {total / (1024 * 1024)} MiB in {allocs.Count} blocks");
    FreeAll();
    Console.WriteLine();
}

// (4) The intended future path: a DEVICE_LOCAL | HOST_VISIBLE block, mapped and
//     written straight from the host -- no staging copy. If a ~2 GiB one works
//     under host pressure, stop diagnosing and build the arena on this type.
void MapWrite()
{
    int t = -1;
    for (int i = 0; i < types.Count; i++)
        if ((types[i].flags & 0x1) != 0 && (types[i].flags & 0x2) != 0) { t = i; break; }
    if (t < 0) { Console.WriteLine("=== mapwrite: no DEVICE_LOCAL|HOST_VISIBLE type, skipped ==="); return; }

    Console.WriteLine($"=== mapwrite: DEVICE_LOCAL|HOST_VISIBLE type {t} (heap {types[t].heapIndex}), map + write ===");
    foreach (long mib in new long[] { 256, 1024, 2047 })
    {
        long bytes = mib * 1024 * 1024;
        var (res, a) = TryAlloc(bytes, t);
        if (res != VK_SUCCESS) { Console.WriteLine($"  {mib,5} MiB -> {ResName(res)}"); continue; }
        unsafe
        {
            int mapRes = vkMapMemory(device, a.memory, 0, (ulong)bytes, 0, out nint p);
            if (mapRes == VK_SUCCESS)
            {
                var sw = Stopwatch.StartNew();
                new Span<byte>((void*)p, (int)Math.Min(bytes, int.MaxValue)).Fill(0xAB);
                sw.Stop();
                vkUnmapMemory(device, a.memory);
                Console.WriteLine($"  {mib,5} MiB -> ok, mapped, host write {bytes / 1e9 / sw.Elapsed.TotalSeconds:F1} GB/s");
            }
            else Console.WriteLine($"  {mib,5} MiB -> allocated but vkMapMemory {ResName(mapRes)}");
        }
        Free(a);
    }
    PrintBudget("  after mapwrite");
    Console.WriteLine();
}

// On failure: transient (clears after a sleep), cumulative (clears after freeing
// half), or is the process poisoned (nothing clears it)? dotLLM's real run showed
// allocations succeeding only after a backoff sleep, which already argues against
// any purely static limit.
void Forensics(long bytes)
{
    Console.WriteLine("  --- forensics ---");
    Thread.Sleep(2000);
    var (r1, a1) = TryAlloc(bytes, probeType);
    Console.WriteLine($"  same size after 2 s sleep      -> {ResName(r1)}   (ok => TRANSIENT)");
    if (r1 == VK_SUCCESS) { Free(a1); return; }

    int half = allocs.Count / 2;
    for (int i = 0; i < half; i++) Free(allocs[i]);
    allocs.RemoveRange(0, half);
    PrintBudget($"  after freeing {half} allocations");
    var (r2, a2) = TryAlloc(bytes, probeType);
    Console.WriteLine($"  same size after freeing half   -> {ResName(r2)}   (ok => CUMULATIVE / commit cap)");
    if (r2 == VK_SUCCESS) { Free(a2); return; }

    FreeAll();
    PrintBudget("  after freeing everything");
    var (r3, a3) = TryAlloc(bytes, probeType);
    Console.WriteLine($"  same size after freeing all    -> {ResName(r3)}   (fail => PROCESS POISONED)");
    if (r3 == VK_SUCCESS) Free(a3);
}

// =========================== plumbing =======================================

// Mirrors VulkanDevice.AllocateInternal: create buffer with the same usage flags,
// query requirements, allocate the requirement size, bind.
(int, Alloc) TryAlloc(long bytes, int memType)
{
    nint buffer = 0;
    ulong allocSize = (ulong)bytes;

    if (!noBuffer)
    {
        unsafe
        {
            var bci = new VkBufferCreateInfo
            {
                sType = 12,
                size = (ulong)bytes,
                usage = 0x20 | 0x1 | 0x2, // STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST
                sharingMode = 0,
            };
            int cr = vkCreateBuffer(device, &bci, 0, out buffer);
            if (cr != VK_SUCCESS) return (cr, default);
            vkGetBufferMemoryRequirements(device, buffer, out var req);
            allocSize = req.size;
            if ((req.memoryTypeBits & (1u << memType)) == 0)
            {
                vkDestroyBuffer(device, buffer, 0);
                Console.WriteLine($"    (type {memType} excluded by buffer typeBits 0x{req.memoryTypeBits:X})");
                return (VK_ERROR_OUT_OF_DEVICE_MEMORY, default);
            }
        }
    }

    int res;
    nint memory;
    unsafe
    {
        var mai = new VkMemoryAllocateInfo
        { sType = 5, allocationSize = allocSize, memoryTypeIndex = (uint)memType };
        res = vkAllocateMemory(device, &mai, 0, out memory);
    }
    if (res != VK_SUCCESS)
    {
        if (buffer != 0) vkDestroyBuffer(device, buffer, 0);
        return (res, default);
    }
    if (buffer != 0)
    {
        int br = vkBindBufferMemory(device, buffer, memory, 0);
        if (br != VK_SUCCESS)
        {
            vkFreeMemory(device, memory, 0);
            vkDestroyBuffer(device, buffer, 0);
            return (br, default);
        }
    }
    return (VK_SUCCESS, new Alloc(buffer, memory, bytes));
}

void Free(Alloc a)
{
    if (a.buffer != 0) vkDestroyBuffer(device, a.buffer, 0);
    if (a.memory != 0) vkFreeMemory(device, a.memory, 0);
}

void FreeAll()
{
    foreach (var a in allocs) Free(a);
    allocs.Clear();
}

// The decisive instrument: heap budget/usage as THIS process sees it.
void PrintBudget(string label)
{
    if (!budgetExt) { Console.WriteLine(label); return; }
    var (budget, usage) = ReadBudget(phys, heaps.Count);
    var parts = new List<string>();
    for (int h = 0; h < heaps.Count; h++)
        parts.Add($"heap{h} use={Mib(usage[h]),7:N0}/bud={Mib(budget[h]),7:N0} MiB");
    Console.WriteLine($"{label,-44} {string.Join("  ", parts)}");
}

static long Mib(ulong b) => (long)(b / (1024 * 1024));

static string ResName(int r) => r switch
{
    VK_SUCCESS => "ok",
    VK_ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    VK_ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY",
    _ => $"VkResult {r}",
};

static string HeapFlagNames(uint f)
{
    var s = new List<string>();
    if ((f & 0x1) != 0) s.Add("DEVICE_LOCAL");
    if ((f & 0x2) != 0) s.Add("MULTI_INSTANCE");
    return s.Count == 0 ? "(none -- system RAM)" : string.Join("|", s);
}

static string TypeFlagNames(uint f)
{
    var s = new List<string>();
    if ((f & 0x01) != 0) s.Add("DEVICE_LOCAL");
    if ((f & 0x02) != 0) s.Add("HOST_VISIBLE");
    if ((f & 0x04) != 0) s.Add("HOST_COHERENT");
    if ((f & 0x08) != 0) s.Add("HOST_CACHED");
    if ((f & 0x10) != 0) s.Add("LAZILY_ALLOCATED");
    return s.Count == 0 ? "(none)" : string.Join("|", s);
}

static void Check(int r, string what)
{
    if (r != VK_SUCCESS) throw new Exception($"{what} failed: {ResName(r)}");
}

static (List<(uint flags, uint heapIndex)>, List<(ulong size, uint flags)>) ReadMemoryProperties(nint phys)
{
    var types = new List<(uint, uint)>();
    var heaps = new List<(ulong, uint)>();
    unsafe
    {
        // VkPhysicalDeviceMemoryProperties: u32 typeCount; VkMemoryType[32] (8 B each);
        // u32 heapCount @260; VkMemoryHeap[16] (16 B each) @264. Total 520 B.
        byte* mp = stackalloc byte[520];
        new Span<byte>(mp, 520).Clear();
        vkGetPhysicalDeviceMemoryProperties(phys, mp);
        uint typeCount = *(uint*)mp;
        for (uint i = 0; i < typeCount; i++)
            types.Add((*(uint*)(mp + 4 + i * 8), *(uint*)(mp + 4 + i * 8 + 4)));
        uint heapCount = *(uint*)(mp + 260);
        for (uint i = 0; i < heapCount; i++)
            heaps.Add((*(ulong*)(mp + 264 + i * 16), *(uint*)(mp + 264 + i * 16 + 8)));
    }
    return (types, heaps);
}

static (ulong[], ulong[]) ReadBudget(nint phys, int heapCount)
{
    var budget = new ulong[16];
    var usage = new ulong[16];
    unsafe
    {
        // VkPhysicalDeviceMemoryBudgetPropertiesEXT: sType @0, pNext @8,
        // heapBudget[16] @16, heapUsage[16] @144. Total 272 B.
        byte* bp = stackalloc byte[272];
        new Span<byte>(bp, 272).Clear();
        *(uint*)bp = 1000237000; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT

        // VkPhysicalDeviceMemoryProperties2: sType @0, pNext @8, memoryProperties @16 (520 B).
        byte* mp2 = stackalloc byte[536];
        new Span<byte>(mp2, 536).Clear();
        *(uint*)mp2 = 1000059006; // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2
        *(nint*)(mp2 + 8) = (nint)bp;

        vkGetPhysicalDeviceMemoryProperties2(phys, mp2);
        for (int h = 0; h < heapCount && h < 16; h++)
        {
            budget[h] = *(ulong*)(bp + 16 + h * 8);
            usage[h] = *(ulong*)(bp + 16 + 128 + h * 8);
        }
    }
    return (budget, usage);
}

readonly record struct Alloc(nint buffer, nint memory, long bytes);

[StructLayout(LayoutKind.Sequential)]
struct VkApplicationInfo
{
    public uint sType; public nint pNext; public nint pApplicationName;
    public uint applicationVersion; public nint pEngineName;
    public uint engineVersion; public uint apiVersion;
}

[StructLayout(LayoutKind.Sequential)]
struct VkInstanceCreateInfo
{
    public uint sType; public nint pNext; public uint flags; public nint pApplicationInfo;
    public uint enabledLayerCount; public nint ppEnabledLayerNames;
    public uint enabledExtensionCount; public nint ppEnabledExtensionNames;
}

[StructLayout(LayoutKind.Sequential)]
struct VkDeviceQueueCreateInfo
{
    public uint sType; public nint pNext; public uint flags;
    public uint queueFamilyIndex; public uint queueCount; public nint pQueuePriorities;
}

[StructLayout(LayoutKind.Sequential)]
struct VkDeviceCreateInfo
{
    public uint sType; public nint pNext; public uint flags;
    public uint queueCreateInfoCount; public nint pQueueCreateInfos;
    public uint enabledLayerCount; public nint ppEnabledLayerNames;
    public uint enabledExtensionCount; public nint ppEnabledExtensionNames;
    public nint pEnabledFeatures;
}

[StructLayout(LayoutKind.Sequential)]
struct VkBufferCreateInfo
{
    public uint sType; public nint pNext; public uint flags; public ulong size;
    public uint usage; public uint sharingMode;
    public uint queueFamilyIndexCount; public nint pQueueFamilyIndices;
}

[StructLayout(LayoutKind.Sequential)]
struct VkMemoryRequirements { public ulong size; public ulong alignment; public uint memoryTypeBits; }

[StructLayout(LayoutKind.Sequential)]
struct VkMemoryAllocateInfo
{
    public uint sType; public nint pNext; public ulong allocationSize; public uint memoryTypeIndex;
}

static unsafe class VkApi
{
    const string L = "vulkan-1";

    [DllImport(L)] public static extern int vkCreateInstance(VkInstanceCreateInfo* pCreateInfo, nint pAllocator, out nint pInstance);
    [DllImport(L)] public static extern void vkDestroyInstance(nint instance, nint pAllocator);
    [DllImport(L)] public static extern int vkEnumeratePhysicalDevices(nint instance, ref uint pCount, nint* pDevices);
    [DllImport(L)] public static extern void vkGetPhysicalDeviceProperties(nint phys, byte* pProps);
    [DllImport(L)] public static extern void vkGetPhysicalDeviceMemoryProperties(nint phys, byte* pProps);
    [DllImport(L)] public static extern void vkGetPhysicalDeviceMemoryProperties2(nint phys, byte* pProps);
    [DllImport(L)] public static extern void vkGetPhysicalDeviceQueueFamilyProperties(nint phys, ref uint pCount, byte* pProps);
    [DllImport(L)] public static extern int vkEnumerateDeviceExtensionProperties(nint phys, nint pLayerName, ref uint pCount, byte* pProps);
    [DllImport(L)] public static extern int vkCreateDevice(nint phys, VkDeviceCreateInfo* pCreateInfo, nint pAllocator, out nint pDevice);
    [DllImport(L)] public static extern void vkDestroyDevice(nint device, nint pAllocator);
    [DllImport(L)] public static extern int vkCreateBuffer(nint device, VkBufferCreateInfo* pCreateInfo, nint pAllocator, out nint pBuffer);
    [DllImport(L)] public static extern void vkDestroyBuffer(nint device, nint buffer, nint pAllocator);
    [DllImport(L)] public static extern void vkGetBufferMemoryRequirements(nint device, nint buffer, out VkMemoryRequirements pReq);
    [DllImport(L)] public static extern int vkAllocateMemory(nint device, VkMemoryAllocateInfo* pInfo, nint pAllocator, out nint pMemory);
    [DllImport(L)] public static extern void vkFreeMemory(nint device, nint memory, nint pAllocator);
    [DllImport(L)] public static extern int vkBindBufferMemory(nint device, nint buffer, nint memory, ulong offset);
    [DllImport(L)] public static extern int vkMapMemory(nint device, nint memory, ulong offset, ulong size, uint flags, out nint ppData);
    [DllImport(L)] public static extern void vkUnmapMemory(nint device, nint memory);
}
