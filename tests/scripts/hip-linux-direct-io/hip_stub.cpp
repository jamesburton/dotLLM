// Test only: CPU allocations and POSIX O_DIRECT reads; never a production HIP fallback.
#define __HIP_PLATFORM_AMD__
#include <hip/amd_detail/hip_storage.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <type_traits>

static_assert(sizeof(hipAmdFileHandle_t) == 8 && offsetof(hipAmdFileHandle_t, fd) == 0);
static_assert(sizeof(struct statx) == 256 && offsetof(struct statx, stx_size) == 40);
static_assert(offsetof(struct statx, stx_dio_mem_align) == 152 && offsetof(struct statx, stx_dio_offset_align) == 156);
static_assert(sizeof(struct statfs) == 120);
using ReadAbi = hipError_t (*)(hipAmdFileHandle_t, void*, uint64_t, int64_t, uint64_t*, int32_t*);
static_assert(std::is_same_v<decltype(&hipAmdFileRead), ReadAbi>);

static int mode, allocations, frees, reads, badFree, badAbi;
static void *owner, *allocation;
static size_t allocationBytes;
extern "C" void probe_mode(int value) { mode = value; reads = 0; }
extern "C" int probe_stat(int index) {
    int stats[] = { allocations, frees, reads, badFree, badAbi };
    return stats[index];
}
extern "C" hipError_t hipInit(unsigned int) { return hipSuccess; }
extern "C" hipError_t hipSetDevice(int device) { return device == 0 ? hipSuccess : hipErrorInvalidDevice; }
extern "C" hipError_t hipDeviceSynchronize() { return hipSuccess; }
extern "C" hipError_t hipRuntimeGetVersion(int* value) { *value = HIP_VERSION; return hipSuccess; }
extern "C" const char* hipGetErrorName(hipError_t) { return "simulated HIP error"; }
extern "C" const char* hipGetErrorString(hipError_t) { return "test runtime fault"; }
extern "C" hipError_t hipMemGetInfo(size_t* available, size_t* total) {
    *total = 1ULL << 32; *available = mode == 5 ? 0 : *total; return hipSuccess;
}
extern "C" hipError_t hipMalloc(void** result, size_t bytes) {
    if (mode == 6) { *result = nullptr; return hipErrorOutOfMemory; }
    if (allocation || posix_memalign(&owner, 4096, bytes + 4096)) return hipErrorOutOfMemory;
    allocation = static_cast<char*>(owner) + 256; // Force the managed view to differ from the free pointer.
    allocationBytes = bytes; *result = allocation; ++allocations; return hipSuccess;
}
extern "C" hipError_t hipFree(void* pointer) {
    if (!allocation || pointer != allocation) { ++badFree; return hipErrorInvalidValue; }
    std::free(owner); owner = allocation = nullptr; ++frees; return hipSuccess;
}
hipError_t hipAmdFileRead(hipAmdFileHandle_t handle, void* pointer, uint64_t bytes,
    int64_t offset, uint64_t* received, int32_t* status) {
    ++reads;
    if ((fcntl(handle.fd, F_GETFL) & O_DIRECT) == 0 || (fcntl(handle.fd, F_GETFD) & FD_CLOEXEC) == 0
        || (uintptr_t)pointer % 4096 || offset % 4096 || bytes % 4096 || bytes > 32 * 1024 * 1024
        || (uintptr_t)pointer < (uintptr_t)allocation || bytes > allocationBytes
        || (uintptr_t)pointer - (uintptr_t)allocation > allocationBytes - bytes) {
        ++badAbi; return hipErrorInvalidValue;
    }
    *received = 0; *status = 0;
    if (mode == 1) { *received = bytes; return hipErrorInvalidValue; }
    if (mode == 2 || (mode == 8 && reads == 2)) { *received = bytes; *status = -5; return hipSuccess; }
    if (mode == 3) { *received = 1; return hipSuccess; }
    if (mode == 4) { *received = bytes + 1; return hipSuccess; }
    ssize_t count = pread(handle.fd, pointer, bytes, offset);
    if (count < 0) { *status = -errno; return hipSuccess; }
    *received = count; return hipSuccess;
}
extern "C" hipError_t hipGetProcAddress(const char* symbol, void** function, int version,
    uint64_t flags, hipDriverProcAddressQueryResult* status) {
    if (std::strcmp(symbol, "hipAmdFileRead") || version != HIP_VERSION || flags) { ++badAbi; return hipErrorInvalidValue; }
    *function = mode == 7 ? nullptr : reinterpret_cast<void*>(&hipAmdFileRead);
    *status = HIP_GET_PROC_ADDRESS_SUCCESS; return hipSuccess;
}
