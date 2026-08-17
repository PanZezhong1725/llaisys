#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::cpu {
namespace {

int deviceCount() { return 1; }
void selectDevice(int) {}
void synchronizeDevice() {}

llaisysStream_t makeStream() { return nullptr; }
void releaseStream(llaisysStream_t) {}
void synchronizeStream(llaisysStream_t) {}

void *allocate(size_t bytes) {
    return bytes == 0 ? nullptr : std::malloc(bytes);
}

void release(void *memory) { std::free(memory); }

void copy(void *destination, const void *source, size_t bytes, llaisysMemcpyKind_t) {
    if (bytes != 0) {
        std::memmove(destination, source, bytes);
    }
}

void copyAsync(
    void *destination,
    const void *source,
    size_t bytes,
    llaisysMemcpyKind_t kind,
    llaisysStream_t) {
    copy(destination, source, bytes, kind);
}

const LlaisysRuntimeAPI CPU_API{
    &deviceCount,
    &selectDevice,
    &synchronizeDevice,
    &makeStream,
    &releaseStream,
    &synchronizeStream,
    &allocate,
    &release,
    &allocate,
    &release,
    &copy,
    &copyAsync,
};

} // namespace

const LlaisysRuntimeAPI *getRuntimeAPI() { return &CPU_API; }

} // namespace llaisys::device::cpu
