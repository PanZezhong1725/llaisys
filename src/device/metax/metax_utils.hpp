#pragma once

#include <mc_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::metax {

inline void checkRuntimeError(
    mcError_t error,
    const char *expr,
    const char *file,
    int line
) {
    if (error == mcSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("[MXMACA] ")
        + mcGetErrorString(error)
        + "\n  expression: "
        + expr
        + "\n  location: "
        + file
        + ":"
        + std::to_string(line)
    );
}

} // namespace llaisys::device::metax

#define METAX_CHECK(expr)                                      \
    do {                                                       \
        ::llaisys::device::metax::checkRuntimeError(           \
            (expr), #expr, __FILE__, __LINE__                  \
        );                                                     \
    } while (0)

#define METAX_KERNEL_CHECK() METAX_CHECK(mcGetLastError())
