target("llaisys-device-suda")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    else
        -- Disable C4819 (file contains characters not representable in current code page)
        -- which is triggered by CUDA headers on non-UTF8 code pages.
        add_cxflags("-wd4819")
    end

    add_files("../src/device/suda/*.cu")
    -- SUDA op kernels (one .cu per op under src/ops/*/suda/)
    add_files("../src/ops/*/suda/*.cu")

    -- Disable Relocatable Device Code (RDC). xmake enables -rdc=true by default
    -- for CUDA (see rules/cuda/env/xmake.lua), which generates a reference to
    -- __cudaRegisterLinkedBinary that requires linking with the RDC-enabled CUDA
    -- runtime. Since each .cu file is self-contained (kernels are defined and
    -- launched in the same file), we do not need RDC.
    set_values("cuda.rdc", false)
    add_cuflags("-std=c++17")

    if not is_plat("windows") then
        -- Forward -fPIC to the host compiler (gcc) so that the CUDA objects can
        -- be linked into the shared library libllaisys.so. Without it, thread-local
        -- storage in suda_resource.cu triggers a relocation error at link time.
        add_cuflags("-Xcompiler=-fPIC")
    else
        -- Forward flags to the host compiler (cl.exe):
        --  - /wd4819: disable C4819 (characters not representable in current code page)
        --  - /MD: use dynamic runtime library to match the rest of the project
        add_cuflags("-Xcompiler=/wd4819", "-Xcompiler=/MD")
    end

    -- Iluvatar CoreX SDK exposes a CUDA-compatible toolchain (cuda_runtime.h,
    -- cublas_v2.h, nvcc, libcudart/libcublas). Use SUDA_PATH to point at the SDK
    -- root; fall back to CUDA_PATH if SUDA_PATH is unset (CoreX machines usually
    -- only define CUDA_PATH).
    local suda_root = os.getenv("SUDA_PATH") or os.getenv("CUDA_PATH")
    if suda_root then
        add_includedirs(suda_root .. "/include")
        if is_plat("windows") then
            add_linkdirs(suda_root .. "/lib/x64")
        else
            add_linkdirs(suda_root .. "/lib64")
        end
    end

    -- Link against the CUDA runtime and BLAS libraries
    add_links("cudart", "cublas", "cublasLt")

    -- NOTE: do NOT use add_cugencodes("native") here. The CoreX SDK is a
    -- CUDA-compatible layer whose nvcc wrapper compiles directly to the
    -- ivcore backend and drops all -gencode/-arch flags. Requesting "native"
    -- gencode forces xmake to run its CUDA device-detection helper, which the
    -- CoreX nvcc wrapper cannot execute (-run is unsupported) and crashes the
    -- build. The ivcore backend requires no explicit gencode.

    on_install(function (target) end)
target_end()
