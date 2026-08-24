target("llaisys-device-nvidia")
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

    add_files("../src/device/nvidia/*.cu")
    -- NVIDIA op kernels (one .cu per op under src/ops/*/nvidia/)
    add_files("../src/ops/*/nvidia/*.cu")

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
        -- storage in nvidia_resource.cu triggers a relocation error at link time.
        add_cuflags("-Xcompiler=-fPIC")
    else
        -- Forward flags to the host compiler (cl.exe):
        --  - /wd4819: disable C4819 (characters not representable in current code page)
        --  - /MD: use dynamic runtime library to match the rest of the project
        add_cuflags("-Xcompiler=/wd4819", "-Xcompiler=/MD")
    end

    -- CUDA include and library search paths
    add_includedirs("$(env CUDA_PATH)/include")
    if is_plat("windows") then
        add_linkdirs("$(env CUDA_PATH)/lib/x64")
    else
        add_linkdirs("$(env CUDA_PATH)/lib64")
    end

    -- Link against the CUDA runtime and BLAS libraries
    add_links("cudart", "cublas", "cublasLt")

    -- Generate code for the native GPU architecture
    add_cugencodes("native")

    on_install(function (target) end)
target_end()
