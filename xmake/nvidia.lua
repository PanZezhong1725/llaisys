-- xmake/nvidia.lua
-- NVIDIA CUDA / MetaX MACA support configuration

-- Detect if we're using MetaX MACA or standard CUDA
local is_metax = os.exists("/opt/maca-3.5.3/tools/cu-bridge/include/cuda_runtime.h")

if is_metax then
    print("Detected MetaX MACA at /opt/maca-3.5.3")
    
    -- Set CUDA SDK path for xmake
    add_includedirs("/opt/maca-3.5.3/tools/cu-bridge/include")
    add_linkdirs("/opt/maca-3.5.3/lib64")
end

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    
    -- Set CUDA as the language
    set_languages("c++17", "cuda")
    
    if is_metax then
        -- MetaX MACA configuration
        print("Configuring for MetaX MACA...")
        
        -- Add MACA include paths
        add_includedirs("/opt/maca-3.5.3/tools/cu-bridge/include")
        add_includedirs("/opt/maca-3.5.3/include")
        
        -- Add MACA library paths
        add_linkdirs("/opt/maca-3.5.3/lib64")
        
        -- Link MACA libraries (CUDA-compatible)
        add_links("mcruntime", "mcblas", "mcdnn")
        
        -- Set CUDA compiler to mxcc

        -- CUDA compilation flags for MetaX C500 (compute capability 7.5)
        add_cuflags("-offload-arch=xcore1000")
        add_cuflags("--expt-relaxed-constexpr")
        add_cuflags("-Xcompiler -fPIC")
        add_cuflags("-I/opt/maca-3.5.3/tools/cu-bridge/include")
    else
        -- Standard NVIDIA CUDA configuration
        print("Configuring for NVIDIA CUDA...")
        
        -- CUDA compilation flags
        add_cuflags("-gencode arch=compute_70,code=sm_70")
        add_cuflags("-gencode arch=compute_75,code=sm_75")
        add_cuflags("-gencode arch=compute_80,code=sm_80")
        add_cuflags("-gencode arch=compute_89,code=sm_89")
        add_cuflags("-gencode arch=compute_90,code=sm_90")
        add_cuflags("--expt-relaxed-constexpr")
        add_cuflags("-Xcompiler -fPIC")
        
        -- Link CUDA libraries
        add_links("cudart", "cublas", "cudnn")
    end
    
    -- Include directories
    add_includedirs("$(projectdir)/include")
    add_includedirs("$(projectdir)/src")
    
    -- Source files
    add_files("$(projectdir)/src/device/nvidia/*.cu")
    
    -- Set warnings
    set_warnings("all")
    
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")
    
    -- Set CUDA as the language
    set_languages("c++17", "cuda")
    
    if is_metax then
        -- MetaX MACA configuration
        add_includedirs("/opt/maca-3.5.3/tools/cu-bridge/include")
        add_includedirs("/opt/maca-3.5.3/include")
        add_linkdirs("/opt/maca-3.5.3/lib64")
        add_links("mcruntime", "mcblas", "mcdnn")
        add_cuflags("-offload-arch=xcore1000")
        add_cuflags("--expt-relaxed-constexpr")
        add_cuflags("-Xcompiler -fPIC")
        add_cuflags("-I/opt/maca-3.5.3/tools/cu-bridge/include")
    else
        -- Standard NVIDIA CUDA configuration
        add_cuflags("-gencode arch=compute_70,code=sm_70")
        add_cuflags("-gencode arch=compute_75,code=sm_75")
        add_cuflags("-gencode arch=compute_80,code=sm_80")
        add_cuflags("-gencode arch=compute_89,code=sm_89")
        add_cuflags("-gencode arch=compute_90,code=sm_90")
        add_cuflags("--expt-relaxed-constexpr")
        add_cuflags("-Xcompiler -fPIC")
        add_links("cudart", "cublas", "cudnn")
    end
    
    -- Include directories
    add_includedirs("$(projectdir)/include")
    add_includedirs("$(projectdir)/src")
    
    -- Source files - all nvidia operator implementations
    add_files("$(projectdir)/src/ops/*/nvidia/*.cu")
    
    -- Set warnings
    set_warnings("all")
    
    on_install(function (target) end)
target_end()
