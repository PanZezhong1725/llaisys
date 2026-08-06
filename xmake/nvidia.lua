-- xmake/nvidia.lua
-- NVIDIA CUDA support configuration

target("llaisys-nvidia")
    set_kind("static")
    add_deps("llaisys-core")
    
    -- CUDA compilation flags
    add_cuflags("-gencode arch=compute_70,code=sm_70")
    add_cuflags("-gencode arch=compute_80,code=sm_80")
    add_cuflags("-gencode arch=compute_90,code=sm_90")
    
    -- Include directories
    add_includedirs("$(projectdir)/include")
    add_includedirs("$(projectdir)/src")
    
    -- Source files
    add_files("$(projectdir)/src/device/nvidia/*.cu")
    add_files("$(projectdir)/src/ops/*/nvidia/*.cu")
    
    -- Link libraries
    add_links("cudart", "cublas", "cudnn")
    
    -- Set CUDA as the language
    set_languages("c++17", "cuda")
target_end()
