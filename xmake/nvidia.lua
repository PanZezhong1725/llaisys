local corex_root = "/usr/local/corex"
local corex_compiler = path.join(corex_root, "bin", "clang++")
local use_corex = os.isfile(corex_compiler)

if use_corex then
    target("llaisys-corex-runtime-placeholder")
        set_kind("static")
        set_basename("cudadevrt")
        set_languages("cxx17")
        add_files("../src/device/nvidia/corex_cudadevrt_stub.cpp")
        on_install(function (target) end)
    target_end()
end

local function configure_cuda_sources()
    set_languages("cxx17")
    if use_corex then
        set_toolset("cu", corex_compiler)
        add_cuflags(
            "-x", "ivcore",
            "-std=c++17",
            "-fPIC",
            "--cuda-path=" .. corex_root,
            "--offload-arch=native",
            {force = true})
        set_values("cuda.rdc", false)
        add_linkdirs(path.join(corex_root, "lib64"))
    else
        add_cugencodes("native")
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
        add_culdflags("-Xcompiler", "-fPIC", {force = true})
        add_values("cuda.build.devlink", true)
    end
end

target("llaisys-device-nvidia")
    set_kind("static")
    add_deps("llaisys-utils")
    if use_corex then
        add_deps("llaisys-corex-runtime-placeholder")
    end
    configure_cuda_sources()
    add_files("../src/device/nvidia/*.cu")
    add_links("cudart", {public = true})
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor", "llaisys-device-nvidia")
    if use_corex then
        add_deps("llaisys-corex-runtime-placeholder")
    end
    configure_cuda_sources()
    add_files("../src/ops/*/nvidia/*.cu")
    add_links("cudart", "cublas", {public = true})
    on_install(function (target) end)
target_end()
