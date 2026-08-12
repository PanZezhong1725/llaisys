local cuda_compiler = os.getenv("LLAISYS_CUDA_COMPILER")

target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    set_languages("cxx17")
    add_defines("ENABLE_NVIDIA_API")
    add_files("../src/device/nvidia/*.cu")
    add_includedirs("../include")
    if cuda_compiler then
        set_toolset("cu", cuda_compiler)
        add_cuflags("--cuda-path=$(env CUDA_PATH)", "--cuda-gpu-arch=sm_86", "--no-offload-new-driver", "-std=c++17", "-fms-runtime-lib=dll", "-fno-gpu-rdc", {force = true})
    else
        add_cugencodes("native")
        add_cuflags("-allow-unsupported-compiler", {force = true})
    end
    add_linkdirs("$(env CUDA_PATH)/lib/x64")
    add_syslinks("cudart")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    set_values("cuda.rdc", false)
    set_policy("build.cuda.devlink", false)
    set_languages("cxx17")
    add_defines("ENABLE_NVIDIA_API")
    add_files("../src/ops/nvidia/*.cu")
    add_includedirs("../include")
    if cuda_compiler then
        set_toolset("cu", cuda_compiler)
        add_cuflags("--cuda-path=$(env CUDA_PATH)", "--cuda-gpu-arch=sm_86", "--no-offload-new-driver", "-std=c++17", "-fms-runtime-lib=dll", "-fno-gpu-rdc", {force = true})
    else
        add_cugencodes("native")
        add_cuflags("-allow-unsupported-compiler", {force = true})
    end
    add_syslinks("cudart")
    on_install(function (target) end)
target_end()
