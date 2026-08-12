local corex_path = os.getenv("COREX_PATH") or os.getenv("CUDA_PATH") or "/usr/local/corex"

target("llaisys-device-iluvatar")
    set_kind("static")
    add_rules("cuda")
    set_languages("cxx17")
    add_defines("ENABLE_ILUVATAR_API")
    set_toolset("cu", corex_path .. "/bin/nvcc")
    add_files("../src/device/iluvatar/*.cu")
    add_includedirs("../include", corex_path .. "/include")
    add_linkdirs(corex_path .. "/lib64")
    add_syslinks("cudart")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end
    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_rules("cuda")
    set_languages("cxx17")
    add_defines("ENABLE_ILUVATAR_API")
    set_toolset("cu", corex_path .. "/bin/nvcc")
    add_files("../src/ops/nvidia/*.cu")
    add_includedirs("../include", corex_path .. "/include")
    add_linkdirs(corex_path .. "/lib64")
    add_syslinks("cudart")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler", "-fPIC", {force = true})
    end
    on_install(function (target) end)
target_end()
