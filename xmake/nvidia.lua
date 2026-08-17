target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_rules("cuda")
    add_files("../src/device/nvidia/nvidia_runtime_api.cu")
    add_links("cudart")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_languages("cxx17")
    add_rules("cuda")
    add_cuflags("-Xcompiler=-fPIC")
    add_links("cudart")
    add_files("../src/ops/*/nvidia/*.cu")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
    on_install(function (target) end)
target_end()
