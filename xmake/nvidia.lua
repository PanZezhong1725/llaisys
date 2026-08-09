target("llaisys-device-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", false)
    add_values("cuda.rdc", false)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("../src/device/nvidia/*.cu")
    add_cugencodes("native")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    set_policy("build.cuda.devlink", false)
    add_values("cuda.rdc", false)
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-nvidia")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("../src/ops/*/nvidia/*.cu")
    add_cugencodes("native")
    if not is_plat("windows") then
        add_cuflags("-Xcompiler=-fPIC")
    end
    on_install(function (target) end)
target_end()
