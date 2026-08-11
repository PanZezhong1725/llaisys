target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    add_cugencodes("native", "sm_80")
    set_values("cuda.rdc", false)
    add_cuflags("-Xcompiler=-fPIC", {force = true})
    add_links("cublas")

    add_files("../src/device/nvidia/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_cugencodes("native", "sm_80")
    set_values("cuda.rdc", false)
    add_cuflags("-Xcompiler=-fPIC", {force = true})
    add_links("cublas")

    add_files("../src/ops/*/nvidia/*.cu")

    on_install(function (target) end)
target_end()
