target("llaisys-device-nvidia")
    set_kind("static")
    set_languages("cxx17")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cuflags("-Xcompiler=-fPIC")
    end
    add_rules("cuda")
    -- CUDA 12.0 cannot parse the RTX 5060's sm_120 reported by the driver.
    -- Use a CUDA-12-compatible code generation target until a newer toolkit
    -- is selected; runtime API compilation itself is architecture agnostic.
    add_cugencodes("compute_89")
    add_files("../src/device/nvidia/nvidia_runtime_api.cpp")
    add_includedirs("/usr/include")
    add_linkdirs("/usr/lib/x86_64-linux-gnu")
    add_links("cudart")
    on_install(function (target) end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("shared")
    set_languages("cxx17")
    add_rules("cuda")
    add_cugencodes("compute_89")
    add_cuflags("-Xcompiler=-fPIC")
    add_includedirs("/usr/include")
    add_linkdirs("/usr/lib/x86_64-linux-gnu")
    add_links("cudart")
    add_files("../src/ops/nvidia/nvidia_ops.cu")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
    on_install(function (target) end)
target_end()
