target("llaisys-device-nvidia")
    set_kind("static")

    set_languages("cxx17")

    add_files("../src/device/nvidia/*.cu")

    add_links("cublas", {public = true})

    --- 动探测当前编译主机的 NVIDIA GPU 显卡硬件，并自动匹配并添加对应的 CUDA gencode 编译参数 ---
    add_cugencodes("native")

    --- CUDA 文件位于 static target，而最终 binary/shared target 本身没有 CUDA 文件时，device-link 可能缺失，需要显式为 static target 开启 device-link ---
    add_values("cuda.build.devlink", true)

    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cuflags("-Xcompiler=-fPIC")
        -- 关键：让 device-link 生成的 host object 也是 PIC
        add_culdflags("-Xcompiler=-fPIC")
    end

    on_install(function (target) end)
target_end()


target("llaisys-ops-nvidia")
    set_kind("static")

    add_deps("llaisys-tensor")

    set_languages("cxx17")

    add_files("../src/ops/*/nvidia/*.cu")

    add_cugencodes("native")

    if not is_plat("windows") then
        add_cxflags("-fPIC")
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
    end

    -- 最终 shared target 本身没有 .cu 时尤其需要注意 device-link
    add_values("cuda.build.devlink", true)

    on_install(function (target) end)
target_end()