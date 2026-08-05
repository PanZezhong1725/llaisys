toolchain("iluvatar")
    set_kind("standalone")

    -- 编译 .cu 文件
    set_toolset("cu", "/usr/local/corex/bin/clang++")

    -- 编译普通 C 文件
    set_toolset("cc", "/usr/bin/gcc")

    -- 编译普通 C++ 文件
    set_toolset("cxx", "/usr/bin/g++")

    -- 链接 C++ 目标
    set_toolset("ld", "/usr/bin/g++")
    set_toolset("sh", "/usr/bin/g++")

    on_check(function (toolchain)
        local find_tool = import("lib.detect.find_tool")

        return find_tool(
            "clang++",
            {paths = "/usr/local/corex/bin"}
        )
    end)
toolchain_end()

target("llaisys-device-iluvatar")
    set_kind("static")
    set_toolchains("iluvatar")
    set_languages("cxx17")
    set_values("cuda.rdc", false)
    add_cuflags("-x", "ivcore", {force = true})
    add_cuflags("-fPIC", {force = true})
    add_links("cublas")
    add_links("cudnn")

    add_files("../src/device/iluvatar/*.cu")
    add_linkdirs("/usr/local/corex-4.4.0/lib64")
    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    set_toolchains("iluvatar")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    add_linkdirs("/usr/local/corex-4.4.0/lib64")
    set_values("cuda.rdc", false)
    add_cuflags("-x", "ivcore", {force = true})
    add_cuflags("-fPIC", {force = true})
    add_links("cublas")
    add_links("cudnn")

    add_files("../src/ops/*/iluvatar/*.cu")

    on_install(function (target) end)
target_end()
