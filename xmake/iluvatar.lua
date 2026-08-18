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
    add_cuflags("-std=c++17", {force = true})
    add_cuflags("-fPIC", {force = true})
    add_links("cublas")
    -- Iluvatar's SDK only ships libcudart.so, not libcudart_static.a. Linking
    -- "cudart" explicitly here satisfies xmake's built-in cuda rule (which
    -- auto-adds "cudart_static" unless a cudart link is already present) so
    -- it picks up the real shared library instead of a static one that
    -- doesn't exist on this platform.
    add_links("cudart")

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
    add_cuflags("-std=c++17", {force = true})
    add_cuflags("-fPIC", {force = true})
    add_links("cublas")
    add_links("cudart")

    add_files("../src/ops/*/iluvatar/*.cu")

    on_install(function (target) end)
target_end()
