local corex_root = "/usr/local/corex"
local corex_compiler = path.join(corex_root, "bin", "clang++")

local function configure_corex_target()
    set_kind("static")
    set_toolset("cxx", corex_compiler)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_includedirs(path.join(corex_root, "include"))
    add_cxflags(
        "-x", "ivcore",
        "--cuda-path=" .. corex_root,
        "--offload-arch=native",
        "-fPIC",
        {force = true}
    )
    on_install(function (target) end)
end

target("llaisys-device-corex")
    configure_corex_target()
    add_files("../src/device/corex/*.cpp")
target_end()

target("llaisys-ops-corex")
    configure_corex_target()
    add_deps("llaisys-tensor")
    add_deps("llaisys-device-corex")
    add_files("../src/ops/*/corex/*.cpp")
target_end()
