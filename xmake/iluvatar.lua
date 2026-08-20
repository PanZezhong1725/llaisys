local corex_root = get_config("corex-root")
local corex_arch = get_config("corex-arch")
local corex_compiler = path.join(corex_root, "bin", "clang++")
local corex_lib = path.join(corex_root, "lib")
local corex_lib64 = path.join(corex_root, "lib64")

local function configure_corex_target()
    set_languages("cxx17")
    set_toolset("cu", corex_compiler)

    add_includedirs(path.join(corex_root, "include"), {public = true})
    add_linkdirs(corex_lib, corex_lib64, {public = true})
    add_rpathdirs(corex_lib, corex_lib64, {public = true})
    add_links("cublas", "cudart", "cuda", {public = true})

    add_cuflags(
        "--cuda-path=" .. corex_root,
        "--cuda-gpu-arch=" .. corex_arch,
        "-fPIC"
    )

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
end

target("llaisys-device-iluvatar")
    set_kind("static")
    configure_corex_target()
    add_files("../src/device/iluvatar/*.cu")

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    configure_corex_target()
    add_files("../src/ops/*/iluvatar/*.cu")

    on_install(function (target) end)
target_end()
