local corex_home = os.getenv("COREX_HOME") or "/usr/local/corex"
local corex_bin = path.join(corex_home, "bin")
local corex_include = path.join(corex_home, "include")
local corex_lib = path.join(corex_home, "lib64")

toolchain("corex")
    set_kind("standalone")
    set_toolset("cc", path.join(corex_bin, "clang"))
    set_toolset("cxx", path.join(corex_bin, "clang++"))
    set_toolset("ld", path.join(corex_bin, "clang++"))
    set_toolset("sh", path.join(corex_bin, "clang++"))
toolchain_end()

rule("corex.ivcore")
    set_extensions(".ivcore")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        import("utils.progress")

        local objectfile = target:objectfile(sourcefile)
        local dependfile = target:dependfile(objectfile)
        depend.on_changed(function ()
            progress.show(opt.progress, "compiling.ivcore %s", sourcefile)
            os.mkdir(path.directory(objectfile))
            os.vrunv(path.join(corex_bin, "clang++"), {
                "-c", "-x", "ivcore", "--cuda-path=" .. corex_home,
                "--cuda-gpu-arch=ivcore11", "-std=c++17", "-O3", "-fPIC",
                "-DENABLE_ILUVATAR_API", "-I" .. path.absolute("include"),
                "-I" .. corex_include, "-o", objectfile, sourcefile
            })
        end, {dependfile = dependfile, files = sourcefile})
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("llaisys-device-iluvatar")
    set_kind("static")
    set_languages("cxx17")
    set_toolchains("corex")
    add_files("../src/device/iluvatar/iluvatar_runtime_api.cpp")
    add_includedirs(corex_include)
    add_cxflags("-fPIC")
    on_install(function (target) end)
target_end()

target("llaisys-icore-ops")
    set_kind("static")
    set_languages("cxx17")
    set_toolchains("corex")
    add_files("../src/ops/icore_all.ivcore", {rule = "corex.ivcore"})
    add_includedirs(corex_include)
    add_ldflags("-L" .. corex_lib, "-lcudart")
    add_linkdirs(corex_lib)
    add_links("cudart")
    add_rpathdirs(corex_lib)
    on_install(function (target) end)
target_end()

target("llaisys")
    add_deps("llaisys-icore-ops")
    add_syslinks("llaisys-icore-ops")
target_end()
