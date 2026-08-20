add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate", { outputdir = "./.vscode" })
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- GPU backends --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

option("iluvatar-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Iluvatar CoreX GPU")
option_end()

option("corex-root")
    set_default(os.getenv("COREX_ROOT") or os.getenv("COREX_HOME") or "/usr/local/corex")
    set_showmenu(true)
    set_description("Iluvatar CoreX SDK root")
option_end()

option("corex-arch")
    set_default("ivcore10")
    set_showmenu(true)
    set_description("Iluvatar CoreX GPU architecture")
option_end()

option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for MetaX MXMACA GPU")
option_end()

option("maca-root")
    set_default(os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or "/opt/maca")
    set_showmenu(true)
    set_description("MetaX MXMACA SDK root")
option_end()

option("maca-arch")
    set_default("native")
    set_showmenu(true)
    set_description("MetaX GPU architecture passed to mxcc")
option_end()

if (has_config("nv-gpu") and has_config("iluvatar-gpu"))
    or (has_config("nv-gpu") and has_config("metax-gpu"))
    or (has_config("iluvatar-gpu") and has_config("metax-gpu")) then
    raise("only one GPU backend can be enabled in the same build")
end

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

if has_config("iluvatar-gpu") then
    add_defines("ENABLE_ILUVATAR_API")
    includes("xmake/iluvatar.lua")
end

if has_config("metax-gpu") then
    add_defines("ENABLE_METAX_API")
    includes("xmake/metax.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    -- Runtime implementations live in backend-specific static targets. Merge
    -- the selected backend into this archive so downstream shared-library
    -- links cannot lose it because of transitive archive ordering.
    set_policy("build.merge_archive", true)
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")

    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    if has_config("iluvatar-gpu") then
        add_deps("llaisys-device-iluvatar")
    end

    if has_config("metax-gpu") then
        add_deps("llaisys-device-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    -- Keep every selected backend implementation in the public ops archive.
    -- This is especially important for custom-compiled .maca object files.
    set_policy("build.merge_archive", true)
    add_deps("llaisys-ops-cpu")

    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    if has_config("iluvatar-gpu") then
        add_deps("llaisys-ops-iluvatar")
    end

    if has_config("metax-gpu") then
        add_deps("llaisys-ops-metax")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    -- 编译 ops 的框架内部的统一算子接口，不同设备的 kernel 由 device.xmake.lua 编译
    -- 再添加依赖即可 add_deps("llaisys-ops-device")
    add_files("src/ops/*/*.cpp")
    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    if has_config("metax-gpu") then
        -- MXMACA objects contain host stubs and embedded device images.  The
        -- final link must be driven by mxcc so its runtime registration
        -- support (__mcRegisterFatBinary, launch configuration, etc.) is
        -- injected into the shared library.
        local maca_root = get_config("maca-root")
        add_shflags(
            "--maca-path=" .. maca_root,
            "-offload-arch",
            get_config("maca-arch"),
            {force = true}
        )

        on_link(function (target)
            import("core.project.config")
            import("core.tool.linker")

            -- Reuse Xmake's dependency ordering and library argument
            -- generation, but run the resulting command through mxcc.  Its
            -- Clang-compatible adapter adds two host flags that mxcc rejects,
            -- so remove exactly those two arguments before invoking it.
            local _, argv = linker.linkargv(
                "shared",
                "cxx",
                target:objectfiles(),
                target:targetfile(),
                {target = target}
            )
            local mxcc_argv = {}
            for _, argument in ipairs(argv) do
                if argument ~= "-m64" and argument ~= "-s" then
                    table.insert(mxcc_argv, argument)
                end
            end

            local configured_root = config.get("maca-root")
            local mxcc = path.join(
                configured_root,
                "mxgpu_llvm",
                "bin",
                "mxcc"
            )
            assert(os.isfile(mxcc), "MetaX compiler was not found: " .. mxcc)
            os.mkdir(path.directory(target:targetfile()))
            os.vrunv(mxcc, mxcc_argv)
        end)
    end
    if not is_plat("windows") then
        add_shflags("-Wl,--no-undefined", {force = true})
    end
    add_files("src/llaisys/*.cc")
    add_files("src/llaisys/models/*.cc")
    set_installdir(".")

    if is_mode("debug") then
        set_symbols("debug")
        set_optimize("none")
    end
    
    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
