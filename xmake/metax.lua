local maca_root = get_config("maca-root")
local maca_arch = get_config("maca-arch")
local maca_compiler = path.join(maca_root, "mxgpu_llvm", "bin", "mxcc")
local project_root = path.directory(os.scriptdir())
local include_dir = path.join(project_root, "include")
local source_dir = path.join(project_root, "src")

rule("metax.compile")
    set_extensions(".maca")

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        table.insert(target:objectfiles(), objectfile)

        batchcmds:show_progress(
            opt.progress,
            "${color.build.object}compiling.metax %s",
            sourcefile
        )
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:vrunv(maca_compiler, {
            "-x", "maca",
            "-offload-arch", maca_arch,
            "--maca-path=" .. maca_root,
            "-std=c++17",
            "-fPIC",
            "-I" .. include_dir,
            "-I" .. source_dir,
            "-c", sourcefile,
            "-o", objectfile
        })

        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()

local function configure_metax_target()
    set_languages("cxx17")
    add_includedirs(path.join(maca_root, "include"), {public = true})
    add_linkdirs(
        path.join(maca_root, "lib"),
        path.join(maca_root, "lib64"),
        {public = true}
    )
    add_rpathdirs(
        path.join(maca_root, "lib"),
        path.join(maca_root, "lib64"),
        {public = true}
    )
    add_links("mcruntime", {public = true})

    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
end

target("llaisys-device-metax")
    set_kind("static")
    configure_metax_target()
    add_files("../src/device/metax/*.maca", {rules = "metax.compile"})

    on_install(function (target) end)
target_end()

target("llaisys-ops-metax")
    set_kind("static")
    add_deps("llaisys-tensor")
    configure_metax_target()
    add_files("../src/ops/*/metax/*.maca", {rules = "metax.compile"})

    on_install(function (target) end)
target_end()
