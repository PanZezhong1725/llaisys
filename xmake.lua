 add_rules("mode.debug", "mode.release")
  set_encodings("utf-8")

  add_includedirs("include")

  -- CPU --
  includes("xmake/cpu.lua")

  -- NVIDIA/MetaX --
  option("nv-gpu")
      set_default(false)
      set_showmenu(true)
      set_description("Whether to compile implementations for Nvidia GPU or MetaX GPU")
  option_end()

  -- MetaX C500 配置
  option("metax-gpu")
      set_default(false)
      set_showmenu(true)
      set_description("Whether to compile implementations for MetaX GPU (曦云 C500)")
  option_end()

  -- 检测 MetaX SDK
  local metax_sdk_path = "/opt/maca-3.5.3"
  local has_metax = os.exists(path.join(metax_sdk_path, "mxgpu_llvm/bin/mxcc"))

  if has_config("nv-gpu") or has_config("metax-gpu") then
      add_defines("ENABLE_NVIDIA_API")
      includes("xmake/nvidia.lua")

      -- 如果是 MetaX GPU，设置编译器
      if has_config("metax-gpu") and has_metax then
          set_config("cu", path.join(metax_sdk_path, "mxgpu_llvm/bin/mxcc"))
          set_config("cu-ld", path.join(metax_sdk_path, "mxgpu_llvm/bin/mxcc"))
          set_config("cuda", path.join(metax_sdk_path, "mxgpu_llvm"))

          -- 添加 MetaX 包含目录和库目录
          add_includedirs(path.join(metax_sdk_path, "include"))
          add_linkdirs(path.join(metax_sdk_path, "lib64"))

          print("Using MetaX C500 GPU with mxcc compiler")
      end
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
      add_deps("llaisys-utils")
      add_deps("llaisys-device-cpu")

      if has_config("nv-gpu") or has_config("metax-gpu") then
          add_deps("llaisys-device-nvidia")
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
      add_deps("llaisys-ops-cpu")

      if has_config("nv-gpu") or has_config("metax-gpu") then
          add_deps("llaisys-ops-nvidia")
      end

      set_languages("cxx17")
      set_warnings("all", "error")
      if not is_plat("windows") then
          add_cxflags("-fPIC", "-Wno-unknown-pragmas")
      end
      add_files("src/ops/*/*.cpp")
      on_install(function (target) end)
  target_end()

  target("llaisys-models")
      set_kind("static")
      add_deps("llaisys-tensor")
      add_deps("llaisys-ops")
      set_languages("cxx17")
      set_warnings("all", "error")
      if not is_plat("windows") then
          add_cxflags("-fPIC", "-Wno-unknown-pragmas")
      end
      add_files("src/models/*.cpp")
      on_install(function (target) end)
  target_end()

  target("llaisys")
      set_kind("shared")
      add_deps("llaisys-utils")
      add_deps("llaisys-device")
      add_deps("llaisys-core")
      add_deps("llaisys-tensor")
      add_deps("llaisys-ops")
      add_deps("llaisys-models")

      if has_config("nv-gpu") or has_config("metax-gpu") then
          add_deps("llaisys-device-nvidia")
          add_deps("llaisys-ops-nvidia")
          add_files("src/llaisys/cuda_link.cu")

          -- Link CUDA/MACA libraries
          if has_config("metax-gpu") and has_metax then
              -- MetaX C500 特定链接
              add_linkdirs(path.join(metax_sdk_path, "lib64"))
              add_links("mcruntime", "mcblas", "mcdnn")
              set_toolset("sh", path.join(os.projectdir(), "scripts/mxcc-ld.sh"))

              -- 添加 MetaX 特定的编译标志
              add_cuflags("-std=c++17", {force = true})
              add_cuflags("-O3", {force = true})
              add_cuflags("-fPIC", {force = true})

              print("Linking MetaX libraries: mcruntime, mcblas, mcdnn")
          else
              -- NVIDIA GPU 链接
              add_links("cudart", "cublas", "cudnn")
          end
      end

      if has_config("nv-gpu") or has_config("metax-gpu") then
          set_languages("cxx17", "cuda")
      else
          set_languages("cxx17")
      end
      if has_config("metax-gpu") then
          set_warnings("all")
      else
          set_warnings("all", "error")
      end
      add_files("src/llaisys/*.cc")
      set_installdir(".")

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
