// CoreX compiles CUDA-compatible translation units without a separate device
// runtime archive. This non-empty object lets build systems satisfy that
// archive dependency when relocatable device code is disabled.
extern "C" int llaisys_corex_device_runtime_anchor() { return 0; }
