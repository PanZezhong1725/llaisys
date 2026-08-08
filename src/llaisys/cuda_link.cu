// Force nvcc to drive the final link so device code in static CUDA
// libraries is registered when libllaisys.so is loaded.
__global__ void llaisys_cuda_link_dummy_kernel() {}

extern "C" void llaisys_cuda_link_dummy() {
    llaisys_cuda_link_dummy_kernel<<<1, 1>>>();
}
