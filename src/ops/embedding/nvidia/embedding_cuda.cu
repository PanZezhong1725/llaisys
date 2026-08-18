#include "embedding_cuda.cuh"
#include <cuda_runtime.h>
#include "../../../utils.hpp"
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { cudaError_t error = (call); if (error != cudaSuccess) { std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); std::abort(); } } while (0)


__global__ void embedding_kernel(unsigned char *out, const int64_t *index, const unsigned char *weight,
               size_t num_indices, size_t rowbytes){
               size_t offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
               size_t total_bytes = num_indices * rowbytes;
               if (offset < total_bytes){
                    size_t row = offset / rowbytes;
                    size_t col = offset % rowbytes;
                    size_t weight_off = index[row] * rowbytes + col;
                    out[offset] =weight[weight_off];   
               }
               }

namespace llaisys::ops::cuda {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embd_dim){
                    size_t esz = llaisys::utils::dsize(type);
                    size_t row_bytes = embd_dim * esz;
                    size_t total_bytes = num_indices * row_bytes; 

                    constexpr int block_size = 256;
                    int grid_size = (total_bytes + block_size -1)/block_size;

                    embedding_kernel<<<grid_size,block_size>>>(
                                   reinterpret_cast<unsigned char *>(out),
                                   reinterpret_cast<const int64_t *>(index),
                                   reinterpret_cast<const unsigned char *>(weight),
                                   num_indices,
                                   row_bytes
                              );
                    CHECK_CUDA(cudaGetLastError());
               }
          }