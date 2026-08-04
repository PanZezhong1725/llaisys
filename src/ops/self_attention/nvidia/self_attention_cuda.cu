#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../../utils.hpp"
#include "self_attention_cuda.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#if CUDNN_MAJOR >= 8
#include <cudnn_frontend.h>
#endif
// V1：手写的两遍 softmax kernel（GQA-aware，一个 block 处理一个 (query token i, head h)）。
// cuDNN 的 SDPA 节点要求 head 维度（d/dv）必须是 8 的倍数，所以这一版留着当 fallback：
// d 或 dv 不是 8 的倍数时用这个，其余情况走下面的 cudnn_frontend 版本。
// 一个 block 处理一个 (query token i, head h)；scores 用动态 shared memory 存一整行
// （total_len 个 float），大小由 launcher 按 total_len*sizeof(float) 申请。
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale) {
    size_t i = blockIdx.x; // query token, [0, seqlen)
    size_t h = blockIdx.y; // query head, [0, nhead)
    size_t tid = threadIdx.x;

    size_t group = nhead / nkvhead; // 每 group 个 query head 共享 1 个 kv head
    size_t kvh = h / group;         // 该 query head 对应的 kv head
    size_t causal_offset = total_len - seqlen;
    size_t limit = i + causal_offset; // 因果掩码: j 的范围是 [0, limit]（闭区间）

    extern __shared__ float scores[]; // 大小 total_len，只用到 [0, limit] 这一段
    __shared__ float sdata[256];
    // 第一遍打分：每个线程按 grid-stride 负责一部分 j（j = tid, tid+stride, ... <= limit）。
    float max_score = -INFINITY;
    // 遍历每一个key token
    for (size_t j = tid; j <= limit; j += blockDim.x) {
        float score = 0.0f;

        for (size_t dim = 0; dim < d; dim++) {
            score += float(q[i * nhead * d + h * d + dim]) * float(k[j * nkvhead * d + kvh * d + dim]);
        }
        score *= scale;
        scores[j] = score;
        max_score = std::fmaxf(max_score, score);
    }
    sdata[tid] = max_score;
    __syncthreads();

    // block 内做 max 规约得到 max_score。
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = max(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_score = sdata[0];

    __shared__ float sumexp[256];
    sumexp[tid] = 0.0;
    // 第二遍:用 exp(score - max_score) 做数值稳定的 softmax 加权求和。
    for (size_t j = tid; j <= limit; j += blockDim.x) {
        scores[j] = std::exp(scores[j] - max_score);
        sumexp[tid] += scores[j];
    }
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sumexp[tid] = sumexp[tid] + sumexp[tid + stride];
        }
        __syncthreads();
    }
    float sum_exp = sumexp[0];

    // 第三遍：按 dv 维度分工，加权求和 v 得到最终输出。
    for (size_t t = tid; t < dv; t += blockDim.x) {
        float acc = 0.0;
        for (size_t j = 0; j <= limit; j += 1) {
            acc += (float)(scores[j]) * (float)(v[j * nkvhead * dv + kvh * dv + t]);
        }
        attn_val[i * nhead * dv + h * dv + t] = acc / sum_exp;
    }
}

// Launcher：负责 grid、block 和 kernel 启动
template <typename T>
void launch_self_attention(T *attn_val, const T *q, const T *k, const T *v,
                           size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                           size_t d, size_t dv, float scale) {
    constexpr int block_size = 256;
    dim3 grid(static_cast<unsigned int>(seqlen), static_cast<unsigned int>(nhead));
    size_t shared_bytes = total_len * sizeof(float);

    self_attention_kernel<<<grid, block_size, shared_bytes>>>(
        attn_val, q, k, v, seqlen, total_len, nhead, nkvhead, d, dv, scale);
}

// V2：hd 是 8 的倍数时走这条路——cudnn_frontend（v1.26.0）的 Graph API + 现成的 SDPA
// 节点（原生支持 GQA，不用自己拆 kv head）。参考实现见 cudnn-frontend 仓库的
// samples/cpp/sdpa/fp16_fwd.cpp。
//
// TODO：现在每次调用都重新建一次 graph、重新 build 一次执行计划——先跑通正确性，
// 回头如果发现这部分开销明显，再考虑要不要把 Graph（按 shape 分组）缓存到 Resource 上。

namespace llaisys::ops::cuda {
// q        : [seqlen,    nhead,   d ]
// k        : [total_len, nkvhead, d ]
// qk^T     : [seqlen,    nhead,   total_len]
// softmax(scale * qk^T) : [seqlen, nhead, total_len]
// v        : [total_len, nkvhead, dv]
// attn_val : [seqlen,    nhead,   dv]
// attn_val = softmax(scale * q * k^T) * v
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type,
                    size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                    float scale, llaisys::device::DeviceResource *resource) {
    // cuDNN 的 SDPA 节点要求 head 维度是 8 的倍数，不满足就走 V1 手写 kernel。
    if (d % 8 != 0 || dv % 8 != 0) {
        switch (type) {
        case LLAISYS_DTYPE_F32:
            launch_self_attention(
                reinterpret_cast<float *>(attn_val),
                reinterpret_cast<const float *>(q),
                reinterpret_cast<const float *>(k),
                reinterpret_cast<const float *>(v),
                seqlen, total_len, nhead, nkvhead, d, dv, scale);
            return;
        case LLAISYS_DTYPE_BF16:
            launch_self_attention(
                reinterpret_cast<__nv_bfloat16 *>(attn_val),
                reinterpret_cast<const __nv_bfloat16 *>(q),
                reinterpret_cast<const __nv_bfloat16 *>(k),
                reinterpret_cast<const __nv_bfloat16 *>(v),
                seqlen, total_len, nhead, nkvhead, d, dv, scale);
            return;
        case LLAISYS_DTYPE_F16:
            launch_self_attention(
                reinterpret_cast<__half *>(attn_val),
                reinterpret_cast<const __half *>(q),
                reinterpret_cast<const __half *>(k),
                reinterpret_cast<const __half *>(v),
                seqlen, total_len, nhead, nkvhead, d, dv, scale);
            return;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
        }
    }
#if CUDNN_MAJOR >= 8
    auto handle = static_cast<llaisys::device::nvidia::Resource *>(resource)->cudnnHandle();

    namespace fe = cudnn_frontend;
    using namespace fe::graph;

    fe::DataType_t io_dtype;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        io_dtype = fe::DataType_t::FLOAT;
        break;
    case LLAISYS_DTYPE_F16:
        io_dtype = fe::DataType_t::HALF;
        break;
    case LLAISYS_DTYPE_BF16:
        io_dtype = fe::DataType_t::BFLOAT16;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // 建 graph，设好 io/intermediate/compute 三个 dtype。
    Graph graph;

    graph.set_io_data_type(io_dtype)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    // 定义 Q/K/V 三个 tensor，dim/stride 对好 llaisys 实际的 [s, h, elem_d] 内存布局。
    auto Q = graph.tensor(
        Tensor_attributes()
            .set_name("Q")
            .set_dim({1, static_cast<int64_t>(nhead), static_cast<int64_t>(seqlen), static_cast<int64_t>(d)})
            .set_stride({static_cast<int64_t>(nhead * seqlen * d), static_cast<int64_t>(d), static_cast<int64_t>(nhead * d), 1})
            .set_uid(1));

    auto K = graph.tensor(
        Tensor_attributes()
            .set_name("K")
            .set_dim({1, static_cast<int64_t>(nkvhead), static_cast<int64_t>(total_len), static_cast<int64_t>(d)})
            .set_stride({static_cast<int64_t>(nkvhead * total_len * d), static_cast<int64_t>(d), static_cast<int64_t>(nkvhead * d), 1})
            .set_uid(2));

    auto V = graph.tensor(
        Tensor_attributes()
            .set_name("V")
            .set_dim({1, static_cast<int64_t>(nkvhead), static_cast<int64_t>(total_len), static_cast<int64_t>(dv)})
            .set_stride({static_cast<int64_t>(nkvhead * total_len * dv), static_cast<int64_t>(dv), static_cast<int64_t>(nkvhead * dv), 1})
            .set_uid(3));

    auto [O, Stats] = graph.sdpa(
        Q,
        K,
        V,
        SDPA_attributes()
            .set_name("attention")
            .set_is_inference(true)
            .set_causal_mask_bottom_right(true)
            .set_attn_scale(scale));

    // O 的 dim/stride 跟 Q 用同一套 {b, h, s, elem_d} 顺序（h=nhead, s=seqlen, elem_d=dv）。
    O->set_name("O")
        .set_dim({1, static_cast<int64_t>(nhead), static_cast<int64_t>(seqlen), static_cast<int64_t>(dv)})
        .set_stride({static_cast<int64_t>(nhead * seqlen * dv), static_cast<int64_t>(dv), static_cast<int64_t>(nhead * dv), 1})
        .set_uid(4)
        .set_output(true);

    auto validate_status = graph.validate();
    if (!validate_status.is_good()) {
        std::cerr << "[cudnn_frontend] validate failed: " << validate_status.get_message() << std::endl;
    }
    auto build_og_status = graph.build_operation_graph(handle);
    if (!build_og_status.is_good()) {
        std::cerr << "[cudnn_frontend] build_operation_graph failed: " << build_og_status.get_message() << std::endl;
    }
    auto plans_status = graph.create_execution_plans({fe::HeurMode_t::A});
    if (!plans_status.is_good()) {
        std::cerr << "[cudnn_frontend] create_execution_plans failed: " << plans_status.get_message() << std::endl;
    }
    auto build_plans_status = graph.build_plans(handle);
    if (!build_plans_status.is_good()) {
        std::cerr << "[cudnn_frontend] build_plans failed: " << build_plans_status.get_message() << std::endl;
    }

    size_t workspace_size = graph.get_workspace_size();
    void *workspace;
    std::unordered_map<int64_t, void *> variant_pack = {
        {1 /* Q 的 uid */, (void *)const_cast<std::byte *>(q)},
        {2 /* K 的 uid */, (void *)const_cast<std::byte *>(k)},
        {3 /* V 的 uid */, (void *)const_cast<std::byte *>(v)},
        {4 /* O 的 uid */, (void *)attn_val},
    };

    cudaMalloc(&workspace, workspace_size);
    auto exec_status = graph.execute(handle, variant_pack, workspace);
    if (!exec_status.is_good()) {
        std::cerr << "[cudnn_frontend] execute failed: " << exec_status.get_message() << std::endl;
    }
    cudaFree(workspace);
#else
    switch (type) {
    case LLAISYS_DTYPE_F32:
        launch_self_attention(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_BF16:
        launch_self_attention(
            reinterpret_cast<__nv_bfloat16 *>(attn_val),
            reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k),
            reinterpret_cast<const __nv_bfloat16 *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    case LLAISYS_DTYPE_F16:
        launch_self_attention(
            reinterpret_cast<__half *>(attn_val),
            reinterpret_cast<const __half *>(q),
            reinterpret_cast<const __half *>(k),
            reinterpret_cast<const __half *>(v),
            seqlen, total_len, nhead, nkvhead, d, dv, scale);
        return;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
#endif
}

} // namespace llaisys::ops::cuda
