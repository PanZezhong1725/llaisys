#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <algorithm> // std::min
#include <thread>    // std::thread（按输出列并行）
#include <vector>    // std::vector（float 解码缓冲区）

namespace llaisys::ops {

namespace {

// 把 count 个元素从 src（dtype 指定的类型）解码成 float 写入 dst。
void decode_to_f32(float *dst, const std::byte *src, llaisysDataType_t dtype, size_t count) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32: {
        const float *p = reinterpret_cast<const float *>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = p[i];
        }
        return;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *p = reinterpret_cast<const bf16_t *>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = utils::cast<float>(p[i]);
        }
        return;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *p = reinterpret_cast<const fp16_t *>(src);
        for (size_t i = 0; i < count; ++i) {
            dst[i] = utils::cast<float>(p[i]);
        }
        return;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

// 长度为 k 的点积。
// 用 4 个独立累加器：浮点加法不满足结合律，编译器在没有 /fp:fast 时不能自行拆分归约链，
// 手动拆成 4 条独立依赖链可让 CPU 流水线并行执行乘加，明显快于单累加器版本。
float dot_f32(const float *a, const float *b, size_t k) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
    size_t p = 0;
    for (; p + 4 <= k; p += 4) {
        s0 += a[p + 0] * b[p + 0];
        s1 += a[p + 1] * b[p + 1];
        s2 += a[p + 2] * b[p + 2];
        s3 += a[p + 3] * b[p + 3];
    }
    float sum = (s0 + s1) + (s2 + s3);
    for (; p < k; ++p) {
        sum += a[p] * b[p];
    }
    return sum;
}

} // namespace

// 计算 Y = X * W^T + b
//   out    : [m, n]，输出 Y，2D 连续张量
//   in     : [m, k]，输入 X，2D 连续张量
//   weight : [n, k]，权重 W，2D 连续张量（注意没有转置，需要在计算中处理）
//   bias   : [n]，偏置 b，1D 张量，可选（不提供时为空指针）
// 全部实现都写在本函数内：m、n、k 等维度都直接从张量对象上读取，不通过参数传入。
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // ---- 参数检查 ----
    // 输出、输入、权重必须在同一设备上。
    CHECK_SAME_DEVICE(out, in, weight);
    // 三者数据类型必须一致。
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 维度约束：三者都是二维。
    ASSERT(out->ndim() == 2, "Linear: out must be a 2D tensor.");
    ASSERT(in->ndim() == 2, "Linear: in must be a 2D tensor.");
    ASSERT(weight->ndim() == 2, "Linear: weight must be a 2D tensor.");
    // 暂时只支持连续张量。
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: all tensors must be contiguous.");

    // 从张量形状取出三个维度：m 为行数，k 为输入特征数，n 为输出特征数（weight 的行数）。
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    // 权重形状必须是 [out_features, in_features]，输出形状必须是 [m, n]。
    ASSERT(weight->shape()[1] == k, "Linear: weight shape must be [out_features, in_features].");
    ASSERT(out->shape()[0] == m && out->shape()[1] == n,
           "Linear: out shape must be [in.shape[0], weight.shape[0]].");

    // bias 是可选的：tensor_t 是 shared_ptr，空指针即表示“不提供偏置”。
    // 必须先判空再访问，否则解引用空 shared_ptr 属于未定义行为。
    const bool has_bias = static_cast<bool>(bias);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);                  // 偏置需与输出同设备
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype()); // 偏置需与输出同类型
        ASSERT(bias->ndim() == 1, "Linear: bias must be a 1D tensor.");
        ASSERT(bias->shape()[0] == n, "Linear: bias length must match out_features.");
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    // ---- 设备分派：CPU 始终可用，其余设备暂不支持 ----
    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        // 先把线程上下文切换到目标设备。
        llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
        if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            // CUDA 版本留待后续作业实现。
            TO_BE_IMPLEMENTED();
            return;
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // ---- 以下是 CPU 实现 ----
    // 第 1 步：把输入与偏置解码成 float。这两块都很小（m*k 与 n），一次性解码即可。
    //
    // 权重故意不在这里整体解码。weight 是全模型最大的张量（lm_head 为 151936x1536），
    // 整体解码需要 n*k 个 float，约 933MB，每次调用都分配一次，推理时完全不可用。
    // 改为在下面按行解码：每行只需 k 个 float 的临时缓冲区，且该行会被 m 行输入复用。
    // f16/bf16 的解码函数现在内联在 types.hpp 中，所以按元素调用不再有跨模块调用开销。
    std::vector<float> in_f(m * k);
    decode_to_f32(in_f.data(), in->data(), in->dtype(), m * k);

    std::vector<float> bias_f;
    if (has_bias) {
        bias_f.resize(n);
        decode_to_f32(bias_f.data(), bias->data(), bias->dtype(), n);
    }

    // 结果先存放在 float 缓冲区中，最后统一转换回原始数据类型。
    std::vector<float> out_f(m * n);

    // 第 2 步：计算 out[i][j] = sum_p in[i][p] * weight[j][p] + bias[j]。
    //
    // 数学式 Y = X * W^T 中的转置，靠“按行读取 weight”隐式完成：
    // W^T 的第 j 列就是 W 的第 j 行，所以 out[i][j] 等于 in 第 i 行与 weight 第 j 行的点积。
    // 这样两个操作数在内存上都是连续的，缓存友好，也不需要真的搬运数据做转置。
    const std::byte *w_base = weight->data();
    const size_t w_elem_size = weight->elementSize();
    const llaisysDataType_t w_dtype = weight->dtype();

    // 每个线程负责一段连续的输出列 [j_begin, j_end)。不同 j 写入的是 out_f 中不同的位置，
    // 线程之间没有重叠，也没有共享的可变状态；每个输出元素的累加顺序与串行版本完全一致，
    // 因此多线程不改变计算结果。
    auto compute_cols = [&](size_t j_begin, size_t j_end) {
        std::vector<float> w_row(k); // 当前 weight 行的解码缓冲区，线程私有
        for (size_t j = j_begin; j < j_end; ++j) {
            decode_to_f32(w_row.data(), w_base + j * k * w_elem_size, w_dtype, k);
            for (size_t i = 0; i < m; ++i) {
                float sum = dot_f32(in_f.data() + i * k, w_row.data(), k);
                if (has_bias) {      // 不提供偏置时跳过累加
                    sum += bias_f[j]; // 偏置按输出列索引取，广播到所有行
                }
                out_f[i * n + j] = sum;
            }
        }
    };

    // 线程数按可用核数与工作量决定；小规模时直接串行，避免线程开销反而变慢。
    size_t nthread = 1;
    if (m * n * k >= (size_t(1) << 20)) {
        const unsigned hw = std::thread::hardware_concurrency();
        nthread = std::min<size_t>(hw == 0 ? 1 : hw, n);
    }

    if (nthread <= 1) {
        compute_cols(0, n);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nthread - 1);
        const size_t chunk = (n + nthread - 1) / nthread;
        for (size_t t = 1; t < nthread; ++t) {
            const size_t j_begin = std::min(t * chunk, n);
            const size_t j_end = std::min(j_begin + chunk, n);
            if (j_begin < j_end) {
                pool.emplace_back(compute_cols, j_begin, j_end);
            }
        }
        compute_cols(0, std::min(chunk, n)); // 主线程也承担一份
        for (auto &th : pool) {
            th.join();
        }
    }

    // 第 3 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        float *op = reinterpret_cast<float *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = out_f[i]; // 目标就是 float，直接写入
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t *op = reinterpret_cast<bf16_t *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = utils::cast<bf16_t>(out_f[i]); // float -> bf16
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t *op = reinterpret_cast<fp16_t *>(out->data());
        for (size_t i = 0; i < m * n; ++i) {
            op[i] = utils::cast<fp16_t>(out_f[i]); // float -> fp16
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
