#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>  // std::sqrt
#include <vector> // std::vector（float 解码缓冲区）

namespace llaisys::ops {

// 对每一行计算：Y[i] = W * X[i] / sqrt( (1/d) * sum_j X[i][j]^2 + eps )
// out、in 为 2D 连续张量，weight 为与最后一维等长的 1D 张量。
// 归一化沿输入张量的最后一个维度（即每一行，长度为 d）进行。
// 全部实现都写在本函数内：行数与列数直接从张量对象上读取，不通过参数传入。
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // ---- 参数检查 ----
    // 三个张量必须在同一设备上。
    CHECK_SAME_DEVICE(out, in, weight);
    // 输出与输入形状相同（不涉及广播）。
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    // 三者数据类型必须一致。
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 维度约束：输入输出二维，权重一维。
    ASSERT(in->ndim() == 2, "RMSNorm: in must be a 2D tensor.");
    ASSERT(out->ndim() == 2, "RMSNorm: out must be a 2D tensor.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be a 1D tensor.");
    // 权重长度必须等于归一化维度的长度。
    ASSERT(weight->shape()[0] == in->shape()[1],
           "RMSNorm: weight length must match the last dimension of in.");
    // 按行连续访存，因此要求三者都连续。
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

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
    // 行数（通常是 seqlen）与归一化长度 d，都直接从张量形状取得。
    const size_t m = in->shape()[0];
    const size_t d = in->shape()[1];
    const size_t numel = m * d;

    // 第 1 步：把输入与权重统一解码成 float。
    // 把数据类型分支集中在这一处，后面的数学计算就只有一条代码路径。
    std::vector<float> x(numel), w(d);
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *xp = reinterpret_cast<const float *>(in->data());
        const float *wp = reinterpret_cast<const float *>(weight->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = xp[i]; // 本身就是 float，直接复制
        }
        for (size_t j = 0; j < d; ++j) {
            w[j] = wp[j];
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *xp = reinterpret_cast<const bf16_t *>(in->data());
        const bf16_t *wp = reinterpret_cast<const bf16_t *>(weight->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = utils::cast<float>(xp[i]); // bf16 -> float
        }
        for (size_t j = 0; j < d; ++j) {
            w[j] = utils::cast<float>(wp[j]);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *xp = reinterpret_cast<const fp16_t *>(in->data());
        const fp16_t *wp = reinterpret_cast<const fp16_t *>(weight->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = utils::cast<float>(xp[i]); // fp16 -> float
        }
        for (size_t j = 0; j < d; ++j) {
            w[j] = utils::cast<float>(wp[j]);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    // 第 2 步：逐行归一化，结果就地覆盖写回 x（每行算完后原值不再需要，可以省一块缓冲区）。
    // RMS 统计量只在行内计算，行与行之间互不影响。
    for (size_t r = 0; r < m; ++r) {
        // 当前行的起始下标（连续存储，行首偏移为 r * d）。
        float *row = x.data() + r * d;

        // 先累加本行所有元素的平方。
        // 累加器用 double 而不是 float：d 可达数千，float 顺序累加的舍入误差会随项数增长，
        // 用 double 可以把这部分误差压到可忽略的水平。
        double sum_sq = 0.0;
        for (size_t j = 0; j < d; ++j) {
            const double xv = static_cast<double>(row[j]);
            sum_sq += xv * xv;
        }

        // 均方值 = 平方和 / d；加上 eps 后开平方得到 RMS。
        const double mean_sq = sum_sq / static_cast<double>(d) + static_cast<double>(eps);
        // 预先算出 1/RMS，把行内的 d 次除法换成 d 次乘法。
        // 结果落回 float：后续与输入、权重的乘法都在 float 上完成，精度已足够。
        const float inv_rms = static_cast<float>(1.0 / std::sqrt(mean_sq));

        // 逐元素完成缩放并乘上权重。
        for (size_t j = 0; j < d; ++j) {
            row[j] = row[j] * inv_rms * w[j];
        }
    }

    // 第 3 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        float *op = reinterpret_cast<float *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = x[i]; // 目标就是 float，直接写入
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t *op = reinterpret_cast<bf16_t *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = utils::cast<bf16_t>(x[i]); // float -> bf16
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t *op = reinterpret_cast<fp16_t *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = utils::cast<fp16_t>(x[i]); // float -> fp16
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
