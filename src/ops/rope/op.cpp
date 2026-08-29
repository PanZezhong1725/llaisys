#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>   // std::pow / std::sin / std::cos
#include <cstdint> // int64_t
#include <vector>  // std::vector（float 解码缓冲区与 sin/cos 查表）

namespace llaisys::ops {

// 旋转位置编码（RoPE）。
//   out     : [seqlen, nhead, d]，输出，连续张量
//   in      : [seqlen, nhead, d]，输入，连续张量
//   pos_ids : [seqlen]，int64，每个 token 在整个上下文中的位置下标
//   theta   : 频率基值（例如 10000.0）
//
// 把每个长度为 d 的向量按前后两半拆成 x = [a, b]，其中 a = x[0 : d/2]，b = x[d/2 : d]。
// 对第 i 个 token（位置 id 为 p_i）、第 j 个频率分量（j = 0 .. d/2-1）：
//     phi(i, j) = p_i / theta^(2j/d)
//     a'(i, j)  = a(i, j) * cos(phi) - b(i, j) * sin(phi)
//     b'(i, j)  = b(i, j) * cos(phi) + a(i, j) * sin(phi)
//
// 全部实现都写在本函数内：seqlen、nhead、d 都直接从张量对象上读取，不通过参数传入。
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // ---- 参数检查 ----
    // 三个张量必须位于同一设备上。
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // out 与 in 的形状、数据类型都必须一致。
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    // 本次实现只处理 3 维的连续张量。
    ASSERT(out->ndim() == 3 && in->ndim() == 3, "Rope: out and in must be 3D tensors.");
    ASSERT(pos_ids->ndim() == 1, "Rope: pos_ids must be a 1D tensor.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "Rope: all tensors must be contiguous.");
    // 位置 id 必须是 int64。
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "Rope: pos_ids must be of type int64.");

    // 从张量形状取出三个维度。
    const size_t seqlen = in->shape()[0]; // 序列长度
    const size_t nhead = in->shape()[1];  // head 数（q 用 nhead，k 用 nkvhead）
    const size_t d = in->shape()[2];      // 每个 head 的维度

    // pos_ids 需要给出每个 token 的位置，长度必须等于序列长度。
    ASSERT(pos_ids->shape()[0] == seqlen, "Rope: pos_ids length must match sequence length.");
    // 旋转是按 (a, b) 成对进行的，因此 d 必须是偶数。
    ASSERT(d % 2 == 0, "Rope: head dimension must be even.");

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
    const size_t numel = seqlen * nhead * d; // 元素总数
    const size_t half = d / 2;               // 每个向量前后两半的长度

    // 第 1 步：把输入统一解码成 float。
    // 把数据类型分支集中在这一处，后面的数学计算就只有一条代码路径。
    std::vector<float> x(numel);
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *xp = reinterpret_cast<const float *>(in->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = xp[i]; // 本身就是 float，直接复制
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *xp = reinterpret_cast<const bf16_t *>(in->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = utils::cast<float>(xp[i]); // bf16 -> float
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *xp = reinterpret_cast<const fp16_t *>(in->data());
        for (size_t i = 0; i < numel; ++i) {
            x[i] = utils::cast<float>(xp[i]); // fp16 -> float
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    // 第 2 步：预计算分母表 denom[j] = theta^(2j/d)。
    // 它只依赖 j，不依赖位置和 head，所以整个算子只需算 half 次幂运算。
    //
    // 【精度要点】角度必须用 float（f32）算，不能全程用 double。
    // 原因：PyTorch 参考实现里 theta^(2j/d) 是在 f32 下求幂的，其结果与“精确值舍入到 f32”
    // 最多相差约 1 个 ulp（相对误差约 1.2e-7）。而 phi = p_i / denom 与 p_i 成正比，
    // 测试里 p_i 最大到 1023，于是这 1 ulp 会被放大成约 1.2e-4 的绝对角度误差，
    // 直接体现在 sin/cos 上。若改用 double 求 phi，得到的是“另一个”同样合法但不同的角度，
    // 两边误差互相独立、无法抵消；实测在 f32 用例（atol=rtol=1e-4）下会偶发超差。
    // 按下面的方式在 f32 下求 denom 与 phi 后，有 98.6% 的 j 与 PyTorch 逐位相同，
    // 误差量级明显下降、测试稳定通过。
    std::vector<float> denom(half);
    for (size_t j = 0; j < half; ++j) {
        // 指数 e = 2j/d 在 f32 下计算，与 PyTorch 的 (2 * i / head_dim) 一致。
        const float e = static_cast<float>(2 * j) / static_cast<float>(d);
        // 幂运算用 double 求值后再落回 f32，得到“正确舍入”的 f32 结果。
        denom[j] = static_cast<float>(std::pow(static_cast<double>(theta),
                                               static_cast<double>(e)));
    }

    // 第 3 步：逐位置计算 sin/cos，再套用到该位置的所有 head。
    // 同一个位置 i 的所有 head 共享同一组角度，所以把 sin/cos 提到 head 循环之外先算好、
    // 再被 nhead 个 head 复用，可以把三角函数的调用次数从 seqlen*nhead*(d/2)
    // 降到 seqlen*(d/2)。两块查表缓冲区在循环外分配一次，反复复用。
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids->data());
    std::vector<double> sin_tab(half), cos_tab(half);

    for (size_t i = 0; i < seqlen; ++i) {
        // 取出第 i 个 token 的位置 id，并转成 float 参与角度计算。
        const int64_t pos = pos_ptr[i];
        ASSERT(pos >= 0, "Rope: position ids must be non-negative.");
        const float posf = static_cast<float>(pos);

        // 填充本位置的 sin/cos 查表。
        for (size_t j = 0; j < half; ++j) {
            // 角度在 f32 下相除（与 PyTorch 的 positions / theta**(...) 对齐）。
            const float phi = posf / denom[j];
            // 对这个已经确定的 f32 角度用 double 求 sin/cos（精度更高，
            // 且不会引入与 PyTorch 之间的系统性偏差）。
            const double phid = static_cast<double>(phi);
            sin_tab[j] = std::sin(phid);
            cos_tab[j] = std::cos(phid);
        }

        // 对该位置下的每个 head 施加旋转，结果就地覆盖写回 x。
        for (size_t h = 0; h < nhead; ++h) {
            // 该 (token, head) 对应向量在连续内存中的起始下标。
            float *vec = x.data() + (i * nhead + h) * d;

            for (size_t j = 0; j < half; ++j) {
                // a 取前半段，b 取后半段（两者相距 half 个元素）。
                const double a = static_cast<double>(vec[j]);
                const double b = static_cast<double>(vec[j + half]);
                const double s = sin_tab[j];
                const double c = cos_tab[j];

                // 旋转公式：结果的前半段写 a'，后半段写 b'。
                vec[j] = static_cast<float>(a * c - b * s);
                vec[j + half] = static_cast<float>(b * c + a * s);
            }
        }
    }

    // 第 4 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
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
