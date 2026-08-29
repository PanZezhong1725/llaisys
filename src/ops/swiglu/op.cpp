#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>  // std::exp
#include <vector> // std::vector（float 解码缓冲区）

namespace llaisys::ops {

// 逐元素计算：out[i] = up[i] * gate[i] / (1 + exp(-gate[i]))
// 其中 gate[i] / (1 + exp(-gate[i])) 就是 SiLU（又称 Swish）激活。
// out、gate、up 是形状相同的 2D 连续张量 [seqlen, intermediate_size]。
// 全部实现都写在本函数内：元素个数直接从张量对象上读取，不通过参数传入。
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // ---- 参数检查 ----
    // 三个张量必须在同一设备上。
    CHECK_SAME_DEVICE(out, gate, up);
    // 逐元素运算，要求形状完全相同（不涉及广播）。
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    // 数据类型也必须一致。
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    // 按线性下标遍历，因此要求三者都连续。
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");

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
    // 元素总数直接从张量取得（三者形状相同，可以拉平成一维遍历）。
    const size_t numel = out->numel();

    // 第 1 步：把两个输入统一解码成 float。
    // 这样做有两个原因：一是 exp 只有浮点版本，二是在 f16/bf16 上直接算指数会明显放大误差。
    // 把数据类型分支集中在这一处，后面的数学计算就只有一条代码路径。
    std::vector<float> g(numel), u(numel);
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *gp = reinterpret_cast<const float *>(gate->data());
        const float *upp = reinterpret_cast<const float *>(up->data());
        for (size_t i = 0; i < numel; ++i) {
            g[i] = gp[i]; // 本身就是 float，直接复制
            u[i] = upp[i];
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *gp = reinterpret_cast<const bf16_t *>(gate->data());
        const bf16_t *upp = reinterpret_cast<const bf16_t *>(up->data());
        for (size_t i = 0; i < numel; ++i) {
            g[i] = utils::cast<float>(gp[i]); // bf16 -> float
            u[i] = utils::cast<float>(upp[i]);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *gp = reinterpret_cast<const fp16_t *>(gate->data());
        const fp16_t *upp = reinterpret_cast<const fp16_t *>(up->data());
        for (size_t i = 0; i < numel; ++i) {
            g[i] = utils::cast<float>(gp[i]); // fp16 -> float
            u[i] = utils::cast<float>(upp[i]);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }

    // 第 2 步：逐元素计算，结果就地覆盖写回 g（g 之后不再被读取，可以省一块缓冲区）。
    for (size_t i = 0; i < numel; ++i) {
        // sigmoid(gate) = 1 / (1 + exp(-gate))。
        // 传入 float 会选中 std::exp 的 float 重载，避免不必要的 double 运算；
        // gate 很负时 exp(-gate) 溢出为 +inf，此时该式自然收敛到 0，符合数学极限，不会产生 NaN。
        const float sigmoid = 1.0f / (1.0f + std::exp(-g[i]));
        // SiLU 与 up 相乘得到最终结果。
        g[i] = u[i] * g[i] * sigmoid;
    }

    // 第 3 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32: {
        float *op = reinterpret_cast<float *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = g[i]; // 目标就是 float，直接写入
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t *op = reinterpret_cast<bf16_t *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = utils::cast<bf16_t>(g[i]); // float -> bf16
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t *op = reinterpret_cast<fp16_t *>(out->data());
        for (size_t i = 0; i < numel; ++i) {
            op[i] = utils::cast<fp16_t>(g[i]); // float -> fp16
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
