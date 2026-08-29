#include "op.hpp"

// 复用工程内已有的公共设施：
//   - core/llaisys_core.hpp 提供 llaisys::core::context()，用于在多设备间切换当前设备。
//   - utils.hpp 汇总了 utils/check.hpp（CHECK_*/ASSERT 等校验宏）与
//     utils/types.hpp（utils::cast<> 数据类型转换工具）。
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstdint> // int64_t
#include <cstring> // std::memcpy
#include <vector>  // std::vector（float 解码缓冲区）

namespace llaisys::ops {

// 获取张量 vals 的最大值及其索引，分别写入 max_val 与 max_idx。
// 当前假设 vals 是一维张量，max_idx 与 max_val 都是只含单个元素的一维张量（即保留了 vals 的维度）。
// 全部实现都写在本函数内：所需的元素个数、元素字节数等信息都直接从张量对象上读取，不通过参数传入。
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // ---- 参数检查 ----
    // 三个张量必须位于同一设备，否则无法在同一段代码里直接访存。
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // max_val 保存的是 vals 中的元素，因此两者数据类型必须一致。
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    // 下标张量固定为 Int64，与 PyTorch 中 torch.max 返回的 indices 类型对齐。
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be of type int64.");
    // 输入为一维张量。
    ASSERT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    // 需要至少一个元素才存在“最大值”。
    ASSERT(vals->numel() > 0, "Argmax: vals must not be empty.");
    // 两个输出张量各自只保存一个标量结果。
    ASSERT(max_idx->numel() == 1 && max_val->numel() == 1,
           "Argmax: max_idx and max_val must contain exactly one element.");
    // 本实现按线性内存扫描，因此要求张量连续。
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    // ---- 设备分派：CPU 始终可用，其余设备暂不支持 ----
    if (vals->deviceType() != LLAISYS_DEVICE_CPU) {
        // 先把线程上下文切换到目标设备。
        llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
#ifdef ENABLE_NVIDIA_API
        if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            // CUDA 版本留待后续作业实现。
            TO_BE_IMPLEMENTED();
            return;
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // ---- 以下是 CPU 实现 ----
    // 元素总数与单个元素的字节数，都直接从张量本身取得。
    const size_t numel = vals->numel();
    const size_t esize = vals->elementSize();
    // 输入数据的原始字节指针（后面写回最大值时还要用它来做整元素拷贝）。
    const std::byte *vals_raw = vals->data();

    // 第 1 步：把输入统一解码成 float。
    // fp16/bf16 是自定义的 16 位类型，没有重载比较运算符，必须先提升到 float 才能比较；
    // 把类型分支集中在这一处，后面的扫描就只有一条代码路径。
    std::vector<float> v(numel);
    switch (vals->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *p = reinterpret_cast<const float *>(vals_raw);
        for (size_t i = 0; i < numel; ++i) {
            v[i] = p[i]; // 本身就是 float，直接复制
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *p = reinterpret_cast<const bf16_t *>(vals_raw);
        for (size_t i = 0; i < numel; ++i) {
            v[i] = utils::cast<float>(p[i]); // bf16 -> float
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *p = reinterpret_cast<const fp16_t *>(vals_raw);
        for (size_t i = 0; i < numel; ++i) {
            v[i] = utils::cast<float>(p[i]); // fp16 -> float
        }
        break;
    }
    default:
        // 其余数据类型暂不支持，抛出带类型名的异常。
        EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
    }

    // 第 2 步：线性扫描求最大值所在的下标。
    // 用第 0 个元素作为初始值（上面已校验 numel >= 1，不会越界）。
    size_t best_idx = 0;
    float best_val = v[0];
    for (size_t i = 1; i < numel; ++i) {
        // 只有“严格大于”才更新：这样在出现并列最大值时保留下标最小的那个，
        // 与 torch.max 返回首个最大值位置的行为一致（低精度类型下并列很常见）。
        if (v[i] > best_val) {
            best_val = v[i];
            best_idx = i;
        }
    }

    // 第 3 步：写回结果。
    // 下标按 int64 写出；显式 static_cast 避免 MSVC 对 size_t -> int64_t 的收窄告警
    //（本工程开启了 -WX，告警会被升级为错误）。
    *reinterpret_cast<int64_t *>(max_idx->data()) = static_cast<int64_t>(best_idx);
    // 最大值直接按字节从原始数据整元素拷贝，而不是把解码后的 float 再转换回去：
    // 这样输出与输入按位完全相等，满足测试中 strict=True 的逐位比较，
    // 同时也不必再为每种数据类型写一遍写回逻辑。
    std::memcpy(max_val->data(), vals_raw + best_idx * esize, esize);
}
} // namespace llaisys::ops
