#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cstdint> // int64_t
#include <cstring> // std::memcpy

namespace llaisys::ops {

// 从 weight（2-D）中按 index（1-D）指定的行号取行，依次写入 out（2-D）。
// index 必须是 Int64 类型（PyTorch 中整型的默认数据类型）。
// 全部实现都写在本函数内：行数、列数、行字节数等信息都直接从张量对象上读取，不通过参数传入。
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // ---- 参数检查 ----
    // 三个张量必须在同一设备上。
    CHECK_SAME_DEVICE(out, index, weight);
    // 输出与词嵌入表保存同类数据，dtype 必须一致。
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    // 下标张量固定为 Int64。
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be of type int64.");
    // 维度约束：下标是一维，权重与输出都是二维。
    ASSERT(index->ndim() == 1, "Embedding: index must be a 1D tensor.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be a 2D tensor.");
    ASSERT(out->ndim() == 2, "Embedding: out must be a 2D tensor.");
    // 输出行数等于下标个数，列数等于词向量维度。
    ASSERT(out->shape()[0] == index->shape()[0],
           "Embedding: out rows must match index length.");
    ASSERT(out->shape()[1] == weight->shape()[1],
           "Embedding: out columns must match weight columns.");
    // 本实现按整行连续拷贝，因此要求张量连续。
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

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
    // 需要查表的下标个数（即输出行数）与词表大小（即 weight 的行数，用于越界检查），
    // 都直接从张量形状取得。
    const size_t nidx = index->shape()[0];
    const size_t nvocab = weight->shape()[0];
    // 一行数据占用的字节数 = 列数 * 单个元素字节数。
    const size_t row_bytes = weight->shape()[1] * weight->elementSize();

    // 下标数组按 int64 解释。
    const int64_t *idx = reinterpret_cast<const int64_t *>(index->data());
    // 词嵌入表与输出都按原始字节访问。
    const std::byte *weight_raw = weight->data();
    std::byte *out_raw = out->data();

    // 这里不需要按数据类型分情况处理：out 与 weight 的 dtype 相同，
    // 查表本质上就是整行的内存搬移，直接按字节复制即可。
    // 这样既天然支持任意数据类型，也保证结果与输入按位一致（满足测试的 strict=True）。
    for (size_t i = 0; i < nidx; ++i) {
        // 取出第 i 个输出行对应的源行号。
        const int64_t row = idx[i];
        // 行号必须落在 [0, nvocab) 内，否则会读到词嵌入表之外的内存。
        // 先判负，再把 row 转成 size_t 与 nvocab 比较，避免有符号/无符号混合比较的告警。
        ASSERT(row >= 0 && static_cast<size_t>(row) < nvocab,
               "Embedding: index out of range.");
        // 整行拷贝：源为 weight 的第 row 行，目标为 out 的第 i 行。
        std::memcpy(out_raw + i * row_bytes,
                    weight_raw + static_cast<size_t>(row) * row_bytes,
                    row_bytes);
    }
}
} // namespace llaisys::ops
