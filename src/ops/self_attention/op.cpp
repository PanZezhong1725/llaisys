#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include <cmath>  // std::exp
#include <vector> // std::vector（float 解码缓冲区与分数暂存区）

namespace llaisys::ops {

// 因果自注意力（causal self-attention）：
//     A = Q * K^T * scale
//     Y = causalsoftmax(A) * V
//
//   attn_val : [seqlen,    nhead,   dv]，输出
//   q        : [seqlen,    nhead,   d ]
//   k        : [total_len, nkvhead, d ]
//   v        : [total_len, nkvhead, dv]
//   scale    : 缩放因子，通常取 1/sqrt(d)
//
// 若需要 kvcache，应在调用本算子之前先把 cache 与新的 k/v 拼接好再传入。
// 全部实现都写在本函数内：seqlen、total_len、nhead、nkvhead、d、dv
// 都直接从张量对象上读取，不通过参数传入。
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // ---- 参数检查 ----
    // 四个张量必须位于同一设备上。
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // 数据类型必须一致。
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    // 本次实现只处理 3 维的连续张量。
    ASSERT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    // 从张量形状取出各个维度。
    const size_t seqlen = q->shape()[0];    // query 长度
    const size_t nhead = q->shape()[1];     // query head 数
    const size_t d = q->shape()[2];         // query/key 的 head 维度
    const size_t total_len = k->shape()[0]; // key/value 长度（含 kvcache）
    const size_t nkvhead = k->shape()[1];   // kv head 数
    const size_t dv = v->shape()[2];        // value 的 head 维度

    // k 的 head 维度必须与 q 相同，否则无法做点积。
    ASSERT(k->shape()[2] == d, "SelfAttention: q and k must have the same head dimension.");
    // v 的长度与 head 数必须与 k 对齐（同一批 token、同一组 kv head）。
    ASSERT(v->shape()[0] == total_len && v->shape()[1] == nkvhead,
           "SelfAttention: k and v must have the same length and number of kv heads.");
    // 输出形状为 [seqlen, nhead, dv]。
    ASSERT(attn_val->shape()[0] == seqlen && attn_val->shape()[1] == nhead && attn_val->shape()[2] == dv,
           "SelfAttention: attn_val shape must be [seqlen, nhead, dv].");
    // GQA（分组查询注意力）要求 query head 数是 kv head 数的整数倍。
    ASSERT(nkvhead > 0 && nhead % nkvhead == 0,
           "SelfAttention: nhead must be a multiple of nkvhead.");
    // 因果掩码要求 key/value 至少和 query 一样长，否则最靠前的 query 没有可见的 key。
    ASSERT(total_len >= seqlen,
           "SelfAttention: total length of k/v must be at least the query length.");

    // ---- 设备分派：CPU 始终可用，其余设备暂不支持 ----
    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        // 先把线程上下文切换到目标设备。
        llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
#ifdef ENABLE_NVIDIA_API
        if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
            // CUDA 版本留待后续作业实现。
            TO_BE_IMPLEMENTED();
            return;
        }
#endif
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    // ---- 以下是 CPU 实现 ----
    // 三个输入各自的元素总数。
    const size_t q_numel = seqlen * nhead * d;
    const size_t k_numel = total_len * nkvhead * d;
    const size_t v_numel = total_len * nkvhead * dv;
    const size_t out_numel = seqlen * nhead * dv;

    // 第 1 步：把 q、k、v 统一解码成 float。
    // 把数据类型分支集中在这一处，后面的数学计算就只有一条代码路径。
    std::vector<float> qf(q_numel), kf(k_numel), vf(v_numel);
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32: {
        const float *qp = reinterpret_cast<const float *>(q->data());
        const float *kp = reinterpret_cast<const float *>(k->data());
        const float *vp = reinterpret_cast<const float *>(v->data());
        for (size_t i = 0; i < q_numel; ++i) {
            qf[i] = qp[i]; // 本身就是 float，直接复制
        }
        for (size_t i = 0; i < k_numel; ++i) {
            kf[i] = kp[i];
        }
        for (size_t i = 0; i < v_numel; ++i) {
            vf[i] = vp[i];
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        const bf16_t *qp = reinterpret_cast<const bf16_t *>(q->data());
        const bf16_t *kp = reinterpret_cast<const bf16_t *>(k->data());
        const bf16_t *vp = reinterpret_cast<const bf16_t *>(v->data());
        for (size_t i = 0; i < q_numel; ++i) {
            qf[i] = utils::cast<float>(qp[i]); // bf16 -> float
        }
        for (size_t i = 0; i < k_numel; ++i) {
            kf[i] = utils::cast<float>(kp[i]);
        }
        for (size_t i = 0; i < v_numel; ++i) {
            vf[i] = utils::cast<float>(vp[i]);
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        const fp16_t *qp = reinterpret_cast<const fp16_t *>(q->data());
        const fp16_t *kp = reinterpret_cast<const fp16_t *>(k->data());
        const fp16_t *vp = reinterpret_cast<const fp16_t *>(v->data());
        for (size_t i = 0; i < q_numel; ++i) {
            qf[i] = utils::cast<float>(qp[i]); // fp16 -> float
        }
        for (size_t i = 0; i < k_numel; ++i) {
            kf[i] = utils::cast<float>(kp[i]);
        }
        for (size_t i = 0; i < v_numel; ++i) {
            vf[i] = utils::cast<float>(vp[i]);
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }

    // 结果先存放在 float 缓冲区中，最后统一转换回原始数据类型。
    std::vector<float> out_f(out_numel);
    // 暂存一个 query 对所有可见 key 的分数，循环外分配一次即可反复复用。
    std::vector<float> scores(total_len);

    // 每个 query head 对应的 kv head 分组大小。
    // GQA：第 h 个 query head 复用第 h/group 个 kv head。这等价于参考实现里对 k、v 做
    // repeat_interleave，但这里只是换个下标读取，不需要真的复制数据。
    const size_t group = nhead / nkvhead;
    // 因果掩码的偏移量：第 i 个 query 能看到的最后一个 key 下标是 i + offset。
    // q 的长度可能小于 k/v 的长度（推理时前面 total_len-seqlen 个 token 来自 kvcache），
    // 参考实现用 tril(diagonal=S-L) 构造掩码，与这里的 offset 含义一致。
    const size_t offset = total_len - seqlen;

    // 第 2 步：逐 (query 位置, query head) 计算注意力。
    for (size_t i = 0; i < seqlen; ++i) {
        for (size_t h = 0; h < nhead; ++h) {
            const size_t kvh = h / group; // 该 query head 使用的 kv head 下标
            // 当前 query 向量的首地址：q[i][h][:]，长度 d。
            const float *q_vec = qf.data() + (i * nhead + h) * d;
            // 本行允许关注的 key 个数：下标 0 .. i+offset，共 i+offset+1 个。
            // 被掩掉的位置分数为 -inf、softmax 后权重为 0，对结果没有贡献，
            // 因此这里直接只遍历允许范围内的 key，省掉无用计算，也不必显式处理 -inf。
            const size_t n_vis = i + offset + 1;

            // 2a：算分数 A[i][j] = dot(q[i], k[j]) * scale，并顺便求最大值。
            // 最大值用于后面 softmax 的数值稳定化（减去最大值再取指数）。
            // 初值随意，因为下面用 j == 0 保证第一轮必定赋值。
            float max_score = 0.f;
            for (size_t j = 0; j < n_vis; ++j) {
                // 当前 key 向量的首地址：k[j][kvh][:]，长度 d。
                const float *k_vec = kf.data() + (j * nkvhead + kvh) * d;
                // 在 float 上累加点积。
                float dot = 0.f;
                for (size_t p = 0; p < d; ++p) {
                    dot += q_vec[p] * k_vec[p];
                }
                // 乘上缩放因子（一般为 1/sqrt(d)）。
                const float s = dot * scale;
                scores[j] = s;
                if (j == 0 || s > max_score) {
                    max_score = s;
                }
            }

            // 2b：causal softmax（只在可见范围内归一化）。
            // 先做 exp(s - max)，同时累加分母。减去 max 不改变 softmax 结果，
            // 但能保证指数的参数 <= 0，从而避免 exp 上溢。
            float denom = 0.f;
            for (size_t j = 0; j < n_vis; ++j) {
                const float e = std::exp(scores[j] - max_score);
                scores[j] = e;
                denom += e;
            }
            // 转成倒数，把后面的除法换成乘法。
            // denom >= 1（取到最大值的那一项恰为 exp(0) == 1），不会出现除零。
            const float inv_denom = 1.f / denom;

            // 2c：用归一化后的权重对 V 加权求和。
            // 输出向量的首地址：attn_val[i][h][:]，长度 dv。
            float *out_vec = out_f.data() + (i * nhead + h) * dv;
            for (size_t c = 0; c < dv; ++c) { // 逐个输出通道
                float acc = 0.f;
                for (size_t j = 0; j < n_vis; ++j) {
                    // v[j][kvh][c]：第 j 个 token、第 kvh 个 kv head 的第 c 维。
                    acc += scores[j] * vf[(j * nkvhead + kvh) * dv + c];
                }
                out_vec[c] = acc * inv_denom;
            }
        }
    }

    // 第 3 步：把 float 结果转换回原始数据类型写出（整个过程只在这里舍入一次）。
    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32: {
        float *op = reinterpret_cast<float *>(attn_val->data());
        for (size_t i = 0; i < out_numel; ++i) {
            op[i] = out_f[i]; // 目标就是 float，直接写入
        }
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        bf16_t *op = reinterpret_cast<bf16_t *>(attn_val->data());
        for (size_t i = 0; i < out_numel; ++i) {
            op[i] = utils::cast<bf16_t>(out_f[i]); // float -> bf16
        }
        break;
    }
    case LLAISYS_DTYPE_F16: {
        fp16_t *op = reinterpret_cast<fp16_t *>(attn_val->data());
        for (size_t i = 0; i < out_numel; ++i) {
            op[i] = utils::cast<fp16_t>(out_f[i]); // float -> fp16
        }
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
