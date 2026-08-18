# `self_attention` 算子实现过程记录

> 本文记录 `src/ops/self_attention/` 从空实现到完整实现(V1 → V4)的完整迭代过程,包含每一版的设计动机、核心代码、遇到的问题和排查过程。已对照当前仓库源码(`src/ops/self_attention/**`)核对。

## 1. 目标与整体接口

实现的是标准的因果(causal)、支持 GQA 的 scaled dot-product self-attention:

```
attn_val = softmax(scale * q @ k^T + causal_mask) @ v
```

最终的张量形状约定(`op.cpp` 从 tensor shape 里推导,传给 CPU kernel):

| 张量 | 形状 | 含义 |
|---|---|---|
| `q` | `[seqlen, nhead, d]` | 本次新增的 query |
| `k`, `v` | `[total_len, nkvhead, d]` / `[total_len, nkvhead, dv]` | 含 KV Cache 历史在内的完整 key/value |
| `attn_val` | `[seqlen, nhead, dv]` | 输出 |

其中 `nhead` 必须是 `nkvhead` 的整数倍(GQA),`causal_offset = total_len - seqlen` 表示 KV Cache 里历史 token 的长度,query 位置 `i` 能看到的 key 范围是 `j <= i + causal_offset`。

最终 CPU kernel 接口(`cpu/self_attention.hpp`):

```cpp
namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                     llaisysDataType_t type,
                     size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead, size_t d, size_t dv,
                     float scale);
}
```

## 2. 迭代路线:V1 → V4

没有一次写全,而是刻意拆成四个版本,每一版只加一个新概念,便于逐步验证:

| 版本 | 新增内容 | 循环方式 | 是否过测试 |
|---|---|---|---|
| V1 | 最朴素实现 | 单遍扫描,直接 `exp(score)` | 部分(仅 `nh==nkvh` 且数值范围温和时) |
| V2 | + 因果掩码 | 单遍扫描,`j` 范围收窄到 `[0, i+causal_offset]` | `nh==nkvh` 用例全过 |
| V3 | + GQA | 单遍扫描,`k`/`v` 按 `kvh = h/group` 取 | 全部用例过 |
| V4 | + 数值稳定 softmax | 两遍扫描(先求 max,再减 max 求 exp) | 全部用例过,且对极端数值更鲁棒 |

### V1:最简单版本

- **设计取舍**:不支持 GQA(假设 `nhead == nkvhead`)、不做因果掩码(每个 `i` 都看全部 `total_len` 个 key)、不做 softmax 数值稳定处理(直接 `exp(score)`,不减 max)。明确"这一版不需要过测试"。
- **核心逻辑**(单遍扫描,打分、`exp`、累加一次做完):

```cpp
float sum_exp = 0.0f;
std::vector<float> acc(dv, 0.0f);
for (size_t j = 0; j < total_len; j++) {
    float score = 0.0f;
    for (size_t dim = 0; dim < d; dim++) {
        score += cast<float>(q[...]) * cast<float>(k[...]);
    }
    score *= scale;
    float e = std::exp(score);
    sum_exp += e;
    for (size_t t = 0; t < dv; t++) {
        acc[t] += e * cast<float>(v[...]);
    }
}
for (size_t t = 0; t < dv; t++) {
    attn_val[...] = cast<T>(acc[t] / sum_exp);
}
```

- **`acc` 的含义**:不是 QKᵀ 矩阵,而是 softmax 权重对 V 的加权和(输出的分子部分),长度是 `dv` 而不是 `total_len`。

### V2:加因果掩码

- 只改一处:`j` 的循环范围从 `[0, total_len)` 收窄成 `[0, i + causal_offset]`(闭区间),其余打分/exp/累加/归一化逻辑不变。
- `causal_offset = total_len - seqlen`,含义是 KV Cache 里历史 token 的长度。
- 验证方式:第一组测试形状 `(qlen=2, kvlen=2, nh=1, nkvh=1, hd=4)` 里,`i=1` 时两种实现(有/无掩码)结果本来就一样(因为 `total_len` 刚好等于看得到的范围),但 `i=0` 时不一样——这正好是验证掩码是否生效的天然信号。

### V3:加 GQA(Grouped-Query Attention)

- `op.cpp` 的校验从 `nhead == nkvhead` 放宽为 `nhead % nkvhead == 0`。
- kernel 里新增 `group = nhead / nkvhead`,每个 query head `h` 对应的 kv head 是 `kvh = h / group`。
- 把原来所有 `k`/`v` 下标里的 `nhead`/`h` 换成 `nkvhead`/`kvh`(`q`、`attn_val` 的下标不用换,它们本来就按 `nhead` 排列)。

```cpp
size_t kvh = h / group;
...
score += cast<float>(q[i*nhead*d + h*d + dim]) * cast<float>(k[j*nkvhead*d + kvh*d + dim]);
...
acc[t] += e * cast<float>(v[j*nkvhead*dv + kvh*dv + t]);
```

### V4:数值稳定 softmax

- **动机**:直接 `exp(score)` 在 `score` 较大时会溢出成 `inf`,进而导致 `sum_exp` 变成 `inf`、最终输出变成 `NaN`。数学上 softmax 对所有输入减去任意常数结果不变(`exp(c)` 项在分子分母间约掉),所以可以安全地统一减去 `max_score`,让指数的输入恒 `<= 0`,分子的每一项被压缩到 `(0, 1]`,`sum_exp` 也因此有下界(至少为 1),彻底避免 `inf`/`NaN`。
- **代价**:单遍扫描不再够用。第二步 `exp(score - max_score)` 依赖一个要扫完整个 `j` 范围才能确定的全局量 `max_score`,所以必须拆成两遍;而两遍之间要把每个 `score[j]` 重新用上,于是引入 `std::vector<float> scores` 把第一遍算出的打分存下来,避免第二遍重新做一次 `O(d)` 的点积(空间换时间)。

```cpp
// 第一遍:算出所有 score[j] 并记录最大值 max_score。
std::vector<float> scores(limit + 1);
float max_score = -std::numeric_limits<float>::infinity();
for (size_t j = 0; j <= limit; j++) {
    float score = 0.0f;
    for (size_t dim = 0; dim < d; dim++) {
        score += cast<float>(q[...]) * cast<float>(k[...]);
    }
    score *= scale;
    scores[j] = score;
    max_score = std::max(max_score, score);
}

float sum_exp = 0.0f;
std::vector<float> acc(dv, 0.0f);
// 第二遍:用 exp(score - max_score) 做数值稳定的 softmax 加权求和。
for (size_t j = 0; j <= limit; j++) {
    float e = std::exp(scores[j] - max_score);
    sum_exp += e;
    for (size_t t = 0; t < dv; t++) {
        acc[t] += e * cast<float>(v[...]);
    }
}
```

## 3. 过程中排查过的问题

实现过程中不是一路顺风,记录几个有代表性的坑:

1. **全角标点导致语法错误**:注释里混入了全角逗号 `、` 和被意外换行拆断的 `//acc` → `/` + `acc[t]...`,编译器把残余的 `/` 当除号解析,报 `expected primary-expression before '/' token`。这类问题的特征是报错行号和实际"看起来有问题"的代码对不上,需要用 `cat -A` 或 `sed -n` 看原始字节才能定位。

2. **`-Werror=unused-variable` / `-Werror=sign-compare`**:项目开了 `-Werror`,每次先加一个后面才会用到的变量(比如 V2 的 `causal_offset`、V3 的 `group`、V4 的 `max_score`)都需要先 `(void)var;` 占位,否则编译不过;循环变量声明成 `int` 和 `size_t` 比较也会报错,统一改成 `size_t`。

3. **`.so` 没同步导致的假报错**:改完代码跑 Python 测试,报错信息(`Shapes mismatch`)和当前源码对不上号。排查后发现是两层缓存没刷新:
   - `xmake build` 编译产物在 `build/`,需要额外 `xmake install` 才会同步到 `python/llaisys/libllaisys/libllaisys.so`;
   - 更隐蔽的是,`test/ops/self_attention.py` 用的 `sys.path.insert` 加的是 `test/` 目录而不是 `python/` 目录,实际 `import llaisys` 落到了之前 `pip install ./python/` 装到 conda `site-packages` 里的旧副本,而不是仓库本地这份。用 `gdb -batch -ex "catch throw" ... --args python ...` 打断点看堆栈,才在帧里看到 `.so` 的真实加载路径,定位到问题。
   - 修复方式:改完代码后固定跑 `xmake build && xmake install && pip install ./python/`,确保三层(编译产物 / xmake install 目标 / pip 包)都同步。

4. **半成品 TODO 被当成代码**:自己实现两遍扫描版本时,把我给的 TODO 提示文字(`max_score = std::max(max_score, scores[j]); // 这里是伪代码...`)直接留在了代码里,`j` 没有对应的循环声明,编译报 `'j' was not declared in this scope`。提醒:TODO 注释只是提示,真正要写一个完整的 `for` 循环替换掉它。

## 4. 最终测试结果

```
qlen=2  kvlen=2  nh=1 nkvh=1 hd=4   f32 / f16 / bf16   ✓
qlen=5  kvlen=11 nh=4 nkvh=2 hd=8   f32 / f16 / bf16   ✓
Test passed!
```

覆盖了 `nh==nkvh`(无分组)和 `nh=4,nkvh=2`(2 倍分组)两种 GQA 配置,三种精度(`f32`/`f16`/`bf16`)全部通过 `test/ops/self_attention.py`。

## 5. 涉及的文件

- `src/ops/self_attention/op.hpp` / `op.cpp`:对外接口,做形状校验、从 tensor shape 推导出 `seqlen/total_len/nhead/nkvhead/d/dv`,按 device 分发。
- `src/ops/self_attention/cpu/self_attention.hpp` / `self_attention.cpp`:CPU kernel,按 dtype(`f32`/`bf16`/`f16`)分发到同一个模板函数 `self_attention_<T>`。
- `test/ops/self_attention.py`:用 PyTorch 的 `scaled_dot_product_attention` 等价实现做基准,对比 `llaisys.Ops.self_attention` 的输出。
