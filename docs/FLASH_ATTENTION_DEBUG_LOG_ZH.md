# `self_attention` NVIDIA Flash Attention 开发笔记

> 个人开发过程记录，不是提交评分用的文档（评分文档见 `REPORT.md`）。记录 `src/ops/self_attention/nvidia/flash_attention_cuda.cu`（prefill）+ `flash_attention_decode_cuda.cu`（decode）从接线到跑通期间踩到的两个真实 bug，方便以后遇到同类问题时回来查。

## 背景

`self_attention` 原来在 NVIDIA 上只有 V1 手写 kernel（一个 block 处理一个 `(query token, head)`，两遍扫描）。这次重写目标是给 prefill/decode 各写一个专门优化过的 flash attention kernel（K/V 分块 + online softmax），`op.cpp` 按 shape 分流：

```cpp
if (seqlen > 1 && total_len == seqlen && d == 128 && dv == 128) {
    flash_attention(...);          // prefill：K/V 分块 + online softmax，TILE_Q=8 行/block
} else if (seqlen == 1 && d == 128 && dv == 128) {
    flash_attention_decode(...);   // decode：单 query 行，一个 warp 处理一个 head
} else {
    self_attention(...);           // 其余情况（非 128 的 head_dim、非首次 prefill）落回 V1
}
```

写完主体逻辑、刚接上 dispatch 时，跑 `test/ops/self_attention.py --device nvidia`（自带的两组小 shape，`hd=4`/`hd=8`，逻辑上根本不会走 flash 分支）就间歇性崩溃——这是第一个 bug；解决之后用专门覆盖 `hd=128` flash 路径的对拍脚本测更大的 shape，又炸出第二个 bug。两个 bug 完全独立，分开记。

## Bug 1：ODR（重复定义）导致的间歇性 `illegal memory access`

### 现象

`test/ops/self_attention.py --device nvidia` 连续跑几次，大概 30%-50% 概率随机报：

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
CUDA error at src/device/nvidia/nvidia_runtime_api.cu:69: an illegal memory access was encountered
```

诡异的地方：这个测试脚本只用 `hd=4`/`hd=8` 两组 shape，`op.cpp` 的分流条件要求 `d==128 && dv==128` 才会调用 flash kernel——按代码逻辑，flash kernel **根本不应该被调用**。而且用 `compute-sanitizer --tool memcheck` 跑同一个命令反而 0 errors、稳定通过；`git stash` 掉 `op.cpp` 的 dispatch 改动（纯 V1）跑 15 次全过。说明问题跟"是否真的调用了 flash kernel"无关，是别的东西在起作用。

### 排查

`git stash`/`pop` 反复横跳（配合多次 `xmake build -r` 强制全量重编）之后，怀疑是不是链接层面的问题，直接对比两个 `.cu` 文件里同名符号的完整签名：

```cpp
// self_attention_cuda.cu（V1）
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale)

// flash_attention_cuda.cu（当时的写法）
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      size_t seqlen, size_t total_len, size_t nhead, size_t nkvhead,
                                      size_t d, size_t dv, float scale)
```

**两个不同 `.cu` 文件，函数模板同名、参数类型完全一样，函数体完全不同。** 两者都在文件全局作用域（没有 `namespace { ... }`、没有 `static`），都是外部链接。C++ 里函数模板的每个实例化默认是弱符号（weak/linkonce），多个 `.o` 里出现同一个弱符号时，链接器只保留一份——具体保留哪一份跟目标文件在静态库里的排列顺序、优化级别等因素有关，是未定义行为，这正好解释了"随机"这个现象：**V1 的调用点有一定概率被链接器悄悄换成了 flash kernel 的机器码**，而 V1 的 launcher 只按 `total_len*sizeof(float)` 分配了很小的 shared memory，flash kernel 却需要 `Bc*(d+dv)*sizeof(float)`（`hd=4/8` 时按 V1 的公式算出来的 shared memory 远小于 flash kernel 实际会访问的范围）——一旦被换过去就是越界访问。

`launch_self_attention`（host 端 launcher）也是同样的重名情况，一并确认。

### 修复

把 `flash_attention_cuda.cu` 里的 `self_attention_kernel`/`launch_self_attention` 改名成 `flash_attention_kernel`/`launch_flash_attention`，跟 V1 不再同名。（`flash_attention_decode_cuda.cu` 从一开始就把所有内部符号包在匿名 `namespace { ... }` 里，天然拿到内部链接，不会有这个问题——这是更稳妥的写法，以后新增 `.cu` 文件时应该照这个来，不要依赖"记得手动改名"。）

### 验证

`xmake build -r`（强制全量重编，排除任何缓存 `.o` 的影响）之后，`test/ops/self_attention.py --device nvidia` 连续跑 10 次，10/10 通过（改之前大概 30%-50% 崩溃率，个别情况下 python 进程直接 core dump 或者 hang 住）。

### 教训

- **C++ 里 `__global__`/普通函数模板默认外部链接**，不同 `.cu` 文件里出现同名同签名的模板函数，就算内容完全不同，也不会在编译期报错——只有链接期弱符号合并的未定义行为，behavior 取决于构建细节，非常难复现和定位。新加一个跟已有 kernel 结构相似的 `.cu` 文件（复制粘贴改一份是常见操作）时，命名要么加前缀避免撞名，要么直接包进匿名 `namespace`。
- **"错误跟代码逻辑对不上"是重要信号**：guard 条件明明排除了 flash 路径，却还是间歇性触发本该只有 flash 路径才会踩到的越界访问——遇到这种"逻辑上不可能，但现象上确实发生"的情况，第一反应不该是继续在业务逻辑里找 bug，而是往"链接/构建层面是不是有问题"这个方向想。`compute-sanitizer` 跑几次全过、纯手跑反而间歇失败，这种"加了诊断工具后 bug 消失"的模式也是一个提示（诊断工具往往会改变时序或强制同步，容易让竞态/未定义行为暂时不出现）。

## Bug 2：跨 chunk 的 shared memory data race

### 现象

修完 Bug 1 之后，用专门覆盖 flash 路径（`hd=128`，触发 `op.cpp` 的分流条件）的对拍脚本测：`qlen ∈ {2, 8, 9, 17, 33}`，GQA(`nh=12,nkvh=2`) 和非 GQA(`nh=4,nkvh=4`) 各一遍，三种精度都测。`qlen=33` 在 f32 精度下断言失败，`qlen<=17` 全过。

单独写脚本把 llaisys 输出和 PyTorch 参考实现逐 query 行比较误差（`(got - ref).abs().amax(dim=(1,2))`）：

```
row  0: max_abs_err=0.000000e+00
row  1: max_abs_err=1.192093e-07   # float32 舍入误差量级，正常
...
row 31: max_abs_err=1.192093e-07
row 32: max_abs_err=2.323037e-02   # 差两个数量级，逻辑错误
```

只有第 32 行（`qlen=33` 里最后一行）误差异常。

### 定位

kernel 行方向 `TILE_Q=8` 行一组、列方向 `Bc=32` 个 key 一组分块。`qlen=33` 时第 32 行单独落在最后一个 tile（`tile_start=32`，这个 tile 只有第 32 行存在）。这个 tile 的 `tile_max_limit = min(32+8-1, 33-1) = 32`，K/V 分块循环 `for (j=0; j<=tile_max_limit; j+=Bc)` 要跑**两轮**（`j=0` 覆盖 key 0-31，`j=32` 只覆盖 key 32）。`qlen<=32` 的所有 tile，`tile_max_limit` 都小于 `Bc=32`，分块循环只跑一轮——也就是说**在这次调试之前，"多轮分块累加"这个 flash attention 真正的核心逻辑，从来没被实际测到过**，小 shape 测试全部只经历单轮分块。

检查 kernel 主循环，每轮分块结束（读完 `K_chunk`/`V_chunk` 算完这一轮 softmax）之后直接回到循环顶部做下一轮的协作搬运，中间没有 `__syncthreads()`：

```cpp
for (int j = 0; j <= tile_max_limit; j += Bc) {
    // 协作搬运 K/V 进 shared memory
    ...
    __syncthreads();          // 搬运完 → 读之前，有同步

    if (i < seqlen) {
        ... 用 K_chunk/V_chunk 算 score/softmax/累加 ...
    }
    // 直接回到 for 循环顶部，没有同步！
}
```

`qlen=33` 那个 tile 里 8 个 warp 只有 `warp_id=0`（第 32 行）满足 `i<seqlen`，其余 7 个 warp 直接跳过整个 `if` 块，抢先跑到下一轮循环顶部开始覆盖 `K_chunk`/`V_chunk`——这时候 `warp_id=0` 那个 warp 可能还在读上一轮的数据（要做 128 维点积 + warp reduce + 累加，比"直接跳过"的空 warp 慢得多）。典型的 shared memory 双缓冲竞态：同一块内存复用给下一轮数据，但没有保证"所有线程都读完当前轮"。

### 修复

循环体末尾（`if (i < seqlen) {...}` 之后、回到循环顶部之前）补一个 `__syncthreads()`：

```cpp
        if (i < seqlen) {
            ...
        }
        __syncthreads();   // 等所有线程读完这一轮的 K_chunk/V_chunk，再进入下一轮覆盖它们
    }
```

### 验证

- 对拍脚本：prefill `qlen ∈ {2,8,9,17,33}`（含跨分块的 33）、decode `total_len ∈ {1,2,31,32,33,65}`（含跨分块的 33/65），GQA + 非 GQA，f32/f16/bf16 全过；`qlen=33` 误差回到 `1e-7` 量级。
- `test/ops/self_attention.py --device nvidia` 连续跑 10 次全过。
- 其余 7 个 NVIDIA 算子回归无变化。
- `test/test_infer.py --model DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --max_steps 32`：和 HF 参考 token 级完全一致。

### 教训

分块类 kernel（flash attention 这种）设计测试 shape 时，光测 `total_len <= 分块大小` 只验证了"单块内 softmax 对不对"，测不出"多块怎么正确拼接（rescale + 跨块状态维护）"——这部分恰恰是算法的核心。以后要故意选一个刚好比分块大小（`Bc`）多一点的值（比如 `Bc+1`），逼出"最后一块只有一行/几行"这种边界场景，才能把跨块逻辑真正测到。

## 两个 bug 的共同点

都不是"小 shape 测试用例覆盖不到"就能免责的那种边界 bug——Bug 1 理论上任何 shape 都可能触发（纯粹是构建/链接层面的运气问题），Bug 2 则是专门针对"跨 chunk"这个此前从未被覆盖到的代码路径。两个 bug 都是先用小 shape 验证"基本没问题"之后，扩大测试范围（换更真实的 shape、专门构造边界值）才暴露出来的——提醒自己：小 shape 测试通过 ≠ 逻辑正确，尤其是涉及分块/tiling 的 kernel，一定要单独设计能触发"多轮/多块"路径的测试用例。
