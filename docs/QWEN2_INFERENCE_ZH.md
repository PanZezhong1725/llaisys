# Assignment #3:Qwen2 大模型推理 —— 任务说明

> 本文梳理 Assignment #3(`README.md` "Large Language Model Inference" 一节)要完成的工作,对照当前仓库里已有的骨架代码(`include/llaisys/models/qwen2.h`、`python/llaisys/models/qwen2.py`、`test/test_infer.py`)整理成一份可执行的任务清单。目标模型是 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),它的注意力机制细节见 [[SELF_ATTENTION_ZH.md]] 同目录下的分析文档。

## 1. 目标

用 C/C++ 在 LLAISYS 后端实现 Qwen2 模型的**完整推理**(prefill + 增量解码),Python 侧只做权重加载和薄封装,**不允许**用 PyTorch 之类的框架在 Python 里代劳计算逻辑。验收标准是 `test/test_infer.py --test` 通过——用 `top_k=1`(贪心采样)时,LLAISYS 生成的 token 序列要和 HuggingFace `transformers` 跑出来的完全一致。

## 2. 现状盘点:哪些是已经搭好的骨架,哪些是空的

### 已经定义好、不用你操心的部分

**C 结构体和函数原型**(`include/llaisys/models/qwen2.h`),已经给出:

```c
struct LlaisysQwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct LlaisysQwen2Weights {
    llaisysTensor_t in_embed;
    llaisysTensor_t out_embed;
    llaisysTensor_t out_norm_w;
    llaisysTensor_t *attn_norm_w;  // 每层一个,数组长度 = nlayer
    llaisysTensor_t *attn_q_w;
    llaisysTensor_t *attn_q_b;
    llaisysTensor_t *attn_k_w;
    llaisysTensor_t *attn_k_b;
    llaisysTensor_t *attn_v_w;
    llaisysTensor_t *attn_v_b;
    llaisysTensor_t *attn_o_w;
    llaisysTensor_t *mlp_norm_w;
    llaisysTensor_t *mlp_gate_w;
    llaisysTensor_t *mlp_up_w;
    llaisysTensor_t *mlp_down_w;
};

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model);
struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model);
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken);
```

注意 `attn_q_w/attn_q_b`、`attn_k_w/attn_k_b`、`attn_v_w/attn_v_b` 都有 bias,但 `attn_o_w` 没有——这正好对应 Qwen2 架构里 QKV 投影带 bias、输出投影不带 bias 的设计(细节见 `SELF_ATTENTION_ZH.md` 里 §5 的说明)。

- **测试脚本** `test/test_infer.py`:已经写好了完整的对比逻辑——用 HuggingFace `transformers` 跑一遍 `model.generate(...)`,再用你实现的 `llaisys.models.Qwen2` 跑一遍,`--test` 模式下强制 `top_k=1, top_p=1.0, temperature=1.0`(即贪心解码),两边 token 序列必须完全一致(`assert llaisys_tokens == tokens`)。
- **算子层**:`self_attention`、`rope`、`rms_norm`、`swiglu`、`linear`、`embedding`、`add`、`argmax` 都已经在 Assignment #2 里实现完了(见 `docs/SELF_ATTENTION_ZH.md`),可以直接复用,不用重新实现计算逻辑。

### 完全是空的、需要你从头写的部分

| 文件 | 现状 |
|---|---|
| `src/llaisys/models/`(或类似目录) | **不存在**,`llaisysQwen2Model*` 四个 C API 函数完全没有实现 |
| `python/llaisys/libllaisys/models.py`(或类似文件) | **不存在**,没有任何 ctypes 包装能调用上面这四个函数 |
| `python/llaisys/models/qwen2.py` | 只有函数签名和两个 `# TODO` 注释,构造函数和 `generate` 都是空的 |
| `xmake.lua` | `src/llaisys/*.cc` 目前只 glob 了 `src/llaisys/` 一层目录(`ops.cc`/`runtime.cc`/`tensor.cc`),如果新文件放在子目录(比如 `src/llaisys/models/qwen2.cc`)需要检查 glob 规则要不要改成 `src/llaisys/**.cc` 或显式加路径 |

## 3. 要做的工作,按调用链从下往上拆解

### 3.1 C++ 后端:实现模型本体和四个导出函数

新建类似 `src/llaisys/models/qwen2.cc`(或直接 `src/llaisys/models.cc`,取决于你想不想拆子目录),内部至少需要:

1. **`LlaisysQwen2Model` 的具体定义**(头文件里只是前向声明 `struct LlaisysQwen2Model;`,具体成员由你定义),通常要持有:
   - `LlaisysQwen2Meta` 的一份拷贝(层数、head 数、维度这些超参数)。
   - `LlaisysQwen2Weights`,以及背后实际的 `tensor_t`(权重张量本身,在 `Create` 时按 `meta` 里的形状分配好,等 Python 侧再通过 `tensorLoad` 把 safetensors 里的数据拷进来)。
   - **KV Cache**:每层一份 K、V 的缓存张量,形状类似 `[maxseq, nkvh, dh]`,以及一个记录"当前已经缓存了多少 token"的计数器(对应 `self_attention` 里的 `total_len`)。
   - 当前设备类型/编号(`device`、`device_ids`、`ndevice`,这次作业阶段只用管 CPU,`ndevice` 大概率恒为 1,多卡是 Assignment #4 CUDA 阶段以后再考虑的事)。

2. **`llaisysQwen2ModelCreate`**:按 `meta` 分配好所有权重张量和 KV Cache 张量,返回一个 `LlaisysQwen2Model*`。

3. **`llaisysQwen2ModelWeights`**:返回内部持有的 `LlaisysQwen2Weights*`,好让 Python 侧能拿到每个 `llaisysTensor_t` 逐个调用 `tensorLoad` 把 safetensors 数据灌进去。

4. **`llaisysQwen2ModelInfer`**:核心推理函数,输入 `token_ids`(长度 `ntoken`),跑一次前向,返回**下一个 token 的 id**(`int64_t`)。内部要做的事情,對每一层(`nlayer` 层)循环:
   - `embedding`:token id 查表得到输入向量(首次调用时对 `ntoken` 个 token 一起做;后续增量解码时 `ntoken` 通常是 1)。
   - `rms_norm`(pre-attn) → `linear` 算 Q/K/V(注意 QKV 带 bias) → `rope` 对 Q、K 做旋转位置编码 → 把新算出的 K/V 写入 KV Cache 对应位置 → `self_attention`(这里要传 `seqlen`=本次新增 token 数,`total_len`=写入后的缓存总长度,`nkvhead`=`meta.nkvh`,跟你在 Assignment #2 里实现的接口完全对应)→ `linear` 算输出投影 → 残差相加(`add`)。
   - `rms_norm`(pre-mlp) → `linear` 算 gate/up → `swiglu` → `linear` 算 down → 残差相加。
   - 最后一层结束后:`rms_norm`(`out_norm_w`)→ `linear`(`out_embed`)算 logits → `argmax` 取概率最大的 token id 返回。

   `end_token`(`meta.end_token`,即 eos token id)由 Python 侧生成循环里判断是否停止,C 层只管返回下一个 token。

5. **`llaisysQwen2ModelDestroy`**:释放上面分配的所有资源。

6. 在 `xmake.lua` 里把新增的 `.cc` 文件纳入 `llaisys` target 的编译范围(检查现有 `add_files("src/llaisys/*.cc")` 这一行的 glob 模式)。

### 3.2 Python ctypes 包装层

新建 `python/llaisys/libllaisys/models.py`(参考同目录下 `ops.py`/`tensor.py` 的写法),需要:

- 用 `ctypes.Structure` 把 `LlaisysQwen2Meta` 和 `LlaisysQwen2Weights` 各自定义一份对应的 Python 结构体(字段名、类型、顺序要跟 C 头文件严格一致,数组字段用 `POINTER(llaisysTensor_t)`)。
- 给 `llaisysQwen2ModelCreate` / `llaisysQwen2ModelDestroy` / `llaisysQwen2ModelWeights` / `llaisysQwen2ModelInfer` 四个函数分别设置 `argtypes`/`restype`(照抄 `load_ops(lib)` 的模式,写一个 `load_models(lib)` 之类的函数)。
- 在 `python/llaisys/libllaisys/__init__.py` 里调用这个新的 `load_xxx(LIB_LLAISYS)`(参照现有的 `load_ops(LIB_LLAISYS)` 那一行)。

### 3.3 Python 模型类:`python/llaisys/models/qwen2.py`

**构造函数 `__init__`**(第一个 TODO):
1. 读 `model_path` 目录下的 `config.json`,填出一份 `LlaisysQwen2Meta`(`nlayer`←`num_hidden_layers`,`hs`←`hidden_size`,`nh`←`num_attention_heads`,`nkvh`←`num_key_value_heads`,`dh`←`hidden_size/num_attention_heads`,`di`←`intermediate_size`,`maxseq`←`max_position_embeddings`,`voc`←`vocab_size`,`epsilon`←`rms_norm_eps`,`theta`←`rope_theta`,`end_token`←`eos_token_id`)。DeepSeek-R1-Distill-Qwen-1.5B 具体数值见 `SELF_ATTENTION_ZH.md`(28 层、12 head、2 kv head、head_dim=128、theta=10000 等)。
2. 调 `llaisysQwen2ModelCreate` 拿到底层模型指针,再调 `llaisysQwen2ModelWeights` 拿到权重张量的句柄。
3. 遍历 `model_path` 下所有 `*.safetensors` 文件(现有代码已经在做这一步),把每个 tensor 名字映射到 `LlaisysQwen2Weights` 里对应的字段,再用 `tensorLoad` 把数据拷进去。命名映射(标准 HF Qwen2 checkpoint 命名):

   | safetensors 里的名字 | 对应字段 |
   |---|---|
   | `model.embed_tokens.weight` | `in_embed` |
   | `lm_head.weight` | `out_embed`(注意 `tie_word_embeddings=false`,这两个不是同一份权重) |
   | `model.norm.weight` | `out_norm_w` |
   | `model.layers.{i}.input_layernorm.weight` | `attn_norm_w[i]` |
   | `model.layers.{i}.self_attn.q_proj.weight` / `.bias` | `attn_q_w[i]` / `attn_q_b[i]` |
   | `model.layers.{i}.self_attn.k_proj.weight` / `.bias` | `attn_k_w[i]` / `attn_k_b[i]` |
   | `model.layers.{i}.self_attn.v_proj.weight` / `.bias` | `attn_v_w[i]` / `attn_v_b[i]` |
   | `model.layers.{i}.self_attn.o_proj.weight` | `attn_o_w[i]`(无 bias) |
   | `model.layers.{i}.post_attention_layernorm.weight` | `mlp_norm_w[i]` |
   | `model.layers.{i}.mlp.gate_proj.weight` | `mlp_gate_w[i]` |
   | `model.layers.{i}.mlp.up_proj.weight` | `mlp_up_w[i]` |
   | `model.layers.{i}.mlp.down_proj.weight` | `mlp_down_w[i]` |

**`generate` 函数**(第二个 TODO):
1. `inputs`(prompt 对应的 token id 列表)整体喂给 `llaisysQwen2ModelInfer` 做一次 **prefill**(`ntoken = len(inputs)`),拿到第一个新 token。
2. 循环:每次只把**上一步生成的那一个 token** 喂给 `llaisysQwen2ModelInfer`(`ntoken = 1`,增量解码,依赖 KV Cache 记住之前的上下文),拿到下一个 token,直到生成满 `max_new_tokens` 个,或者遇到 `end_token` 提前停止。
3. `--test` 模式下 `top_k=1/top_p=1.0/temperature=1.0` 等价于贪心采样,C 层的 `argmax` 已经是贪心的,所以这个模式下 Python 侧基本不需要额外采样逻辑;如果要支持非 `--test` 模式下的 `top_k`/`top_p`/`temperature` 采样,那部分逻辑需要在 Python 侧或者额外的 C 接口里处理(具体实现方式取决于你想把随机采样放在哪一层)。

## 4. KV Cache:为什么必须做

`README.md` 原文强调"不实现 KV Cache 模型会慢到不可用",原因和你在 Assignment #2 里实现 `self_attention` 时看到的参数是同一件事:

- 没有 KV Cache:每生成一个新 token,都要把**从头到当前位置的所有 token** 重新过一遍 embedding → attention → mlp 全部 28 层,复杂度随生成长度呈平方增长。
- 有 KV Cache:每层的 K、V 一旦算出来就存住,新 token 只需要算它自己的 Q/K/V,新的 K/V 追加进缓存,`self_attention` 用**全部缓存的 K/V**(`total_len` 个)去和**这一个新 token 的 Q**(`seqlen=1`)做注意力——这正是你已经实现好的 `causal_offset = total_len - seqlen` 逻辑在真实推理场景里的用法:prefill 阶段 `causal_offset=0`,增量解码阶段 `causal_offset` 等于已经生成的历史长度。

## 5. 调试建议

- 用 `tensor.debug()`(Python 侧,底层调 `tensorDebug`)在每一步(embedding 之后、每层 attention/mlp 之后、最终 logits 之前)打印张量数据,和 HuggingFace 模型在同样位置的中间结果对比,定位第一个数值开始偏离的地方,比逐字排查整个前向过程高效得多。
- 建议先只跑 1~2 层、固定输入(比如全 0 或固定 token id),确认单层结果对得上,再扩展到全部层数,最后再接上 KV Cache 和多步生成。

## 6. 验收

```bash
python test/test_infer.py --model <本地模型目录> --test
```

`--test` 模式下用贪心解码(`top_k=1`),要求 `llaisys` 生成的 `tokens` 和 HuggingFace `transformers` 生成的 `tokens` 完全一致(`assert llaisys_tokens == tokens`),打印 `Test passed!` 即算通过。之后按 README 里的说明 commit + push,CI 里 `Assignment-3` 那一步(`python test/test_infer.py --test`)应该跑绿——但注意 CI 环境要能拿到模型权重(`load_hf_model` 在没传 `--model` 时会用 `huggingface_hub.snapshot_download` 自动下载),如果 CI 里没联网或没配好缓存,这一步可能需要额外配置。
