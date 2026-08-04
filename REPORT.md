# LLAISYS Assignment #4 报告（草稿）

## 1. 复现步骤

- 环境：TODO（GPU 型号、CUDA 版本、cuDNN 版本）
- 编译：

  ```bash
  xmake f --nv-gpu=y
  xmake
  xmake install
  pip install -e ./python
  ```

- 测试：

  ```bash
  python test/ops/<op>.py --device nvidia
  python test/test_infer.py --model <dir_path> --test --device nvidia
  ```

## 2. 结果记录

| 算子 | 状态 |
| --- | --- |
| add | TODO |
| embedding | TODO |
| argmax | TODO |
| rope | TODO |
| linear | TODO |
| swiglu | TODO |
| rms_norm | TODO |
| self_attention | TODO |

- Assignment #3 端到端推理（`test_infer.py --test --device nvidia`）：TODO

## 3. 支持平台与状态

| 平台 | 状态 |
| --- | --- |
| Nvidia | TODO |
| （第二个平台，待定） | TODO |
