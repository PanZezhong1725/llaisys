#!/usr/bin/env bash
# 端到端推理验证：确认自 self_attention 去掉 cuDNN 加速之后（commit 171bb0b/a72386d）,
# 完整模型推理在天数机器上没有回归——这一步之前只做过 8 个算子的单测，没跑过完整的
# test_infer.py --device iluvatar --test。
#
# 用法: cd 到 llaisys 仓库根目录，
#   bash scripts/iluvatar/verify_infer.sh
# 把完整输出贴回来。

set -u
REPO_DIR="$(pwd)"

echo "===================================================="
echo "[0] 拉最新代码并确认版本"
echo "===================================================="
git pull 2>&1
git log --oneline -3
echo "(期望最新 commit 是 04f5738 或更新)"

echo
echo "===================================================="
echo "[1] 确保 libcudadevrt 桩库存在"
echo "===================================================="
STUB_DIR="/usr/local/corex-4.4.0/lib64"
STUB_LIB="${STUB_DIR}/libcudadevrt.a"
if [ -f "$STUB_LIB" ]; then
    echo "已存在: $STUB_LIB"
else
    echo "创建空桩库: $STUB_LIB"
    ar rcs "$STUB_LIB"
fi

echo
echo "===================================================="
echo "[2] xmake 配置 + 完整构建 (iluvatar-gpu=y)"
echo "===================================================="
xmake f -c --iluvatar-gpu=y 2>&1
echo
xmake build -v llaisys 2>&1
BUILD_STATUS=$?
echo "--- build exit code: $BUILD_STATUS ---"
if [ $BUILD_STATUS -ne 0 ]; then
    echo ">>> 构建失败，后面步骤大概率也会失败，但继续跑方便一次性看到所有问题"
fi

echo
echo "===================================================="
echo "[3] xmake install + pip 可编辑安装"
echo "===================================================="
xmake install 2>&1
pip install -e ./python 2>&1

echo
echo "===================================================="
echo "[4] 8 个算子回归（确认 cuDNN 移除 + 注释清理没有引入问题）"
echo "===================================================="
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    echo "---- test/ops/${op}.py --device iluvatar ----"
    python3 test/ops/${op}.py --device iluvatar 2>&1
    echo "---- exit code: $? ----"
done

echo
echo "===================================================="
echo "[5] 查找本地模型权重（DeepSeek-R1-Distill-Qwen-1.5B）"
echo "===================================================="
MODEL_DIR=""
for candidate in \
    "$REPO_DIR/DeepSeek-R1-Distill-Qwen-1.5B" \
    "$HOME/DeepSeek-R1-Distill-Qwen-1.5B" \
    /root/DeepSeek-R1-Distill-Qwen-1.5B \
    /data/DeepSeek-R1-Distill-Qwen-1.5B \
    /data/*/DeepSeek-R1-Distill-Qwen-1.5B \
    /models/DeepSeek-R1-Distill-Qwen-1.5B
do
    if [ -f "$candidate/config.json" ] && [ -f "$candidate/model.safetensors" ]; then
        MODEL_DIR="$candidate"
        break
    fi
done
if [ -z "$MODEL_DIR" ]; then
    echo "没在常见路径下找到，扩大范围用 find 搜（可能会慢一点）..."
    FOUND=$(find / -maxdepth 6 -iname "config.json" -path "*DeepSeek*" 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        MODEL_DIR=$(dirname "$FOUND")
    fi
fi

if [ -n "$MODEL_DIR" ]; then
    echo "找到模型目录: $MODEL_DIR"
    ls -la "$MODEL_DIR"
else
    echo ">>> 没找到 DeepSeek-R1-Distill-Qwen-1.5B 的权重目录。"
    echo ">>> 需要包含 config.json / tokenizer 相关文件 / model.safetensors（约 3.5GB）的目录，"
    echo ">>> 放到这台机器上（比如 repo 根目录下），然后重跑这个脚本，或手动执行："
    echo ">>>   python3 test/test_infer.py --model <目录路径> --device iluvatar --test"
fi

echo
echo "===================================================="
echo "[6] 完整端到端推理 test_infer.py --device iluvatar --test"
echo "===================================================="
if [ -n "$MODEL_DIR" ]; then
    python3 test/test_infer.py --model "$MODEL_DIR" --device iluvatar --test 2>&1
    echo "--- exit code: $? ---"
else
    echo "跳过（第 [5] 步没找到模型权重）"
fi

echo
echo "===================================================="
echo "全部结束，请把以上完整输出贴回去"
echo "===================================================="
