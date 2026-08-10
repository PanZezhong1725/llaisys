#!/usr/bin/env bash
# 全新天数（Iluvatar）实例的一键 clone + build + 验证脚本。
# 前提：跟上次用的是同一个课程镜像/模板（corex SDK、天数版 PyTorch 已经预装好，
# 只是这次是全新开的机器，代码、模型权重、xmake 都没有）。
#
# 用法（在新实例的任意目录下）：
#   curl -fsSL https://gitee.com/tang-jinchi/llaisys/raw/main/scripts/iluvatar/bootstrap_and_verify.sh -o bootstrap_and_verify.sh
#   bash bootstrap_and_verify.sh
# 或者先手动 clone 仓库、cd 进去，再 bash scripts/iluvatar/bootstrap_and_verify.sh
#
# 把完整输出贴回来。

set -u
REPO_URL="https://gitee.com/tang-jinchi/llaisys.git"
REPO_DIR="$HOME/llaisys"

echo "===================================================="
echo "[0] 环境自检：确认这是预期的天数镜像"
echo "===================================================="
echo "-- clang++ (真正的编译器，不是那个假 nvcc) --"
/usr/local/corex/bin/clang++ --version 2>&1
echo
echo "-- corex SDK 版本目录 --"
ls -d /usr/local/corex-*/ 2>&1
echo
echo "-- 天数版 PyTorch（应该已经预装，脚本不会碰它）--"
python3 -c "import torch; print('torch', torch.__version__); print('torch.cuda.is_available():', torch.cuda.is_available())" 2>&1

echo
echo "===================================================="
echo "[1] xmake（这次是全新机器，大概率没装）"
echo "===================================================="
export XMAKE_ROOT=y
if command -v xmake >/dev/null 2>&1; then
    echo "xmake 已存在: $(xmake --version | head -1)"
else
    echo "安装 xmake（注意不带 --branch 参数，装脚本对这个 flag 解析有 bug）..."
    curl -fsSL https://xmake.io/shget.text | bash
    export PATH="$HOME/.local/bin:$PATH"
    echo "'export XMAKE_ROOT=y' 和 PATH 已在这次 shell 里生效；" \
         "如果重开一个 shell，记得先手动 export 一遍或者把它们写进 ~/.bashrc。"
fi
xmake --version 2>&1

echo
echo "===================================================="
echo "[2] clone / 更新仓库到 $REPO_DIR"
echo "===================================================="
if [ -d "$REPO_DIR/.git" ]; then
    echo "仓库已存在，git pull"
    (cd "$REPO_DIR" && git pull 2>&1)
else
    git clone "$REPO_URL" "$REPO_DIR" 2>&1
fi
cd "$REPO_DIR" || { echo ">>> cd 失败，终止"; exit 1; }
git log --oneline -3
echo "(期望最新 commit 是 fc46265 或更新)"

echo
echo "===================================================="
echo "[3] Python 依赖检查（只装跟平台无关的库，不碰 torch）"
echo "===================================================="
for pkg_import in "transformers:transformers" "huggingface_hub:huggingface_hub" "safetensors:safetensors" "ml_dtypes:ml_dtypes"; do
    mod="${pkg_import%%:*}"
    pipname="${pkg_import##*:}"
    if python3 -c "import ${mod}" >/dev/null 2>&1; then
        echo "${mod}: 已安装"
    else
        echo "${mod}: 缺失，尝试 pip install ${pipname}"
        pip install "${pipname}" 2>&1
    fi
done

echo
echo "===================================================="
echo "[4] 确保 libcudadevrt 桩库存在"
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
echo "[5] xmake 配置 + 完整构建 (iluvatar-gpu=y)"
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
echo "[6] xmake install + pip 可编辑安装"
echo "===================================================="
xmake install 2>&1
pip install -e ./python 2>&1

echo
echo "===================================================="
echo "[7] 8 个算子回归"
echo "===================================================="
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    echo "---- test/ops/${op}.py --device iluvatar ----"
    python3 test/ops/${op}.py --device iluvatar 2>&1
    echo "---- exit code: $? ----"
done

echo
echo "===================================================="
echo "[8] 查找本地模型权重（没有的话让脚本自己从 HuggingFace 下载）"
echo "===================================================="
MODEL_DIR=""
for candidate in \
    "$REPO_DIR/DeepSeek-R1-Distill-Qwen-1.5B" \
    "$HOME/DeepSeek-R1-Distill-Qwen-1.5B" \
    /root/DeepSeek-R1-Distill-Qwen-1.5B \
    /data/DeepSeek-R1-Distill-Qwen-1.5B \
    /models/DeepSeek-R1-Distill-Qwen-1.5B
do
    if [ -f "$candidate/config.json" ] && [ -f "$candidate/model.safetensors" ]; then
        MODEL_DIR="$candidate"
        break
    fi
done
if [ -n "$MODEL_DIR" ]; then
    echo "找到本地模型目录: $MODEL_DIR"
else
    echo "没找到本地权重。test_infer.py 在不传 --model 时会自动从 HuggingFace" \
         "(deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, 约 3.5GB) 下载到本地 HF 缓存并复用同一份路径，" \
         "前提是这台机器能访问外网——接下来这步会自动尝试。"
fi

echo
echo "===================================================="
echo "[9] 完整端到端推理 test_infer.py --device iluvatar --test"
echo "===================================================="
if [ -n "$MODEL_DIR" ]; then
    python3 test/test_infer.py --model "$MODEL_DIR" --device iluvatar --test 2>&1
else
    python3 test/test_infer.py --device iluvatar --test 2>&1
fi
echo "--- exit code: $? ---"

echo
echo "===================================================="
echo "全部结束，请把以上完整输出贴回去"
echo "===================================================="
