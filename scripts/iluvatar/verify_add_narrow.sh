#!/usr/bin/env bash
# 在天数智芯远程机器上运行，验证 iluvatar 工具链是否真的能编译 llaisys 代码。
# 把这个脚本拷到远程机器上（比如 scp 或直接粘贴内容），cd 到 llaisys 仓库根目录后执行：
#   bash verify_iluvatar.sh
# 然后把全部输出贴回来。

set -u  # 不用 -e，因为我们要收集失败信息而不是中途退出

REPO_DIR="$(pwd)"
echo "===================================================="
echo "[0] 基本信息"
echo "===================================================="
echo "repo dir: $REPO_DIR"
if [ ! -d "$REPO_DIR/src/ops/add/iluvatar" ]; then
    echo "!! 当前目录看起来不是 llaisys 仓库根目录（找不到 src/ops/add/iluvatar），请 cd 到正确目录后重跑"
    exit 1
fi

echo
echo "===================================================="
echo "[1] 工具链 / SDK 路径检查"
echo "===================================================="
for p in /usr/local/corex/bin/clang++ /usr/local/corex/bin/nvcc /usr/local/corex-4.4.0/lib64 /usr/local/corex-4.4.0/include/cudnn.h; do
    if [ -e "$p" ]; then
        echo "OK   $p"
    else
        echo "MISSING $p"
    fi
done

echo
echo "clang++ 真实身份 (file):"
file /usr/local/corex/bin/clang++ 2>&1
echo
echo "clang++ --version:"
/usr/local/corex/bin/clang++ --version 2>&1

echo
echo "cudnn 版本 (来自 cudnn.h):"
grep -E "CUDNN_MAJOR|CUDNN_MINOR|CUDNN_PATCHLEVEL" /usr/local/corex-4.4.0/include/cudnn.h 2>&1

echo
echo "libcublas / libcudnn 是否在 lib64 里:"
ls /usr/local/corex-4.4.0/lib64/ 2>&1 | grep -iE "cublas|cudnn"

echo
echo "GPU 是否可见 (torch):"
python3 -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1

echo
echo "===================================================="
echo "[2] 独立编译测试：直接用 clang++ 编译 add_iluvatar.cu（不经过 xmake）"
echo "===================================================="
TESTOUT="/tmp/add_iluvatar_test.o"
rm -f "$TESTOUT"

CMD=(/usr/local/corex/bin/clang++
    -x ivcore
    -std=c++17
    -fPIC
    -c "$REPO_DIR/src/ops/add/iluvatar/add_iluvatar.cu"
    -I "$REPO_DIR/src"
    -I "$REPO_DIR/include"
    -o "$TESTOUT")

echo "执行命令:"
printf ' %q' "${CMD[@]}"
echo
echo "---- 输出 ----"
"${CMD[@]}"
STATUS=$?
echo "---- exit code: $STATUS ----"

if [ -f "$TESTOUT" ]; then
    echo "产物存在，大小/类型："
    ls -la "$TESTOUT"
    file "$TESTOUT"
    if [ -s "$TESTOUT" ]; then
        echo ">>> 看起来是真实的非空目标文件"
    else
        echo ">>> 警告：产物是空文件！"
    fi
else
    echo ">>> 没有产物文件，编译失败"
fi

echo
echo "===================================================="
echo "[3] xmake 真实构建测试：只编 add 这一个 iluvatar 算子"
echo "===================================================="
echo "临时把 xmake/iluvatar.lua 里 llaisys-ops-iluvatar 的 add_files 收窄成只有 add_iluvatar.cu"

cp xmake/iluvatar.lua /tmp/iluvatar.lua.bak

python3 - "$REPO_DIR/xmake/iluvatar.lua" <<'PYEOF'
import sys, re
path = sys.argv[1]
with open(path) as f:
    content = f.read()

target_re = re.compile(r'(target\("llaisys-ops-iluvatar"\).*?)add_files\("\.\./src/ops/\*/iluvatar/\*\.cu"\)', re.S)
new_content, n = target_re.subn(
    r'\1add_files("../src/ops/add/iluvatar/add_iluvatar.cu")',
    content
)
if n != 1:
    print("!! 没能找到预期的 add_files 行，脚本假设与当前 iluvatar.lua 内容不匹配，跳过第 3 步", file=sys.stderr)
    sys.exit(1)

with open(path, "w") as f:
    f.write(new_content)
print("已临时收窄 add_files")
PYEOF

if [ $? -eq 0 ]; then
    echo
    echo "--- xmake f -c --iluvatar-gpu=y ---"
    xmake f -c --iluvatar-gpu=y 2>&1
    echo
    echo "--- xmake build -v llaisys-ops-iluvatar ---"
    xmake build -v llaisys-ops-iluvatar 2>&1
    XBUILD_STATUS=$?
    echo "--- exit code: $XBUILD_STATUS ---"

    echo
    echo "查找生成的 .o 产物："
    find build -iname "*add_iluvatar*" 2>&1
    find build -iname "*add_iluvatar*" -exec file {} \; 2>&1
else
    echo "跳过 xmake 构建测试（收窄 add_files 失败）"
fi

echo
echo "===================================================="
echo "[4] 恢复 xmake/iluvatar.lua"
echo "===================================================="
cp /tmp/iluvatar.lua.bak xmake/iluvatar.lua
git -C "$REPO_DIR" diff --stat xmake/iluvatar.lua
echo "已恢复原始 xmake/iluvatar.lua（如果 git diff 还有差异，说明恢复有问题，检查 /tmp/iluvatar.lua.bak）"

echo
echo "===================================================="
echo "全部测试结束，请把以上完整输出贴回去"
echo "===================================================="
