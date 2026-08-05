#!/usr/bin/env bash
# 第三阶段验证：派发层(op.cpp / runtime_api.cpp / device_resource.cpp)、
# xmake 聚合 deps、测试脚本 --device 选项都已经接好了。
# 这一步做完整的 `xmake build llaisys --iluvatar-gpu=y`（不再是单独编某个 target），
# 然后 `xmake install` 把 .so 拷进 python 包，再逐个跑 8 个算子的
# `test/ops/*.py --device iluvatar`，最后跑 test_runtime.py。
#
# 用法: cd 到 llaisys 仓库根目录（先 git pull 到最新代码），
#   bash scripts/iluvatar/verify_ops.sh
# 把完整输出贴回来。

set -u
REPO_DIR="$(pwd)"

echo "===================================================="
echo "[0] 确认代码版本"
echo "===================================================="
git log --oneline -3

echo
echo "===================================================="
echo "[1] 确保 libcudadevrt 桩库存在"
echo "===================================================="
# 天数智芯的 corex SDK 没有 libcudadevrt.a（用于 relocatable device code 链接），
# 但我们两个 target 都设了 cuda.rdc=false，实际功能上用不到这个库——
# xmake 内置的 cuda 规则却无条件在链接命令里加 -lcudadevrt，不管 rdc 是否为 false。
# 建一个空的桩静态库满足链接器即可，不会被真正用到任何符号。
STUB_DIR="/usr/local/corex-4.4.0/lib64"
STUB_LIB="${STUB_DIR}/libcudadevrt.a"
if [ -f "$STUB_LIB" ]; then
    echo "已存在: $STUB_LIB"
    file "$STUB_LIB"
else
    echo "创建空桩库: $STUB_LIB"
    ar rcs "$STUB_LIB"
    ls -la "$STUB_LIB"
    file "$STUB_LIB"
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
    echo ">>> 构建失败，后面的测试大概率也会失败，但还是继续跑，方便一次性看到所有问题"
fi

echo
echo "===================================================="
echo "[3] xmake install（把 .so 拷进 python 包）"
echo "===================================================="
xmake install 2>&1

echo
echo "===================================================="
echo "[4] 逐个跑 8 个算子的 --device iluvatar 测试"
echo "===================================================="
for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    echo "---- test/ops/${op}.py --device iluvatar ----"
    python3 test/ops/${op}.py --device iluvatar 2>&1
    echo "---- exit code: $? ----"
    echo
done

echo
echo "===================================================="
echo "[5] test_runtime.py --device iluvatar"
echo "===================================================="
python3 test/test_runtime.py --device iluvatar 2>&1
echo "--- exit code: $? ---"

echo
echo "===================================================="
echo "全部测试结束，请把以上完整输出贴回去"
echo "===================================================="
