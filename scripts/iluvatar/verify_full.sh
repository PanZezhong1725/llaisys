#!/usr/bin/env bash
# 第二阶段验证：add 算子已经在 xmake 下编译通过（含 -std=c++17 修复）。
# 这一步不再收窄 add_files，直接用 xmake/iluvatar.lua 里已有的完整 glob
# （../src/ops/*/iluvatar/*.cu 和 ../src/device/iluvatar/*.cu），
# 编译全部 9 个算子 + device-iluvatar，看哪些算子会因为平台差异报错。
#
# 用法: cd 到 llaisys 仓库根目录（先 git pull 到带 -std=c++17 修复的最新代码），
#   bash verify_iluvatar_full.sh
# 把完整输出贴回来。

set -u
REPO_DIR="$(pwd)"

echo "===================================================="
echo "[0] 确认代码是最新的（应该能看到 -std=c++17 那次提交）"
echo "===================================================="
git log --oneline -5
echo
grep -n "std=c++17" xmake/iluvatar.lua

echo
echo "===================================================="
echo "[1] xmake 配置 (iluvatar-gpu=y)"
echo "===================================================="
xmake f -c --iluvatar-gpu=y 2>&1

echo
echo "===================================================="
echo "[2] 构建 llaisys-device-iluvatar（全部文件，不收窄）"
echo "===================================================="
xmake build -v llaisys-device-iluvatar 2>&1
echo "--- exit code: $? ---"

echo
echo "===================================================="
echo "[3] 构建 llaisys-ops-iluvatar（全部 9 个算子，不收窄）"
echo "===================================================="
xmake build -v llaisys-ops-iluvatar 2>&1
echo "--- exit code: $? ---"

echo
echo "===================================================="
echo "[4] 产物检查"
echo "===================================================="
find build -iname "*iluvatar*.o" 2>&1
echo
find build -iname "libllaisys-*iluvatar*.a" -exec file {} \; 2>&1

echo
echo "===================================================="
echo "全部测试结束，请把以上完整输出贴回去"
echo "===================================================="
