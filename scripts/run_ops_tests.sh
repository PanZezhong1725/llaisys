#!/bin/bash
# Run all NVIDIA operator tests.
export PATH=/usr/local/cuda/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/data/llaisys-26s/lib
export PYTHONPATH=/data/llaisys-26s/python
cd /data/llaisys-26s

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    echo "===== $op ====="
    python test/ops/$op.py --device nvidia 2>&1 | tail -15
    echo "exit code: ${PIPESTATUS[0]}"
done
