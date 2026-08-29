#!/bin/bash
args=()
for a in "$@"; do
  if [ "$a" = "-m64" ] || [ "$a" = "-s" ]; then
    continue
  fi
  args+=("$a")
done
exec /opt/maca-3.5.3/mxgpu_llvm/bin/mxcc "${args[@]}"
