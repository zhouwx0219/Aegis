#!/usr/bin/env bash
# 编译 cast_core Python 扩展（C++ 事务内核 + pybind11 桥接）。
# 用法: bash build.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PYINC=$(python3 -m pybind11 --includes)
EXT=$(python3-config --extension-suffix)

echo "compiling cast_core${EXT} ..."
g++ -O3 -std=c++17 -shared -fPIC \
  -I "$HERE" \
  ${PYINC} \
  core/bindings/cast_bindings.cpp \
  -o "cast_core${EXT}"

echo "built: $HERE/cast_core${EXT}"
