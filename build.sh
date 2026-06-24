#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CXX="${CXX:-g++}"
PYINC="$(python3 -m pybind11 --includes)"
EXT="$(python3-config --extension-suffix)"

echo "compiling cast_core${EXT} ..."
# PYINC intentionally expands to multiple compiler arguments.
# shellcheck disable=SC2086
"$CXX" -O3 -std=c++17 -Wall -Wextra -shared -fPIC \
  -DASTRA_DBX1000_EMBEDDED=1 -DNOGRAPHITE=1 \
  -I "$ROOT" \
  -I "$ROOT/third_party/dbx1000" \
  -I "$ROOT/third_party/dbx1000/storage" \
  -I "$ROOT/third_party/dbx1000/system" \
  -I "$ROOT/third_party/dbx1000/concurrency_control" \
  ${PYINC} \
  core/bindings/cast_bindings.cpp \
  core/storage/dbx1000_versioned_kv.cpp \
  third_party/dbx1000/storage/catalog.cpp \
  third_party/dbx1000/storage/table.cpp \
  third_party/dbx1000/storage/row.cpp \
  third_party/dbx1000/storage/index_hash.cpp \
  -pthread \
  -o "cast_core${EXT}"

echo "built: $ROOT/cast_core${EXT}"
