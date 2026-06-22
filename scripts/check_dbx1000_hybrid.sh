#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DBX="$ROOT/third_party/dbx1000"
CFG="$DBX/config.h"
BAK="$(mktemp)"

cleanup() {
  cp "$BAK" "$CFG"
  rm -f "$BAK"
  make -C "$DBX" clean >/dev/null 2>&1 || true
}
trap cleanup EXIT

cp "$CFG" "$BAK"
perl -0pi -e 's/#define\s+CC_ALG\s+\w+/#define CC_ALG HYBRID/' "$CFG"

echo "[dbx1000-hybrid] build DBx1000 with CC_ALG=HYBRID"
make -C "$DBX" clean >/dev/null
make -C "$DBX" -j"${JOBS:-2}" >/tmp/dbx1000_hybrid_build.log
echo "[dbx1000-hybrid] OK"
