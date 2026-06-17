#!/usr/bin/env bash
# 准备真实 VitaBench 环境（一次性）。VM /tmp 会在会话间重置，需要时重跑。
set -e
VB=${1:-/tmp/vb}
if [ ! -d "$VB/.git" ]; then
  git clone --depth 1 https://github.com/meituan-longcat/vitabench "$VB"
fi
pip install -e "$VB" --break-system-packages -q || pip install -e "$VB" --break-system-packages -q
echo "VitaBench ready at $VB"
