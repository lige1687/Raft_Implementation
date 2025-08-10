#!/bin/bash

set -e

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "=== 启动Go HTTP服务 ==="
cd "$ROOT_DIR/src/main"
# 构建并后台运行
GO111MODULE=on go build -o raft_server || { echo "构建失败"; exit 1; }
./raft_server -id=0 -port=8080 -lr=0.001 >/tmp/raft_http_server.log 2>&1 &
SERVER_PID=$!
echo "Go服务PID: $SERVER_PID"

# 健康等待
sleep 1

# 运行Python集成测试
echo "=== 运行Python集成测试（HTTP） ==="
cd "$ROOT_DIR/pytorch_client"
python3 -m pip install -r requirements.txt
python3 test_integration.py

# 结束Go服务
kill $SERVER_PID || true
wait $SERVER_PID 2>/dev/null || true

echo "=== 全流程结束 ===" 