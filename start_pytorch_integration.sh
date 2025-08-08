#!/bin/bash

echo "=== Raft-PyTorch集成测试启动脚本 ==="

# 检查Python环境
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查PyTorch
echo "检查PyTorch安装..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" || {
    echo "❌ PyTorch未安装，请运行: pip install torch"
    exit 1
}

# 安装依赖
echo "安装Python依赖..."
cd pytorch_client
pip install -r requirements.txt || {
    echo "❌ 依赖安装失败"
    exit 1
}

# 运行测试
echo "开始运行集成测试..."
python3 test_integration.py

echo "=== 测试完成 ===" 