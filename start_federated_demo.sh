#!/bin/bash

# TrustedFL 联邦学习演示启动脚本
# 基于Raft共识的可信联邦学习平台

echo "🚀 Starting TrustedFL Federated Learning Demo"
echo "=============================================="

# 检查Go环境
if ! command -v go &> /dev/null; then
    echo "❌ Go is not installed. Please install Go first."
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first."
    exit 1
fi

# 进入项目目录
cd "$(dirname "$0")"
echo "📁 Working directory: $(pwd)"

# 安装Python依赖
echo "📦 Installing Python dependencies..."
pip3 install torch torchvision numpy requests > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Warning: Failed to install some Python dependencies"
fi

# 构建Go项目
echo "🔨 Building Go project..."
cd src
go mod tidy > /dev/null 2>&1

# 创建日志目录
mkdir -p ../logs

# 启动Raft集群 (3个节点)
echo "🌐 Starting Raft cluster..."

# 节点0 (Leader候选)
echo "  Starting Node 0 (HTTP: 8080)..."
go run main/diskvd.go 0 > ../logs/node0.log 2>&1 &
NODE0_PID=$!

# 节点1
echo "  Starting Node 1 (HTTP: 8081)..."
go run main/diskvd.go 1 > ../logs/node1.log 2>&1 &
NODE1_PID=$!

# 节点2  
echo "  Starting Node 2 (HTTP: 8082)..."
go run main/diskvd.go 2 > ../logs/node2.log 2>&1 &
NODE2_PID=$!

# 等待Raft集群启动
echo "⏳ Waiting for Raft cluster to initialize..."
sleep 5

# 检查Raft节点状态
echo "🔍 Checking Raft cluster status..."
for port in 8080 8081 8082; do
    if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
        echo "✅ Node on port $port is healthy"
    else
        echo "❌ Node on port $port is not responding"
    fi
done

# 启动联邦学习客户端演示
echo ""
echo "🧠 Starting Federated Learning clients..."
cd ../pytorch_client

# 启动医疗场景演示
echo "🏥 Medical Scenario Demo:"
echo "  - Hospital A (Cardiology)"
echo "  - Hospital B (Neurology)" 
echo "  - Hospital C (General)"

# 医院A
echo "  Starting Hospital A..."
python3 federated_client.py --scenario medical --client-id hospital_a_cardiology --rounds 3 --wait-time 15 > ../logs/hospital_a.log 2>&1 &
HOSPITAL_A_PID=$!

# 医院B
echo "  Starting Hospital B..."
python3 federated_client.py --scenario medical --client-id hospital_b_neurology --rounds 3 --wait-time 15 > ../logs/hospital_b.log 2>&1 &
HOSPITAL_B_PID=$!

# 医院C
echo "  Starting Hospital C..."
python3 federated_client.py --scenario medical --client-id hospital_c_general --rounds 3 --wait-time 15 > ../logs/hospital_c.log 2>&1 &
HOSPITAL_C_PID=$!

echo ""
echo "🎉 Demo started successfully!"
echo "📊 Monitor progress:"
echo "  - Raft logs: tail -f logs/node*.log"
echo "  - Client logs: tail -f logs/hospital_*.log"
echo "  - Health check: curl http://localhost:8080/health"
echo "  - Raft status: curl http://localhost:8080/raft/status"
echo ""

# 创建停止脚本
cat > stop_demo.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping TrustedFL Demo..."

# 停止所有相关进程
pkill -f "go run main/diskvd.go"
pkill -f "federated_client.py" 

echo "✅ Demo stopped"
EOF

chmod +x stop_demo.sh

echo "💡 Tips:"
echo "  - Run './stop_demo.sh' to stop all processes"
echo "  - Check logs/hospital_*.log for training progress"
echo "  - Visit http://localhost:8080 for API endpoints"
echo ""
echo "🔄 Training will run for 3 rounds, ~5 minutes total"

# 函数：清理并退出
cleanup() {
    echo ""
    echo "🛑 Shutting down demo..."
    kill $NODE0_PID $NODE1_PID $NODE2_PID 2>/dev/null
    kill $HOSPITAL_A_PID $HOSPITAL_B_PID $HOSPITAL_C_PID 2>/dev/null
    echo "✅ Cleanup completed"
    exit 0
}

# 捕获中断信号
trap cleanup SIGINT SIGTERM

# 等待训练完成或用户中断
echo "⌨️  Press Ctrl+C to stop the demo"
wait $HOSPITAL_A_PID $HOSPITAL_B_PID $HOSPITAL_C_PID

echo ""
echo "🎊 Federated Learning Demo Completed!"
echo "📈 Check the logs for training results and model convergence"

# 清理进程
cleanup
