#!/usr/bin/env python3
"""
立即可运行的联邦学习演示
展示项目的核心技术价值
"""

import sys
import os
sys.path.append('pytorch_client')

def main():
    print("🎯 TrustedFL 核心技术演示")
    print("=" * 50)
    print("📋 演示内容：")
    print("   ✅ MNIST联邦学习算法")
    print("   ✅ 3个客户端协作训练")
    print("   ✅ 非IID数据分布模拟")
    print("   ✅ 模型聚合和评估")
    print()
    print("⏱️  预计用时：2-3分钟")
    print("📊 将生成训练曲线图")
    print()
    
    choice = input("🚀 是否开始演示？(y/n): ").lower().strip()
    if choice == 'y' or choice == 'yes':
        from pytorch_client.mnist_federated_demo import FederatedConfig, run_federated_mnist_demo
        
        # 快速配置
        config = FederatedConfig(
            num_clients=3,
            local_epochs=2,
            batch_size=128,
            learning_rate=0.01,
            num_rounds=5,
            client_data_size=1000
        )
        
        print("\n🔄 开始训练...")
        history, aggregator = run_federated_mnist_demo(config)
        
        print(f"\n🎉 演示完成！")
        print(f"   📈 最终准确率: {history['global_accuracy'][-1]:.2f}%")
        print(f"   📊 训练曲线已保存为 federated_mnist_results.png")
        print(f"   📄 Raft日志已保存为 simulated_raft_logs.json")
    else:
        print("👋 演示取消")

if __name__ == "__main__":
    main()


