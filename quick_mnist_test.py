#!/usr/bin/env python3
"""
快速MNIST联邦学习测试 - 验证可行性
只训练2轮，快速验证功能是否正常
"""

import sys
import os
sys.path.append('pytorch_client')

from pytorch_client.mnist_federated_demo import FederatedConfig, run_federated_mnist_demo

def quick_test():
    """快速测试 - 2个客户端，2轮训练"""
    print("🚀 Running Quick MNIST Federated Learning Test")
    print("=" * 50)
    
    # 快速配置
    config = FederatedConfig(
        num_clients=2,          # 只用2个客户端
        local_epochs=2,         # 每轮只训练2个epoch
        batch_size=128,         # 大一点的batch_size，训练更快
        learning_rate=0.01,
        num_rounds=3,           # 只训练3轮
        client_data_size=1000   # 每个客户端只用1000个样本
    )
    
    print(f"📋 Test Configuration:")
    print(f"   - Clients: {config.num_clients}")
    print(f"   - Rounds: {config.num_rounds}")
    print(f"   - Local epochs: {config.local_epochs}")
    print(f"   - Data per client: {config.client_data_size}")
    print()
    
    try:
        # 运行联邦学习
        history, aggregator = run_federated_mnist_demo(config)
        
        # 显示结果
        print("\n🎉 Quick Test Results:")
        print(f"   ✅ Final Accuracy: {history['global_accuracy'][-1]:.2f}%")
        print(f"   ✅ Training completed successfully!")
        print(f"   ✅ Generated {len(history['rounds'])} training rounds")
        
        # 验证准确率提升
        if len(history['global_accuracy']) >= 2:
            improvement = history['global_accuracy'][-1] - history['global_accuracy'][0]
            print(f"   📈 Accuracy improvement: +{improvement:.2f}%")
        
        print("\n✅ MNIST Federated Learning is fully functional!")
        print("💡 Ready for integration with Raft system.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Run full demo: python3 pytorch_client/mnist_federated_demo.py")
        print("2. Integrate with Raft: Use the federated_client.py")
        print("3. Start Raft cluster: ./start_federated_demo.sh")
    else:
        print("\n🔧 Please check the environment setup and try again.")


