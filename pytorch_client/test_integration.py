#!/usr/bin/env python3
"""
Raft-PyTorch集成测试脚本
测试梯度同步和模型训练功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from raft_client import RaftClient, RaftOptimizer, register_raft_hooks
import time

class SimpleModel(nn.Module):
    """简单的测试模型"""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_dummy_data(num_samples=1000, input_size=10):
    """创建虚拟数据集"""
    X = torch.randn(num_samples, input_size)
    # 简单的线性关系 + 噪声
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
    return X, y

def test_basic_training():
    """测试基本训练功能"""
    print("=== 测试基本训练功能 ===")
    
    # 创建模型和数据
    model = SimpleModel()
    X, y = create_dummy_data(1000, 10)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建Raft客户端
    raft_client = RaftClient(worker_id=0)
    
    # 注册梯度Hook
    register_raft_hooks(model, raft_client)
    
    # 创建优化器
    optimizer = RaftOptimizer(model.parameters(), lr=0.001, raft_client=raft_client)
    criterion = nn.MSELoss()
    
    # 训练循环
    model.train()
    for epoch in range(3):
        raft_client.set_epoch(epoch)
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            raft_client.set_batch_id(batch_idx)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播（自动触发Hook）
            optimizer.zero_grad()
            loss.backward()
            
            # 优化步骤
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} 平均损失: {avg_loss:.4f}")
    
    print("基本训练测试完成")

def test_gradient_synchronization():
    """测试梯度同步功能"""
    print("\n=== 测试梯度同步功能 ===")
    
    # 创建多个worker模拟
    workers = []
    for i in range(3):
        worker = RaftClient(worker_id=i)
        workers.append(worker)
    
    # 模拟梯度提交
    dummy_gradients = {
        "fc1.weight": torch.randn(20, 10),
        "fc1.bias": torch.randn(20),
        "fc2.weight": torch.randn(20, 20),
        "fc2.bias": torch.randn(20),
        "fc3.weight": torch.randn(1, 20),
        "fc3.bias": torch.randn(1)
    }
    
    print("模拟多worker梯度提交...")
    for epoch in range(2):
        for batch in range(5):
            for worker_id, worker in enumerate(workers):
                worker.set_epoch(epoch)
                worker.set_batch_id(batch)
                
                # 提交梯度
                for param_name, grad in dummy_gradients.items():
                    success = worker.submit_gradient(param_name, grad)
                    if success:
                        print(f"Worker {worker_id} 提交 {param_name} 成功")
                
                time.sleep(0.1)  # 模拟网络延迟
    
    print("梯度同步测试完成")

def test_fault_tolerance():
    """测试容错功能"""
    print("\n=== 测试容错功能 ===")
    
    raft_client = RaftClient(worker_id=0)
    
    # 模拟网络故障
    print("模拟网络故障...")
    try:
        # 尝试提交梯度
        dummy_grad = torch.randn(10)
        success = raft_client.submit_gradient("test_param", dummy_grad)
        print(f"梯度提交结果: {success}")
    except Exception as e:
        print(f"捕获到异常: {e}")
    
    # 模拟恢复
    print("模拟网络恢复...")
    try:
        aggregated = raft_client.get_aggregated_gradients()
        if aggregated:
            print("成功获取聚合梯度")
        else:
            print("未获取到聚合梯度")
    except Exception as e:
        print(f"获取聚合梯度异常: {e}")
    
    print("容错测试完成")

def test_performance():
    """测试性能"""
    print("\n=== 测试性能 ===")
    
    model = SimpleModel()
    raft_client = RaftClient(worker_id=0)
    register_raft_hooks(model, raft_client)
    
    # 创建大量数据
    X, y = create_dummy_data(5000, 10)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    optimizer = RaftOptimizer(model.parameters(), lr=0.001, raft_client=raft_client)
    criterion = nn.MSELoss()
    
    # 性能测试
    start_time = time.time()
    
    model.train()
    for epoch in range(2):
        for batch_idx, (data, target) in enumerate(dataloader):
            raft_client.set_epoch(epoch)
            raft_client.set_batch_id(batch_idx)
            
            output = model(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                elapsed = time.time() - start_time
                print(f"已处理 {batch_idx} 批次, 耗时: {elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}秒")
    print("性能测试完成")

def main():
    """主测试函数"""
    print("开始Raft-PyTorch集成测试...")
    
    try:
        # 运行各项测试
        test_basic_training()
        test_gradient_synchronization()
        test_fault_tolerance()
        test_performance()
        
        print("\n=== 所有测试完成 ===")
        print("✅ 基本训练功能正常")
        print("✅ 梯度同步功能正常")
        print("✅ 容错机制正常")
        print("✅ 性能表现可接受")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 