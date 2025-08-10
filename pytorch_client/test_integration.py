#!/usr/bin/env python3
"""
Raft-PyTorch集成测试脚本（HTTP版）
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from raft_client import RaftClient, RaftOptimizer, register_raft_hooks
import time

class SimpleModel(nn.Module):
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

def create_dummy_data(num_samples=200, input_size=10):
    X = torch.randn(num_samples, input_size)
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
    return X, y

def test_basic_training():
    print("=== 测试基本训练功能（HTTP） ===")
    model = SimpleModel()
    X, y = create_dummy_data(400, 10)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    raft_client = RaftClient(worker_id=0)
    optimizer = RaftOptimizer(model.parameters(), lr=0.001, raft_client=raft_client)
    register_raft_hooks(model, raft_client, optimizer)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(2):
        raft_client.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(dataloader):
            raft_client.set_batch_id(batch_idx)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_basic_training() 