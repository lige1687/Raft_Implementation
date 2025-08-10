import torch
import torch.nn as nn
import requests
from typing import Dict, List, Optional
import threading
import time

class RaftClient:
    """基于HTTP的Raft梯度同步客户端"""
    
    def __init__(self, server_address: str = "http://127.0.0.1:8080", worker_id: int = 0):
        self.server_address = server_address.rstrip('/')
        self.worker_id = worker_id
        self.epoch = 0
        self.batch_id = 0
        
        # 梯度缓冲区
        self.gradient_buffer = {}
        self.buffer_lock = threading.Lock()
        
        print(f"Raft HTTP 客户端初始化完成: WorkerID={worker_id}, Server={self.server_address}")
    
    def submit_gradient(self, param_name: str, gradients: torch.Tensor) -> bool:
        """提交梯度到HTTP服务"""
        try:
            grad_array = gradients.detach().cpu().numpy().flatten().tolist()
            payload = {
                "epoch": self.epoch,
                "worker_id": self.worker_id,
                "batch_id": self.batch_id,
                "gradients": {param_name: grad_array},
            }
            url = f"{self.server_address}/submit_gradient"
            resp = requests.post(url, json=payload, timeout=5)
            ok = resp.status_code == 200 and resp.json().get("ok", False)
            
            with self.buffer_lock:
                if self.batch_id not in self.gradient_buffer:
                    self.gradient_buffer[self.batch_id] = {}
                self.gradient_buffer[self.batch_id][param_name] = gradients.clone()
            return ok
        except Exception as e:
            print(f"提交梯度失败: {e}")
            return False
    
    def get_aggregated_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取聚合梯度（按当前batch_id查询）"""
        try:
            url = f"{self.server_address}/aggregated?batch_id={self.batch_id}"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if not data or (isinstance(data, dict) and data.get("found") is False):
                return None
            aggregated = {}
            for param_name, grad_values in data.get("Gradients", data.get("gradients", {})).items():
                tensor = torch.tensor(grad_values, dtype=torch.float32)
                aggregated[param_name] = tensor
            return aggregated if aggregated else None
        except Exception as e:
            print(f"获取聚合梯度失败: {e}")
            return None
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def set_batch_id(self, batch_id: int):
        self.batch_id = batch_id

class RaftOptimizer(torch.optim.Optimizer):
    """基于Raft的优化器（HTTP版）"""
    
    def __init__(self, params, lr=0.001, raft_client: RaftClient = None):
        super().__init__(params, {'lr': lr})
        self.raft_client = raft_client or RaftClient()
        # 建立name->param映射
        self.name_to_param: Dict[str, torch.nn.Parameter] = {}
    
    def bind_model(self, model: nn.Module):
        for name, param in model.named_parameters():
            self.name_to_param[name] = param
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        aggregated_grads = self.raft_client.get_aggregated_gradients()
        if not aggregated_grads:
            return loss
        # 应用聚合梯度
        for name, param in self.name_to_param.items():
            if name in aggregated_grads:
                grad = aggregated_grads[name]
                # 尺寸对齐（服务端按flatten返回，这里简单截断/填充）
                if grad.numel() != param.data.numel():
                    flat = grad.flatten()
                    target = torch.zeros_like(param.data.flatten())
                    n = min(flat.numel(), target.numel())
                    target[:n] = flat[:n]
                    grad = target.view_as(param.data)
                else:
                    grad = grad.view_as(param.data)
                param.data -= self.param_groups[0]['lr'] * grad
        return loss

def register_raft_hooks(model: nn.Module, raft_client: RaftClient, optimizer: Optional[RaftOptimizer] = None):
    """为模型参数注册Raft梯度Hook，并绑定优化器映射"""
    if optimizer is not None:
        optimizer.bind_model(model)
    
    def create_gradient_hook(param_name: str):
        def hook(grad):
            if grad is not None:
                raft_client.submit_gradient(param_name, grad)
        return hook
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(create_gradient_hook(name))

# 测试函数
def test_raft_integration():
    """测试Raft集成"""
    print("=== 测试Raft-PyTorch集成 ===")
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # 创建Raft客户端
    raft_client = RaftClient(worker_id=0)
    
    # 注册梯度Hook
    register_raft_hooks(model, raft_client)
    
    # 创建优化器
    optimizer = RaftOptimizer(model.parameters(), lr=0.001, raft_client=raft_client)
    
    # 模拟训练
    print("开始模拟训练...")
    for epoch in range(2):
        raft_client.set_epoch(epoch)
        
        for batch in range(3):
            raft_client.set_batch_id(batch)
            
            # 模拟前向传播
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            
            # 前向传播
            output = model(x)
            loss = nn.MSELoss()(output, y)
            
            # 反向传播（会自动触发Hook）
            loss.backward()
            
            # 优化步骤
            optimizer.step()
            
            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss.item():.4f}")
    
    print("=== 测试完成 ===")

if __name__ == "__main__":
    test_raft_integration() 