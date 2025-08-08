import torch
import torch.nn as nn
import grpc
import numpy as np
from typing import Dict, List, Optional
import threading
import time

# 简化的protobuf消息结构
class GradientMessage:
    def __init__(self, epoch: int, worker_id: int, batch_id: int, 
                 param_name: str, gradients: List[float]):
        self.epoch = epoch
        self.worker_id = worker_id
        self.batch_id = batch_id
        self.param_name = param_name
        self.gradients = gradients

class AggregatedGradientMessage:
    def __init__(self, epoch: int, batch_id: int, worker_count: int,
                 gradients: Dict[str, List[float]]):
        self.epoch = epoch
        self.batch_id = batch_id
        self.worker_count = worker_count
        self.gradients = gradients

class MockGRPCStub:
    """模拟gRPC客户端，用于测试"""
    
    def __init__(self):
        self.submitted_gradients = []
        self.aggregated_gradients = {}
    
    def SubmitGradient(self, request):
        """模拟提交梯度"""
        grad_msg = GradientMessage(
            epoch=request.epoch,
            worker_id=request.worker_id,
            batch_id=request.batch_id,
            param_name=request.param_name,
            gradients=request.gradients
        )
        self.submitted_gradients.append(grad_msg)
        print(f"模拟提交梯度: Worker={request.worker_id}, Batch={request.batch_id}, Param={request.param_name}")
        return type('Response', (), {'success': True})()
    
    def GetAggregatedGradients(self, request):
        """模拟获取聚合梯度"""
        # 模拟聚合逻辑
        if self.submitted_gradients:
            latest = self.submitted_gradients[-1]
            aggregated = AggregatedGradientMessage(
                epoch=latest.epoch,
                batch_id=latest.batch_id,
                worker_count=3,
                gradients={
                    "weight1": [0.2, 0.3, 0.4],
                    "weight2": [0.5, 0.6, 0.7]
                }
            )
            return aggregated
        return None

class RaftClient:
    """Raft梯度同步客户端"""
    
    def __init__(self, server_address: str = "localhost:50051", worker_id: int = 0):
        self.server_address = server_address
        self.worker_id = worker_id
        self.epoch = 0
        self.batch_id = 0
        
        # 初始化gRPC连接（这里使用模拟客户端）
        self.stub = MockGRPCStub()
        
        # 梯度缓冲区
        self.gradient_buffer = {}
        self.buffer_lock = threading.Lock()
        
        print(f"Raft客户端初始化完成: WorkerID={worker_id}")
    
    def submit_gradient(self, param_name: str, gradients: torch.Tensor) -> bool:
        """提交梯度到Raft集群"""
        try:
            # 转换为numpy数组
            grad_array = gradients.detach().cpu().numpy().flatten().tolist()
            
            # 创建梯度消息
            grad_msg = GradientMessage(
                epoch=self.epoch,
                worker_id=self.worker_id,
                batch_id=self.batch_id,
                param_name=param_name,
                gradients=grad_array
            )
            
            # 提交到Raft集群
            response = self.stub.SubmitGradient(grad_msg)
            
            # 缓存梯度
            with self.buffer_lock:
                if self.batch_id not in self.gradient_buffer:
                    self.gradient_buffer[self.batch_id] = {}
                self.gradient_buffer[self.batch_id][param_name] = gradients.clone()
            
            return response.success
            
        except Exception as e:
            print(f"提交梯度失败: {e}")
            return False
    
    def get_aggregated_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取聚合梯度"""
        try:
            response = self.stub.GetAggregatedGradients(None)
            if response is None:
                return None
            
            # 转换为PyTorch张量
            aggregated = {}
            for param_name, grad_values in response.gradients.items():
                tensor = torch.tensor(grad_values, dtype=torch.float32)
                aggregated[param_name] = tensor
            
            return aggregated
            
        except Exception as e:
            print(f"获取聚合梯度失败: {e}")
            return None
    
    def set_epoch(self, epoch: int):
        """设置当前训练轮次"""
        self.epoch = epoch
    
    def set_batch_id(self, batch_id: int):
        """设置当前批次ID"""
        self.batch_id = batch_id

class RaftOptimizer(torch.optim.Optimizer):
    """基于Raft的优化器"""
    
    def __init__(self, params, lr=0.001, raft_client: RaftClient = None):
        super().__init__(params, {'lr': lr})
        self.raft_client = raft_client or RaftClient()
        self.param_groups = list(params)
    
    def step(self, closure=None):
        """执行优化步骤"""
        loss = None
        if closure is not None:
            loss = closure()
        
        # 获取聚合梯度
        aggregated_grads = self.raft_client.get_aggregated_gradients()
        if aggregated_grads is None:
            print("未获取到聚合梯度，跳过更新")
            return loss
        
        # 应用聚合梯度到参数
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                param_name = param._get_name()
                if param_name in aggregated_grads:
                    # 使用聚合梯度更新参数
                    param.data -= group['lr'] * aggregated_grads[param_name]
                    print(f"应用聚合梯度到参数: {param_name}")
        
        return loss

def register_raft_hooks(model: nn.Module, raft_client: RaftClient):
    """为模型参数注册Raft梯度Hook"""
    
    def create_gradient_hook(param_name: str):
        def hook(grad):
            if grad is not None:
                # 提交梯度到Raft集群
                success = raft_client.submit_gradient(param_name, grad)
                if success:
                    print(f"梯度已提交: {param_name}")
                else:
                    print(f"梯度提交失败: {param_name}")
        return hook
    
    # 为每个参数注册hook
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = create_gradient_hook(name)
            param.register_hook(hook)
            print(f"已为参数注册Raft Hook: {name}")

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