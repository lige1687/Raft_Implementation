#!/usr/bin/env python3
"""
TrustedFL - 基于Raft共识的可信联邦学习客户端

这个客户端演示了如何在联邦学习场景中使用Raft共识来保证模型聚合的可靠性。
特别适用于需要高可信度的场景，如医疗、金融等领域。
"""

import json
import time
import hashlib
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """客户端配置"""
    client_id: str
    raft_nodes: List[str]  # Raft集群节点地址
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    device: str = "cpu"
    data_size: int = 1000
    
class SimpleModel(nn.Module):
    """简单的神经网络模型用于演示"""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class FederatedLearningClient:
    """联邦学习客户端"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.client_id = config.client_id
        self.raft_nodes = config.raft_nodes
        self.current_leader = None
        self.current_round = 0
        
        # 初始化模型
        self.model = SimpleModel()
        self.model.to(config.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 生成模拟数据
        self.train_loader = self._generate_synthetic_data()
        
        logger.info(f"Initialized client {self.client_id}")
        
    def _generate_synthetic_data(self) -> DataLoader:
        """生成模拟训练数据（每个客户端数据分布不同）"""
        np.random.seed(hash(self.client_id) % 2**32)  # 基于client_id的固定随机种子
        
        # 生成不同分布的数据（模拟非IID场景）
        if "hospital" in self.client_id.lower():
            # 医院数据：偏向某些类别（模拟疾病分布差异）
            data = np.random.randn(self.config.data_size, 784) * 0.5
            labels = np.random.choice(10, self.config.data_size, 
                                    p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01])
        elif "bank" in self.client_id.lower():
            # 银行数据：偏向不同类别（模拟交易类型差异）
            data = np.random.randn(self.config.data_size, 784) * 0.8
            labels = np.random.choice(10, self.config.data_size,
                                    p=[0.1, 0.1, 0.1, 0.3, 0.2, 0.1, 0.05, 0.02, 0.02, 0.01])
        else:
            # 通用数据
            data = np.random.randn(self.config.data_size, 784)
            labels = np.random.choice(10, self.config.data_size)
        
        # 转换为张量
        X = torch.FloatTensor(data)
        y = torch.LongTensor(labels)
        
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
    
    def _find_leader(self) -> Optional[str]:
        """寻找当前的Raft Leader"""
        for node in self.raft_nodes:
            try:
                response = requests.get(f"http://{node}/raft/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    if status.get("is_leader", False):
                        self.current_leader = node
                        return node
            except Exception as e:
                logger.debug(f"Failed to contact node {node}: {e}")
                continue
        return None
    
    def local_train(self) -> Tuple[Dict[str, List[float]], float, float]:
        """本地训练"""
        logger.info(f"Client {self.client_id} starting local training...")
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            total_loss += epoch_loss
            logger.debug(f"Epoch {epoch + 1}/{self.config.local_epochs}, Loss: {epoch_loss:.4f}")
        
        avg_loss = total_loss / (self.config.local_epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        # 提取模型参数
        parameters = {}
        for name, param in self.model.named_parameters():
            parameters[name] = param.data.cpu().numpy().tolist()
        
        logger.info(f"Local training completed. Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return parameters, avg_loss, accuracy
    
    def _calculate_model_hash(self, model_update: Dict) -> str:
        """计算模型更新的哈希值"""
        # 排除hash字段本身
        update_copy = model_update.copy()
        if 'hash' in update_copy:
            del update_copy['hash']
        
        # 序列化并计算哈希
        json_str = json.dumps(update_copy, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def submit_model_update(self, parameters: Dict[str, List[float]], 
                          loss: float, accuracy: float) -> bool:
        """提交模型更新到Raft集群"""
        if not self.current_leader:
            leader = self._find_leader()
            if not leader:
                logger.error("No leader found in Raft cluster")
                return False
        
        # 构造模型更新
        model_update = {
            "client_id": self.client_id,
            "round_id": self.current_round,
            "parameters": parameters,
            "gradients": {},  # 这里可以添加梯度信息
            "data_size": self.config.data_size,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "metadata": {
                "local_epochs": self.config.local_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "model_type": "SimpleModel"
            }
        }
        
        # 计算哈希
        model_update["hash"] = self._calculate_model_hash(model_update)
        
        # 提交请求
        request_data = {
            "update": model_update,
            "term": 0,  # 简化处理
            "leader_id": 0,
            "timestamp": time.time()
        }
        
        try:
            response = requests.post(
                f"http://{self.current_leader}/federated/submit_update",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    logger.info(f"Successfully submitted model update for round {self.current_round}")
                    return True
                else:
                    logger.error(f"Failed to submit model update: {result.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"HTTP error {response.status_code} when submitting model update")
                return False
                
        except Exception as e:
            logger.error(f"Exception when submitting model update: {e}")
            # 尝试重新寻找Leader
            self.current_leader = None
            return False
    
    def request_aggregation(self) -> Optional[Dict]:
        """请求模型聚合"""
        if not self.current_leader:
            leader = self._find_leader()
            if not leader:
                logger.error("No leader found in Raft cluster")
                return None
        
        request_data = {
            "round_id": self.current_round,
            "client_id": self.client_id,
            "term": 0,
            "leader_id": 0,
            "timestamp": time.time()
        }
        
        try:
            response = requests.post(
                f"http://{self.current_leader}/federated/request_aggregation",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    logger.info(f"Successfully requested aggregation for round {self.current_round}")
                    return result.get("global_model")
                else:
                    logger.error(f"Failed to request aggregation: {result.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"HTTP error {response.status_code} when requesting aggregation")
                return None
                
        except Exception as e:
            logger.error(f"Exception when requesting aggregation: {e}")
            return None
    
    def get_global_model(self) -> Optional[Dict]:
        """获取全局模型"""
        if not self.current_leader:
            leader = self._find_leader()
            if not leader:
                logger.error("No leader found in Raft cluster")
                return None
        
        request_data = {
            "client_id": self.client_id,
            "round_id": self.current_round,
            "term": 0,
            "timestamp": time.time()
        }
        
        try:
            response = requests.post(
                f"http://{self.current_leader}/federated/get_global_model",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    return result.get("global_model")
                else:
                    logger.error(f"Failed to get global model: {result.get('message', 'Unknown error')}")
                    return None
            else:
                logger.error(f"HTTP error {response.status_code} when getting global model")
                return None
                
        except Exception as e:
            logger.error(f"Exception when getting global model: {e}")
            return None
    
    def update_local_model(self, global_model: Dict):
        """用全局模型更新本地模型"""
        if not global_model or "parameters" not in global_model:
            logger.error("Invalid global model received")
            return
        
        global_params = global_model["parameters"]
        
        # 更新本地模型参数
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_params:
                    param.data = torch.tensor(global_params[name], dtype=param.dtype)
                    logger.debug(f"Updated parameter {name}")
        
        logger.info(f"Local model updated with global model from round {global_model.get('round_id', 'unknown')}")
    
    def run_federated_training(self, total_rounds: int = 10, 
                             wait_time: int = 30):
        """运行联邦训练主循环"""
        logger.info(f"Starting federated training for {total_rounds} rounds")
        
        for round_num in range(total_rounds):
            self.current_round = round_num
            logger.info(f"\n=== Round {round_num + 1}/{total_rounds} ===")
            
            # 1. 本地训练
            parameters, loss, accuracy = self.local_train()
            
            # 2. 提交模型更新
            success = self.submit_model_update(parameters, loss, accuracy)
            if not success:
                logger.error(f"Failed to submit update for round {round_num}")
                continue
            
            # 3. 等待其他客户端
            logger.info(f"Waiting {wait_time}s for other clients...")
            time.sleep(wait_time)
            
            # 4. 请求聚合（只有第一个客户端请求）
            if "client_0" in self.client_id or "hospital_a" in self.client_id:
                global_model = self.request_aggregation()
                if global_model:
                    logger.info(f"Aggregation completed. Global accuracy: {global_model.get('global_accuracy', 'N/A'):.2f}%")
            
            # 5. 获取全局模型
            time.sleep(5)  # 等待聚合完成
            global_model = self.get_global_model()
            if global_model:
                self.update_local_model(global_model)
                logger.info(f"Round {round_num + 1} completed successfully")
            else:
                logger.error(f"Failed to get global model for round {round_num}")
        
        logger.info("Federated training completed!")

def create_medical_scenario_clients():
    """创建医疗场景的演示客户端"""
    raft_nodes = ["localhost:8080", "localhost:8081", "localhost:8082"]
    
    clients = []
    
    # 医院A：心脏病专科
    config_a = ClientConfig(
        client_id="hospital_a_cardiology",
        raft_nodes=raft_nodes,
        data_size=800,
        local_epochs=3,
        learning_rate=0.01
    )
    clients.append(FederatedLearningClient(config_a))
    
    # 医院B：神经科
    config_b = ClientConfig(
        client_id="hospital_b_neurology", 
        raft_nodes=raft_nodes,
        data_size=600,
        local_epochs=3,
        learning_rate=0.01
    )
    clients.append(FederatedLearningClient(config_b))
    
    # 医院C：综合医院
    config_c = ClientConfig(
        client_id="hospital_c_general",
        raft_nodes=raft_nodes,
        data_size=1200,
        local_epochs=3,
        learning_rate=0.01
    )
    clients.append(FederatedLearningClient(config_c))
    
    return clients

def create_financial_scenario_clients():
    """创建金融场景的演示客户端"""
    raft_nodes = ["localhost:8080", "localhost:8081", "localhost:8082"]
    
    clients = []
    
    # 银行A：投资银行
    config_a = ClientConfig(
        client_id="bank_a_investment",
        raft_nodes=raft_nodes,
        data_size=1500,
        local_epochs=5,
        learning_rate=0.005
    )
    clients.append(FederatedLearningClient(config_a))
    
    # 银行B：零售银行
    config_b = ClientConfig(
        client_id="bank_b_retail",
        raft_nodes=raft_nodes,
        data_size=2000,
        local_epochs=5,
        learning_rate=0.005
    )
    clients.append(FederatedLearningClient(config_b))
    
    return clients

if __name__ == "__main__":
    import argparse
    import threading
    
    parser = argparse.ArgumentParser(description="TrustedFL Federated Learning Client")
    parser.add_argument("--scenario", choices=["medical", "financial", "simple"], 
                       default="medical", help="Scenario to run")
    parser.add_argument("--client-id", type=str, help="Specific client ID to run")
    parser.add_argument("--rounds", type=int, default=5, help="Number of training rounds")
    parser.add_argument("--wait-time", type=int, default=20, help="Wait time between rounds")
    
    args = parser.parse_args()
    
    if args.scenario == "medical":
        clients = create_medical_scenario_clients()
    elif args.scenario == "financial":
        clients = create_financial_scenario_clients()
    else:
        # 简单场景
        config = ClientConfig(
            client_id=args.client_id or "client_0",
            raft_nodes=["localhost:8080", "localhost:8081", "localhost:8082"]
        )
        clients = [FederatedLearningClient(config)]
    
    if args.client_id:
        # 运行特定客户端
        target_client = None
        for client in clients:
            if client.client_id == args.client_id:
                target_client = client
                break
        
        if target_client:
            target_client.run_federated_training(args.rounds, args.wait_time)
        else:
            logger.error(f"Client {args.client_id} not found")
    else:
        # 并行运行所有客户端
        threads = []
        for client in clients:
            thread = threading.Thread(
                target=client.run_federated_training,
                args=(args.rounds, args.wait_time)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有客户端完成
        for thread in threads:
            thread.join()
        
        logger.info("All clients completed federated training!")
