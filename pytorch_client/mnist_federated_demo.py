#!/usr/bin/env python3
"""
MNIST联邦学习演示 - 可以真正运行的版本

这个演示不依赖Raft后端，可以独立运行来验证联邦学习算法的正确性。
然后可以逐步集成到Raft系统中。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
import copy
import json
import hashlib
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FederatedConfig:
    """联邦学习配置"""
    num_clients: int = 3
    local_epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 0.01
    num_rounds: int = 10
    client_data_size: int = 2000  # 每个客户端的数据量

class SimpleNet(nn.Module):
    """简单的CNN模型用于MNIST"""
    
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class FederatedClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: int, train_loader: DataLoader, test_loader: DataLoader, config: FederatedConfig):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        
        # 初始化模型
        self.model = SimpleNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Client {client_id} initialized with {len(train_loader.dataset)} training samples")
    
    def local_train(self, global_model_params: Dict = None) -> Tuple[Dict, float, float]:
        """本地训练"""
        # 如果有全局模型参数，先加载
        if global_model_params:
            self.load_model_params(global_model_params)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            total_loss += epoch_loss
            logger.debug(f"Client {self.client_id} - Epoch {epoch + 1}/{self.config.local_epochs}, Loss: {epoch_loss:.4f}")
        
        avg_loss = total_loss / (self.config.local_epochs * len(self.train_loader))
        accuracy = 100.0 * correct / total
        
        # 提取模型参数
        model_params = self.get_model_params()
        
        logger.info(f"Client {self.client_id} training completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return model_params, avg_loss, accuracy
    
    def get_model_params(self) -> Dict:
        """获取模型参数"""
        params = {}
        for name, param in self.model.named_parameters():
            # 直接存储numpy数组，避免list嵌套问题
            params[name] = param.data.cpu().numpy()
        return params
    
    def load_model_params(self, params: Dict):
        """加载模型参数"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in params:
                    param.data = torch.from_numpy(params[name]).to(param.dtype)
    
    def evaluate(self) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / len(self.test_loader.dataset)
        
        return test_loss, accuracy

class FederatedAggregator:
    """联邦聚合器 - 模拟Raft的聚合功能"""
    
    def __init__(self):
        self.global_model = SimpleNet()
        self.round_history = []
    
    def federated_averaging(self, client_updates: List[Tuple[Dict, int]]) -> Dict:
        """联邦平均算法"""
        import numpy as np
        
        # client_updates: [(params, data_size), ...]
        
        # 计算权重
        total_data_size = sum(data_size for _, data_size in client_updates)
        weights = [data_size / total_data_size for _, data_size in client_updates]
        
        # 初始化聚合参数
        aggregated_params = {}
        
        # 加权平均 - 使用numpy进行高效计算
        for i, (params, _) in enumerate(client_updates):
            weight = weights[i]
            for param_name, param_array in params.items():
                if param_name not in aggregated_params:
                    # 初始化为零数组，形状与参数相同
                    aggregated_params[param_name] = np.zeros_like(param_array)
                
                # 加权累加
                aggregated_params[param_name] += weight * param_array
        
        logger.info(f"Aggregated {len(client_updates)} client updates with weights: {[f'{w:.3f}' for w in weights]}")
        return aggregated_params
    
    def update_global_model(self, aggregated_params: Dict):
        """更新全局模型"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_params:
                    param.data = torch.from_numpy(aggregated_params[name]).to(param.dtype)
    
    def get_global_model_params(self) -> Dict:
        """获取全局模型参数"""
        params = {}
        for name, param in self.global_model.named_parameters():
            params[name] = param.data.cpu().numpy()
        return params
    
    def evaluate_global_model(self, test_loader: DataLoader) -> Tuple[float, float]:
        """评估全局模型"""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100.0 * correct / len(test_loader.dataset)
        
        return test_loss, accuracy

def create_non_iid_data(dataset, num_clients: int, samples_per_client: int):
    """创建非IID数据分布（每个客户端偏向不同的数字）"""
    # 按标签分组
    label_to_indices = {}
    for idx, (_, label) in enumerate(dataset):
        if label not in label_to_indices:
            label_to_indices[label] = []
        label_to_indices[label].append(idx)
    
    client_datasets = []
    
    for client_id in range(num_clients):
        client_indices = []
        
        # 每个客户端主要包含2-3个类别的数据
        main_labels = [(client_id * 2) % 10, (client_id * 2 + 1) % 10]
        if client_id < 3:  # 前3个客户端再加一个类别
            main_labels.append((client_id * 2 + 2) % 10)
        
        # 80%数据来自主要类别，20%来自其他类别
        main_samples = int(samples_per_client * 0.8)
        other_samples = samples_per_client - main_samples
        
        # 从主要类别采样
        samples_per_main_label = main_samples // len(main_labels)
        for label in main_labels:
            indices = np.random.choice(label_to_indices[label], samples_per_main_label, replace=False)
            client_indices.extend(indices)
        
        # 从其他类别随机采样
        other_labels = [i for i in range(10) if i not in main_labels]
        for _ in range(other_samples):
            label = np.random.choice(other_labels)
            idx = np.random.choice(label_to_indices[label])
            client_indices.append(idx)
        
        client_datasets.append(Subset(dataset, client_indices))
        logger.info(f"Client {client_id} - Main labels: {main_labels}, Total samples: {len(client_indices)}")
    
    return client_datasets

def run_federated_mnist_demo(config: FederatedConfig):
    """运行MNIST联邦学习演示"""
    logger.info("🚀 Starting MNIST Federated Learning Demo")
    logger.info(f"Config: {config.num_clients} clients, {config.num_rounds} rounds, {config.local_epochs} local epochs")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    logger.info("📥 Loading MNIST dataset...")
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 创建非IID客户端数据
    logger.info("📊 Creating non-IID client datasets...")
    client_datasets = create_non_iid_data(train_dataset, config.num_clients, config.client_data_size)
    
    # 创建客户端
    clients = []
    for i in range(config.num_clients):
        train_loader = DataLoader(client_datasets[i], batch_size=config.batch_size, shuffle=True)
        client = FederatedClient(i, train_loader, test_loader, config)
        clients.append(client)
    
    # 创建聚合器
    aggregator = FederatedAggregator()
    
    # 记录训练历史
    history = {
        'rounds': [],
        'global_accuracy': [],
        'global_loss': [],
        'client_accuracies': [[] for _ in range(config.num_clients)]
    }
    
    # 联邦学习主循环
    logger.info("🔄 Starting federated training...")
    
    for round_num in range(config.num_rounds):
        logger.info(f"\n=== Round {round_num + 1}/{config.num_rounds} ===")
        
        # 获取当前全局模型参数
        global_params = aggregator.get_global_model_params()
        
        # 客户端本地训练
        client_updates = []
        for client in clients:
            params, loss, accuracy = client.local_train(global_params)
            client_updates.append((params, len(client.train_loader.dataset)))
            
            # 记录客户端准确率
            history['client_accuracies'][client.client_id].append(accuracy)
        
        # 聚合模型
        logger.info("🔄 Aggregating client updates...")
        aggregated_params = aggregator.federated_averaging(client_updates)
        aggregator.update_global_model(aggregated_params)
        
        # 评估全局模型
        global_loss, global_accuracy = aggregator.evaluate_global_model(test_loader)
        logger.info(f"Global Model - Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.2f}%")
        
        # 记录历史
        history['rounds'].append(round_num + 1)
        history['global_accuracy'].append(global_accuracy)
        history['global_loss'].append(global_loss)
        
        # 每5轮显示详细信息
        if (round_num + 1) % 5 == 0:
            logger.info(f"📈 Round {round_num + 1} Summary:")
            logger.info(f"   Global Accuracy: {global_accuracy:.2f}%")
            client_accs = [history['client_accuracies'][i][-1] for i in range(config.num_clients)]
            logger.info(f"   Client Accuracies: {[f'{acc:.1f}%' for acc in client_accs]}")
    
    # 最终评估
    logger.info("\n🎉 Federated Learning Completed!")
    final_loss, final_accuracy = aggregator.evaluate_global_model(test_loader)
    logger.info(f"Final Global Model - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
    
    # 绘制训练曲线
    plot_training_history(history, config)
    
    return history, aggregator

def plot_training_history(history, config):
    """绘制训练历史"""
    try:
        plt.figure(figsize=(15, 5))
        
        # 全局准确率曲线
        plt.subplot(1, 3, 1)
        plt.plot(history['rounds'], history['global_accuracy'], 'b-', linewidth=2, label='Global Model')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Global Model Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 全局损失曲线
        plt.subplot(1, 3, 2)
        plt.plot(history['rounds'], history['global_loss'], 'r-', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Global Model Loss')
        plt.grid(True, alpha=0.3)
        
        # 客户端准确率对比
        plt.subplot(1, 3, 3)
        for i in range(config.num_clients):
            plt.plot(history['rounds'], history['client_accuracies'][i], 
                    label=f'Client {i}', marker='o', markersize=3)
        plt.xlabel('Round')
        plt.ylabel('Local Accuracy (%)')
        plt.title('Client Local Accuracies')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('federated_mnist_results.png', dpi=150, bbox_inches='tight')
        logger.info("📊 Training curves saved to 'federated_mnist_results.png'")
        plt.show()
        
    except Exception as e:
        logger.warning(f"Could not plot results: {e}")

def simulate_raft_integration(history, aggregator):
    """模拟Raft集成 - 展示如何与Raft系统集成"""
    logger.info("\n🔗 Simulating Raft Integration...")
    
    # 模拟Raft日志条目
    raft_logs = []
    
    for round_num in range(len(history['rounds'])):
        # 模拟客户端更新日志
        for client_id in range(len(history['client_accuracies'])):
            log_entry = {
                "type": "federated_log_entry",
                "action": "model_update",
                "round_id": round_num,
                "client_id": f"client_{client_id}",
                "local_accuracy": history['client_accuracies'][client_id][round_num],
                "timestamp": f"2024-12-{round_num+1:02d}T10:00:00Z",
                "hash": hashlib.sha256(f"round_{round_num}_client_{client_id}".encode()).hexdigest()[:16]
            }
            raft_logs.append(log_entry)
        
        # 模拟聚合日志
        aggregation_entry = {
            "type": "federated_log_entry", 
            "action": "model_aggregation",
            "round_id": round_num,
            "global_accuracy": history['global_accuracy'][round_num],
            "participants": [f"client_{i}" for i in range(len(history['client_accuracies']))],
            "timestamp": f"2024-12-{round_num+1:02d}T10:05:00Z",
            "hash": hashlib.sha256(f"aggregation_round_{round_num}".encode()).hexdigest()[:16]
        }
        raft_logs.append(aggregation_entry)
    
    # 保存模拟的Raft日志
    with open('simulated_raft_logs.json', 'w') as f:
        json.dump(raft_logs, f, indent=2)
    
    logger.info(f"✅ Generated {len(raft_logs)} simulated Raft log entries")
    logger.info("📄 Saved to 'simulated_raft_logs.json'")
    
    # 显示部分日志
    logger.info("📋 Sample Raft Log Entries:")
    for i, entry in enumerate(raft_logs[:3]):
        logger.info(f"   Entry {i+1}: {entry['action']} - {entry.get('client_id', 'aggregator')} - Round {entry['round_id']}")

if __name__ == "__main__":
    # 配置
    config = FederatedConfig(
        num_clients=3,
        local_epochs=3,
        batch_size=64,
        learning_rate=0.01,
        num_rounds=10,
        client_data_size=2000
    )
    
    # 运行演示
    history, aggregator = run_federated_mnist_demo(config)
    
    # 模拟Raft集成
    simulate_raft_integration(history, aggregator)
    
    logger.info("\n🎯 Demo completed! Key achievements:")
    logger.info(f"   ✅ Trained federated model on MNIST with {config.num_clients} clients")
    logger.info(f"   ✅ Achieved {history['global_accuracy'][-1]:.2f}% accuracy on test set")
    logger.info(f"   ✅ Demonstrated non-IID data handling")
    logger.info(f"   ✅ Simulated Raft integration with audit logs")
    logger.info("\n💡 Next steps: Integrate this with the actual Raft cluster!")
