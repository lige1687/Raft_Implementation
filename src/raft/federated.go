package raft

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// FederatedLogEntry 联邦学习日志条目
type FederatedLogEntry struct {
	Type        string          `json:"type"`         // "model_update", "aggregation", "validation"
	RoundID     int            `json:"round_id"`     // 联邦学习轮次
	ClientID    string         `json:"client_id"`    // 客户端标识
	ModelUpdate *ModelUpdate   `json:"model_update"` // 模型更新
	Signature   string         `json:"signature"`    // 数字签名
	Timestamp   time.Time      `json:"timestamp"`
}

// ModelUpdate 客户端模型更新
type ModelUpdate struct {
	ClientID     string                 `json:"client_id"`
	RoundID      int                   `json:"round_id"`
	Parameters   map[string][]float32  `json:"parameters"`   // 模型参数
	Gradients    map[string][]float32  `json:"gradients"`    // 梯度更新
	DataSize     int                   `json:"data_size"`    // 本地数据量
	Loss         float32               `json:"loss"`         // 本地损失
	Accuracy     float32               `json:"accuracy"`     // 本地准确率
	Metadata     map[string]interface{} `json:"metadata"`     // 元数据
	Hash         string                `json:"hash"`         // 内容哈希
}

// GlobalModel 全局模型状态
type GlobalModel struct {
	RoundID       int                   `json:"round_id"`
	Parameters    map[string][]float32  `json:"parameters"`
	Participants  []string              `json:"participants"` // 参与客户端
	Aggregator    string                `json:"aggregator"`   // 聚合节点
	TotalDataSize int                   `json:"total_data_size"`
	GlobalLoss    float32               `json:"global_loss"`
	GlobalAccuracy float32              `json:"global_accuracy"`
	CreatedAt     time.Time             `json:"created_at"`
	Hash          string                `json:"hash"`
}

// FederatedLearningManager 联邦学习管理器
type FederatedLearningManager struct {
	mu sync.RWMutex
	
	// 当前状态
	currentRound  int
	globalModel   *GlobalModel
	
	// 客户端管理
	clientUpdates map[int]map[string]*ModelUpdate // roundID -> clientID -> update
	clientStatus  map[string]ClientStatus          // clientID -> status
	
	// 聚合配置
	minClientsPerRound int
	maxWaitTime        time.Duration
	aggregationMethod  string // "fedavg", "fedprox", "scaffold"
	
	// 安全配置
	enableByzantineTolerance bool
	maxMaliciousClients      int
	reputationScores         map[string]float32 // clientID -> reputation
	
	// 审计日志
	auditLog []AuditEntry
	
	// 统计信息
	stats FederatedStats
}

// ClientStatus 客户端状态
type ClientStatus struct {
	ClientID     string    `json:"client_id"`
	Status       string    `json:"status"`        // "active", "inactive", "malicious", "pending"
	LastSeen     time.Time `json:"last_seen"`
	Reputation   float32   `json:"reputation"`
	TotalRounds  int       `json:"total_rounds"`
	SuccessRounds int      `json:"success_rounds"`
	DataQuality  float32   `json:"data_quality"`
}

// AuditEntry 审计日志条目
type AuditEntry struct {
	Timestamp time.Time              `json:"timestamp"`
	Action    string                 `json:"action"`
	ClientID  string                 `json:"client_id"`
	RoundID   int                   `json:"round_id"`
	Details   map[string]interface{} `json:"details"`
	Hash      string                 `json:"hash"`
}

// FederatedStats 联邦学习统计
type FederatedStats struct {
	TotalRounds        int       `json:"total_rounds"`
	ActiveClients      int       `json:"active_clients"`
	TotalUpdates       int       `json:"total_updates"`
	MaliciousDetected  int       `json:"malicious_detected"`
	LastRoundTime      time.Time `json:"last_round_time"`
	AverageRoundTime   float32   `json:"average_round_time"`
	ConsensusFailures  int       `json:"consensus_failures"`
}

// NewFederatedLearningManager 创建联邦学习管理器
func NewFederatedLearningManager() *FederatedLearningManager {
	return &FederatedLearningManager{
		currentRound:      0,
		clientUpdates:     make(map[int]map[string]*ModelUpdate),
		clientStatus:      make(map[string]ClientStatus),
		minClientsPerRound: 3,
		maxWaitTime:       30 * time.Second,
		aggregationMethod: "fedavg",
		enableByzantineTolerance: true,
		maxMaliciousClients: 1,
		reputationScores:  make(map[string]float32),
		auditLog:         make([]AuditEntry, 0),
		globalModel: &GlobalModel{
			RoundID:    0,
			Parameters: make(map[string][]float32),
			CreatedAt:  time.Now(),
		},
	}
}

// SubmitModelUpdate 提交客户端模型更新
func (flm *FederatedLearningManager) SubmitModelUpdate(update *ModelUpdate) error {
	flm.mu.Lock()
	defer flm.mu.Unlock()
	
	// 验证更新
	if err := flm.validateModelUpdate(update); err != nil {
		return fmt.Errorf("validation failed: %v", err)
	}
	
	// 检查客户端状态
	status, exists := flm.clientStatus[update.ClientID]
	if !exists {
		// 新客户端，初始化状态
		status = ClientStatus{
			ClientID:    update.ClientID,
			Status:      "active",
			LastSeen:    time.Now(),
			Reputation:  1.0,
			TotalRounds: 0,
			DataQuality: 1.0,
		}
		flm.clientStatus[update.ClientID] = status
		flm.reputationScores[update.ClientID] = 1.0
	}
	
	if status.Status == "malicious" {
		return fmt.Errorf("client %s is marked as malicious", update.ClientID)
	}
	
	// 初始化轮次映射
	if flm.clientUpdates[update.RoundID] == nil {
		flm.clientUpdates[update.RoundID] = make(map[string]*ModelUpdate)
	}
	
	// 存储更新
	flm.clientUpdates[update.RoundID][update.ClientID] = update
	
	// 更新客户端状态
	status.LastSeen = time.Now()
	status.TotalRounds++
	flm.clientStatus[update.ClientID] = status
	
	// 记录审计日志
	flm.recordAuditEntry("model_update_received", update.ClientID, update.RoundID, 
		map[string]interface{}{
			"data_size": update.DataSize,
			"loss":      update.Loss,
			"accuracy":  update.Accuracy,
		})
	
	flm.stats.TotalUpdates++
	
	return nil
}

// TriggerAggregation 触发模型聚合
func (flm *FederatedLearningManager) TriggerAggregation(roundID int) (*GlobalModel, error) {
	flm.mu.Lock()
	defer flm.mu.Unlock()
	
	updates := flm.clientUpdates[roundID]
	if len(updates) < flm.minClientsPerRound {
		return nil, fmt.Errorf("insufficient clients: got %d, need %d", 
			len(updates), flm.minClientsPerRound)
	}
	
	// 拜占庭容错检测
	if flm.enableByzantineTolerance {
		updates = flm.detectAndFilterMaliciousUpdates(updates)
	}
	
	// 执行聚合
	aggregatedModel, err := flm.aggregateModels(updates, roundID)
	if err != nil {
		return nil, fmt.Errorf("aggregation failed: %v", err)
	}
	
	// 更新全局模型
	flm.globalModel = aggregatedModel
	flm.currentRound = roundID
	
	// 更新客户端信誉分数
	flm.updateReputationScores(updates)
	
	// 记录审计日志
	flm.recordAuditEntry("model_aggregated", "system", roundID, 
		map[string]interface{}{
			"participants":    len(updates),
			"global_loss":     aggregatedModel.GlobalLoss,
			"global_accuracy": aggregatedModel.GlobalAccuracy,
		})
	
	flm.stats.TotalRounds++
	flm.stats.LastRoundTime = time.Now()
	
	return aggregatedModel, nil
}

// validateModelUpdate 验证模型更新
func (flm *FederatedLearningManager) validateModelUpdate(update *ModelUpdate) error {
	if update == nil {
		return fmt.Errorf("update is nil")
	}
	
	if update.ClientID == "" {
		return fmt.Errorf("client ID is empty")
	}
	
	if update.RoundID < 0 {
		return fmt.Errorf("invalid round ID: %d", update.RoundID)
	}
	
	if len(update.Parameters) == 0 && len(update.Gradients) == 0 {
		return fmt.Errorf("no parameters or gradients provided")
	}
	
	if update.DataSize <= 0 {
		return fmt.Errorf("invalid data size: %d", update.DataSize)
	}
	
	// 验证哈希
	expectedHash := flm.calculateUpdateHash(update)
	if update.Hash != expectedHash {
		return fmt.Errorf("hash mismatch: expected %s, got %s", expectedHash, update.Hash)
	}
	
	return nil
}

// detectAndFilterMaliciousUpdates 检测并过滤恶意更新
func (flm *FederatedLearningManager) detectAndFilterMaliciousUpdates(
	updates map[string]*ModelUpdate) map[string]*ModelUpdate {
	
	if len(updates) <= flm.maxMaliciousClients {
		return updates // 如果客户端数量太少，无法过滤
	}
	
	// 计算参数统计信息
	paramStats := flm.calculateParameterStatistics(updates)
	
	// 检测异常更新
	suspiciousClients := make([]string, 0)
	for clientID, update := range updates {
		if flm.isUpdateSuspicious(update, paramStats) {
			suspiciousClients = append(suspiciousClients, clientID)
		}
	}
	
	// 标记恶意客户端
	for _, clientID := range suspiciousClients {
		if len(suspiciousClients) <= flm.maxMaliciousClients {
			status := flm.clientStatus[clientID]
			status.Status = "malicious"
			status.Reputation *= 0.5 // 降低信誉分数
			flm.clientStatus[clientID] = status
			flm.reputationScores[clientID] = status.Reputation
			
			delete(updates, clientID)
			flm.stats.MaliciousDetected++
			
			flm.recordAuditEntry("malicious_client_detected", clientID, updates[clientID].RoundID,
				map[string]interface{}{
					"reason": "statistical_anomaly",
				})
		}
	}
	
	return updates
}

// aggregateModels 聚合模型更新
func (flm *FederatedLearningManager) aggregateModels(
	updates map[string]*ModelUpdate, roundID int) (*GlobalModel, error) {
	
	switch flm.aggregationMethod {
	case "fedavg":
		return flm.federatedAveraging(updates, roundID)
	case "fedprox":
		return flm.federatedProx(updates, roundID)
	default:
		return flm.federatedAveraging(updates, roundID)
	}
}

// federatedAveraging 联邦平均算法
func (flm *FederatedLearningManager) federatedAveraging(
	updates map[string]*ModelUpdate, roundID int) (*GlobalModel, error) {
	
	// 计算总数据量
	totalDataSize := 0
	for _, update := range updates {
		totalDataSize += update.DataSize
	}
	
	// 初始化聚合结果
	aggregatedParams := make(map[string][]float32)
	participants := make([]string, 0, len(updates))
	
	var totalLoss, totalAccuracy float32
	
	// 加权聚合
	for clientID, update := range updates {
		participants = append(participants, clientID)
		weight := float32(update.DataSize) / float32(totalDataSize)
		
		// 聚合参数
		for paramName, paramValues := range update.Parameters {
			if _, exists := aggregatedParams[paramName]; !exists {
				aggregatedParams[paramName] = make([]float32, len(paramValues))
			}
			
			for i, val := range paramValues {
				aggregatedParams[paramName][i] += weight * val
			}
		}
		
		// 聚合指标
		totalLoss += weight * update.Loss
		totalAccuracy += weight * update.Accuracy
	}
	
	// 创建全局模型
	globalModel := &GlobalModel{
		RoundID:       roundID,
		Parameters:    aggregatedParams,
		Participants:  participants,
		Aggregator:    "fedavg",
		TotalDataSize: totalDataSize,
		GlobalLoss:    totalLoss,
		GlobalAccuracy: totalAccuracy,
		CreatedAt:     time.Now(),
	}
	
	// 计算哈希
	globalModel.Hash = flm.calculateGlobalModelHash(globalModel)
	
	return globalModel, nil
}

// federatedProx FedProx算法（带正则化项）
func (flm *FederatedLearningManager) federatedProx(
	updates map[string]*ModelUpdate, roundID int) (*GlobalModel, error) {
	
	// 基本实现，实际中需要更复杂的正则化
	return flm.federatedAveraging(updates, roundID)
}

// 辅助函数
func (flm *FederatedLearningManager) calculateUpdateHash(update *ModelUpdate) string {
	// 排除Hash字段本身
	temp := *update
	temp.Hash = ""
	
	data, _ := json.Marshal(temp)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

func (flm *FederatedLearningManager) calculateGlobalModelHash(model *GlobalModel) string {
	temp := *model
	temp.Hash = ""
	
	data, _ := json.Marshal(temp)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

func (flm *FederatedLearningManager) calculateParameterStatistics(
	updates map[string]*ModelUpdate) map[string]interface{} {
	// 简化实现，实际中需要更复杂的统计分析
	return map[string]interface{}{
		"mean": 0.0,
		"std":  1.0,
	}
}

func (flm *FederatedLearningManager) isUpdateSuspicious(
	update *ModelUpdate, stats map[string]interface{}) bool {
	// 简化实现，实际中需要更复杂的异常检测
	return false
}

func (flm *FederatedLearningManager) updateReputationScores(updates map[string]*ModelUpdate) {
	for clientID := range updates {
		// 简化实现：根据参与情况更新信誉分数
		if score, exists := flm.reputationScores[clientID]; exists {
			flm.reputationScores[clientID] = score * 0.95 + 0.05 // 缓慢提升
		}
	}
}

func (flm *FederatedLearningManager) recordAuditEntry(action, clientID string, roundID int, details map[string]interface{}) {
	entry := AuditEntry{
		Timestamp: time.Now(),
		Action:    action,
		ClientID:  clientID,
		RoundID:   roundID,
		Details:   details,
	}
	
	// 计算审计条目哈希
	data, _ := json.Marshal(entry)
	hash := sha256.Sum256(data)
	entry.Hash = fmt.Sprintf("%x", hash)
	
	flm.auditLog = append(flm.auditLog, entry)
}

// GetGlobalModel 获取当前全局模型
func (flm *FederatedLearningManager) GetGlobalModel() *GlobalModel {
	flm.mu.RLock()
	defer flm.mu.RUnlock()
	
	return flm.globalModel
}

// GetClientStatus 获取客户端状态
func (flm *FederatedLearningManager) GetClientStatus(clientID string) (ClientStatus, bool) {
	flm.mu.RLock()
	defer flm.mu.RUnlock()
	
	status, exists := flm.clientStatus[clientID]
	return status, exists
}

// GetStats 获取统计信息
func (flm *FederatedLearningManager) GetStats() FederatedStats {
	flm.mu.RLock()
	defer flm.mu.RUnlock()
	
	stats := flm.stats
	stats.ActiveClients = len(flm.clientStatus)
	
	return stats
}

// GetAuditLog 获取审计日志
func (flm *FederatedLearningManager) GetAuditLog() []AuditEntry {
	flm.mu.RLock()
	defer flm.mu.RUnlock()
	
	// 返回副本
	log := make([]AuditEntry, len(flm.auditLog))
	copy(log, flm.auditLog)
	
	return log
}
