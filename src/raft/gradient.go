package raft

import (
	"sync"
	"time"
)

// GradientLog 梯度日志结构
type GradientLog struct {
	Epoch     int                  `json:"epoch"`
	WorkerID  int                  `json:"worker_id"`
	BatchID   int                  `json:"batch_id"`
	Gradients map[string][]float32 `json:"gradients"` // 参数名 -> 梯度值
	Timestamp time.Time            `json:"timestamp"`
}

// AggregatedGradient 聚合后的梯度
type AggregatedGradient struct {
	Epoch       int                  `json:"epoch"`
	BatchID     int                  `json:"batch_id"`
	Gradients   map[string][]float32 `json:"gradients"`
	WorkerCount int                  `json:"worker_count"`
	Timestamp   time.Time            `json:"timestamp"`
}

// ModelState 模型状态
type ModelState struct {
	Epoch     int                  `json:"epoch"`
	Params    map[string][]float32 `json:"params"` // 参数名 -> 参数值
	Timestamp time.Time            `json:"timestamp"`
}

// GradientManager 梯度管理器
type GradientManager struct {
	mu sync.RWMutex

	// 梯度缓冲区
	gradientBuffer map[int][]GradientLog // batchID -> gradients
	bufferSize     int

	// 模型状态
	currentModel *ModelState
	learningRate float32

	// 统计信息
	stats struct {
		TotalGradients    int
		AggregatedBatches int
		LastAggregation   time.Time
	}
}

// NewGradientManager 创建梯度管理器
func NewGradientManager(learningRate float32) *GradientManager {
	return &GradientManager{
		gradientBuffer: make(map[int][]GradientLog),
		bufferSize:     10, // 每10个梯度聚合一次
		learningRate:   learningRate,
		currentModel: &ModelState{
			Epoch:     0,
			Params:    make(map[string][]float32),
			Timestamp: time.Now(),
		},
	}
}

// AddGradient 添加梯度到缓冲区
func (gm *GradientManager) AddGradient(grad GradientLog) bool {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	if gm.gradientBuffer[grad.BatchID] == nil {
		gm.gradientBuffer[grad.BatchID] = make([]GradientLog, 0)
	}

	gm.gradientBuffer[grad.BatchID] = append(gm.gradientBuffer[grad.BatchID], grad)
	gm.stats.TotalGradients++

	// 检查是否达到聚合条件
	return len(gm.gradientBuffer[grad.BatchID]) >= gm.bufferSize
}

// AggregateGradients 聚合梯度（模拟AllReduce）
func (gm *GradientManager) AggregateGradients(batchID int) *AggregatedGradient {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	gradients := gm.gradientBuffer[batchID]
	if len(gradients) == 0 {
		return nil
	}

	// 初始化聚合结果
	aggregated := &AggregatedGradient{
		Epoch:       gradients[0].Epoch,
		BatchID:     batchID,
		Gradients:   make(map[string][]float32),
		WorkerCount: len(gradients),
		Timestamp:   time.Now(),
	}

	// 聚合所有worker的梯度
	for _, grad := range gradients {
		for paramName, gradValues := range grad.Gradients {
			if _, exists := aggregated.Gradients[paramName]; !exists {
				aggregated.Gradients[paramName] = make([]float32, len(gradValues))
			}

			// 累加梯度
			for i, val := range gradValues {
				aggregated.Gradients[paramName][i] += val
			}
		}
	}

	// 求平均（模拟AllReduce）
	for paramName := range aggregated.Gradients {
		for i := range aggregated.Gradients[paramName] {
			aggregated.Gradients[paramName][i] /= float32(len(gradients))
		}
	}

	// 清理缓冲区
	delete(gm.gradientBuffer, batchID)
	gm.stats.AggregatedBatches++
	gm.stats.LastAggregation = time.Now()

	return aggregated
}

// ApplyGradientToModel 将聚合梯度应用到模型
func (gm *GradientManager) ApplyGradientToModel(aggGrad *AggregatedGradient) {
	gm.mu.Lock()
	defer gm.mu.Unlock()

	// 更新模型参数（梯度下降）
	for paramName, gradValues := range aggGrad.Gradients {
		if _, exists := gm.currentModel.Params[paramName]; !exists {
			gm.currentModel.Params[paramName] = make([]float32, len(gradValues))
		}

		for i, grad := range gradValues {
			gm.currentModel.Params[paramName][i] -= gm.learningRate * grad
		}
	}

	gm.currentModel.Epoch = aggGrad.Epoch
	gm.currentModel.Timestamp = time.Now()
}

// GetModelState 获取当前模型状态
func (gm *GradientManager) GetModelState() *ModelState {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	// 返回副本避免并发修改
	state := &ModelState{
		Epoch:     gm.currentModel.Epoch,
		Params:    make(map[string][]float32),
		Timestamp: gm.currentModel.Timestamp,
	}

	for paramName, params := range gm.currentModel.Params {
		state.Params[paramName] = make([]float32, len(params))
		copy(state.Params[paramName], params)
	}

	return state
}

// GetStats 获取统计信息
func (gm *GradientManager) GetStats() map[string]interface{} {
	gm.mu.RLock()
	defer gm.mu.RUnlock()

	return map[string]interface{}{
		"total_gradients":    gm.stats.TotalGradients,
		"aggregated_batches": gm.stats.AggregatedBatches,
		"last_aggregation":   gm.stats.LastAggregation,
		"buffer_size":        len(gm.gradientBuffer),
		"current_epoch":      gm.currentModel.Epoch,
		"learning_rate":      gm.learningRate,
	}
}
