package raft

import (
	"fmt"
	"testing"
	"time"
)

// TestGradientSynchronization 测试梯度同步功能
func TestGradientSynchronization(t *testing.T) {
	servers := 3
	cfg := make_config(t, servers, false, false)
	defer cfg.cleanup()

	cfg.begin("Test: Gradient Synchronization")

	// 等待选举完成
	cfg.checkOneLeader()
	time.Sleep(100 * time.Millisecond)

	// 模拟多个worker提交梯度
	leader := cfg.checkOneLeader()

	// 创建测试梯度数据
	testGradients := []GradientLog{
		{
			Epoch:    1,
			WorkerID: 0,
			BatchID:  1,
			Gradients: map[string][]float32{
				"weight1": {0.1, 0.2, 0.3},
				"weight2": {0.4, 0.5, 0.6},
			},
			Timestamp: time.Now(),
		},
		{
			Epoch:    1,
			WorkerID: 1,
			BatchID:  1,
			Gradients: map[string][]float32{
				"weight1": {0.2, 0.3, 0.4},
				"weight2": {0.5, 0.6, 0.7},
			},
			Timestamp: time.Now(),
		},
		{
			Epoch:    1,
			WorkerID: 2,
			BatchID:  1,
			Gradients: map[string][]float32{
				"weight1": {0.3, 0.4, 0.5},
				"weight2": {0.6, 0.7, 0.8},
			},
			Timestamp: time.Now(),
		},
	}

	// 提交梯度到Raft集群
	for i, grad := range testGradients {
		index, term, isLeader := cfg.rafts[leader].SubmitGradient(grad)
		if !isLeader {
			t.Fatalf("Expected leader to accept gradient")
		}
		fmt.Printf("Submitted gradient %d: index=%d, term=%d\n", i, index, term)
	}

	// 等待日志复制和应用
	time.Sleep(500 * time.Millisecond)

	// 检查所有节点的模型状态是否一致
	modelStates := make([]*ModelState, servers)
	for i := 0; i < servers; i++ {
		modelStates[i] = cfg.rafts[i].GetModelState()
		if modelStates[i] == nil {
			t.Fatalf("Node %d has no model state", i)
		}
	}

	// 验证模型状态一致性
	for i := 1; i < servers; i++ {
		if modelStates[i].Epoch != modelStates[0].Epoch {
			t.Fatalf("Model epoch mismatch: node 0=%d, node %d=%d",
				modelStates[0].Epoch, i, modelStates[i].Epoch)
		}

		// 检查参数是否一致（允许小的浮点误差）
		for paramName, params0 := range modelStates[0].Params {
			paramsI, exists := modelStates[i].Params[paramName]
			if !exists {
				t.Fatalf("Parameter %s missing in node %d", paramName, i)
			}

			if len(params0) != len(paramsI) {
				t.Fatalf("Parameter %s length mismatch: node 0=%d, node %d=%d",
					paramName, len(params0), i, len(paramsI))
			}

			for j := range params0 {
				diff := params0[j] - paramsI[j]
				if diff < -0.001 || diff > 0.001 {
					t.Fatalf("Parameter %s[%d] mismatch: node 0=%f, node %d=%f",
						paramName, j, params0[j], i, paramsI[j])
				}
			}
		}
	}

	// 检查梯度统计信息
	for i := 0; i < servers; i++ {
		stats := cfg.rafts[i].GetGradientStats()
		if stats["error"] != nil {
			t.Fatalf("Node %d gradient stats error: %v", i, stats["error"])
		}

		totalGradients := stats["total_gradients"].(int)
		if totalGradients < 3 {
			t.Fatalf("Node %d expected at least 3 gradients, got %d", i, totalGradients)
		}
	}

	fmt.Printf("✓ Gradient synchronization test passed\n")
	cfg.end()
}

// TestGradientFaultTolerance 测试梯度同步的容错能力
func TestGradientFaultTolerance(t *testing.T) {
	servers := 5
	cfg := make_config(t, servers, false, false)
	defer cfg.cleanup()

	cfg.begin("Test: Gradient Fault Tolerance")

	// 等待选举完成
	cfg.checkOneLeader()
	time.Sleep(100 * time.Millisecond)

	// 提交一些梯度
	leader := cfg.checkOneLeader()
	testGrad := GradientLog{
		Epoch:    1,
		WorkerID: 0,
		BatchID:  1,
		Gradients: map[string][]float32{
			"weight1": {0.1, 0.2, 0.3},
		},
		Timestamp: time.Now(),
	}

	_, _, isLeader := cfg.rafts[leader].SubmitGradient(testGrad)
	if !isLeader {
		t.Fatalf("Expected leader to accept gradient")
	}

	// 等待日志复制
	time.Sleep(200 * time.Millisecond)

	// 断开leader
	cfg.disconnect(leader)

	// 等待新leader选举
	time.Sleep(1000 * time.Millisecond)
	newLeader := cfg.checkOneLeader()
	if newLeader == leader {
		t.Fatalf("Expected new leader after disconnection")
	}

	// 提交更多梯度到新leader
	testGrad2 := GradientLog{
		Epoch:    1,
		WorkerID: 1,
		BatchID:  2,
		Gradients: map[string][]float32{
			"weight1": {0.4, 0.5, 0.6},
		},
		Timestamp: time.Now(),
	}

	_, _, isLeader2 := cfg.rafts[newLeader].SubmitGradient(testGrad2)
	if !isLeader2 {
		t.Fatalf("Expected new leader to accept gradient")
	}

	// 重新连接旧leader
	cfg.connect(leader)
	time.Sleep(500 * time.Millisecond)

	// 检查所有节点是否都应用了梯度
	for i := 0; i < servers; i++ {
		modelState := cfg.rafts[i].GetModelState()
		if modelState == nil {
			t.Fatalf("Node %d has no model state", i)
		}

		// 检查是否有参数更新
		if len(modelState.Params) == 0 {
			t.Fatalf("Node %d has no model parameters", i)
		}
	}

	fmt.Printf("✓ Gradient fault tolerance test passed\n")
	cfg.end()
}

// TestGradientPerformance 测试梯度同步性能
func TestGradientPerformance(t *testing.T) {
	servers := 3
	cfg := make_config(t, servers, false, false)
	defer cfg.cleanup()

	cfg.begin("Test: Gradient Performance")

	cfg.checkOneLeader()
	time.Sleep(100 * time.Millisecond)

	leader := cfg.checkOneLeader()

	// 批量提交梯度
	startTime := time.Now()
	numGradients := 50

	for i := 0; i < numGradients; i++ {
		grad := GradientLog{
			Epoch:    1,
			WorkerID: i % 3,
			BatchID:  i/10 + 1, // 每10个梯度一个batch
			Gradients: map[string][]float32{
				"weight1": {float32(i) * 0.01, float32(i) * 0.02, float32(i) * 0.03},
				"weight2": {float32(i) * 0.04, float32(i) * 0.05, float32(i) * 0.06},
			},
			Timestamp: time.Now(),
		}

		_, _, isLeader := cfg.rafts[leader].SubmitGradient(grad)
		if !isLeader {
			t.Fatalf("Expected leader to accept gradient")
		}
	}

	// 等待所有梯度处理完成
	time.Sleep(2 * time.Second)

	duration := time.Since(startTime)
	throughput := float64(numGradients) / duration.Seconds()

	fmt.Printf("Processed %d gradients in %v (%.2f gradients/sec)\n",
		numGradients, duration, throughput)

	// 检查统计信息
	for i := 0; i < servers; i++ {
		stats := cfg.rafts[i].GetGradientStats()
		totalGradients := stats["total_gradients"].(int)
		aggregatedBatches := stats["aggregated_batches"].(int)

		fmt.Printf("Node %d: %d gradients, %d aggregated batches\n",
			i, totalGradients, aggregatedBatches)
	}

	fmt.Printf("✓ Gradient performance test passed\n")
	cfg.end()
}
