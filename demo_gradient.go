package main

import (
	"fmt"
	"time"

	"./src/raft"
)

func main() {
	fmt.Println("=== Raft 梯度同步功能演示 ===")

	// 创建梯度管理器
	gm := raft.NewGradientManager(0.001)

	// 模拟多个worker提交梯度
	gradients := []raft.GradientLog{
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

	fmt.Println("提交梯度...")
	for _, grad := range gradients {
		shouldAggregate := gm.AddGradient(grad)
		fmt.Printf("Worker %d 提交梯度: Epoch=%d, BatchID=%d, 参数数量=%d\n",
			grad.WorkerID, grad.Epoch, grad.BatchID, len(grad.Gradients))

		if shouldAggregate {
			fmt.Println("达到聚合条件，执行梯度聚合...")
			aggGrad := gm.AggregateGradients(grad.BatchID)
			if aggGrad != nil {
				fmt.Printf("聚合完成: %d个worker, 参数数量=%d\n",
					aggGrad.WorkerCount, len(aggGrad.Gradients))

				// 应用聚合梯度到模型
				gm.ApplyGradientToModel(aggGrad)
				fmt.Println("梯度已应用到模型")
			}
		}
	}

	// 获取模型状态
	modelState := gm.GetModelState()
	fmt.Printf("\n当前模型状态:\n")
	fmt.Printf("Epoch: %d\n", modelState.Epoch)
	fmt.Printf("参数数量: %d\n", len(modelState.Params))
	for paramName, params := range modelState.Params {
		fmt.Printf("  %s: [", paramName)
		for i, val := range params {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.4f", val)
		}
		fmt.Println("]")
	}

	// 获取统计信息
	stats := gm.GetStats()
	fmt.Printf("\n统计信息:\n")
	fmt.Printf("总梯度数: %d\n", stats["total_gradients"])
	fmt.Printf("聚合批次: %d\n", stats["aggregated_batches"])
	fmt.Printf("学习率: %f\n", stats["learning_rate"])

	fmt.Println("\n=== 演示完成 ===")
}
