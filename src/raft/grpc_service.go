package raft

import (
	"context"
	"fmt"
	"log"
	"net"
	"sync"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// 定义protobuf消息结构（简化版）
type GradientProto struct {
	Epoch     int32
	WorkerID  int32
	BatchID   int32
	ParamName string
	Gradients []float32
}

type AggregatedGradientProto struct {
	Epoch       int32
	BatchID     int32
	WorkerCount int32
	Gradients   map[string][]float32
}

type Empty struct{}

// GradientService gRPC服务
type GradientService struct {
	mu sync.RWMutex

	// 复用现有的Raft和GradientManager
	raft            *Raft
	gradientManager *GradientManager

	// 梯度缓冲区（按batchID组织）
	gradBuffer map[int32]map[int32]*GradientLog // batchID -> workerID -> gradient
}

// NewGradientService 创建梯度服务
func NewGradientService(raft *Raft, learningRate float32) *GradientService {
	return &GradientService{
		raft:            raft,
		gradientManager: NewGradientManager(learningRate),
		gradBuffer:      make(map[int32]map[int32]*GradientLog),
	}
}

// SubmitGradient 提交梯度
func (s *GradientService) SubmitGradient(ctx context.Context, req *GradientProto) (*Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// 检查是否为Leader
	_, isLeader := s.raft.GetState()
	if !isLeader {
		return nil, status.Errorf(codes.Unavailable, "not leader")
	}

	// 初始化batch缓冲区
	if s.gradBuffer[req.BatchID] == nil {
		s.gradBuffer[req.BatchID] = make(map[int32]*GradientLog)
	}

	// 创建或更新梯度日志
	if s.gradBuffer[req.BatchID][req.WorkerID] == nil {
		s.gradBuffer[req.BatchID][req.WorkerID] = &GradientLog{
			Epoch:     int(req.Epoch),
			WorkerID:  int(req.WorkerID),
			BatchID:   int(req.BatchID),
			Gradients: make(map[string][]float32),
		}
	}

	// 添加梯度数据
	s.gradBuffer[req.BatchID][req.WorkerID].Gradients[req.ParamName] = req.Gradients

	log.Printf("收到梯度: Worker=%d, Batch=%d, Param=%s", req.WorkerID, req.BatchID, req.ParamName)

	// 检查是否达到聚合条件
	if len(s.gradBuffer[req.BatchID]) >= 3 { // 至少3个worker
		go s.aggregateAndCommit(req.BatchID)
	}

	return &Empty{}, nil
}

// GetAggregatedGradients 获取聚合梯度
func (s *GradientService) GetAggregatedGradients(ctx context.Context, req *Empty) (*AggregatedGradientProto, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// 获取最新的聚合梯度（这里简化处理）
	// 实际应该从Raft日志中获取已共识的聚合梯度

	// 模拟返回聚合结果
	result := &AggregatedGradientProto{
		Epoch:       1,
		BatchID:     1,
		WorkerCount: 3,
		Gradients: map[string][]float32{
			"weight1": {0.2, 0.3, 0.4},
			"weight2": {0.5, 0.6, 0.7},
		},
	}

	return result, nil
}

// aggregateAndCommit 聚合并提交梯度
func (s *GradientService) aggregateAndCommit(batchID int32) {
	s.mu.Lock()
	defer s.mu.Unlock()

	gradients := s.gradBuffer[batchID]
	if len(gradients) < 3 {
		return
	}

	// 转换为GradientLog列表
	var gradLogs []GradientLog
	for _, grad := range gradients {
		gradLogs = append(gradLogs, *grad)
	}

	// 使用现有的GradientManager进行聚合
	for _, grad := range gradLogs {
		shouldAggregate := s.gradientManager.AddGradient(grad)
		if shouldAggregate {
			aggGrad := s.gradientManager.AggregateGradients(int(batchID))
			if aggGrad != nil {
				// 应用梯度到模型
				s.gradientManager.ApplyGradientToModel(aggGrad)
				log.Printf("梯度聚合完成: Batch=%d, Workers=%d", batchID, aggGrad.WorkerCount)
			}
		}
	}

	// 清理缓冲区
	delete(s.gradBuffer, batchID)
}

// StartGRPCServer 启动gRPC服务器
func StartGRPCServer(raft *Raft, port int) error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return fmt.Errorf("failed to listen: %v", err)
	}

	server := grpc.NewServer()
	_ = NewGradientService(raft, 0.001) // 暂时不使用service变量

	// 注册服务（这里需要生成protobuf）
	// RegisterGradientServiceServer(server, service)

	log.Printf("gRPC服务器启动在端口 %d", port)
	return server.Serve(lis)
}
